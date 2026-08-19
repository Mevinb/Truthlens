#!/usr/bin/env python3
"""
TruthLens — src/features_clip.py
=================================
Cache frozen CLIP ViT-L/14 image features for the modern corpus, once, so a
linear head can be fit and refit in seconds instead of hours.

Why a frozen backbone at all
----------------------------
The ResNet-18 runs established that capacity was not the limit — the
representation was. A backbone trained end-to-end on this corpus learns the
corpus, and the corpus is what the audit says it is: format, resolution and
aspect ratio all correlate with the label. Freezing the backbone removes the
model's ability to fit those correlations into its features at all. Whatever the
linear head then achieves off-tier is attributable to CLIP's pretrained
representation rather than to anything this corpus taught it, which is the
property the current phase is trying to buy.

The checkpoint pairing is not interchangeable
---------------------------------------------
``ViT-L-14-quickgelu`` with ``openai``, and specifically not ``ViT-L-14`` with
``openai``. OpenAI trained these weights with the QuickGELU activation and
open_clip declares that in the pretrained tag (``quick_gelu: True``), but the
plain ``ViT-L-14`` config leaves it off, so that pairing loads the right tensors
into the wrong activation and open_clip emits only a warning. Measured on a
corpus image the two builds agree to a cosine of 0.943 — close enough to look
like it works, far enough to cost real accuracy. A weak linear probe on those
features would read as "frozen CLIP does not help here", which would be a false
conclusion drawn from a silent bug.

Why views are cached rather than augmentations sampled
------------------------------------------------------
The training pipeline in :mod:`src.preprocessing` samples a *random* view per
epoch — native-scale crop or whole frame, random JPEG quality, random aspect pad
— and relies on the model seeing every image many ways across many epochs. A
cached feature cannot be re-augmented; whatever transform ran at extraction time
is frozen into the vector. So the randomness is replaced by a small fixed set of
deterministic views, chosen so that each one answers a different question, and a
separate head is fit per view. That turns what would be an invisible averaging
into something measurable: if a view's head is strong in-tier and collapses
off-tier, that view is carrying a shortcut, and the per-view numbers say so.

The one confound this closes by construction
--------------------------------------------
Every image is re-encoded through the same JPEG encoder at the same quality
before any view is taken. The corpus stores original bytes, so container format
correlates with the label — PNG runs 95.7% fake today — and
:class:`~src.preprocessing.RandomRecompress` exists to break exactly that for
the CNN path. Randomising quality is not available here, but it is also not
needed: passing *all* images through one fixed encoder means format and encoder
history are literally constant across the corpus and can carry no information at
all. That is a stronger guarantee than the random version, not a weaker one.

The one confound this does *not* close
--------------------------------------
The ``frame`` view leaks source resolution and no deterministic transform can
fix it. Showing a whole frame at 224px means a 4000px original is averaged 18x
and a 300px original 1.3x, and "how smooth is this" then reads out the source's
resolution — which the audit measures as the corpus's strongest leak. The
training pipeline dilutes this by resizing to a random common target first;
determinism forbids that. The three native views are immune (a 1:1 crop looks
the same whichever size frame it came from), so the honest arrangement is to keep
``frame`` for the global cues it genuinely carries, fit its head separately, and
report it as the view whose off-tier drop is expected. Quantifying a leak is
worth more here than hiding it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.utils import ROOT_DIR, get_logger

logger = get_logger(__name__)

MODEL_NAME = "ViT-L-14-quickgelu"
PRETRAINED = "openai"
EMBED_DIM = 768

DEFAULT_CORPUS = ROOT_DIR / "datasets" / "prepared" / "modern_v2"
DEFAULT_CACHE = ROOT_DIR / "features" / "clip_vitl14"

# Fixed re-encode quality. High enough that the generator fingerprint the native
# views depend on survives, low enough to be a real encode rather than a no-op.
REENCODE_QUALITY = 90

# Below this short side a 224 crop is most of the frame, so "native scale" stops
# meaning anything and the view degrades to a resize. Matches the 1.25x factor
# NativeScaleCrop uses so the two pipelines agree on which images are too small.
MIN_NATIVE_SHORT = int(224 * 1.25)


# ─── Views ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class View:
    """One deterministic way of looking at an image.

    ``leaks_resolution`` is carried here rather than left as lore in a docstring
    because the training script uses it to decide which per-view heads may be
    combined into a headline number, and the evaluation prints it next to the
    view's off-tier score.
    """

    name: str
    kind: str                      # "native" | "frame"
    anchor: Tuple[float, float] = (0.5, 0.5)
    leaks_resolution: bool = False
    why: str = ""


VIEWS: Tuple[View, ...] = (
    View("native_c", "native", (0.5, 0.5), False,
         "1:1 centre crop — the high-frequency forensic band at source scale"),
    View("native_tl", "native", (0.25, 0.25), False,
         "1:1 crop off-centre, so a single unlucky patch is not the whole vote"),
    View("native_br", "native", (0.75, 0.75), False,
         "1:1 crop off-centre on the other diagonal"),
    View("frame", "frame", (0.5, 0.5), True,
         "whole frame — global composition and anatomy, but reads out source "
         "resolution, so its off-tier drop is expected"),
)
VIEW_BY_NAME = {v.name: v for v in VIEWS}


def reencode(img: Image.Image, quality: int = REENCODE_QUALITY) -> Image.Image:
    """Put every image through one identical encoder.

    Applied before cropping for the same reason
    :class:`~src.preprocessing.RandomRecompress` is: the corpus holds genuine
    20-60MP camera originals, and re-encoding those at full resolution to produce
    a 224px view was measured at 61ms/image and half the preprocessing cost. The
    decorrelation this exists for is unaffected by the order — all that matters
    is that both classes pass through the same encoder.
    """
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render(img: Image.Image, view: View, size: int = 224) -> Tuple[Image.Image, bool]:
    """Produce one view. Returns the image and whether it is at native scale.

    The flag matters for honesty rather than for control flow: a small image has
    no native scale to preserve, so its "native" views are upscales that
    contain none of the high-frequency evidence the view exists to capture.
    Counting them lets the report say how much of each tier is actually being
    judged on forensic texture.
    """
    w, h = img.size
    short = min(w, h)

    if view.kind == "frame":
        # Centre-crop to square first, then one resize. Cropping rather than
        # squashing keeps aspect ratio from surviving as distortion, which the
        # audit flags as its third confound (portrait ran 73% fake).
        side = short
        left, top = (w - side) // 2, (h - side) // 2
        return (img.crop((left, top, left + side, top + side))
                .resize((size, size), Image.BICUBIC), False)

    if short < MIN_NATIVE_SHORT:
        return img.resize((size, size), Image.BICUBIC), False

    ax, ay = view.anchor
    x = int(round((w - size) * ax))
    y = int(round((h - size) * ay))
    return img.crop((x, y, x + size, y + size)), True


# ─── Corpus ───────────────────────────────────────────────────────────────────
def load_manifest(corpus: Path, splits: Sequence[str]) -> List[dict]:
    path = corpus / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    wanted = set(splits)
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") in wanted:
                rows.append(row)
    # Sort so the cache order is reproducible across runs and machines; the
    # sha256 is already in the manifest and is unique per image.
    rows.sort(key=lambda r: (r["split"], r["sha256"]))
    return rows


class ViewDataset:
    """Yields all views of one image from a single decode.

    Decoding dominates cost — the corpus averages well over a megapixel and some
    rows are 60MP — so the four views are produced from one decode and one
    re-encode rather than by iterating the corpus once per view.
    """

    def __init__(self, rows: List[dict], corpus: Path, views: Sequence[View],
                 size: int = 224, quality: int = REENCODE_QUALITY):
        self.rows = rows
        self.corpus = corpus
        self.views = list(views)
        self.size = size
        self.quality = quality
        import torchvision.transforms as T
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch
        row = self.rows[i]
        try:
            with Image.open(self.corpus / row["path"]) as im:
                im.load()
                img = reencode(im, self.quality)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("unreadable %s: %s", row["path"], exc)
            return (torch.zeros(len(self.views), 3, self.size, self.size),
                    torch.zeros(len(self.views), dtype=torch.bool), i, False)

        tensors, native = [], []
        for v in self.views:
            rendered, is_native = render(img, v, self.size)
            tensors.append(self.to_tensor(rendered))
            native.append(is_native)
        return (torch.stack(tensors),
                torch.tensor(native, dtype=torch.bool), i, True)


def _collate(batch):
    import torch
    views = torch.cat([b[0] for b in batch], dim=0)     # (B*V, 3, S, S)
    native = torch.stack([b[1] for b in batch])         # (B, V)
    idx = torch.tensor([b[2] for b in batch])
    ok = torch.tensor([b[3] for b in batch], dtype=torch.bool)
    return views, native, idx, ok


# ─── Extraction ───────────────────────────────────────────────────────────────
def build_backbone(device: str):
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model.eval().to(device)
    if device.startswith("cuda"):
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract(
    rows: List[dict],
    corpus: Path,
    cache: Path,
    views: Sequence[View],
    batch_images: int = 16,
    workers: int = 6,
    device: Optional[str] = None,
) -> Dict[str, Path]:
    """Extract and cache features for every (split, view) pair.

    Writes one ``.npy`` per split+view plus one shared index per split, so a
    head can be fit on any subset of views without touching the images again.
    """
    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache.mkdir(parents=True, exist_ok=True)
    model = build_backbone(device)

    by_split: Dict[str, List[dict]] = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    written: Dict[str, Path] = {}
    for split, split_rows in sorted(by_split.items()):
        out_paths = {v.name: cache / f"{split}__{v.name}.npy" for v in views}
        index_path = cache / f"{split}__index.json"
        if all(p.exists() for p in out_paths.values()) and index_path.exists():
            logger.info("%s: cached, skipping (%d rows)", split, len(split_rows))
            written.update({f"{split}__{k}": v for k, v in out_paths.items()})
            continue

        ds = ViewDataset(split_rows, corpus, views)
        loader = DataLoader(ds, batch_size=batch_images, shuffle=False,
                           num_workers=workers, collate_fn=_collate,
                           pin_memory=device.startswith("cuda"))

        n, nv = len(split_rows), len(views)
        feats = np.zeros((nv, n, EMBED_DIM), dtype=np.float16)
        native = np.zeros((n, nv), dtype=bool)
        readable = np.zeros(n, dtype=bool)

        done = 0
        for batch_views, batch_native, idx, ok in loader:
            b = idx.numel()
            x = batch_views.to(device, non_blocking=True)
            if device.startswith("cuda"):
                x = x.half()
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            # (B*V, D) -> (B, V, D) -> (V, B, D)
            f = f.reshape(b, nv, EMBED_DIM).permute(1, 0, 2).float().cpu().numpy()
            rows_idx = idx.numpy()
            feats[:, rows_idx, :] = f.astype(np.float16)
            native[rows_idx, :] = batch_native.numpy()
            readable[rows_idx] = ok.numpy()
            done += b
            if done % (batch_images * 20) < batch_images:
                logger.info("%s: %d/%d", split, done, n)

        for v in views:
            vi = views.index(v)
            np.save(out_paths[v.name], feats[vi])
            written[f"{split}__{v.name}"] = out_paths[v.name]

        index_path.write_text(json.dumps({
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "embed_dim": EMBED_DIM,
            "reencode_quality": REENCODE_QUALITY,
            "views": [{"name": v.name, "kind": v.kind, "anchor": list(v.anchor),
                       "leaks_resolution": v.leaks_resolution, "why": v.why}
                      for v in views],
            "rows": [{
                "path": r["path"], "label": r["label"], "generator": r["generator"],
                "source_key": r["source_key"], "split": r["split"],
                "sha256": r["sha256"], "width": r["width"], "height": r["height"],
                "orig_format": r.get("orig_format", ""),
                "rendition": r.get("rendition", "pristine"),
            } for r in split_rows],
            "native_scale": native.tolist(),
            "readable": readable.tolist(),
        }, indent=1))
        logger.info("%s: wrote %d rows x %d views (%d unreadable, "
                    "%d not at native scale)",
                    split, n, nv, int((~readable).sum()),
                    int((~native[:, 0]).sum()))

    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test", "holdout"])
    ap.add_argument("--views", nargs="+", default=[v.name for v in VIEWS],
                    choices=[v.name for v in VIEWS])
    ap.add_argument("--batch-images", type=int, default=16,
                    help="images per batch; each becomes len(views) crops")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows per split, for a smoke run")
    args = ap.parse_args(argv)

    rows = load_manifest(args.corpus, args.splits)
    if args.limit:
        capped: Dict[str, int] = {}
        kept = []
        for r in rows:
            c = capped.get(r["split"], 0)
            if c < args.limit:
                kept.append(r)
                capped[r["split"]] = c + 1
        rows = kept
    if not rows:
        print("no rows matched", file=sys.stderr)
        return 1

    views = [VIEW_BY_NAME[n] for n in args.views]
    logger.info("%d rows, %d views, backbone %s/%s",
                len(rows), len(views), MODEL_NAME, PRETRAINED)
    extract(rows, args.corpus, args.cache, views,
            batch_images=args.batch_images, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
