#!/usr/bin/env python3
"""
TruthLens — src/score_clip.py
==============================
Serve the frozen-CLIP linear head as a scorer, so it can be measured by exactly
the protocol that measured the ResNet.

Why this exists as a separate module
-----------------------------------
:mod:`src.features_clip` caches features for the corpus splits, and
:mod:`src.train_head` fits heads on that cache. Neither can score an arbitrary
image path, which is what the tiered board and the app both need — the board
 reaches into ``datasets/evaluation/internet`` for its wild tier, and the app is handed one
upload at a time. This module closes that gap and is the single place where a
cached-feature head becomes a callable model.

The one invariant that matters
------------------------------
An image scored here must be rendered *identically* to the way it was rendered
at extraction time: the same fixed re-encode, the same four deterministic views,
the same normalisation. So this imports :class:`~src.features_clip.ViewDataset`
and :data:`~src.features_clip.VIEWS` rather than reimplementing them — a
reimplementation is exactly where a silent train/serve skew would live, and the
skew would look like a generalisation failure rather than a bug.

Which number it serves
----------------------
``ensemble_all`` by default: the mean calibrated probability over the three 1:1
native crops and the whole-frame view. The native views retain fine forensic
detail while the frame view contributes global composition. Every view remains
reported separately during evaluation, including the frame view's known
resolution sensitivity. ``--views`` can request the native-only ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.utils import ROOT_DIR, get_logger

logger = get_logger(__name__)

DEFAULT_HEADS = ROOT_DIR / "models" / "clip_linear" / "heads.joblib"


class ClipLinearScorer:
    """P(FAKE) from frozen CLIP features plus the saved per-view linear heads.

    Batched on the GPU because the backbone is the whole cost: a ViT-L/14 forward
    over four views is ~80ms per image against the ResNet's ~14ms, and scoring a
    few thousand board images one at a time would be minutes of avoidable idle.
    """

    #: The board resolves a ResNet checkpoint before constructing a scorer. This
    #: one carries its own weights and its own heads, so that step is skipped for
    #: it rather than silently satisfied by whatever ``.pth`` happens to be newest.
    manages_own_checkpoint = True

    def __init__(self, cfg=None, model_type: str = "clip",
                 heads_path: Optional[Path] = None,
                 views: Optional[Sequence[str]] = None,
                 batch_images: int = 16, workers: int = 6,
                 device: Optional[str] = None) -> None:
        import joblib
        import torch

        from src.features_clip import VIEWS, build_backbone

        if model_type not in ("clip", "clip_linear"):
            raise ValueError(
                f"ClipLinearScorer cannot serve model_type={model_type!r}")

        path = Path(heads_path or DEFAULT_HEADS)
        if not path.exists():
            raise FileNotFoundError(
                f"no fitted heads at {path}; run:\n"
                f"  python -m src.features_clip\n"
                f"  python -m src.train_head")
        bundle = joblib.load(path)
        self.backbone_meta = bundle["backbone"]
        self.thresholds = bundle["thresholds"]
        self.target_fpr = bundle.get("target_fpr")
        self.fusion = bundle.get("fusion")
        self.adaptation = bundle.get("adaptation")

        # Default to the complete multi-view detector. Asking for a set the heads
        # were not fitted on is an error rather than a silent subset, because the
        # ensemble mean is only comparable across runs if membership is stable.
        by_view = {h["view"]: h for h in bundle["heads"]}
        if views:
            missing = [v for v in views if v not in by_view]
            if missing:
                raise KeyError(f"no head for views {missing} "
                               f"(have {sorted(by_view)})")
            want = list(views)
        else:
            want = [h["view"] for h in bundle["heads"]]
        self.heads = [by_view[v] for v in want]
        self.view_names = want

        self.views = [v for v in VIEWS if v.name in set(want)]
        if len(self.views) != len(want):
            raise KeyError(
                f"views {want} are in the head bundle but not in "
                f"features_clip.VIEWS ({[v.name for v in VIEWS]}); the render "
                f"spec has changed since the heads were fitted, so scoring "
                f"would not match extraction")
        # Preserve the bundle's order so the ensemble mean is order-stable.
        self.views.sort(key=lambda v: want.index(v.name))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_images = batch_images
        self.workers = workers
        self.model = build_backbone(self.device)

        label = "+".join(want)
        self.name = (f"clip-linear:{self.backbone_meta['model']}/"
                     f"{self.backbone_meta['pretrained']}:{label}")
        resolved_path = path.resolve()
        try:
            shown_path = resolved_path.relative_to(ROOT_DIR)
        except ValueError:
            shown_path = resolved_path
        self.meta = {
            **self.backbone_meta,
            "views": want,
            "threshold_from_val": self._served_threshold_name(),
            "heads_path": str(shown_path),
            "adaptation": self.adaptation,
            "fusion": bool(self.fusion),
        }

    def _served_threshold_name(self) -> str:
        """Which stored threshold corresponds to the ensemble being served."""
        from src.features_clip import VIEWS
        leaks = {v.name: v.leaks_resolution for v in VIEWS}
        if len(self.view_names) == 1:
            return self.view_names[0]
        all_native = all(not leaks.get(v, False) for v in self.view_names)
        if all_native and len(self.view_names) == sum(1 for v in VIEWS
                                                     if not v.leaks_resolution):
            return "ensemble_native"
        if len(self.view_names) == len(VIEWS):
            return "ensemble_all"
        return "(no stored threshold for this view subset)"

    def served_threshold(self) -> Optional[float]:
        """The validation-fitted threshold for the served ensemble, if stored."""
        v = self.thresholds.get(self._served_threshold_name())
        return None if v is None or not np.isfinite(v) else float(v)

    def _probabilities_from_features(self, feats: np.ndarray) -> np.ndarray:
        """Convert ``(views, images, embed_dim)`` features to P(FAKE)."""
        ps = self._view_probabilities_from_features(feats)
        stacked = np.column_stack(ps)
        if self.fusion:
            expected = self.fusion.get("input_views", [])
            if list(self.view_names) != list(expected):
                raise ValueError(
                    "adapted fusion requires views "
                    f"{expected}, got {self.view_names}"
                )
            return self.fusion["model"].predict_proba(stacked)[:, 1]
        return stacked.mean(axis=1)

    def _view_probabilities_from_features(
        self, feats: np.ndarray
    ) -> List[np.ndarray]:
        """One calibrated P(FAKE) vector per view."""
        ps = []
        for k, h in enumerate(self.heads):
            raw = h["model"].decision_function(feats[k]).reshape(-1, 1)
            ps.append(h["calibrator"].predict_proba(raw)[:, 1])
        return ps

    def score_image_details(self, image: object) -> Optional[dict]:
        """Return fused and per-view probabilities for one decoded image."""
        feats, readable = self._encode_images([image])
        if not readable[0]:
            return None
        view_ps = self._view_probabilities_from_features(feats)
        fused = self._probabilities_from_features(feats)
        values = {name: float(view_ps[k][0])
                  for k, name in enumerate(self.view_names)}
        arr = np.array(list(values.values()), dtype=float)
        return {
            "p_fake": float(fused[0]),
            "views": values,
            "view_min": float(arr.min()),
            "view_max": float(arr.max()),
            "view_std": float(arr.std()),
        }

    def _encode_images(
        self, images: Sequence[object]
    ) -> tuple[np.ndarray, List[bool]]:
        """Render and encode decoded images, returning features and readability."""
        import torch
        import torchvision.transforms as T
        from PIL import Image

        from src.features_clip import EMBED_DIM, reencode, render

        to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])
        tensors: List[torch.Tensor] = []
        readable: List[bool] = []
        for image in images:
            try:
                if not isinstance(image, Image.Image):
                    raise TypeError(f"expected PIL image, got {type(image)!r}")
                img = reencode(image.convert("RGB"))
                tensors.extend(to_tensor(render(img, v)[0]) for v in self.views)
                readable.append(True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("unreadable in-memory image: %s", exc)
                tensors.extend(torch.zeros(3, 224, 224) for _ in self.views)
                readable.append(False)

        if not images:
            return np.empty((len(self.views), 0, EMBED_DIM)), []
        batch = torch.stack(tensors).to(self.device)
        if self.device.startswith("cuda"):
            batch = batch.half()
        with torch.no_grad():
            encoded = self.model.encode_image(batch)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        n, nv = len(images), len(self.views)
        feats = (encoded.reshape(n, nv, EMBED_DIM)
                 .permute(1, 0, 2).float().cpu().numpy())
        return feats, readable

    def p_fake_images(self, images: Sequence[object]) -> List[Optional[float]]:
        """Score decoded PIL images without writing temporary files."""
        feats, readable = self._encode_images(images)
        if not images:
            return []
        p = self._probabilities_from_features(feats)
        return [float(p[i]) if readable[i] else None for i in range(len(images))]

    # ── the scorer interface the board calls ─────────────────────────────────
    def p_fake(self, rows: Sequence[dict]) -> List[Optional[float]]:
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from src.features_clip import EMBED_DIM, ViewDataset, _collate

        # ViewDataset joins corpus/path. Board rows carry absolute paths, and
        # joining an absolute path onto any root yields the absolute path, so the
        # same class serves both callers without a second code path.
        ds = ViewDataset([{"path": str(r["path"])} for r in rows],
                         ROOT_DIR, self.views)
        loader = DataLoader(ds, batch_size=self.batch_images, shuffle=False,
                            num_workers=self.workers, collate_fn=_collate,
                            pin_memory=self.device.startswith("cuda"))

        nv = len(self.views)
        feats = np.zeros((nv, len(rows), EMBED_DIM), dtype=np.float32)
        ok = np.zeros(len(rows), dtype=bool)
        with tqdm(total=len(rows), desc="  scoring", ncols=72,
                  unit="img", leave=False) as bar:
            for batch_views, _native, idx, readable in loader:
                b = idx.numel()
                x = batch_views.to(self.device, non_blocking=True)
                if self.device.startswith("cuda"):
                    x = x.half()
                with torch.no_grad():
                    f = self.model.encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True)
                f = f.reshape(b, nv, EMBED_DIM).permute(1, 0, 2).float().cpu().numpy()
                feats[:, idx.numpy(), :] = f
                ok[idx.numpy()] = readable.numpy()
                bar.update(b)

        # Mean of *calibrated* probabilities, matching src.train_head.ensemble_p.
        # Averaging raw margins instead would weight whichever head is most
        # confident, and the heads were fitted with different C.
        p = self._probabilities_from_features(feats)

        # An unreadable image gets None, not a probability. A zero feature vector
        # would still produce a confident-looking number.
        return [float(p[i]) if ok[i] else None for i in range(len(rows))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score explicit image paths — a smoke test for the serve path.

    Useful on its own: if these probabilities disagree with what
    :mod:`src.train_head` reported for the same corpus images, the serve path has
    drifted from the extraction path and every board number downstream is wrong.
    """
    ap = argparse.ArgumentParser(description="Score images with the CLIP head")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--heads", type=Path, default=DEFAULT_HEADS)
    ap.add_argument("--views", nargs="+", default=None)
    args = ap.parse_args(argv)

    scorer = ClipLinearScorer(heads_path=args.heads, views=args.views)
    print(f"{scorer.name}")
    thr = scorer.served_threshold()
    print(f"served threshold ({scorer.meta['threshold_from_val']}): "
          f"{thr if thr is not None else '—'}  "
          f"@ target FPR {scorer.target_fpr}")
    rows = [{"path": p.resolve()} for p in args.paths]
    for r, p in zip(rows, scorer.p_fake(rows)):
        if p is None:
            print(f"  {r['path'].name:<44} unreadable")
            continue
        call = "?" if thr is None else ("FAKE" if p >= thr else "REAL")
        print(f"  {r['path'].name:<44} P(fake) {p:.4f}   {call}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
