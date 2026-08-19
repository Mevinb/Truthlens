#!/usr/bin/env python3
"""
TruthLens — Modern Generator Corpus Collector
==============================================
Builds ``datasets/prepared/modern_v2/`` to cover the generators the existing corpus does
not contain, which is why the detector fails on images made with ChatGPT today.

Why this exists
---------------
The fake class of ``datasets/prepared/multires`` is CIFAKE (latent diffusion on CIFAR,
32px), two mixed archives of older SD/GAN output, and 1,380 DALL-E 3 images.
ChatGPT has not served DALL-E 3 since March 2025; it serves the native
multimodal image models — GPT-Image-1, and GPT-Image-2 since 2026-04-21. Those
leave a different fingerprint than a diffusion U-Net, so a detector trained on
diffusion artifacts has no reason to generalise to them, and does not.

Design decisions that matter, and the failure each one avoids
-------------------------------------------------------------
1. **Bytes are stored verbatim — never re-encoded.** ``build_multires_dataset``
   re-saves every image as JPEG q95. That erases the high-frequency generator
   fingerprint and, worse, makes container format a shortcut: if every fake has
   been through the same JPEG encoder and no real upload has, the model can
   learn the encoder instead of the generator. Here the original bytes land on
   disk untouched and ``orig_format`` is recorded, so re-encoding becomes a
   *training-time augmentation* that can be randomised over both classes.

2. **Splits are assigned from the content hash, not a shuffle.** Re-running
   with a larger ``--limit`` keeps every image in the split it was already in,
   so a later top-up cannot leak a training image into test.

3. **Eval-only sources are quarantined into ``holdout/``.** Two sources must
   never reach training: ``felix-chatgpt-generated-images`` is uploader-
   "all rights reserved" (fine to measure against, not to train on), and
   ``liars-dividend`` is a caption-paired real/fake set whose value is entirely
   as a clean GPT-Image-2 benchmark. They go to ``holdout/`` which the training
   loader does not read.

4. **Every image carries provenance in ``manifest.jsonl``** — source repo,
   generator, licence, original dimensions, aspect ratio. This is what makes
   per-generator accuracy measurable; the current corpus has no per-generator
   labels, which is why the app's generator guess is unvalidated.

5. **``--report`` audits for shortcuts before any training happens.** Chiefly
   aspect ratio: GPT-Image output is 1024x1536 / 1536x1024 / 1024x1024, while
   Flickr reals are mostly 3:2 landscape. If portrait implies fake in the
   corpus, the model learns orientation, scores well offline, and still fails
   on a real upload. Run the report and match the distributions before training.

Usage
-----
    python dataset/collect_modern_generators.py --report
    python dataset/collect_modern_generators.py --sources gptimage2_wild,gptimage1
    python dataset/collect_modern_generators.py --limit 1500
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "prepared" / "modern_v2"
MANIFEST_PATH = OUTPUT_DIR / "manifest.jsonl"

# Content-hash split boundaries. Deterministic in the image bytes, so a re-run
# with a bigger --limit never moves an image between splits.
SPLIT_TRAIN, SPLIT_VAL = 0.80, 0.90


# ─── Source registry ─────────────────────────────────────────────────────────
@dataclass
class Source:
    key: str
    repo_id: str
    label: str                 # "real" | "fake"
    generator: str             # per-image provenance label
    method: str                # glob | parquet | parquet_rowgroups | jsonl_filtered
    license: str
    url: str
    pattern: str = ""          # glob: repo path pattern
    image_column: str = "image"
    holdout: bool = False      # quarantine from training splits
    notes: str = ""
    limit: Optional[int] = None  # source-specific cap, overrides --limit
    # parquet: configs to draw from in order. Multi-config repos (megalith ships
    # 12 chunks) need one named explicitly, and spreading the take across
    # several avoids pulling every real photo from one upload batch.
    configs: List[str] = field(default_factory=list)
    # parquet: HF *split* names, for repos that shard by split rather than by
    # config (bm-subnet-real-fullsize ships three weekly splits and no "train").
    hf_splits: List[str] = field(default_factory=list)
    # parquet_rowgroups: how many shards to spread the take across. Each shard
    # costs one footer read (~224 KB, ~1.5s) before any image arrives, so this
    # trades a fixed setup cost for protection against shard-level ordering.
    # Twelve is enough spread to notice a skew and cheap enough to ignore.
    max_shards: int = 12
    # Resolution window, in short-side pixels, applied after decoding. 0 = no
    # bound. Used to draw a specific resolution band out of a source rather than
    # whatever it happens to be dominated by; see ``real_hires_camera``.
    min_short: int = 0
    max_short: int = 0
    # How many source rows to pull per image kept. Only matters with a
    # resolution window: the fetchers stop after yielding their budget, so a
    # source that rejects 97% of rows needs its budget scaled up or collection
    # ends silently short of target.
    overfetch: float = 1.0
    # Packaging regime, recorded per image. "pristine" is direct model output or
    # a camera original; "wild" has been through a platform that recompressed it.
    # The two fail differently — the corpus scored 93% on wild GPT-Image-2 and
    # 12% on pristine — so the regime has to be a field, not a note.
    rendition: str = "pristine"
    rendition_column: str = ""
    rendition_map: Dict[str, str] = field(default_factory=dict)

    # ── Per-row provenance ───────────────────────────────────────────────────
    # For repos that pack several generators, both classes, and the intended
    # train/holdout division into one stream. So-Fake-OOD is the case that forced
    # this: label, generator, platform and holdout status all vary row to row,
    # and one Source per partition would mean a separate full pass over a 135 GB
    # repo for each.
    where: Dict[str, List[str]] = field(default_factory=dict)      # keep if all match
    where_not: Dict[str, List[str]] = field(default_factory=dict)  # drop if any match
    label_column: str = ""
    label_map: Dict[str, str] = field(default_factory=dict)
    generator_column: str = ""
    # Upstream spelling → corpus spelling. Load-bearing rather than cosmetic: a
    # generator already in the corpus must map onto its *existing* name, or the
    # evaluation counts a third source of a known generator as a brand-new one
    # and the held-out-source tier silently loses its subject.
    generator_map: Dict[str, str] = field(default_factory=dict)
    # Rows matching any entry here are quarantined, and take a distinct
    # source_key so the protocol reads them as separate provenance.
    holdout_when: Dict[str, List[str]] = field(default_factory=dict)
    holdout_suffix: str = "_holdout"
    # Extra row columns to carry into the manifest, e.g. the platform a real
    # photo was scraped from.
    extra_fields: List[str] = field(default_factory=list)


SOURCES: List[Source] = [
    # ── FAKE: the actual failure case ────────────────────────────────────────
    Source(
        key="gptimage2_wild",
        repo_id="Scam-AI/gpt-image-2",
        label="fake",
        generator="gpt-image-2",
        method="jsonl_filtered",
        license="CC-BY-NC-SA-4.0 (non-commercial research)",
        url="https://huggingface.co/datasets/Scam-AI/gpt-image-2",
        notes="GPT-Image-2 scraped from Twitter/X during launch week "
              "(2026-04-21..28). Gated, auto-approve. Only the 4,959 rows with "
              "classification=confirmed are taken; the other 5,258 are "
              "self-reported/uncertain and would be ~50% label noise. "
              "Twitter re-encodes to JPEG, so this is the in-the-wild regime "
              "rather than pristine model output — which is what a shared "
              "image actually looks like.",
        limit=3000,
    ),
    Source(
        key="gptimage1",
        repo_id="a3xrfgb/gpt-image-mega-4k",
        label="fake",
        generator="gpt-image-1",
        method="glob",
        pattern="GPTIMG_*.png",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/a3xrfgb/gpt-image-mega-4k",
        notes="4,000 pristine GPT-Image-1 PNGs at 1024x1536 — never "
              "recompressed, so this is the regime of a direct ChatGPT "
              "download. Complements the Twitter-recompressed set above.",
        limit=1500,
    ),
    Source(
        key="seedream45",
        repo_id="ash12321/seedream-4.5-generated-2k",
        label="fake",
        generator="seedream-4.5",
        method="glob",
        pattern="image_*.png",
        license="see dataset card",
        url="https://huggingface.co/datasets/ash12321/seedream-4.5-generated-2k",
        notes="Non-OpenAI modern generator, to stop the model keying on one "
              "vendor's signature.",
        limit=200,
    ),
    Source(
        key="flux1dev",
        repo_id="ash12321/flux-1-dev-generated-10k",
        label="fake",
        generator="flux.1-dev",
        method="parquet",
        license="see dataset card",
        url="https://huggingface.co/datasets/ash12321/flux-1-dev-generated-10k",
        notes="Rectified-flow transformer — a third architecture family "
              "alongside diffusion and the GPT-Image models.",
        limit=1500,
    ),
    Source(
        key="midjourney_v6",
        repo_id="Photoroom/midjourney-v6-recap",
        label="fake",
        generator="midjourney-v6",
        method="parquet",
        license="see dataset card",
        url="https://huggingface.co/datasets/Photoroom/midjourney-v6-recap",
        notes="Closed-model output, heavily stylised.",
        limit=1500,
    ),

    # ── SO-FAKE-OOD: one repo, thirteen modern generators, three platforms ────
    # The single highest-value addition available, for four reasons measured
    # rather than assumed:
    #
    #  1. It carries 1,334 GPT-Image-2 images — a *third* independent source for
    #     the exact generator the corpus scores 12% on from an unseen repo. That
    #     is the failure this rebuild exists to fix, and more sources per
    #     generator is the only intervention the evidence supports.
    #  2. Its fakes are JPEG, at 1024x1024. The corpus audit currently reports
    #     PNG as 100% fake — a pure container shortcut — and a large JPEG fake
    #     population is what removes it.
    #  3. Its reals are social-platform photographs at 1.5-1.9MP, which is the
    #     band the audit reports as 98.5% fake. They are also a fourth real
    #     provenance, against three curated photo sets today.
    #  4. Four generators and two platforms are quarantined below, so the
    #     held-out-generator tier stops resting on a single stylised source whose
    #     95.5% score was traced to it being a statistical outlier.
    #
    # The card states images "preserve their original file bytes. We do not
    # resize, crop, decode/re-encode, or normalize" — which is the same guarantee
    # this collector makes, so nothing here is laundered through a re-encode.
    Source(
        key="sofake",
        repo_id="saberzl/So-Fake-OOD",
        label="fake",                     # overridden per row by label_column
        generator="?",                    # overridden per row by generator_column
        method="parquet_rowgroups",
        license="CC-BY-NC-4.0 (non-commercial research)",
        url="https://huggingface.co/datasets/saberzl/So-Fake-OOD",
        hf_splits=["test_image"],
        rendition="mixed",                # per row, below
        # Direct API output vs. an image that has been through a social platform's
        # recompression. Kept as a field because the two regimes fail differently:
        # the shipped model scores 93% on wild GPT-Image-2 and 12% on pristine.
        rendition_column="source_type",
        rendition_map={"generator": "pristine", "platform": "wild"},
        # Keep the two whole-image classes; drop TAMPERED, which is a locally
        # retouched real photograph. Calling those fake would train the model to
        # reject a genuine photo because part of it was edited — a different task,
        # and one that would poison the fake class with mostly-real pixels.
        where={"label": ["REAL", "FULL_SYNTHETIC"]},
        label_column="authenticity",
        label_map={"REAL": "real", "FAKE": "fake"},
        generator_column="source_name",
        # Upstream spelling → corpus spelling. Densities are measured, from a
        # 1,999-row census of shard 0, and they matter because this repo is
        # thoroughly interleaved: what arrives comes in these proportions, so the
        # share is what decides whether a generator is usable as a tier.
        generator_map={
            "GPT-image-2": "gpt-image-2",              # 1.3%
            "GPT4o": "chatgpt-4o-native",              # 3.5%
            "GPT-image-1.5": "gpt-image-1.5",          # 3.4%
            "Flux.1_pro": "flux.1-pro",                # 3.5%
            "Hidream": "hidream",                      # 3.9%
            "Ideogram3": "ideogram-3",                 # 2.8%
            "Ideogram2": "ideogram-2",                 # 1.0%
            "ideogram": "ideogram-1",                  # 1.5%
            "imagen3": "imagen-3",                     # 2.7%
            "nano_banana": "nano-banana",              # 1.4%
            # Present in the data and absent from the first version of this map,
            # so both were being skipped. seedream4.5 is the costly one: the
            # corpus holds 200 seedream-4.5 images from one repo and nothing else,
            # and this is a second source for exactly that generator.
            "seedream4.5": "seedream-4.5",             # 1.2%
            # Quarantined below — five separate vendors, so the held-out tier is
            # not one lab's signature repeated.
            "FLUX_2": "flux-2",                        # 1.8%
            "Imagen4": "imagen-4",                     # 3.2%
            "Seedream3.0": "seedream-3.0",             # 3.4%
            "Recraftv3": "recraft-v3",                 # 2.8%
            "nano_banana_2": "nano-banana-2",          # 1.2%
            # Reals, by platform. Reddit trains; the other two are quarantined as
            # unseen-source negatives, which the held-out tiers badly need — 5%
            # FPR on the 100 reals available today is five images, a resolution
            # the protocol already flags as too coarse to report.
            "Reddit": "camera/social-reddit",          # 20.4%
            "Tumblr": "camera/social-tumblr",          # 3.2%
            "Bluesky": "camera/social-bluesky",        # 1.4%
            # source_name "openai" is 36.8% of rows and is the TAMPERED class,
            # dropped by `where` above before this map is consulted.
        },
        # Quarantine rule: a generator goes here when it appears *nowhere else*
        # in the corpus, so the held-out tier it feeds is honestly a held-out
        # GENERATOR. gpt-image-2, chatgpt-4o-native and seedream-4.5 are
        # deliberately not quarantined even though the corpus already holds them
        # from other repos — routing them to a `sofake_holdout` key would let them
        # read as a held-out *source* when the provenance, packaging and
        # rendition are all shared with the training rows. They train instead,
        # which is the honest use of them.
        holdout_when={"source_name": ["FLUX_2", "Imagen4", "Seedream3.0",
                                      "Recraftv3", "nano_banana_2",
                                      "Tumblr", "Bluesky"]},
        extra_fields=["source_type", "source_platform"],
        notes="So-Fake-OOD v3, an out-of-distribution benchmark built from "
              "social-platform reals and 13 modern photoreal generators. Taken "
              "here as a *training* source rather than an eval set, because the "
              "measured failure is per-source and this is the only verified repo "
              "with modern pristine output, declared licensing and per-row "
              "provenance. Its own X and Instagram rows are link-only and are "
              "not in the parquet, so only Reddit/Tumblr/Bluesky reals arrive.",
        # Sized by measured throughput, not by appetite. A 45-image slice ran
        # end-to-end at 2 row groups + 12 footers = 206 MB in 619 s — 333 KB/s,
        # and 40 of the 68 rows in a group survived the TAMPERED filter, so:
        #
        #   2.55 MB downloaded per usable image, ~8.0 s each, ~450 images/hour
        #   3,000 images ≈ 7.6 GB down ≈ 6.7 h → ~4.5 GB kept → corpus ~13.7 GB
        #   9,000 images ≈ 23 GB down  ≈ 20 h  →                corpus ~22.7 GB
        #
        # The earlier 9,000 was written before anything was measured and commits
        # most of a day to one source. 3,000 fits the stated 20-30 GB budget with
        # room for the others, and at the densities above it yields roughly 1,200
        # reals (mostly Reddit, which is the corpus's scarcest class — the
        # held-out tiers currently share 100 negatives, so 5% FPR is one image)
        # and ~1,800 fakes over 16 generators, most of them *pristine* API output.
        # Pristine is the regime the shipped model fails hardest on: 27% recall on
        # held-out-source gpt-image-2 against 97% on the wild variety it trained
        # on. Raise this if an overnight run is on.
        limit=3000,
    ),

    # ── REAL: modern photographs ─────────────────────────────────────────────
    Source(
        key="megalith",
        repo_id="bitmind/megalith-small",
        label="real",
        generator="camera/flickr-cc0",
        method="parquet",
        license="CC0",
        url="https://huggingface.co/datasets/bitmind/megalith-small",
        configs=[f"chunk_{i:04d}" for i in range(12)],
        notes="Flickr CC0 photographs with EXIF intact — genuine camera "
              "provenance. Capped at 1024px long edge and mostly 3:2 "
              "landscape, so check --report for aspect skew against the "
              "portrait-heavy GPT-Image sets before training.",
        limit=4500,
    ),
    Source(
        key="openimages",
        repo_id="bitmind/open-images-v7-subset",
        label="real",
        generator="camera/open-images",
        method="parquet",
        license="CC-BY-4.0 (per-image, see upstream)",
        url="https://huggingface.co/datasets/bitmind/open-images-v7-subset",
        notes="Broadens real-class subject matter beyond Flickr aesthetics. "
              "Measured: capped at 1024px long edge, mean 0.75MP, short sides "
              "split evenly between the 384-768 and 768-1400 bins — so it widens "
              "subject coverage but does not by itself reach the 1-2MP band "
              "where every modern generator sits.",
        limit=3200,
    ),
    Source(
        key="real_hires_camera",
        repo_id="bitmind/bm-subnet-real-fullsize",
        label="real",
        generator="camera/hires-original",
        method="parquet",
        license="see dataset card (aggregated real-image feed)",
        url="https://huggingface.co/datasets/bitmind/bm-subnet-real-fullsize",
        hf_splits=["2024_10_13_weekly", "2024_10_06_weekly", "2024_09_29_weekly"],
        min_short=1030,
        overfetch=50.0,
        notes="Camera originals at 2-22MP — the resolution band the real class "
              "was missing entirely. Megalith is capped at 1024px long edge, so "
              "before this source every image in the corpus above 768px short "
              "side was generated, and the audit measured 0.82 fake share in "
              "that bin.\n"
              "min_short=1030 is not a resolution preference, it excludes a "
              "specific hazard. 90% of this repo is exactly 1024x1024, and those "
              "rows are resampled rather than native: median high-pass residual "
              "std on a 224 native crop is 2.37, against 7.76 for Megalith, 6.2 "
              "for GPT-Image-1 and 10.1 for GPT-Image-2. Every real *and* fake "
              "source in the corpus is 5-10; the squares sit far below all of "
              "them. Importing 2,700 of them as real would teach 'smooth, "
              "texture-poor ⇒ real' — the opposite of the evidence the "
              "native-crop branch is built to use, and a direct route to missing "
              "smooth AI images. The 1030px floor drops them and keeps the "
              "genuine originals, at ~2.5% of rows; hence overfetch=50.",
        limit=1300,
    ),

    # ── HOLDOUT: measured against, never trained on ──────────────────────────
    Source(
        key="chatgpt4o_holdout",
        repo_id="wafflefan/felix-chatgpt-generated-images",
        label="fake",
        generator="chatgpt-4o-native",
        method="glob",
        pattern="originals/*.png",
        license="ALL RIGHTS RESERVED by uploader — eval only, do not train",
        url="https://huggingface.co/datasets/wafflefan/felix-chatgpt-generated-images",
        holdout=True,
        notes="1,202 original-byte PNGs saved straight out of ChatGPT from "
              "2025-03-26 on. Closest available proxy for 'an image a user "
              "made with ChatGPT and handed to the app'. Licence forbids "
              "training use; held out for measurement only.",
        limit=400,
    ),
    Source(
        key="liars_dividend_fake",
        repo_id="Sarim-Hash/liars-dividend-GPT-image-2-photoreal",
        label="fake",
        generator="gpt-image-2",
        method="glob",
        pattern="fake/*.png",
        license="see dataset card",
        url="https://huggingface.co/datasets/Sarim-Hash/liars-dividend-GPT-image-2-photoreal",
        holdout=True,
        notes="100 photoreal GPT-Image-2 images, caption-paired with the 100 "
              "reals below. Paired design means a real-vs-fake score "
              "difference cannot be explained by subject matter.",
        limit=100,
    ),
    Source(
        key="liars_dividend_real",
        repo_id="Sarim-Hash/liars-dividend-GPT-image-2-photoreal",
        label="real",
        generator="camera/coco-paired",
        method="glob",
        pattern="real/*.png",
        license="see dataset card",
        url="https://huggingface.co/datasets/Sarim-Hash/liars-dividend-GPT-image-2-photoreal",
        holdout=True,
        notes="Caption-matched real counterparts to liars_dividend_fake.",
        limit=100,
    ),
]

SOURCE_BY_KEY = {s.key: s for s in SOURCES}


# ─── Manifest / split helpers ────────────────────────────────────────────────
def split_for(digest: str) -> str:
    """Deterministic train/val/test assignment from the content hash."""
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < SPLIT_TRAIN:
        return "train"
    if bucket < SPLIT_VAL:
        return "val"
    return "test"


def load_manifest() -> Tuple[List[dict], set]:
    """Existing manifest rows plus the set of hashes already on disk.

    Makes the collector resumable and idempotent: an interrupted run resumes
    where it stopped, and the same image pulled from two sources is stored once.
    """
    if not MANIFEST_PATH.exists():
        return [], set()
    rows, seen = [], set()
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
        seen.add(row["sha256"])
    return rows, seen


class ManifestWriter:
    """Append-only manifest, flushed per image so a kill loses nothing."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def probe_image(raw: bytes) -> Optional[dict]:
    """Decode enough to validate and to record the shortcut-audit fields.

    A source that hands back HTML error pages or truncated files would
    otherwise land silently in the corpus as unreadable "images".
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            fmt, mode = im.format, im.mode
    except Exception:
        return None
    if not w or not h:
        return None
    return {
        "width": w,
        "height": h,
        "orig_format": (fmt or "?").upper(),
        "mode": mode,
        "aspect": round(w / h, 4),
        "long_edge": max(w, h),
        "orientation": "square" if 0.95 <= w / h <= 1.05
                       else ("landscape" if w > h else "portrait"),
    }


def store(raw: bytes, src: Source, seen: set, manifest: ManifestWriter,
          overrides: Optional[dict] = None) -> Optional[str]:
    """Write one image verbatim and record it. Returns the split, or None if
    skipped (duplicate, undecodable, or outside the source's resolution window).

    ``overrides`` carries per-row provenance for multi-generator sources —
    ``label``, ``generator``, ``holdout``, plus any ``extra_fields``. A row
    promoted to holdout here also gets a distinct ``source_key``: without that
    the evaluation would see a held-out image whose source it also trained on and
    correctly refuse to treat it as evidence.
    """
    digest = hashlib.sha256(raw).hexdigest()
    if digest in seen:
        return None
    meta = probe_image(raw)
    if meta is None:
        return None

    short = min(meta["width"], meta["height"])
    if src.min_short and short < src.min_short:
        return None
    if src.max_short and short > src.max_short:
        return None

    ov = dict(overrides or {})
    label = ov.pop("label", src.label)
    generator = ov.pop("generator", src.generator)
    row_holdout = bool(ov.pop("holdout", False))
    source_key = src.key + (src.holdout_suffix if row_holdout and not src.holdout
                            else "")

    split = "holdout" if (src.holdout or row_holdout) else split_for(digest)
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(meta["orig_format"], "bin")
    out_dir = OUTPUT_DIR / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{generator.replace('/', '-')}__{digest[:16]}.{ext}"
    (out_dir / filename).write_bytes(raw)          # verbatim: no re-encode

    seen.add(digest)
    manifest.write({
        "sha256": digest,
        "path": f"{split}/{label}/{filename}",
        "split": split,
        "label": label,
        "generator": generator,
        "source_key": source_key,
        "repo_id": src.repo_id,
        "license": src.license,
        "url": src.url,
        "rendition": src.rendition,
        **ov,                                      # whatever extra_fields carried
        **meta,
    })
    return split


# ─── Fetch strategies ────────────────────────────────────────────────────────
def _parallel_fetch(repo_id: str, names: List[str], workers: int = 12) -> Iterator[bytes]:
    """Yield the bytes of many repo files, downloading several at a time.

    One-at-a-time ``hf_hub_download`` measured ~3.7 s/image against
    ``Scam-AI/gpt-image-2``, which is over three hours for 3,000 images and is
    almost entirely request latency rather than bandwidth. A thread pool turns
    that into minutes. ``hf_hub_download`` is thread-safe and shares the local
    cache, so re-runs still cost nothing.

    Yields in completion order — irrelevant here because split assignment comes
    from the content hash, not arrival order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    def grab(name: str) -> Optional[bytes]:
        try:
            return Path(hf_hub_download(repo_id, name, repo_type="dataset")).read_bytes()
        except Exception as exc:                                   # noqa: BLE001
            print(f"    ! {name}: {type(exc).__name__}: {str(exc)[:90]}",
                  file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(grab, n): n for n in names}
        for fut in as_completed(futures):
            raw = fut.result()
            if raw:
                yield raw


def _repo_files(repo_id: str) -> set:
    """Every filename in the dataset repo, for pre-flight existence checks."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id)
    return {s.rfilename for s in (info.siblings or [])}


def iter_glob(src: Source, limit: int) -> Iterator[Tuple[bytes, dict]]:
    """Files matching a path pattern in the repo, fetched in parallel."""
    import fnmatch

    names = sorted(f for f in _repo_files(src.repo_id)
                   if fnmatch.fnmatch(f, src.pattern))
    if not names:
        raise RuntimeError(f"no files match {src.pattern!r} in {src.repo_id}")
    for raw in _parallel_fetch(src.repo_id, names[:limit]):
        yield raw, {}


def iter_jsonl_filtered(src: Source, limit: int) -> Iterator[Tuple[bytes, dict]]:
    """Scam-AI layout: metadata.jsonl selects which images to pull.

    Filtered to classification=confirmed. Taking every row would mean training
    on ~5,258 images whose generator was never verified — roughly 50% label
    noise in the class this whole effort is meant to fix.
    """
    from huggingface_hub import hf_hub_download

    meta_path = hf_hub_download(src.repo_id, "metadata.jsonl", repo_type="dataset")
    lines = [l for l in Path(meta_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("classification") != "confirmed":
            continue
        if str(row.get("download_success")) != "True":
            continue
        rows.append(row)

    # Highest-confidence first, so a --limit smaller than the set keeps the
    # cleanest labels rather than an arbitrary slice.
    rank = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda r: (rank.get(r.get("classification_confidence"), 3),
                             str(r.get("tweet_id"))))

    # Intersect with the actual file list: a few metadata rows have no image in
    # the repo, and skipping them here avoids paying a 404 round-trip each.
    present = _repo_files(src.repo_id)
    names = [n for n in (f"images/{r['tweet_id']}_{r['media_key']}.jpg" for r in rows)
             if n in present]
    print(f"    {len(rows)} confirmed of {len(lines)} rows; "
          f"{len(names)} have images in the repo")
    for raw in _parallel_fetch(src.repo_id, names[:limit]):
        yield raw, {}


def iter_parquet(src: Source, limit: int) -> Iterator[Tuple[bytes, dict]]:
    """Streamed parquet rows, so a 100k-row repo costs only what we take.

    Draws round-robin across ``src.configs`` when the repo is chunked, rather
    than draining chunk_0000 — consecutive rows in one chunk tend to share an
    upload batch, and for the real class that means correlated cameras.

    ``src.hf_splits`` covers repos that shard by split instead of by config. The
    two are crossed rather than treated as alternatives, so a repo that uses both
    is drawn from evenly; in practice a source sets one or the other.

    Rows failing ``where``/``where_not`` are skipped but still count against the
    scan budget, because streaming pays for them either way.
    """
    from datasets import load_dataset

    targets = [(c, s) for c in (src.configs or [None])
               for s in (src.hf_splits or ["train"])]
    per_target = max(1, limit // len(targets)) if len(targets) > 1 else limit
    taken = 0
    for cfg, hf_split in targets:
        if taken >= limit:
            return
        try:
            ds = (load_dataset(src.repo_id, cfg, split=hf_split, streaming=True)
                  if cfg else load_dataset(src.repo_id, split=hf_split,
                                           streaming=True))
            features = dict(getattr(ds, "features", None) or {})
            ds = _undecoded(ds)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    ! config={cfg} split={hf_split}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        from_this = 0
        for row in ds:
            if from_this >= per_target or taken >= limit:
                break
            if not _row_matches(src, row, features):
                continue
            overrides = _row_overrides(src, row, features)
            if overrides is None:
                continue
            raw = _raw_from_row(row, src.image_column)
            if raw is None:
                continue
            yield raw, overrides
            from_this += 1
            taken += 1


class RangeFile:
    """Minimal seekable read-only file over HTTP range requests.

    Exists because the obvious approach does not work on a slow link. Streaming
    the same repo through ``datasets`` moved bytes at roughly 58 KB/s against a
    measured 400 KB/s line — it buffers whole row groups and pays for the 37% of
    rows that are TAMPERED and dropped — and pointing pyarrow at fsspec's
    ``HfFileSystem`` never finished reading a footer at all, because parquet asks
    for the last handful of bytes in a file and a filesystem tuned for bulk
    sequential reads answers that by pulling blocks.

    Doing no read-ahead at all is what makes it fast: a footer then costs 1.5s
    and 224 KB in two requests, and a row group costs one coalesced read at line
    speed. Measured on ``saberzl/So-Fake-OOD``.
    """

    def __init__(self, url: str, session) -> None:
        # Resolve the CDN redirect once. Re-resolving per range doubles the
        # request count and every one of those is a round trip.
        head = session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.session = session
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.bytes_fetched = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (offset if whence == 0 else
                    self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self.pos
        if length == 0:
            return b""
        end = min(self.pos + length, self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"},
                             timeout=600)
        r.raise_for_status()
        data = r.content
        self.bytes_fetched += len(data)
        self.pos += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        # A property, not a method: pyarrow reads this as an attribute, and a
        # bound method is always truthy, which reads as "already closed".
        return False

    def close(self) -> None:
        pass


def _provenance_columns(src: Source) -> List[str]:
    """Every non-image column this source's filters and overrides consult."""
    cols = set(src.where) | set(src.where_not) | set(src.holdout_when)
    for c in (src.label_column, src.generator_column, src.rendition_column):
        if c:
            cols.add(c)
    cols.update(src.extra_fields)
    return sorted(cols)


def iter_parquet_rowgroups(src: Source, limit: int) -> Iterator[Tuple[bytes, dict]]:
    """Row groups fetched by HTTP range, round-robin across shards.

    Why not just download the shards
    --------------------------------
    ``hf_hub_download`` reads at line speed, which is the main thing streaming
    got wrong, but it commits to a whole file — 2.91 GB and about two hours per
    shard here — and cannot stop when the quota is met. A row group is ~102 MB, so
    ranging gives 28x finer stopping granularity, resumes at row-group boundaries
    rather than restarting a multi-GB file, and never leaves a partial download
    behind.

    Why round-robin rather than draining shard 0
    -------------------------------------------
    Same reason :func:`iter_parquet` crosses configs: adjacent rows in one shard
    share an upload batch. Here it was measured rather than assumed — a census of
    shard 0 found 15.67 distinct generators per 68-row row group, so the repo is
    thoroughly interleaved and any prefix is representative in *composition*.
    Spreading across shards still costs nothing and protects against the one
    thing the census could not rule out, which is a shard-level ordering the
    single-shard sample cannot see.

    What this deliberately does not do
    ----------------------------------
    It does not try to fetch only the rows that match ``where``. Fetching is
    whole-column-chunk, so a row group costs its full ~102 MB however few of its
    68 rows are wanted. With generators interleaved at ~1-4% each, selecting one
    would cost ~102 MB per image against 2.35 MB for taking everything keepable.
    So it takes everything that passes the filters and lets the composition fall
    where the repo puts it.
    """
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import HfApi

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(src.repo_id, repo_type="dataset")
                   if f.endswith(".parquet")
                   and (not src.hf_splits
                        or any(s in f for s in src.hf_splits)))
    if not files:
        print(f"    ! no parquet files matched {src.hf_splits} in {src.repo_id}",
              file=sys.stderr)
        return

    # Spread the chosen shards across the whole file list rather than taking a
    # prefix, so shard-level ordering cannot concentrate the sample.
    n = min(src.max_shards, len(files))
    picks = [files[round(i * (len(files) - 1) / max(1, n - 1))] for i in range(n)] \
        if n > 1 else files[:1]
    seen_picks, ordered = set(), []
    for p in picks:                     # dedupe while keeping the spread order
        if p not in seen_picks:
            seen_picks.add(p)
            ordered.append(p)

    session = requests.Session()
    columns = _provenance_columns(src) + [src.image_column or "image"]
    readers: List[Tuple[str, object, int]] = []
    for path in ordered:
        url = f"https://huggingface.co/datasets/{src.repo_id}/resolve/main/{path}"
        try:
            pf = pq.ParquetFile(RangeFile(url, session))
            readers.append((path, pf, pf.metadata.num_row_groups))
        except Exception as exc:                                   # noqa: BLE001
            print(f"    ! footer {path}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    if not readers:
        return
    available = {name for r in readers
                 for name in r[1].schema_arrow.names}
    want = [c for c in columns if c in available]
    missing = [c for c in columns if c not in available]
    if missing:
        print(f"    ! columns absent from {src.repo_id}: {missing}",
              file=sys.stderr)

    taken = 0
    rg_index = 0
    while taken < limit:
        progressed = False
        for path, pf, n_rg in readers:
            if taken >= limit:
                break
            if rg_index >= n_rg:
                continue
            progressed = True
            try:
                tbl = pf.read_row_group(rg_index, columns=want)
            except Exception as exc:                               # noqa: BLE001
                print(f"    ! {path} rg{rg_index}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                continue
            prov = {c: tbl.column(c).to_pylist()
                    for c in want if c != (src.image_column or "image")}
            images = tbl.column(src.image_column or "image")
            for i in range(tbl.num_rows):
                if taken >= limit:
                    break
                row = {c: prov[c][i] for c in prov}
                if not _row_matches(src, row, {}):
                    continue
                overrides = _row_overrides(src, row, {})
                if overrides is None:
                    continue
                val = images[i].as_py()
                raw = (val.get("bytes") if isinstance(val, dict)
                       else val if isinstance(val, (bytes, bytearray)) else None)
                if not raw:
                    continue
                yield bytes(raw), overrides
                taken += 1
            del tbl
        if not progressed:
            break
        rg_index += 1


def _row_value(row: dict, features: dict, column: str) -> Optional[str]:
    """One row field as a string, resolving ClassLabel ints to their names.

    A ClassLabel column streams as an integer. Filtering on the integer would
    encode positions that move if the upstream reorders its class list, so the
    name is the only stable thing to match on.
    """
    if column not in row:
        return None
    val = row[column]
    names = getattr(features.get(column), "names", None)
    if names is not None and isinstance(val, int):
        return names[val] if 0 <= val < len(names) else str(val)
    if val is None:
        return ""
    return val if isinstance(val, str) else str(val)


def _row_matches(src: Source, row: dict, features: dict) -> bool:
    """Allowlist then denylist. Both are explicit so an exclusion is visible.

    So-Fake-OOD's third class, TAMPERED, is a locally edited real rather than a
    generated image. Sweeping it into the fake class would train the model to
    call a genuine photograph fake because part of it was retouched, which is a
    different task; it is dropped by name here so the choice is on the record.
    """
    for col, allowed in src.where.items():
        if _row_value(row, features, col) not in allowed:
            return False
    for col, blocked in src.where_not.items():
        if _row_value(row, features, col) in blocked:
            return False
    return True


def _row_overrides(src: Source, row: dict, features: dict) -> Optional[dict]:
    """Per-row label, generator, holdout flag and extras. None means skip.

    Unmapped values are skipped with a warning rather than guessed at. An
    unrecognised generator spelling is the dangerous case: passed through, it
    would enter the corpus as a novel generator when it may be one already
    present under a different spelling, which quietly corrupts the
    held-out-generator tier.
    """
    ov: dict = {}

    if src.label_column:
        raw = _row_value(row, features, src.label_column)
        mapped = src.label_map.get(raw)
        if mapped is None:
            _warn_unmapped(src.label_column, raw)
            return None
        ov["label"] = mapped

    if src.generator_column:
        raw = _row_value(row, features, src.generator_column)
        mapped = src.generator_map.get(raw)
        if mapped is None:
            _warn_unmapped(src.generator_column, raw)
            return None
        ov["generator"] = mapped

    if src.rendition_column:
        raw = _row_value(row, features, src.rendition_column)
        mapped = src.rendition_map.get(raw)
        if mapped is None:
            _warn_unmapped(src.rendition_column, raw)
            return None
        ov["rendition"] = mapped

    for col, values in src.holdout_when.items():
        if _row_value(row, features, col) in values:
            ov["holdout"] = True
            break

    for col in src.extra_fields:
        val = _row_value(row, features, col)
        if val:
            ov[col] = val
    return ov


def _warn_unmapped(column: str, value: Optional[str]) -> None:
    """Once per unseen value — a new upstream class must not scroll past."""
    token = f"{column}={value!r}"
    if token in _WARNED_UNMAPPED:
        return
    _WARNED_UNMAPPED.add(token)
    print(f"    ! {token} has no mapping; skipping those rows. Add it to the "
          f"source's map to include them.", file=sys.stderr)


_WARNED_UNMAPPED: set = set()


def _undecoded(ds):
    """Turn off automatic image decoding so the stored bytes stay original.

    ``datasets`` decodes an ``Image`` feature to a PIL object on access. Writing
    that back out re-encodes it — which silently defeated this collector's
    verbatim guarantee and, because the fallback encoder is PNG, made every
    parquet-sourced real photo a PNG while the JPEG-sourced fakes stayed JPEG.
    That is a format shortcut pointing the wrong way, invented by the loader.
    ``decode=False`` yields the ``{"bytes", "path"}`` struct instead.
    """
    from datasets import Image as HFImage

    features = getattr(ds, "features", None) or {}
    for name, feat in features.items():
        if isinstance(feat, HFImage):
            ds = ds.cast_column(name, HFImage(decode=False))
    return ds


def _raw_from_row(row: dict, preferred: str) -> Optional[bytes]:
    """Pull original bytes out of a HF row without a decode/re-encode cycle.

    Returns ``None`` rather than re-encoding. A PIL object here means
    ``_undecoded`` failed to catch the column, and silently re-encoding it would
    stamp this source with a uniform container format — exactly the shortcut
    this collector exists to avoid. Better to drop the source loudly.
    """
    from PIL import Image

    candidates = [preferred, "image", "media_image", "jpg", "jpeg", "png",
                  "webp", "img"]
    for key in candidates:
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, dict) and val.get("bytes"):
            return val["bytes"]
        if isinstance(val, (bytes, bytearray)):
            return bytes(val)
        if isinstance(val, Image.Image):
            if key not in _WARNED_DECODED:
                _WARNED_DECODED.add(key)
                print(f"    ! column {key!r} arrived pre-decoded; skipping this "
                      f"source rather than re-encoding it (would create a "
                      f"format shortcut)", file=sys.stderr)
            return None
    return None


_WARNED_DECODED: set = set()


FETCHERS: Dict[str, Callable[[Source, int], Iterator[Tuple[bytes, dict]]]] = {
    "glob": iter_glob,
    "parquet": iter_parquet,
    "parquet_rowgroups": iter_parquet_rowgroups,
    "jsonl_filtered": iter_jsonl_filtered,
}


# ─── Collection ──────────────────────────────────────────────────────────────
def collect(keys: List[str], default_limit: int,
            target_override: Optional[int] = None) -> None:
    from tqdm import tqdm

    rows, seen = load_manifest()
    already = collections.Counter(r["source_key"] for r in rows)
    manifest = ManifestWriter(MANIFEST_PATH)

    print("=" * 78)
    print("  TruthLens — collecting modern-generator corpus")
    print(f"  → {OUTPUT_DIR}")
    print(f"  {len(rows):,} images already present")
    print("=" * 78)

    try:
        for key in keys:
            src = SOURCE_BY_KEY[key]
            cap = target_override if target_override is not None else (
                src.limit or default_limit
            )
            # Rows this source quarantined per-row live under a suffixed
            # source_key; counting only the bare key would make every re-run
            # re-collect them.
            have = already[key] + already[key + src.holdout_suffix]
            todo = max(0, cap - have)
            tag = "  [HOLDOUT — never trained on]" if src.holdout else ""
            print(f"\n[{key}] {src.repo_id}")
            what = (f"{src.label_column or 'label'}+{src.generator_column} per row"
                    if src.label_column or src.generator_column
                    else f"{src.label}/{src.generator}")
            print(f"    {what} — have {have}, target {cap}{tag}")
            if todo == 0:
                print("    already at target, skipping")
                continue

            counts: collections.Counter = collections.Counter()
            bar = tqdm(total=todo, desc=f"  {key}", ncols=78, unit="img")
            try:
                # Small additive margin, not a percentage: duplicates and
                # undecodable files are rare, and the parallel fetcher submits
                # every requested name up front — an over-fetch multiplier would
                # download thousands of files only to discard them.
                #
                # ``overfetch`` is the exception, for sources with a resolution
                # window. There the reject rate is the point rather than an
                # accident, so the budget has to be scaled or the fetcher stops
                # yielding long before the target is met.
                budget = int(todo * src.overfetch) + 32
                if src.min_short or src.max_short:
                    window = (f"{src.min_short or 0}-"
                              f"{src.max_short or '∞'}px short side")
                    print(f"    resolution window {window}; "
                          f"scanning up to {budget:,} rows")
                for raw, overrides in FETCHERS[src.method](src, budget):
                    split = store(raw, src, seen, manifest, overrides)
                    if split is None:
                        continue
                    counts[split] += 1
                    bar.update(1)
                    if sum(counts.values()) >= todo:
                        break
            except Exception as exc:                               # noqa: BLE001
                print(f"\n    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                bar.close()
            print(f"    +{sum(counts.values())} "
                  f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items())) or 'none'})")
    finally:
        manifest.close()

    print()
    report()


# ─── Shortcut audit ──────────────────────────────────────────────────────────
def report() -> None:
    """Print composition and the confounds that would inflate offline scores."""
    rows, _ = load_manifest()
    if not rows:
        print("No manifest yet — run a collection first.")
        return

    train_like = [r for r in rows if r["split"] != "holdout"]

    print("=" * 78)
    print("  CORPUS COMPOSITION")
    print("=" * 78)
    print(f"{'source':<24}{'label':<6}{'generator':<22}{'n':>7}")
    print("-" * 78)
    by_src = collections.Counter(
        (r["source_key"], r["label"], r["generator"]) for r in rows
    )
    # Read the holdout flag off the manifest rather than the registry: a source
    # that quarantines per row emits a source_key the registry has never heard of.
    quarantined = {r["source_key"] for r in rows if r["split"] == "holdout"}
    for (key, label, gen), n in sorted(by_src.items(), key=lambda kv: -kv[1]):
        flag = " *" if key in quarantined else ""
        print(f"{key:<24}{label:<6}{gen:<22}{n:>7}{flag}")
    print("-" * 78)
    print("  * = holdout, excluded from train/val/test")

    print(f"\n{'split':<12}{'real':>8}{'fake':>8}{'total':>8}")
    print("-" * 40)
    for split in ("train", "val", "test", "holdout"):
        r = sum(1 for x in rows if x["split"] == split and x["label"] == "real")
        f = sum(1 for x in rows if x["split"] == split and x["label"] == "fake")
        if r or f:
            print(f"{split:<12}{r:>8}{f:>8}{r + f:>8}")
    print("-" * 40)
    print(f"{'TOTAL':<12}{sum(1 for x in rows if x['label'] == 'real'):>8}"
          f"{sum(1 for x in rows if x['label'] == 'fake'):>8}{len(rows):>8}")

    # ── Shortcut checks ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SHORTCUT AUDIT — train/val/test only")
    print("  A feature that separates the classes here but not in the wild "
          "will\n  inflate offline accuracy and still fail on a real upload.")
    print("=" * 78)

    for field_name, title in (("orientation", "ASPECT / ORIENTATION"),
                              ("orig_format", "CONTAINER FORMAT")):
        print(f"\n{title}")
        vals = sorted({r[field_name] for r in train_like})
        print(f"  {'':<12}" + "".join(f"{v:>12}" for v in vals))
        share: Dict[str, Dict[str, float]] = {}
        for label in ("real", "fake"):
            sub = [r for r in train_like if r["label"] == label]
            cnt = collections.Counter(r[field_name] for r in sub)
            total = max(len(sub), 1)
            share[label] = {v: cnt[v] / total for v in vals}
            print(f"  {label:<12}" + "".join(
                f"{cnt[v]:>7} {share[label][v] * 100:>3.0f}%" for v in vals))
        worst = max(vals, key=lambda v: abs(share["real"][v] - share["fake"][v]),
                    default=None)
        if worst is not None:
            gap = abs(share["real"][worst] - share["fake"][worst]) * 100
            verdict = "OK" if gap < 20 else ("WATCH" if gap < 45 else "SHORTCUT RISK")
            print(f"  → largest gap: {worst} differs by {gap:.0f} points "
                  f"between classes  [{verdict}]")

    print("\nLONG EDGE (px)")
    for label in ("real", "fake"):
        edges = sorted(r["long_edge"] for r in train_like if r["label"] == label)
        if not edges:
            continue
        mid = edges[len(edges) // 2]
        print(f"  {label:<6} min={edges[0]:>5}  median={mid:>5}  max={edges[-1]:>5}")

    print("\nPER-GENERATOR COUNTS (train split — what the model actually sees)")
    tr = collections.Counter(r["generator"] for r in rows if r["split"] == "train")
    for gen, n in tr.most_common():
        print(f"  {gen:<24}{n:>7}")
    print()


def purge(keys: List[str]) -> None:
    """Delete every image and manifest row belonging to the given sources.

    Needed whenever a source turns out to have been ingested wrongly — the
    manifest is append-only and ``collect`` treats existing rows as done, so a
    bad batch would otherwise persist and count toward the target forever.
    """
    rows, _ = load_manifest()
    if not rows:
        print("Nothing to purge — no manifest.")
        return
    # A source that quarantines per row wrote some images under a suffixed
    # source_key. Purging the bare key alone would leave those behind, and they
    # would keep counting toward the target on the next run.
    targets = set(keys) | {k + s.holdout_suffix
                           for k in keys
                           for s in [SOURCE_BY_KEY.get(k)] if s}
    doomed = [r for r in rows if r["source_key"] in targets]
    kept = [r for r in rows if r["source_key"] not in targets]
    if not doomed:
        print(f"No rows for {', '.join(keys)}.")
        return

    removed = 0
    for r in doomed:
        path = OUTPUT_DIR / r["path"]
        if path.exists():
            path.unlink()
            removed += 1
    MANIFEST_PATH.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept),
        encoding="utf-8")
    print(f"Purged {len(doomed)} manifest rows for {', '.join(keys)} "
          f"({removed} files deleted). {len(kept)} rows remain.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="all",
                    help="comma-separated source keys, or 'all' / 'fake' / 'real' / 'holdout'")
    ap.add_argument("--limit", type=int, default=1200,
                    help="max images per source (default 1200)")
    ap.add_argument(
        "--target", type=int, default=None,
        help="absolute target for each selected source, overriding its built-in "
             "limit; useful for small resumable collection batches",
    )
    ap.add_argument("--report", action="store_true",
                    help="print composition + shortcut audit and exit")
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    ap.add_argument("--purge", default="",
                    help="comma-separated source keys to delete from the corpus")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            tag = " [HOLDOUT]" if s.holdout else ""
            print(f"{s.key:<24}{s.label:<6}{s.generator:<22}{s.repo_id}{tag}")
            print(f"{'':<24}licence: {s.license}")
        return 0

    if args.purge:
        purge([k.strip() for k in args.purge.split(",") if k.strip()])
        return 0

    if args.report:
        report()
        return 0

    sel = args.sources.strip()
    if sel == "all":
        keys = [s.key for s in SOURCES]
    elif sel in ("fake", "real"):
        keys = [s.key for s in SOURCES if s.label == sel and not s.holdout]
    elif sel == "holdout":
        keys = [s.key for s in SOURCES if s.holdout]
    elif sel == "train":
        keys = [s.key for s in SOURCES if not s.holdout]
    else:
        keys = [k.strip() for k in sel.split(",") if k.strip()]
        unknown = [k for k in keys if k not in SOURCE_BY_KEY]
        if unknown:
            print(f"unknown source(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(SOURCE_BY_KEY)}", file=sys.stderr)
            return 2

    collect(keys, args.limit, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
