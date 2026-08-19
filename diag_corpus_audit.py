#!/usr/bin/env python3
"""
TruthLens — Combined Corpus Shortcut Audit
===========================================
Audits the *union* of corpus roots that training will actually see, looking for
properties that correlate with the label but have nothing to do with whether an
image was generated.

Why audit the union and not each corpus
---------------------------------------
Each corpus can be individually balanced and the union still leak.
``datasets/prepared/cifake`` and ``datasets/prepared/multires`` can be balanced
within themselves; ``datasets/prepared/modern_v2`` can be balanced within itself.
But if one contributes mostly 32px images to both classes and the other
contributes 1024px images that are mostly fake, the union hands the model
"large ⇒ fake" for free. Training reads the union, so the union is what has to be
checked.

What each panel means
---------------------
* **format** — container the bytes arrived in. A split here is the classic
  provenance leak: generators emit PNG, photo archives emit JPEG.
  ``RandomRecompress`` in the train pipeline is the mitigation; this panel says
  how much work it has to do.
* **short side** — resolution band. Matters more than it used to, because
  ``NativeScaleCrop`` samples 1:1 crops only from images large enough to have a
  native scale. If large images skew one way, "was cropped at native scale"
  becomes a class hint on its own.
* **orientation** — portrait/landscape/square. Generators default to square or
  portrait; photo archives are mostly 3:2 landscape. ``RandomAspectPad`` is the
  mitigation.
* **megapixels** — a coarser view of the same resolution question, useful when
  the short-side bins are too granular to read.

Reading the output: the **skew** column is the fake share of each bin. 0.50 means
the bin carries no class information. Bins flagged ``LEAK`` hold enough of the
corpus to matter *and* lean hard one way — those are the ones worth fixing at the
corpus level rather than papering over with augmentation.

Usage
-----
    python diag_corpus_audit.py datasets/prepared/modern_v2 datasets/prepared/multires
    python diag_corpus_audit.py datasets/prepared/modern_v2 --split test
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# A bin has to hold at least this share of the corpus before a skew in it is
# worth reporting — otherwise every long-tail resolution reads as a leak.
MIN_SHARE = 0.02
# Fake share further than this from 0.50 counts as leaning.
SKEW_TOL = 0.20

SHORT_BINS = [(0, 64), (64, 192), (192, 384), (384, 768), (768, 1400),
              (1400, 10 ** 9)]


def bin_label(value: int, bins: List[Tuple[int, int]]) -> str:
    for lo, hi in bins:
        if lo <= value < hi:
            return f"{lo}-{hi if hi < 10 ** 9 else '∞'}px"
    return "?"


def scan(roots: List[Path], split: str, limit: int) -> List[dict]:
    """Read image metadata without decoding pixels where possible."""
    from PIL import Image

    from tqdm import tqdm

    files: List[Tuple[Path, str, str]] = []
    for root in roots:
        base = root / split
        if not base.exists():
            print(f"  ! {base} does not exist — skipped", file=sys.stderr)
            continue
        for label in ("real", "fake"):
            d = base / label
            if not d.exists():
                continue
            found = [p for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
            if limit:
                # Stride rather than truncate: taking the first N of a sorted
                # listing samples one generator's filename prefix, not the corpus.
                step = max(1, len(found) // limit)
                found = found[::step][:limit]
            files += [(p, label, root.name) for p in found]

    rows: List[dict] = []
    for path, label, corpus in tqdm(files, desc="  scanning", ncols=70,
                                    unit="img", leave=False):
        try:
            with Image.open(path) as im:
                w, h = im.size
                fmt = (im.format or "?").upper()
        except Exception:                                          # noqa: BLE001
            continue
        rows.append({
            "label": label, "corpus": corpus, "format": fmt,
            "short": min(w, h), "mp": w * h / 1e6,
            "orientation": ("square" if 0.95 <= w / h <= 1.05
                            else "landscape" if w > h else "portrait"),
        })
    return rows


def panel(rows: List[dict], name: str, keyfn) -> dict:
    """Print one distribution panel and return it as data."""
    total = len(rows)
    counts: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for r in rows:
        counts[keyfn(r)][r["label"]] += 1

    print(f"\n  {name}")
    print(f"    {'bin':<16}{'real':>7}{'fake':>7}{'share':>8}{'skew':>8}")
    print("    " + "-" * 46)

    out = {}
    for key in sorted(counts, key=lambda k: -sum(counts[k].values())):
        real = counts[key]["real"]
        fake = counts[key]["fake"]
        n = real + fake
        share = n / total
        skew = fake / n if n else 0.0
        flag = ""
        if share >= MIN_SHARE and abs(skew - 0.5) > SKEW_TOL:
            flag = "  LEAK"
        print(f"    {key:<16}{real:>7}{fake:>7}{share * 100:>7.1f}%"
              f"{skew:>8.2f}{flag}")
        out[key] = {"real": real, "fake": fake,
                    "share": round(share, 4), "fake_share": round(skew, 4),
                    "leak": bool(flag)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+",
                    help="corpus roots, e.g. datasets/prepared/modern_v2 datasets/prepared/multires")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=4000,
                    help="max images sampled per root+class (0 = all)")
    args = ap.parse_args()

    roots = [ROOT_DIR / r for r in args.roots]
    rows = scan(roots, args.split, args.limit)
    if not rows:
        sys.exit("Nothing scanned.")

    print("=" * 74)
    print(f"  CORPUS SHORTCUT AUDIT — split={args.split}")
    print(f"  {len(rows)} images from {', '.join(r.name for r in roots)}")
    print("=" * 74)
    n_real = sum(1 for r in rows if r["label"] == "real")
    print(f"  class balance: real {n_real}  fake {len(rows) - n_real}"
          f"  (fake share {(len(rows) - n_real) / len(rows):.2f})")

    payload = {
        "split": args.split,
        "roots": [r.name for r in roots],
        "n": len(rows),
        "class_balance": {"real": n_real, "fake": len(rows) - n_real},
        "panels": {
            "corpus":      panel(rows, "corpus contribution", lambda r: r["corpus"]),
            "format":      panel(rows, "container format", lambda r: r["format"]),
            "short_side":  panel(rows, "short side (native-crop eligibility)",
                                 lambda r: bin_label(r["short"], SHORT_BINS)),
            "orientation": panel(rows, "orientation", lambda r: r["orientation"]),
            "megapixels":  panel(rows, "megapixels",
                                 lambda r: ("<0.5MP" if r["mp"] < 0.5 else
                                            "0.5-1MP" if r["mp"] < 1 else
                                            "1-2MP" if r["mp"] < 2 else
                                            "2-6MP" if r["mp"] < 6 else "6MP+")),
        },
    }

    leaks = [(p, b) for p, panel_data in payload["panels"].items()
             for b, v in panel_data.items() if v["leak"]]
    print("\n" + "=" * 74)
    if leaks:
        print(f"  {len(leaks)} leaning bin(s) — each is a property the model can")
        print("  read instead of the generator fingerprint:")
        for p, b in leaks:
            v = payload["panels"][p][b]
            print(f"    {p:<12}{b:<16}fake share {v['fake_share']:.2f}"
                  f"  ({v['share'] * 100:.0f}% of corpus)")
        print("\n  Augmentation covers format and orientation "
              "(RandomRecompress / RandomAspectPad).")
        print("  A resolution leak is not augmentable — fix it by adding "
              "images to the thin class.")
    else:
        print("  No leaning bins above the reporting threshold.")
    print("=" * 74)

    out = ROOT_DIR / "results" / "metrics" / f"corpus_audit_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
