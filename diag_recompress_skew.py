"""Measure the train/serve recompression skew.

Training applies ``RandomRecompress(p=1.0, quality=(40, 98))`` to every image and
validation applies it at a fixed q88, so *every* sample the model has ever been
scored against during fitting carries a fresh JPEG grid. The serving path in
``src/predict.py`` applies no recompression at all. For a source that is already
JPEG the gap is only double- versus single-compression, but for a lossless PNG
it is "has an 8x8 JPEG grid" versus "has none whatsoever" — a large shift in
exactly the high-frequency band the classifier reads. A ChatGPT download is a
lossless PNG.

This script scores the same images twice: as they arrive, and after an in-memory
JPEG round trip. If the round trip moves P(FAKE) substantially on the PNG
sources, the skew is real and the fix belongs in training (lower ``p`` so the
model also sees pristine images) rather than in serving (which would stamp a
JPEG grid over the sensor noise that is the actual evidence).

Reals are measured alongside the fakes on purpose: a round trip that pushes
*everything* toward FAKE is a global shift, not recovered evidence, and would
show up here as both classes moving together.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import Config, resolve_cnn_checkpoint   # noqa: E402
from src.predict import predict_image                   # noqa: E402

OUT_PATH = Path("results/metrics/recompress_skew.json")


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode to JPEG in memory and decode back, exactly as RandomRecompress does."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="datasets/prepared/modern_v2")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--limit", type=int, default=60,
                    help="images per generator")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--qualities", default="95,88,75",
                    help="comma-separated JPEG qualities to test")
    args = ap.parse_args()

    qualities = [int(q) for q in args.qualities.split(",")]
    base = Path(args.corpus)
    df = pd.read_json(base / "manifest.jsonl", lines=True)
    df = df[df.split == args.split]

    # Deterministic subsample per generator so reruns are comparable.
    rows = (df.sort_values("sha256")
              .groupby("generator", group_keys=False)
              .head(args.limit))

    cfg = resolve_cnn_checkpoint(Config())
    # Accumulators: gen -> variant -> list of P(FAKE)
    probs: dict = defaultdict(lambda: defaultdict(list))
    labels: dict = {}

    variants = ["as-is"] + [f"jpeg-q{q}" for q in qualities]

    print("=" * 74)
    print(f"  Recompression skew — {len(rows)} images from "
          f"{args.corpus} (split={args.split})")
    print(f"  variants: {', '.join(variants)}")
    print("=" * 74)

    for n, row in enumerate(rows.itertuples(), 1):
        path = base / row.path
        try:
            src = Image.open(path).convert("RGB")
        except Exception as exc:                     # pragma: no cover
            print(f"  ! skip {row.path}: {exc}")
            continue

        labels[row.generator] = row.label
        for variant in variants:
            img = src if variant == "as-is" else \
                jpeg_roundtrip(src, int(variant.split("q")[1]))
            result, _ = predict_image(img, cfg, model_type=args.model,
                                      with_gradcam=False)
            # _build_result reports percentages under upper-case keys.
            p_fake = float(result["probabilities"]["FAKE"]) / 100.0
            probs[row.generator][variant].append(p_fake)

        if n % 20 == 0:
            print(f"  ... {n}/{len(rows)}")

    # ---------------------------------------------------------------- report
    print()
    print("=" * 74)
    print("  MEAN P(FAKE) BY VARIANT")
    print("  A JPEG round trip that recovers real evidence moves the fakes up")
    print("  while leaving the reals put. One that moves both is a global shift.")
    print("=" * 74)
    header = f"{'generator':<26}{'label':<7}{'n':>4}  " + \
             "".join(f"{v:>11}" for v in variants)
    print(header)
    print("-" * len(header))

    summary = {}
    for gen in sorted(probs, key=lambda g: (labels[g], g)):
        per = probs[gen]
        n = len(per["as-is"])
        cells = "".join(f"{sum(per[v]) / len(per[v]) * 100:>10.1f}%"
                        for v in variants)
        print(f"{gen:<26}{labels[gen]:<7}{n:>4}  {cells}")
        summary[gen] = {
            "label": labels[gen], "n": n,
            "mean_p_fake": {v: sum(per[v]) / len(per[v]) for v in variants},
        }

    print()
    print("=" * 74)
    print("  ACCURACY BY VARIANT  (threshold 0.5)")
    print("=" * 74)
    print(header)
    print("-" * len(header))
    for gen in sorted(probs, key=lambda g: (labels[g], g)):
        per = probs[gen]
        want_fake = labels[gen] == "fake"
        cells = ""
        for v in variants:
            correct = sum(1 for p in per[v] if (p >= 0.5) == want_fake)
            acc = correct / len(per[v])
            cells += f"{acc * 100:>10.1f}%"
            summary[gen].setdefault("acc", {})[v] = acc
        print(f"{gen:<26}{labels[gen]:<7}{len(per['as-is']):>4}  {cells}")

    # Balanced accuracy per variant across all generators present.
    print()
    print("  BALANCED ACCURACY (macro over real/fake pools)")
    for v in variants:
        pools = {"real": [], "fake": []}
        for gen, per in probs.items():
            pools[labels[gen]].extend(per[v])
        accs = []
        for lab, ps in pools.items():
            if not ps:
                continue
            want_fake = lab == "fake"
            accs.append(sum(1 for p in ps if (p >= 0.5) == want_fake) / len(ps))
        if len(accs) == 2:
            print(f"    {v:<12} {sum(accs) / len(accs) * 100:5.1f}%")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"corpus": args.corpus, "split": args.split, "model": args.model,
         "variants": variants, "per_generator": summary}, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
