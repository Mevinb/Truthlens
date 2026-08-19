#!/usr/bin/env python3
"""
TruthLens — Per-Generator Baseline Evaluation
==============================================
Scores the *current* checkpoint against ``datasets/prepared/modern_v2``, broken out by
generator, to turn "it failed on the two images I made with ChatGPT" into a
number.

Why per-generator and not one accuracy
--------------------------------------
``diag_benchmark.py`` reports accuracy on ``datasets/prepared/multires/test``, whose fake
class is CIFAKE + older SD/GAN + DALL-E 3. A model can score 92.8% there and
still be at chance on GPT-Image-2, because that generator is absent from both
the train and test halves of that corpus — the test split cannot detect a gap
that the corpus itself does not contain. This script reads the ``generator``
field from ``manifest.jsonl`` and reports accuracy per generator, so a
collapse on one family is visible instead of averaged away.

Read the output as follows: for the fake class, accuracy *is* recall, so a
per-generator number near 0 means every image of that generator was waved
through as REAL. That is the failure being measured.

Usage
-----
    python diag_modern_baseline.py                       # all splits
    python diag_modern_baseline.py --split holdout       # the clean eval sets
    python diag_modern_baseline.py --limit 150 --model resnet18
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = ROOT_DIR / "datasets" / "prepared" / "modern_v2"
MANIFEST = CORPUS_DIR / "manifest.jsonl"
OUT_PATH = ROOT_DIR / "results" / "metrics" / "modern_generator_baseline.json"


def load_rows(split: str | None, limit_per_gen: int) -> List[dict]:
    if not MANIFEST.exists():
        sys.exit(f"No manifest at {MANIFEST}\n"
                 f"Run: python dataset/collect_modern_generators.py")
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if split and split != "all":
        rows = [r for r in rows if r["split"] == split]

    # Cap per (generator, label) so one large source does not dominate runtime.
    kept: List[dict] = []
    seen: collections.Counter = collections.Counter()
    for r in sorted(rows, key=lambda x: x["sha256"]):
        key = (r["generator"], r["label"])
        if seen[key] >= limit_per_gen:
            continue
        seen[key] += 1
        kept.append(r)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all",
                    help="train | val | test | holdout | all (default all)")
    ap.add_argument("--limit", type=int, default=120,
                    help="max images per generator+label (default 120)")
    ap.add_argument("--model", default="resnet18",
                    help="model_type passed to predict_image (default resnet18)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT_DIR))
    from PIL import Image
    from tqdm import tqdm

    from src.predict import predict_image
    from src.utils import Config, resolve_cnn_checkpoint

    cfg = resolve_cnn_checkpoint(Config())

    rows = load_rows(args.split, args.limit)
    if not rows:
        sys.exit(f"No manifest rows for split={args.split!r}")

    print("=" * 74)
    print(f"  Per-generator baseline — model={args.model}  "
          f"checkpoint={cfg.cnn_model_name}")
    print(f"  {len(rows)} images from {CORPUS_DIR.relative_to(ROOT_DIR)} (split={args.split})")
    print("=" * 74)

    # correct/total and mean P(FAKE) per (generator, label)
    stats: Dict[tuple, Dict[str, float]] = collections.defaultdict(
        lambda: {"n": 0, "correct": 0, "p_fake_sum": 0.0})
    failures: List[dict] = []

    for r in tqdm(rows, desc="scoring", ncols=74, unit="img"):
        path = CORPUS_DIR / r["path"]
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
            result, _ = predict_image(img, cfg, model_type=args.model,
                                      with_gradcam=False)
        except Exception as exc:                                   # noqa: BLE001
            print(f"\n  ! {r['path']}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue

        truth = r["label"].upper()          # REAL | FAKE
        pred = result["prediction"]
        p_fake = float(result["probabilities"]["FAKE"])

        s = stats[(r["generator"], r["label"])]
        s["n"] += 1
        s["correct"] += int(pred == truth)
        s["p_fake_sum"] += p_fake
        if pred != truth:
            failures.append({"path": r["path"], "truth": truth, "pred": pred,
                             "p_fake": round(p_fake, 2),
                             "generator": r["generator"]})

    if not stats:
        sys.exit("Nothing scored — check that image files exist under "
                 f"{CORPUS_DIR}")

    print("\n" + "=" * 74)
    print("  ACCURACY BY GENERATOR")
    print("  For fake rows accuracy == recall: 0% means every image of that")
    print("  generator was passed through as REAL.")
    print("=" * 74)
    print(f"{'generator':<26}{'label':<7}{'n':>5}{'acc':>8}{'mean P(FAKE)':>14}")
    print("-" * 74)

    payload = {"checkpoint": cfg.cnn_model_name, "model_type": args.model,
               "split": args.split, "by_generator": {}}

    for (gen, label), s in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        acc = s["correct"] / s["n"]
        mean_p = s["p_fake_sum"] / s["n"]
        flag = ""
        if label == "fake" and acc < 0.25:
            flag = "  << MISSED"
        elif label == "real" and acc < 0.5:
            flag = "  << FALSE ALARMS"
        print(f"{gen:<26}{label:<7}{s['n']:>5}{acc * 100:>7.1f}%"
              f"{mean_p:>13.1f}%{flag}")
        payload["by_generator"][f"{gen}|{label}"] = {
            "n": s["n"], "accuracy": round(acc, 4),
            "mean_p_fake": round(mean_p, 2),
        }

    print("-" * 74)
    for label in ("real", "fake"):
        sub = [s for (g, l), s in stats.items() if l == label]
        n = sum(s["n"] for s in sub)
        c = sum(s["correct"] for s in sub)
        if n:
            print(f"{'ALL ' + label:<26}{'':<7}{n:>5}{c / n * 100:>7.1f}%")
            payload[f"overall_{label}"] = {"n": n, "accuracy": round(c / n, 4)}

    n_all = sum(s["n"] for s in stats.values())
    c_all = sum(s["correct"] for s in stats.values())
    print(f"{'OVERALL':<26}{'':<7}{n_all:>5}{c_all / n_all * 100:>7.1f}%")
    payload["overall"] = {"n": n_all, "accuracy": round(c_all / n_all, 4)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT_DIR)}")
    print(f"{len(failures)} misclassified of {n_all}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
