#!/usr/bin/env python3
"""
TruthLens — Unified Evaluation Board
=====================================
Scores one checkpoint against every evaluation set that exists and prints them
side by side, so "did this change help?" has a single answer instead of four
scripts to reconcile.

The sets, and what each is evidence of
--------------------------------------
================  ===========================================================
``multires``      ``datasets/prepared/multires/test`` — the corpus the shipped checkpoint
                  was trained on (CIFAKE, older SD/GAN, DALL-E 3). Watched for
                  *regression*: a fix for modern generators that destroys this
                  has traded one failure for another.
``modern``        ``datasets/prepared/modern_v2/test`` — held out from the modern corpus,
                  broken out per generator. Same uploads as the training half,
                  so a high score here is necessary but not sufficient.
``holdout``       ``datasets/prepared/modern_v2/holdout`` — quarantined at collection
                  time and never trained on, from *different* uploaders than
                  the training sources. Independent evidence.
``internet``      ``datasets/evaluation/internet`` — Wikimedia Commons, unrelated to every
                  training source, both classes passed through one thumbnailer
                  so format and resolution carry no class information. The
                  closest thing here to what a user actually uploads.
================  ===========================================================

Read ``internet`` and ``holdout`` first. ``modern`` rising while those stay flat
means the model learned the corpus, not the generators.

Everything goes through ``predict_image``, i.e. the full TTA path the app serves,
so these numbers describe the product rather than a bare forward pass.

Usage
-----
    python diag_eval_all.py                      # every set, 150 imgs per group
    python diag_eval_all.py --sets internet holdout
    python diag_eval_all.py --limit 300 --tag after-retrain
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "results" / "metrics"

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ─── Set definitions ──────────────────────────────────────────────────────────
def rows_from_manifest(corpus: Path, split: Optional[str],
                       group_field: str) -> List[dict]:
    """Rows from a manifest.jsonl, tagged with the field used to group results."""
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        return []
    out = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if split and r.get("split") != split:
            continue
        out.append({
            "path": corpus / r["path"],
            "label": r["label"],
            "group": r.get(group_field, "?"),
        })
    return out


def rows_from_tree(root: Path) -> List[dict]:
    """Rows from a plain real/ + fake/ directory tree (no manifest)."""
    out = []
    for label in ("real", "fake"):
        d = root / label
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                out.append({"path": p, "label": label, "group": label})
    return out


def collect_sets(names: List[str]) -> Dict[str, List[dict]]:
    sets: Dict[str, List[dict]] = {}
    if "multires" in names:
        sets["multires"] = rows_from_tree(ROOT_DIR / "datasets" / "prepared" / "multires" / "test")
    if "modern" in names:
        sets["modern"] = rows_from_manifest(
            ROOT_DIR / "datasets" / "prepared" / "modern_v2", "test", "generator")
    if "holdout" in names:
        sets["holdout"] = rows_from_manifest(
            ROOT_DIR / "datasets" / "prepared" / "modern_v2", "holdout", "generator")
    if "internet" in names:
        sets["internet"] = rows_from_manifest(
            ROOT_DIR / "datasets" / "evaluation" / "internet", None, "tag")
    return {k: v for k, v in sets.items() if v}


def cap_per_group(rows: List[dict], limit: int) -> List[dict]:
    """Cap rows per (group, label). Sorted by filename so runs are comparable."""
    kept, seen = [], collections.Counter()
    for r in sorted(rows, key=lambda x: x["path"].name):
        key = (r["group"], r["label"])
        if seen[key] >= limit:
            continue
        seen[key] += 1
        kept.append(r)
    return kept


# ─── Scoring ──────────────────────────────────────────────────────────────────
def score(rows: List[dict], cfg, model_type: str) -> Tuple[dict, List[dict]]:
    from PIL import Image
    from tqdm import tqdm

    from src.predict import predict_image

    stats: Dict[tuple, Dict[str, float]] = collections.defaultdict(
        lambda: {"n": 0, "correct": 0, "p_fake_sum": 0.0})
    misses: List[dict] = []

    for r in tqdm(rows, desc="  scoring", ncols=70, unit="img", leave=False):
        if not r["path"].exists():
            continue
        try:
            with Image.open(r["path"]) as im:
                img = im.convert("RGB")
            result, _ = predict_image(img, cfg, model_type=model_type,
                                      with_gradcam=False)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {r['path'].name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        truth = r["label"].upper()
        pred = result["prediction"]
        p_fake = float(result["probabilities"]["FAKE"])
        s = stats[(r["group"], r["label"])]
        s["n"] += 1
        s["correct"] += int(pred == truth)
        s["p_fake_sum"] += p_fake
        if pred != truth:
            misses.append({"path": str(r["path"].relative_to(ROOT_DIR)),
                            "truth": truth, "pred": pred,
                            "p_fake": round(p_fake, 3), "group": r["group"]})
    return stats, misses


def summarise(stats: dict) -> dict:
    """Per-group table plus per-class and overall rollups."""
    out = {"by_group": {}}
    for (group, label), s in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out["by_group"][f"{group}|{label}"] = {
            "n": s["n"],
            "accuracy": round(s["correct"] / s["n"], 4),
            "mean_p_fake": round(s["p_fake_sum"] / s["n"], 4),
        }
    for label in ("real", "fake"):
        sub = [s for (g, l), s in stats.items() if l == label]
        n = sum(s["n"] for s in sub)
        if n:
            out[label] = {"n": n,
                          "accuracy": round(sum(s["correct"] for s in sub) / n, 4)}
    n = sum(s["n"] for s in stats.values())
    c = sum(s["correct"] for s in stats.values())
    if n:
        out["overall"] = {"n": n, "accuracy": round(c / n, 4)}
        # Balanced accuracy: the headline the class rollups can hide. A model
        # that calls everything FAKE scores 50% here no matter the class mix.
        accs = [out[l]["accuracy"] for l in ("real", "fake") if l in out]
        out["balanced_accuracy"] = round(sum(accs) / len(accs), 4)
    return out


def print_set(name: str, summary: dict) -> None:
    print(f"\n  {name.upper()}")
    print(f"  {'group':<26}{'label':<7}{'n':>5}{'acc':>8}{'mean P(FAKE)':>14}")
    print("  " + "-" * 58)
    for key, v in summary["by_group"].items():
        group, label = key.rsplit("|", 1)
        flag = ""
        if label == "fake" and v["accuracy"] < 0.5:
            flag = "  << MISSED"
        elif label == "real" and v["accuracy"] < 0.7:
            flag = "  << FALSE ALARMS"
        print(f"  {group:<26}{label:<7}{v['n']:>5}{v['accuracy'] * 100:>7.1f}%"
              f"{v['mean_p_fake'] * 100:>13.1f}%{flag}")
    print("  " + "-" * 58)
    for label in ("real", "fake"):
        if label in summary:
            print(f"  {'ALL ' + label:<26}{'':<7}{summary[label]['n']:>5}"
                  f"{summary[label]['accuracy'] * 100:>7.1f}%")
    if "overall" in summary:
        print(f"  {'OVERALL':<26}{'':<7}{summary['overall']['n']:>5}"
              f"{summary['overall']['accuracy'] * 100:>7.1f}%")
        print(f"  {'BALANCED':<26}{'':<7}{'':>5}"
              f"{summary['balanced_accuracy'] * 100:>7.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*",
                    default=["internet", "holdout", "modern", "multires"],
                    choices=["internet", "holdout", "modern", "multires"])
    ap.add_argument("--limit", type=int, default=150,
                    help="max images per group+label (default 150)")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint filename under models/, a repo-relative "
                         "path, or an absolute path (default: autodetect)")
    ap.add_argument("--tag", default="latest",
                    help="label for the output file, e.g. before / after-retrain")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT_DIR))
    from src.utils import Config, resolve_cnn_checkpoint, set_cnn_checkpoint

    cfg = Config()
    if args.checkpoint:
        set_cnn_checkpoint(cfg, args.checkpoint)
    else:
        resolve_cnn_checkpoint(cfg)

    sets = collect_sets(args.sets)
    if not sets:
        sys.exit("No evaluation sets found. Build at least one of:\n"
                 "  dataset/collect_modern_generators.py\n"
                 "  dataset/build_internet_testset.py")

    print("=" * 74)
    print(f"  TruthLens evaluation board — checkpoint={cfg.cnn_model_name}  "
          f"model={args.model}")
    print("=" * 74)

    payload = {"checkpoint": cfg.cnn_model_name, "model_type": args.model,
               "limit_per_group": args.limit, "sets": {}}
    headline: List[Tuple[str, dict]] = []

    for name, rows in sets.items():
        capped = cap_per_group(rows, args.limit)
        print(f"\n  [{name}] {len(capped)} of {len(rows)} images")
        stats, misses = score(capped, cfg, args.model)
        if not stats:
            print(f"  [{name}] nothing scored — files missing?")
            continue
        summary = summarise(stats)
        payload["sets"][name] = {**summary, "misclassified": misses[:200]}
        print_set(name, summary)
        headline.append((name, summary))

    print("\n" + "=" * 74)
    print("  SUMMARY — balanced accuracy is the number to watch")
    print("=" * 74)
    print(f"  {'set':<14}{'n':>6}{'real':>9}{'fake':>9}{'overall':>10}{'balanced':>11}")
    print("  " + "-" * 60)
    for name, s in headline:
        real = f"{s['real']['accuracy'] * 100:.1f}%" if "real" in s else "—"
        fake = f"{s['fake']['accuracy'] * 100:.1f}%" if "fake" in s else "—"
        print(f"  {name:<14}{s['overall']['n']:>6}{real:>9}{fake:>9}"
              f"{s['overall']['accuracy'] * 100:>9.1f}%"
              f"{s['balanced_accuracy'] * 100:>10.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"eval_board_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
