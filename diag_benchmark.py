#!/usr/bin/env python3
"""Benchmark every shipped model on the held-out datasets/prepared/multires test split.

Produces the numbers quoted in README.md and app/app.py. The CNN is evaluated on
the full test split; the classical models on a stratified subsample, because
HOG+LBP extraction is the bottleneck (~0.5s/image).

Writes results/metrics/benchmark_test_split.json.
"""
import argparse
import glob
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

from src.predict import ClassicalPredictor, CNNPredictor
from src.utils import Config

CLASSICAL = ("random_forest", "svm", "logistic_regression",
             "knn", "decision_tree", "naive_bayes")


def collect(per_class: int | None):
    files, labels = [], []
    for cls, lbl in (("real", 0), ("fake", 1)):
        a = sorted(glob.glob(f"datasets/prepared/multires/test/{cls}/*"))
        if per_class is not None and per_class < len(a):
            a = a[:: max(1, len(a) // per_class)][:per_class]
        files += a
        labels += [lbl] * len(a)
    return files, np.array(labels)


def score(y, p):
    pred = (p >= 50).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()) * 100, 2),
        "f1": round(float(f1_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "real_accuracy": round(float((pred[y == 0] == 0).mean()) * 100, 2),
        "fake_accuracy": round(float((pred[y == 1] == 1).mean()) * 100, 2),
    }


def score_by_tier(files, y, p):
    """Break the CNN score down by resolution tier.

    The overall figure is a blend of three tiers and hides a real gap: the
    high-resolution tier — the regime a user uploading a photo or an SDXL render
    is actually in — scores several points below the 256px and 32px tiers. The
    app quotes both numbers so the headline does not overstate the case that
    matters.
    """
    tiers = {}
    for prefix in ("highres_", "medres256_", "lowres32_"):
        idx = np.array([Path(f).name.startswith(prefix) for f in files])
        if not idx.any():
            continue
        tiers[prefix.rstrip("_")] = score(y[idx], p[idx])
    return tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-per-class", type=int, default=None,
                    help="Subsample the CNN eval too (default: full split).")
    ap.add_argument("--classical-per-class", type=int, default=300)
    args = ap.parse_args()

    cfg = Config()
    out = {"split": "datasets/prepared/multires/test", "models": {}}

    # ── CNN, via the exact path the app serves (TTA, no forensic calibration) ──
    files, y = collect(args.cnn_per_class)
    pred = CNNPredictor(cfg)
    t0 = time.time()
    p = []
    for i, f in enumerate(files, 1):
        im = Image.open(f).convert("RGB")
        with torch.no_grad():
            probs = pred.tta_probs(im, im)
        p.append(probs[1] * 100)
        if i % 250 == 0:
            print(f"    cnn {i}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    out["models"]["resnet18_tta"] = score(y, np.array(p))
    out["models"]["resnet18_tta"]["seconds"] = round(time.time() - t0, 1)
    out["by_resolution"] = score_by_tier(files, y, np.array(p))

    # ── Same checkpoint, single 224 crop, for the TTA delta ──
    from src.preprocessing import preprocess_single_image
    p1 = []
    for f in files:
        t = preprocess_single_image(Image.open(f).convert("RGB"), cfg.img_size).to(pred.device)
        with torch.no_grad():
            p1.append(float(torch.softmax(pred.model(t), dim=1)[0, 1]) * 100)
    out["models"]["resnet18_single_crop"] = score(y, np.array(p1))

    # ── Classical models on a subsample ──
    cfiles, cy = collect(args.classical_per_class)
    images = [Image.open(f).convert("RGB") for f in cfiles]
    for key in CLASSICAL:
        t0 = time.time()
        try:
            cp = ClassicalPredictor(key, cfg)
            cp_p = np.array([cp.predict(im)["probabilities"]["FAKE"] for im in images])
            out["models"][key] = score(cy, cp_p)
            out["models"][key]["seconds"] = round(time.time() - t0, 1)
        except Exception as exc:
            out["models"][key] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"    {key} done ({time.time() - t0:.0f}s)", flush=True)

    dest = Path("results/metrics/benchmark_test_split.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    print(f"\n  {'model':<24s} {'n':>5s} {'acc':>7s} {'F1':>7s} {'AUC':>7s} "
          f"{'real':>7s} {'fake':>7s}")
    print("  " + "-" * 70)
    for name, m in out["models"].items():
        if "error" in m:
            print(f"  {name:<24s} {m['error']}")
            continue
        print(f"  {name:<24s} {m['n']:5d} {m['accuracy']:6.2f}% {m['f1']:7.4f} "
              f"{m['auc']:7.4f} {m['real_accuracy']:6.1f}% {m['fake_accuracy']:6.1f}%")

    if out.get("by_resolution"):
        print("\n  ResNet18 + TTA, by resolution tier:")
        print("  " + "-" * 70)
        for tier, m in out["by_resolution"].items():
            print(f"  {tier:<24s} {m['n']:5d} {m['accuracy']:6.2f}% {m['f1']:7.4f} "
                  f"{m['auc']:7.4f} {m['real_accuracy']:6.1f}% {m['fake_accuracy']:6.1f}%")

    print(f"\n  written -> {dest}")


if __name__ == "__main__":
    main()
