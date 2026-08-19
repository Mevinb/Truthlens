#!/usr/bin/env python3
"""
TruthLens — TTA View Weighting and Threshold Diagnostic
========================================================
Asks whether the served verdict is being diluted by its own view set.

The hypothesis
--------------
``CNNPredictor.tta_probs`` averages twelve views with equal weight:

    1  whole frame resized to 224
    1  whole frame flipped, resized to 224
    5  FiveCrop of a 256 resize      (still a downscaled frame)
    5  native 1:1 crops              (only for images with a native scale)

On a 1024x1536 generation the first seven average roughly 7x7 source pixels per
output pixel, which is exactly the band that separates a neural decoder from a
camera sensor. So 58% of the vote is cast by views that cannot see the evidence,
and their near-0.5 outputs pull the mean toward the middle no matter how
confident the native crops are.

The measured symptom fits precisely. On the held-out modern test split the old
checkpoint scored gpt-image-2 at 48.5% recall with a mean P(FAKE) of 49.6% —
dead centre, the signature of averaging a confident signal against noise rather
than of a model that has no signal.

If that is what is happening, the fix costs no training: reweight the groups.
If it is not, this rules the idea out and the answer has to come from the
weights themselves.

What it does
------------
Each image gets ONE forward pass over all twelve views. The per-group mean
probabilities are cached, and every weighting scheme is then scored offline from
that cache — so comparing a dozen schemes costs the same GPU time as scoring
once. Schemes are reported with balanced accuracy at the served 0.5 threshold,
AUC (threshold-free, so it measures evidence quality rather than calibration),
and the balanced accuracy each would reach at its own best threshold.

Reading the output
------------------
* **AUC** says which views carry the signal. If ``native_only`` beats
  ``current`` on AUC, the downscaled views are diluting a real signal and
  reweighting is a free win.
* **bal@0.5 vs bal@best** separates two different problems. A scheme with good
  AUC and a poor ``bal@0.5`` is mis-calibrated and wants a threshold, not a
  retrain. A scheme with poor AUC cannot be rescued by either.
* **per-generator recall** is where an average hides a hole: a scheme can gain
  overall while losing the one generator that matters.

Usage
-----
    python diag_tta_weighting.py --split holdout
    python diag_tta_weighting.py --split test --limit 150
    python diag_tta_weighting.py --internet
    python diag_tta_weighting.py --split holdout --checkpoint resnet18_modern.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
MODERN_MANIFEST = ROOT_DIR / "datasets" / "prepared" / "modern_v2" / "manifest.jsonl"
INTERNET_MANIFEST = ROOT_DIR / "datasets" / "evaluation" / "internet" / "manifest.jsonl"
OUT_DIR = ROOT_DIR / "results" / "metrics"

GROUPS = ("frame", "flip", "five", "native")

# Weight per view group. Normalised over whichever groups are present, so an
# image with no native scale still sums to 1 and small-image behaviour is
# unchanged by any scheme.
SCHEMES: Dict[str, Dict[str, float]] = {
    "current (served)":  {"frame": 1, "flip": 1, "five": 5, "native": 5},
    "frame_only":        {"frame": 1, "flip": 1, "five": 5, "native": 0},
    "native_only":       {"frame": 0, "flip": 0, "five": 0, "native": 1},
    "native_75":         {"frame": 1, "flip": 1, "five": 1, "native": 9},
    "native_60":         {"frame": 1, "flip": 1, "five": 2, "native": 6},
    "no_fivecrop":       {"frame": 1, "flip": 1, "five": 0, "native": 5},
    "equal_per_group":   {"frame": 1, "flip": 1, "five": 1, "native": 1},
}


def load_rows(split: str, limit: int, internet: bool) -> Tuple[List[dict], Path]:
    """Rows plus the directory their ``path`` field is relative to."""
    base = (ROOT_DIR / "datasets" / "evaluation" / "internet") if internet else \
        (ROOT_DIR / "datasets" / "prepared" / "modern_v2")
    manifest = base / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"Missing {manifest.relative_to(ROOT_DIR)}")
    rows = [json.loads(l) for l in
            manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not internet and split and split != "all":
        rows = [r for r in rows if r.get("split") == split]

    key = "tag" if internet else "generator"
    by: Dict[str, List[dict]] = {}
    for r in rows:
        by.setdefault(r.get(key) or "?", []).append(r)
    out: List[dict] = []
    for group in by.values():
        step = max(1, len(group) // limit) if limit else 1
        out += group[::step][:limit]
    return out, base


def group_probs(predictor, pil_img) -> Dict[str, float]:
    """Mean P(fake) within each view group, from a single forward pass."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD

    size = predictor.cfg.img_size
    norm = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    base = T.Compose([T.Resize((size, size)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    views: List["torch.Tensor"] = [base(pil_img).unsqueeze(0)]
    views.append(base(pil_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0))
    views.append(T.Compose([
        T.Resize((256, 256)), T.FiveCrop(size),
        T.Lambda(lambda cs: torch.stack([norm(c) for c in cs])),
    ])(pil_img))

    native = predictor._native_crops(pil_img)
    counts = [1, 1, 5]
    if native:
        views.append(torch.stack([norm(c) for c in native]))
        counts.append(len(native))

    batch = torch.cat(views, dim=0).to(predictor.device)
    with torch.no_grad():
        probs = torch.softmax(predictor.model(batch), dim=1)[:, 1].cpu().numpy()

    out, at = {}, 0
    for name, n in zip(GROUPS, counts):
        out[name] = float(probs[at:at + n].mean())
        at += n
    return out


def combine(cached: Dict[str, float], weights: Dict[str, float]) -> float:
    num = sum(weights.get(g, 0.0) * p for g, p in cached.items())
    den = sum(weights.get(g, 0.0) for g in cached)
    # Every scheme must stay defined when a group is absent. native_only on a
    # 32px thumbnail has no native group at all; falling back to the full mean
    # keeps it scoring the image rather than dropping it from the comparison.
    if den <= 0:
        return float(np.mean(list(cached.values())))
    return num / den


def balanced_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = (p >= thr).astype(int)
    tpr = float((pred[y == 1] == 1).mean()) if (y == 1).any() else 0.0
    tnr = float((pred[y == 0] == 0).mean()) if (y == 0).any() else 0.0
    return (tpr + tnr) / 2


def best_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    grid = np.unique(np.round(np.concatenate([p, np.arange(0.05, 0.96, 0.01)]), 3))
    scored = [(balanced_at(y, p, t), t) for t in grid]
    bal, thr = max(scored)
    return float(thr), float(bal)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="holdout",
                    help="modern-corpus split: train/val/test/holdout/all")
    ap.add_argument("--internet", action="store_true",
                    help="score the internet evaluation set instead of the modern corpus")
    ap.add_argument("--limit", type=int, default=200,
                    help="max images per generator (or per tag)")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT_DIR))
    from src.predict import CNNPredictor
    from src.utils import Config, resolve_cnn_checkpoint
    from PIL import Image
    from tqdm import tqdm

    cfg = Config()
    if args.checkpoint:
        cfg.cnn_model_name = args.checkpoint
    else:
        resolve_cnn_checkpoint(cfg)
    predictor = CNNPredictor(cfg)

    rows, base = load_rows(args.split, args.limit, args.internet)
    if not rows:
        sys.exit("No manifest rows matched")

    label_key = "label"
    tag_key = "tag" if args.internet else "generator"
    scope = "internet" if args.internet else f"split={args.split}"

    print("=" * 78)
    print("  TTA VIEW WEIGHTING — checkpoint="
          f"{cfg.cnn_model_name}  {scope}")
    print(f"  {len(rows)} images")
    print("=" * 78)

    cache: List[Tuple[Dict[str, float], int, str]] = []
    n_native = 0
    for row in tqdm(rows, desc="  scoring", ncols=70, unit="img", leave=False):
        # Manifest paths are relative to the corpus root, not the repo root.
        path = base / row["path"] if not Path(row["path"]).is_absolute() \
            else Path(row["path"])
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                pil = im.convert("RGB")
        except Exception:                                          # noqa: BLE001
            continue
        g = group_probs(predictor, pil)
        n_native += "native" in g
        cache.append((g, 1 if row[label_key] == "fake" else 0,
                      row.get(tag_key) or "?"))

    if not cache:
        sys.exit("Nothing scored — check manifest paths")

    y = np.array([c[1] for c in cache])
    tags = [c[2] for c in cache]
    print(f"\n  {len(cache)} scored · {n_native} had a native scale "
          f"({n_native / len(cache):.0%}) · real {int((y == 0).sum())} "
          f"fake {int((y == 1).sum())}\n")

    from sklearn.metrics import roc_auc_score

    print(f"  {'scheme':20} {'AUC':>7} {'bal@0.5':>9} {'bal@best':>9} "
          f"{'thr':>6}  {'real@0.5':>9} {'fake@0.5':>9}")
    print("  " + "-" * 76)

    results = {}
    for name, weights in SCHEMES.items():
        p = np.array([combine(g, weights) for g, _, _ in cache])
        auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
        thr, bal_best = best_threshold(y, p)
        pred = (p >= 0.5).astype(int)
        real_acc = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
        fake_acc = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
        results[name] = {
            "auc": round(auc, 4), "bal_at_0.5": round(balanced_at(y, p, 0.5), 4),
            "bal_at_best": round(bal_best, 4), "best_threshold": round(thr, 3),
            "real_acc_at_0.5": round(real_acc, 4),
            "fake_acc_at_0.5": round(fake_acc, 4),
            "per_generator_recall_at_0.5": {},
            "mean_p_fake": round(float(p.mean()), 4),
        }
        for t in sorted(set(tags)):
            m = np.array([tg == t for tg in tags])
            if not m.any():
                continue
            correct = (pred[m] == y[m]).mean()
            results[name]["per_generator_recall_at_0.5"][t] = round(float(correct), 4)
        print(f"  {name:20} {auc:7.4f} {balanced_at(y, p, 0.5):9.4f} "
              f"{bal_best:9.4f} {thr:6.2f}  {real_acc:9.4f} {fake_acc:9.4f}")

    baseline = results["current (served)"]["auc"]
    gains = sorted(((v["auc"] - baseline, k) for k, v in results.items()),
                   reverse=True)
    print("\n  AUC vs served weighting")
    for delta, name in gains:
        if name == "current (served)":
            continue
        verdict = ("dilution confirmed" if delta > 0.01 else
                   "no better" if delta > -0.01 else "worse")
        print(f"    {name:20} {delta:+.4f}   {verdict}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  best AUC: {best_name} ({results[best_name]['auc']:.4f})")
    print("  per-generator accuracy at 0.5, served vs best:")
    served_pg = results["current (served)"]["per_generator_recall_at_0.5"]
    best_pg = results[best_name]["per_generator_recall_at_0.5"]
    for t in sorted(served_pg):
        s, b = served_pg[t], best_pg.get(t, float("nan"))
        flag = "  <<" if b - s > 0.05 else ("  !!" if b - s < -0.05 else "")
        print(f"    {t:26} {s:6.1%} -> {b:6.1%}  {b - s:+.1%}{flag}")

    print("=" * 78)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "internet" if args.internet else args.split
    out = OUT_DIR / f"tta_weighting_{stem}.json"
    out.write_text(json.dumps(
        {"checkpoint": cfg.cnn_model_name, "scope": scope, "n": len(cache),
         "schemes": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
