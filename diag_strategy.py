#!/usr/bin/env python3
"""Select the best inference strategy on val, then report it on test.

Compares single-crop, flip-TTA, five-crop TTA, and combinations, plus the
classical models, so ensemble weights can be chosen from measured AUC rather
than hardcoded guesses.
"""
import glob
import warnings

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from src.predict import CNNPredictor
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from src.utils import Config

PER_CLASS = 200
cfg = Config()
pred = CNNPredictor(cfg)
dev = pred.device
model = pred.model

norm = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
base = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
five = T.Compose([T.Resize((256, 256)), T.FiveCrop(224),
                  T.Lambda(lambda cs: torch.stack([norm(c) for c in cs]))])


def collect(split):
    fs, ys = [], []
    for cls, lbl in (("real", 0), ("fake", 1)):
        a = sorted(glob.glob(f"datasets/prepared/multires/{split}/{cls}/*"))
        step = max(1, len(a) // PER_CLASS)
        sel = a[::step][:PER_CLASS]
        fs += sel
        ys += [lbl] * len(sel)
    return fs, np.array(ys)


def scores(files):
    """Return dict of strategy -> fake probability array."""
    out = {k: [] for k in ("single", "flip", "five", "single+flip+five")}
    for f in files:
        im = Image.open(f).convert("RGB")
        t1 = base(im).unsqueeze(0)
        t2 = base(im.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
        t5 = five(im)
        batch = torch.cat([t1, t2, t5], dim=0).to(dev)
        with torch.no_grad():
            p = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
        out["single"].append(p[0])
        out["flip"].append(p[:2].mean())
        out["five"].append(p[2:].mean())
        out["single+flip+five"].append(p.mean())
    return {k: np.array(v) * 100 for k, v in out.items()}


def report(tag, ys, sc):
    print(f"\n  {tag}  (n={len(ys)}, {(ys==0).sum()} real / {(ys==1).sum()} fake)")
    print(f"    {'strategy':<20s} {'AUC':>7s} {'acc':>7s} {'real acc':>9s} {'fake acc':>9s}")
    best, best_auc = None, -1
    for k, v in sc.items():
        auc = roc_auc_score(ys, v)
        pr = (v >= 50).astype(int)
        acc = (pr == ys).mean() * 100
        ra = (pr[ys == 0] == 0).mean() * 100
        fa = (pr[ys == 1] == 1).mean() * 100
        print(f"    {k:<20s} {auc:7.4f} {acc:6.1f}% {ra:8.1f}% {fa:8.1f}%")
        if auc > best_auc:
            best, best_auc = k, auc
    print(f"    -> best by AUC: {best} ({best_auc:.4f})")
    return best


vf, vy = collect("val")
tf, ty = collect("test")
vs = scores(vf)
best = report("VAL (selection)", vy, vs)
ts = scores(tf)
report("TEST (held out)", ty, ts)
print(f"\n  SELECTED on val: {best}")

# Classical model AUCs, to decide ensemble weights honestly.
print("\n  Classical models on TEST:")
from src.predict import ClassicalPredictor
for key in ("svm", "random_forest", "logistic_regression"):
    try:
        cp = ClassicalPredictor(key, cfg)
        pv = []
        for f in tf:
            r = cp.predict(Image.open(f).convert("RGB"))
            pv.append(r["probabilities"]["FAKE"])
        pv = np.array(pv)
        auc = roc_auc_score(ty, pv)
        acc = ((pv >= 50).astype(int) == ty).mean() * 100
        print(f"    {key:<22s} AUC {auc:.4f}  acc {acc:5.1f}%")
    except Exception as exc:
        print(f"    {key:<22s} FAILED: {type(exc).__name__}: {exc}")

# Does adding classical or forensic signal to the best CNN score help at all?
from src.predict import compute_ela_analysis, compute_fft_spectral_score
cnn_t = ts[best]
fft_t, ela_t = [], []
for f in tf:
    im = Image.open(f).convert("RGB")
    fft_t.append(compute_fft_spectral_score(np.array(im))["spectral_ai_score"])
    ela_t.append(compute_ela_analysis(im)[0]["ela_ai_score"])
fft_t, ela_t = np.array(fft_t), np.array(ela_t)
print("\n  Blending forensic signal into the CNN (TEST AUC):")
print(f"    CNN alone                      {roc_auc_score(ty, cnn_t):.4f}")
for w in (0.10, 0.25, 0.45):
    mix = cnn_t * (1 - w) + ((fft_t + ela_t) / 2.0) * w
    print(f"    CNN {1-w:.2f} + forensic {w:.2f}       {roc_auc_score(ty, mix):.4f}")
