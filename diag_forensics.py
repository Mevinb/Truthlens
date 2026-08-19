#!/usr/bin/env python3
"""Diagnostic: compare raw CNN vs forensic-calibrated output on the test split."""
import glob
import warnings

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from src.predict import CNNPredictor, compute_ela_analysis, compute_fft_spectral_score
from src.preprocessing import preprocess_single_image
from src.utils import Config

N = 80
cfg = Config()
pred = CNNPredictor(cfg)  # load ONCE

files, labels = [], []
for cls, lbl in (("real", 0), ("fake", 1)):
    fs = sorted(glob.glob(f"datasets/prepared/multires/test/{cls}/*"))
    step = max(1, len(fs) // N)
    sel = fs[::step][:N]
    files += sel
    labels += [lbl] * len(sel)
labels = np.array(labels)

raw, calib, fft_s, ela_s, ela_std_s = [], [], [], [], []
for f in files:
    im = Image.open(f).convert("RGB")
    t = preprocess_single_image(im, cfg.img_size).to(pred.device)
    with torch.no_grad():
        p = torch.softmax(pred.model(t), dim=1)[0]
    raw.append(float(p[1]) * 100)

    a = np.array(im)
    sp = compute_fft_spectral_score(a)
    el, _ = compute_ela_analysis(im)
    fft_s.append(sp["spectral_ai_score"])
    ela_s.append(el["ela_ai_score"])
    ela_std_s.append(el["ela_std"])

    r = pred.predict(im)  # full calibrated path
    calib.append(r["probabilities"]["FAKE"])

raw = np.array(raw); calib = np.array(calib)
fft_s = np.array(fft_s); ela_s = np.array(ela_s); ela_std_s = np.array(ela_std_s)

print(f"\n n = {len(labels)}  ({(labels==0).sum()} real / {(labels==1).sum()} fake)\n")
print(f" {'signal':<26s} {'AUC':>7s} {'acc@50':>8s}  {'mean real':>10s} {'mean fake':>10s}")
print(" " + "-" * 68)


def row(name, score, higher_is_fake=True):
    s = score if higher_is_fake else -score
    auc = roc_auc_score(labels, s)
    acc = ((score >= 50).astype(int) == labels).mean() * 100 if higher_is_fake else float("nan")
    print(f" {name:<26s} {auc:7.3f} {acc:7.1f}%  {score[labels==0].mean():10.2f} {score[labels==1].mean():10.2f}")


row("raw CNN (single crop)", raw)
row("CALIBRATED (shipped)", calib)
row("fft spectral_ai_score", fft_s)
row("ela_ai_score", ela_s)
row("ela_std (as AI signal)", ela_std_s)

print("\n Per-class accuracy @ threshold 50:")
for name, score in (("raw CNN", raw), ("calibrated", calib)):
    pr = (score >= 50).astype(int)
    tr = (pr[labels == 0] == 0).mean() * 100
    tf = (pr[labels == 1] == 1).mean() * 100
    print(f"   {name:<12s} real correct {tr:6.1f}%   fake correct {tf:6.1f}%   overall {(pr==labels).mean()*100:6.1f}%")

print("\n Calibration branch taken:")
branches = {"authentic_camera": 0, "synthetic": 0, "middle": 0}
for i, f in enumerate(files):
    ela_std, ela_ai, fft_ai = ela_std_s[i], ela_s[i], fft_s[i]
    is_auth = ela_std >= 18.0 and fft_ai <= 40.0
    is_syn = (ela_std < 14.0 and ela_ai > 50.0) or (fft_ai > 55.0) or (raw[i] >= 60.0 and not is_auth)
    if is_auth and not (fft_ai > 60.0):
        branches["authentic_camera"] += 1
    elif is_syn:
        branches["synthetic"] += 1
    else:
        branches["middle"] += 1
for k, v in branches.items():
    print(f"   {k:<18s} {v:4d} / {len(files)}")
