#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthLens — Notebook 01: Exploratory Data Analysis
====================================================
Run this script to generate EDA plots and statistics.
For interactive use: jupytext --to notebook notebooks/01_EDA.py

Author: TruthLens Team
"""

# %% [markdown]
# # 📊 TruthLens — Exploratory Data Analysis
#
# This notebook explores the CIFAKE dataset used to train TruthLens.
#
# **Dataset:** CIFAKE — Real vs AI-Generated Images
# - 60,000 REAL images (from CIFAR-10)
# - 60,000 FAKE images (Stable Diffusion v1.4 generated)

# %% Imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils import Config, get_logger, RESULTS_DIR

logger = get_logger("EDA")
cfg    = Config()

# %% [markdown]
# ## 1. Dataset Statistics

# %% Count images
print("=" * 55)
print("  TruthLens — Dataset Statistics")
print("=" * 55)

split_counts = {}
for split in ["train", "val", "test"]:
    split_counts[split] = {}
    for cls in ["real", "fake"]:
        d = cfg.data_dir / split / cls
        if d.exists():
            files = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            split_counts[split][cls] = len(files)
            print(f"  {split:6s}/{cls:5s}  →  {len(files):6,} images")
        else:
            split_counts[split][cls] = 0
            print(f"  {split:6s}/{cls:5s}  →  [NOT FOUND]")

total = sum(v for split in split_counts.values() for v in split.values())
print(f"\n  Total: {total:,} images")
print("=" * 55)

# %% [markdown]
# ## 2. Class Balance Plot

# %% Class balance bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = {"real": "#4ade80", "fake": "#f87171"}

for ax, split in zip(axes, ["train", "val", "test"]):
    counts = split_counts.get(split, {"real": 0, "fake": 0})
    bars = ax.bar(
        ["REAL", "FAKE"],
        [counts.get("real", 0), counts.get("fake", 0)],
        color=[colors["real"], colors["fake"]],
        edgecolor="white", linewidth=0.5, width=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            f"{bar.get_height():,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_title(f"{split.title()} Split", fontsize=13, fontweight="bold")
    ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(counts.values(), default=1) * 1.2 or 1000)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("CIFAKE — Class Distribution", fontsize=15, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "plots" / "eda_class_balance.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {out}")

# %% [markdown]
# ## 3. Sample Images (Real vs Fake)

# %% Visualise sample images
def load_samples(cls_dir: Path, n: int = 8) -> list:
    files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
    if not files:
        return [Image.new("RGB", (32, 32), color=(100, 100, 100)) for _ in range(n)]
    chosen = random.sample(files, min(n, len(files)))
    return [Image.open(f).convert("RGB") for f in chosen]


real_dir = cfg.data_dir / "train" / "real"
fake_dir = cfg.data_dir / "train" / "fake"

real_imgs = load_samples(real_dir, n=8)
fake_imgs = load_samples(fake_dir, n=8)

fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for i, img in enumerate(real_imgs):
    axes[0, i].imshow(img)
    axes[0, i].axis("off")
    if i == 0: axes[0, i].set_ylabel("✅ REAL", fontsize=11, color="#4ade80", fontweight="bold")

for i, img in enumerate(fake_imgs):
    axes[1, i].imshow(img)
    axes[1, i].axis("off")
    if i == 0: axes[1, i].set_ylabel("🤖 FAKE", fontsize=11, color="#f87171", fontweight="bold")

plt.suptitle("CIFAKE — Sample Images (Train Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "plots" / "eda_samples.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {out}")

# %% [markdown]
# ## 4. Pixel Intensity Distributions

# %% Channel histograms
def compute_channel_hist(cls_dir: Path, n_samples: int = 500) -> np.ndarray:
    files  = list(cls_dir.glob("*.jpg"))[:n_samples]
    hists  = {c: np.zeros(256) for c in range(3)}
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))
        for c in range(3):
            h, _ = np.histogram(img[:, :, c].ravel(), bins=256, range=(0,255))
            hists[c] += h
    return np.stack([hists[c] for c in range(3)], axis=0)


print("  Computing pixel histograms (this may take a moment)...")
real_hists = compute_channel_hist(real_dir)
fake_hists = compute_channel_hist(fake_dir)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
channel_names = ["Red", "Green", "Blue"]
colors_ch     = ["#ef4444", "#22c55e", "#3b82f6"]

for c, (ax, name, color) in enumerate(zip(axes, channel_names, colors_ch)):
    x = np.arange(256)
    ax.fill_between(x, real_hists[c], alpha=0.5, color=color, label="REAL")
    ax.fill_between(x, fake_hists[c], alpha=0.3, color="orange", label="FAKE")
    ax.plot(x, real_hists[c], color=color, lw=1.5, alpha=0.8)
    ax.plot(x, fake_hists[c], color="orange", lw=1.5, alpha=0.8)
    ax.set_title(f"{name} Channel", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel Value"); ax.set_ylabel("Frequency")
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Pixel Intensity — Real vs Fake", fontsize=14, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "plots" / "eda_pixel_histograms.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {out}")

# %% [markdown]
# ## 5. Frequency Domain Analysis (FFT)

# %% FFT analysis
def avg_fft_magnitude(cls_dir: Path, n_samples: int = 200) -> np.ndarray:
    files  = list(cls_dir.glob("*.jpg"))[:n_samples]
    accum  = None
    for f in files:
        img  = np.array(Image.open(f).convert("L").resize((32, 32)), dtype=np.float32)
        dft  = np.fft.fftshift(np.fft.fft2(img))
        mag  = np.log1p(np.abs(dft))
        accum = mag if accum is None else accum + mag
    return (accum / len(files)) if accum is not None else np.zeros((32, 32))


print("  Computing FFT analysis...")
real_fft = avg_fft_magnitude(real_dir)
fake_fft = avg_fft_magnitude(fake_dir)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, fft_mag, label, cmap in zip(
    axes, [real_fft, fake_fft], ["REAL", "FAKE"], ["Blues_r", "Reds_r"]
):
    im = ax.imshow(fft_mag, cmap=cmap, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{label} — Avg FFT Magnitude", fontsize=12, fontweight="bold")
    ax.axis("off")

plt.suptitle("Frequency Domain Analysis — Real vs AI-Generated", fontsize=13, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "plots" / "eda_fft_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {out}")
print("\n  ✅ EDA complete. All plots saved to results/plots/")
