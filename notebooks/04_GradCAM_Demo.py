#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthLens — Notebook 04: Grad-CAM Explainability Demo
======================================================
Visualise what regions of an image ResNet18 focuses on.
Run: python notebooks/04_GradCAM_Demo.py --image path/to/image.jpg
"""

# %% Imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import Config, RESULTS_DIR, get_logger

logger = get_logger("Notebook04")
cfg    = Config()

# %% Parse args (for CLI usage)
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="",
                    help="Path to image. If empty, uses a random test image.")
parser.add_argument("--n-demo", type=int, default=4,
                    help="Number of demo images to visualise.")
args, _ = parser.parse_known_args()

print("=" * 60)
print("  TruthLens — Notebook 04: Grad-CAM Demo")
print("=" * 60)

# %% [markdown]
# ## Load Model + Predictor

# %% Load CNN predictor
ckpt_path = cfg.models_dir / cfg.cnn_model_name
if not ckpt_path.exists():
    print(f"\n  ⚠️  Checkpoint not found: {ckpt_path}")
    print("  Train first: python src/train.py")
    sys.exit(0)

from src.predict import CNNPredictor, generate_explanation, detect_likely_generator
predictor = CNNPredictor(cfg)
print("  ✅ Model loaded successfully\n")

# %% [markdown]
# ## Single Image Grad-CAM

# %% Get demo images
def get_demo_images(n: int) -> list:
    """Collect n images from test set (mix of real and fake)."""
    paths = []
    for cls in ["real", "fake"]:
        d = cfg.data_dir / "test" / cls
        if d.exists():
            files = list(d.glob("*.jpg"))[:n//2]
            paths.extend(files)
    if not paths:
        # Return synthetic images if dataset not present
        print("  No dataset found — using synthetic images for demo.")
        return [None] * n
    return paths[:n]


if args.image:
    demo_sources = [args.image] * args.n_demo
else:
    demo_sources = get_demo_images(args.n_demo)

# %% Run Grad-CAM on demo images
fig, axes = plt.subplots(len(demo_sources), 3, figsize=(12, len(demo_sources) * 4))
if len(demo_sources) == 1:
    axes = [axes]

for row, source in enumerate(demo_sources):
    if source is None:
        # Use random synthetic image
        img_np  = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_np)
        label   = "SYNTHETIC"
    else:
        img_pil = Image.open(source).convert("RGB")
        img_np  = np.array(img_pil)
        label   = Path(source).parent.name.upper() if hasattr(source, '__class__') else "IMAGE"
        if isinstance(source, Path):
            label = source.parent.name.upper()

    # Predict + Grad-CAM
    try:
        result, heatmap = predictor.predict_with_gradcam(img_pil)
        pred     = result["prediction"]
        conf     = result["confidence"]
        emoji    = result["emoji"]
        reasons  = generate_explanation(result, img_np)

        axes[row][0].imshow(img_pil)
        axes[row][0].set_title(f"Original [{label}]", fontsize=10, fontweight="bold")
        axes[row][0].axis("off")

        if heatmap is not None:
            axes[row][1].imshow(heatmap)
        else:
            axes[row][1].imshow(img_np)
        axes[row][1].set_title("Grad-CAM Overlay", fontsize=10, fontweight="bold")
        axes[row][1].axis("off")

        color = "#4ade80" if pred == "REAL" else "#f87171"
        axes[row][2].axis("off")
        axes[row][2].text(
            0.1, 0.85, f"{emoji} {pred}", transform=axes[row][2].transAxes,
            fontsize=14, fontweight="bold", color=color,
        )
        axes[row][2].text(
            0.1, 0.70, f"Confidence: {conf:.1f}%", transform=axes[row][2].transAxes,
            fontsize=11, color="white",
        )
        reason_text = "\n".join(f"• {r[:50]}" for r in reasons[:3])
        axes[row][2].text(
            0.1, 0.45, reason_text, transform=axes[row][2].transAxes,
            fontsize=8, color="#94a3b8", va="top",
        )
        axes[row][2].set_facecolor("#0d1b2a")

    except Exception as exc:
        logger.error(f"  Error on row {row}: {exc}")
        for ax in axes[row]:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")

plt.suptitle("TruthLens — Grad-CAM Explainability Demo", fontsize=15, fontweight="bold")
plt.tight_layout()

out = RESULTS_DIR / "gradcam" / "gradcam_demo.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
plt.show()
print(f"\n  ✅ Grad-CAM demo saved: {out}")

# %% [markdown]
# ## Interpretation
#
# **Red/Hot regions** = areas that most strongly activated the "FAKE" or "REAL" neurons.
#
# For AI-generated images, the model often focuses on:
# - Unnatural texture patterns (too smooth or too regular)
# - Lighting inconsistencies at object boundaries
# - Abnormal frequency content in flat regions
#
# For real images, the model uses natural noise and texture cues.
