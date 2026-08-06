#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthLens — Notebook 03: CNN Training & Evaluation
====================================================
ResNet18 transfer learning training walkthrough.
Run: python notebooks/03_CNN_Training.py
"""

# %% Imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import Config, get_device, get_logger, RESULTS_DIR
from src.train import build_model

logger = get_logger("Notebook03")
cfg    = Config()

print("=" * 60)
print("  TruthLens — Notebook 03: CNN Training")
print("=" * 60)

# %% [markdown]
# ## Model Architecture

# %% Print model summary
device = get_device()
model  = build_model(num_classes=2, pretrained=False)

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  ResNet18 Architecture Summary:")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"\n  Classifier Head:")
print(model.fc)

# %% [markdown]
# ## Training (Phase 1: Head Only → Phase 2: Full Fine-tune)

# %% Train the model
print("\n  Starting CNN training...")
print("  (Make sure dataset is downloaded first)")

from src.train import train
try:
    history = train(cfg)
except RuntimeError as e:
    print(f"\n  ⚠️  {e}")
    history = None

# %% [markdown]
# ## Training Curves

# %% Load and plot training history
history_path = RESULTS_DIR / "metrics" / "cnn_training_history.json"
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
elif history is None:
    print("  No training history found. Train the model first.")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

if history["train_loss"]:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", color="#6366f1", lw=2, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "s-", color="#f59e0b", lw=2, label="Val")
    axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", color="#6366f1", lw=2, label="Train")
    axes[1].plot(epochs, history["val_acc"],   "s-", color="#f59e0b", lw=2, label="Val")
    axes[1].set_title("Training Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("ResNet18 — TruthLens Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "plots" / "notebook_cnn_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out}")

    best_epoch = history["val_acc"].index(max(history["val_acc"])) + 1
    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Best val acc: {max(history['val_acc']):.2f}%")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")

# %% [markdown]
# ## Load Best Checkpoint & Quick Inference Test

# %% Test inference
ckpt_path = cfg.models_dir / cfg.cnn_model_name
if ckpt_path.exists():
    model = build_model(num_classes=2).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dummy forward pass
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out  = model(dummy)
        prob = torch.softmax(out, dim=1)

    print(f"\n  ✅ Model loaded successfully")
    print(f"  Dummy forward pass → logits: {out.shape}")
    print(f"  Probabilities → REAL: {prob[0,0]:.4f}  FAKE: {prob[0,1]:.4f}")
else:
    print("  No checkpoint found yet. Train with: python src/train.py")
