#!/usr/bin/env python3
"""
TruthLens — src/utils.py
=========================
Central utilities: configuration, logging, plotting, and model I/O helpers.
"""

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ─── Root Paths ───────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "dataset"
MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "gradcam").mkdir(parents=True, exist_ok=True)


# ─── Configuration ────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Dataset
    data_dir:    Path = field(default_factory=lambda: DATA_DIR)
    img_size:    int  = 224          # ResNet input size
    cifake_size: int  = 32           # CIFAKE native size
    num_classes: int  = 2
    class_names: Tuple[str, ...] = ("REAL", "FAKE")

    # Training — CNN
    batch_size:  int   = 64
    num_epochs:  int   = 20
    lr_head:     float = 1e-3        # LR for head-only phase
    lr_finetune: float = 1e-4        # LR for full fine-tune phase
    freeze_epochs: int = 5           # epochs to train head only
    weight_decay: float = 1e-4
    patience:    int   = 5           # early stopping patience
    num_workers: int   = 4

    # Training — Classical ML
    n_jobs:       int = -1
    cv_folds:     int = 5
    random_state: int = 42

    # Paths
    models_dir:  Path = field(default_factory=lambda: MODELS_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    # Model filenames
    cnn_model_name:   str = "resnet18_truthlens.pth"
    svm_model_name:   str = "svm_model.pkl"
    rf_model_name:    str = "rf_model.pkl"
    lr_model_name:    str = "lr_model.pkl"
    dt_model_name:    str = "dt_model.pkl"
    knn_model_name:   str = "knn_model.pkl"
    nb_model_name:    str = "nb_model.pkl"

    def to_json(self, path: Path) -> None:
        d = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        d["data_dir"]    = Path(d["data_dir"])
        d["models_dir"]  = Path(d["models_dir"])
        d["results_dir"] = Path(d["results_dir"])
        d["class_names"] = tuple(d["class_names"])
        return cls(**d)


# ─── Logger ───────────────────────────────────────────────────────────────────
def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a configured logger with stream + optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── Model I/O ────────────────────────────────────────────────────────────────
def save_sklearn_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_sklearn_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ─── Confusion Matrix Plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, ...],
    title:  str,
    save_path: Optional[Path] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── ROC Curve Plot ───────────────────────────────────────────────────────────
def plot_roc_curve(
    y_true:  np.ndarray,
    y_score: np.ndarray,
    title:   str,
    save_path: Optional[Path] = None,
) -> float:
    """Plot ROC curve and return AUC."""
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}", color="#6366f1")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#6366f1")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return auc


# ─── Training Curve Plot ──────────────────────────────────────────────────────
def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    train_accs:   List[float],
    val_accs:     List[float],
    save_path: Optional[Path] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs, train_losses, label="Train", color="#6366f1", lw=2)
    axes[0].plot(epochs, val_losses,   label="Val",   color="#f59e0b", lw=2)
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_accs, label="Train", color="#6366f1", lw=2)
    axes[1].plot(epochs, val_accs,   label="Val",   color="#f59e0b", lw=2)
    axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("TruthLens — ResNet18 Training Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Model Comparison Table ───────────────────────────────────────────────────
def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing all models across key metrics."""
    models  = list(results.keys())
    metrics = ["accuracy", "f1", "auc"]
    colors  = ["#6366f1", "#10b981", "#f59e0b"]

    x   = np.arange(len(models))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * w, vals, w, label=metric.upper(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("TruthLens — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Quick Report ─────────────────────────────────────────────────────────────
def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, ...],
    model_name: str,
) -> Dict[str, Any]:
    report_str  = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{'='*55}")
    print(f"  {model_name} — Classification Report")
    print("="*55)
    print(report_str)
    return report_dict


# ─── Seed Everything ──────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Device Helper ────────────────────────────────────────────────────────────
def get_device() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA — {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no GPU detected — training will be slow)")
    return device
