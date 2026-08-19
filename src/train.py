#!/usr/bin/env python3
"""
TruthLens — src/train.py
=========================
ResNet18 Transfer Learning Training Pipeline

Strategy:
  Phase 1 (epochs 1 – freeze_epochs):  Freeze backbone, train classifier head.
  Phase 2 (epochs freeze_epochs+1 – N): Unfreeze all layers, fine-tune end-to-end.

Features:
  • Mixed-precision training (torch.cuda.amp)
  • Early stopping
  • Cosine annealing LR scheduler
  • TensorBoard logging
  • Best-checkpoint saving

Usage:
    python src/train.py
    python src/train.py --epochs 30 --batch-size 128
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm

from src.preprocessing import get_dataloaders
from src.utils import (
    Config,
    get_device,
    get_logger,
    plot_training_curves,
    save_metrics,
    seed_everything,
)

logger = get_logger(__name__, log_file=Path("results/logs/train_cnn.log"))


# ─── Model Builder ────────────────────────────────────────────────────────────
def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Load ResNet18 with a custom two-class classifier head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    # Replace the final fully-connected layer
    in_features = model.fc.in_features          # 512
    model.fc    = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classifier head (model.fc)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ─── One Epoch ────────────────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    is_train:  bool,
) -> tuple[float, float]:
    """Run one training or validation epoch. Returns (avg_loss, accuracy%)."""
    model.train() if is_train else model.eval()

    running_loss  = 0.0
    correct       = 0
    total         = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * images.size(0)
            preds  = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = running_loss / (total + 1e-8)
    accuracy = 100.0 * correct / (total + 1e-8)
    return avg_loss, accuracy


# ─── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─── Main Training Loop ───────────────────────────────────────────────────────
def train(cfg: Config, initial_checkpoint: Path | None = None) -> dict:
    """Full training loop. Returns history dict."""
    if cfg.num_epochs <= cfg.freeze_epochs:
        raise ValueError(
            "num_epochs must be greater than freeze_epochs so the ResNet "
            "backbone is fine-tuned. Increase --epochs or reduce --freeze-epochs."
        )

    seed_everything(cfg.random_state)
    device  = get_device()
    loaders = get_dataloaders(cfg)

    if "train" not in loaders:
        raise RuntimeError(
            "Train loader not found. Download the dataset first:\n"
            "  python dataset/download_cifake.py --method kaggle\n"
            "  python dataset/download_cifake.py --method manual"
        )

    if "val" not in loaders:
        raise RuntimeError(
            f"Validation split not found under {cfg.data_dir / 'val'}.\n"
            "Training without it would validate on shuffled, augmented training "
            "data, making early stopping and best-checkpoint selection "
            "meaningless. Create a val/real + val/fake split first."
        )

    model     = build_model(cfg.num_classes).to(device)
    if initial_checkpoint is not None:
        checkpoint = torch.load(
            initial_checkpoint, map_location=device, weights_only=False
        )
        class_names = tuple(
            checkpoint.get(
                "class_names",
                checkpoint.get("config", {}).get("class_names", ()),
            )
        )
        if class_names and class_names != cfg.class_names:
            raise ValueError(
                f"Checkpoint class order {class_names} does not match expected "
                f"order {cfg.class_names}."
            )
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"  Initial weights loaded from {initial_checkpoint}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = GradScaler(device.type, enabled=(device.type == "cuda"))
    stopper   = EarlyStopping(patience=cfg.patience)

    # TensorBoard — optional, falls back gracefully if not installed
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/resnet18")
    except ImportError:
        logger.warning("  tensorboard not installed — skipping TensorBoard logging.")
        writer = None

    # ── Phase 1: head-only training ──
    freeze_backbone(model)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_head, weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.freeze_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc": [],
    }
    best_val_loss = float("inf")
    best_ckpt     = cfg.models_dir / cfg.cnn_model_name

    logger.info("=" * 55)
    logger.info("  TruthLens — ResNet18 Training")
    logger.info("=" * 55)
    logger.info(f"  Device : {device}")
    logger.info(f"  Epochs : {cfg.num_epochs}  (freeze for {cfg.freeze_epochs})")
    logger.info(f"  Batch  : {cfg.batch_size}")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        # Switch to full fine-tune at freeze_epochs + 1
        if epoch == cfg.freeze_epochs + 1:
            logger.info("─" * 55)
            logger.info(f"  [Epoch {epoch}] Unfreezing backbone → full fine-tune")
            logger.info("─" * 55)
            unfreeze_all(model)
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.lr_finetune, weight_decay=cfg.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=cfg.num_epochs - cfg.freeze_epochs
            )

        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"],
            criterion, optimizer, scaler, device, is_train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # TensorBoard
        if writer is not None:
            writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - t0
        logger.info(
            f"  Epoch [{epoch:02d}/{cfg.num_epochs:02d}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  │  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  "
            f"[{elapsed:.1f}s]"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cfg.models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss":    val_loss,
                    "val_acc":     val_acc,
                    "class_names": cfg.class_names,
                    # Paths are stringified so the checkpoint holds only plain
                    # types; embedding PosixPath forced weights_only=False on
                    # every load, which future torch versions default away from.
                    "config": {
                        k: (str(v) if isinstance(v, Path) else v)
                        for k, v in cfg.__dict__.items()
                    },
                },
                best_ckpt,
            )
            logger.info(f"  ✔ Best checkpoint saved → {best_ckpt.name}")

        # Early stopping
        if stopper(val_loss):
            logger.info(f"  Early stopping triggered at epoch {epoch}.")
            break

    if writer is not None:
        writer.close()

    # Save training curves
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        save_path=cfg.results_dir / "plots" / "training_curves.png",
    )

    save_metrics(history, cfg.results_dir / "metrics" / "cnn_training_history.json")
    logger.info(f"\n  Training complete. Best Val Loss: {best_val_loss:.4f}")
    return history


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train TruthLens ResNet18 model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/prepared/cifake",
        help="Dataset directory (default: datasets/prepared/cifake)",
    )
    parser.add_argument(
        "--extra-data-dir", type=str, action="append", default=[],
        metavar="DIR",
        help="Additional corpus root merged into every split; repeatable. "
             "e.g. --data-dir datasets/prepared/multires --extra-data-dir datasets/prepared/modern_v2",
    )
    parser.add_argument(
        "--modern-aug", action="store_true",
        help="Use the shortcut-breaking train pipeline (JPEG recompression and "
             "aspect padding applied to both classes, lighter photometric "
             "jitter). Required when training on datasets/prepared/modern_v2, whose "
             "container format and image orientation otherwise correlate with "
             "the label.",
    )
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Raise for the modern pipeline: "
                             "decoding and re-encoding megapixel images is the "
                             "bottleneck, not the GPU.")
    parser.add_argument("--model-name", type=str,   default="resnet18_truthlens.pth", help="Model checkpoint output filename")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr-head",    type=float, default=1e-3)
    parser.add_argument("--lr-tune",    type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Existing CNN checkpoint to fine-tune instead of starting from ImageNet.",
    )
    args = parser.parse_args()

    cfg = Config(
        data_dir       = Path(args.data_dir),
        extra_data_dirs= tuple(args.extra_data_dir),
        modern_augment = args.modern_aug,
        cnn_model_name = args.model_name,
        num_epochs     = args.epochs,
        batch_size     = args.batch_size,
        lr_head        = args.lr_head,
        lr_finetune    = args.lr_tune,
        freeze_epochs  = args.freeze_epochs,
        patience       = args.patience,
        num_workers    = args.num_workers,
    )
    train(cfg, initial_checkpoint=args.initial_checkpoint)


if __name__ == "__main__":
    main()
