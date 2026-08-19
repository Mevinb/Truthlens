#!/usr/bin/env python3
"""
TruthLens — src/train_swin.py
==============================
SwinV2-Tiny binary real/fake detector trained on the *unified* manifest
(``datasets/prepared/swin_union/manifest.jsonl``), whose rows are sha256
deduplicated across splits so no image is fitted and scored.

Why SwinV2-Tiny
---------------
The union corpus is ~275K images spanning 32px CIFAKE thumbnails to 20MP
camera originals, with every modern generator family mixed in. SwinV2's
shifted-window attention reads local high-frequency structure (the evidence
that separates a neural image decoder from a camera sensor) at native scale,
and the Tiny variant fits the 6 GB RTX 4050 at 512×512 (batch 4) and at
1024×1024 (batch 1 + gradient accumulation).

Two-stage protocol (stage selected with ``--img-size``)
-------------------------------------------------------
Stage 1 (512×512):  the baseline. Fast, establishes the pipeline and the
                    first honest numbers.
Stage 2 (1024×1024): the resolution experiment. Same data, same seed, same
                    schedule — only the input size changes, so the comparison
                    is the experiment, not incidental differences.

Both stages: AMP, gradient accumulation, AdamW + cosine schedule, early
stopping on validation loss, best-checkpoint saving.

Usage
-----
    .venv/bin/python src/train_swin.py --img-size 512 --epochs 20 \\
        --batch-size 4 --grad-accum 4 --tag swinv2_512
    .venv/bin/python src/train_swin.py --img-size 1024 --epochs 20 \\
        --batch-size 1 --grad-accum 8 --tag swinv2_1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Swin_V2_T_Weights, swin_v2_t

from src.utils import get_logger, seed_everything

logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("REAL", "FAKE")
LABEL_MAP = {"real": 0, "fake": 1}


# ─── Transforms ───────────────────────────────────────────────────────────────
def _train_transform(size: int):
    """Shortcut-breaking train pipeline, sized for ``size`` input.

    Mirrors the reasoning in :mod:`src.preprocessing`: recompress both classes
    (the union corpus stores original bytes, so container format otherwise
    correlates with source, not with label), pad to a random aspect (generators
    lean portrait/square, photo archives lean landscape), and sample a
    resized crop so the model sees both whole-frame and local structure across
    epochs.
    """
    from src.preprocessing import RandomAspectPad, RandomRecompress
    return transforms.Compose([
        RandomAspectPad(p=0.3),
        transforms.RandomResizedCrop(
            size, scale=(0.4, 1.0), ratio=(0.75, 1.333), antialias=True),
        RandomRecompress(p=1.0, quality=(60, 98)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_transform(size: int):
    """Deterministic val/test view.

    Plain square resize (no aspect crop) keeps the transform identical for
    every row regardless of source resolution, which is what a fair
    cross-corpus comparison needs. The fixed-quality recompress closes the
    container shortcut on the eval path the same way it is closed in train.
    """
    from src.preprocessing import RandomRecompress
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        RandomRecompress(p=1.0, quality=(88, 88)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── Manifest dataset ─────────────────────────────────────────────────────────
class ManifestDataset(Dataset):
    """Rows from the unified manifest, carrying provenance through.

    ``with_meta`` switches the return contract: (tensor, label) for training,
    (tensor, label, meta_dict) for evaluation. The meta dict is the manifest
    row, so per-source and per-generator scoring in ``src/eval_swin.py`` can
    group by any recorded field without a second manifest pass.
    """

    def __init__(self, manifest: Path, split: str, transform,
                 with_meta: bool = False, limit: int = 0) -> None:
        self.transform = transform
        self.with_meta = with_meta
        self.rows: List[dict] = []
        manifest = Path(manifest)
        with manifest.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") != split:
                    continue
                self.rows.append(r)
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No rows in {manifest} for split={split!r}")
        labels = [LABEL_MAP[r["label"]] for r in self.rows]
        self.class_balance = {
            "real": labels.count(0), "fake": labels.count(1),
            "n": len(labels),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        from PIL import Image
        try:
            with Image.open(r["path"]) as im:
                img = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            # One bad file must not kill a 60-hour run.
            img = Image.new("RGB", (64, 64), (128, 128, 128))
            logger.warning("decode failed for %s — %s", r["path"], exc)
        x = self.transform(img)
        y = LABEL_MAP[r["label"]]
        if self.with_meta:
            return x, y, r
        return x, y


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(pretrained: bool = True) -> nn.Module:
    """SwinV2-Tiny with a two-class head."""
    weights = Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_v2_t(weights=weights)
    in_features = model.head.in_features          # 768
    model.head = nn.Linear(in_features, len(CLASS_NAMES))
    return model


# ─── One epoch ────────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    device: torch.device,
    is_train: bool,
    grad_accum: int = 1,
) -> Dict[str, float]:
    """One train/val epoch with gradient accumulation. Returns metrics dict."""
    model.train() if is_train else model.eval()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    running_loss = 0.0
    correct = 0
    total = 0
    probs_all: List[float] = []
    labels_all: List[int] = []

    with ctx:
        for step, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                # Scale once per effective batch: divide by accum so the mean
                # over the effective batch is the loss actually minimised.
                # Must go through scaler.scale() or _scale stays None and
                # unscale_ below fails.
                scaler.scale(loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            probs_all.extend(probs.detach().float().cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    if is_train and (step + 1) % grad_accum != 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    p = np.asarray(probs_all)
    y = np.asarray(labels_all)
    metrics = {
        "loss": running_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }
    if (y == 0).any() and (y == 1).any():
        from sklearn.metrics import roc_auc_score
        metrics["auc"] = float(roc_auc_score(y, p))
    return metrics


# ─── Early stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─── Main ─────────────────────────────────────────────────────────────────────
def train(args) -> int:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    if device.type == "cpu":
        print("WARNING: no GPU found — training will be extremely slow")

    manifest = ROOT_DIR / args.manifest
    if not manifest.exists():
        print(f"Manifest not found: {manifest}\n"
              "Build it first:\n  python dataset/build_swin_union_manifest.py")
        return 1

    train_ds = ManifestDataset(manifest, "train", _train_transform(args.img_size))
    val_ds = ManifestDataset(manifest, "val", _eval_transform(args.img_size))
    print(f"\n  train: {train_ds.class_balance}")
    print(f"  val  : {val_ds.class_balance}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")
    best_ckpt = ROOT_DIR / "models" / args.model_name
    history = []

    print(f"\n  SwinV2-Tiny  {args.img_size}x{args.img_size}"
          f"  batch {args.batch_size}  grad_accum {args.grad_accum}"
          f"  (effective {args.batch_size * args.grad_accum})"
          f"  epochs {args.epochs}  lr {args.lr}")
    print(f"  steps/epoch {steps_per_epoch}  total steps {total_steps}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler,
                            device, True, args.grad_accum)
        scheduler.step()
        val_m = run_epoch(model, val_loader, criterion, None, scaler,
                          device, False)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)
        print(f"  Epoch {epoch:02d}/{args.epochs:02d}  "
              f"train loss {train_m['loss']:.4f} acc {train_m['accuracy']:.2f}%"
              f" auc {train_m.get('auc', float('nan')):.4f}  │  "
              f"val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.2f}%"
              f" auc {val_m.get('auc', float('nan')):.4f}"
              f"  [{elapsed:.0f}s]")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "swin_v2_tiny",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"],
                "class_names": CLASS_NAMES,
                "img_size": args.img_size,
                "config": {
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "manifest": str(args.manifest),
                    "pretrained": not args.no_pretrained,
                },
            }, best_ckpt)
            print(f"  ✔ Best checkpoint saved → {best_ckpt}")

        if stopper(val_m["loss"]):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    out_dir = ROOT_DIR / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"swin_history_{args.tag}.json"
    out_path.write_text(json.dumps(
        {"model": "swin_v2_tiny", "img_size": args.img_size,
         "tag": args.tag, "best_val_loss": best_val_loss,
         "epochs": history}, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(ROOT_DIR)}")
    print(f"Best val loss {best_val_loss:.4f} at {best_ckpt}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=str,
                    default="datasets/prepared/swin_union/manifest.jsonl")
    ap.add_argument("--img-size", type=int, default=512,
                    help="input resolution: 512 (stage 1) or 1024 (stage 2)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="gradient accumulation steps; effective batch = "
                         "batch-size × grad-accum")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4,
                    help="early-stopping patience on val loss")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="swinv2_512")
    ap.add_argument("--model-name", type=str, default=None,
                    help="checkpoint filename under models/ "
                         "(default: swin_v2_tiny_{img_size}.pth)")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    if args.model_name is None:
        args.model_name = f"swin_v2_tiny_{args.img_size}.pth"
    return train(args)


if __name__ == "__main__":
    sys.exit(main())