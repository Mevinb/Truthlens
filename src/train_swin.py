#!/usr/bin/env python3
"""
TruthLens — src/train_swin.py
==============================
Improved SwinV2-Tiny binary real/fake detector trained on the *unified* manifest
(``datasets/prepared/swin_union/manifest.jsonl``), with:
  • Balanced class-weighted loss / Focal loss to resolve dataset imbalance & false alarms on reals
  • Shortcut-breaking data augmentations (blur/sharpening, saturation jitter, multiscale crops, recompression)
  • Real-time Live tracking of Real Accuracy, Fake Recall, Balanced Acc, AUC, Recall@5% FPR, and GPU VRAM
  • Automatic decision threshold calibration saved inside checkpoint metadata
  • Mid-epoch and epoch-boundary atomic resume checkpoints with data-order persistence
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
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


# ─── Loss Functions & Class Balancing ──────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss with alpha class-balancing for binary/multiclass classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        p = torch.softmax(logits, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        modulating_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * modulating_factor * ce_loss
        else:
            loss = modulating_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def calculate_class_weights(class_balance: Dict[str, int]) -> torch.Tensor:
    """Calculate balanced inverse-frequency class weights for [real, fake]."""
    n_real = max(class_balance.get("real", 1), 1)
    n_fake = max(class_balance.get("fake", 1), 1)
    n_total = n_real + n_fake
    w_real = n_total / (2.0 * n_real)
    w_fake = n_total / (2.0 * n_fake)
    return torch.tensor([w_real, w_fake], dtype=torch.float32)


def calculate_calibrated_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05
) -> Tuple[float, float]:
    """Calculate optimal decision threshold on reals to maintain FPR <= target_fpr."""
    real_scores = y_score[y_true == 0]
    fake_scores = y_score[y_true == 1]
    if len(real_scores) == 0 or len(fake_scores) == 0:
        return 0.5, 0.0
    threshold = float(np.percentile(real_scores, 100.0 * (1.0 - target_fpr)))
    recall = float(np.mean(fake_scores >= threshold))
    return threshold, recall


def build_criterion(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.05,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build the loss criterion based on configuration."""
    if loss_type == "focal":
        return FocalLoss(
            alpha=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing
        )
    elif loss_type in ("weighted", "weighted_ce"):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Choose 'weighted_ce', 'focal', or 'ce'"
        )


# ─── Transforms ───────────────────────────────────────────────────────────────
def _train_transform(size: int, strong_aug: bool = True):
    """Shortcut-breaking train pipeline, sized for ``size`` input.

    Mirrors the reasoning in :mod:`src.preprocessing`: recompress both classes
    (the union corpus stores original bytes, so container format otherwise
    correlates with source, not with label), pad to a random aspect (generators
    lean portrait/square, photo archives lean landscape), and sample a
    resized crop so the model sees both whole-frame and local structure across
    epochs.
    """
    from src.preprocessing import RandomAspectPad, RandomRecompress

    aug_list = [
        RandomAspectPad(p=0.3),
        transforms.RandomResizedCrop(
            size, scale=(0.35, 1.0), ratio=(0.75, 1.333), antialias=True
        ),
        RandomRecompress(p=0.9, quality=(50, 98)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    ]
    if strong_aug:
        aug_list.extend([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
            ], p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.2),
            transforms.RandomAutocontrast(p=0.1),
        ])
    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(aug_list)


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

    def __init__(
        self,
        manifest: Path,
        split: str,
        transform,
        with_meta: bool = False,
        limit: int = 0,
    ) -> None:
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
            "real": labels.count(0),
            "fake": labels.count(1),
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
            # One bad file must not kill a multi-hour run.
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
    in_features = model.head.in_features  # 768
    model.head = nn.Linear(in_features, len(CLASS_NAMES))
    return model


# ─── Sampler slice for step-resume ────────────────────────────────────────────
def _atomic_torch_save(obj, path: Path) -> None:
    """torch.save to a temp file + atomic rename.

    A step-resume snapshot is written in the middle of training, where a kill
    can land at any instant; an interrupted write would corrupt the only copy
    of the training state. Writing to a sibling temp file first and then
    ``os.replace`` guarantees the checkpoint path always holds either the old
    or the new complete state, never a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _manifest_identity(path: Path) -> Tuple[str, int]:
    """Hash the manifest bytes and count its non-empty records."""
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as fh:
        for line in fh:
            digest.update(line)
            rows += bool(line.strip())
    return digest.hexdigest(), rows


def _resume_compatibility_errors(
    state: Dict,
    args,
    manifest_sha256: str,
    manifest_rows: int,
    train_rows: int,
) -> List[str]:
    """Return reasons a checkpoint cannot safely continue this invocation."""
    config = state.get("config", {})
    expected = {
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "patience": args.patience,
        "manifest_sha256": manifest_sha256,
        "manifest_rows": manifest_rows,
    }
    errors = []
    for key, current in expected.items():
        saved = config.get(key)
        if saved is None:
            errors.append(f"checkpoint has no {key}")
        elif saved != current:
            errors.append(f"{key}: checkpoint={saved!r}, current={current!r}")
    if "pretrained" in config:
        expected_pretrained = not getattr(args, "no_pretrained", False)
        if config["pretrained"] != expected_pretrained:
            errors.append(
                f"pretrained: checkpoint={config['pretrained']!r}, "
                f"current={expected_pretrained!r}"
            )
    perm = state.get("perm")
    if state.get("step_in_epoch", 0) and (
        not isinstance(perm, list) or len(perm) != train_rows
    ):
        errors.append(
            f"mid-epoch permutation length is "
            f"{len(perm) if isinstance(perm, list) else 'missing'}, "
            f"but train has {train_rows} rows"
        )
    return errors


def _resume_position(state: Dict) -> Tuple[int, int]:
    """Return (epoch, completed batches) represented by a resume state."""
    step = int(state.get("step_in_epoch", 0))
    epoch = int(state.get("epoch", 0))
    return (epoch, step) if step else (epoch + 1, 0)


# ─── Per-epoch weight archive ─────────────────────────────────────────────────
def _eval_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    val_m: Dict,
    args,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """Evaluation-checkpoint contents shared by the best file and the archive."""
    return {
        "model_type": "swin_v2_tiny",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_loss": val_m["loss"],
        "val_acc": val_m["accuracy"],
        "balanced_acc": val_m.get("balanced_acc", val_m["accuracy"]),
        "real_acc": val_m.get("real_acc", 0.0),
        "fake_recall": val_m.get("fake_recall", 0.0),
        "auc": val_m.get("auc", 0.5),
        "threshold": val_m.get("threshold_5fpr", 0.5),
        "calibrated_threshold": val_m.get("threshold_5fpr", 0.5),
        "recall_at_5fpr": val_m.get("recall_at_5fpr", 0.0),
        "val_metrics": val_m,
        "class_names": CLASS_NAMES,
        "img_size": args.img_size,
        "loss_type": getattr(args, "loss_type", "weighted_ce"),
        "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
        "config": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "seed": args.seed,
            "patience": getattr(args, "patience", None),
            "loss_type": getattr(args, "loss_type", "weighted_ce"),
            "manifest": str(args.manifest),
            "manifest_sha256": getattr(args, "manifest_sha256", None),
            "manifest_rows": getattr(args, "manifest_rows", None),
            "pretrained": not args.no_pretrained,
        },
    }


def _epoch_archive_path(tag: str, epoch: int) -> Path:
    """Where an epoch's weights are archived: models/swin_epochs/<tag>_ep<NNN>.pth."""
    return ROOT_DIR / "models" / "swin_epochs" / f"{tag}_ep{epoch:03d}.pth"


def _epoch_index_path(tag: str) -> Path:
    """Metric index for a tag's archive: models/swin_epochs/<tag>_index.json."""
    return ROOT_DIR / "models" / "swin_epochs" / f"{tag}_index.json"


def _update_epoch_index(index_path: Path, entry: Dict) -> None:
    """Upsert one epoch's metrics into the archive index (atomic)."""
    entries: List[Dict] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []  # corrupt index must never cost a training run
    entries = [e for e in entries if e.get("epoch") != entry.get("epoch")]
    entries.append(entry)
    entries.sort(key=lambda e: e.get("epoch", 0))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp, index_path)


class _SliceSampler:
    """Yield a fixed subsequence of dataset indices for step-resume."""

    def __init__(self, indices: List[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


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
    start_step: int = 0,
    snapshot_every: int = 0,
    snapshot_cb=None,
    progress_every: int = 0,
    target_fpr: float = 0.05,
) -> Dict[str, float]:
    """One train/val epoch with gradient accumulation & comprehensive metrics."""
    from tqdm import tqdm

    model.train() if is_train else model.eval()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    running_loss = 0.0
    correct = 0
    total = 0
    correct_real = 0
    total_real = 0
    correct_fake = 0
    total_fake = 0
    probs_all: List[float] = []
    labels_all: List[int] = []
    epoch_started = time.time()

    # tqdm only when stdout is a real terminal
    use_bar = is_train and sys.stdout.isatty()
    iterator = (
        loader
        if not use_bar
        else tqdm(loader, desc="train", unit="batch", dynamic_ncols=True)
    )

    with ctx:
        processed_batches = 0
        for step, (images, labels) in enumerate(iterator):
            processed_batches += 1
            gstep = start_step + step
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss / grad_accum).backward()
                if (gstep + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if (
                        snapshot_every
                        and snapshot_cb
                        and (gstep + 1) % snapshot_every == 0
                        and step + 1 < len(loader)
                    ):
                        snapshot_cb(gstep + 1)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            real_mask = labels == 0
            fake_mask = labels == 1
            total_real += real_mask.sum().item()
            total_fake += fake_mask.sum().item()
            correct_real += (preds[real_mask] == 0).sum().item()
            correct_fake += (preds[fake_mask] == 1).sum().item()

            probs_all.extend(probs.detach().float().cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

            if is_train and use_bar:
                r_acc = (100.0 * correct_real / max(total_real, 1)) if total_real > 0 else 0.0
                f_rec = (100.0 * correct_fake / max(total_fake, 1)) if total_fake > 0 else 0.0
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    real_acc=f"{r_acc:.1f}%",
                    fake_rec=f"{f_rec:.1f}%",
                )

            if progress_every and (
                (gstep + 1) % progress_every == 0 or step + 1 == len(loader)
            ):
                elapsed = max(time.time() - epoch_started, 1e-6)
                done = step + 1
                remaining = max(len(loader) - done, 0)
                eta = datetime.timedelta(seconds=int(elapsed / done * remaining))
                phase = "train" if is_train else "val"
                lr = (
                    optimizer.param_groups[0]["lr"]
                    if optimizer is not None
                    else float("nan")
                )

                cur_real_acc = (
                    (100.0 * correct_real / max(total_real, 1)) if total_real > 0 else 0.0
                )
                cur_fake_rec = (
                    (100.0 * correct_fake / max(total_fake, 1)) if total_fake > 0 else 0.0
                )
                cur_bal_acc = 0.5 * (cur_real_acc + cur_fake_rec)

                vram_str = ""
                if device.type == "cuda" and torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    vram_res = torch.cuda.memory_reserved() / (1024 * 1024)
                    vram_str = f"  VRAM {vram_mb:.0f}/{vram_res:.0f}MB"

                print(
                    f"  [{phase:5s}] batch {gstep + 1:,} "
                    f"({done / max(len(loader), 1):6.2%})  "
                    f"loss {loss.item():.4f} (avg {running_loss / max(total, 1):.4f})  "
                    f"acc {100.0 * correct / max(total, 1):.2f}% "
                    f"[Real: {cur_real_acc:.1f}% | Fake: {cur_fake_rec:.1f}% | Bal: {cur_bal_acc:.1f}%]  "
                    f"lr {lr:.2e}{vram_str}  "
                    f"{total / elapsed:.1f} smp/s  ETA {eta}",
                    flush=True,
                )

    if processed_batches == 0:
        raise ValueError(
            "epoch loader is empty; resume checkpoint is at or past the end of the epoch"
        )

    if is_train and (start_step + step + 1) % grad_accum != 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    p = np.asarray(probs_all)
    y = np.asarray(labels_all)

    real_acc = 100.0 * correct_real / max(total_real, 1) if total_real > 0 else 0.0
    fake_rec = 100.0 * correct_fake / max(total_fake, 1) if total_fake > 0 else 0.0
    bal_acc = 0.5 * (real_acc + fake_rec)

    metrics = {
        "loss": running_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
        "real_acc": real_acc,
        "fake_recall": fake_rec,
        "balanced_acc": bal_acc,
        "n_real": total_real,
        "n_fake": total_fake,
    }

    if (y == 0).any():
        metrics["mean_p_fake_real"] = float(p[y == 0].mean())
    if (y == 1).any():
        metrics["mean_p_fake_fake"] = float(p[y == 1].mean())

    if (y == 0).any() and (y == 1).any():
        from sklearn.metrics import roc_auc_score

        try:
            metrics["auc"] = float(roc_auc_score(y, p))
        except Exception:
            metrics["auc"] = 0.5

        thresh_5fpr, rec_5fpr = calculate_calibrated_threshold(
            y, p, target_fpr=target_fpr
        )
        metrics["threshold_5fpr"] = thresh_5fpr
        metrics["recall_at_5fpr"] = rec_5fpr * 100.0

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
    print(
        f"[Device] {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )
    if device.type == "cpu":
        print("WARNING: no GPU found — training will be slow")

    manifest = ROOT_DIR / args.manifest
    if not manifest.exists():
        print(
            f"Manifest not found: {manifest}\n"
            "Build it first:\n  python dataset/build_swin_union_manifest.py"
        )
        return 1

    train_ds = ManifestDataset(
        manifest,
        "train",
        _train_transform(args.img_size, strong_aug=not args.no_strong_aug),
    )
    val_ds = ManifestDataset(manifest, "val", _eval_transform(args.img_size))
    manifest_sha256, manifest_rows = _manifest_identity(manifest)
    args.manifest_sha256 = manifest_sha256
    args.manifest_rows = manifest_rows

    class_weights = calculate_class_weights(train_ds.class_balance).to(device)
    print(f"\n  Dataset Summary:")
    print(
        f"    Train split : {train_ds.class_balance['n']:,} images "
        f"(Real: {train_ds.class_balance['real']:,} | Fake: {train_ds.class_balance['fake']:,})"
    )
    print(
        f"    Val split   : {val_ds.class_balance['n']:,} images "
        f"(Real: {val_ds.class_balance['real']:,} | Fake: {val_ds.class_balance['fake']:,})"
    )
    print(
        f"    Loss Type   : {args.loss_type} (Class Weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f})"
    )
    print(f"    Manifest ID : {manifest_sha256[:16]}… ({manifest_rows:,} total rows)")

    train_gen = torch.Generator()
    train_gen.set_state(torch.random.get_rng_state())
    epoch_perm: List[int] = []

    def _make_train_loader(
        use_perm: Optional[List[int]] = None, slice_from: int = 0
    ) -> DataLoader:
        nonlocal epoch_perm
        if use_perm is not None:
            epoch_perm = use_perm
        else:
            epoch_perm = torch.randperm(len(train_ds), generator=train_gen).tolist()
        sampler = _SliceSampler(epoch_perm[slice_from:])
        return DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)

    if getattr(args, "init_weights", None) and not args.resume:
        init_path = ROOT_DIR / args.init_weights if not Path(args.init_weights).is_absolute() else Path(args.init_weights)
        if init_path.exists():
            print(f"  [Weight Init] Loading starting weights from {init_path.name}...")
            init_state = torch.load(init_path, map_location="cpu", weights_only=False)
            state_dict = init_state.get("model_state", init_state.get("state_dict", init_state))
            model.load_state_dict(state_dict)
            print(f"  [Weight Init] ✔ Successfully initialized model from {init_path.name}")
        else:
            print(f"  [Weight Init] ⚠ Warning: Checkpoint {init_path} not found; falling back to default initialization.")

    criterion = build_criterion(
        loss_type=args.loss_type,
        class_weights=class_weights if args.loss_type != "ce" else None,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
    )
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(train_ds) // args.batch_size // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")
    best_bal_acc = 0.0
    best_ckpt = ROOT_DIR / "models" / args.model_name
    resume_ckpt = ROOT_DIR / "models" / f"swin_resume_{args.tag}.pth"
    history = []
    epoch_times: List[float] = []
    start_epoch = 1
    resume_step = 0
    resume_perm: Optional[List[int]] = None

    if args.resume:
        rp = ROOT_DIR / args.resume
        if not rp.exists():
            print(f"Resume checkpoint not found: {rp}")
            return 1
        state = torch.load(rp, map_location="cpu", weights_only=False)
        compatibility_errors = _resume_compatibility_errors(
            state, args, manifest_sha256, manifest_rows, len(train_ds)
        )
        if compatibility_errors and not args.allow_resume_mismatch:
            print("Resume checkpoint is incompatible with this dataset/config:")
            for error in compatibility_errors:
                print(f"  - {error}")
            print(
                "Use a new TAG for a fresh run or pass --allow-resume-mismatch if intentional."
            )
            return 1
        if compatibility_errors:
            print("WARNING: forcing an incompatible resume:")
            for error in compatibility_errors:
                print(f"  - {error}")
        model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])
        if "stopper" in state:
            stopper.best = state["stopper"]["best"]
            stopper.counter = state["stopper"]["counter"]
        stopper.stop = False
        best_val_loss = state.get("best_val_loss", float("inf"))
        best_bal_acc = state.get("best_bal_acc", 0.0)
        history = state.get("history", [])
        epoch_times = state.get("epoch_times", [])
        resume_epoch, step_in_epoch = _resume_position(state)
        if step_in_epoch:
            start_epoch = resume_epoch
            resume_step = step_in_epoch
            resume_perm = state.get("perm")
            if "train_gen_state" in state:
                train_gen.set_state(state["train_gen_state"].cpu())
            print(
                f"  Resuming epoch {start_epoch} at batch {resume_step} (mid-epoch step-resume)"
            )
        else:
            start_epoch = resume_epoch
            if "train_gen_state" in state:
                train_gen.set_state(state["train_gen_state"].cpu())
            print(
                f"  Resuming from epoch {start_epoch} (best_val_loss {best_val_loss:.4f}, {len(history)} epoch(s) recorded)"
            )
        if "scheduler_state" not in state:
            scheduler.last_epoch = start_epoch - 1

    print(
        f"\n  Training Configuration:\n"
        f"    Model Architecture : SwinV2-Tiny ({args.img_size}x{args.img_size})\n"
        f"    Batch Size / Accum : {args.batch_size} / {args.grad_accum} (Effective: {args.batch_size * args.grad_accum})\n"
        f"    Learning Rate      : {args.lr:.2e} (Cosine Annealing to {args.lr * 0.01:.2e})\n"
        f"    Target FPR on Reals: {args.target_fpr * 100:.1f}%\n"
        f"    Checkpoints        : Best -> models/{best_ckpt.name} | Resume -> models/{resume_ckpt.name}\n"
    )

    def _save_step_state(epoch: int, completed_batches: int) -> None:
        _atomic_torch_save(
            {
                "model_type": "swin_v2_tiny",
                "epoch": epoch,
                "step_in_epoch": completed_batches,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "stopper": {"best": stopper.best, "counter": stopper.counter},
                "best_val_loss": best_val_loss,
                "best_bal_acc": best_bal_acc,
                "history": history,
                "epoch_times": epoch_times,
                "perm": epoch_perm,
                "train_gen_state": train_gen.get_state(),
                "class_names": CLASS_NAMES,
                "img_size": args.img_size,
                "loss_type": args.loss_type,
                "config": {
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "patience": args.patience,
                    "loss_type": args.loss_type,
                    "manifest": str(args.manifest),
                    "manifest_sha256": manifest_sha256,
                    "manifest_rows": manifest_rows,
                    "pretrained": not args.no_pretrained,
                },
            },
            resume_ckpt,
        )
        print(f"  💾 Step-resume snapshot @ batch {completed_batches} -> {resume_ckpt.name}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if resume_perm is not None:
            slice_from = resume_step * args.batch_size
            train_loader = _make_train_loader(resume_perm, slice_from=slice_from)
            start_step = resume_step
            print(f"  ↪ Continuing epoch {epoch} from batch {resume_step}")
            resume_perm, resume_step = None, 0
        else:
            train_loader = _make_train_loader()
            start_step = 0

        train_m = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            True,
            args.grad_accum,
            start_step=start_step,
            snapshot_every=args.resume_every,
            snapshot_cb=lambda completed: _save_step_state(epoch, completed),
            progress_every=args.progress_every,
            target_fpr=args.target_fpr,
        )
        scheduler.step()
        val_m = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            scaler,
            device,
            False,
            progress_every=args.progress_every,
            target_fpr=args.target_fpr,
        )
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
            "train_metrics_partial": bool(start_step),
        }
        history.append(record)

        # Print Beautiful Formatted Epoch Summary Table
        print("\n" + "═" * 84)
        print(f"  📊 EPOCH {epoch:02d}/{args.epochs:02d} SUMMARY  (Elapsed: {elapsed:.1f}s)")
        print("─" * 84)
        print(f"  {'Metric':<22} | {'Train':<26} | {'Validation':<26}")
        print("─" * 84)
        print(f"  {'Loss':<22} | {train_m['loss']:<26.4f} | {val_m['loss']:<26.4f}")
        print(f"  {'Overall Accuracy':<22} | {train_m['accuracy']:<25.2f}% | {val_m['accuracy']:<25.2f}%")
        print(f"  {'Balanced Accuracy':<22} | {train_m.get('balanced_acc', 0.0):<25.2f}% | {val_m.get('balanced_acc', 0.0):<25.2f}%")
        print(f"  {'Real Accuracy':<22} | {train_m.get('real_acc', 0.0):<25.2f}% | {val_m.get('real_acc', 0.0):<25.2f}%")
        print(f"  {'Fake Recall':<22} | {train_m.get('fake_recall', 0.0):<25.2f}% | {val_m.get('fake_recall', 0.0):<25.2f}%")
        print(f"  {'ROC AUC':<22} | {train_m.get('auc', 0.0):<26.4f} | {val_m.get('auc', 0.0):<26.4f}")
        if "threshold_5fpr" in val_m:
            print(f"  {'Calibrated Threshold':<22} | {'-':<26} | {val_m['threshold_5fpr']:<26.4f}")
            print(f"  {'Recall @ 5% FPR':<22} | {'-':<26} | {val_m['recall_at_5fpr']:<25.2f}%")
        print("═" * 84 + "\n")

        should_stop = stopper(val_m["loss"])

        mean_epoch_s = float(np.mean(epoch_times))
        finish_epoch = args.epochs + 1
        if not stopper.stop:
            eta_s = mean_epoch_s * (finish_epoch - epoch)
            eta_dt = datetime.timedelta(seconds=int(eta_s))
            eta_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta_s)
            print(
                f"  ⏱  Pace: {mean_epoch_s/3600:.2f} h/epoch -> Est. Finish: {eta_finish.strftime('%Y-%m-%d %H:%M')} ({eta_dt.days}d {eta_dt.seconds//3600:02d}h {(eta_dt.seconds//60)%60:02d}m left)"
            )

        # Archive per-epoch weights
        archive = _epoch_archive_path(args.tag, epoch)
        try:
            _atomic_torch_save(
                _eval_checkpoint_payload(epoch, model, val_m, args, class_weights), archive
            )
            _update_epoch_index(
                _epoch_index_path(args.tag),
                {
                    "epoch": epoch,
                    "val_loss": round(val_m["loss"], 5),
                    "val_acc": round(val_m["accuracy"], 4),
                    "val_bal_acc": round(val_m.get("balanced_acc", 0.0), 4),
                    "val_real_acc": round(val_m.get("real_acc", 0.0), 4),
                    "val_fake_recall": round(val_m.get("fake_recall", 0.0), 4),
                    "val_auc": round(val_m.get("auc", 0.0), 4),
                    "val_threshold": round(val_m.get("threshold_5fpr", 0.5), 4),
                    "train_loss": round(train_m["loss"], 5),
                    "file": archive.name,
                },
            )
            print(f"  🗄  Epoch weights archived -> {archive.relative_to(ROOT_DIR)}")
        except OSError as exc:
            print(f"  ⚠ Could not archive epoch {epoch} weights: {exc}")

        # Save Best Model Checkpoint
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_bal_acc = val_m.get("balanced_acc", val_m["accuracy"])
            _atomic_torch_save(
                _eval_checkpoint_payload(epoch, model, val_m, args, class_weights), best_ckpt
            )
            print(f"  ✔ Best checkpoint saved (Val Loss: {best_val_loss:.4f}, BalAcc: {best_bal_acc:.2f}%) -> {best_ckpt.name}")

        # Save Resume Checkpoint
        _atomic_torch_save(
            {
                "model_type": "swin_v2_tiny",
                "epoch": epoch,
                "step_in_epoch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "stopper": {"best": stopper.best, "counter": stopper.counter},
                "best_val_loss": best_val_loss,
                "best_bal_acc": best_bal_acc,
                "history": history,
                "epoch_times": epoch_times,
                "perm": epoch_perm,
                "train_gen_state": train_gen.get_state(),
                "class_names": CLASS_NAMES,
                "img_size": args.img_size,
                "loss_type": args.loss_type,
                "config": {
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "patience": args.patience,
                    "loss_type": args.loss_type,
                    "manifest": str(args.manifest),
                    "manifest_sha256": manifest_sha256,
                    "manifest_rows": manifest_rows,
                    "pretrained": not args.no_pretrained,
                },
            },
            resume_ckpt,
        )
        print(f"  💾 Resume state saved -> {resume_ckpt.name}")

        if should_stop:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    out_dir = ROOT_DIR / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"swin_history_{args.tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": "swin_v2_tiny",
                "img_size": args.img_size,
                "tag": args.tag,
                "best_val_loss": best_val_loss,
                "best_bal_acc": best_bal_acc,
                "loss_type": args.loss_type,
                "epochs": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote training history to {out_path.relative_to(ROOT_DIR)}")
    print(f"Best val loss: {best_val_loss:.4f} at {best_ckpt}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=str,
        default="datasets/prepared/swin_union/manifest.jsonl",
    )
    ap.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="input resolution: 512 (stage 1) or 1024 (stage 2)",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="gradient accumulation steps; effective batch = batch-size * grad-accum",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--loss-type",
        type=str,
        choices=["weighted_ce", "focal", "ce"],
        default="weighted_ce",
        help="loss function formulation: weighted_ce (balanced cross-entropy), focal (focal loss), or ce",
    )
    ap.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="gamma exponent for focal loss",
    )
    ap.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="label smoothing factor",
    )
    ap.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="target false positive rate on reals for decision threshold calibration (default: 0.05 / 5%)",
    )
    ap.add_argument(
        "--no-strong-aug",
        action="store_true",
        help="disable enhanced anti-shortcut augmentations (gaussian blur, sharpness jitter, etc.)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early-stopping patience on val loss",
    )
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--resume-every",
        type=int,
        default=2000,
        help="save a step-resume snapshot every N train batches (0 disables)",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume from a training-state checkpoint (e.g. models/swin_resume_swin_v2_512.pth)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print detailed train/validation metrics every N batches (default: 50)",
    )
    ap.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="force loading a checkpoint whose dataset identity or batch configuration differs",
    )
    ap.add_argument("--tag", type=str, default="swin_v2_512_improved")
    ap.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="checkpoint filename under models/ (default: swin_v2_tiny_{img_size}_improved.pth)",
    )
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch without ImageNet pretrained weights")
    ap.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="initialize model backbone/head from an existing checkpoint file (e.g. models/swin_v2_tiny_512_hardstyles.pth)",
    )
    args = ap.parse_args()

    if args.model_name is None:
        args.model_name = f"swin_v2_tiny_{args.img_size}_improved.pth"

    if args.resume_every > 0:
        args.resume_every = max(args.resume_every // args.grad_accum, 1) * args.grad_accum
    else:
        args.resume_every = 0

    return train(args)


if __name__ == "__main__":
    sys.exit(main())
