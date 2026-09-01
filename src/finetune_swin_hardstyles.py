#!/usr/bin/env python3
"""TruthLens — src/finetune_swin_hardstyles.py
=============================================
Targeted fine-tune of the trained SwinV2-Tiny on hard styles + modern replay.

Root cause being fixed (measured on a 242-image fresh-internet corpus):
  * real paintings / night / HDR / minimal / B&W photography flagged FAKE
  * painterly AI, AI product shots, photoreal AI people passed as REAL
  * training fakes were 82.5% legacy (CIFAKE 32px / SD-1.4 era), so modern
    photoreal and stylistic generators were under-represented.

Data = datasets/prepared/hard_styles (new style buckets) + a stratified replay
sample from the swin_union TRAIN split (never val/test/holdout) that keeps the
model calibrated on the distribution it must not forget.

The fresh-internet test corpus and the union val/test/holdout splits are NEVER
trained on.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.train_swin import (LABEL_MAP, ManifestDataset, _atomic_torch_save,
                            _eval_transform, _train_transform, build_model,
                            _epoch_archive_path, _update_epoch_index,
                            _eval_checkpoint_payload)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial-checkpoint", default="models/swin_v2_tiny_512.pth")
    ap.add_argument("--out", default="models/swin_v2_tiny_512_hardstyles.pth")
    ap.add_argument("--hard-styles-dir", default="datasets/prepared/hard_styles")
    ap.add_argument("--manifest", default="datasets/prepared/swin_union/manifest.jsonl")
    ap.add_argument("--replay-real", type=int, default=5000)
    ap.add_argument("--replay-fake-modern", type=int, default=4000)
    ap.add_argument("--replay-fake-legacy", type=int, default=2000)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-every-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-limit", type=int, default=4000,
                    help="union-val subsample for the periodic check")
    ap.add_argument("--hold-frac", type=float, default=0.1,
                    help="fraction of hard_styles held out as its own val")
    return ap.parse_args()


# ── data assembly ────────────────────────────────────────────────────────────
MODERN_FAKE_KEYS = ("gpt-image", "nano-banana", "flux", "midjourney",
                    "chatgpt", "seedream", "imagen", "hidream", "ideogram",
                    "grok", "firefly", "meta", "recraft", "dalle3", "qwen")


def load_union_rows(manifest: Path):
    rows = {"real": [], "modern": [], "legacy": []}
    with manifest.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "train":
                continue
            if not Path(r["path"]).exists():
                continue
            if r["label"] == "real":
                rows["real"].append(r)
            else:
                g = (r.get("generator") or "?").lower()
                rows["modern" if any(k in g for k in MODERN_FAKE_KEYS) else "legacy"].append(r)
    return rows


def load_hard_styles(hard_dir: Path, hold_frac: float, rng: random.Random):
    """Split hard_styles rows into train/val, stratified per bucket."""
    meta = hard_dir / "manifest.jsonl"
    buckets: dict[tuple, list] = {}
    for line in meta.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not Path(r["path"]).exists():
            continue
        buckets.setdefault((r["label"], r["generator"]), []).append(r)
    train_rows, val_rows = [], []
    for key, rows in sorted(buckets.items()):
        rows = rows[:]
        rng.shuffle(rows)
        n_hold = max(6, int(len(rows) * hold_frac)) if len(rows) >= 20 else max(2, len(rows) // 5)
        val_rows += rows[:n_hold]
        train_rows += rows[n_hold:]
    return train_rows, val_rows


class MixedDataset(Dataset):
    """Union of manifest rows; returns (tensor, label) for training."""

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        from PIL import Image
        try:
            with Image.open(r["path"]) as im:
                img = im.convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64), (128, 128, 128))
        return self.transform(img), LABEL_MAP[r["label"]]


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    from sklearn.metrics import roc_auc_score
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    ps, ys = [], []
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
        p = torch.softmax(logits, 1)[:, 1]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        ps += p.float().cpu().tolist()
        ys += y.cpu().tolist()
    import numpy as np
    y = np.array(ys)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": 100.0 * correct / max(total, 1),
        "real_acc": float((np.array(ps)[y == 0] < 0.5).mean()),
        "fake_recall": float((np.array(ps)[y == 1] >= 0.5).mean()),
        "auc": float(roc_auc_score(y, ps)),
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    # ── hard styles ───────────────────────────────────────────────────────
    hs_train, hs_val = load_hard_styles(ROOT_DIR / args.hard_styles_dir,
                                        args.hold_frac, rng)
    n_f = sum(1 for r in hs_train if r["label"] == "fake")
    n_r = len(hs_train) - n_f
    print(f"hard_styles train: {len(hs_train)} (fake {n_f} / real {n_r})  "
          f"val: {len(hs_val)}")

    # ── replay from union train ───────────────────────────────────────────
    union = load_union_rows(ROOT_DIR / args.manifest)
    rng.shuffle(union["real"])
    rng.shuffle(union["modern"])
    rng.shuffle(union["legacy"])
    replay = (union["real"][:args.replay_real]
              + union["modern"][:args.replay_fake_modern]
              + union["legacy"][:args.replay_fake_legacy])
    print(f"replay: real {min(args.replay_real, len(union['real']))}  "
          f"modern-fake {min(args.replay_fake_modern, len(union['modern']))}  "
          f"legacy-fake {min(args.replay_fake_legacy, len(union['legacy']))}")

    train_rows = hs_train + replay
    rng.shuffle(train_rows)
    print(f"TOTAL train rows: {len(train_rows)}")

    train_ds = MixedDataset(train_rows, _train_transform(args.img_size))
    hs_val_ds = MixedDataset(hs_val, _eval_transform(args.img_size))

    # union-val subsample (regression guard, never trained on)
    val_manifest = ROOT_DIR / args.manifest
    val_rows = []
    with val_manifest.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "val" and Path(r["path"]).exists():
                val_rows.append(r)
    rng.shuffle(val_rows)
    val_rows = val_rows[:args.val_limit]
    union_val_ds = ManifestDataset.__new__(ManifestDataset)
    union_val_ds.rows = val_rows
    union_val_ds.transform = _eval_transform(args.img_size)
    union_val_ds.with_meta = False
    union_val_ds.class_balance = {}

    g = torch.Generator(); g.set_state(torch.random.get_rng_state())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, generator=g,
                              persistent_workers=args.num_workers > 0)
    hs_val_loader = DataLoader(hs_val_ds, batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=4, pin_memory=True)
    union_val_loader = DataLoader(union_val_ds, batch_size=args.batch_size * 2,
                                  shuffle=False, num_workers=4, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────
    ckpt_path = ROOT_DIR / args.initial_checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {ckpt_path.name} (epoch {ckpt.get('epoch')}, "
          f"val_acc {ckpt.get('val_acc'):.2f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05)
    print(f"steps/epoch {steps_per_epoch}  total {total_steps}")

    best = {"key": float("inf"), "payload": None}
    archive_dir = ROOT_DIR / "models" / "swin_epochs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    out_path = ROOT_DIR / args.out
    history = []
    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % 50 == 0:
                    el = time.time() - t0
                    eta = el / global_step * (total_steps - global_step)
                    print(f"  e{epoch} step {global_step}/{total_steps} "
                          f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                          f"[{el/60:.0f}m elapsed, {eta/60:.0f}m left]", flush=True)

                if global_step % args.val_every_steps == 0:
                    hs_m = evaluate(model, hs_val_loader, device)
                    un_m = evaluate(model, union_val_loader, device)
                    print(f"  ⤏ val hard_styles: acc {hs_m['acc']:.2f}% "
                          f"real {hs_m['real_acc']:.3f} fake {hs_m['fake_recall']:.3f} "
                          f"auc {hs_m['auc']:.4f} || union-val: acc {un_m['acc']:.2f}% "
                          f"real {un_m['real_acc']:.3f} fake {un_m['fake_recall']:.3f} "
                          f"auc {un_m['auc']:.4f}", flush=True)
                    history.append({"step": global_step, "hard": hs_m, "union": un_m})
                    # selection: hard-val loss must improve without union-val
                    # collapsing more than 1 point of accuracy
                    key = hs_m["loss"] + max(0.0, (96.5 - un_m["acc"])) * 0.05
                    if key < best["key"]:
                        best["key"] = key
                        best["payload"] = _atomic_payload(model, epoch, hs_m, un_m, args)
                        _atomic_torch_save(best["payload"], out_path)
                        print(f"  ✔ saved → {out_path.name}", flush=True)

        # end of epoch: always evaluate + save if better
        hs_m = evaluate(model, hs_val_loader, device)
        un_m = evaluate(model, union_val_loader, device)
        history.append({"epoch": epoch, "hard": hs_m, "union": un_m})
        print(f"  Epoch {epoch} done — hard acc {hs_m['acc']:.2f}% "
              f"(real {hs_m['real_acc']:.3f} fake {hs_m['fake_recall']:.3f}) "
              f"union acc {un_m['acc']:.2f}%", flush=True)
        key = hs_m["loss"] + max(0.0, (96.5 - un_m["acc"])) * 0.05
        if key < best["key"]:
            best["key"] = key
            best["payload"] = _atomic_payload(model, epoch, hs_m, un_m, args)
            _atomic_torch_save(best["payload"], out_path)
            print(f"  ✔ saved → {out_path.name}", flush=True)

        # Save every completed epoch as a durable archive.
        epoch_archive = _epoch_archive_path("hardstyles", epoch)
        epoch_payload = _eval_checkpoint_payload(epoch, model, hs_m, args)
        _atomic_torch_save(epoch_payload, epoch_archive)
        _update_epoch_index(
            _epoch_archive_path("hardstyles", epoch).with_name("hardstyles_index.json"),
            {"epoch": epoch,
             "val_loss": round(hs_m["loss"], 5),
             "val_acc": round(hs_m["acc"], 4),
             "train_loss": round(hs_m["loss"], 5),
             "file": epoch_archive.name})
        print(f"  🗄 epoch archive → {epoch_archive.relative_to(ROOT_DIR)}", flush=True)

    (ROOT_DIR / "results/metrics/swin_hardstyles_history.json").write_text(
        json.dumps({"args": vars(args), "history": history}, indent=2) + "\n")
    print(f"\nDONE in {(time.time()-t0)/60:.0f}m — best at {out_path}")
    return 0


def _atomic_payload(model, epoch, hs_m, un_m, args) -> dict:
    return {
        "model_type": "swin_v2_tiny",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_loss": hs_m["loss"],
        "val_acc": hs_m["acc"],
        "class_names": ("REAL", "FAKE"),
        "img_size": args.img_size,
        "config": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "seed": args.seed,
            "manifest": "hard_styles+union-replay",
            "pretrained": False,
            "finetuned_from": args.initial_checkpoint,
            "union_val_acc": un_m["acc"],
        },
    }


if __name__ == "__main__":
    sys.exit(main())
