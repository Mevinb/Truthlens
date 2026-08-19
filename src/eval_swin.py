#!/usr/bin/env python3
"""
TruthLens — src/eval_swin.py
============================
Evaluate a trained SwinV2-Tiny checkpoint on the unified manifest's held-out
splits, with the metrics the retrain plan actually needs:

* overall accuracy / precision / recall / F1 / ROC-AUC / balanced accuracy,
* the same numbers broken out **by source** and **by generator** (per-class,
  like ``diag_eval_all``) so a detector that memorised one generator's visual
  quirks shows up instead of hiding in the aggregate.

Splits are scored exactly as the manifest declares them — ``test`` is the
primary held-out split; ``holdout`` is the quarantined modern_v2 set (unseen
generators/sources) and is scored separately, never pooled into ``test``.

Per-image scores are cached to
``results/metrics/scores/swin_<tag>_<split>.jsonl`` so the breakdowns can be
re-reported without a second GPU pass.

Usage
-----
    .venv/bin/python src/eval_swin.py --checkpoint models/swin_v2_tiny_512.pth \\
        --tag swinv2_512 --img-size 512
    .venv/bin/python src/eval_swin.py --reuse-scores --tag swinv2_512
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from src.train_swin import (
    LABEL_MAP, ManifestDataset, _eval_transform, build_model,
)

SCORE_DIR = ROOT_DIR / "results" / "metrics" / "scores"


def _meta_collate(batch):
    """Keep meta as a list of dicts instead of a dict-of-lists.

    ``torch``'s default collate recurses into dicts and stacks each key, which
    is wrong for manifest rows (mixed str/int/None values) and destroys the
    one-to-one correspondence with tensors that ``zip`` in :func:`score_split`
    relies on.
    """
    xs, ys, metas = zip(*batch)
    return torch.stack(list(xs)), torch.tensor(list(ys)), list(metas)


def per_group_metrics(records: List[dict], group_field: str) -> Dict[str, dict]:
    """Accuracy / recall / precision / F1 / mean-P(FAKE) per group+label."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        groups[r[group_field]].append(r)
    out: Dict[str, dict] = {}
    for g, rows in sorted(groups.items()):
        y = np.array([LABEL_MAP[r["label"]] for r in rows])
        p = np.array([r["p_fake"] for r in rows])
        pred = (p >= 0.5).astype(int)
        real_idx = y == 0
        fake_idx = y == 1
        entry = {
            "n": int(len(rows)),
            "n_real": int(real_idx.sum()),
            "n_fake": int(fake_idx.sum()),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "mean_p_fake": round(float(p.mean()), 4),
        }
        if real_idx.any():
            entry["real_acc"] = round(float((pred[real_idx] == 0).mean()), 4)
        if fake_idx.any():
            entry["fake_recall"] = round(float((pred[fake_idx] == 1).mean()), 4)
            entry["fake_precision"] = round(
                float(precision_score(y, pred, zero_division=0)), 4)
            entry["fake_f1"] = round(
                float(f1_score(y, pred, zero_division=0)), 4)
        out[g] = entry
    return out


def overall_metrics(records: List[dict]) -> dict:
    y = np.array([LABEL_MAP[r["label"]] for r in records])
    p = np.array([r["p_fake"] for r in records])
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(records)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "mean_p_fake_real": round(float(p[y == 0].mean()), 4),
        "mean_p_fake_fake": round(float(p[y == 1].mean()), 4),
    }
    return out


def score_split(model, loader, device) -> List[dict]:
    model.eval()
    records: List[dict] = []
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            for m, p in zip(meta, probs):
                records.append({
                    "path": m["path"],
                    "label": m["label"],
                    "p_fake": round(float(p), 6),
                    "source": m.get("source", "?"),
                    "generator": m.get("generator", "?"),
                    "architecture": m.get("architecture", "?"),
                    "corpus": m.get("corpus", "?"),
                    "width": m.get("width"),
                    "height": m.get("height"),
                    "sha256": m.get("sha256"),
                })
    return records


def write_scores(tag: str, split: str, records: List[dict]) -> Path:
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    dest = SCORE_DIR / f"swin_{tag}_{split}.jsonl"
    with dest.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return dest


def read_scores(tag: str, split: str) -> List[dict]:
    dest = SCORE_DIR / f"swin_{tag}_{split}.jsonl"
    if not dest.exists():
        return []
    out = []
    for line in dest.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def report(tag: str, split: str, records: List[dict], label: str) -> None:
    print(f"\n  {split} — {label}")
    print(f"    {len(records)} images scored")
    m = overall_metrics(records)
    print(f"    accuracy {m['accuracy']:.3f}  balanced {m['balanced_acc']:.3f}"
          f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}"
          f"  F1 {m['f1']:.3f}  AUC {m['auc']:.3f}")
    print(f"    real n={m['n_real']} mean P(FAKE) {m['mean_p_fake_real']:.3f}  "
          f"fake n={m['n_fake']} mean P(FAKE) {m['mean_p_fake_fake']:.3f}")
    for field in ("source", "generator"):
        table = per_group_metrics(records, field)
        print(f"\n    by {field}:")
        print(f"      {'group':<28}{'n':>6}{'acc':>8}{'real':>8}"
              f"{'fake_rec':>10}{'prec':>8}{'F1':>7}")
        for g, v in table.items():
            rl = v.get("real_acc", float("nan"))
            fr = v.get("fake_recall", float("nan"))
            pr = v.get("fake_precision", float("nan"))
            f1 = v.get("fake_f1", float("nan"))
            print(f"      {g:<28}{v['n']:>6}{v['accuracy']:>8.3f}"
                  f"{rl:>8.3f}{fr:>10.3f}{pr:>8.3f}{f1:>7.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=str,
                    default="datasets/prepared/swin_union/manifest.jsonl")
    ap.add_argument("--checkpoint", type=str,
                    default="models/swin_v2_tiny_512.pth")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--tag", type=str, default="swinv2_512")
    ap.add_argument("--splits", nargs="*", default=["test", "holdout"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--reuse-scores", action="store_true",
                    help="re-report from cached per-image scores")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.reuse_scores:
        ckpt_path = ROOT_DIR / args.checkpoint
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return 1
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        img_size = ckpt.get("img_size", args.img_size)
        print(f"  checkpoint {ckpt_path}  (epoch {ckpt.get('epoch')}, "
              f"val_acc {ckpt.get('val_acc'):.2f}%)")

    manifest = ROOT_DIR / args.manifest
    payload = {}
    for split in args.splits:
        records = read_scores(args.tag, split) if args.reuse_scores else []
        if not records and not args.reuse_scores:
            ds = ManifestDataset(manifest, split, _eval_transform(img_size),
                                 with_meta=True)
            loader = DataLoader(ds, batch_size=16, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True, collate_fn=_meta_collate)
            records = score_split(model, loader, device)
            dest = write_scores(args.tag, split, records)
            print(f"  [{split}] wrote {dest} ({len(records)} rows)")
        elif records:
            print(f"  [{split}] reusing cached scores ({len(records)} rows)")
        if records:
            report(args.tag, split, records, split)
            payload[split] = {
                "overall": overall_metrics(records),
                "by_source": per_group_metrics(records, "source"),
                "by_generator": per_group_metrics(records, "generator"),
                "by_corpus": per_group_metrics(records, "corpus"),
            }

    out_path = ROOT_DIR / "results" / "metrics" / f"swin_eval_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())