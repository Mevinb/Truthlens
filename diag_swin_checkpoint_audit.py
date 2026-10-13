#!/usr/bin/env python3
"""Validate every locally available SwinV2 checkpoint.

This is an integrity and inference-compatibility audit, not an accuracy
benchmark: it discovers all ``models/**/swin*.pth`` files, loads every
checkpoint on CPU, strictly restores its model weights, and performs one
512-pixel forward pass.  It also fingerprints the model state so aliases and
training-resume snapshots cannot be mistaken for independent models.

Run with:

    .venv/bin/python diag_swin_checkpoint_audit.py

The JSON report is written to ``results/metrics/swin_checkpoint_audit.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_swin import build_model


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Return a deterministic digest of names, shapes, dtypes, and weights."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name!r} is not a tensor")
        cpu = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_paths() -> list[Path]:
    return sorted((ROOT / "models").glob("**/swin*.pth"))


def audit_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    started = time.perf_counter()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint payload is {type(raw).__name__}, not a mapping")
    state = raw.get("model_state", raw.get("state_dict"))
    if not isinstance(state, dict):
        raise KeyError("checkpoint has no model_state/state_dict mapping")

    # strict=True verifies architecture as well as the serialisation format.
    model.load_state_dict(state, strict=True)
    model.eval()
    img_size = int(raw.get("img_size", 512))
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, img_size, img_size))
        probs = torch.softmax(logits, dim=1)
    if logits.shape != (1, 2) or not torch.isfinite(probs).all():
        raise ValueError(f"invalid output: shape={tuple(logits.shape)} finite={torch.isfinite(probs).all()}")

    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "epoch": raw.get("epoch"),
        "img_size": img_size,
        "model_type": raw.get("model_type"),
        "is_resume_snapshot": "optimizer_state" in raw,
        "state_sha256": state_fingerprint(state),
        "state_tensors": len(state),
        "output_shape": list(logits.shape),
        "zero_input_p_fake": round(float(probs[0, 1]), 6),
        "load_and_forward_seconds": round(time.perf_counter() - started, 3),
        "status": "passed",
    }


def main() -> int:
    paths = checkpoint_paths()
    if not paths:
        print("No Swin checkpoints found.")
        return 1

    print(f"Auditing {len(paths)} Swin checkpoint file(s) on CPU...")
    model = build_model(pretrained=False).cpu()
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        try:
            row = audit_checkpoint(path, model)
            print(f"  [{index:>2}/{len(paths)}] PASS {path.name}")
        except Exception as exc:  # report all bad files, not only the first one
            row = {
                "checkpoint": str(path.relative_to(ROOT)),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  [{index:>2}/{len(paths)}] FAIL {path.name}: {row['error']}")
        rows.append(row)

    aliases: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["status"] == "passed":
            aliases[row["state_sha256"]].append(row["checkpoint"])
    for row in rows:
        if row["status"] == "passed":
            row["identical_weights_to"] = aliases[row["state_sha256"]]

    payload = {
        "purpose": "Checkpoint integrity and 512px inference compatibility audit",
        "files_discovered": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unique_weight_states": len(aliases),
        "identical_weight_groups": [group for group in aliases.values() if len(group) > 1],
        "checkpoints": rows,
    }
    out = ROOT / "results" / "metrics" / "swin_checkpoint_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0 if payload["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
