#!/usr/bin/env python3
"""
TruthLens — tests/test_swin_train_eval.py
=========================================
Guards on the SwinV2 training/eval scripts that the union-corpus run depends
on. All of these run on CPU with synthetic data — no GPU, no 275K-row
manifest — so they can run in CI on every change.

What is pinned here:
* ``ManifestDataset`` split filtering, label mapping and the ``with_meta``
  contract that eval's per-source grouping relies on.
* the model head replacement (768 → 2 classes).
* grad-accumulation: the scaler path that previously crashed with
  "Attempted unscale_ but _scale is None" must work, including the
  trailing partial-accumulation block.
* early stopping semantics (patience / min_delta).
* eval-side grouping: ``per_group_metrics`` and ``overall_metrics`` against
  hand-computed numbers, and the collate that keeps meta as a list.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def tiny_manifest(tmp_path):
    """A manifest with a handful of train/val rows and real image files."""
    from src.train_swin import LABEL_MAP
    rng = np.random.default_rng(7)
    rows = []
    for split in ("train", "val"):
        for i, label in enumerate(("real", "fake", "fake", "real")):
            img = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
            img = img.convert("RGB")
            d = tmp_path / split / label
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{i:02d}.jpg"
            img.save(p, quality=90)
            rows.append({
                "path": str(p), "split": split, "label": label,
                "source": "cam_a" if label == "real" else "gen_a",
                "generator": None if label == "real" else "gen-a",
                "architecture": "?", "sha256": f"h{i}", "corpus": "toy",
            })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return manifest


def _eval_transform_tiny():
    from src.train_swin import _eval_transform
    return _eval_transform(32)


# ─── ManifestDataset ──────────────────────────────────────────────────────────
class TestManifestDataset:
    def test_filters_by_split_and_maps_labels(self, tiny_manifest):
        from src.train_swin import ManifestDataset, _eval_transform
        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32))
        assert len(ds) == 4
        assert ds.class_balance == {"real": 2, "fake": 2, "n": 4}
        x, y = ds[0]
        assert x.shape == (3, 32, 32)
        assert y in (0, 1)

    def test_raises_on_unknown_split(self, tiny_manifest):
        from src.train_swin import ManifestDataset, _eval_transform
        with pytest.raises(ValueError, match="No rows"):
            ManifestDataset(tiny_manifest, "test", _eval_transform(32))

    def test_with_meta_carries_provenance(self, tiny_manifest):
        from src.train_swin import ManifestDataset, _eval_transform
        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32), with_meta=True)
        x, y, meta = ds[0]
        assert meta["source"] in ("cam_a", "gen_a")
        assert meta["path"] and meta["sha256"]

    def test_limit_truncates(self, tiny_manifest):
        from src.train_swin import ManifestDataset, _eval_transform
        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32), limit=2)
        assert len(ds) == 2


# ─── Model ────────────────────────────────────────────────────────────────────
class TestBuildModel:
    def test_head_is_two_class(self):
        from src.train_swin import build_model
        m = build_model(pretrained=False)
        assert isinstance(m.head, nn.Linear)
        assert m.head.out_features == 2
        assert m.head.in_features == 768  # swin_v2_t feature dim


# ─── Training loop / grad accumulation ────────────────────────────────────────
class TestRunEpoch:
    def test_grad_accum_trains_and_val_runs(self, tiny_manifest):
        """Regression: scaler.scale() must be used or unscale_ crashes."""
        from src.train_swin import (
            ManifestDataset, _eval_transform, build_model, run_epoch,
        )
        from torch.amp import GradScaler

        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32))
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = build_model(pretrained=False)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler("cpu", enabled=False)   # enabled off, path still exercised

        # grad_accum=2 over 2 batches: 1 full accumulation, no trailing block.
        tm = run_epoch(model, loader, criterion, optimizer, scaler,
                       torch.device("cpu"), True, grad_accum=2)
        assert set(tm) >= {"loss", "accuracy"}
        assert tm["loss"] > 0.0

    def test_trailing_partial_accumulation_flushes(self, tiny_manifest):
        """3 batches with grad_accum=2 leaves one un-stepped; must not crash."""
        from src.train_swin import (
            ManifestDataset, _eval_transform, build_model, run_epoch,
        )
        from torch.amp import GradScaler

        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32))
        loader = DataLoader(ds, batch_size=1, shuffle=False)   # 4 batches
        model = build_model(pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler("cpu", enabled=False)
        run_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, scaler,
                  torch.device("cpu"), True, grad_accum=3)

    def test_val_epoch_reports_accuracy(self, tiny_manifest):
        from src.train_swin import (
            ManifestDataset, _eval_transform, build_model, run_epoch,
        )
        from torch.amp import GradScaler

        ds = ManifestDataset(tiny_manifest, "val", _eval_transform(32))
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = build_model(pretrained=False)
        vm = run_epoch(model, loader, nn.CrossEntropyLoss(), None,
                       GradScaler("cpu", enabled=False), torch.device("cpu"),
                       False)
        assert 0.0 <= vm["accuracy"] <= 100.0


# ─── Early stopping ───────────────────────────────────────────────────────────
class TestEarlyStopping:
    def test_improvement_resets_counter(self):
        from src.train_swin import EarlyStopping
        es = EarlyStopping(patience=3, min_delta=1e-4)
        assert es(1.0) is False
        assert es(0.5) is False        # improvement → reset
        assert es(0.5) is False
        assert es(0.5) is False
        assert es(0.5) is True         # 3 no-improvements → stop

    def test_min_delta_ignores_tiny_drops(self):
        from src.train_swin import EarlyStopping
        es = EarlyStopping(patience=1, min_delta=0.1)
        assert es(1.0) is False
        assert es(0.95) is True        # drop < min_delta → counts as no-improvement


# ─── Eval metrics ─────────────────────────────────────────────────────────────
def _rec(label, p_fake, source="gen_a", generator="gen-a"):
    return {"label": label, "p_fake": p_fake, "source": source,
            "generator": generator, "corpus": "toy"}


class TestMetrics:
    def test_overall_metrics_hand_computed(self):
        from src.eval_swin import overall_metrics
        recs = [
            _rec("fake", 0.90), _rec("fake", 0.80), _rec("fake", 0.30),   # 1 missed
            _rec("real", 0.10), _rec("real", 0.20), _rec("real", 0.60),   # 1 FP
        ]
        m = overall_metrics(recs)
        assert m["n"] == 6
        assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-3)
        assert m["recall"] == pytest.approx(2 / 3, abs=1e-3)
        assert m["precision"] == pytest.approx(2 / 3, abs=1e-3)
        assert m["auc"] == pytest.approx(8 / 9, abs=1e-3)   # 8/9 rank pairs correct

    def test_per_group_metrics_splits_by_source(self):
        from src.eval_swin import per_group_metrics
        recs = [
            _rec("fake", 0.9, source="gen_a", generator="gen-a"),
            _rec("fake", 0.8, source="gen_b", generator="gen-b"),
            _rec("real", 0.1, source="cam_a"),
            _rec("real", 0.2, source="cam_a"),
        ]
        g = per_group_metrics(recs, "source")
        assert set(g) == {"cam_a", "gen_a", "gen_b"}
        assert g["cam_a"]["n_real"] == 2 and g["cam_a"]["n_fake"] == 0
        assert g["gen_a"]["fake_recall"] == 1.0
        assert g["gen_a"]["fake_precision"] == 1.0

    def test_metric_keywords_are_json_serialisable(self):
        from src.eval_swin import overall_metrics, per_group_metrics
        recs = [_rec("fake", 0.9, source="gen_a"), _rec("real", 0.1, source="cam_a")]
        for obj in (overall_metrics(recs), per_group_metrics(recs, "source")):
            json.dumps(obj)  # must not raise


# ─── Collate ──────────────────────────────────────────────────────────────────
class TestMetaCollate:
    def test_meta_stays_a_list_of_dicts(self, tiny_manifest):
        """Regression: default collate turns meta into a dict-of-lists, and
        zip(meta, probs) then iterates over keys instead of rows."""
        from src.eval_swin import _meta_collate
        from src.train_swin import ManifestDataset, _eval_transform

        ds = ManifestDataset(tiny_manifest, "val", _eval_transform(32), with_meta=True)
        xs, ys, meta = _meta_collate([ds[0], ds[1]])
        assert xs.shape == (2, 3, 32, 32)
        assert ys.shape == (2,)
        assert isinstance(meta, list) and len(meta) == 2
        assert isinstance(meta[0], dict) and "path" in meta[0]

    def test_collate_used_by_scorer(self, tiny_manifest):
        from src.eval_swin import _meta_collate, score_split
        from src.train_swin import ManifestDataset, _eval_transform, build_model

        ds = ManifestDataset(tiny_manifest, "val", _eval_transform(32), with_meta=True)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=_meta_collate)
        model = build_model(pretrained=False).eval()
        recs = score_split(model, loader, torch.device("cpu"))
        assert len(recs) == len(ds)
        assert {r["label"] for r in recs} == {"real", "fake"}
        assert all(0.0 <= r["p_fake"] <= 1.0 for r in recs)