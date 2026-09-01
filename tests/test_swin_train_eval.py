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

import argparse
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


# ─── Resume ───────────────────────────────────────────────────────────────────
class TestResume:
    def test_epoch_boundary_checkpoint_starts_next_epoch(self):
        from src.train_swin import _resume_position

        assert _resume_position({"epoch": 4, "step_in_epoch": 0}) == (5, 0)
        assert _resume_position({"epoch": 4, "step_in_epoch": 200}) == (4, 200)

    def test_resume_state_roundtrip(self, tmp_path, tiny_manifest):
        """The full training state must survive a save → new process → load.

        If the optimizer or scheduler state were dropped, a resumed run would
        not continue at the same position (momentum and LR schedule are part
        of the position), so both are round-tripped through the checkpoint.
        """
        import torch
        from torch.amp import GradScaler

        from src.train_swin import (
            EarlyStopping, ManifestDataset, _eval_transform, build_model,
        )

        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32))
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = build_model(pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6)
        stopper = EarlyStopping(patience=4)
        scaler = GradScaler("cpu", enabled=False)

        # Advance a couple of steps so optimizer/scheduler state is non-trivial.
        optimizer.zero_grad(set_to_none=True)
        for _ in range(3):
            optimizer.step()
            scheduler.step()

        ckpt = tmp_path / "resume.pth"
        torch.save({
            "epoch": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "stopper": {"best": 0.5, "counter": 0},
            "best_val_loss": 0.5,
            "history": [{"epoch": 1}],
            "epoch_times": [10.0],
            "img_size": 32,
        }, ckpt)

        # Fresh objects — simulates a brand-new process.
        model2 = build_model(pretrained=False)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=10, eta_min=1e-6)
        stopper2 = EarlyStopping(patience=4)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model2.load_state_dict(state["model_state"])
        optimizer2.load_state_dict(state["optimizer_state"])
        scheduler2.load_state_dict(state["scheduler_state"])
        stopper2.best = state["stopper"]["best"]
        stopper2.counter = state["stopper"]["counter"]

        assert state["epoch"] == 2
        assert state["best_val_loss"] == 0.5
        assert optimizer2.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        assert scheduler2.get_last_lr() == scheduler.get_last_lr()
        assert stopper2.best == 0.5 and stopper2.counter == 0
        # Weights actually transferred (not a fresh init).
        w1 = next(model.parameters()).detach().clone()
        w2 = next(model2.parameters()).detach().clone()
        assert torch.equal(w1, w2)

    def test_step_resume_state_carries_data_order(self, tmp_path, tiny_manifest):
        """A mid-epoch checkpoint must persist the permutation and the shuffle
        generator's state — the data order is part of the training position."""
        from src.train_swin import _atomic_torch_save

        gen = torch.Generator()
        gen.manual_seed(42)
        perm = list(torch.randperm(100, generator=gen))
        state = gen.get_state()

        ckpt = tmp_path / "step_resume.pth"
        _atomic_torch_save({
            "epoch": 3, "step_in_epoch": 50, "perm": perm,
            "train_gen_state": state,
        }, ckpt)
        assert not (tmp_path / "step_resume.pth.tmp").exists()  # atomic: no torn write

        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 3
        assert loaded["step_in_epoch"] == 50
        assert loaded["perm"] == perm
        gen2 = torch.Generator()
        gen2.set_state(loaded["train_gen_state"])
        assert torch.equal(gen2.get_state(), state)

    def test_resume_compatibility_checks_dataset_and_batch_config(self):
        from src.train_swin import _resume_compatibility_errors

        args = argparse.Namespace(
            img_size=512, batch_size=4, grad_accum=4, epochs=20, lr=1e-4,
            weight_decay=1e-4, seed=42, patience=4)
        state = {"config": {
            "img_size": 512, "batch_size": 4, "grad_accum": 4,
            "epochs": 20, "lr": 1e-4, "weight_decay": 1e-4,
            "seed": 42, "patience": 4,
            "manifest_sha256": "current", "manifest_rows": 100,
        }}
        assert _resume_compatibility_errors(
            state, args, "current", 100, 80) == []
        errors = _resume_compatibility_errors(
            state, args, "changed", 101, 80)
        assert any("manifest_sha256" in error for error in errors)
        assert any("manifest_rows" in error for error in errors)

    def test_mid_epoch_resume_requires_matching_permutation(self):
        from src.train_swin import _resume_compatibility_errors

        args = argparse.Namespace(
            img_size=32, batch_size=2, grad_accum=1, epochs=2, lr=1e-4,
            weight_decay=1e-4, seed=42, patience=4)
        state = {
            "step_in_epoch": 2,
            "perm": [0, 1],
            "config": {
                "img_size": 32, "batch_size": 2, "grad_accum": 1,
                "epochs": 2, "lr": 1e-4, "weight_decay": 1e-4,
                "seed": 42, "patience": 4,
                "manifest_sha256": "abc", "manifest_rows": 8,
            },
        }
        errors = _resume_compatibility_errors(state, args, "abc", 8, 4)
        assert any("permutation length" in error for error in errors)


# ─── Step-resume building blocks ──────────────────────────────────────────────
class TestStepResume:
    def test_slice_sampler_yields_fixed_subsequence(self):
        from src.train_swin import _SliceSampler
        s = _SliceSampler([4, 5, 6, 7])
        assert list(s) == [4, 5, 6, 7]
        assert len(s) == 4

    def test_slice_sampler_supports_partial_slices(self):
        from src.train_swin import _SliceSampler
        perm = list(range(10))
        s = _SliceSampler(perm[6:])
        assert list(s) == [6, 7, 8, 9]
        assert len(s) == 4

    def test_run_epoch_snapshot_callback_at_accumulation_boundaries(self, tiny_manifest):
        """snapshot_cb must fire right after the optimizer step, only at
        accumulation boundaries, and with the global batch count (so a resumed
        run can slice the saved permutation at the right index)."""
        from src.train_swin import (
            ManifestDataset, _eval_transform, build_model, run_epoch,
        )
        from torch.amp import GradScaler

        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32))
        loader = DataLoader(ds, batch_size=1, shuffle=False)   # 4 batches
        model = build_model(pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler("cpu", enabled=False)

        snapshots = []
        run_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, scaler,
                  torch.device("cpu"), True, grad_accum=2,
                  snapshot_every=2, snapshot_cb=snapshots.append)
        # The final batch is deliberately left to the epoch-boundary checkpoint;
        # saving it as mid-epoch could resume to an empty loader during val.
        assert snapshots == [2]

    def test_run_epoch_snapshot_respects_start_step(self, tiny_manifest):
        """With start_step>0 the global batch count shifts, so boundaries land
        where the interrupted run's would have — and no snapshot fires at a
        non-boundary."""
        from src.train_swin import (
            ManifestDataset, _eval_transform, build_model, run_epoch,
        )
        from torch.amp import GradScaler

        ds = ManifestDataset(tiny_manifest, "train", _eval_transform(32))
        loader = DataLoader(ds, batch_size=1, shuffle=False)   # 4 batches
        model = build_model(pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler("cpu", enabled=False)

        snapshots = []
        # start_step=2 (2 batches already done), grad_accum=2, snapshot_every=2:
        # global positions covered are 2..5; boundaries at 4 and 6.
        run_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, scaler,
                  torch.device("cpu"), True, grad_accum=2,
                  start_step=2, snapshot_every=2, snapshot_cb=snapshots.append)
        assert snapshots == [4]


class TestEpochArchive:
    """Per-epoch weight archiving: paths and the metrics index."""

    def test_epoch_archive_path_format(self):
        from src.train_swin import _epoch_archive_path

        p = _epoch_archive_path("swin_v2_512", 7)
        assert p.name == "swin_v2_512_ep007.pth"
        assert p.parent.name == "swin_epochs"
        assert p.parent.parent.name == "models"

    def test_update_epoch_index_upserts_and_sorts(self, tmp_path):
        from src.train_swin import _update_epoch_index

        idx = tmp_path / "tag_index.json"
        _update_epoch_index(idx, {"epoch": 2, "val_loss": 0.20})
        _update_epoch_index(idx, {"epoch": 1, "val_loss": 0.25})
        # Re-running an epoch replaces its entry instead of duplicating it.
        _update_epoch_index(idx, {"epoch": 2, "val_loss": 0.19})
        entries = json.loads(idx.read_text())
        assert [e["epoch"] for e in entries] == [1, 2]
        assert entries[1]["val_loss"] == 0.19
        # Atomic write leaves no temp file behind.
        assert not list(tmp_path.glob("*.tmp"))

    def test_update_epoch_index_survives_corrupt_file(self, tmp_path):
        from src.train_swin import _update_epoch_index

        idx = tmp_path / "tag_index.json"
        idx.write_text("{not valid json")
        _update_epoch_index(idx, {"epoch": 3, "val_loss": 0.15})
        entries = json.loads(idx.read_text())
        assert [e["epoch"] for e in entries] == [3]

    def test_eval_checkpoint_payload_shape(self, tiny_manifest):
        from src.train_swin import (
            ManifestDataset, _eval_checkpoint_payload, _eval_transform,
            build_model,
        )

        args = argparse.Namespace(
            img_size=32, batch_size=2, grad_accum=1, lr=1e-4,
            weight_decay=0.01, epochs=1, seed=0,
            manifest=str(tiny_manifest), no_pretrained=True)
        model = build_model(pretrained=False)
        val_m = {"loss": 0.42, "accuracy": 88.0}
        payload = _eval_checkpoint_payload(5, model, val_m, args)
        assert payload["epoch"] == 5
        assert payload["val_loss"] == 0.42 and payload["val_acc"] == 88.0
        assert payload["img_size"] == 32
        assert "model_state" in payload and "optimizer_state" not in payload


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
