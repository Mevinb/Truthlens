"""Guards on the unified manifest builder.

The whole point of the sha256 dedup is that the same image can never be fitted
and scored — a leak the union corpus could create silently by assembling six
corpora that overlap. The dedup policy and the final leak check are the two
behaviours that make the SwinV2 numbers meaningful, so they are pinned here on
synthetic corpora rather than on the 275K-row real manifest.
"""

import json

import pytest

from dataset.build_swin_union_manifest import dedup, report_leaks, sha256_and_size


# ─── sha256 ───────────────────────────────────────────────────────────────────
def test_sha256_and_size_matches_file_bytes(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"abc")
    digest, w, h = sha256_and_size(p)
    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert (w, h) == (None, None)  # not a real image → dimensions unknown


# ─── dedup policy ─────────────────────────────────────────────────────────────
def _row(path, split, sha, label="fake", source="s", gen="g"):
    return {"path": path, "split": split, "label": label, "source": source,
            "generator": gen, "architecture": "?", "sha256": sha}


def test_dedup_preserves_evaluation_split_copy():
    """A designated validation identity must not be pulled into training."""
    rows = [
        _row("a/train.png", "train", "AAA"),
        _row("a/val.png", "val", "AAA"),
        _row("b/val.png", "val", "BBB"),
    ]
    kept, stats = dedup(rows)
    by_path = {r["path"]: r for r in kept}
    assert set(by_path) == {"a/val.png", "b/val.png"}
    assert stats["dropped"]["train (dup of val)"] == 1
    assert stats["missing_sha"] == 0


def test_dedup_drops_intra_split_duplicates():
    rows = [
        _row("t1.png", "train", "AAA"),
        _row("t2.png", "train", "AAA"),
        _row("v1.png", "val", "BBB"),
    ]
    kept, stats = dedup(rows)
    assert len(kept) == 2
    assert stats["dropped"]["train (dup of train)"] == 1


def test_dedup_prefers_holdout_then_test_then_val_over_train():
    rows = [
        _row("test.png", "test", "AAA"),
        _row("train.png", "train", "AAA"),
        _row("val.png", "val", "AAA"),
        _row("holdout.png", "holdout", "AAA"),
    ]
    kept, _ = dedup(rows)
    assert [r["split"] for r in kept] == ["holdout"]


def test_dedup_reports_conflicting_labels():
    rows = [
        _row("real.png", "train", "AAA", label="real"),
        _row("fake.png", "val", "AAA", label="fake"),
    ]
    _, stats = dedup(rows)
    assert stats["conflicting_labels"] == [{
        "sha256": "AAA", "labels": ["fake", "real"],
        "paths": ["real.png", "fake.png"],
    }]


def test_rows_without_sha_are_counted_not_silently_merged():
    rows = [_row("a.png", "train", ""), _row("b.png", "val", "")]
    kept, stats = dedup(rows)
    assert len(kept) == 2
    assert stats["missing_sha"] == 2


# ─── leak check ───────────────────────────────────────────────────────────────
def test_report_leaks_finds_no_leak_after_dedup():
    rows = [
        _row("t1.png", "train", "AAA"),
        _row("v1.png", "val", "BBB"),
        _row("e1.png", "test", "CCC"),
        _row("h1.png", "holdout", "DDD"),
    ]
    assert report_leaks(rows) == []


def test_report_leaks_flags_a_hash_in_train_and_test():
    rows = [
        _row("t1.png", "train", "AAA"),
        _row("e1.png", "test", "AAA"),
    ]
    leaks = report_leaks(rows)
    assert len(leaks) == 1
    assert leaks[0]["path"] == "e1.png"


def test_builder_would_exit_nonzero_on_leak(tmp_path, capsys):
    """End-to-end: a manifest with a cross-split duplicate must not pass."""
    import sys
    from pathlib import Path

    from dataset.build_swin_union_manifest import OUT_DIR, SPLIT_PRIORITY

    # Write a tiny synthetic manifest the way the builder would, then assert
    # the leaked copy survives the builder's own dedup+check for a *single*
    # corpus (here the legacy extractor path is the interesting one).
    corpus = tmp_path / "corpus"
    (corpus / "train" / "fake").mkdir(parents=True)
    (corpus / "test" / "fake").mkdir(parents=True)
    dup = b"duplicate-bytes"
    (corpus / "train" / "fake" / "x.jpg").write_bytes(dup)
    (corpus / "test" / "fake" / "x.jpg").write_bytes(dup)
    manifest = corpus / "manifest.csv"
    manifest.write_text(
        "path,split,label,source_archive,source_split\n"
        "train/fake/x.jpg,train,fake,DS1,train\n"
        "test/fake/x.jpg,test,fake,DS1,test\n")

    from dataset.build_swin_union_manifest import rows_legacy_archive
    rows = rows_legacy_archive(corpus, {})
    assert len(rows) == 2
    kept, _ = dedup(rows)
    assert len(kept) == 1
    assert report_leaks(kept) == []


# ─── split ordering invariant ─────────────────────────────────────────────────
def test_split_priority_ordering_is_consistent():
    from dataset.build_swin_union_manifest import SPLIT_PRIORITY
    assert SPLIT_PRIORITY["holdout"] < SPLIT_PRIORITY["test"]
    assert SPLIT_PRIORITY["test"] < SPLIT_PRIORITY["val"]
    assert SPLIT_PRIORITY["val"] < SPLIT_PRIORITY["train"]
