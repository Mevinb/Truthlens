import hashlib
import json

import numpy as np
from PIL import Image

from dataset.verify_swin_dataset import verify_manifest


def _write_image(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (12, 12, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(path, split, label, digest):
    return {"path": str(path), "split": split, "label": label,
            "source": "toy", "generator": "toy", "sha256": digest,
            "corpus": "toy"}


def test_full_verification_accepts_clean_manifest(tmp_path):
    rows = []
    seed = 0
    for split in ("train", "val"):
        for label in ("real", "fake"):
            image = tmp_path / split / label / "image.png"
            rows.append(_row(image, split, label, _write_image(image, seed)))
            seed += 1
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)

    report = verify_manifest(manifest, full=True, workers=2, progress_every=0)

    assert report["ok"] is True
    assert report["rows"] == 4
    assert report["missing_paths"] == 0
    assert report["hash_mismatches"] == []


def test_verification_rejects_content_leak_and_bad_hash(tmp_path):
    train_real = tmp_path / "train" / "real" / "real.png"
    train_fake = tmp_path / "train" / "fake" / "fake.png"
    val_real = tmp_path / "val" / "real" / "real.png"
    val_fake = tmp_path / "val" / "fake" / "fake.png"
    real_hash = _write_image(train_real, 1)
    fake_hash = _write_image(train_fake, 2)
    val_real.parent.mkdir(parents=True, exist_ok=True)
    val_real.write_bytes(train_real.read_bytes())
    val_fake_hash = _write_image(val_fake, 3)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        _row(train_real, "train", "real", real_hash),
        _row(train_fake, "train", "fake", fake_hash),
        _row(val_real, "val", "real", real_hash),
        _row(val_fake, "val", "fake", "not-the-real-hash"),
    ])

    report = verify_manifest(manifest, full=True, workers=1, progress_every=0)

    assert report["ok"] is False
    assert len(report["duplicate_hash_groups"]) == 1
    assert len(report["hash_mismatches"]) == 1
    assert val_fake_hash == report["hash_mismatches"][0]["actual"]


def test_quick_verification_rejects_missing_file(tmp_path):
    missing = tmp_path / "train" / "real" / "missing.png"
    rows = [_row(missing, "train", "real", "abc")]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)

    report = verify_manifest(manifest, full=False)

    assert report["ok"] is False
    assert report["missing_paths"] == 1


def test_verification_requires_new_corpus_size(tmp_path):
    rows = []
    seed = 0
    for split in ("train", "val"):
        for label in ("real", "fake"):
            image = tmp_path / split / label / "image.png"
            rows.append(_row(image, split, label, _write_image(image, seed)))
            seed += 1
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)

    report = verify_manifest(
        manifest, full=False, required_corpora={"grok_aurora": 500})

    assert report["ok"] is False
    assert any("grok_aurora" in error for error in report["errors"])
