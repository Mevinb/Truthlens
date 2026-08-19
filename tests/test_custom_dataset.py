from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image


def test_prepare_custom_dataset_creates_balanced_splits(tmp_path: Path):
    from dataset.prepare_custom_dataset import prepare_dataset

    fake_dir = tmp_path / "gemini"
    real_dir = tmp_path / "real"
    output_dir = tmp_path / "output"
    fake_dir.mkdir()
    real_dir.mkdir()

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for index in range(10):
        Image.fromarray(image).save(fake_dir / f"fake_{index}.png")
        Image.fromarray(image).save(real_dir / f"real_{index}.png")

    prepare_dataset(fake_dir, real_dir, output_dir, seed=42)

    for class_name in ("fake", "real"):
        assert len(list((output_dir / "train" / class_name).glob("*.jpg"))) == 8
        assert len(list((output_dir / "val" / class_name).glob("*.jpg"))) == 1
        assert len(list((output_dir / "test" / class_name).glob("*.jpg"))) == 1


def test_prepare_generator_dataset_extracts_nano_banana_parquet(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dataset.prepare_generator_dataset import prepare_dataset

    fake_root = tmp_path / "generators"
    nano_dir = fake_root / "nano-banana" / "data"
    real_dir = tmp_path / "real"
    output_dir = tmp_path / "output"
    nano_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (real_dir / split / "real").mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)

    # Build a tiny parquet shard with the same nested image schema used upstream.
    rows = 10
    buffer = []
    for _ in range(rows):
        bio = BytesIO()
        Image.fromarray(image).save(bio, format="PNG")
        buffer.append(bio.getvalue())

    table = pa.table(
        {
            "image": pa.array([{"bytes": data} for data in buffer], type=pa.struct([("bytes", pa.binary())])),
            "format": pa.array(["png"] * rows),
        }
    )
    pq.write_table(table, nano_dir / "train-00000-of-00001.parquet")

    for split in ("train", "val", "test"):
        for index in range(10):
            Image.fromarray(image).save(real_dir / split / "real" / f"real_{index}.png")

    prepare_dataset(fake_root, real_dir, output_dir, seed=42)

    assert len(list((output_dir / "train" / "fake" / "nano-banana").glob("*.png"))) == 8
    assert len(list((output_dir / "val" / "fake" / "nano-banana").glob("*.png"))) == 1
    assert len(list((output_dir / "test" / "fake" / "nano-banana").glob("*.png"))) == 1
    assert len(list((output_dir / "train" / "real").glob("*.png"))) == 10
    assert len(list((output_dir / "val" / "real").glob("*.png"))) == 10
    assert len(list((output_dir / "test" / "real").glob("*.png"))) == 10


def test_prepare_legacy_archive_keeps_splits(tmp_path: Path):
    from dataset.prepare_legacy_archive import prepare_dataset

    source_root = tmp_path / "legacy_archive"
    archive = source_root / "Data Set 1" / "Data Set 1"
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    for split in ("train", "validation", "test"):
        for label in ("real", "fake"):
            split_dir = archive / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(split_dir / f"{split}_{label}.png")

    output_dir = tmp_path / "prepared"
    counts = prepare_dataset(source_root, output_dir)

    assert counts == {"train": 2, "val": 2, "test": 2}
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            assert len(list((output_dir / split / label).glob("*.png"))) == 1
