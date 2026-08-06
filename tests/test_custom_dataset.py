from pathlib import Path

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
