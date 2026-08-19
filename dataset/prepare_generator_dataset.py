#!/usr/bin/env python3
"""Build a filesystem dataset from the generator images kept under
``datasets/raw/generator_sources``.

The checked-in source data is intentionally left untouched:

* ordinary image files are linked into the prepared dataset;
* Nano-Banana parquet rows are materialised once into a cache and then linked;
* a CSV manifest records the source and split of every sample.

This keeps the dataset in the directory layout expected by TruthLens while
avoiding a second copy of the GPT-Image images.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


def _split(items: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    n_val = int(len(shuffled) * 0.1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _parquet_images(parquet_dir: Path, cache_dir: Path) -> Iterable[Path]:
    """Yield cached image paths for Nano-Banana rows."""
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Nano-Banana parquet support requires pyarrow. Install pyarrow to prepare "
            "datasets/raw/generator_sources/nano-banana."
        ) from exc

    for shard in sorted(parquet_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch_index, batch in enumerate(
            parquet.iter_batches(columns=["image", "format"], batch_size=64)
        ):
            images = batch.column("image").to_pylist()
            formats = batch.column("format").to_pylist()
            for row_index, (image, image_format) in enumerate(zip(images, formats)):
                raw = image["bytes"]
                extension = (image_format or "png").lower().lstrip(".")
                if extension == "jpeg":
                    extension = "jpg"
                name = (
                    f"{shard.stem}_{batch_index:05d}_{row_index:03d}."
                    f"{_safe_name(extension)}"
                )
                cached = cache_dir / name
                if not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(raw)
                yield cached


def _source_images(fake_root: Path) -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {}
    gpt = fake_root / "gpt-image-2" / "images"
    if gpt.exists():
        sources["gpt-image-2"] = sorted(
            p for p in gpt.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    nano = fake_root / "nano-banana" / "data"
    if nano.exists():
        sources["nano-banana"] = list(_parquet_images(nano, fake_root / "_cache" / "nano-banana"))

    if not sources or not any(sources.values()):
        raise FileNotFoundError(
            f"No GPT-Image or Nano-Banana data found below {fake_root}. "
            "Expected datasets/raw/generator_sources/gpt-image-2/images and "
            "datasets/raw/generator_sources/nano-banana/data."
        )
    return sources


def _real_images(real_root: Path, split: str) -> list[Path]:
    split_dir = real_root / split / "real"
    if split_dir.exists():
        return sorted(
            p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(
        p for p in real_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_dataset(fake_root: Path, real_root: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
    """Prepare ``train/val/test/{real,fake}`` and return per-class counts."""
    fake_sources = _source_images(fake_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    counts = {split: 0 for split in SPLITS}

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "split", "label", "generator"))
        writer.writeheader()

        for generator, files in fake_sources.items():
            for split, split_files in _split(files, seed).items():
                for index, source in enumerate(split_files):
                    destination = output_dir / split / "fake" / generator / f"{index:06d}{source.suffix.lower()}"
                    _link(source, destination)
                    writer.writerow(
                        {"path": str(destination.relative_to(output_dir)), "split": split,
                         "label": "fake", "generator": generator}
                    )
                    counts[split] += 1

        for split in SPLITS:
            for index, source in enumerate(_real_images(real_root, split)):
                destination = output_dir / split / "real" / f"{index:06d}{source.suffix.lower()}"
                _link(source, destination)
                writer.writerow(
                    {"path": str(destination.relative_to(output_dir)), "split": split,
                     "label": "real", "generator": "real"}
                )
                counts[split] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake-root",
        type=Path,
        default=Path("datasets/raw/generator_sources"),
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("datasets/prepared/modern_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/prepared/gpt_nano"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = prepare_dataset(args.fake_root, args.real_root, args.output_dir, args.seed)
    print(f"Prepared {args.output_dir} with {counts} samples; manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
