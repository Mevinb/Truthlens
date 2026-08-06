#!/usr/bin/env python3
"""Prepare labeled generator-specific images for TruthLens fine-tuning."""

import argparse
import random
from pathlib import Path

from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def split_files(files: list[Path], seed: int) -> dict[str, list[Path]]:
    files = files.copy()
    random.Random(seed).shuffle(files)
    train_end = int(len(files) * 0.8)
    val_end = train_end + int(len(files) * 0.1)
    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }


def prepare_dataset(fake_dir: Path, real_dir: Path, output_dir: Path, seed: int) -> None:
    """Prepare fake and real trees; fake subdirectories remain valid sources."""
    sources = {"fake": image_files(fake_dir), "real": image_files(real_dir)}
    if any(len(files) < 10 for files in sources.values()):
        raise ValueError("At least 10 readable image files are required per class.")

    for class_name, files in sources.items():
        for split, split_sources in split_files(files, seed).items():
            destination = output_dir / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(split_sources):
                try:
                    with Image.open(source) as image:
                        image.convert("RGB").save(
                            destination / f"{class_name}_{index:06d}.jpg",
                            quality=95,
                        )
                except Exception as exc:
                    print(f"Skipping unreadable image {source}: {exc}")

    print(f"Custom dataset prepared at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split images from multiple AI generators and real references. "
            "Put Gemini, DALL-E, Midjourney, Stable Diffusion, Flux, etc. "
            "under one fake directory."
        )
    )
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_gemini"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.fake_dir, args.real_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
