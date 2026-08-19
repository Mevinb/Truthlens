#!/usr/bin/env python3
"""Download a balanced real/AI image corpus while retaining source metadata."""

import argparse
import csv
import io
import json
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm


DATASET_ID = "OwensLab/CommunityForensics-Small"
ARCHITECTURES = ("real", "GAN", "LatDiff", "PixDiff", "other")
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def normalize_architecture(value: object, label: int) -> str | None:
    """Return one of the supported source groups for a dataset row."""
    if label == 0:
        return "real"
    text = str(value).strip().lower()
    aliases = {
        "gan": "GAN",
        "latdiff": "LatDiff",
        "latent diffusion": "LatDiff",
        "pixdiff": "PixDiff",
        "pixel diffusion": "PixDiff",
        "other": "other",
    }
    return aliases.get(text)


def decode_image(value: object) -> Image.Image:
    """Decode the image representations emitted by Hugging Face datasets."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value))
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def choose_split(index: int, target: int) -> str:
    train_end = int(target * SPLIT_RATIOS[0])
    val_end = train_end + int(target * SPLIT_RATIOS[1])
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def existing_state(manifest_path: Path) -> tuple[Counter, set[str]]:
    counts: Counter = Counter()
    source_ids: set[str] = set()
    if not manifest_path.exists():
        return counts, source_ids
    with manifest_path.open(newline="", encoding="utf-8") as manifest:
        for row in csv.DictReader(manifest):
            counts[row["architecture"]] += 1
            if row.get("source_id"):
                source_ids.add(row["source_id"])
    return counts, source_ids


def check_disk_space(output_dir: Path, remaining: int, max_bytes: int) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    estimate = remaining * max_bytes
    reserve = 2 * 1024**3
    if estimate + reserve > free_bytes:
        needed_gib = (estimate + reserve) / 1024**3
        free_gib = free_bytes / 1024**3
        raise RuntimeError(
            f"Insufficient disk space: need up to {needed_gib:.1f} GiB "
            f"including reserve, but only {free_gib:.1f} GiB is free"
        )


def download(
    output_dir: Path,
    per_architecture: int,
    image_size: int,
    quality: int,
    seed: int,
) -> Counter:
    manifest_path = output_dir / "manifest.csv"
    counts, source_ids = existing_state(manifest_path)
    targets = {name: per_architecture for name in ARCHITECTURES}
    targets["real"] = per_architecture * (len(ARCHITECTURES) - 1)
    remaining = sum(max(0, targets[name] - counts[name]) for name in ARCHITECTURES)
    check_disk_space(output_dir, remaining, 100_000)

    for split in SPLITS:
        for class_name in ("real", "fake"):
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    write_header = not manifest_path.exists()
    errors = 0

    with manifest_path.open("a", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=(
                "path",
                "label",
                "architecture",
                "model_name",
                "real_source",
                "source_id",
            ),
        )
        if write_header:
            writer.writeheader()

        progress = tqdm(total=remaining, desc="Saving balanced architecture dataset", unit="image")
        for row in dataset:
            label = int(row["label"])
            architecture = normalize_architecture(row.get("architecture"), label)
            source_id = f"{row.get('model_name', '')}/{row.get('image_name', '')}"
            if (
                architecture is None
                or counts[architecture] >= targets[architecture]
                or source_id in source_ids
                or row.get("nsfw_flag", False)
            ):
                continue

            index = counts[architecture]
            split = choose_split(index, targets[architecture])
            class_name = "real" if label == 0 else "fake"
            relative_path = Path(split) / class_name / f"{architecture.lower()}_{index:06d}.jpg"
            try:
                with decode_image(row.get("image_data", row.get("image"))) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                    image.save(output_dir / relative_path, "JPEG", quality=quality, optimize=True)
            except (OSError, ValueError, TypeError):
                errors += 1
                continue

            writer.writerow(
                {
                    "path": relative_path.as_posix(),
                    "label": class_name,
                    "architecture": architecture,
                    "model_name": row.get("model_name", ""),
                    "real_source": row.get("real_source", ""),
                    "source_id": source_id,
                }
            )
            manifest.flush()
            counts[architecture] += 1
            source_ids.add(source_id)
            progress.update(1)
            if all(counts[name] >= targets[name] for name in ARCHITECTURES):
                break
        progress.close()

    summary = {
        "dataset": DATASET_ID,
        "target_per_architecture": per_architecture,
        "real_target": targets["real"],
        "image_size": image_size,
        "counts": {name: counts[name] for name in ARCHITECTURES},
        "decode_errors": errors,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream balanced real, GAN, latent diffusion, pixel diffusion, and other images."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/prepared/architectures"))
    parser.add_argument("--per-architecture", type=int, default=20_000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_architecture < 1 or args.image_size < 32 or not 1 <= args.quality <= 95:
        parser.error("targets must be positive, image size >= 32, and quality between 1 and 95")
    return args


def main() -> None:
    args = parse_args()
    counts = download(
        args.output_dir,
        args.per_architecture,
        args.image_size,
        args.quality,
        args.seed,
    )
    print("Downloaded counts:")
    for architecture in ARCHITECTURES:
        print(f"  {architecture:8s}: {counts[architecture]:,}")


if __name__ == "__main__":
    main()
