#!/usr/bin/env python3
"""Build a resumable, deduplicated multi-source TruthLens corpus.

The output is compatible with the existing image-folder training pipeline and
also includes ``manifest.csv`` and ``sources.json`` for auditability. Images
are never loaded into memory as a complete dataset; Hugging Face sources are
consumed as streams.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from datasets import DownloadConfig, load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


SPLITS = ("train", "val", "test")
ARCHITECTURES = ("GAN", "LatDiff", "PixDiff", "other")
MANIFEST_FIELDS = ("path", "label", "source", "source_id", "architecture", "width", "height", "sha256")


@dataclass(frozen=True)
class Source:
    name: str
    dataset_id: str
    license: str
    url: str


SOURCES = (
    Source("cifake", "dragonintelligence/CIFAKE-image-dataset", "See upstream dataset card", "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset"),
    Source("highres_mixed", "Hemg/AI-Generated-vs-Real-Images-Datasets", "See upstream dataset card", "https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets"),
    Source("modern_mixed", "date3k2/raw_real_fake_images", "See upstream dataset card", "https://huggingface.co/datasets/date3k2/raw_real_fake_images"),
    Source("community_forensics", "OwensLab/CommunityForensics-Small", "CC-BY-NC-SA-4.0", "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small"),
)


def decode_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes):
        return Image.open(__import__("io").BytesIO(value))
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(__import__("io").BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def split_for(index: int, target: int) -> str:
    if index < int(target * 0.8):
        return "train"
    if index < int(target * 0.9):
        return "val"
    return "test"


def image_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def architecture_for(row: dict, label: str, source: str) -> str:
    if label == "real":
        return "real"
    if source == "community_forensics":
        value = str(row.get("architecture", "other")).strip().lower()
        return {"gan": "GAN", "latdiff": "LatDiff", "pixdiff": "PixDiff"}.get(value, "other")
    return "other"


def existing_hashes(manifest_path: Path) -> tuple[set[str], Counter]:
    hashes: set[str] = set()
    counts: Counter = Counter()
    if not manifest_path.exists():
        return hashes, counts
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            hashes.add(row.get("sha256", ""))
            counts[(row.get("source", ""), row.get("architecture", ""), row.get("label", ""))] += 1
    return hashes, counts


def iter_source(source: Source) -> Iterable[dict]:
    # Sequential shard access avoids opening many multi-gigabyte Parquet files
    # simultaneously, which commonly stalls on constrained connections.
    config = DownloadConfig(max_retries=20)
    return load_dataset(source.dataset_id, split="train", streaming=True, download_config=config)


def label_for(row: dict, source: str) -> str:
    if source == "community_forensics":
        return "real" if int(row["label"]) == 0 else "fake"
    return "real" if int(row["label"]) == 1 else "fake"


def minimum_edge(source: str) -> int:
    """Keep CIFAKE as a low-resolution stress set, not the quality baseline."""
    return 32 if source == "cifake" else 512


def build(output_dir: Path, per_architecture: int, highres_per_class: int, cifake_per_class: int, image_size: int, quality: int, source_names: set[str] | None = None) -> Counter:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    seen, counts = existing_hashes(manifest_path)
    targets = {("community_forensics", arch, "fake"): per_architecture for arch in ARCHITECTURES}
    targets[("community_forensics", "real", "real")] = per_architecture * len(ARCHITECTURES)
    targets.update({(source, "other", label): highres_per_class for source in ("highres_mixed", "modern_mixed") for label in ("real", "fake")})
    targets.update({("cifake", "other", label): cifake_per_class for label in ("real", "fake")})
    for split in SPLITS:
        for label in ("real", "fake"):
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    field_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
    with manifest_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        if not field_exists:
            writer.writeheader()
        for source in SOURCES:
            if source_names is not None and source.name not in source_names:
                continue
            source_targets = {key: value for key, value in targets.items() if key[0] == source.name}
            if not source_targets or all(counts[key] >= target for key, target in source_targets.items()):
                continue
            progress = tqdm(desc=f"{source.name}: saving images", unit="image")
            try:
                rows = iter_source(source)
                for row in rows:
                    label = label_for(row, source.name)
                    architecture = architecture_for(row, label, source.name)
                    key = (source.name, architecture if source.name == "community_forensics" else "other", label)
                    if counts[key] >= targets.get(key, 0) or row.get("nsfw_flag", False):
                        continue
                    source_id = str(row.get("image_name") or row.get("id") or counts[key])
                    try:
                        with decode_image(row.get("image_data", row.get("image"))) as image:
                            image = ImageOps.exif_transpose(image).convert("RGB")
                            width, height = image.size
                            if min(width, height) < minimum_edge(source.name):
                                continue
                            digest = image_hash(image)
                            if digest in seen:
                                continue
                            image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                            split = split_for(counts[key], targets[key])
                            relative = Path(split) / label / f"{source.name}_{counts[key]:07d}.jpg"
                            image.save(output_dir / relative, "JPEG", quality=quality, optimize=True)
                    except (OSError, ValueError, TypeError, KeyError):
                        continue
                    writer.writerow({"path": relative.as_posix(), "label": label, "source": source.name, "source_id": source_id, "architecture": architecture, "width": width, "height": height, "sha256": digest})
                    stream.flush()
                    seen.add(digest)
                    counts[key] += 1
                    progress.update(1)
                    if all(counts[key] >= target for key, target in source_targets.items()):
                        break
            except Exception as error:
                print(f"Warning: {source.name} could not be completed: {error}")
            progress.close()
    (output_dir / "sources.json").write_text(json.dumps([asdict(source) for source in SOURCES], indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps({"counts": {"/".join(key): value for key, value in counts.items()}, "image_size": image_size, "quality": quality}, indent=2) + "\n", encoding="utf-8")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a diverse, high-resolution, deduplicated image corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_corpus"))
    parser.add_argument("--per-architecture", type=int, default=10_000, help="Community Forensics fake images per generator family.")
    parser.add_argument("--highres-per-class", type=int, default=10_000)
    parser.add_argument("--cifake-per-class", type=int, default=0, help="Optional low-resolution CIFAKE stress samples; excluded by default.")
    parser.add_argument("--image-size", type=int, default=1024, help="Maximum saved edge; source resolution is checked before resizing.")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--sources", nargs="+", choices=[source.name for source in SOURCES], default=None)
    args = parser.parse_args()
    if min(args.per_architecture, args.highres_per_class, args.cifake_per_class) < 0 or args.image_size < 1 or not 1 <= args.quality <= 95:
        parser.error("counts must be non-negative, image size must be positive, and quality must be 1..95")
    return args


if __name__ == "__main__":
    args = parse_args()
    build(args.output_dir, args.per_architecture, args.highres_per_class, args.cifake_per_class, args.image_size, args.quality, set(args.sources) if args.sources else None)
