#!/usr/bin/env python3
"""Flatten the raw legacy archive into a trainable prepared corpus.

The archive already ships with train/validation/test splits, so this script
keeps those splits intact and only standardises the layout to
``datasets/prepared/legacy_archive/{train,val,test}/{real,fake}``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _legacy_roots(source_root: Path) -> list[Path]:
    roots: list[Path] = []
    for archive_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        nested = archive_dir / archive_dir.name
        if nested.exists():
            roots.append(nested)
        else:
            roots.append(archive_dir)
    return roots


def _images(directory: Path):
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_dataset(source_root: Path, output_dir: Path) -> dict[str, int]:
    """Prepare the archive without changing its original split structure."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    counts = {split: 0 for split in SPLIT_MAP.values()}

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "split", "label", "source_archive", "source_split"),
        )
        writer.writeheader()

        for archive_root in _legacy_roots(source_root):
            source_name = _safe_name(archive_root.parent.name)
            for source_split, target_split in SPLIT_MAP.items():
                for label in ("real", "fake"):
                    split_dir = archive_root / source_split / label
                    if not split_dir.exists():
                        continue
                    for index, source in enumerate(_images(split_dir)):
                        destination = (
                            output_dir / target_split / label /
                            f"{source_name}_{source_split}_{index:06d}{source.suffix.lower()}"
                        )
                        _link(source, destination)
                        writer.writerow(
                            {
                                "path": str(destination.relative_to(output_dir)),
                                "split": target_split,
                                "label": label,
                                "source_archive": archive_root.parent.name,
                                "source_split": source_split,
                            }
                        )
                        counts[target_split] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/raw/legacy_archive"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/prepared/legacy_archive"),
    )
    args = parser.parse_args()
    counts = prepare_dataset(args.source_root, args.output_dir)
    print(f"Prepared {args.output_dir} with {counts} samples; manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
