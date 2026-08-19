#!/usr/bin/env python3
"""
TruthLens — Pure DALL-E 3 Dataset Extractor
=============================================
Streams and extracts authentic 1024x1024 DALL-E 3 generated images
from `ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions`.
"""

import sys
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "datasets" / "prepared" / "multires"

def main(n_images: int = 2500):
    print("=" * 70)
    print(f"  TruthLens — Extracting {n_images:,} Authentic DALL-E 3 Images (1024x1024)")
    print("=" * 70)

    ds = load_dataset(
        "ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions",
        split="train",
        streaming=True
    )

    counts = {"train": 0, "val": 0, "test": 0}
    target_counts = {
        "train": int(n_images * 0.8),
        "val": int(n_images * 0.1),
        "test": int(n_images * 0.1)
    }

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "fake").mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=n_images, desc="Downloading DALL-E 3 Images", ncols=80)
    total_saved = 0

    for item in ds:
        if total_saved >= n_images:
            break

        img = item.get("jpg") or item.get("jpeg") or item.get("png")
        if not img:
            continue

        # Choose split
        if counts["train"] < target_counts["train"]:
            split = "train"
        elif counts["val"] < target_counts["val"]:
            split = "val"
        elif counts["test"] < target_counts["test"]:
            split = "test"
        else:
            break

        out_path = OUTPUT_DIR / split / "fake" / f"pure_dalle3_{total_saved:05d}.jpg"
        try:
            img.convert("RGB").save(out_path, quality=95)
            counts[split] += 1
            total_saved += 1
            pbar.update(1)
        except Exception as exc:
            continue

    pbar.close()
    print("\n" + "=" * 70)
    print("  DALL-E 3 Image Extraction Complete!")
    print(f"  Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    main(n)
