#!/usr/bin/env python3
"""
TruthLens — High-Resolution (512px+) Dataset Extractor
=========================================================
Streams and extracts high-resolution AI-generated vs Real images
(resolutions: 512x512 to 2940x1960) from Hugging Face Hub.

Dataset: `Hemg/AI-Generated-vs-Real-Images-Datasets`
Labels:  0 -> AiArtData (fake), 1 -> RealArt (real)
Output:  dataset_highres/ {train/, val/, test/} (50% real, 50% fake)
"""

import sys
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent
HIGHRES_DIR = ROOT_DIR / "dataset_highres"

def main(n_per_class: int = 500):
    print("=" * 65)
    print(f"  TruthLens — High-Res Dataset Extractor ({n_per_class*2:,} images)")
    print("=" * 65)

    # Clean old directory
    if HIGHRES_DIR.exists():
        shutil.rmtree(HIGHRES_DIR)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (HIGHRES_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)
    label_map = {0: "fake", 1: "real"}

    counts = {"fake": 0, "real": 0}
    pbar = tqdm(total=n_per_class * 2, desc="Extracting High-Res Images", ncols=85)

    for item in ds:
        label_idx = item["label"]
        cls = label_map[label_idx]

        if counts[cls] >= n_per_class:
            if counts["real"] >= n_per_class and counts["fake"] >= n_per_class:
                break
            continue

        img = item["image"]
        w, h = img.size
        if w < 200 or h < 200:
            continue

        idx = counts[cls]
        if idx < int(n_per_class * 0.8):
            split = "train"
        elif idx < int(n_per_class * 0.9):
            split = "val"
        else:
            split = "test"

        out_path = HIGHRES_DIR / split / cls / f"{idx:05d}.jpg"
        img.convert("RGB").save(out_path, quality=95)
        counts[cls] += 1
        pbar.update(1)

    pbar.close()

    print("\n" + "=" * 65)
    print("  High-Resolution Dataset Extraction Complete!")
    print("=" * 65)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            path = HIGHRES_DIR / split / cls
            n = len(list(path.glob("*.jpg")))
            print(f"  dataset_highres/{split:6s}/{cls:5s} → {n:5,} images")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    main(n)
