#!/usr/bin/env python3
"""
TruthLens — Multi-Domain & Multi-Resolution Combined Dataset Ingestion
========================================================================
Combines:
  1. High-Resolution DSLR & Smartphone Camera Photos + SDXL/Midjourney AI Art (dataset_highres_master)
  2. CIFAKE Benchmark Real & Synthetic AI Images (dataset)

This produces a robust model capable of accurately detecting BOTH:
  • Modern 4K smartphone photos (no false positive FAKE)
  • Low-res, compressed, or older AI-generated images (no false positive REAL)

Outputs to: dataset_combined/{train/, val/, test/}
"""

import os
import shutil
import random
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
COMBINED_DIR = ROOT_DIR / "dataset_combined"
HIGHRES_DIR = ROOT_DIR / "dataset_highres_master"
CIFAKE_DIR = ROOT_DIR / "dataset"

def main():
    print("=" * 70)
    print("  TruthLens — Creating Multi-Domain & Multi-Resolution Dataset")
    print("=" * 70)

    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (COMBINED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}

    # 1. Copy High-Res Images
    print("[1/2] Ingesting High-Resolution DSLR/Camera & SDXL Images...")
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            src_folder = HIGHRES_DIR / split / cls
            if not src_folder.exists(): continue
            for img_file in src_folder.glob("*.jpg"):
                out_name = f"highres_{counts[split][cls]:06d}.jpg"
                shutil.copy(img_file, COMBINED_DIR / split / cls / out_name)
                counts[split][cls] += 1

    # 2. Copy CIFAKE Images (5,000 per class for train, 1,000 for val/test)
    print("[2/2] Ingesting CIFAKE Benchmark Real & Synthetic Images...")
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            src_folder = CIFAKE_DIR / split / cls
            if not src_folder.exists(): continue
            img_list = list(src_folder.glob("*.jpg"))
            random.seed(42)
            random.shuffle(img_list)
            
            # Limit CIFAKE so it doesn't overwhelm high-res photos
            limit = 4000 if split == "train" else 800
            for img_file in img_list[:limit]:
                out_name = f"cifake_{counts[split][cls]:06d}.jpg"
                shutil.copy(img_file, COMBINED_DIR / split / cls / out_name)
                counts[split][cls] += 1

    print("\n" + "=" * 70)
    print("  Multi-Domain Combined Dataset Created!")
    print("=" * 70)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            n = len(list((COMBINED_DIR / split / cls).glob("*.jpg")))
            print(f"  dataset_combined/{split:6s}/{cls:5s} → {n:6,} images")
    
    total = sum(counts[s][c] for s in counts for c in counts[s])
    print(f"  TOTAL COMBINED SAMPLES: {total:,}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
