#!/usr/bin/env python3
"""
TruthLens — 150,000 High-Resolution Dataset Extractor
=====================================================
Ingests from `Hemg/AI-Generated-vs-Real-Images-Datasets` (152,710 images):
  • 71,536 Real High-Res Camera & Art Photos (label = 1)
  • 81,174 AI-Generated Synthetic Images & SDXL (label = 0)

Filters out any low-res thumbnail (< 400x400) to ensure high-res training.
"""

import sys
import shutil
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "dataset_highres_master"

def main(n_per_class: int = 5000):
    print("=" * 70)
    print(f"  TruthLens — Ingesting High-Res Dataset ({n_per_class*2:,} images)")
    print("=" * 70)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    print("Loading 152,710 High-Res Dataset from Hugging Face...")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")

    counts = {"real": 0, "fake": 0}
    pbar = tqdm(total=n_per_class * 2, desc="Extracting High-Res Real & Fake", ncols=85)

    random.seed(42)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for i in indices:
        if counts["real"] >= n_per_class and counts["fake"] >= n_per_class:
            break

        item = ds[i]
        label = item["label"] # 1 -> real, 0 -> fake
        cls = "real" if label == 1 else "fake"

        if counts[cls] >= n_per_class:
            continue

        img = item["image"]
        w, h = img.size
        if w < 400 or h < 400:
            continue

        idx = counts[cls]
        if idx < int(n_per_class * 0.8):
            target_split = "train"
        elif idx < int(n_per_class * 0.9):
            target_split = "val"
        else:
            target_split = "test"

        out_path = OUTPUT_DIR / target_split / cls / f"highres_{idx:05d}.jpg"
        try:
            img.convert("RGB").save(out_path, quality=95)
            counts[cls] += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()

    print("\n" + "=" * 70)
    print("  High-Resolution Dataset Ingestion Complete!")
    print("=" * 70)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            n = len(list((OUTPUT_DIR / split / cls).glob("*.jpg")))
            print(f"  dataset_highres_master/{split:6s}/{cls:5s} → {n:6,} images")
    print(f"  TOTAL HIGH-RES IMAGES: {counts['real'] + counts['fake']:,}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(n)
