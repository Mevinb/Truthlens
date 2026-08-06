#!/usr/bin/env python3
"""
TruthLens — HuggingFace CIFAKE Dataset Downloader & Extractor
=============================================================
Downloads CIFAKE from Hugging Face (`dragonintelligence/CIFAKE-image-dataset`)
and saves images into local folder structure:
  dataset/train/real/ & fake/
  dataset/val/real/   & fake/
  dataset/test/real/  & fake/
"""

import os
import random
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "dataset"

def main():
    print("=" * 60)
    print("  TruthLens — Downloading CIFAKE via Hugging Face Hub")
    print("=" * 60)

    ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    label_names = ds["train"].features["label"].names # ['FAKE', 'REAL']

    # Map HF label names ('REAL'/'FAKE') to TruthLens folder names ('real'/'fake')
    name_map = {"REAL": "real", "FAKE": "fake"}

    # 1. Process Train set & split 10% into Val set
    train_ds = ds["train"]
    print(f"\n[1/2] Processing train dataset ({len(train_ds):,} images)...")

    # Create directories
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    # Generate split indices (90% train, 10% val)
    n_train = len(train_ds)
    val_indices = set(random.sample(range(n_train), int(n_train * 0.1)))

    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}

    for i, item in enumerate(tqdm(train_ds, desc="Extracting Train/Val", ncols=80)):
        img = item["image"]
        raw_label = label_names[item["label"]] # 'REAL' or 'FAKE'
        cls = name_map[raw_label]
        split = "val" if i in val_indices else "train"

        out_path = DATA_DIR / split / cls / f"{i:06d}.jpg"
        img.convert("RGB").save(out_path, quality=95)
        counts[split][cls] += 1

    # 2. Process Test set
    test_ds = ds["test"]
    print(f"\n[2/2] Processing test dataset ({len(test_ds):,} images)...")
    for i, item in enumerate(tqdm(test_ds, desc="Extracting Test", ncols=80)):
        img = item["image"]
        raw_label = label_names[item["label"]] # 'REAL' or 'FAKE'
        cls = name_map[raw_label]

        out_path = DATA_DIR / "test" / cls / f"{i:06d}.jpg"
        img.convert("RGB").save(out_path, quality=95)
        counts["test"][cls] += 1

    print("\n" + "=" * 60)
    print("  CIFAKE Dataset Extraction Complete!")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            print(f"  dataset/{split}/{cls:5s} → {counts[split][cls]:6,} images")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
