#!/usr/bin/env python3
"""
TruthLens — 100,000+ Master Dataset Extractor
================================================
Downloads and extracts 100,000+ Real & AI-Generated images:
  1. dragonintelligence/CIFAKE-image-dataset (116,000 images: 58k Real, 58k Fake)
  2. Hemg/AI-Generated-vs-Real-Images-Datasets (High-res 512px+ real photos & AI art)

Outputs to: dataset_master/{train/, val/, test/}
"""

import sys
import shutil
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent
MASTER_DIR = ROOT_DIR / "dataset_master"

def main():
    print("=" * 70)
    print("  TruthLens — 100,000+ Image Master Dataset Extraction")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (MASTER_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"real": 0, "fake": 0}

    # 1. Extract CIFAKE (100,000 images)
    print("\n[1/2] Ingesting 100,000 CIFAKE Images...")
    try:
        ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")
        label_names = ds["train"].features["label"].names # ['FAKE', 'REAL']
        name_map = {"REAL": "real", "FAKE": "fake"}

        for split_name in ["train", "test"]:
            sub_ds = ds[split_name]
            for i, item in enumerate(tqdm(sub_ds, desc=f"Extracting CIFAKE {split_name}", ncols=80)):
                img = item["image"]
                raw_label = label_names[item["label"]]
                cls = name_map[raw_label]

                # Split 80% train, 10% val, 10% test
                r = random.random()
                if r < 0.8:
                    target_split = "train"
                elif r < 0.9:
                    target_split = "val"
                else:
                    target_split = "test"

                idx = counts[cls]
                out_path = MASTER_DIR / target_split / cls / f"cifake_{idx:06d}.jpg"
                if not out_path.exists():
                    img.convert("RGB").save(out_path, quality=95)
                counts[cls] += 1

    except Exception as exc:
        print(f"  Warning: CIFAKE extraction error: {exc}")

    # 2. Extract High-Res Images (Hemg/AI-Generated-vs-Real-Images-Datasets)
    print("\n[2/2] Streaming & Extracting High-Res Photos & AI Art...")
    try:
        ds_highres = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)
        highres_count = 0
        for item in tqdm(ds_highres, desc="Extracting High-Res", ncols=80):
            if highres_count >= 5000:
                break

            label = item.get("label")
            cls = "real" if label == 1 else "fake"
            img = item["image"]

            r = random.random()
            if r < 0.8:
                target_split = "train"
            elif r < 0.9:
                target_split = "val"
            else:
                target_split = "test"

            idx = counts[cls]
            out_path = MASTER_DIR / target_split / cls / f"highres_{idx:06d}.jpg"
            if not out_path.exists():
                try:
                    img.convert("RGB").save(out_path, quality=95)
                    counts[cls] += 1
                    highres_count += 1
                except Exception:
                    continue
    except Exception as exc:
        print(f"  Warning: High-res extraction error: {exc}")

    print("\n" + "=" * 70)
    print("  100,000+ Master Dataset Extraction Complete!")
    print("=" * 70)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            n = len(list((MASTER_DIR / split / cls).glob("*.jpg")))
            print(f"  dataset_master/{split:6s}/{cls:5s} → {n:6,} images")
    print(f"  TOTAL IMAGES IN MASTER DATASET: {counts['real'] + counts['fake']:,}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
