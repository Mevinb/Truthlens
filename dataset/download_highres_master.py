#!/usr/bin/env python3
"""
TruthLens — High-Resolution Master Dataset Extractor (512px+ Only)
====================================================================
Streams and extracts ONLY High-Resolution (512px+ to 4K) Real Photos & AI Images.
Filters out low-resolution images (< 400x400) to ensure the model learns
real camera sensor characteristics, HDR, and modern generator artifacts.

Sources:
  • Hemg/AI-Generated-vs-Real-Images-Datasets (High-Res 512px to 2940px)
  • assassinwizz/AI-Generated-vs-Real-Images-Dataset

Outputs to: dataset_highres_master/{train/, val/, test/}
"""

import sys
import shutil
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent
HIGHRES_MASTER_DIR = ROOT_DIR / "dataset_highres_master"

def main(n_per_class: int = 5000):
    print("=" * 70)
    print(f"  TruthLens — High-Res 512px+ Dataset Extractor ({n_per_class*2:,} High-Res Images)")
    print("=" * 70)

    if HIGHRES_MASTER_DIR.exists():
        shutil.rmtree(HIGHRES_MASTER_DIR)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (HIGHRES_MASTER_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"real": 0, "fake": 0}

    print("\n[Streaming High-Resolution Real Photos & AI Images...]")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)
    
    pbar = tqdm(total=n_per_class * 2, desc="Extracting High-Res (512px+)", ncols=85)

    for item in ds:
        if counts["real"] >= n_per_class and counts["fake"] >= n_per_class:
            break

        label_idx = item.get("label", 0) # 0 -> fake, 1 -> real
        cls = "real" if label_idx == 1 else "fake"

        if counts[cls] >= n_per_class:
            continue

        img = item["image"]
        w, h = img.size
        # STRICT FILTER: Reject any image smaller than 400x400
        if w < 400 or h < 400:
            continue

        idx = counts[cls]
        if idx < int(n_per_class * 0.8):
            target_split = "train"
        elif idx < int(n_per_class * 0.9):
            target_split = "val"
        else:
            target_split = "test"

        out_path = HIGHRES_MASTER_DIR / target_split / cls / f"highres_{idx:05d}.jpg"
        try:
            img.convert("RGB").save(out_path, quality=95)
            counts[cls] += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()

    print("\n" + "=" * 70)
    print("  High-Resolution Dataset Extraction Complete!")
    print("=" * 70)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            n = len(list((HIGHRES_MASTER_DIR / split / cls).glob("*.jpg")))
            print(f"  dataset_highres_master/{split:6s}/{cls:5s} → {n:6,} images")
    print(f"  TOTAL HIGH-RES IMAGES: {counts['real'] + counts['fake']:,}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(n)
