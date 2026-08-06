#!/usr/bin/env python3
"""
TruthLens — Download date3k2/raw_real_fake_images
==================================================
Downloads high-resolution modern AI images (DALL-E 3, Midjourney v6, etc.)
and real photos at 1024px-1536px resolution.

label=0 -> FAKE (modern AI generators)
label=1 -> REAL (actual photographs)
"""

import sys
import shutil
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "dataset_dalle"

def main(n_per_class: int = 1000):
    print("=" * 70)
    print(f"  Downloading Modern AI Images (DALL-E 3 / Midjourney v6 style)")
    print(f"  Target: {n_per_class} Real + {n_per_class} Fake @ 1024px+")
    print("=" * 70)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("date3k2/raw_real_fake_images", split="train", streaming=True)
    counts = {"real": 0, "fake": 0}
    pbar = tqdm(total=n_per_class * 2, desc="Extracting 1024px+ Real & DALL-E Images", ncols=80)

    for item in ds:
        if counts["real"] >= n_per_class and counts["fake"] >= n_per_class:
            break

        label = item["label"]  # 1=real, 0=fake
        cls = "real" if label == 1 else "fake"

        if counts[cls] >= n_per_class:
            continue

        img = item["image"]
        w, h = img.size
        # Only keep genuinely high-res images
        if w < 512 or h < 512:
            continue

        idx = counts[cls]
        if idx < int(n_per_class * 0.8):
            target_split = "train"
        elif idx < int(n_per_class * 0.9):
            target_split = "val"
        else:
            target_split = "test"

        out_path = OUTPUT_DIR / target_split / cls / f"dalle_{idx:05d}.jpg"
        try:
            img.convert("RGB").save(out_path, quality=95)
            counts[cls] += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()
    print("\n" + "=" * 70)
    print("  DALL-E / Modern AI Dataset Download Complete!")
    print("=" * 70)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            n = len(list((OUTPUT_DIR / split / cls).glob("*.jpg")))
            print(f"  dataset_dalle/{split:6s}/{cls:5s} → {n:5,} images")
    print(f"  TOTAL: {counts['real'] + counts['fake']:,}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n)
