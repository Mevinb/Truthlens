#!/usr/bin/env python3
"""
TruthLens — Ingest Authentic 1024px DALL-E 3 & High-Res Real Photos
===================================================================
Streams:
  - 2,500 DALL-E 3 synthetic images into dataset_multires/*/fake
  - 2,500 authentic high-resolution real camera photos into dataset_multires/*/real
"""

import sys
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "dataset_multires"

def main(n_images_per_class: int = 2500):
    print("=" * 70)
    print(f"  TruthLens — Ingesting {n_images_per_class:,} DALL-E 3 & Real High-Res Photos")
    print("=" * 70)

    target_train = int(n_images_per_class * 0.8)
    target_val   = int(n_images_per_class * 0.1)
    target_test  = int(n_images_per_class * 0.1)

    # 1. Download DALL-E 3
    print("\n[1/2] Streaming DALL-E 3 (1024x1024) Synthetic Images...")
    ds_fake = load_dataset(
        "ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions",
        split="train",
        streaming=True
    )
    counts_fake = {"train": 0, "val": 0, "test": 0}
    saved_fake = 0
    pbar_f = tqdm(total=n_images_per_class, desc="DALL-E 3 Fake", ncols=80)

    for item in ds_fake:
        if saved_fake >= n_images_per_class:
            break
        img = item.get("jpg") or item.get("jpeg") or item.get("png")
        if not img:
            continue

        if counts_fake["train"] < target_train:
            split = "train"
        elif counts_fake["val"] < target_val:
            split = "val"
        elif counts_fake["test"] < target_test:
            split = "test"
        else:
            break

        out_path = OUTPUT_DIR / split / "fake" / f"pure_dalle3_{saved_fake:06d}.jpg"
        try:
            img.convert("RGB").save(out_path, quality=95)
            counts_fake[split] += 1
            saved_fake += 1
            pbar_f.update(1)
        except Exception:
            continue
    pbar_f.close()

    # 2. Download High-Res Real Photos
    print("\n[2/2] Streaming High-Res Real Camera & Art Photos...")
    ds_real = load_dataset(
        "Hemg/AI-Generated-vs-Real-Images-Datasets",
        split="train",
        streaming=True
    )
    counts_real = {"train": 0, "val": 0, "test": 0}
    saved_real = 0
    pbar_r = tqdm(total=n_images_per_class, desc="Real High-Res Photos", ncols=80)

    for item in ds_real:
        if saved_real >= n_images_per_class:
            break
        if item.get("label") != 1:  # 1 = RealArt
            continue
        img = item.get("image")
        if not img:
            continue
        w, h = img.size
        if w < 512 or h < 512:
            continue

        if counts_real["train"] < target_train:
            split = "train"
        elif counts_real["val"] < target_val:
            split = "val"
        elif counts_real["test"] < target_test:
            split = "test"
        else:
            break

        out_path = OUTPUT_DIR / split / "real" / f"pure_real_photo_{saved_real:06d}.jpg"
        try:
            img.convert("RGB").save(out_path, quality=95)
            counts_real[split] += 1
            saved_real += 1
            pbar_r.update(1)
        except Exception:
            continue
    pbar_r.close()

    print("\n" + "=" * 70)
    print("  Ingestion Complete!")
    print(f"  DALL-E 3 Fake Ingested: {saved_fake:,} (train={counts_fake['train']}, val={counts_fake['val']}, test={counts_fake['test']})")
    print(f"  Real Photos Ingested : {saved_real:,} (train={counts_real['train']}, val={counts_real['val']}, test={counts_real['test']})")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    main(n)
