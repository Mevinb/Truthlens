#!/usr/bin/env python3
"""
TruthLens — Multi-Resolution Balanced Dataset Builder
======================================================
Builds a curated, balanced dataset covering ALL resolution tiers:
  Tier 1: High-Resolution (512px - 2048px+) — Modern AI (DALL-E 3, SDXL, Midjourney) + DSLR/Phone Cameras
  Tier 2: Medium-Resolution (256px) — Multi-domain photo/synthetic archive
  Tier 3: Low-Resolution / Compressed (32px - 64px) — CIFAKE benchmark

Outputs to: dataset_multires/{train/, val/, test/} with balanced real & fake classes.
"""

import sys
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "dataset_multires"

def copy_or_save_image(src_path: Path, dst_path: Path, quality: int = 95):
    try:
        with Image.open(src_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, "JPEG", quality=quality, optimize=True)
            return True
    except Exception:
        return False

def add_split_samples(samples, target_counts, prefix="sample"):
    """Distributes a list of file paths (or tuples) into train/val/test splits."""
    random.shuffle(samples)
    total = len(samples)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:]
    }
    return splits

def main(
    n_highres_per_class: int = 4500,
    n_medres_per_class: int = 6000,
    n_lowres_per_class: int = 3000
):
    print("=" * 75)
    print("  TruthLens — Building Comprehensive Multi-Resolution Dataset")
    print(f"  Target per class: {n_highres_per_class} High-Res + {n_medres_per_class} 256px + {n_lowres_per_class} Low-Res")
    print(f"  Total target images: {(n_highres_per_class + n_medres_per_class + n_lowres_per_class) * 2:,}")
    print("=" * 75)

    if OUTPUT_DIR.exists():
        print(f"Removing previous {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}

    # =========================================================================
    # 1. High-Resolution Tier (512px - 2048px+)
    # =========================================================================
    print("\n[Tier 1/3] Ingesting High-Resolution Real Photos & Modern AI Images...")
    highres_real = []
    highres_fake = []

    # Local modern corpus
    modern_dir = ROOT_DIR / "dataset_modern_corpus"
    if modern_dir.exists():
        highres_real.extend(list((modern_dir / "train" / "real").glob("*.jpg")))
        highres_fake.extend(list((modern_dir / "train" / "fake").glob("*.jpg")))

    # Local highres corpus
    highres_corpus_dir = ROOT_DIR / "dataset_highres_corpus"
    if highres_corpus_dir.exists():
        highres_real.extend(list((highres_corpus_dir / "train" / "real").glob("*.jpg")))
        highres_fake.extend(list((highres_corpus_dir / "train" / "fake").glob("*.jpg")))

    print(f"  Found {len(highres_real):,} local High-Res Real, {len(highres_fake):,} local High-Res Fake.")

    # Balance counts to minimum available if not streaming
    n_high = min(n_highres_per_class, len(highres_real), len(highres_fake))

    # Process and save High-Res images
    for cls, img_list in [("real", highres_real), ("fake", highres_fake)]:
        img_list = img_list[:n_high]
        random.shuffle(img_list)
        splits = add_split_samples(img_list, len(img_list))
        for split, items in splits.items():
            for p in items:
                dst = OUTPUT_DIR / split / cls / f"highres_{counts[split][cls]:06d}.jpg"
                if copy_or_save_image(p, dst):
                    counts[split][cls] += 1

    print(f"  Tier 1 Complete: High-Res Real={sum(counts[s]['real'] for s in counts):,}, High-Res Fake={sum(counts[s]['fake'] for s in counts):,}")

    # =========================================================================
    # 2. Medium-Resolution Tier (256px)
    # =========================================================================
    print("\n[Tier 2/3] Ingesting Medium-Resolution (256px) Images...")
    med_real = []
    med_fake = []

    archive_dir = ROOT_DIR / "dataset" / "archive"
    if archive_dir.exists():
        for d in ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]:
            real_path = archive_dir / d / d / "train" / "real"
            fake_path = archive_dir / d / d / "train" / "fake"
            if real_path.exists():
                med_real.extend(list(real_path.glob("*.jpg")))
            if fake_path.exists():
                med_fake.extend(list(fake_path.glob("*.jpg")))

    print(f"  Found {len(med_real):,} 256px Real, {len(med_fake):,} 256px Fake.")
    for cls, img_list in [("real", med_real), ("fake", med_fake)]:
        random.shuffle(img_list)
        img_list = img_list[:n_medres_per_class]
        splits = add_split_samples(img_list, len(img_list))
        for split, items in splits.items():
            for p in items:
                dst = OUTPUT_DIR / split / cls / f"medres256_{counts[split][cls]:06d}.jpg"
                if copy_or_save_image(p, dst):
                    counts[split][cls] += 1

    print(f"  Tier 2 Complete. Running totals: Real={sum(counts[s]['real'] for s in counts):,}, Fake={sum(counts[s]['fake'] for s in counts):,}")

    # =========================================================================
    # 3. Low-Resolution Tier (32px - 64px CIFAKE Benchmark)
    # =========================================================================
    print("\n[Tier 3/3] Ingesting Low-Resolution (32px) Benchmark Images...")
    cifake_dir = ROOT_DIR / "dataset"
    cifake_real = list((cifake_dir / "train" / "real").glob("*.jpg"))
    cifake_fake = list((cifake_dir / "train" / "fake").glob("*.jpg"))

    print(f"  Found {len(cifake_real):,} CIFAKE Real, {len(cifake_fake):,} CIFAKE Fake.")
    for cls, img_list in [("real", cifake_real), ("fake", cifake_fake)]:
        random.shuffle(img_list)
        img_list = img_list[:n_lowres_per_class]
        splits = add_split_samples(img_list, len(img_list))
        for split, items in splits.items():
            for p in items:
                dst = OUTPUT_DIR / split / cls / f"lowres32_{counts[split][cls]:06d}.jpg"
                if copy_or_save_image(p, dst):
                    counts[split][cls] += 1

    # =========================================================================
    # Final Dataset Summary
    # =========================================================================
    print("\n" + "=" * 75)
    print("  🎉 Multi-Resolution TruthLens Dataset Construction Complete!")
    print("=" * 75)
    total_all = 0
    for split in ["train", "val", "test"]:
        n_real = counts[split]["real"]
        n_fake = counts[split]["fake"]
        total_split = n_real + n_fake
        total_all += total_split
        print(f"  dataset_multires/{split:6s} → Real: {n_real:6,} | Fake: {n_fake:6,} | Total: {total_split:6,}")

    print("-" * 75)
    print(f"  GRAND TOTAL MULTI-RESOLUTION IMAGES: {total_all:,}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    n_high = int(sys.argv[1]) if len(sys.argv) > 1 else 4500
    n_med  = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    n_low  = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    main(n_high, n_med, n_low)
