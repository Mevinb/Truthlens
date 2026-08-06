#!/usr/bin/env python3
"""
TruthLens — CIFAKE Dataset Downloader
======================================
Downloads the CIFAKE dataset (Real vs AI-Generated Images).

Usage:
    python dataset/download_cifake.py --method kaggle   # requires ~/.kaggle/kaggle.json
    python dataset/download_cifake.py --method manual   # prints manual download link

Dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
Classes: REAL (60,000) | FAKE (60,000)
Format : Pre-split train/test directories with REAL/ and FAKE/ subdirectories
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "dataset"
KAGGLE_ID  = "birdy654/cifake-real-and-ai-generated-synthetic-images"
MANUAL_URL = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"


# ─── Kaggle Download ──────────────────────────────────────────────────────────
def download_via_kaggle(data_dir: Path) -> None:
    """Download CIFAKE using the Kaggle API."""
    try:
        import kaggle  # type: ignore
    except ImportError:
        print("[ERROR] kaggle package not found. Install it with: pip install kaggle")
        sys.exit(1)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print(
            "[ERROR] Kaggle credentials not found.\n"
            "  1. Go to https://www.kaggle.com/account → 'Create New API Token'\n"
            "  2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json\n"
            "  3. Run: chmod 600 ~/.kaggle/kaggle.json"
        )
        sys.exit(1)

    print(f"[INFO] Downloading CIFAKE via Kaggle API → {data_dir} ...")
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_ID, path=data_dir, unzip=True)
    print("[INFO] Download complete.")
    _reorganise(data_dir)


def _reorganise(data_dir: Path) -> None:
    """
    CIFAKE zip extracts as:
        train/REAL/  train/FAKE/
        test/REAL/   test/FAKE/

    We reorganise to lowercase and create a val split (10% of train).
    """
    print("[INFO] Reorganising dataset structure ...")

    for split in ["train", "test"]:
        for cls_src, cls_dst in [("REAL", "real"), ("FAKE", "fake")]:
            src = data_dir / split / cls_src
            dst = data_dir / split / cls_dst
            if src.exists() and not dst.exists():
                src.rename(dst)
                print(f"  {split}/{cls_src}  →  {split}/{cls_dst}")

    # Rename test → test (already good), but create val from train
    _create_val_split(data_dir, val_fraction=0.1)
    print("[INFO] Reorganisation complete.")


def _create_val_split(data_dir: Path, val_fraction: float = 0.1) -> None:
    """Move a fraction of train images into val/."""
    import random
    random.seed(42)

    for cls in ["real", "fake"]:
        src_dir = data_dir / "train" / cls
        val_dir = data_dir / "val" / cls
        val_dir.mkdir(parents=True, exist_ok=True)

        files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        n_val  = int(len(files) * val_fraction)
        chosen = random.sample(files, n_val)

        for f in chosen:
            shutil.move(str(f), val_dir / f.name)

        print(f"  [val/{cls}] Created {n_val} images from train/{cls}")


# ─── Manual Instructions ──────────────────────────────────────────────────────
def print_manual_instructions() -> None:
    print(
        "\n" + "="*60 + "\n"
        "  CIFAKE — Manual Download Instructions\n"
        "="*60 + "\n"
        f"  URL: {MANUAL_URL}\n\n"
        "  Steps:\n"
        "  1. Open the URL above in your browser.\n"
        "  2. Click 'Download' (requires free Kaggle account).\n"
        "  3. Extract the zip file.\n"
        "  4. Place extracted folders so your structure looks like:\n\n"
        "       dataset/\n"
        "         train/ real/  fake/\n"
        "         val/   real/  fake/\n"
        "         test/  real/  fake/\n\n"
        "  5. If the zip gives REAL/ and FAKE/ (uppercase), run:\n"
        "       python dataset/download_cifake.py --method reorganise\n"
        "="*60 + "\n"
    )


def reorganise_existing(data_dir: Path) -> None:
    """Reorganise an already-downloaded and extracted dataset."""
    _reorganise(data_dir)


# ─── Dataset Info ─────────────────────────────────────────────────────────────
def print_dataset_info(data_dir: Path) -> None:
    print("\n" + "="*50)
    print("  TruthLens — Dataset Status")
    print("="*50)
    total = 0
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            d = data_dir / split / cls
            if d.exists():
                count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                print(f"  {split:6s}/{cls:5s}  →  {count:6,} images")
                total += count
            else:
                print(f"  {split:6s}/{cls:5s}  →  [NOT FOUND]")
    print(f"  {'TOTAL':12s}  →  {total:6,} images")
    print("="*50 + "\n")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & prepare the CIFAKE dataset for TruthLens."
    )
    parser.add_argument(
        "--method",
        choices=["kaggle", "manual", "reorganise", "info"],
        default="info",
        help=(
            "kaggle     : Download using Kaggle API (requires credentials)\n"
            "manual     : Print manual download instructions\n"
            "reorganise : Reorganise an already-downloaded dataset\n"
            "info       : Print current dataset statistics"
        ),
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.method == "kaggle":
        download_via_kaggle(DATA_DIR)
    elif args.method == "manual":
        print_manual_instructions()
    elif args.method == "reorganise":
        reorganise_existing(DATA_DIR)
    elif args.method == "info":
        print_dataset_info(DATA_DIR)

    print_dataset_info(DATA_DIR)


if __name__ == "__main__":
    main()
