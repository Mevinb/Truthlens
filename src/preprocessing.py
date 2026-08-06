#!/usr/bin/env python3
"""
TruthLens — src/preprocessing.py
==================================
Dataset classes, transforms, and feature extraction for both:
  • Deep Learning  → PyTorch Dataset + DataLoaders
  • Classical ML   → HOG + LBP feature vectors
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from src.utils import Config, get_logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

logger = get_logger(__name__)


# ─── ImageNet normalisation constants ─────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split: str, img_size: int = 224):
    """
    Return augmentation pipeline for train / val+test splits.

    Train augmentations are intentionally strong because CIFAKE images are
    32×32 upsampled to 224×224 — extra augmentation helps prevent overfitting.
    """
    from torchvision import transforms
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class CIFAKEDataset(Dataset):
    """
    Expects a directory tree:
        root/
          real/  *.jpg | *.png
          fake/  *.jpg | *.png

    Labels: real → 0, fake → 1
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
    ) -> None:
        self.root      = Path(root)
        self.split     = split
        self.img_size  = img_size
        self.transform = transform or get_transforms(split, img_size)

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        class_map = {"real": 0, "fake": 1}
        for cls_name, label in class_map.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                logger.warning(f"Class directory not found: {cls_dir}")
                continue
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((fpath, label))

        if not self.samples:
            logger.warning(
                f"No images found in {self.root}. "
                "Did you run `python dataset/download_cifake.py --method kaggle`?"
            )
        else:
            n_real = sum(1 for _, lbl in self.samples if lbl == 0)
            n_fake = sum(1 for _, lbl in self.samples if lbl == 1)
            logger.info(
                f"[{self.split}] Loaded {len(self.samples)} images "
                f"(real={n_real}, fake={n_fake})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.error(f"Failed to open image {path}: {exc}")
            img = Image.new("RGB", (self.img_size, self.img_size))
        return self.transform(img), label

    @property
    def class_weights(self) -> "torch.Tensor":
        """Inverse-frequency class weights for imbalanced datasets."""
        import torch
        labels = [lbl for _, lbl in self.samples]
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = 1.0 / (counts + 1e-8)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoader Factory ───────────────────────────────────────────────────────
def get_dataloaders(cfg: Config) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders from config."""
    import torch

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dir = cfg.data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory missing: {split_dir} — skipping.")
            continue

        dataset = CIFAKEDataset(
            root=split_dir,
            split=split,
            img_size=cfg.img_size,
        )

        if len(dataset) == 0:
            logger.warning(f"Empty dataset for split '{split}' — skipping.")
            continue

        shuffle     = split == "train"
        pin_memory  = torch.cuda.is_available()

        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=split == "train",
        )
        logger.info(
            f"[{split}] DataLoader: {len(dataset)} samples, "
            f"{len(loaders[split])} batches"
        )

    return loaders


# ─── HOG Feature Extractor ────────────────────────────────────────────────────
def extract_hog_features(
    img_array: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract Histogram of Oriented Gradients (HOG) features."""
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


# ─── LBP Feature Extractor ────────────────────────────────────────────────────
def extract_lbp_features(
    img_array: np.ndarray,
    n_points: int = 24,
    radius: int = 3,
    n_bins: int = 64,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract Local Binary Pattern (LBP) texture histogram."""
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size)

    gray  = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lbp   = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ─── Combined Feature Extractor ───────────────────────────────────────────────
def extract_features(img_array: np.ndarray) -> np.ndarray:
    """Combine HOG + LBP into a single feature vector."""
    hog_feat = extract_hog_features(img_array)
    lbp_feat = extract_lbp_features(img_array)
    return np.concatenate([hog_feat, lbp_feat])


# ─── Dataset Feature Extractor ────────────────────────────────────────────────
def build_feature_dataset(
    split_dir: Path,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk a split directory (real/ + fake/) and extract features for every image.
    Returns (X, y) numpy arrays ready for scikit-learn.
    """
    from tqdm import tqdm

    X_list, y_list = [], []
    class_map = {"real": 0, "fake": 1}

    for cls_name, label in class_map.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            logger.warning(f"[Feature Build] Missing: {cls_dir}")
            continue

        files = sorted(cls_dir.iterdir())
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if max_samples_per_class:
            files = files[:max_samples_per_class]

        logger.info(f"  Extracting features from {cls_name} ({len(files)} images) ...")
        for fpath in tqdm(files, desc=f"{cls_name}", leave=False, ncols=80):
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                feat = extract_features(img)
                X_list.append(feat)
                y_list.append(label)
            except Exception as exc:
                logger.debug(f"  Skip {fpath.name}: {exc}")

    if not X_list:
        raise RuntimeError(
            f"No features extracted from {split_dir}. "
            "Check that the dataset is downloaded and organised correctly."
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ─── Single Image Preprocessor ────────────────────────────────────────────────
def preprocess_single_image(
    image_source,
    img_size: int = 224,
    return_tensor: bool = True,
):
    """
    Preprocess a single image for CNN inference.

    Args:
        image_source : PIL.Image, numpy array, or file path string/Path.
        img_size     : Target size (default 224 for ResNet).
        return_tensor: If True, returns a torch.Tensor [1, 3, H, W].
                       If False, returns a numpy array [H, W, 3].
    """
    import torch

    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_source)}")

    transform = get_transforms("test", img_size)
    tensor    = transform(img)

    if return_tensor:
        return tensor.unsqueeze(0)  # [1, 3, H, W]
    else:
        return np.array(img.resize((img_size, img_size)))
