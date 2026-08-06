#!/usr/bin/env python3
"""
TruthLens — tests/test_preprocessing.py
Pytest unit tests for the preprocessing pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def random_rgb_image() -> np.ndarray:
    """Synthetic 64×64 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def pil_image(random_rgb_image) -> Image.Image:
    return Image.fromarray(random_rgb_image)


# ─── Transform Tests ─────────────────────────────────────────────────────────
class TestTransforms:
    def test_train_transform_output_shape(self, pil_image):
        from src.preprocessing import get_transforms
        tf = get_transforms("train", img_size=224)
        tensor = tf(pil_image)
        assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"

    def test_val_transform_output_shape(self, pil_image):
        from src.preprocessing import get_transforms
        tf = get_transforms("val", img_size=224)
        tensor = tf(pil_image)
        assert tensor.shape == (3, 224, 224)

    def test_test_transform_output_shape(self, pil_image):
        from src.preprocessing import get_transforms
        tf = get_transforms("test", img_size=112)
        tensor = tf(pil_image)
        assert tensor.shape == (3, 112, 112)

    def test_normalisation_applied(self, pil_image):
        """After ImageNet normalisation, pixel values should not be in [0,1] raw range."""
        from src.preprocessing import get_transforms
        tf = get_transforms("test", img_size=224)
        tensor = tf(pil_image)
        # After normalisation, some values will be negative
        assert tensor.min() < 0 or tensor.max() > 1


# ─── Feature Extraction Tests ─────────────────────────────────────────────────
class TestFeatureExtraction:
    def test_hog_feature_shape(self, random_rgb_image):
        from src.preprocessing import extract_hog_features
        feat = extract_hog_features(random_rgb_image)
        assert feat.ndim == 1
        assert feat.shape[0] > 0
        assert feat.dtype == np.float32

    def test_lbp_feature_shape(self, random_rgb_image):
        from src.preprocessing import extract_lbp_features
        feat = extract_lbp_features(random_rgb_image)
        assert feat.ndim == 1
        assert feat.shape[0] == 64    # n_bins=64
        assert feat.dtype == np.float32

    def test_combined_features_concatenated(self, random_rgb_image):
        from src.preprocessing import extract_features, extract_hog_features, extract_lbp_features
        hog = extract_hog_features(random_rgb_image)
        lbp = extract_lbp_features(random_rgb_image)
        combined = extract_features(random_rgb_image)
        assert combined.shape[0] == hog.shape[0] + lbp.shape[0]

    def test_lbp_histogram_sums_to_one(self, random_rgb_image):
        from src.preprocessing import extract_lbp_features
        feat = extract_lbp_features(random_rgb_image)
        # density=True → should integrate to ~1
        assert abs(feat.sum() - 1.0) < 0.5   # loose check


# ─── Single Image Preprocessor Tests ─────────────────────────────────────────
class TestSingleImagePreprocessor:
    def test_from_pil_returns_tensor(self, pil_image):
        from src.preprocessing import preprocess_single_image
        tensor = preprocess_single_image(pil_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_from_numpy_returns_tensor(self, random_rgb_image):
        from src.preprocessing import preprocess_single_image
        tensor = preprocess_single_image(random_rgb_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_unsupported_type_raises(self):
        from src.preprocessing import preprocess_single_image
        with pytest.raises(TypeError):
            preprocess_single_image(12345)

    def test_return_numpy_when_flagged(self, pil_image):
        from src.preprocessing import preprocess_single_image
        arr = preprocess_single_image(pil_image, return_tensor=False)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3    # H, W, C


# ─── Config Tests ─────────────────────────────────────────────────────────────
class TestConfig:
    def test_default_config_valid(self):
        from src.utils import Config
        cfg = Config()
        assert cfg.img_size   == 224
        assert cfg.num_epochs == 20
        assert cfg.num_classes == 2
        assert cfg.class_names == ("REAL", "FAKE")

    def test_config_serialisation(self, tmp_path):
        from src.utils import Config
        cfg  = Config()
        path = tmp_path / "config.json"
        cfg.to_json(path)
        cfg2 = Config.from_json(path)
        assert cfg2.img_size    == cfg.img_size
        assert cfg2.num_epochs  == cfg.num_epochs
        assert cfg2.class_names == cfg.class_names
