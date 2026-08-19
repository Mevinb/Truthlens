#!/usr/bin/env python3
"""
TruthLens — tests/test_predict.py
Pytest unit tests for the inference pipeline (no trained model required).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def random_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def pil_image(random_image) -> Image.Image:
    return Image.fromarray(random_image)


# ─── Generator Detection ──────────────────────────────────────────────────────
class TestGeneratorDetection:
    def test_returns_string(self, random_image):
        from src.predict import detect_likely_generator
        result = detect_likely_generator(random_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_known_generators_in_options(self, random_image):
        from src.predict import detect_likely_generator
        valid = {
            "Midjourney", "Midjourney v6", "FLUX", "FLUX.1",
            "Stable Diffusion", "Stable Diffusion 1.5/2.1",
            "SDXL (Stable Diffusion XL)", "SDXL / FLUX",
            "DALL-E", "DALL-E 3", "Unknown Generator", "Stable Diffusion / Midjourney"
        }
        result = detect_likely_generator(random_image)
        assert result in valid

    def test_handles_small_image(self):
        from src.predict import detect_likely_generator
        small = np.zeros((16, 16, 3), dtype=np.uint8)
        result = detect_likely_generator(small)
        assert isinstance(result, str)


# ─── Explanation Generator ─────────────────────────────────────────────────────
class TestExplanationGenerator:
    def test_fake_explanation_not_empty(self, random_image):
        from src.predict import generate_explanation
        result = {
            "prediction": "FAKE",
            "confidence": 95.0,
        }
        reasons = generate_explanation(result, random_image)
        assert isinstance(reasons, list)
        assert len(reasons) >= 1

    def test_real_explanation_not_empty(self, random_image):
        from src.predict import generate_explanation
        result = {
            "prediction": "REAL",
            "confidence": 88.0,
        }
        reasons = generate_explanation(result, random_image)
        assert isinstance(reasons, list)
        assert len(reasons) >= 1

    def test_max_4_reasons(self, random_image):
        from src.predict import generate_explanation
        result = {"prediction": "FAKE", "confidence": 99.0}
        reasons = generate_explanation(result, random_image)
        assert len(reasons) <= 4


# ─── Grad-CAM Overlay ─────────────────────────────────────────────────────────
class TestGradCAM:
    def test_overlay_output_shape(self, random_image):
        from src.predict import CNNPredictor
        heatmap = np.random.rand(7, 7).astype(np.float32)
        overlay = CNNPredictor._overlay_heatmap(random_image, heatmap)
        assert overlay.shape == random_image.shape
        assert overlay.dtype == np.uint8

    def test_overlay_value_range(self, random_image):
        from src.predict import CNNPredictor
        heatmap = np.ones((7, 7), dtype=np.float32)
        overlay = CNNPredictor._overlay_heatmap(random_image, heatmap)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_build_result_structure(self):
        from src.predict import CNNPredictor
        probs  = np.array([0.1, 0.9])
        result = CNNPredictor._build_result(
            pred_idx=1, probs=probs, confidence=90.0
        )
        assert result["prediction"]  == "FAKE"
        assert result["label_index"] == 1
        assert result["confidence"]  == 90.0
        assert "probabilities" in result
        assert "REAL" in result["probabilities"]
        assert "FAKE" in result["probabilities"]

    def test_build_result_real_class(self):
        from src.predict import CNNPredictor
        probs  = np.array([0.85, 0.15])
        result = CNNPredictor._build_result(
            pred_idx=0, probs=probs, confidence=85.0
        )
        assert result["prediction"]  == "REAL"
        assert result["label_index"] == 0

    def test_clip_threshold_adjustment_maps_threshold_to_half(self):
        from src.predict import ClipPredictor

        adjusted = ClipPredictor._threshold_adjusted_probability(0.4, 0.4)

        assert adjusted == pytest.approx(0.5)

    def test_clip_threshold_adjustment_preserves_decision_order(self):
        from src.predict import ClipPredictor

        assert ClipPredictor._threshold_adjusted_probability(0.7, 0.4) > 0.5
        assert ClipPredictor._threshold_adjusted_probability(0.2, 0.4) < 0.5

    def test_clip_routes_boundary_scores_to_review(self):
        from src.predict import ClipPredictor

        assert ClipPredictor._review_recommended(0.50, 0.01)
        assert not ClipPredictor._review_recommended(0.95, 0.01)
        assert not ClipPredictor._review_recommended(0.03, 0.01)
        assert ClipPredictor._review_recommended(0.08, 0.01)

    def test_clip_routes_view_disagreement_to_review(self):
        from src.predict import ClipPredictor

        assert ClipPredictor._review_recommended(0.95, 0.30)

    def test_classical_probability_uses_estimator_class_order(self, monkeypatch, pil_image):
        from src.predict import ClassicalPredictor

        class ReversedClassPipeline:
            classes_ = np.array([1, 0])

            def predict(self, features):
                return np.array([1])

            def predict_proba(self, features):
                return np.array([[0.8, 0.2]])

        predictor = ClassicalPredictor.__new__(ClassicalPredictor)
        predictor.pipeline = ReversedClassPipeline()
        monkeypatch.setattr(
            "src.predict.extract_features", lambda image: np.zeros(4, dtype=np.float32)
        )

        result = predictor.predict(pil_image)

        assert result["prediction"] == "FAKE"
        assert result["confidence"] == 80.0
        assert result["probabilities"] == {"REAL": 20.0, "FAKE": 80.0}


# ─── Utils ────────────────────────────────────────────────────────────────────
class TestUtils:
    def test_seed_everything(self):
        from src.utils import seed_everything
        seed_everything(42)
        import random
        a = random.random()
        seed_everything(42)
        b = random.random()
        assert a == b

    def test_get_device_returns_device(self):
        import torch
        from src.utils import get_device
        device = get_device()
        assert isinstance(device, torch.device)

    def test_logger_created(self):
        from src.utils import get_logger
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
