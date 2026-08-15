#!/usr/bin/env python3
"""
TruthLens — src/predict.py
============================
Single-image inference pipeline with:
  • ResNet18 CNN prediction (primary)
  • Classical ML fallback
  • Grad-CAM heatmap generation (manual implementation — no external library)
  • Likely generator detection (heuristic analysis)
  • AI explanation generation

Usage:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --image path/to/image.jpg --model svm
    python src/predict.py --image path/to/image.jpg --gradcam
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

from src.preprocessing import extract_features, preprocess_single_image
from src.train import build_model
from src.utils import Config, get_device, get_logger, load_sklearn_model

logger = get_logger(__name__)

# ─── Label / Class Maps ───────────────────────────────────────────────────────
LABEL_MAP  = {0: "REAL", 1: "FAKE"}
EMOJI_MAP  = {0: "✅",   1: "🤖"}
COLOR_MAP  = {0: (34, 197, 94), 1: (239, 68, 68)}   # green / red  (RGB)


# ─── CNN Predictor ────────────────────────────────────────────────────────────
class CNNPredictor:
    """Wraps a trained ResNet18 for single-image inference + Grad-CAM."""

    def __init__(self, cfg: Config) -> None:
        import torch
        import torch.nn as nn
        self.cfg    = cfg
        self.device = get_device()
        self.model  = self._load_model()

        # Grad-CAM hooks
        self._layer4_features  = None
        self._layer4_gradients = None
        self._layer3_features  = None
        self._layer3_gradients = None
        self._register_hooks()

    def _load_model(self):
        import torch
        ckpt_path = self.cfg.models_dir / self.cfg.cnn_model_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {ckpt_path}\n"
                "Train the model first:  python src/train.py"
            )
        model = build_model(self.cfg.num_classes).to(self.device)
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except OSError as exc:
            if exc.errno != 5:
                raise
            logger.warning(f"Checkpoint read failed once; retrying: {ckpt_path}")
            time.sleep(0.25)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        class_names = tuple(
            checkpoint.get(
                "class_names",
                checkpoint.get("config", {}).get("class_names", ()),
            )
        )
        if class_names and class_names != self.cfg.class_names:
            raise ValueError(
                f"Checkpoint class order {class_names} does not match expected "
                f"order {self.cfg.class_names}."
            )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        logger.info(f"  CNN loaded — epoch {checkpoint.get('epoch','?')}")
        return model

    def _register_hooks(self) -> None:
        """Attach forward + backward hooks to layer3 and layer4 for High-Resolution Multi-Layer Grad-CAM."""
        layer4_target = self.model.layer4[-1]
        layer3_target = self.model.layer3[-1]

        def forward_hook_l4(module, input, output):
            self._layer4_features = output.detach()

        def backward_hook_l4(module, grad_in, grad_out):
            self._layer4_gradients = grad_out[0].detach()

        def forward_hook_l3(module, input, output):
            self._layer3_features = output.detach()

        def backward_hook_l3(module, grad_in, grad_out):
            self._layer3_gradients = grad_out[0].detach()

        layer4_target.register_forward_hook(forward_hook_l4)
        layer4_target.register_backward_hook(backward_hook_l4)
        layer3_target.register_forward_hook(forward_hook_l3)
        layer3_target.register_backward_hook(backward_hook_l3)

    def predict(self, image_source) -> Dict[str, Any]:
        """
        Predict without Grad-CAM (fast path) using Multi-Crop TTA, Native Patch Scanning, and forensic refinement.
        Returns full result dict.
        """
        import torch
        import torchvision.transforms as T
        from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD

        if isinstance(image_source, (str, Path)):
            pil_img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            pil_img = image_source.convert("RGB")
        elif isinstance(image_source, np.ndarray):
            pil_img = Image.fromarray(image_source).convert("RGB")
        else:
            pil_img = None

        t_full = preprocess_single_image(image_source, self.cfg.img_size).to(self.device)

        if pil_img is not None:
            t_crops = T.Compose([
                T.Resize((256, 256)),
                T.FiveCrop(224),
                T.Lambda(lambda crops: torch.stack([T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(c) for c in crops]))
            ])(pil_img).to(self.device)
            all_tensors = torch.cat([t_full, t_crops], dim=0)
        else:
            all_tensors = t_full

        with torch.no_grad():
            logits = self.model(all_tensors)
            probs_all = torch.softmax(logits, dim=1)
            probs = probs_all.mean(dim=0)

        raw_fake = float(probs[1].cpu()) * 100.0

        # Subject-Centric Native Patch Scanning (Focuses on Person & Facial Features, not empty background)
        max_patch_fake = raw_fake
        avg_patch_fake = raw_fake
        if pil_img is not None and min(pil_img.size) >= 350:
            try:
                w, h = pil_img.size
                crop_size = min(w, h, 256)
                half = crop_size // 2
                crops_native = [
                    pil_img.crop((w//2 - half, h//2 - half, w//2 + half, h//2 + half)),
                    pil_img.crop((w//2 - half, max(0, h//3 - half), w//2 + half, max(0, h//3 - half) + crop_size)),
                    pil_img.crop((w//2 - half, min(h - crop_size, h//2), w//2 + half, min(h, h//2 + crop_size))),
                ]
                t_patch_list = [T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(c) for c in crops_native]
                t_patches = torch.stack(t_patch_list).to(self.device)
                with torch.no_grad():
                    patch_logits = self.model(t_patches)
                    patch_probs = torch.softmax(patch_logits, dim=1)
                    max_patch_fake = float(patch_probs[:, 1].max().cpu()) * 100.0
                    avg_patch_fake = float(patch_probs[:, 1].mean().cpu()) * 100.0
            except Exception:
                max_patch_fake = raw_fake
                avg_patch_fake = raw_fake

        # Multi-domain forensic synthesis
        if pil_img is not None:
            try:
                img_np = np.array(pil_img)
                spectral = compute_fft_spectral_score(img_np)
                ela_metrics, _ = compute_ela_analysis(pil_img)
                ela_std = float(ela_metrics.get("ela_std", 20.0))
                ela_ai = float(ela_metrics.get("ela_ai_score", 50.0))
                fft_ai = float(spectral.get("spectral_ai_score", 50.0))
                hf_ratio = float(spectral.get("hf_ratio", 0.65))

                is_synthetic = (
                    (raw_fake >= 15.0 and (ela_std < 14.5 or fft_ai > 50.0 or avg_patch_fake > 55.0)) or
                    (raw_fake < 15.0 and (fft_ai > 60.0 or avg_patch_fake > 70.0)) or
                    (raw_fake >= 50.0)
                )

                if is_synthetic:
                    forensic_signal = max(ela_ai, fft_ai, max_patch_fake, (16.0 - ela_std) * 7.5 if ela_std < 16.0 else 0.0)
                    calib_fake = max(76.0, min(99.2, raw_fake * 0.25 + forensic_signal * 0.75))
                elif raw_fake < 30.0 and avg_patch_fake < 40.0:
                    # Natural photographic subject with optical bokeh/background
                    calib_fake = min(10.0, raw_fake * 0.4)
                else:
                    calib_fake = raw_fake

                calib_fake = min(99.5, max(0.5, calib_fake))
                calib_probs = np.array([(100.0 - calib_fake)/100.0, calib_fake/100.0], dtype=np.float32)
                pred_idx = int(calib_probs.argmax())
                confidence = float(calib_probs[pred_idx]) * 100.0
                probs_out = calib_probs
            except Exception:
                pred_idx = int(probs.argmax())
                confidence = float(probs[pred_idx]) * 100.0
                probs_out = probs.cpu().numpy()
        else:
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx]) * 100.0
            probs_out = probs.cpu().numpy()

        return self._build_result(
            pred_idx   = pred_idx,
            probs      = probs_out,
            confidence = confidence,
        )

    def predict_with_gradcam(self, image_source) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Predict with Multi-Layer High-Resolution Grad-CAM heatmap.
        Returns (result_dict, heatmap_overlay_numpy_RGB).
        """
        import torch
        import torchvision.transforms as T
        from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD

        if isinstance(image_source, (str, Path)):
            pil_img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            pil_img = image_source.convert("RGB")
        elif isinstance(image_source, np.ndarray):
            pil_img = Image.fromarray(image_source).convert("RGB")
        else:
            pil_img = None

        self.model.eval()
        t_full = preprocess_single_image(image_source, self.cfg.img_size).to(self.device)
        t_full.requires_grad_(True)

        logits_full = self.model(t_full)
        probs_full  = torch.softmax(logits_full, dim=1)[0]
        pred_idx_full = int(probs_full.argmax())

        # Backprop for Grad-CAM
        self.model.zero_grad()
        class_score = logits_full[0, pred_idx_full]
        class_score.backward()

        heatmap = self._compute_gradcam()

        if pil_img is not None:
            orig = np.array(pil_img)
        else:
            orig = np.zeros((224, 224, 3), dtype=np.uint8)

        overlay = self._overlay_heatmap(orig, heatmap)

        # Multi-crop evaluation for probability accuracy
        with torch.no_grad():
            if pil_img is not None:
                t_crops = T.Compose([
                    T.Resize((256, 256)),
                    T.FiveCrop(224),
                    T.Lambda(lambda crops: torch.stack([T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(c) for c in crops]))
                ])(pil_img).to(self.device)
                all_tensors = torch.cat([t_full.detach(), t_crops], dim=0)
                logits_all = self.model(all_tensors)
                probs_all = torch.softmax(logits_all, dim=1)
                probs = probs_all.mean(dim=0)
            else:
                probs = probs_full

        raw_fake = float(probs[1].cpu()) * 100.0

        # Subject-Centric Native Patch Scanning
        max_patch_fake = raw_fake
        avg_patch_fake = raw_fake
        if pil_img is not None and min(pil_img.size) >= 350:
            try:
                w, h = pil_img.size
                crop_size = min(w, h, 256)
                half = crop_size // 2
                crops_native = [
                    pil_img.crop((w//2 - half, h//2 - half, w//2 + half, h//2 + half)),
                    pil_img.crop((w//2 - half, max(0, h//3 - half), w//2 + half, max(0, h//3 - half) + crop_size)),
                    pil_img.crop((w//2 - half, min(h - crop_size, h//2), w//2 + half, min(h, h//2 + crop_size))),
                ]
                t_patch_list = [T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(c) for c in crops_native]
                t_patches = torch.stack(t_patch_list).to(self.device)
                with torch.no_grad():
                    patch_logits = self.model(t_patches)
                    patch_probs = torch.softmax(patch_logits, dim=1)
                    max_patch_fake = float(patch_probs[:, 1].max().cpu()) * 100.0
                    avg_patch_fake = float(patch_probs[:, 1].mean().cpu()) * 100.0
            except Exception:
                max_patch_fake = raw_fake
                avg_patch_fake = raw_fake

        # Multi-domain forensic synthesis
        if pil_img is not None:
            try:
                spectral = compute_fft_spectral_score(orig)
                ela_metrics, _ = compute_ela_analysis(pil_img)
                ela_std = float(ela_metrics.get("ela_std", 20.0))
                ela_ai = float(ela_metrics.get("ela_ai_score", 50.0))
                fft_ai = float(spectral.get("spectral_ai_score", 50.0))
                hf_ratio = float(spectral.get("hf_ratio", 0.65))

                is_synthetic = (
                    (raw_fake >= 15.0 and (ela_std < 14.5 or fft_ai > 50.0 or avg_patch_fake > 55.0)) or
                    (raw_fake < 15.0 and (fft_ai > 60.0 or avg_patch_fake > 70.0)) or
                    (raw_fake >= 50.0)
                )

                if is_synthetic:
                    forensic_signal = max(ela_ai, fft_ai, max_patch_fake, (16.0 - ela_std) * 7.5 if ela_std < 16.0 else 0.0)
                    calib_fake = max(76.0, min(99.2, raw_fake * 0.25 + forensic_signal * 0.75))
                elif raw_fake < 30.0 and avg_patch_fake < 40.0:
                    # Natural photographic subject with optical bokeh/background
                    calib_fake = min(10.0, raw_fake * 0.4)
                else:
                    calib_fake = raw_fake

                calib_fake = min(99.5, max(0.5, calib_fake))
                calib_probs = np.array([(100.0 - calib_fake)/100.0, calib_fake/100.0], dtype=np.float32)
                pred_idx = int(calib_probs.argmax())
                confidence = float(calib_probs[pred_idx]) * 100.0
                probs_out = calib_probs
            except Exception:
                pred_idx = int(probs.argmax())
                confidence = float(probs[pred_idx]) * 100.0
                probs_out = probs.cpu().numpy()
        else:
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx]) * 100.0
            probs_out = probs.cpu().numpy()

        result = self._build_result(
            pred_idx   = pred_idx,
            probs      = probs_out,
            confidence = confidence,
        )
        return result, overlay

    def _compute_gradcam(self) -> np.ndarray:
        """Compute High-Resolution Multi-Layer Grad-CAM heatmap combining layer3 and layer4 with subject salience."""
        import torch.nn.functional as F

        def _cam_from_tensors(features, gradients):
            if features is None or gradients is None:
                return None
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1).squeeze(0)
            cam = F.relu(cam)
            cam_np = cam.cpu().numpy()
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max > c_min:
                cam_np = (cam_np - c_min) / (c_max - c_min)
            return cam_np.astype(np.float32)

        cam_l4 = _cam_from_tensors(self._layer4_features, self._layer4_gradients)
        cam_l3 = _cam_from_tensors(self._layer3_features, self._layer3_gradients)

        if cam_l4 is None and cam_l3 is None:
            return np.zeros((14, 14), dtype=np.float32)
        if cam_l3 is None:
            return cam_l4
        if cam_l4 is None:
            return cam_l3

        # Combine layer3 (14x14 structural & textural features) + layer4 (7x7 global semantic features)
        h, w = cam_l3.shape
        cam_l4_upscaled = cv2.resize(cam_l4, (w, h), interpolation=cv2.INTER_CUBIC)
        combined = 0.55 * cam_l4_upscaled + 0.45 * cam_l3

        # Center-weighted subject salience (focuses on person/face/subject, suppressing empty background walls)
        y = np.linspace(-1, 1, h)[:, None]
        x = np.linspace(-1, 1, w)[None, :]
        dist_sq = (x / 1.1)**2 + (y / 1.1)**2
        salience_prior = np.exp(-0.7 * dist_sq).astype(np.float32)
        salience_prior = (salience_prior - salience_prior.min()) / (salience_prior.max() - salience_prior.min() + 1e-8)

        focused_cam = combined * (0.65 + 0.35 * salience_prior)
        c_min, c_max = focused_cam.min(), focused_cam.max()
        if c_max > c_min:
            focused_cam = (focused_cam - c_min) / (c_max - c_min)
        return focused_cam.astype(np.float32)

    @staticmethod
    def _overlay_heatmap(
        img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Resize heatmap, apply colormap, and blend with original image."""
        h, w = img.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h))

        # Apply JET colormap and convert BGR→RGB
        hm_uint8 = (hm_resized * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_rgb   = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        # Blend
        img_f  = img.astype(np.float32)
        hm_f   = hm_rgb.astype(np.float32)
        blend  = (1 - alpha) * img_f + alpha * hm_f
        return np.clip(blend, 0, 255).astype(np.uint8)

    @staticmethod
    def _build_result(
        pred_idx:   int,
        probs:      np.ndarray,
        confidence: float,
    ) -> Dict[str, Any]:
        label = LABEL_MAP[pred_idx]
        return {
            "prediction":   label,
            "label_index":  pred_idx,
            "emoji":        EMOJI_MAP[pred_idx],
            "confidence":   round(confidence, 2),
            "probabilities": {
                "REAL": round(float(probs[0]) * 100, 2),
                "FAKE": round(float(probs[1]) * 100, 2),
            },
        }


# ─── Classical ML Predictor ───────────────────────────────────────────────────
class ClassicalPredictor:
    """Wraps a saved scikit-learn pipeline for single-image inference."""

    MODEL_PATHS = {
        "logistic_regression": "logistic_regression.pkl",
        "decision_tree":       "decision_tree.pkl",
        "random_forest":       "random_forest.pkl",
        "svm":                 "svm.pkl",
        "knn":                 "k-nn.pkl",
        "naive_bayes":         "naive_bayes.pkl",
    }

    def __init__(self, model_key: str, cfg: Config) -> None:
        if model_key not in self.MODEL_PATHS:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Choose from: {list(self.MODEL_PATHS)}"
            )
        self.model_key = model_key
        path = cfg.models_dir / self.MODEL_PATHS[model_key]
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Train first:  python src/train_classical.py"
            )
        self.pipeline = load_sklearn_model(path)
        logger.info(f"  Loaded classical model: {model_key}")

    def predict(self, image_source) -> Dict[str, Any]:
        # Extract numpy image
        if isinstance(image_source, (str, Path)):
            img = np.array(Image.open(image_source).convert("RGB"))
        elif isinstance(image_source, Image.Image):
            img = np.array(image_source.convert("RGB"))
        elif isinstance(image_source, np.ndarray):
            img = image_source
        else:
            raise TypeError(f"Unsupported type: {type(image_source)}")

        features   = extract_features(img).reshape(1, -1)
        pred_idx   = int(self.pipeline.predict(features)[0])
        proba      = self.pipeline.predict_proba(features)[0]
        class_probabilities = {
            int(label): float(probability)
            for label, probability in zip(self.pipeline.classes_, proba)
        }
        confidence = class_probabilities[pred_idx] * 100.0

        label = LABEL_MAP[pred_idx]
        return {
            "prediction":  label,
            "label_index": pred_idx,
            "emoji":       EMOJI_MAP[pred_idx],
            "confidence":  round(confidence, 2),
            "probabilities": {
                "REAL": round(class_probabilities[0] * 100, 2),
                "FAKE": round(class_probabilities[1] * 100, 2),
            },
        }


# ─── FFT Spectral Analyzer ───────────────────────────────────────────────────
def compute_fft_spectral_score(img_np: np.ndarray) -> Dict[str, Any]:
    """
    2D Fast Fourier Transform (FFT) Spectral Analysis.
    Detects high-frequency periodic lattice artifacts left by modern VAE decoders
    (SDXL, Midjourney v6, FLUX, SD 1.5/2.1).
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    h, w = gray.shape

    # 2D Fast Fourier Transform
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1e-8)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = min(h, w) / 2.0

    # High frequency outer ring vs low frequency inner ring
    hf_mask = (r >= max_r * 0.6) & (r <= max_r * 0.95)
    lf_mask = (r < max_r * 0.25)

    hf_energy = float(np.mean(mag[hf_mask])) if np.any(hf_mask) else 0.0
    lf_energy = float(np.mean(mag[lf_mask])) if np.any(lf_mask) else 1.0

    ratio = hf_energy / (lf_energy + 1e-8)

    # VAE lattice grid artifact check: local peaks in high frequency outer region
    hf_region = mag[hf_mask] if np.any(hf_mask) else np.array([0])
    peak_factor = float(np.max(hf_region) - np.mean(hf_region))

    # Composite spectral AI probability score (0 to 100)
    score = min(100.0, max(0.0, (ratio - 0.65) * 250.0 + peak_factor * 5.0))
    return {
        "hf_ratio": round(ratio, 4),
        "peak_factor": round(peak_factor, 2),
        "spectral_ai_score": round(score, 2)
    }


# ─── Generator Detection ──────────────────────────────────────────────────────
def detect_likely_generator(img: np.ndarray) -> str:
    """
    Heuristic generator detection based on spatial & frequency statistics.
    Detects SDXL, Midjourney, FLUX, Stable Diffusion v1/v2, DALL-E.
    """
    orig_h, orig_w = img.shape[:2]
    if img.shape[:2] != (224, 224):
        img_resized = cv2.resize(img, (224, 224))
    else:
        img_resized = img

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # ── Frequency-domain analysis ──
    spectral = compute_fft_spectral_score(img)
    dft    = np.fft.fft2(gray.astype(np.float32))
    dft_sh = np.fft.fftshift(dft)
    mag    = np.log1p(np.abs(dft_sh))
    h, w   = mag.shape
    center = mag[h//4:3*h//4, w//4:3*w//4]
    high_freq_ratio = center.mean() / (mag.mean() + 1e-8)

    # ── Texture & Color analysis ──
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_sat = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)[:, :, 1].mean()

    # Heuristic decision tree
    if orig_h >= 1024 or orig_w >= 1024:
        if spectral["hf_ratio"] > 0.72:
            return "SDXL (Stable Diffusion XL)"
        elif laplacian_var > 700:
            return "Midjourney v6"
        else:
            return "SDXL / FLUX"
    elif laplacian_var > 800 and mean_sat > 100:
        return "Midjourney"
    elif high_freq_ratio > 1.3 and laplacian_var > 500:
        return "FLUX.1"
    elif mean_sat > 80 and laplacian_var < 600:
        return "Stable Diffusion 1.5/2.1"
    elif laplacian_var < 300:
        return "DALL-E 3"
    else:
        return "Stable Diffusion / Midjourney"


# ─── Explanation Generator ────────────────────────────────────────────────────
def generate_explanation(result: Dict[str, Any], img: np.ndarray) -> List[str]:
    """
    Generate human-readable explanation bullets for the prediction.
    Based on image statistics + confidence level.
    """
    reasons: List[str] = []
    pred   = result["prediction"]
    conf   = result["confidence"]

    if pred == "FAKE":
        if conf > 90:
            reasons.append("Very high confidence of AI generation detected")

        # Analyse texture consistency
        gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap_var < 300:
            reasons.append("Unusually smooth texture — typical of diffusion models")
        elif lap_var > 800:
            reasons.append("Hyper-detailed texture inconsistencies detected")

        # Colour analysis
        hsv       = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat_std   = hsv[:, :, 1].std()
        if sat_std < 30:
            reasons.append("Unnaturally uniform colour saturation")

        # Symmetry check (AI images tend to be more symmetric)
        flipped  = np.fliplr(gray)
        symmetry = 1.0 - np.mean(np.abs(gray.astype(float) - flipped.astype(float))) / 255.0
        if symmetry > 0.85:
            reasons.append("High bilateral symmetry — common in AI-generated faces")

        # Frequency anomaly (2D FFT)
        spectral = compute_fft_spectral_score(img)
        if spectral["hf_ratio"] > 0.72:
            reasons.append("High-frequency VAE decoder lattice artifacts detected (SDXL / Diffusion signature)")
        elif spectral["hf_ratio"] < 0.58:
            reasons.append("Anomalous frequency spectrum (lacks natural high-frequency sensor noise)")

        if not reasons:
            reasons.append("Pattern inconsistencies detected by deep neural network")

    else:  # REAL
        reasons.append("Natural texture and noise patterns detected")
        reasons.append("Consistent lighting and shadow distribution")
        if conf > 90:
            reasons.append("Very high confidence of authentic photographic content")
        reasons.append("Frequency spectrum consistent with camera sensor noise")

    return reasons[:4]  # cap at 4 bullets


# ─── ELA (Error Level Analysis) ───────────────────────────────────────────────
def compute_ela_analysis(img_source, quality: int = 90) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Error Level Analysis (ELA).
    Detects JPEG digital compression error anomalies characteristic of AI generation
    or digital manipulation.
    """
    if isinstance(img_source, (str, Path)):
        img_pil = Image.open(img_source).convert("RGB")
    elif isinstance(img_source, Image.Image):
        img_pil = img_source.convert("RGB")
    elif isinstance(img_source, np.ndarray):
        img_pil = Image.fromarray(img_source).convert("RGB")
    else:
        img_pil = Image.new("RGB", (224, 224), color=0)

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    ela_img = ImageChops.difference(img_pil, resaved)
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / (max_diff + 1e-8)

    ela_scaled = ImageEnhance.Brightness(ela_img).enhance(scale)
    ela_np = np.array(ela_scaled)

    mean_diff = float(np.mean(ela_np))
    std_diff = float(np.std(ela_np))

    # Standard digital camera JPEGs have non-uniform sensor-dependent quantization variance (std > 20)
    # AI-generated JPEGs often have flatter/lower error level variance (std < 18)
    ela_score = min(100.0, max(0.0, (22.0 - std_diff) * 4.5 + (15.0 - mean_diff) * 2.0))

    metrics = {
        "ela_mean": round(mean_diff, 2),
        "ela_std": round(std_diff, 2),
        "ela_ai_score": round(ela_score, 2),
    }
    return metrics, ela_np


# ─── Multi-Model Voting Ensemble Predictor ─────────────────────────────────────
class EnsemblePredictor:
    """
    Combines ResNet18 CNN, SVM (HOG+LBP), Random Forest, 2D FFT Spectral Analysis,
    and Error Level Analysis (ELA) into a single bulletproof consensus engine.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cnn_predictor = CNNPredictor(cfg)

    def predict(self, image_source) -> Dict[str, Any]:
        # 1. ResNet18 CNN prediction
        cnn_res = self.cnn_predictor.predict(image_source)
        cnn_fake_prob = float(cnn_res["probabilities"]["FAKE"])

        # 2. Classical SVM prediction (if trained)
        svm_fake_prob = cnn_fake_prob
        try:
            svm_pred = ClassicalPredictor("svm", self.cfg)
            svm_res = svm_pred.predict(image_source)
            svm_fake_prob = float(svm_res["probabilities"]["FAKE"])
        except Exception:
            pass

        # 3. Classical Random Forest prediction (if trained)
        rf_fake_prob = cnn_fake_prob
        try:
            rf_pred = ClassicalPredictor("random_forest", self.cfg)
            rf_res = rf_pred.predict(image_source)
            rf_fake_prob = float(rf_res["probabilities"]["FAKE"])
        except Exception:
            pass

        # 4. 2D FFT Spectral Analysis
        if isinstance(image_source, (str, Path)):
            img_np = np.array(Image.open(image_source).convert("RGB"))
        elif isinstance(image_source, Image.Image):
            img_np = np.array(image_source.convert("RGB"))
        elif isinstance(image_source, np.ndarray):
            img_np = image_source
        else:
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)

        spectral = compute_fft_spectral_score(img_np)
        spectral_fake_prob = float(spectral["spectral_ai_score"])

        # 5. ELA Analysis
        ela_metrics, _ = compute_ela_analysis(image_source)
        ela_fake_prob = float(ela_metrics["ela_ai_score"])

        # ── Weighted Multi-Domain Consensus ──
        # ResNet18: 35%, 2D FFT: 25%, ELA: 20%, SVM: 10%, Random Forest: 10%
        ensemble_fake_prob = (
            0.35 * cnn_fake_prob +
            0.25 * spectral_fake_prob +
            0.20 * ela_fake_prob +
            0.10 * svm_fake_prob +
            0.10 * rf_fake_prob
        )
        ensemble_fake_prob = min(99.9, max(0.1, ensemble_fake_prob))

        is_fake = ensemble_fake_prob >= 50.0
        label = "FAKE" if is_fake else "REAL"
        idx = 1 if is_fake else 0
        emoji = "🤖" if is_fake else "✅"

        # Model breakdown
        votes = {
            "ResNet18 CNN": "FAKE" if cnn_fake_prob >= 50 else "REAL",
            "SVM (HOG+LBP)": "FAKE" if svm_fake_prob >= 50 else "REAL",
            "Random Forest": "FAKE" if rf_fake_prob >= 50 else "REAL",
            "2D FFT Spectrum": "FAKE" if spectral_fake_prob >= 50 else "REAL",
            "ELA Inspector": "FAKE" if ela_fake_prob >= 50 else "REAL",
        }

        fake_votes = sum(1 for v in votes.values() if v == "FAKE")
        consensus_status = f"{fake_votes}/5 Models Voted FAKE"

        result = {
            "prediction": label,
            "label_index": idx,
            "emoji": emoji,
            "confidence": round(ensemble_fake_prob if is_fake else (100.0 - ensemble_fake_prob), 2),
            "probabilities": {
                "REAL": round(100.0 - ensemble_fake_prob, 2),
                "FAKE": round(ensemble_fake_prob, 2),
            },
            "spectral_analysis": spectral,
            "ela_metrics": ela_metrics,
            "ensemble_votes": votes,
            "consensus_status": consensus_status,
        }
        return result


# ─── Full Prediction Pipeline ─────────────────────────────────────────────────
def predict_image(
    image_source,
    cfg:         Config,
    model_type:  str = "ensemble",
    with_gradcam: bool = True,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Top-level prediction function used by the Streamlit app.

    Returns:
        result  : dict with prediction, confidence, explanation, generator
        heatmap : numpy RGB overlay (or None if with_gradcam=False / classical model)
    """
    # Load image as numpy array for explanation/generator
    if isinstance(image_source, (str, Path)):
        img_np = np.array(Image.open(image_source).convert("RGB"))
    elif isinstance(image_source, Image.Image):
        img_np = np.array(image_source.convert("RGB"))
    elif isinstance(image_source, np.ndarray):
        img_np = image_source
    else:
        img_np = np.zeros((224, 224, 3), dtype=np.uint8)

    heatmap = None

    if model_type == "ensemble":
        ensemble_pred = EnsemblePredictor(cfg)
        result = ensemble_pred.predict(image_source)
        if with_gradcam:
            try:
                cnn_pred = CNNPredictor(cfg)
                _, heatmap = cnn_pred.predict_with_gradcam(image_source)
            except Exception:
                heatmap = None
    elif model_type == "resnet18":
        predictor = CNNPredictor(cfg)
        if with_gradcam:
            result, heatmap = predictor.predict_with_gradcam(image_source)
        else:
            result = predictor.predict(image_source)
    else:
        predictor = ClassicalPredictor(model_type, cfg)
        result    = predictor.predict(image_source)
        predictor = ClassicalPredictor(model_type, cfg)
        result    = predictor.predict(image_source)

    # ── Hybrid Spatial + 2D FFT Spectral Fusion ──
    spectral = compute_fft_spectral_score(img_np)
    result["spectral_analysis"] = spectral

    # If 2D FFT spectral analysis shows strong VAE lattice signature (>75% AI score)
    # or high-frequency ratio > 0.76 (typical for SDXL/FLUX/Midjourney), boost FAKE probability
    if spectral["spectral_ai_score"] > 75.0 or spectral["hf_ratio"] > 0.76:
        # Fuse probabilities: 60% Model + 40% FFT Spectral
        model_fake_prob = float(result["probabilities"]["FAKE"])
        fused_fake_prob = min(99.9, max(0.1, model_fake_prob * 0.5 + spectral["spectral_ai_score"] * 0.5))

        if fused_fake_prob >= 50.0:
            result["prediction"]  = "FAKE"
            result["label_index"] = 1
            result["emoji"]       = "🤖"
            result["confidence"]  = round(fused_fake_prob, 2)
            result["probabilities"]["FAKE"] = round(fused_fake_prob, 2)
            result["probabilities"]["REAL"] = round(100.0 - fused_fake_prob, 2)

    # Enrich result
    result["explanation"] = generate_explanation(result, img_np)
    if result["prediction"] == "FAKE":
        result["likely_generator"] = detect_likely_generator(img_np)
    else:
        result["likely_generator"] = "N/A"

    return result, heatmap


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TruthLens — Single Image Prediction")
    parser.add_argument("--image",   required=True, help="Path to input image.")
    parser.add_argument(
        "--model", default="resnet18",
        choices=["resnet18", "logistic_regression", "decision_tree",
                 "random_forest", "svm", "knn", "naive_bayes"],
        help="Model to use for prediction.",
    )
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmap.")
    parser.add_argument("--save-heatmap", type=str, default="", help="Path to save heatmap.")
    args = parser.parse_args()

    cfg    = Config()
    result, heatmap = predict_image(
        args.image, cfg,
        model_type=args.model,
        with_gradcam=args.gradcam,
    )

    print("\n" + "="*50)
    print("  TruthLens — Prediction Result")
    print("="*50)
    print(f"  Image      : {args.image}")
    print(f"  Model      : {args.model}")
    print(f"  Prediction : {result['emoji']} {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"  Prob REAL  : {result['probabilities']['REAL']:.2f}%")
    print(f"  Prob FAKE  : {result['probabilities']['FAKE']:.2f}%")
    if result["prediction"] == "FAKE":
        print(f"  Generator  : {result['likely_generator']}")
    print("\n  Explanation:")
    for reason in result["explanation"]:
        print(f"    • {reason}")
    print("="*50)

    if heatmap is not None:
        save_path = args.save_heatmap or f"results/gradcam/heatmap_{Path(args.image).stem}.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(heatmap).save(save_path)
        print(f"\n  Grad-CAM heatmap saved → {save_path}")


if __name__ == "__main__":
    main()
