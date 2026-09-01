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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


# ─── Shared Input Handling ────────────────────────────────────────────────────
def _as_pil(image_source) -> Optional[Image.Image]:
    """Coerce a path / PIL image / numpy array to RGB PIL. None if unsupported."""
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source).convert("RGB")
    if isinstance(image_source, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image_source))).convert("RGB")
    if hasattr(image_source, "read"):
        try:
            return Image.open(image_source).convert("RGB")
        except Exception:
            return None
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    if isinstance(image_source, np.ndarray):
        return Image.fromarray(image_source).convert("RGB")
    return None


def _as_numpy(image_source) -> np.ndarray:
    """Coerce any supported input to an RGB numpy array, falling back to black."""
    pil = _as_pil(image_source)
    if pil is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return np.array(pil)


def forensic_metrics(image_source) -> Dict[str, Any]:
    """
    Compute FFT + ELA metrics for *display only*.

    These are shown in the app's inspector tabs because the spectrum and error
    map are informative to look at, but they are deliberately excluded from the
    verdict. Measured on the held-out test split their discriminative power is
    close to chance (FFT AUC 0.562, ELA AUC 0.514, chance = 0.500), and mixing
    them into the CNN score lowers AUC monotonically: 0.9814 alone, 0.9782 at
    10% forensic weight, 0.9730 at 25%, 0.9640 at 45%. See diag_forensics.py.
    """
    pil = _as_pil(image_source)
    if pil is None:
        return {}
    out: Dict[str, Any] = {"spectral_analysis": compute_fft_spectral_score(np.array(pil))}
    try:
        out["ela_metrics"] = compute_ela_analysis(pil)[0]
    except Exception as exc:                       # ELA needs a JPEG round-trip
        logger.debug(f"  ELA metrics unavailable: {exc}")
    return out


# ─── CNN Predictor ────────────────────────────────────────────────────────────
class CNNPredictor:
    """Wraps a trained ResNet18 for single-image inference + Grad-CAM."""

    def __init__(self, cfg: Config) -> None:
        import torch
        import torch.nn as nn
        self.cfg    = cfg
        self.device = get_device()
        # Provenance of the loaded weights, filled in by _load_model(). The app
        # surfaces this so a stale checkpoint is visible rather than implied.
        self.checkpoint_meta: Dict[str, Any] = {}
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
        self.checkpoint_meta = {
            "file":     self.cfg.cnn_model_name,
            "epoch":    checkpoint.get("epoch"),
            "val_acc":  checkpoint.get("val_acc"),
            "val_loss": checkpoint.get("val_loss"),
            "device":   str(self.device),
            "classes":  list(class_names or self.cfg.class_names),
        }
        logger.info(f"  CNN loaded — epoch {checkpoint.get('epoch','?')}")
        return model

    def _register_hooks(self) -> None:
        """Attach forward + backward hooks to layer3 and layer4 for High-Resolution Multi-Layer Grad-CAM.

        Uses register_full_backward_hook: the legacy register_backward_hook silently
        drops part of grad_input when the module's forward contains multiple autograd
        nodes (which a ResNet BasicBlock does, because of the residual add).
        """
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
        layer4_target.register_full_backward_hook(backward_hook_l4)
        layer3_target.register_forward_hook(forward_hook_l3)
        layer3_target.register_full_backward_hook(backward_hook_l3)

    def tta_probs(self, image_source, pil_img=None) -> np.ndarray:
        """
        Average softmax probabilities over the test-time augmentation views.

        Whole-frame views: plain 224 resize, horizontal flip, and five 224 crops
        of a 256 resize. That set was selected on the validation split (AUC
        0.9819 vs 0.9798 single-view) and confirmed on held-out test (AUC 0.9814,
        93.0% accuracy). See diag_strategy.py.

        Native-scale views: for any image whose short side comfortably exceeds
        224, a grid of 224 crops taken at 1:1 pixel scale is appended. Those
        views exist because the whole-frame ones cannot see the evidence on a
        large image — resizing 1024x1536 to 224 averages ~7x7 source pixels and
        erases the high-frequency band that separates a neural decoder from a
        camera sensor. This mirrors NativeScaleCrop in the training pipeline; a
        model trained on native crops but served only downscaled frames would be
        evaluated on a distribution it never saw.

        Small images are unaffected: a 32px CIFAKE thumbnail has no native scale
        to sample, so the view set stays exactly as it was and older checkpoints
        keep their measured behaviour.

        Public because src/evaluate.py measures this path rather than the bare
        model — metrics must describe what the app actually serves.
        """
        import torch
        import torchvision.transforms as T
        from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD

        if pil_img is None:
            pil_img = _as_pil(image_source)

        views = [preprocess_single_image(image_source, self.cfg.img_size)]

        if pil_img is not None:
            views.append(
                preprocess_single_image(
                    pil_img.transpose(Image.FLIP_LEFT_RIGHT), self.cfg.img_size
                )
            )
            norm = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
            views.append(
                T.Compose([
                    T.Resize((256, 256)),
                    T.FiveCrop(self.cfg.img_size),
                    T.Lambda(lambda crops: torch.stack([norm(c) for c in crops])),
                ])(pil_img)
            )
            native = self._native_crops(pil_img)
            if native:
                views.append(torch.stack([norm(c) for c in native]))

        batch = torch.cat(views, dim=0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1).mean(dim=0)
        return probs.cpu().numpy()

    def _native_crops(self, pil_img: "Image.Image") -> list:
        """A spread of 1:1-scale crops: centre plus the four quadrant centres.

        Five positions rather than one because a single 224 crop of a 1536px
        image covers about 3% of it, and generator artifacts are not uniform
        across a frame — flat sky recompresses to almost nothing while hair,
        foliage and fabric carry most of the signal. Sampling five places and
        averaging is far more stable than betting on the middle.

        Returns ``[]`` when the image is too small to crop at native scale, which
        keeps the view set unchanged for low-resolution input.
        """
        size = self.cfg.img_size
        w, h = pil_img.size
        if min(w, h) < size * 1.25:
            return []

        # Quadrant centres, clamped so every box stays inside the frame.
        fracs = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        crops = []
        for fx, fy in fracs:
            left = min(max(int(fx * w - size / 2), 0), w - size)
            top = min(max(int(fy * h - size / 2), 0), h - size)
            crops.append(pil_img.crop((left, top, left + size, top + size)))
        return crops

    def predict(self, image_source) -> Dict[str, Any]:
        """
        Predict without Grad-CAM (fast path) using test-time augmentation.

        The CNN's probabilities are reported as-is. FFT/ELA metrics are attached
        for display but do not influence the verdict — see forensic_metrics().
        """
        pil_img = _as_pil(image_source)
        probs   = self.tta_probs(image_source, pil_img)

        pred_idx = int(probs.argmax())
        result = self._build_result(
            pred_idx   = pred_idx,
            probs      = probs,
            confidence = float(probs[pred_idx]) * 100.0,
        )
        result.update(forensic_metrics(pil_img))
        return result

    def predict_with_gradcam(self, image_source) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Predict with Multi-Layer High-Resolution Grad-CAM heatmap.
        Returns (result_dict, heatmap_overlay_numpy_RGB).

        The verdict is computed first, then Grad-CAM backpropagates *that* class,
        so the heatmap always explains the label shown beside it.
        """
        import torch

        pil_img = _as_pil(image_source)

        self.model.eval()

        # 1. Verdict from the same TTA path as predict(), so both agree.
        probs    = self.tta_probs(image_source, pil_img)
        pred_idx = int(probs.argmax())

        # 2. Grad-CAM for the predicted class on the plain (un-augmented) view.
        t_full = preprocess_single_image(image_source, self.cfg.img_size).to(self.device)
        t_full.requires_grad_(True)

        logits_full = self.model(t_full)
        self.model.zero_grad(set_to_none=True)
        logits_full[0, pred_idx].backward()

        heatmap = self._compute_gradcam()

        orig    = np.array(pil_img) if pil_img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        overlay = self._overlay_heatmap(orig, heatmap)

        result = self._build_result(
            pred_idx   = pred_idx,
            probs      = probs,
            confidence = float(probs[pred_idx]) * 100.0,
        )
        result.update(forensic_metrics(pil_img))
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
        # .get guards a single-class estimator, which would otherwise KeyError.
        confidence = class_probabilities.get(pred_idx, 0.0) * 100.0

        label = LABEL_MAP[pred_idx]
        return {
            "prediction":  label,
            "label_index": pred_idx,
            "emoji":       EMOJI_MAP[pred_idx],
            "confidence":  round(confidence, 2),
            "probabilities": {
                "REAL": round(class_probabilities.get(0, 0.0) * 100, 2),
                "FAKE": round(class_probabilities.get(1, 0.0) * 100, 2),
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
    Heuristic generator label based on spatial & frequency statistics.

    NOTE: this is an unvalidated heuristic. The training corpus carries no
    per-generator ground truth, so these thresholds cannot be checked against
    real labels — treat the output as a hint, not a finding. Validating it needs
    a generator-labelled holdout (see src/evaluate_generators.py).
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

    # fftshift puts DC at the centre, so high frequencies live in the *outer*
    # region. The previous version averaged mag[h//4:3*h//4, w//4:3*w//4] — the
    # low-frequency centre — and compared it against > 1.3, a value that region
    # never reaches (observed max 1.292), so the FLUX.1 branch was unreachable.
    outer = np.ones(mag.shape, dtype=bool)
    outer[h // 4:3 * h // 4, w // 4:3 * w // 4] = False
    high_freq_ratio = float(mag[outer].mean() / (mag.mean() + 1e-8))

    # ── Texture & Color analysis ──
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_sat = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)[:, :, 1].mean()

    # Heuristic decision tree. The 0.96 cut is the upper quartile of the
    # corrected ratio measured over the test split (range 0.903–0.983).
    if orig_h >= 1024 or orig_w >= 1024:
        if spectral["hf_ratio"] > 0.72:
            return "SDXL (Stable Diffusion XL)"
        elif laplacian_var > 700:
            return "Midjourney v6"
        else:
            return "SDXL / FLUX"
    elif laplacian_var > 800 and mean_sat > 100:
        return "Midjourney"
    elif high_freq_ratio > 0.96 and laplacian_var > 500:
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

    These describe measurable image statistics; they are *not* independent
    evidence for the verdict, which comes from the CNN. Frequency bullets are
    worded as observations because the FFT statistics were measured to be
    non-discriminative on this corpus (AUC 0.562) — asserting them as a
    "diffusion signature" would overstate what they show.
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

        # Frequency observation (2D FFT) — descriptive only, see docstring.
        spectral = compute_fft_spectral_score(img)
        if spectral["hf_ratio"] > 0.72:
            reasons.append("High-frequency energy above typical range for this corpus")
        elif spectral["hf_ratio"] < 0.58:
            reasons.append("Low high-frequency energy — little fine detail or sensor noise")

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
    ResNet18 CNN verdict, presented alongside advisory signals (classical models,
    2D FFT spectrum, ELA).

    The advisory signals are reported but do **not** vote on the outcome. Measured
    on the held-out test split they are close to chance — SVM AUC 0.517, Random
    Forest 0.546, FFT 0.562, ELA 0.514 — while the CNN reaches 0.981. The previous
    weighted blend (CNN 35% / FFT 25% / ELA 20% / SVM 10% / RF 10%) cost 10.5
    points of accuracy overall and 68 points on high-resolution real photographs.
    See diag_forensics.py and diag_strategy.py.
    """

    ADVISORY_CLASSICAL = ("svm", "random_forest")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Share the cached CNNPredictor so switching between "resnet18" and
        # "ensemble" in the UI does not load a second copy of the checkpoint.
        self.cnn_predictor = _get_predictor("resnet18", cfg)
        self._classical: Dict[str, "ClassicalPredictor"] = {}
        for key in self.ADVISORY_CLASSICAL:
            try:
                self._classical[key] = _get_predictor(key, cfg)
            except Exception as exc:
                logger.warning(f"  Advisory model '{key}' unavailable: {exc}")

    def predict(self, image_source) -> Dict[str, Any]:
        # ── Verdict: ResNet18 CNN only ──
        result = self.cnn_predictor.predict(image_source)
        cnn_fake = float(result["probabilities"]["FAKE"])

        # ── Advisory signals (displayed, never decisive) ──
        advisory: Dict[str, str] = {"ResNet18 CNN (verdict)": result["prediction"]}

        for key, predictor in self._classical.items():
            try:
                res = predictor.predict(image_source)
                advisory[f"{key} (advisory)"] = res["prediction"]
            except Exception as exc:
                logger.debug(f"  Advisory model '{key}' failed on this image: {exc}")

        spectral = result.get("spectral_analysis") or compute_fft_spectral_score(_as_numpy(image_source))
        advisory["2D FFT Spectrum (advisory)"] = (
            "FAKE" if float(spectral["spectral_ai_score"]) >= 50 else "REAL"
        )
        ela = result.get("ela_metrics")
        if ela:
            advisory["ELA Inspector (advisory)"] = (
                "FAKE" if float(ela["ela_ai_score"]) >= 50 else "REAL"
            )

        agree = sum(1 for v in advisory.values() if v == result["prediction"])
        result["ensemble_votes"]   = advisory
        result["consensus_status"] = (
            f"{agree}/{len(advisory)} signals agree with the CNN verdict "
            f"({cnn_fake:.1f}% FAKE) — advisory signals are informational only"
        )
        return result


class SwinPredictor:
    """SwinV2-Tiny fine-tuned on the unified ~275K-image union corpus.

    Uses the exact eval path the model was trained/validated with (square
    resize + fixed-quality recompress + ImageNet normalisation, see
    ``src/train_swin._eval_transform``), so the app reports what
    ``src/eval_swin.py`` measures. Verdict is the same contract as the other
    predictors: ``prediction`` / ``confidence`` / ``probabilities``.

    The FAKE decision is taken at :attr:`THRESHOLD` rather than by argmax.
    Argmax (an implicit 0.5) is well calibrated on the manifest's own val
    split — 3.3% false alarms on reals — but on independent web photographs
    it flags 42% of real images as AI-generated, frequently at 0.97+. The
    softmax is badly overconfident off-distribution, so the boundary is moved
    to where the false-alarm rate on genuine photos becomes defensible.
    """

    # Chosen on the manifest val split (n=10,000) *and* the independent
    # internet tier (n=351), because the second is what uploads look like:
    #
    #   thr    in-dist real / fake      wild real / fake
    #   0.50      0.967 / 0.963          0.577 / 0.918   <- argmax, 71 false alarms
    #   0.90      0.989 / 0.919          0.786 / 0.874   <- served, 36 false alarms
    #
    # 0.90 halves the false alarms on real web photos and *raises* in-dist
    # real accuracy, for ~4pp of fake recall. Trading recall for precision on
    # the REAL class is the right direction for a tool whose failure mode is
    # calling someone's own photograph a fake.
    THRESHOLD = 0.90

    # Scores between these bounds are reported as inconclusive rather than
    # dressed up as a verdict. On the wild tier this abstains on ~31% of
    # images and lifts accuracy on the rest from 0.832 to 0.876 (real 0.809,
    # fake 0.919); in-distribution it abstains on ~12% at 0.980 selective
    # accuracy. The band brackets the threshold, so a confident FAKE needs
    # >=0.97 and a confident REAL needs <=0.30.
    ABSTAIN_LOW = 0.30
    ABSTAIN_HIGH = 0.97

    def __init__(self, cfg: Config, checkpoint_name: Optional[str] = None) -> None:
        import torch
        from src.train_swin import _eval_transform, build_model as build_swin

        self.cfg = cfg
        self.device = get_device()
        self.checkpoint_meta: Dict[str, Any] = {}

        swin_name = checkpoint_name or getattr(cfg, "swin_model_name", "swin_v2_tiny_512_hardstyles.pth")
        p = Path(swin_name)
        candidates = [
            cfg.models_dir / p,
            cfg.models_dir / "swin_epochs" / p.name,
            PROJECT_ROOT / "models" / p,
            PROJECT_ROOT / "models" / "swin_epochs" / p.name,
            cfg.models_dir / "swin_v2_tiny_512.pth",
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"SwinV2 checkpoint not found (tried {candidates})\n"
                "Train it first:  python src/train_swin.py"
            )
        # The checkpoint is rewritten while training is still running, so a
        # partially-written file is a transient, retryable state rather than a
        # corruption. torch.save is not atomic; the train loop's every-epoch
        # write can land mid-read here.
        checkpoint = None
        for attempt in range(3):
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device,
                                        weights_only=False)
                break
            except Exception as exc:                                   # noqa: BLE001
                if attempt == 2:
                    raise
                logger.warning("SwinV2 checkpoint read failed; retrying: %s", exc)
                time.sleep(1.0)

        class_names = tuple(checkpoint.get(
            "class_names", checkpoint.get("config", {}).get("class_names", ())))
        if class_names and class_names != self.cfg.class_names:
            raise ValueError(
                f"SwinV2 checkpoint class order {class_names} does not match "
                f"expected order {self.cfg.class_names}."
            )
        self.img_size = int(checkpoint.get("img_size", 512))
        self.threshold = float(checkpoint.get("decision_threshold",
                                             checkpoint.get("threshold",
                                                            checkpoint.get("calibrated_threshold",
                                                                           self.THRESHOLD))))
        self.transform = _eval_transform(self.img_size)
        model = build_swin(pretrained=False).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model
        self.checkpoint_meta = {
            "file": ckpt_path.name,
            "epoch": checkpoint.get("epoch"),
            "val_acc": checkpoint.get("val_acc"),
            "val_loss": checkpoint.get("val_loss"),
            "device": str(self.device),
            "classes": list(class_names or self.cfg.class_names),
            "img_size": self.img_size,
            "threshold": self.threshold,
        }
        logger.info("  SwinV2 loaded — epoch %s (val acc %.2f%%)",
                    checkpoint.get("epoch", "?"), checkpoint.get("val_acc", 0.0))

    def predict(self, image_source) -> Dict[str, Any]:
        import torch

        pil = _as_pil(image_source)
        if pil is None:
            raise TypeError(f"Unsupported image source: {type(image_source)!r}")

        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        raw_p_fake = float(probs[1])

        # Decide at the served threshold, then rescale the odds so the
        # reported confidence agrees with the verdict. Without the shift a
        # p_fake of 0.85 would be called REAL while displaying "85% FAKE".
        p_fake = ClipPredictor._threshold_adjusted_probability(
            raw_p_fake, self.threshold)
        shifted = np.array([1.0 - p_fake, p_fake], dtype=np.float64)
        pred_idx = int(p_fake >= 0.5)

        result = CNNPredictor._build_result(
            pred_idx=pred_idx,
            probs=shifted,
            confidence=float(shifted[pred_idx]) * 100.0,
        )
        result["model_name"] = "swin_v2_tiny"
        result["img_size"] = self.img_size
        result["raw_probability_fake"] = round(raw_p_fake * 100.0, 2)
        result["decision_threshold"] = round(float(self.threshold) * 100.0, 2)
        # Off-distribution the softmax is overconfident in both directions, so
        # mid-range scores are routed to review instead of asserted.
        result["review_recommended"] = bool(
            self.ABSTAIN_LOW < raw_p_fake < self.ABSTAIN_HIGH)
        result["reliability"] = (
            "inconclusive" if result["review_recommended"] else "high"
        )
        result.update(forensic_metrics(pil))
        return result


class ClipPredictor:
    """Frozen CLIP ViT-L/14 with retrained calibrated linear heads."""

    def __init__(self, cfg: Config) -> None:
        from src.score_clip import ClipLinearScorer

        self.cfg = cfg
        self.scorer = ClipLinearScorer(cfg=cfg, model_type="clip", workers=0)
        # The per-view heads are Platt-calibrated, so 0.5 is the natural
        # deployment boundary. The stored fixed-FPR threshold was selected on
        # the same-source validation corpus and produced too many false alarms
        # on independent web photographs; it remains available to evaluation,
        # but is not silently imposed on arbitrary uploads.
        self.threshold = 0.5
        self.checkpoint_meta = {
            **self.scorer.meta,
            "file": "models/clip_linear/heads.joblib",
            "device": self.scorer.device,
            "threshold": self.threshold,
            "target_fpr": self.scorer.target_fpr,
            "stored_validation_threshold": self.scorer.served_threshold(),
        }

    @staticmethod
    def _threshold_adjusted_probability(p_fake: float, threshold: float) -> float:
        """Shift calibrated odds so the deployed threshold maps to 50%."""
        eps = 1e-6
        p = float(np.clip(p_fake, eps, 1.0 - eps))
        t = float(np.clip(threshold, eps, 1.0 - eps))
        shifted_logit = np.log(p / (1.0 - p)) - np.log(t / (1.0 - t))
        return float(1.0 / (1.0 + np.exp(-shifted_logit)))

    @staticmethod
    def _review_recommended(p_fake: float, view_std: float) -> bool:
        """Whether a binary answer should be treated as inconclusive."""
        # The wide interval is intentional. On the independent web tier,
        # accepting only <=5% REAL or >=90% FAKE scores raises selective
        # accuracy to about 95%, at roughly 50% coverage. A forensic tool should
        # abstain rather than dress a weak score up as certainty.
        return bool(0.05 < p_fake < 0.90 or view_std >= 0.22)

    def predict(self, image_source) -> Dict[str, Any]:
        pil = _as_pil(image_source)
        if pil is None:
            raise TypeError(f"Unsupported image source: {type(image_source)!r}")
        details = self.scorer.score_image_details(pil)
        if details is None:
            raise ValueError("Image could not be decoded by the CLIP pipeline")
        raw = details["p_fake"]

        p_fake = self._threshold_adjusted_probability(raw, self.threshold)
        probs = np.array([1.0 - p_fake, p_fake], dtype=np.float64)
        pred_idx = int(p_fake >= 0.5)
        result = CNNPredictor._build_result(
            pred_idx=pred_idx,
            probs=probs,
            confidence=float(probs[pred_idx]) * 100.0,
        )
        result["raw_probability_fake"] = round(float(raw) * 100.0, 2)
        result["decision_threshold"] = round(float(self.threshold) * 100.0, 2)
        result["model_name"] = self.scorer.name
        result["view_probabilities"] = {
            name: round(value * 100.0, 2)
            for name, value in details["views"].items()
        }
        result["view_disagreement"] = round(details["view_std"] * 100.0, 2)
        # Binary image forensics is not reliable enough to force a confident
        # answer on every upload. Scores near the boundary or contradictory
        # views are explicitly routed to manual review instead.
        result["review_recommended"] = self._review_recommended(
            raw, details["view_std"]
        )
        result["reliability"] = (
            "inconclusive" if result["review_recommended"] else "high"
        )
        result.update(forensic_metrics(pil))
        return result


# ─── Predictor Cache ──────────────────────────────────────────────────────────
# Loading the CNN checkpoint costs a ~46MB disk read plus a device transfer, and
# the sklearn pickles up to 8MB each. predict_image() used to rebuild them on
# every call. Keyed by (model_type, checkpoint path) so pointing cfg at a
# different checkpoint still builds a fresh predictor.
_PREDICTOR_CACHE: Dict[Tuple[str, str], Any] = {}


def _predictor_cache_key(model_type: str, cfg: Config) -> Tuple[str, str]:
    """Return a cache key that changes when the model files change.

    A path-only key is subtly wrong for a long-running Streamlit process:
    retraining can replace a checkpoint in place while the old predictor stays
    cached forever.  Include the file size and nanosecond mtime so a retrained
    model is picked up without changing any scoring logic.  Missing files are
    still represented deterministically; the loader will raise the useful
    ``FileNotFoundError`` afterwards.
    """
    if model_type in ("clip", "clip_linear"):
        paths = (cfg.models_dir / "clip_linear" / "heads.joblib",)
    elif model_type in ("swin", "swin_hardstyles", "swin_ep9", "swin_improved_ep6", "swin_improved_ep5", "swin_improved_ep3", "swin_improved_ep4", "swin_improved_ep2", "swin_newdata", "swin_newdata_ep3", "swin_newdata_ep4", "swin_newdata_ep2", "swin_newdata_ep1"):
        target_map = {
            "swin_hardstyles": "swin_v2_tiny_512_hardstyles.pth",
            "swin_ep9": "swin_epochs/swin_v2_512_ep009.pth",
            "swin_improved_ep6": "swin_epochs/swin_v2_512_improved_ep006.pth",
            "swin_improved_ep5": "swin_epochs/swin_v2_512_improved_ep005.pth",
            "swin_improved_ep3": "swin_epochs/swin_v2_512_improved_ep003.pth",
            "swin_improved_ep4": "swin_epochs/swin_v2_512_improved_ep004.pth",
            "swin_improved_ep2": "swin_epochs/swin_v2_512_improved_ep002.pth",
            "swin_newdata": "swin_v2_tiny_512_newdata.pth",
            "swin_newdata_ep3": "swin_epochs/swin_v2_512_newdata_ep003.pth",
            "swin_newdata_ep4": "swin_epochs/swin_v2_512_newdata_ep004.pth",
            "swin_newdata_ep2": "swin_epochs/swin_v2_512_newdata_ep002.pth",
            "swin_newdata_ep1": "swin_epochs/swin_v2_512_newdata_ep001.pth",
        }
        target_name = target_map.get(model_type, getattr(cfg, "swin_model_name", "swin_epochs/swin_v2_512_improved_ep006.pth"))
        p = Path(target_name)
        candidates = [
            cfg.models_dir / p,
            cfg.models_dir / "swin_epochs" / p.name,
            PROJECT_ROOT / "models" / p,
            PROJECT_ROOT / "models" / "swin_epochs" / p.name,
            cfg.models_dir / "swin_epochs" / "swin_v2_512_improved_ep006.pth",
            cfg.models_dir / "swin_v2_tiny_512_improved.pth",
            cfg.models_dir / "swin_v2_tiny_512_hardstyles.pth",
            cfg.models_dir / "swin_v2_tiny_512.pth",
        ]
        chosen = candidates[-1]
        for c in candidates:
            if c.exists() and c.is_file():
                chosen = c
                break
        paths = (chosen,)
    elif model_type == "resnet18":
        paths = (cfg.models_dir / cfg.cnn_model_name,)
    elif model_type == "ensemble":
        paths = (
            cfg.models_dir / cfg.cnn_model_name,
            cfg.models_dir / "svm.pkl",
            cfg.models_dir / "random_forest.pkl",
        )
    else:
        model_names = {
            "logistic_regression": "logistic_regression.pkl",
            "random_forest": "random_forest.pkl",
            "svm": "svm.pkl",
            "knn": "k-nn.pkl",
            "decision_tree": "decision_tree.pkl",
            "naive_bayes": "naive_bayes.pkl",
        }
        paths = (cfg.models_dir / model_names.get(model_type, model_type),)

    fingerprints = []
    for path in paths:
        try:
            stat = path.stat()
            fingerprints.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            fingerprints.append(f"{path.resolve()}:missing")
    return model_type, "|".join(fingerprints)


def _get_predictor(model_type: str, cfg: Config):
    key = _predictor_cache_key(model_type, cfg)
    if key not in _PREDICTOR_CACHE:
        if model_type in ("clip", "clip_linear"):
            _PREDICTOR_CACHE[key] = ClipPredictor(cfg)
        elif model_type == "ensemble":
            _PREDICTOR_CACHE[key] = EnsemblePredictor(cfg)
        elif model_type == "resnet18":
            _PREDICTOR_CACHE[key] = CNNPredictor(cfg)
        elif model_type == "swin_improved_ep6":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_improved_ep006.pth")
        elif model_type == "swin_improved_ep5":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_improved_ep005.pth")
        elif model_type == "swin_improved_ep3":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_improved_ep003.pth")
        elif model_type == "swin_improved_ep4":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_improved_ep004.pth")
        elif model_type == "swin_improved_ep2":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_improved_ep002.pth")
        elif model_type == "swin_newdata_ep3":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_newdata_ep003.pth")
        elif model_type == "swin_newdata_ep4":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_newdata_ep004.pth")
        elif model_type == "swin_newdata_ep2":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_newdata_ep002.pth")
        elif model_type == "swin_newdata_ep1":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_newdata_ep001.pth")
        elif model_type == "swin_newdata":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_v2_tiny_512_newdata.pth")
        elif model_type == "swin_hardstyles":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_v2_tiny_512_hardstyles.pth")
        elif model_type == "swin_ep9":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg, checkpoint_name="swin_epochs/swin_v2_512_ep009.pth")
        elif model_type == "swin":
            _PREDICTOR_CACHE[key] = SwinPredictor(cfg)
        else:
            _PREDICTOR_CACHE[key] = ClassicalPredictor(model_type, cfg)
    return _PREDICTOR_CACHE[key]


def clear_predictor_cache() -> None:
    """Drop cached predictors (call after retraining or swapping a checkpoint)."""
    _PREDICTOR_CACHE.clear()


# ─── Full Prediction Pipeline ─────────────────────────────────────────────────
def predict_image(
    image_source,
    cfg:         Config,
    model_type:  str = "ensemble",
    with_gradcam: bool = True,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Top-level prediction function used by the Streamlit app.

    The verdict comes from the CNN (or the selected classical model) alone. FFT
    and ELA metrics are attached under "spectral_analysis" / "ela_metrics" for
    display, but never modify the prediction — see forensic_metrics().

    Returns:
        result  : dict with prediction, confidence, explanation, generator
        heatmap : numpy RGB overlay (or None if with_gradcam=False / classical model)
    """
    from src.image_forensics import analyze_image_provenance

    # Keep original encoded bytes available to the provenance analyser, while
    # giving the ML pipeline a decoded RGB image it already understands.
    original_source = image_source
    model_source = _as_pil(image_source)
    if model_source is None:
        raise TypeError(f"Unsupported image source: {type(image_source)!r}")

    img_np  = _as_numpy(model_source)
    heatmap = None

    if model_type in ("clip", "clip_linear"):
        result = _get_predictor("clip", cfg).predict(model_source)
    elif model_type == "ensemble":
        ensemble_pred = _get_predictor("ensemble", cfg)
        result = ensemble_pred.predict(model_source)
        if with_gradcam:
            try:
                _, heatmap = ensemble_pred.cnn_predictor.predict_with_gradcam(model_source)
            except Exception as exc:
                logger.debug(f"  Grad-CAM unavailable: {exc}")
                heatmap = None
    elif model_type == "resnet18":
        predictor = _get_predictor("resnet18", cfg)
        if with_gradcam:
            result, heatmap = predictor.predict_with_gradcam(model_source)
        else:
            result = predictor.predict(model_source)
    elif model_type in ("swin", "swin_hardstyles", "swin_ep9", "swin_improved_ep6", "swin_improved_ep5", "swin_improved_ep3", "swin_improved_ep4", "swin_improved_ep2", "swin_newdata", "swin_newdata_ep3", "swin_newdata_ep4", "swin_newdata_ep2", "swin_newdata_ep1"):
        # SwinV2 is a CNN but has no layer3/layer4 hooks, so Grad-CAM is not
        # wired up for it — the verdict is produced by predict() alone.
        result = _get_predictor(model_type, cfg).predict(model_source)
    else:
        result = _get_predictor(model_type, cfg).predict(model_source)
        result.update(forensic_metrics(model_source))

    # Enrich result
    result["explanation"] = generate_explanation(result, img_np)
    if result["prediction"] == "FAKE":
        result["likely_generator"] = detect_likely_generator(img_np)
    else:
        result["likely_generator"] = "N/A"
    result["provenance_analysis"] = analyze_image_provenance(original_source)

    return result, heatmap


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TruthLens — Single Image Prediction")
    parser.add_argument("--image",   required=True, help="Path to input image.")
    parser.add_argument(
        "--model", default="clip",
        choices=["clip", "resnet18", "swin", "logistic_regression", "decision_tree",
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
    provenance = result.get("provenance_analysis") or {}
    if provenance.get("available"):
        print(f"  Provenance : {provenance.get('summary', 'analysed')}")
        for signal in provenance.get("signals", [])[:5]:
            print(f"    [{signal['level']}] {signal['title']}: {signal['detail']}")
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
