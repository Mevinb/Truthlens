#!/usr/bin/env python3
"""
TruthLens — app/inference.py
============================
Everything the frontend needs from the model layer, kept out of app.py so the
UI file is about layout only.

The single rule enforced here: all inference goes through
``src.predict.predict_image``. The app used to re-implement preprocessing and
softmax itself, which is how it ended up serving results whose numbers did not
match anything the project measured. There is now one path, so
``src/evaluate.py`` and ``diag_benchmark.py`` describe what the app actually
returns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Model registry ──────────────────────────────────────────────────────────
# Display label → the model_type key understood by src.predict.predict_image.
# One CNN entry, not three: only one ResNet18 checkpoint exists
# (resnet18_highres.pth is a symlink to resnet18_truthlens.pth), and the old
# selector listed three variants that loaded the same weights while quoting
# three different accuracies.
MODEL_OPTIONS: Dict[str, str] = {
    "CLIP ViT-L/14 (retrained)":  "clip",
    "ResNet18 (CNN)":              "resnet18",
    "ResNet18 + advisory signals": "ensemble",
    "Logistic Regression":         "logistic_regression",
    "Random Forest":               "random_forest",
    "SVM (RBF kernel)":            "svm",
    "k-NN":                        "knn",
    "Decision Tree":               "decision_tree",
    "Naive Bayes":                 "naive_bayes",
}

CNN_TYPES = ("clip", "resnet18", "ensemble")
GRADCAM_TYPES = ("resnet18", "ensemble")

# model_type → key inside results/metrics/benchmark_test_split.json
_BENCH_KEY = {"resnet18": "resnet18_tta", "ensemble": "resnet18_tta"}

MODEL_BLURB: Dict[str, str] = {
    "clip":                "Frozen CLIP ViT-L/14 with three native-resolution "
                           "views, one whole-frame view, and retrained calibrated "
                           "linear heads. It is "
                           "the default because it generalises much better to "
                           "unseen generators and source pipelines.",
    "resnet18":            "ResNet18 fine-tuned from ImageNet, with horizontal-flip and "
                           "five-crop test-time augmentation.",
    "ensemble":            "Same ResNet18 verdict, with the classical models and the "
                           "forensic signals shown alongside as advisory only.",
    "logistic_regression": "Linear classifier on PCA-reduced HOG + LBP texture features.",
    "random_forest":       "300-tree forest on HOG + LBP texture features.",
    "svm":                 "Support vector machine, RBF kernel, on HOG + LBP features.",
    "knn":                 "k-nearest neighbours in PCA-reduced HOG + LBP space.",
    "decision_tree":       "Single CART tree, depth 20, on HOG + LBP features.",
    "naive_bayes":         "Gaussian naive Bayes on raw HOG + LBP features.",
}

# Measured AUC of each forensic signal on its own, from diag_forensics.py.
# Both sit near the 0.500 chance line, which is why neither is allowed to move
# the verdict and why the panels that show them are labelled descriptive.
FORENSIC_AUC = {"fft": 0.562, "ela": 0.514}

# Per-class means ± std of each forensic metric, measured on a 200-per-class
# stratified sample of datasets/prepared/multires/test. The point of showing these next to
# a single image's numbers is that the two classes overlap almost completely —
# e.g. hf_ratio differs by 0.005 between real and fake against a spread of 0.07.
# The previous UI called anything above hf_ratio 0.72 a "VAE grid signature",
# a threshold that sits directly on the mean for *real* photographs.
FORENSIC_REFERENCE = {
    "hf_ratio": {"real": (0.7197, 0.0740), "fake": (0.7249, 0.0701)},
    "ela_mean": {"real": (22.92, 12.19),   "fake": (21.47, 9.16)},
    "ela_std":  {"real": (25.78, 9.50),    "fake": (25.35, 7.97)},
}


# ─── Config / predictors ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def resolve_config():
    """Config pointed at whichever CNN checkpoint is actually on disk."""
    from src.utils import Config, resolve_cnn_checkpoint

    return resolve_cnn_checkpoint(Config())


@st.cache_resource(show_spinner=False)
def _warm_predictor(model_type: str) -> str:
    """Build the predictor once so load failures surface before inference and
    the checkpoint is not re-read per image."""
    from src.predict import _get_predictor

    _get_predictor(model_type, resolve_config())
    return model_type


def load_predictor(display_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(model_type, None)`` on success, or ``(None, message)``."""
    model_type = MODEL_OPTIONS.get(display_name, display_name)
    if model_type not in MODEL_OPTIONS.values():
        return None, f"Unknown model: {display_name!r}"
    try:
        return _warm_predictor(model_type), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:                                  # noqa: BLE001
        logger.exception("Failed to load predictor %s", model_type)
        return None, f"{type(exc).__name__}: {exc}"


def run_prediction(model_type: str, img, with_gradcam: bool):
    """Run inference. Returns ``(result, heatmap_or_None)``, or
    ``(None, message)`` if inference raised."""
    from src.predict import predict_image

    try:
        return predict_image(
            img, resolve_config(), model_type=model_type, with_gradcam=with_gradcam
        )
    except Exception as exc:                                  # noqa: BLE001
        logger.exception("Prediction failed for %s", model_type)
        return None, f"{type(exc).__name__}: {exc}"


def checkpoint_provenance() -> Dict[str, Any]:
    """Metadata for the loaded CNN weights, or ``{}`` if the CNN is unavailable.

    Read off the predictor rather than re-opening the 45 MB checkpoint, and
    surfaced in the UI so a stale or unexpected checkpoint is visible instead of
    implied by an accuracy figure.
    """
    from src.predict import _get_predictor

    try:
        return dict(_get_predictor("clip", resolve_config()).checkpoint_meta)
    except Exception as exc:                                  # noqa: BLE001
        logger.debug("Checkpoint provenance unavailable: %s", exc)
        return {}


# ─── Measured performance ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def benchmark() -> Dict[str, Any]:
    """Load results/metrics/benchmark_test_split.json.

    Read from disk rather than hardcoded in the UI, so re-running
    ``python diag_benchmark.py`` updates every number the app displays and the
    two can never disagree. Returns ``{}`` when the file is absent.
    """
    path = PROJECT_ROOT / "results" / "metrics" / "benchmark_test_split.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        logger.warning("No benchmark file at %s — run diag_benchmark.py", path)
        return {}
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("Could not read benchmark file: %s", exc)
        return {}


def model_scores(model_type: str) -> Dict[str, Any]:
    """Measured test-split scores for one model_type, or ``{}``."""
    if model_type == "clip":
        path = PROJECT_ROOT / "results" / "metrics" / "clip_linear.json"
        try:
            block = json.loads(path.read_text())["splits"]["test"]["ensemble_all"]
            return {
                "n": block["n"],
                "accuracy": block["accuracy_at_0.5"] * 100,
                "f1": block["f1_at_0.5"],
                "auc": block["auc"],
                "real_accuracy": block["real_acc_at_0.5"] * 100,
                "fake_accuracy": block["fake_acc_at_0.5"] * 100,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return {}
    key = _BENCH_KEY.get(model_type, model_type)
    return benchmark().get("models", {}).get(key, {})


def benchmark_rows() -> List[Dict[str, Any]]:
    """Every benchmarked model as display rows, best accuracy first."""
    labels = {
        "resnet18_tta":         "ResNet18 (CNN + TTA)",
        "resnet18_single_crop": "ResNet18 (single crop)",
        "logistic_regression":  "Logistic Regression",
        "random_forest":        "Random Forest",
        "svm":                  "SVM (RBF kernel)",
        "knn":                  "k-NN",
        "decision_tree":        "Decision Tree",
        "naive_bayes":          "Naive Bayes",
    }
    rows = []
    clip = model_scores("clip")
    if clip:
        rows.append({
            "Model": "CLIP ViT-L/14 + linear heads",
            "Accuracy": clip["accuracy"],
            "F1": clip["f1"],
            "AUC": clip["auc"],
            "Real acc": clip.get("real_accuracy"),
            "Fake acc": clip.get("fake_accuracy"),
            "Images": clip.get("n"),
        })
    for key, m in benchmark().get("models", {}).items():
        if "error" in m or "accuracy" not in m:
            continue
        rows.append(
            {
                "Model":     labels.get(key, key),
                "Accuracy":  m["accuracy"],
                "F1":        m["f1"],
                "AUC":       m["auc"],
                "Real acc":  m.get("real_accuracy"),
                "Fake acc":  m.get("fake_accuracy"),
                "Images":    m.get("n"),
            }
        )
    return sorted(rows, key=lambda r: r["Accuracy"], reverse=True)


# ─── Resolution tiers ───────────────────────────────────────────────────────
# The corpus is built in three tiers and stored at fixed sizes: 32×32,
# 256×256, and 455–2048px on the long edge. Thresholds on the shorter edge
# recover which tier an arbitrary image belongs to.
TIER_LABEL = {
    "highres":    "high-resolution (512px+)",
    "medres256":  "medium-resolution (256px)",
    "lowres32":   "low-resolution (32px)",
}


def tier_for_size(width: int, height: int) -> str:
    short = min(width, height)
    if short <= 64:
        return "lowres32"
    if short <= 320:
        return "medres256"
    return "highres"


def tier_scores(tier: str) -> Dict[str, Any]:
    """CNN scores restricted to one resolution tier, or ``{}``.

    Worth quoting per tier rather than only in aggregate: the blended headline
    is several points above the high-resolution tier, which is the regime almost
    every real upload falls into.
    """
    return benchmark().get("by_resolution", {}).get(tier, {})


def resolution_rows() -> List[Dict[str, Any]]:
    """Per-tier CNN scores as display rows, coarsest resolution last."""
    order = ("highres", "medres256", "lowres32")
    by = benchmark().get("by_resolution", {})
    return [
        {
            "Resolution tier": TIER_LABEL.get(t, t),
            "Accuracy": by[t]["accuracy"],
            "F1":       by[t]["f1"],
            "AUC":      by[t]["auc"],
            "Real acc": by[t].get("real_accuracy"),
            "Fake acc": by[t].get("fake_accuracy"),
            "Images":   by[t].get("n"),
        }
        for t in order
        if t in by and "accuracy" in by[t]
    ]


# ─── Sample images ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def sample_images(per_class: int = 3) -> List[Dict[str, str]]:
    """A few held-out test images so the app is usable without an upload.

    Deliberately drawn from ``test/``, never ``train/`` — a demo that shows off
    training images is showing off memorisation. Returns ``[]`` if the corpus
    is not present, and the caller degrades to upload-only.
    """
    out: List[Dict[str, str]] = []
    for cls, truth in (("real", "REAL"), ("fake", "FAKE")):
        folder = PROJECT_ROOT / "datasets" / "prepared" / "multires" / "test" / cls
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.glob("highres_*") if p.is_file())
        step = max(1, len(files) // max(per_class, 1))
        for path in files[::step][:per_class]:
            out.append({"path": str(path), "truth": truth, "name": path.name})
    return out
