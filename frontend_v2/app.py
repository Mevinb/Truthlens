#!/usr/bin/env python3
"""TruthLens V2: a separate Streamlit frontend for the existing model layer."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

HERO_IMAGE = HERE / "assets" / "forensic-loupe.webp"

st.set_page_config(
    page_title="TruthLens | Image authenticity review",
    page_icon=str(HERO_IMAGE),
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "TruthLens screens images for signs of AI generation. "
            "Training is still in progress, so results are not proof of authenticity."
        )
    },
)

from app.inference import (  # noqa: E402
    FORENSIC_AUC,
    FORENSIC_REFERENCE,
    GRADCAM_TYPES,
    TIER_LABEL,
    benchmark_rows,
    load_predictor,
    model_scores,
    resolution_rows,
    resolve_config,
    run_prediction,
    sample_images,
    tier_for_size,
    tier_scores,
)
from frontend_v2 import theme as T  # noqa: E402


logger = logging.getLogger(__name__)

MAX_PIXELS = 40_000_000
ACCEPTED = ["jpg", "jpeg", "png", "webp", "bmp"]

MODEL_CHOICES: Dict[str, str] = {
    "SwinV2 Tiny, improved checkpoint 6": "swin_improved_ep6",
    "SwinV2 Tiny, improved checkpoint 5": "swin_improved_ep5",
    "SwinV2 Tiny, improved checkpoint 3": "swin_improved_ep3",
    "SwinV2 Tiny, improved checkpoint 4": "swin_improved_ep4",
    "SwinV2 Tiny, hard styles": "swin_hardstyles",
    "SwinV2 Tiny, checkpoint 9": "swin_ep9",
    "CLIP ViT-L/14, retrained": "clip",
    "ResNet18 with test-time augmentation": "resnet18",
    "ResNet18 with advisory signals": "ensemble",
    "Logistic regression": "logistic_regression",
    "Random forest": "random_forest",
    "Support vector machine": "svm",
    "k-nearest neighbours": "knn",
    "Decision tree": "decision_tree",
    "Naive Bayes": "naive_bayes",
}

SWIN_FILES = {
    "swin_improved_ep6": "swin_v2_512_improved_ep006.pth",
    "swin_improved_ep5": "swin_v2_512_improved_ep005.pth",
    "swin_improved_ep3": "swin_v2_512_improved_ep003.pth",
    "swin_improved_ep4": "swin_v2_512_improved_ep004.pth",
    "swin_hardstyles": "swin_v2_tiny_512_hardstyles.pth",
    "swin_ep9": "swin_v2_512_ep009.pth",
}


@st.cache_data(show_spinner=False, max_entries=48)
def analyse(raw: bytes, model_type: str, with_gradcam: bool):
    """Run the shared production inference path and cache identical work."""
    return run_prediction(model_type, raw, with_gradcam=with_gradcam)


@st.cache_data(show_spinner=False, max_entries=48)
def fft_spectrum(raw: bytes) -> Optional[np.ndarray]:
    import cv2

    try:
        arr = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        magnitude = np.log(
            np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))) + 1e-8
        )
        normalized = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return cv2.cvtColor(
            cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS),
            cv2.COLOR_BGR2RGB,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("FFT spectrum failed: %s", exc)
        return None


@st.cache_data(show_spinner=False, max_entries=48)
def ela_image(raw: bytes) -> Optional[np.ndarray]:
    from src.predict import compute_ela_analysis

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return compute_ela_analysis(image)[1]
    except Exception as exc:  # noqa: BLE001
        logger.debug("ELA image failed: %s", exc)
        return None


def clean_text(value: object) -> str:
    return str(value).translate({0x2014: ord("-"), 0x2013: ord("-"), 0x00B7: ord("/")})


def digest(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:10]


def number(value: Any, pattern: str = "{:.4f}", empty: str = "Not available") -> str:
    try:
        return pattern.format(float(value))
    except (TypeError, ValueError):
        return empty


def load_image(raw: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width * height > MAX_PIXELS:
            megapixels = width * height / 1_000_000
            return None, f"{width} x {height} ({megapixels:.0f} MP) is too large to process."
        return image.convert("RGB"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"The file could not be decoded ({type(exc).__name__})."


def make_thumbnail_data_url(raw: bytes, max_size: int = 96) -> str:
    """Generate a lightweight Base64 data URL thumbnail for dataframe and card display."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception:
        return ""


def model_label(model_type: str) -> str:
    for label, value in MODEL_CHOICES.items():
        if value == model_type:
            return label
    return model_type


def find_swin_evaluation(model_type: str) -> Dict[str, Any]:
    target = SWIN_FILES.get(model_type)
    if not target:
        return {}
    preferred = (
        "swin_model_comparison.json"
        if model_type in {"swin_hardstyles", "swin_ep9"}
        else "improved_swin_benchmark.json"
    )
    filenames = [preferred]
    filenames.append(
        "improved_swin_benchmark.json"
        if preferred == "swin_model_comparison.json"
        else "swin_model_comparison.json"
    )
    for filename in filenames:
        path = PROJECT_ROOT / "results" / "metrics" / filename
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("name") == target:
                return row
    return {}


def selected_model_metrics(model_type: str) -> Dict[str, Any]:
    measured = model_scores(model_type)
    if measured:
        return {
            "split": "held-out test",
            "accuracy": measured.get("accuracy"),
            "auc": measured.get("auc"),
            "real_accuracy": measured.get("real_accuracy"),
            "fake_accuracy": measured.get("fake_accuracy"),
            "n": measured.get("n"),
        }
    row = find_swin_evaluation(model_type)
    holdout = row.get("splits", {}).get("holdout", {}) if row else {}
    if not holdout:
        return {}
    return {
        "split": "independent holdout",
        "accuracy": holdout.get("balanced_acc", holdout.get("accuracy")),
        "auc": holdout.get("auc"),
        "real_accuracy": holdout.get("real_acc"),
        "fake_accuracy": holdout.get("fake_recall"),
        "n": holdout.get("n"),
    }


def preferred_dark_mode() -> bool:
    try:
        return str(st.context.theme.type).lower() == "dark"
    except Exception:  # noqa: BLE001
        return False


def render_settings() -> Dict[str, Any]:
    with st.popover("Model settings", width="stretch"):
        st.markdown("#### Review settings")
        labels = list(MODEL_CHOICES)
        display = st.selectbox(
            "Detection model",
            labels,
            index=0,
            help="The selected checkpoint produces the verdict. Evidence panels stay descriptive.",
        )
        model_type = MODEL_CHOICES[display]
        with_gradcam = st.toggle(
            "Create attention map",
            value=False,
            disabled=model_type not in GRADCAM_TYPES,
            help="Attention maps are available only for the ResNet18 models.",
        )
        flag_below = st.slider(
            "Flag confidence below",
            min_value=50,
            max_value=99,
            value=70,
            step=1,
            format="%d%%",
            help="This changes the caution message, not the underlying model label.",
        )
        st.toggle("Dark appearance", key="theme_dark")

        metrics = selected_model_metrics(model_type)
        if metrics:
            st.markdown(
                T.kv_grid(
                    [
                        ("Measured split", metrics["split"]),
                        ("Accuracy", number(metrics.get("accuracy"), "{:.1f}%")),
                        ("AUC", number(metrics.get("auc"), "{:.3f}")),
                        ("Images", f"{int(metrics['n']):,}" if metrics.get("n") else "Not available"),
                    ]
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                T.callout("No matching benchmark record is available for this model."),
                unsafe_allow_html=True,
            )

    return {
        "display": display,
        "model_type": model_type,
        "with_gradcam": with_gradcam and model_type in GRADCAM_TYPES,
        "flag_below": flag_below,
    }


def render_header_and_settings() -> Dict[str, Any]:
    left, right = st.columns([5.4, 1], vertical_alignment="center")
    with left:
        st.markdown(T.header(training=True), unsafe_allow_html=True)
    with right:
        return render_settings()


def render_sample_picker() -> Optional[Dict[str, str]]:
    samples = sample_images(per_class=2)
    if not samples:
        return st.session_state.get("active_sample")

    st.markdown(
        T.section_head(
            "Try a known example",
            "These images come from the held-out test split and include their known class.",
        ),
        unsafe_allow_html=True,
    )
    with st.container(key="samples"):
        columns = st.columns(len(samples), gap="small")
        for column, sample in zip(columns, samples):
            with column:
                st.image(sample["path"], width="stretch")
                truth = "Known real" if sample["truth"] == "REAL" else "Known AI generated"
                st.markdown(
                    f'<div class="tl-sample-label"><b>{T.safe(truth)}</b>'
                    f"<span>{T.safe(sample['name'])}</span></div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Analyze sample",
                    key=f"sample_{sample['truth']}_{sample['name']}",
                    width="stretch",
                ):
                    st.session_state["active_sample"] = sample

        if st.session_state.get("active_sample") and st.button(
            "Clear selected sample", key="clear_sample"
        ):
            st.session_state.pop("active_sample", None)
            st.rerun()

    return st.session_state.get("active_sample")


def render_workspace() -> List[Dict[str, Any]]:
    uploads = None
    with st.container(key="workspace"):
        intro, upload = st.columns([0.84, 1.16], gap="large", vertical_alignment="center")
        with intro:
            st.markdown(T.hero_intro(), unsafe_allow_html=True)
        with upload:
            st.markdown(
                '<div class="tl-workspace-label">Choose images to inspect</div>'
                '<p class="tl-workspace-help">JPEG, PNG, WebP or BMP. Batch upload is supported.</p>',
                unsafe_allow_html=True,
            )
            uploads = st.file_uploader(
                "Choose images to inspect",
                type=ACCEPTED,
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploads:
                preview, error = load_image(uploads[0].getvalue())
                if preview is not None:
                    st.image(preview, width="stretch")
                elif error:
                    st.markdown(T.callout(error, "error"), unsafe_allow_html=True)
            else:
                st.image(HERO_IMAGE, width="stretch")

    items = [
        {"name": upload.name, "bytes": upload.getvalue(), "truth": None}
        for upload in (uploads or [])
    ]
    active_sample = render_sample_picker() if not items else st.session_state.get("active_sample")
    if not items and active_sample:
        try:
            items.append(
                {
                    "name": f"{active_sample['truth'].lower()}/{active_sample['name']}",
                    "bytes": Path(active_sample["path"]).read_bytes(),
                    "truth": active_sample["truth"],
                }
            )
        except OSError as exc:
            st.markdown(
                T.callout(f"The selected sample could not be read: {exc}", "error"),
                unsafe_allow_html=True,
            )
            st.session_state.pop("active_sample", None)
    return items


def render_result_summary(
    item: Dict[str, Any], result: Dict[str, Any], settings: Dict[str, Any]
) -> None:
    prediction = str(result["prediction"])
    confidence = float(result["confidence"])
    is_fake = prediction == "FAKE"
    inconclusive = bool(result.get("review_recommended"))
    tone = "uncertain" if inconclusive else ("fake" if is_fake else "real")
    label = "Inconclusive" if inconclusive else ("Likely AI generated" if is_fake else "Likely real")

    metrics = selected_model_metrics(settings["model_type"])
    reliability = "No matching benchmark score is available for this checkpoint."
    if metrics.get("accuracy") is not None:
        reliability = (
            f"Measured {float(metrics['accuracy']):.1f}% accuracy on its "
            f"{metrics['split']} split."
        )

    prov_verified = result.get("provenance_verified", False)
    if prov_verified:
        prov_basis = result.get("consensus_basis") or "Verified provenance"
        subtitle = f"🛡️ <b>{T.safe(prov_basis)}</b>"
    else:
        subtitle = (
            f"Analyzed with <b>{T.safe(settings['display'])}</b>. {T.safe(reliability)}"
        )

    if settings["model_type"] in {"resnet18", "ensemble"} and not prov_verified:
        width, height = item.get("size", (0, 0))
        if width and height:
            tier = tier_for_size(width, height)
            tier_metric = tier_scores(tier)
            if tier_metric.get("accuracy") is not None:
                reliability = (
                    f"Measured {float(tier_metric['accuracy']):.1f}% accuracy on held-out "
                    f"{TIER_LABEL[tier]} images."
                )
                subtitle = (
                    f"Analyzed with <b>{T.safe(settings['display'])}</b>. "
                    f"{T.safe(reliability)}"
                )
    st.markdown(
        T.verdict(label, confidence, subtitle, tone),
        unsafe_allow_html=True,
    )

    if result.get("provenance_overridden"):
        attr = result.get("ai_attribution") or {}
        provider = T.safe(str(attr.get("provider", "AI")))
        wm_type = T.safe(str(attr.get("watermark_type", "watermark")))
        st.markdown(
            T.callout(
                f"Authoritative Provenance Override: This image carries verified {provider} watermarking or metadata ({wm_type}). Calibrated consensus classifies it as AI-GENERATED.",
                "caution",
            ),
            unsafe_allow_html=True,
        )

    if inconclusive:
        st.markdown(
            T.callout(
                "The score is too close to the model's review boundary. Use manual review and provenance evidence before making a claim.",
                "caution",
            ),
            unsafe_allow_html=True,
        )

    if confidence < settings["flag_below"]:
        st.markdown(
            T.callout(
                f"Confidence is below your {settings['flag_below']}% review threshold. Treat the result as undecided.",
                "caution",
            ),
            unsafe_allow_html=True,
        )

    truth = item.get("truth")
    if truth:
        comparison = "matches" if truth == prediction else "does not match"
        st.markdown(
            T.callout(
                f"Known label: {truth}. The model's underlying label {comparison} this sample.",
                "info" if truth == prediction else "caution",
            ),
            unsafe_allow_html=True,
        )

    # ─── Watermark & Provenance Scan Status Card ───
    prov = result.get("provenance_analysis") or {}
    gem = prov.get("gemini_provenance") or {}
    gpt = prov.get("chatgpt_provenance") or {}
    inv = prov.get("invisible_watermark") or {}
    c2pa = prov.get("content_credentials") or {}

    gem_detected = bool(gem.get("detected"))
    gpt_detected = bool(gpt.get("detected") or c2pa.get("is_ai_generated"))
    inv_score = float(inv.get("score", 0.0))
    inv_suspected = bool(inv.get("suspected"))

    st.markdown(
        f'<div style="margin: 14px 0 18px 0; padding: 12px 16px; border: 1px solid rgba(255,255,255,0.08); '
        f'border-radius: 8px; background: rgba(255,255,255,0.02);">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">'
        f'<span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; color:{T.TEXT_3};">'
        f'🛡️ Watermark & Provenance Verification Scan</span>'
        f'<span style="font-size:0.76rem; color:{T.FAKE if result.get("provenance_verified") else T.TEXT_3}; font-weight:600;">'
        f'{"⚠️ Verified AI Watermark Found" if result.get("provenance_verified") else "⚪ Scanned — No Hidden Watermarks Found"}'
        f'</span></div>'
        f'<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px;">'
        f'<div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:6px; border-left:3px solid '
        f'{T.FAKE if gem_detected else T.REAL};">'
        f'<div style="font-size:0.72rem; color:{T.TEXT_3}; font-weight:600;">Google Gemini (SynthID & IPTC)</div>'
        f'<div style="font-size:0.83rem; font-weight:700; color:{T.FAKE if gem_detected else T.REAL}; margin-top:2px;">'
        f'{"🛡️ " + T.safe(str(gem.get("watermark_type") or "SynthID Detected")) if gem_detected else "✅ Scanned (Not Detected)"}'
        f'</div></div>'
        f'<div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:6px; border-left:3px solid '
        f'{T.FAKE if gpt_detected else T.REAL};">'
        f'<div style="font-size:0.72rem; color:{T.TEXT_3}; font-weight:600;">OpenAI ChatGPT (C2PA & DALL·E)</div>'
        f'<div style="font-size:0.83rem; font-weight:700; color:{T.FAKE if gpt_detected else T.REAL}; margin-top:2px;">'
        f'{"🛡️ " + T.safe(str(gpt.get("watermark_type") or "C2PA Detected")) if gpt_detected else "✅ Scanned (Not Detected)"}'
        f'</div></div>'
        f'<div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:6px; border-left:3px solid '
        f'{T.FAKE if inv_suspected else T.TEXT_3};">'
        f'<div style="font-size:0.72rem; color:{T.TEXT_3}; font-weight:600;">Invisible Perturbation / LSB</div>'
        f'<div style="font-size:0.83rem; font-weight:700; color:{T.FAKE if inv_suspected else T.TEXT_1}; margin-top:2px;">'
        f'{"⚠️ Anomaly Detected" if inv_suspected else "Normal Baseline (" + f"{inv_score:.1f}/100)"}'
        f'</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    summary, preview = st.columns([1.04, 0.96], gap="large")
    with summary:
        real = float(result["probabilities"]["REAL"])
        fake = float(result["probabilities"]["FAKE"])
        st.markdown(T.probabilities(real, fake), unsafe_allow_html=True)

        reasons = [clean_text(reason) for reason in result.get("explanation") or []]
        if reasons:
            st.markdown("#### Signals behind the score")
            st.markdown(T.signals(reasons), unsafe_allow_html=True)

        generator = result.get("likely_generator")
        if is_fake and generator not in (None, "", "N/A", "Unknown"):
            st.markdown(
                T.callout(
                    f"Generator hint: {clean_text(generator)}. This is an unvalidated heuristic, not a measured finding."
                ),
                unsafe_allow_html=True,
            )

    with preview:
        image = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        with st.container(key=f"image_{digest(item['bytes'])}"):
            st.image(image, width="stretch")
        width, height = item.get("size", image.size)
        st.markdown(
            f'<div class="tl-file-meta">{T.safe(item["name"])} | {width} x {height}</div>',
            unsafe_allow_html=True,
        )
        export = {"file": item["name"], "model": settings["display"], **result}
        st.download_button(
            "Download result JSON",
            data=json.dumps(export, indent=2, default=str),
            file_name=f"truthlens_{digest(item['bytes'])}.json",
            mime="application/json",
            width="stretch",
            key=f"json_{digest(item['bytes'])}_{hashlib.sha1(str(item.get('name', '')).encode('utf-8', 'ignore')).hexdigest()[:8]}_{settings['model_type']}_{id(item)}",
        )

    if "ensemble_votes" in result:
        votes = {
            clean_text(name): clean_text(vote)
            for name, vote in result["ensemble_votes"].items()
        }
        st.markdown("#### Advisory model votes")
        st.markdown(T.kv_grid(votes), unsafe_allow_html=True)
        st.markdown(
            T.callout("These votes are context only. They do not change the ResNet18 verdict."),
            unsafe_allow_html=True,
        )


def reference_line(metric: str) -> str:
    ref = FORENSIC_REFERENCE.get(metric, {})
    if not ref:
        return "Not available"
    pattern = "{:.4f}" if metric == "hf_ratio" else "{:.2f}"
    return " | ".join(
        f"{label}: {pattern.format(mean)} +/- {pattern.format(std)}"
        for label, (mean, std) in ref.items()
    )


def render_attention(raw: bytes, heatmap: Optional[np.ndarray]) -> None:
    if heatmap is None:
        st.markdown(
            T.empty_state(
                "No attention map for this run",
                "Select a ResNet18 model and enable the attention map in model settings.",
            ),
            unsafe_allow_html=True,
        )
        return
    original, overlay = st.columns(2, gap="large")
    with original:
        st.image(Image.open(io.BytesIO(raw)).convert("RGB"), caption="Input image", width="stretch")
    with overlay:
        st.image(heatmap, caption="Model attention", width="stretch")
    st.markdown(
        T.callout(
            "Warmer regions contributed more to the predicted class. The map explains the selected ResNet18 label but does not validate it."
        ),
        unsafe_allow_html=True,
    )


def render_provenance(result: Dict[str, Any]) -> None:
    provenance = result.get("provenance_analysis") or {}
    if not provenance.get("available"):
        error = provenance.get("error", "unknown error")
        st.markdown(
            T.callout(f"Provenance analysis was unavailable: {clean_text(error)}", "error"),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        T.callout(
            "Metadata and credentials can support a review, but their absence does not prove an image is authentic. Platforms often remove them."
        ),
        unsafe_allow_html=True,
    )

    for signal in provenance.get("signals") or []:
        title = T.safe(signal.get("title", "Finding"))
        detail = T.safe(signal.get("detail", ""))
        kind = "caution" if signal.get("level") == "strong" else "info"
        st.markdown(
            T.callout(f"<strong>{title}</strong><br>{detail}", kind, allow_html=True),
            unsafe_allow_html=True,
        )

    attribution = provenance.get("ai_attribution") or {}
    if attribution.get("detected"):
        prov_title = f"Verified AI Provenance: {attribution.get('provider', 'AI')}"
        ev_text = " • ".join(attribution.get("evidence", []))
        st.markdown(
            T.callout(
                f"<strong>{prov_title}</strong><br>"
                f"Watermark Type: {attribution.get('watermark_type')}<br>"
                f"Evidence: {ev_text}",
                "caution",
                allow_html=True,
            ),
            unsafe_allow_html=True,
        )

    metadata = provenance.get("metadata") or {}
    credentials = provenance.get("content_credentials") or {}
    visible = provenance.get("visible_watermark") or {}
    invisible = provenance.get("invisible_watermark") or {}
    container = provenance.get("container") or {}
    fingerprints = provenance.get("fingerprints") or {}

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Container and credentials")
        st.markdown(
            T.kv_grid(
                [
                    ("Format", container.get("format", "Not available")),
                    ("Dimensions", " x ".join(map(str, container.get("size", []))) or "Not available"),
                    ("AI provider", attribution.get("provider", "Not detected")),
                    ("Watermark type", attribution.get("watermark_type", "None")),
                    ("EXIF", "Present" if metadata.get("exif_present") else "Not found"),
                    ("XMP", "Present" if metadata.get("xmp_present") else "Not found"),
                    ("ICC profile", "Present" if container.get("has_icc_profile") else "Not found"),
                    ("C2PA or JUMBF", "Marker found" if credentials.get("detected") else "Not found"),
                    ("Trailing data", f"{container.get('trailing_bytes', 0)} bytes"),
                    ("Perceptual dHash", fingerprints.get("dhash64", "Not available")),
                ],
                mono=True,
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Watermark checks")
        st.markdown(
            T.kv_grid(
                [
                    ("Visible overlay score", number(visible.get("score"), "{:.1f} / 100")),
                    ("Possible regions", ", ".join(visible.get("regions") or []) or "None"),
                    ("Embedded signal score", number(invisible.get("score"), "{:.1f} / 100")),
                    ("LSB entropy", number(invisible.get("lsb_entropy"))),
                    ("Explicit marker", ", ".join(invisible.get("explicit_markers") or []) or "None"),
                ]
            ),
            unsafe_allow_html=True,
        )

    fields = metadata.get("fields") or {}
    if fields:
        rows = [
            {"Field": clean_text(key), "Value": clean_text(value)}
            for key, value in fields.items()
        ]
        st.markdown("#### Embedded metadata")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_frequency(raw: bytes, result: Dict[str, Any]) -> None:
    spectrum = result.get("spectral_analysis") or {}
    visual, values = st.columns([1.08, 0.92], gap="large")
    with visual:
        image = fft_spectrum(raw)
        if image is None:
            st.markdown(
                T.callout("The frequency spectrum could not be computed.", "error"),
                unsafe_allow_html=True,
            )
        else:
            st.image(image, caption="Log magnitude frequency spectrum", width="stretch")
    with values:
        st.markdown("#### Frequency measurements")
        st.markdown(
            T.kv_grid(
                [
                    ("High-frequency energy ratio", number(spectrum.get("hf_ratio"))),
                    ("Descriptive spectral score", number(spectrum.get("spectral_ai_score"), "{:.1f}")),
                    ("Held-out reference", reference_line("hf_ratio")),
                    ("Signal AUC", f"{FORENSIC_AUC['fft']:.3f}"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.callout(
                "Descriptive only. Real and AI images overlap heavily on this metric, so it never changes the verdict."
            ),
            unsafe_allow_html=True,
        )


def render_compression(raw: bytes, result: Dict[str, Any]) -> None:
    metrics = result.get("ela_metrics") or {}
    visual, values = st.columns([1.08, 0.92], gap="large")
    with visual:
        image = ela_image(raw)
        if image is None:
            st.markdown(
                T.callout("The compression residual could not be computed.", "error"),
                unsafe_allow_html=True,
            )
        else:
            st.image(image, caption="Error-level difference map", width="stretch")
    with values:
        st.markdown("#### Compression measurements")
        st.markdown(
            T.kv_grid(
                [
                    ("Mean residual", number(metrics.get("ela_mean"), "{:.2f}")),
                    ("Residual spread", number(metrics.get("ela_std"), "{:.2f}")),
                    ("Mean reference", reference_line("ela_mean")),
                    ("Signal AUC", f"{FORENSIC_AUC['ela']:.3f}"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.callout(
                "This map shows where JPEG recompression changes the image most. It is descriptive and does not change the model score."
            ),
            unsafe_allow_html=True,
        )


def render_evidence(raw: bytes, result: Dict[str, Any], heatmap: Optional[np.ndarray]) -> None:
    st.markdown(
        T.section_head(
            "Inspect the evidence",
            "These panels add context for a human reviewer. Only the selected model produces the verdict.",
        ),
        unsafe_allow_html=True,
    )
    attention, provenance, frequency, compression = st.tabs(
        ["Attention", "Provenance", "Frequency", "Compression"]
    )
    with attention:
        render_attention(raw, heatmap)
    with provenance:
        render_provenance(result)
    with frequency:
        render_frequency(raw, result)
    with compression:
        render_compression(raw, result)


def render_batch_summary(rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    reals = sum(1 for r in rows if r["Verdict"] == "REAL")
    fakes = sum(1 for r in rows if r["Verdict"] == "FAKE")

    st.markdown(
        T.section_head(
            f"Batch overview — {total} images",
            f"Detected {reals} Real photographs and {fakes} AI-generated images.",
        ),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Scanned", total)
    c2.metric("Real Photos", reals)
    c3.metric("AI Fakes", fakes)

    frame = pd.DataFrame(rows)
    col_cfg: Dict[str, Any] = {
        "File": st.column_config.TextColumn("File", width="medium"),
        "Verdict": st.column_config.TextColumn("Verdict", width="small"),
        "Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        "P(fake)": st.column_config.ProgressColumn(
            "P(fake)", format="%.1f%%", min_value=0, max_value=100
        ),
        "Watermark / Provenance": st.column_config.TextColumn(
            "Watermark / Provenance", width="medium"
        ),
        "Basis": st.column_config.TextColumn("Basis", width="medium"),
    }
    if "Preview" in frame.columns:
        col_cfg = {
            "Preview": st.column_config.ImageColumn(
                "Image", width="small", help="Double-click to enlarge thumbnail"
            ),
            **col_cfg,
        }

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=col_cfg,
    )
    csv_df = frame.drop(columns=["Preview"]) if "Preview" in frame.columns else frame
    st.download_button(
        "Download batch CSV",
        data=csv_df.to_csv(index=False),
        file_name="truthlens_batch.csv",
        mime="text/csv",
        key="dl_batch_csv_frontend_v2",
    )


def score_items(items: List[Dict[str, Any]], settings: Dict[str, Any]) -> None:
    model_type, error = load_predictor(settings["model_type"])
    if model_type is None:
        st.markdown(
            T.callout(
                f"The selected model could not be loaded: {clean_text(error)}",
                "error",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.callout("Choose another checkpoint in Model settings or finish training this model."),
            unsafe_allow_html=True,
        )
        return

    loading = st.empty()
    loading.markdown(T.skeleton(), unsafe_allow_html=True)
    progress = st.progress(0.0, text="Analyzing images") if len(items) > 1 else None

    scored: List[Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]] = []
    summary: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        image, decode_error = load_image(item["bytes"])
        if image is None:
            st.markdown(
                T.callout(f"{item['name']}: {decode_error}", "error"),
                unsafe_allow_html=True,
            )
            continue
        item["size"] = image.size
        result, heatmap = analyse(item["bytes"], model_type, settings["with_gradcam"])
        if result is None:
            st.markdown(
                T.callout(f"{item['name']}: analysis failed. {clean_text(heatmap)}", "error"),
                unsafe_allow_html=True,
            )
            continue
        scored.append((item, result, heatmap))
        prov = result.get("provenance_analysis") or {}
        attr = result.get("ai_attribution") or prov.get("ai_attribution") or {}
        if result.get("provenance_verified") and attr.get("detected"):
            prov_badge = f"🛡️ {attr.get('provider', 'AI')} ({attr.get('watermark_type', 'watermark')})"
        else:
            prov_badge = "⚪ Checked (None detected)"

        thumb_uri = make_thumbnail_data_url(item["bytes"], max_size=96)
        item["thumbnail_uri"] = thumb_uri
        summary.append(
            {
                "Preview": thumb_uri,
                "File": item["name"],
                "Verdict": result["prediction"],
                "Confidence": float(result["confidence"]),
                "P(fake)": float(result["probabilities"]["FAKE"]),
                "Watermark / Provenance": prov_badge,
                "Basis": result.get("consensus_basis", "Neural model"),
            }
        )
        if progress is not None:
            progress.progress(index / len(items), text=f"Analyzed {index} of {len(items)}")

    loading.empty()
    if progress is not None:
        progress.empty()
    if not scored:
        st.markdown(
            T.empty_state(
                "No usable results",
                "Try a supported image or choose a model with an available checkpoint.",
            ),
            unsafe_allow_html=True,
        )
        return

    if len(scored) > 1:
        render_batch_summary(summary)
        st.markdown("<hr style='margin:16px 0; border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
        for item, result, heatmap in scored:
            w, h = item.get("size", (0, 0))
            dims = f"{w}×{h} px" if w and h else ""
            pred = result["prediction"]
            is_fake = pred == "FAKE"
            badge_color = T.FAKE if is_fake else T.REAL
            badge_bg = T.FAKE_BG if is_fake else T.REAL_BG

            dim_badge = f'<span style="font-size:0.75rem; color:{T.TEXT_3}; margin-left:8px;">{dims}</span>' if dims else ""
            with st.container():
                c_img, c_info = st.columns([1, 4], gap="medium")
                with c_img:
                    st.image(
                        Image.open(io.BytesIO(item["bytes"])).convert("RGB"),
                        width="stretch",
                    )
                with c_info:
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">'
                        f'<div>'
                        f'<span style="font-weight:700; font-size:1.02rem; color:{T.TEXT};">{clean_text(item["name"])}</span>'
                        f'{dim_badge}'
                        f'</div>'
                        f'<span style="padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.82rem; '
                        f'background:{badge_bg}; color:{badge_color}; border:1px solid {badge_color};">'
                        f'{pred} ({result["confidence"]:.1f}%)</span>'
                        f'</div>'
                        f'<div style="font-size:0.8rem; color:{T.TEXT_2}; margin-top:4px; margin-bottom:8px;">'
                        f'P(fake): <b>{result["probabilities"]["FAKE"]:.1f}%</b> · '
                        f'P(real): <b>{result["probabilities"]["REAL"]:.1f}%</b> · '
                        f'Basis: <b>{clean_text(str(result.get("consensus_basis", "Neural model")))}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("🔬 View full forensic analysis & details", expanded=False):
                        render_result_summary(item, result, settings)
                        render_evidence(item["bytes"], result, heatmap)
                st.markdown("<hr style='margin:16px 0; border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
        return

    item, result, heatmap = scored[0]
    render_result_summary(item, result, settings)
    render_evidence(item["bytes"], result, heatmap)


def render_analyze(settings: Dict[str, Any]) -> None:
    items = render_workspace()
    if items:
        score_items(items, settings)
    else:
        st.markdown(
            T.empty_state(
                "Your review will appear here",
                "Upload an image above or choose one of the held-out examples.",
            ),
            unsafe_allow_html=True,
        )


def swin_benchmark_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_type in (
        "swin_improved_ep6",
        "swin_hardstyles",
        "swin_ep9",
        "swin_improved_ep5",
    ):
        evaluation = find_swin_evaluation(model_type)
        holdout = evaluation.get("splits", {}).get("holdout", {}) if evaluation else {}
        if not holdout:
            continue
        rows.append(
            {
                "Model": model_label(model_type),
                "Accuracy": holdout.get("balanced_acc", holdout.get("accuracy")),
                "AUC": holdout.get("auc"),
                "Real acc": holdout.get("real_acc"),
                "Fake acc": holdout.get("fake_recall"),
            }
        )
    return rows


def render_benchmarks() -> None:
    swin_rows = swin_benchmark_rows()
    leader = swin_rows[0] if swin_rows else None
    headline = float(leader["Accuracy"]) if leader else 0.0
    lead_name = leader["Model"] if leader else "No Swin result available"
    st.markdown(
        '<div class="tl-benchmark-lead">'
        '<div><h2>Performance needs context.</h2>'
        '<p>Scores come directly from saved evaluation files. Different model families use different test sets, so they are shown separately.</p></div>'
        f'<div class="tl-headline-metric"><b>{headline:.1f}%</b>'
        f'<span>balanced accuracy on the independent holdout for {T.safe(lead_name)}</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if swin_rows:
        st.markdown(
            T.section_head(
                "Latest Swin checkpoints",
                "Independent holdout results. Accuracy here is balanced accuracy because the holdout is class-imbalanced.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(T.model_rows(swin_rows), unsafe_allow_html=True)

    legacy = benchmark_rows()
    if legacy:
        st.markdown(
            T.section_head(
                "Established baselines",
                "CLIP and ResNet use the held-out project test set. Classical models use a stratified 600-image subset.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(T.model_rows(legacy), unsafe_allow_html=True)

    st.markdown(
        T.callout(
            "Do not compare a number across sections without checking its evaluation split. A high score on the union test can fall on unfamiliar web images.",
            "caution",
        ),
        unsafe_allow_html=True,
    )

    tiers = resolution_rows()
    if tiers:
        st.markdown(
            T.section_head(
                "Resolution changes the result",
                "The ResNet benchmark spans three resolution tiers, and full-resolution uploads are its hardest group.",
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(tiers),
            width="stretch",
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.2f%%"),
                "Real acc": st.column_config.NumberColumn(format="%.1f%%"),
                "Fake acc": st.column_config.NumberColumn(format="%.1f%%"),
                "F1": st.column_config.NumberColumn(format="%.4f"),
                "AUC": st.column_config.NumberColumn(format="%.4f"),
                "Images": st.column_config.NumberColumn(format="%d"),
            },
        )


def render_method(settings: Dict[str, Any]) -> None:
    config = resolve_config()
    metrics = selected_model_metrics(settings["model_type"])
    metric_values = [
        ("Selected model", settings["display"]),
        ("Measured split", metrics.get("split", "Not available")),
        ("Accuracy", number(metrics.get("accuracy"), "{:.1f}%")),
        ("AUC", number(metrics.get("auc"), "{:.3f}")),
        ("Training", "In progress"),
        ("Upload storage", "Not persisted by this UI"),
    ]

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown(
            '<div class="tl-method-copy">'
            "<h2>What the result means</h2>"
            "<p>TruthLens compares visual patterns in an uploaded image with patterns learned from real photographs and generated images. The selected model alone produces the label and confidence.</p>"
            "<h3>Evidence stays separate</h3>"
            "<p>Metadata, content credentials, frequency spectra, compression residuals, and attention maps help a person inspect the file. Weak forensic heuristics never get blended into the verdict.</p>"
            "<h3>Known limits</h3>"
            "<ul>"
            "<li>New generators, heavy edits, screenshots, recompression, and unusual crops can shift an image outside the training distribution.</li>"
            "<li>Confidence is a model score, not a guarantee that the image history is known.</li>"
            "<li>Missing metadata is common and does not prove authenticity.</li>"
            "<li>Training is unfinished, so checkpoint behavior and published benchmarks can change.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Current review setup")
        st.markdown(T.kv_grid(metric_values), unsafe_allow_html=True)
        st.markdown("#### Project paths")
        st.markdown(
            T.kv_grid(
                [
                    ("CNN checkpoint", getattr(config, "cnn_model_name", "Not available")),
                    ("Swin checkpoint", getattr(config, "swin_model_name", "Not available")),
                    ("Data directory", getattr(config, "data_dir", "Not available")),
                    ("Inference entry", "src.predict.predict_image"),
                ],
                mono=True,
            ),
            unsafe_allow_html=True,
        )


def main() -> None:
    if "theme_dark" not in st.session_state:
        st.session_state["theme_dark"] = preferred_dark_mode()
    st.markdown(T.stylesheet(bool(st.session_state["theme_dark"])), unsafe_allow_html=True)
    settings = render_header_and_settings()

    analyze_tab, benchmark_tab, method_tab = st.tabs(["Analyze", "Benchmarks", "Method"])
    with analyze_tab:
        render_analyze(settings)
    with benchmark_tab:
        render_benchmarks()
    with method_tab:
        render_method(settings)


if __name__ == "__main__":
    main()
