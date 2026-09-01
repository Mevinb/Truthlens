#!/usr/bin/env python3
"""
TruthLens — app/app.py
======================
Streamlit frontend for AI-generated image detection.

This file is layout only. The model layer lives in ``app/inference.py`` and the
design system in ``app/theme.py``. Everything on screen goes through
``src.predict.predict_image``, so what the app reports is exactly what
``src/evaluate.py`` and ``diag_benchmark.py`` measure.

Run:
    streamlit run app/app.py
"""

import os
import sys

# Ensure the project root is importable when launched from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import html
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Page config (must precede every other Streamlit call) ───────────────────
st.set_page_config(
    page_title="TruthLens — AI image detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**TruthLens** — AI-generated image detection.\n\n"
            "Frozen CLIP ViT-L/14 with retrained calibrated linear heads is the "
            "default detector. The forensic panels are descriptive and never "
            "change the verdict."
        ),
    },
)

from app import theme as T                                        # noqa: E402
from app.inference import (                                       # noqa: E402
    CNN_TYPES,
    GRADCAM_TYPES,
    FORENSIC_AUC,
    FORENSIC_REFERENCE,
    MODEL_BLURB,
    MODEL_OPTIONS,
    benchmark,
    benchmark_rows,
    checkpoint_provenance,
    load_predictor,
    model_scores,
    resolution_rows,
    resolve_config,
    run_prediction,
    sample_images,
    tier_for_size,
    tier_scores,
    TIER_LABEL,
)

MAX_PIXELS = 40_000_000          # refuse decompression-bomb sized uploads
ACCEPTED = ["jpg", "jpeg", "png", "webp", "bmp"]


# ─── Cached work ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=48)
def analyse(img_bytes: bytes, model_type: str, with_gradcam: bool):
    """Predict on raw bytes.

    Cached, so moving the flag slider, switching tabs or any other rerun never
    re-runs inference on an image already scored. This replaces a session-state
    flag that keyed off the filename alone and so went stale whenever two
    different images shared a name.
    """
    # Pass original bytes, not a decoded RGB copy. Metadata, C2PA/JUMBF boxes,
    # PNG text chunks, alpha payloads, and trailing data only exist in the
    # encoded container. src.predict decodes a separate copy for the model.
    return run_prediction(model_type, img_bytes, with_gradcam=with_gradcam)


@st.cache_data(show_spinner=False, max_entries=48)
def fft_spectrum(img_bytes: bytes) -> Optional[np.ndarray]:
    """Log-magnitude 2D FFT of the luminance channel, as a viewable image."""
    import cv2

    try:
        arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))) + 1e-8)
        norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    except Exception as exc:                                      # noqa: BLE001
        logger.debug("FFT spectrum failed: %s", exc)
        return None


@st.cache_data(show_spinner=False, max_entries=48)
def ela_image(img_bytes: bytes) -> Optional[np.ndarray]:
    """Error-level-analysis difference map (needs a JPEG round-trip)."""
    from src.predict import compute_ela_analysis

    try:
        return compute_ela_analysis(Image.open(io.BytesIO(img_bytes)).convert("RGB"))[1]
    except Exception as exc:                                      # noqa: BLE001
        logger.debug("ELA image failed: %s", exc)
        return None


# ─── Small helpers ───────────────────────────────────────────────────────────
def load_image(raw: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Decode bytes to RGB, rejecting anything unreadable or absurdly large."""
    try:
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if w * h > MAX_PIXELS:
            return None, f"{w}×{h} ({w * h / 1e6:.0f} MP) is too large to process."
        return img.convert("RGB"), None
    except Exception as exc:                                      # noqa: BLE001
        return None, f"could not be decoded ({type(exc).__name__})."


def pct(v: Optional[float], digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}f}%"


def num(v: Any, fmt: str = "{:.4f}") -> str:
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "—"


def ref_line(metric: str) -> str:
    """'real 0.7197 ± 0.074 · fake 0.7249 ± 0.070' for a forensic metric."""
    ref = FORENSIC_REFERENCE.get(metric, {})
    if not ref:
        return ""
    fmt = "{:.4f}" if metric == "hf_ratio" else "{:.2f}"
    parts = [
        f'<span style="color:{T.REAL if cls == "real" else T.FAKE}">{cls}</span> '
        f"{fmt.format(m)} ± {fmt.format(s)}"
        for cls, (m, s) in ref.items()
    ]
    return " · ".join(parts)


def digest(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:10]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.15rem; font-weight:700; letter-spacing:-0.01em;">'
            f'Truth<span style="color:{T.ACCENT}">Lens</span></div>'
            f'<div style="font-size:0.75rem; color:{T.TEXT_3}; margin-top:2px;">'
            "detector controls</div><hr>",
            unsafe_allow_html=True,
        )

        display = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS),
            index=0,
            help=(
                "The SwinV2-Tiny transformer fine-tuned on the unified union "
                "corpus is the default. The retrained CLIP detector is kept for "
                "comparison, and the classical models only for reference."
            ),
        )
        model_type = MODEL_OPTIONS[display]
        is_cnn = model_type in GRADCAM_TYPES

        with_gradcam = st.toggle(
            "Grad-CAM heatmap",
            value=True,
            help="ResNet18 only — SwinV2 has no layer3/layer4 hooks for it.",
            disabled=not is_cnn,
        )

        flag_below = st.slider(
            "Flag results below (%)",
            min_value=50, max_value=99, value=70, step=1,
            help=(
                "Display only. Results under this confidence get a caution banner. "
                "The verdict is always the argmax at 50%, so moving this slider "
                "never changes a REAL/FAKE label."
            ),
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Measured scores read from results/metrics/benchmark_test_split.json —
        # not hardcoded, so re-running diag_benchmark.py updates the whole UI and
        # the displayed figures cannot drift from what was measured.
        m = model_scores(model_type)
        st.markdown(T.eyebrow("measured on held-out test"), unsafe_allow_html=True)
        if m:
            st.markdown(
                '<div class="tl-card tl-flush">'
                + T.stats([
                    ("accuracy", pct(m.get("accuracy"))),
                    ("f1", num(m.get("f1"), "{:.3f}")),
                    ("auc", num(m.get("auc"), "{:.3f}")),
                ])
                + '<div style="height:12px"></div>'
                + T.kv_rows([
                    ("catches real", pct(m.get("real_accuracy"))),
                    ("catches fake", pct(m.get("fake_accuracy"))),
                    ("scored on", f"{m['n']:,} images" if m.get("n") else "—"),
                ])
                + f'<div style="margin-top:12px; font-size:0.78rem; color:{T.TEXT_3}; '
                f'line-height:1.5;">{MODEL_BLURB.get(model_type, "")}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if (m.get("accuracy") or 0) < 70:
                st.markdown(
                    T.note(
                        "Chance is <b>50%</b>. This model is barely above it and clears "
                        "that bar mostly by calling things REAL. Use ResNet18 for "
                        "anything you intend to act on.",
                        "caution",
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                T.note(
                    "No benchmark on disk. Run <code>python diag_benchmark.py</code> to "
                    "populate the measured scores.",
                ),
                unsafe_allow_html=True,
            )

        # Which weights are actually loaded — surfaced so a stale or unexpected
        # checkpoint is visible rather than implied by an accuracy figure.
        prov = checkpoint_provenance(model_type)
        if prov:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(T.eyebrow("loaded checkpoint"), unsafe_allow_html=True)
            rows = [("file", prov.get("file", "—")), ("device", prov.get("device", "—"))]
            if prov.get("epoch") is not None:
                rows.append(("epoch", str(prov["epoch"])))
            if prov.get("val_acc") is not None:
                rows.append(("val acc", num(prov["val_acc"], "{:.2f}%")))
            st.markdown(
                f'<div class="tl-card tl-flush">{T.kv_rows(rows, mono=True)}</div>',
                unsafe_allow_html=True,
            )

    return {
        "display": display,
        "model_type": model_type,
        "with_gradcam": with_gradcam and is_cnn,
        "flag_below": flag_below,
    }


# ─── Input ───────────────────────────────────────────────────────────────────
def collect_inputs() -> List[Dict[str, Any]]:
    """Gather the images to score: uploads first, then any picked sample."""
    items: List[Dict[str, Any]] = []

    uploads = st.file_uploader(
        "Upload images",
        type=ACCEPTED,
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="JPEG, PNG, WebP or BMP. Several at once is fine.",
    )
    for up in uploads or []:
        items.append({"name": up.name, "bytes": up.getvalue(), "truth": None})

    samples = sample_images()
    if samples:
        with st.expander(f"No image handy? Try one of {len(samples)} held-out test images"):
            st.markdown(
                f'<div style="font-size:0.82rem; color:{T.TEXT_3}; margin-bottom:12px; '
                'line-height:1.6;">Drawn from <code>datasets/prepared/multires/test</code>, never '
                "from the training split — the model has not seen these. The true label "
                "is shown so you can check the answer against it.</div>",
                unsafe_allow_html=True,
            )
            for col, s in zip(st.columns(len(samples)), samples):
                with col:
                    try:
                        st.image(s["path"], width="stretch")
                    except Exception:                             # noqa: BLE001
                        pass
                    tone = T.FAKE if s["truth"] == "FAKE" else T.REAL
                    st.markdown(
                        '<div style="text-align:center; font-size:0.7rem; font-weight:700;'
                        f' letter-spacing:0.06em; color:{tone}; margin:-4px 0 8px;">'
                        f'{s["truth"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Use this", key=f"samp_{s['truth']}_{s['name']}",
                                 width="stretch"):
                        st.session_state["picked_sample"] = s

    picked = st.session_state.get("picked_sample")
    if picked:
        try:
            with open(picked["path"], "rb") as fh:
                items.append({
                    "name": f"{picked['truth'].lower()}/{picked['name']}",
                    "bytes": fh.read(),
                    "truth": picked["truth"],
                })
        except OSError as exc:
            st.markdown(T.note(f"Sample unreadable: {exc}", "error"),
                        unsafe_allow_html=True)
            st.session_state.pop("picked_sample", None)

    return items


# ─── Verdict ─────────────────────────────────────────────────────────────────
def render_verdict(item: Dict[str, Any], result: Dict[str, Any],
                   settings: Dict[str, Any]) -> None:
    pred = result["prediction"]
    conf = float(result["confidence"])
    is_fake = pred == "FAKE"
    real_pct = float(result["probabilities"]["REAL"])
    fake_pct = float(result["probabilities"]["FAKE"])

    acc = model_scores(settings["model_type"]).get("accuracy")
    reliability = (
        f"right {acc:.1f}% of the time on the held-out test split"
        if acc else "no measured accuracy on disk for this model"
    )

    # Quote the accuracy for this image's resolution tier when one exists. The
    # blended 92.8% headline runs several points above the high-resolution tier,
    # which is where essentially every real upload lands, so the aggregate on its
    # own overstates the case that matters most.
    tier_note = ""
    if settings["model_type"] in ("resnet18", "ensemble"):
        w, h = item.get("size", (0, 0))
        tier = tier_for_size(w, h) if w and h else None
        ts = tier_scores(tier) if tier else {}
        if ts.get("accuracy") is not None:
            reliability = (
                f"right {ts['accuracy']:.1f}% of the time on held-out "
                f"{TIER_LABEL[tier]} images"
            )
            if acc and abs(ts["accuracy"] - acc) >= 1.5:
                tier_note = (
                    f"Calibrated to this image's resolution. Across all tiers the "
                    f"model scores {acc:.1f}%, but on {TIER_LABEL[tier]} images it "
                    f"scores {ts['accuracy']:.1f}% — and catches "
                    f"{ts['fake_accuracy']:.1f}% of AI images in that tier."
                )

    inconclusive = bool(result.get("review_recommended"))
    st.markdown(
        T.verdict(
            "INCONCLUSIVE" if inconclusive
            else ("AI-GENERATED" if is_fake else "REAL PHOTOGRAPH"),
            is_fake,
            (
                f"Underlying score leans <b>{pred}</b> at {conf:.1f}%, but this "
                "image is outside the detector's reliable decision region."
                if inconclusive else
                f"<b>{conf:.1f}%</b> confidence · {settings['display']} · {reliability}"
            ),
            uncertain=inconclusive,
        ),
        unsafe_allow_html=True,
    )

    if tier_note:
        st.markdown(T.note(tier_note), unsafe_allow_html=True)

    if result.get("review_recommended"):
        why = (
            "the four CLIP views disagree"
            if settings["model_type"] == "clip"
            else "the score is not decisive enough off-distribution"
        )
        st.markdown(
            T.note(
                "<b>Inconclusive — manual review recommended.</b> The calibrated "
                f"score is close to the decision boundary or {why}. This is safer "
                "than presenting a forced binary answer as reliable evidence.",
                "caution",
            ),
            unsafe_allow_html=True,
        )

    # Samples carry a known label, so the app can be checked rather than believed.
    if item.get("truth"):
        correct = item["truth"] == pred
        if inconclusive:
            sample_message = (
                f"Abstained — this sample's true label is <b>{item['truth']}</b>. "
                "The underlying model lean is shown for inspection but was not "
                "accepted as a reliable verdict."
            )
            sample_tone = "info"
        else:
            sample_message = (
                f"{'Correct' if correct else 'Wrong'} — this sample's true label is "
                f"<b>{item['truth']}</b>."
            )
            sample_tone = "info" if correct else "caution"
        st.markdown(
            T.note(sample_message, sample_tone),
            unsafe_allow_html=True,
        )

    if conf < settings["flag_below"]:
        extra = (
            "" if settings["model_type"] in CNN_TYPES
            else " Try ResNet18 — the classical models score near chance on this corpus."
        )
        st.markdown(
            T.note(
                f"Confidence {conf:.1f}% is under your {settings['flag_below']}% flag "
                f"threshold. Treat this as undecided rather than as a call.{extra}",
                "caution",
            ),
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(T.eyebrow("probability split"), unsafe_allow_html=True)
        st.markdown(T.split_bar(real_pct, fake_pct), unsafe_allow_html=True)

        reasons = [str(r) for r in (result.get("explanation") or [])]
        if reasons:
            st.markdown('<div style="height:16px"></div>' + T.eyebrow("why"),
                        unsafe_allow_html=True)
            st.markdown(T.reasons(reasons), unsafe_allow_html=True)

        gen = result.get("likely_generator")
        if is_fake and gen not in (None, "", "N/A", "Unknown"):
            st.markdown(
                '<div style="height:10px"></div>'
                + T.eyebrow("generator guess — unvalidated")
                + T.chip(f"🎨 {gen}")
                + f'<div style="font-size:0.76rem; color:{T.TEXT_3}; margin-top:10px; '
                'line-height:1.55;">A rule-based heuristic over resolution and colour '
                "statistics. The corpus carries no per-generator labels, so this has "
                "never been measured against ground truth — read it as a hint, not a "
                "finding.</div>",
                unsafe_allow_html=True,
            )

    with right:
        st.image(Image.open(io.BytesIO(item["bytes"])).convert("RGB"),
                 caption=item["name"], width="stretch")
        st.download_button(
            "Download result JSON",
            data=json.dumps(
                {"file": item["name"], "model": settings["display"], **result},
                indent=2, default=str,
            ),
            file_name=f"truthlens_{digest(item['bytes'])}.json",
            mime="application/json",
            width="stretch",
            key=f"dl_{digest(item['bytes'])}_{settings['model_type']}",
        )

    if "ensemble_votes" in result:
        st.markdown(
            '<div style="height:8px"></div>'
            + T.eyebrow("advisory signals — none of these change the verdict"),
            unsafe_allow_html=True,
        )
        votes = list(result["ensemble_votes"].items())
        cols = st.columns(min(len(votes), 4) or 1)
        for i, (name, vote) in enumerate(votes):
            tone = T.FAKE if vote == "FAKE" else T.REAL
            with cols[i % len(cols)]:
                st.markdown(
                    '<div class="tl-card tl-flush" style="margin-bottom:8px;">'
                    f'<div style="font-size:0.72rem; color:{T.TEXT_3};">{name}</div>'
                    f'<div style="font-weight:700; color:{tone}; font-size:0.95rem;">'
                    f"{vote}</div></div>",
                    unsafe_allow_html=True,
                )


# ─── Evidence panels ─────────────────────────────────────────────────────────
def render_evidence(item: Dict[str, Any], result: Dict[str, Any],
                    heatmap: Optional[np.ndarray]) -> None:
    """Grad-CAM, frequency/compression diagnostics, and provenance checks.

    The forensic panels are descriptive. Measured alone they reach AUC 0.562
    (FFT) and 0.514 (ELA) against a 0.500 chance line, and blending either into
    the CNN score lowers AUC monotonically — see diag_forensics.py. The previous
    version of this screen announced "high-frequency VAE grid signature
    detected!" above hf_ratio 0.72 and printed an "% AI probability" for both
    signals. Real photographs in this corpus average hf_ratio 0.7197, i.e. that
    threshold sits on top of the real-image mean and fires for about half of
    them. Each panel now shows the per-class reference range instead, so the
    overlap is visible rather than hidden behind a verdict-shaped claim.
    """
    raw = item["bytes"]
    st.markdown("<hr>", unsafe_allow_html=True)
    tab_cam, tab_prov, tab_fft, tab_ela = st.tabs(
        ["Attention", "Provenance & watermarks", "Frequency spectrum", "Compression residual"]
    )

    with tab_cam:
        if heatmap is not None:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(T.eyebrow("input"), unsafe_allow_html=True)
                st.image(Image.open(io.BytesIO(raw)).convert("RGB"),
                         width="stretch")
            with c2:
                st.markdown(T.eyebrow("grad-cam overlay"), unsafe_allow_html=True)
                st.image(heatmap, width="stretch")
            st.markdown(
                T.note(
                    f"Warm regions contributed most to the <b>{result['prediction']}</b> "
                    "score, cool regions least. Gradients are taken from the predicted "
                    "class at <code>layer3</code> and <code>layer4</code>, so the map "
                    "always explains the label shown above it."
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                T.note(
                    "Grad-CAM needs a convolutional network. Pick <b>ResNet18</b> in the "
                    "sidebar and leave the heatmap toggle on."
                ),
                unsafe_allow_html=True,
            )

    with tab_prov:
        prov = result.get("provenance_analysis") or {}
        if not prov.get("available"):
            st.markdown(
                T.note(
                    "Container and metadata analysis was unavailable: "
                    f"<code>{html.escape(str(prov.get('error', 'unknown error')))}</code>",
                    "error",
                ),
                unsafe_allow_html=True,
            )
        else:
            signals = prov.get("signals") or []
            metadata = prov.get("metadata") or {}
            credentials = prov.get("content_credentials") or {}
            visible = prov.get("visible_watermark") or {}
            invisible = prov.get("invisible_watermark") or {}
            container = prov.get("container") or {}
            fingerprints = prov.get("fingerprints") or {}

            st.markdown(
                T.note(
                    "<b>Independent provenance checks.</b> Explicit generator metadata "
                    "or a verifiable content credential can be meaningful. Visible and "
                    "invisible watermark scores are heuristics only and never change the "
                    "REAL/FAKE verdict. Provider watermarks such as SynthID require the "
                    "provider's detector for confirmation."
                ),
                unsafe_allow_html=True,
            )
            if signals:
                st.markdown(T.eyebrow("findings"), unsafe_allow_html=True)
                for signal in signals:
                    title = html.escape(str(signal.get("title", "Finding")))
                    detail = html.escape(str(signal.get("detail", "")))
                    st.markdown(
                        T.note(f"<b>{title}</b><br>{detail}", "caution"
                               if signal.get("level") == "strong" else "info"),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    T.note(
                        "No metadata, content-credential, watermark, or container "
                        "indicator was found. This is not proof of authenticity: most "
                        "platforms strip metadata and watermarks can be removed."
                    ),
                    unsafe_allow_html=True,
                )

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(T.eyebrow("container & credentials"),
                            unsafe_allow_html=True)
                st.markdown(
                    '<div class="tl-card tl-flush">'
                    + T.kv_rows([
                        ("format", container.get("format", "—")),
                        ("dimensions", "×".join(map(str, container.get("size", []))) or "—"),
                        ("EXIF", "present" if metadata.get("exif_present") else "not found"),
                        ("XMP", "present" if metadata.get("xmp_present") else "not found"),
                        ("ICC profile", "present" if container.get("has_icc_profile") else "not found"),
                        ("C2PA / JUMBF", "marker found" if credentials.get("detected") else "not found"),
                        ("trailing data", f"{container.get('trailing_bytes', 0)} bytes"),
                        ("perceptual dHash", fingerprints.get("dhash64", "—")),
                    ], mono=True)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(T.eyebrow("watermark checks"), unsafe_allow_html=True)
                st.markdown(
                    '<div class="tl-card tl-flush">'
                    + T.kv_rows([
                        ("visible overlay heuristic", num(visible.get("score"), "{:.1f}/100")),
                        ("possible regions", ", ".join(visible.get("regions") or []) or "none"),
                        ("invisible/embed heuristic", num(invisible.get("score"), "{:.1f}/100")),
                        ("LSB entropy", num(invisible.get("lsb_entropy"))),
                        ("explicit marker", ", ".join(invisible.get("explicit_markers") or []) or "none"),
                    ])
                    + "</div>",
                    unsafe_allow_html=True,
                )

            fields = metadata.get("fields") or {}
            if fields:
                import pandas as pd

                st.markdown(
                    T.eyebrow(
                        f"metadata fields shown — {len(fields)} of "
                        f"{metadata.get('field_count', len(fields))}"
                    ),
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"Field": str(k), "Value": str(v)} for k, v in fields.items()]
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with tab_fft:
        spec = result.get("spectral_analysis") or {}
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown(T.eyebrow("log-magnitude spectrum"), unsafe_allow_html=True)
            img = fft_spectrum(raw)
            if img is not None:
                st.image(img, width="stretch")
            else:
                st.markdown(T.note("Spectrum could not be computed.", "error"),
                            unsafe_allow_html=True)
        with c2:
            st.markdown(T.eyebrow("this image"), unsafe_allow_html=True)
            st.markdown(
                '<div class="tl-card tl-flush">'
                + T.kv_rows([
                    ("high-frequency energy ratio", num(spec.get("hf_ratio"))),
                    ("descriptive spectral score",
                     num(spec.get("spectral_ai_score"), "{:.1f}")),
                ])
                + f'<div style="margin-top:12px; font-size:0.76rem; color:{T.TEXT_3}; '
                f'line-height:1.6;">test-split mean ± std<br>{ref_line("hf_ratio")}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                T.note(
                    "Descriptive only — this never affects the verdict. Real and AI "
                    "images overlap almost completely here: the class means differ by "
                    "0.005 against a spread of 0.07. Used alone as a classifier the "
                    f"signal reaches <b>AUC {FORENSIC_AUC['fft']:.3f}</b> where 0.500 "
                    "is a coin flip, and adding it to the CNN score makes the CNN "
                    "worse. Sharpness and clean upscaling push it up regardless of "
                    "origin."
                ),
                unsafe_allow_html=True,
            )

    with tab_ela:
        ela = result.get("ela_metrics") or {}
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown(T.eyebrow("ela difference map"), unsafe_allow_html=True)
            img = ela_image(raw)
            if img is not None:
                st.image(img, width="stretch")
            else:
                st.markdown(
                    T.note("ELA needs a JPEG round-trip and it failed for this image.",
                           "error"),
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown(T.eyebrow("this image"), unsafe_allow_html=True)
            st.markdown(
                '<div class="tl-card tl-flush">'
                + T.kv_rows([
                    ("mean error", num(ela.get("ela_mean"), "{:.2f}")),
                    ("error std", num(ela.get("ela_std"), "{:.2f}")),
                    ("descriptive score", num(ela.get("ela_ai_score"), "{:.1f}")),
                ])
                + f'<div style="margin-top:12px; font-size:0.76rem; color:{T.TEXT_3}; '
                'line-height:1.6;">test-split mean &#177; std<br>'
                f'mean error — {ref_line("ela_mean")}<br>'
                f'error std — {ref_line("ela_std")}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                T.note(
                    "Re-compresses the image and maps where the error is largest — edges "
                    "and texture light up, flat areas stay dark. Descriptive only: alone "
                    f"it reaches <b>AUC {FORENSIC_AUC['ela']:.3f}</b> against 0.500 for "
                    "chance, and the two classes are statistically indistinguishable on "
                    "both metrics above. The common claim that synthetic images show a "
                    "low error spread does not hold on this corpus, so no threshold here "
                    "is a reliable tell."
                ),
                unsafe_allow_html=True,
            )


# ─── Batch summary ───────────────────────────────────────────────────────────
def render_batch_summary(rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    st.markdown(T.eyebrow(f"batch summary — {len(rows)} images"), unsafe_allow_html=True)
    df = pd.DataFrame(rows)
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "Confidence": st.column_config.NumberColumn(format="%.1f%%"),
            "P(fake)": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )
    st.download_button(
        "Download batch CSV",
        data=df.to_csv(index=False),
        file_name="truthlens_batch.csv",
        mime="text/csv",
        key="dl_batch_csv",
    )


# ─── Detect tab ──────────────────────────────────────────────────────────────
def render_detect(settings: Dict[str, Any]) -> None:
    items = collect_inputs()

    if not items:
        st.markdown(
            T.empty_state(
                "Upload an image to check whether it is a photograph or AI-generated",
                "JPEG · PNG · WebP · BMP — or open the sample drawer above",
            ),
            unsafe_allow_html=True,
        )
        return

    model_type, load_error = load_predictor(settings["display"])
    if model_type is None:
        st.markdown(
            T.note(f"<b>{settings['display']} could not be loaded.</b><br>"
                   f"<code>{load_error}</code>", "error"),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.note(
                "Train that model first, or pick another in the sidebar. "
                "<code>python src/train.py --data-dir datasets/prepared/multires</code> for the "
                "CNN; <code>python src/train_classical.py</code> for the rest."
            ),
            unsafe_allow_html=True,
        )
        return

    scored: List[Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]] = []
    summary: List[Dict[str, Any]] = []
    progress = st.progress(0.0, text="Analysing…") if len(items) > 1 else None

    for i, item in enumerate(items, start=1):
        img, err = load_image(item["bytes"])
        if img is None:
            st.markdown(T.note(f"<b>{item['name']}</b> {err}", "error"),
                        unsafe_allow_html=True)
            continue
        item["size"] = img.size

        with st.spinner(f"Analysing {item['name']} with {settings['display']}…"):
            result, heatmap = analyse(
                item["bytes"], model_type, settings["with_gradcam"]
            )

        if result is None:
            st.markdown(
                T.note(f"<b>{item['name']}</b> — prediction failed: "
                       f"<code>{heatmap}</code>", "error"),
                unsafe_allow_html=True,
            )
            continue

        scored.append((item, result, heatmap))
        summary.append({
            "File":       item["name"],
            "Verdict":    (
                "INCONCLUSIVE" if result.get("review_recommended")
                else result["prediction"]
            ),
            "Model lean": result["prediction"],
            "Confidence": float(result["confidence"]),
            "P(fake)":    float(result["probabilities"]["FAKE"]),
            "Provenance findings": len(
                (result.get("provenance_analysis") or {}).get("signals") or []
            ),
            "Truth":      item.get("truth") or "—",
        })
        if progress:
            progress.progress(i / len(items), text=f"Analysing… {i}/{len(items)}")

    if progress:
        progress.empty()
    if not scored:
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    if len(scored) > 1:
        render_batch_summary(summary)
        st.markdown("<hr>", unsafe_allow_html=True)
        for item, result, heatmap in scored:
            with st.expander(f"{item['name']} → {result['prediction']} "
                             f"({result['confidence']:.1f}%)"):
                render_verdict(item, result, settings)
                render_evidence(item, result, heatmap)
    else:
        item, result, heatmap = scored[0]
        render_verdict(item, result, settings)
        render_evidence(item, result, heatmap)


# ─── Benchmarks tab ──────────────────────────────────────────────────────────
def render_benchmarks() -> None:
    rows = benchmark_rows()
    if not rows:
        st.markdown(
            T.note(
                "No benchmark file found. Run <code>python diag_benchmark.py</code> — it "
                "writes <code>results/metrics/benchmark_test_split.json</code>, which "
                "both this page and the sidebar read."
            ),
            unsafe_allow_html=True,
        )
        return

    split = benchmark().get("split", "the held-out test split")
    st.markdown(
        '<div class="tl-card"><h4>Measured performance</h4>'
        f"<p>Every shipped model, scored on <code>{split}</code> by "
        "<code>diag_benchmark.py</code>. The CNN is scored on all 2,620 test images; "
        "the classical models on a 600-image stratified subsample, because HOG+LBP "
        "extraction is the bottleneck. Nothing here is hardcoded — the page reads the "
        "benchmark file, so it cannot drift from what was measured.</p></div>",
        unsafe_allow_html=True,
    )

    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_bar(
        x=df["Model"], y=df["Accuracy"],
        marker_color=[T.ACCENT if a >= 70 else T.TEXT_3 for a in df["Accuracy"]],
        hovertemplate="%{x}<br>accuracy %{y:.2f}%<extra></extra>",
    )
    fig.add_hline(
        y=50, line_dash="dot", line_color=T.CAUTION,
        annotation_text="chance (50%)", annotation_position="top left",
        annotation_font_color=T.CAUTION,
    )
    fig.update_layout(
        height=330, margin=dict(l=0, r=0, t=14, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T.TEXT_2, size=11),
        yaxis=dict(title="accuracy (%)", range=[0, 100], gridcolor=T.LINE),
        xaxis=dict(tickangle=-20), showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.2f%%"),
            "Real acc": st.column_config.NumberColumn(format="%.1f%%"),
            "Fake acc": st.column_config.NumberColumn(format="%.1f%%"),
            "F1":       st.column_config.NumberColumn(format="%.4f"),
            "AUC":      st.column_config.NumberColumn(format="%.4f"),
            "Images":   st.column_config.NumberColumn(format="%d"),
        },
    )

    st.markdown(
        T.note(
            "<b>Read the fake-accuracy column.</b> Every classical model clears 50% "
            "overall only by calling almost everything REAL — they catch 21–48% of "
            "actual fakes. HOG+LBP texture statistics do not separate modern generators "
            "from photographs on this corpus. Earlier revisions of this project quoted "
            "CIFAKE-era figures (CNN ~98%, Random Forest ~78%, SVM ~76%); none of those "
            "held here.",
            "caution",
        ),
        unsafe_allow_html=True,
    )

    tiers = resolution_rows()
    if tiers:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="tl-card"><h4>The headline hides a gap</h4>'
            "<p>The corpus spans three resolution tiers, and the CNN does not perform "
            "equally across them. Almost every real upload is a full-resolution image, "
            "which is the <i>worst</i> tier — so the blended figure above flatters the "
            "case that matters most. Each result page quotes the tier-specific number "
            "for the image you gave it rather than the blended one.</p></div>",
            unsafe_allow_html=True,
        )
        import pandas as pd

        st.dataframe(
            pd.DataFrame(tiers), width="stretch", hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.2f%%"),
                "Real acc": st.column_config.NumberColumn(format="%.1f%%"),
                "Fake acc": st.column_config.NumberColumn(format="%.1f%%"),
                "F1":       st.column_config.NumberColumn(format="%.4f"),
                "AUC":      st.column_config.NumberColumn(format="%.4f"),
                "Images":   st.column_config.NumberColumn(format="%d"),
            },
        )


# ─── About tab ───────────────────────────────────────────────────────────────
def render_about() -> None:
    cfg = resolve_config()
    prov = checkpoint_provenance()
    cnn = model_scores("resnet18")

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown(
            "<div class='tl-card'><h4>What this does</h4>"
            "<p>TruthLens takes an image and returns one of two labels — a real "
            "photograph, or AI-generated — with a confidence score and a visual "
            "explanation of which regions drove the decision.</p>"
            "<p>The verdict comes from a single ResNet18, fine-tuned from ImageNet "
            "weights and run with horizontal-flip and five-crop test-time augmentation "
            "averaged in probability space. The frequency and compression panels beside "
            "each result are diagnostics for a human to look at; they are never mixed "
            "into the score.</p></div>"

            "<div class='tl-card'><h4>How it was measured</h4>"
            "<p>Model selection used the validation split. Every number in this app "
            "comes from a held-out test split that was never trained or tuned on, and "
            "the evaluation runs the same code the app serves — "
            "<code>src/evaluate.py</code> drives <code>CNNPredictor.tta_probs</code> "
            "rather than the bare checkpoint. Measuring a path that nothing serves is "
            "how a ten-point accuracy gap went unnoticed here for a while.</p></div>"

            "<div class='tl-card'><h4>Known limits</h4><ul>"
            "<li>Trained on one specific corpus. Generators released after it, heavy "
            "editing, screenshots, recompression and unusual crops all move an image "
            "away from what the model has seen.</li>"
            "<li>Accuracy is not uniform across resolutions. The blended 92.8% is a mix "
            "of three tiers; on full-resolution images — where nearly every real upload "
            "lands — it is 88.5%, and about one AI image in seven gets through. Each "
            "result quotes the tier-specific figure.</li>"
            "<li>The generator guess on fake images is an unvalidated heuristic — the "
            "corpus has no per-generator labels, so its accuracy is simply unknown.</li>"
            "<li>The six classical models sit near chance and are kept for comparison, "
            "not for use.</li>"
            f"<li>The forensic panels reach AUC {FORENSIC_AUC['fft']:.3f} and "
            f"{FORENSIC_AUC['ela']:.3f} on their own, against 0.500 for chance. They are "
            "texture to look at, not evidence.</li>"
            "<li>A confident answer is not a certain one. At 92.8% accuracy, roughly one "
            "image in fourteen is called wrong.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(T.eyebrow("headline result"), unsafe_allow_html=True)
        st.markdown(
            '<div class="tl-card tl-flush">'
            + T.stats([
                ("accuracy", pct(cnn.get("accuracy"))),
                ("auc", num(cnn.get("auc"), "{:.3f}")),
                ("f1", num(cnn.get("f1"), "{:.3f}")),
            ])
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(T.eyebrow("corpus"), unsafe_allow_html=True)
        st.markdown(
            '<div class="tl-card tl-flush">'
            + T.kv_rows([
                ("high-res", "512–2048px · DALL·E 3 / SDXL / Midjourney vs. photos"),
                ("medium-res", "256px multi-domain"),
                ("low-res", "32px CIFAKE"),
                ("train", "20,960 images"),
                ("val", "2,620 images"),
                ("test", "2,620 images"),
                ("balance", "50 / 50 in every split"),
            ])
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(T.eyebrow("runtime"), unsafe_allow_html=True)
        st.markdown(
            '<div class="tl-card tl-flush">'
            + T.kv_rows([
                ("checkpoint", prov.get("file", cfg.cnn_model_name)),
                ("device", prov.get("device", "—")),
                ("data dir", str(cfg.data_dir)),
                ("stack", "PyTorch · scikit-learn · OpenCV · Streamlit"),
            ], mono=True)
            + "</div>",
            unsafe_allow_html=True,
        )


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(T.stylesheet(), unsafe_allow_html=True)
    settings = render_sidebar()

    cnn = model_scores("resnet18")
    badge = (
        T.chip(f"ResNet18 · {cnn['accuracy']:.1f}% test accuracy", neutral=True)
        if cnn.get("accuracy") else ""
    )
    st.markdown(T.masthead(badge), unsafe_allow_html=True)

    tab_detect, tab_bench, tab_about = st.tabs(["Detect", "Benchmarks", "About"])
    with tab_detect:
        render_detect(settings)
    with tab_bench:
        render_benchmarks()
    with tab_about:
        render_about()


if __name__ == "__main__":
    # Streamlit execs this file with __name__ == "__main__", so the page renders
    # under `streamlit run` but importers (tests, diag_app_paths.py) get the
    # helpers without a rendered page or a ScriptRunContext warning.
    main()
