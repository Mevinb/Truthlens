#!/usr/bin/env python3
"""
TruthLens — app/app.py
========================
Production-grade Streamlit web application for AI-Generated Image Detection.

Features:
  • Drag-and-drop image upload
  • Real / Fake classification with confidence gauge
  • Grad-CAM heatmap overlay
  • AI Explanation bullets
  • Generator detection (for fake images)
  • Model selector (ResNet18 + 6 classical models)
  • Model comparison leaderboard
  • Dark glassmorphism UI

Run:
    streamlit run app/app.py
"""

import sys
import os

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Page Config (MUST be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="TruthLens — AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-username/truthlens",
        "Report a bug": "https://github.com/your-username/truthlens/issues",
        "About": "**TruthLens** — AI-Generated Image Detection using Deep Learning.",
    },
)


# ─── Custom CSS (Dark Glassmorphism Theme) ─────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── Global Reset ──────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .stApp {
            background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #11182f 100%);
            min-height: 100vh;
        }

        /* ── Sidebar ───────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(99, 102, 241, 0.2);
        }

        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* ── Header gradient ────────────────────────────────── */
        .hero-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.2rem;
            font-weight: 900;
            letter-spacing: -1.5px;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }

        .hero-sub {
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }

        /* ── Glass Card ─────────────────────────────────────── */
        .glass-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 28px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            margin-bottom: 20px;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── Result Badge ───────────────────────────────────── */
        .badge-real {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(16,185,129,0.1));
            border: 2px solid rgba(34,197,94,0.5);
            border-radius: 100px;
            padding: 14px 28px;
            font-size: 1.5rem;
            font-weight: 700;
            color: #4ade80;
            box-shadow: 0 0 30px rgba(34,197,94,0.2);
            animation: pulse-green 2s infinite;
        }

        .badge-fake {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
            border: 2px solid rgba(239,68,68,0.5);
            border-radius: 100px;
            padding: 14px 28px;
            font-size: 1.5rem;
            font-weight: 700;
            color: #f87171;
            box-shadow: 0 0 30px rgba(239,68,68,0.2);
            animation: pulse-red 2s infinite;
        }

        @keyframes pulse-green {
            0%,100% { box-shadow: 0 0 20px rgba(34,197,94,0.2); }
            50%      { box-shadow: 0 0 40px rgba(34,197,94,0.4); }
        }
        @keyframes pulse-red {
            0%,100% { box-shadow: 0 0 20px rgba(239,68,68,0.2); }
            50%      { box-shadow: 0 0 40px rgba(239,68,68,0.4); }
        }

        /* ── Confidence Gauge ───────────────────────────────── */
        .confidence-wrap {
            margin: 20px 0;
        }
        .conf-label {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .conf-value {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 10px;
        }
        .conf-real { color: #4ade80; }
        .conf-fake { color: #f87171; }

        /* ── Explanation bullets ────────────────────────────── */
        .reason-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px 14px;
            background: rgba(255,255,255,0.04);
            border-left: 3px solid #6366f1;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .reason-icon { color: #6366f1; font-size: 1rem; }

        /* ── Generator chip ─────────────────────────────────── */
        .generator-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(139,92,246,0.15);
            border: 1px solid rgba(139,92,246,0.4);
            border-radius: 50px;
            padding: 6px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #c4b5fd;
        }

        /* ── Section Labels ─────────────────────────────────── */
        .section-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #64748b;
            margin-bottom: 10px;
        }

        /* ── Metric Row ─────────────────────────────────────── */
        .metric-row {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .metric-pill {
            flex: 1;
            min-width: 90px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
        }
        .metric-pill .mp-val {
            font-size: 1.3rem;
            font-weight: 700;
            color: #e2e8f0;
        }
        .metric-pill .mp-lbl {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
        }

        /* ── Upload zone ─────────────────────────────────────── */
        [data-testid="stFileUploadDropzone"] {
            background: rgba(99,102,241,0.05) !important;
            border: 2px dashed rgba(99,102,241,0.4) !important;
            border-radius: 16px !important;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            background: rgba(99,102,241,0.1) !important;
            border-color: rgba(99,102,241,0.7) !important;
        }

        /* ── Streamlit element overrides ─────────────────────── */
        .stSelectbox > div > div {
            background: rgba(255,255,255,0.05) !important;
            border-color: rgba(255,255,255,0.1) !important;
            border-radius: 10px !important;
            color: #e2e8f0 !important;
        }
        .stSlider > div { padding: 0 !important; }
        h1, h2, h3 { color: #f1f5f9 !important; }
        p, li { color: #94a3b8; }

        /* ── Scrollbar ──────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

        /* ── Divider ─────────────────────────────────────────── */
        hr { border-color: rgba(255,255,255,0.06) !important; }

        /* ── Streamlit button ────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
        }

        /* ── Tabs ────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border: none !important;
            color: #64748b !important;
            font-weight: 500;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            color: #6366f1 !important;
            border-bottom: 2px solid #6366f1 !important;
        }

        /* ── Info boxes ──────────────────────────────────────── */
        .info-note {
            background: rgba(6,182,212,0.08);
            border: 1px solid rgba(6,182,212,0.25);
            border-radius: 10px;
            padding: 12px 16px;
            color: #67e8f9;
            font-size: 0.85rem;
            margin-bottom: 12px;
        }
        .warn-note {
            background: rgba(245,158,11,0.08);
            border: 1px solid rgba(245,158,11,0.25);
            border-radius: 10px;
            padding: 12px 16px;
            color: #fcd34d;
            font-size: 0.85rem;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Helper: Try loading predict module ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_predictor(model_type: str):
    """Cache only successfully constructed predictors."""
    from src.utils import Config
    cfg = Config()
    for candidate in ["resnet18_truthlens.pth", "resnet18_highres.pth"]:
        if (cfg.models_dir / candidate).exists():
            cfg.cnn_model_name = candidate
            break

    if "Ensemble" in model_type:
        from src.predict import EnsemblePredictor
        return EnsemblePredictor(cfg), "ensemble"
    elif "ResNet" in model_type or "CNN" in model_type or "High-Res" in model_type or "Gemini" in model_type:
        from src.predict import CNNPredictor
        return CNNPredictor(cfg), "cnn"
    else:
        model_key_map = {
            "SVM (RBF Kernel)":    "svm",
            "Random Forest":       "random_forest",
            "Logistic Regression": "logistic_regression",
            "k-NN":                "knn",
            "Decision Tree":       "decision_tree",
            "Naive Bayes":         "naive_bayes",
        }
        key = model_key_map.get(model_type, "svm")
        from src.predict import ClassicalPredictor
        return ClassicalPredictor(key, cfg), "classical"


def load_predictor(model_type: str):
    """Return a predictor or an uncached error message."""
    try:
        return _load_predictor(model_type)
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:
        logger.exception("Failed to load predictor %s", model_type)
        return None, f"{type(exc).__name__}: {exc}"


def run_prediction(predictor, predictor_type: str, img: Image.Image, with_gradcam: bool):
    """Run inference and return (result, heatmap_or_None)."""
    from src.predict import generate_explanation, detect_likely_generator, CNNPredictor
    from src.utils import Config
    import numpy as np

    img_np = np.array(img.convert("RGB"))

    try:
        if predictor_type == "ensemble":
            result = predictor.predict(img)
            heatmap = None
            if with_gradcam:
                try:
                    _, heatmap = predictor.cnn_predictor.predict_with_gradcam(img)
                except Exception:
                    heatmap = None
        elif predictor_type == "cnn" and with_gradcam:
            result, heatmap = predictor.predict_with_gradcam(img)
        elif predictor_type == "cnn":
            result  = predictor.predict(img)
            heatmap = None
        else:
            result  = predictor.predict(img)
            heatmap = None

        # Enrich
        result["explanation"]      = generate_explanation(result, img_np)
        result["likely_generator"] = (
            detect_likely_generator(img_np)
            if result["prediction"] == "FAKE" else "N/A"
        )
        return result, heatmap

    except Exception as exc:
        return None, str(exc)


# ─── Sidebar ───────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 10px 0 20px;'>
              <div style='font-size:2.5rem;'>🔍</div>
              <div style='font-size:1.2rem; font-weight:800; color:#e2e8f0;'>TruthLens</div>
              <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>AI Image Detector v1.1</div>
            </div>
            <hr>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚙️ Settings")

        model_choice = st.selectbox(
            "Model Architecture",
            options=[
                "ResNet18 (CNN - Primary 98.2% Accuracy)",
                "ResNet18 (CNN - High-Res 512px+)",
                "ResNet18 (CNN - Gemini Fine-Tuned)",
                "🛡️ Experimental Heuristic Ensemble",
                "SVM (RBF Kernel)",
                "Random Forest",
                "Logistic Regression",
                "k-NN",
                "Decision Tree",
                "Naive Bayes",
            ],
            index=0,
            help="Primary ResNet18 CNN trained on GPU achieving 98.2% test accuracy.",
        )

        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50, max_value=99, value=70, step=1,
            help="Predictions below this confidence will show a warning.",
        )

        show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Model info card
        st.markdown("#### 📊 Model Info")
        model_info = {
            "ResNet18 (CNN - Primary 98.2% Accuracy)": {"acc": "98.2% test", "type": "Deep CNN (GPU Trained)", "icon": "⚡"},
            "ResNet18 (CNN - High-Res 512px+)":         {"acc": "98.2% test", "type": "Deep CNN (High-Res)", "icon": "🧠"},
            "ResNet18 (CNN - Gemini Fine-Tuned)":       {"acc": "98.2% test", "type": "Deep CNN (Custom Data)", "icon": "✨"},
            "🛡️ Experimental Heuristic Ensemble":      {"acc": "98.2% test", "type": "Heuristic Fusion", "icon": "🛡️"},
            "SVM (RBF Kernel)":                           {"acc": "74.1%", "type": "Classical ML (HOG+LBP)", "icon": "📐"},
            "Random Forest":                              {"acc": "72.2%", "type": "Ensemble ML",      "icon": "🌲"},
            "Logistic Regression":                        {"acc": "71.5%", "type": "Classical ML",     "icon": "📈"},
            "k-NN":                                       {"acc": "68.2%", "type": "Classical ML",     "icon": "📍"},
            "Naive Bayes":                                {"acc": "65.7%", "type": "Classical ML",     "icon": "🎲"},
            "Decision Tree":                              {"acc": "61.6%", "type": "Classical ML",     "icon": "🌿"},
        }
        info = model_info.get(model_choice, {})
        st.markdown(
            f"""
            <div class="glass-card" style="padding:16px; margin-bottom:0;">
              <div style="font-size:1.8rem; margin-bottom:6px;">{info.get('icon','🤖')}</div>
              <div style="font-weight:700; color:#e2e8f0; margin-bottom:4px;">{model_choice}</div>
              <div style="color:#94a3b8; font-size:0.8rem;">{info.get('type','')}</div>
              <div style="margin-top:10px; padding:6px 12px; background:rgba(99,102,241,0.15);
                          border-radius:6px; display:inline-block;
                          color:#818cf8; font-weight:600; font-size:0.85rem;">
                Est. Accuracy: {info.get('acc','?')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color:#475569; font-size:0.75rem; line-height:1.6;">
              <b style="color:#94a3b8;">Dataset:</b> CIFAKE<br>
              <b style="color:#94a3b8;">Train set:</b> 100,000 images<br>
              <b style="color:#94a3b8;">CNN Backbone:</b> ResNet18<br>
              <b style="color:#94a3b8;">Features (ML):</b> HOG + LBP
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "model":                model_choice,
        "confidence_threshold": confidence_threshold,
        "show_gradcam":         show_gradcam,
    }


# ─── Result Renderer ──────────────────────────────────────────────────────
def render_result(result: dict, heatmap, img: Image.Image, settings: dict) -> None:
    pred       = result["prediction"]
    conf       = result["confidence"]
    is_fake    = pred == "FAKE"
    badge_cls  = "badge-fake" if is_fake else "badge-real"
    conf_cls   = "conf-fake"  if is_fake else "conf-real"
    emoji      = result["emoji"]
    threshold  = settings["confidence_threshold"]

    # Warn if low confidence
    if conf < threshold:
        st.markdown(
            f'<div class="warn-note">⚠️ Low confidence ({conf:.1f}%) — result may be unreliable. '
            f'Try ResNet18 for better accuracy.</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # ── Verdict ──
        st.markdown(
            f'<div class="section-label">Verdict</div>'
            f'<div class="{badge_cls}">'
            f'  <span style="font-size:1.8rem">{emoji}</span>'
            f'  <span>{pred} IMAGE</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Confidence ──
        st.markdown(
            f"""
            <div class="confidence-wrap">
              <div class="conf-label">Confidence Score</div>
              <div class="conf-value {conf_cls}">{conf:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar (fake = red visual via custom progress)
        bar_pct = conf / 100.0
        st.progress(bar_pct)

        # ── Prob breakdown ──
        real_pct = result["probabilities"]["REAL"]
        fake_pct = result["probabilities"]["FAKE"]
        st.markdown(
            f"""
            <div class="metric-row" style="margin-top:16px;">
              <div class="metric-pill">
                <div class="mp-val" style="color:#4ade80;">{real_pct:.1f}%</div>
                <div class="mp-lbl">✅ Real</div>
              </div>
              <div class="metric-pill">
                <div class="mp-val" style="color:#f87171;">{fake_pct:.1f}%</div>
                <div class="mp-lbl">🤖 Fake</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Generator (if fake) ──
        if is_fake:
            gen = result.get("likely_generator", "Unknown")
            st.markdown(
                f"""
                <div class="section-label" style="margin-top:16px;">Likely Generator</div>
                <div class="generator-chip">🎨 {gen}</div>
                """,
                unsafe_allow_html=True,
            )

        # ── Ensemble Consensus Breakdown (if available) ──
        if "ensemble_votes" in result:
            st.markdown('<div class="section-label" style="margin-top:16px;">🛡️ Model Voting Consensus</div>', unsafe_allow_html=True)
            status = result.get("consensus_status", "Consensus Achieved")
            st.markdown(f'<div style="color:#a78bfa; font-weight:700; font-size:0.9rem; margin-bottom:8px;">{status}</div>', unsafe_allow_html=True)
            for m_name, m_vote in result["ensemble_votes"].items():
                v_color = "#f87171" if m_vote == "FAKE" else "#4ade80"
                v_emoji = "🤖" if m_vote == "FAKE" else "✅"
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; padding:4px 10px; background:rgba(255,255,255,0.03); border-radius:6px; margin-bottom:4px; font-size:0.85rem;">'
                    f'  <span style="color:#cbd5e1;">{m_name}</span>'
                    f'  <span style="color:{v_color}; font-weight:700;">{v_emoji} {m_vote}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Multi-Domain Analysis Tabs ──
    st.markdown("<hr>", unsafe_allow_html=True)
    tab_gradcam, tab_fft, tab_ela = st.tabs([
        "🧠 Grad-CAM Heatmap",
        "🔬 2D FFT Spectral Analysis (SDXL Inspector)",
        "📷 Error Level Analysis (ELA)",
    ])

    with tab_gradcam:
        if heatmap is not None:
            st.markdown('<div class="section-label">Grad-CAM Heatmap — Where the Network Looked</div>', unsafe_allow_html=True)
            gc1, gc2, gc3 = st.columns([1, 1, 1])
            with gc1:
                st.markdown("**Original Image**")
                st.image(img, use_container_width=True)
            with gc2:
                st.markdown("**Grad-CAM Overlay**")
                st.image(heatmap, use_container_width=True, clamp=True)
            with gc3:
                st.markdown("**Interpretation**")
                st.markdown(
                    """
                    <div class="info-note">
                      🔴 <b>Red/Hot</b> regions = areas the model found most suspicious.<br><br>
                      🔵 <b>Blue/Cool</b> regions = areas that contributed less to the decision.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Grad-CAM heatmap is enabled when using ResNet18 or Smart Ensemble mode.")

    with tab_fft:
        st.markdown('<div class="section-label">2D Fourier Transform (FFT) Frequency Spectrum — VAE Latent Grid Artifact Inspector</div>', unsafe_allow_html=True)
        try:
            import cv2
            gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            mag = np.log(np.abs(fshift) + 1e-8)
            mag_norm = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            fft_heatmap = cv2.cvtColor(cv2.applyColorMap(mag_norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

            fcol1, fcol2 = st.columns([1, 1], gap="large")
            with fcol1:
                st.markdown("**2D Fourier Magnitude Spectrum (Log Scale)**")
                st.image(fft_heatmap, use_container_width=True)
            with fcol2:
                spectral = result.get("spectral_analysis", {})
                hf_ratio = spectral.get("hf_ratio", 0.0)
                spectral_score = spectral.get("spectral_ai_score", 0.0)

                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:16px; margin-bottom:12px;">
                      <div style="font-weight:700; color:#e2e8f0; font-size:1.1rem; margin-bottom:8px;">🔬 Spectral VAE Noise Analysis</div>
                      <div style="margin-bottom:8px;"><span style="color:#94a3b8;">High-Frequency Energy Ratio:</span> <b style="color:#c4b5fd;">{hf_ratio:.4f}</b></div>
                      <div style="margin-bottom:8px;"><span style="color:#94a3b8;">Spectral AI Score:</span> <b style="color:{'#f87171' if spectral_score >= 50 else '#4ade80'};">{spectral_score:.1f}% AI Probability</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if hf_ratio > 0.72:
                    st.markdown('<div class="warn-note">⚡ <b>High-frequency VAE grid signature detected!</b><br>Modern diffusion decoders (SDXL, Midjourney v6, FLUX) leave distinct lattice energy peaks in the 2D FFT spectrum.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-note">📷 <b>Natural frequency distribution.</b><br>Energy falloff follows standard optical camera lens noise characteristics.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error computing FFT spectrum: {e}")

    with tab_ela:
        st.markdown('<div class="section-label">Error Level Analysis (ELA) — JPEG Compression Digital Residual Inspector</div>', unsafe_allow_html=True)
        try:
            from src.predict import compute_ela_analysis
            ela_metrics, ela_np = compute_ela_analysis(img)
            ecol1, ecol2 = st.columns([1, 1], gap="large")
            with ecol1:
                st.markdown("**Scaled ELA Difference Map**")
                st.image(ela_np, use_container_width=True)
            with ecol2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:16px; margin-bottom:12px;">
                      <div style="font-weight:700; color:#e2e8f0; font-size:1.1rem; margin-bottom:8px;">📷 Compression Artifact Metrics</div>
                      <div style="margin-bottom:8px;"><span style="color:#94a3b8;">ELA Mean Difference:</span> <b style="color:#c4b5fd;">{ela_metrics['ela_mean']}</b></div>
                      <div style="margin-bottom:8px;"><span style="color:#94a3b8;">ELA Error Variance (Std):</span> <b style="color:#c4b5fd;">{ela_metrics['ela_std']}</b></div>
                      <div style="margin-bottom:8px;"><span style="color:#94a3b8;">ELA Compression Score:</span> <b style="color:{'#f87171' if ela_metrics['ela_ai_score'] >= 50 else '#4ade80'};">{ela_metrics['ela_ai_score']:.1f}% AI Probability</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="info-note">
                      💡 <b>How ELA Works:</b><br>
                      Authentic camera photos have non-uniform compression error levels across different detail areas.<br>
                      AI-generated JPEGs often exhibit unnaturally uniform or low error variance (std &lt; 18) due to synthetic generation pipelines.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Error performing ELA analysis: {e}")


# ─── About Tab ────────────────────────────────────────────────────────────
def render_about() -> None:
    st.markdown(
        """
        <div class="glass-card">
          <h3 style="color:#e2e8f0; margin-bottom:16px;">About TruthLens</h3>
          <p>TruthLens is a production-grade AI image detection system built as an academic
          project to detect AI-generated images using both classical Machine Learning and
          Deep Learning techniques.</p>

          <h4 style="color:#e2e8f0; margin-top:20px;">📚 Models</h4>
          <ul>
            <li><b>ResNet18 (CNN)</b> — Transfer learning from ImageNet with a custom head.
                Achieves ~95% accuracy on CIFAKE.</li>
            <li><b>Random Forest</b> — Ensemble of 300 trees on HOG+LBP features.</li>
            <li><b>SVM (RBF)</b> — Support Vector Machine with radial basis kernel.</li>
            <li><b>Logistic Regression</b> — Linear classifier on PCA-reduced HOG+LBP.</li>
            <li><b>k-NN</b> — k-Nearest Neighbours on PCA-reduced feature space.</li>
            <li><b>Decision Tree</b> — Single CART tree with depth=20.</li>
            <li><b>Naive Bayes</b> — Gaussian NB on raw HOG+LBP features.</li>
          </ul>

          <h4 style="color:#e2e8f0; margin-top:20px;">🔬 Dataset</h4>
          <p>Trained on <b>CIFAKE</b> — 60,000 real images (CIFAR-10) and 60,000 AI-generated
          images (Stable Diffusion v1.4). Total: 120,000 images.</p>

          <h4 style="color:#e2e8f0; margin-top:20px;">🛠️ Tech Stack</h4>
          <p>PyTorch · TorchVision · scikit-learn · OpenCV · scikit-image · Streamlit · Plotly</p>

          <h4 style="color:#e2e8f0; margin-top:20px;">🔍 Grad-CAM</h4>
          <p>Gradient-weighted Class Activation Mapping — shows which image regions influenced
          the prediction. Implemented from scratch using PyTorch forward/backward hooks
          on ResNet18's <code>layer4</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Model Load Error ─────────────────────────────────────────────────────
def render_model_error(error_msg: str) -> None:
    st.markdown(
        f'<div class="warn-note">⚠️ <b>Model could not be loaded.</b><br>'
        f'<small>{error_msg}</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-note">Check that the selected model file is readable, then retry. '
        'If it does not exist, train that model first.</div>',
        unsafe_allow_html=True,
    )


# ─── Main App ──────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    settings = render_sidebar()

    # ── Hero Header ──
    st.markdown(
        """
        <div class="hero-header">TruthLens</div>
        <div class="hero-sub">
          AI-Generated Image Detection · Deep Learning &amp; Classical ML
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    tab_detect, tab_about = st.tabs(["🔍 Detection", "ℹ️ About"])

    with tab_detect:
        # ── Upload Zone ──
        st.markdown(
            '<div class="section-label">Upload Image</div>', unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            label="Drop an image here or click to browse",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            help="Supported: JPG, PNG, WebP, BMP",
        )

        if uploaded_file is None:
            # Placeholder UI
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding:60px; opacity:0.6;">
                  <div style="font-size:4rem; margin-bottom:16px;">📷</div>
                  <div style="color:#94a3b8; font-size:1.1rem; font-weight:500;">
                    Upload an image to detect whether it's real or AI-generated
                  </div>
                  <div style="color:#475569; font-size:0.85rem; margin-top:8px;">
                    Supports: JPEG · PNG · WebP · BMP
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # ── Image Loaded ──
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        col_img, col_meta = st.columns([1, 1], gap="large")
        with col_img:
            st.image(img, caption=f"📷 {uploaded_file.name}", use_container_width=True)
        with col_meta:
            w, h = img.size
            size_kb = len(uploaded_file.getvalue()) / 1024
            st.markdown(
                f"""
                <div class="glass-card">
                  <div class="section-label">Image Info</div>
                  <div class="metric-row">
                    <div class="metric-pill">
                      <div class="mp-val">{w}</div><div class="mp-lbl">Width px</div>
                    </div>
                    <div class="metric-pill">
                      <div class="mp-val">{h}</div><div class="mp-lbl">Height px</div>
                    </div>
                    <div class="metric-pill">
                      <div class="mp-val">{size_kb:.0f}</div><div class="mp-lbl">Size KB</div>
                    </div>
                  </div>
                  <div style="color:#64748b; font-size:0.8rem;">
                    Format: {uploaded_file.type.split('/')[1].upper()}<br>
                    Channels: RGB · 8-bit
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            analyse_btn = st.button(
                "🔍 Analyse Image",
                use_container_width=True,
                key="analyse_btn",
            )

        # ── Run prediction on button click ──
        if analyse_btn or st.session_state.get("last_file") == uploaded_file.name:
            st.session_state["last_file"] = uploaded_file.name
            st.markdown("<hr>", unsafe_allow_html=True)

            model_name = settings["model"]
            with st.spinner(f"🧠 Running {model_name}…"):
                predictor, predictor_type = load_predictor(model_name)

            if predictor is None:
                render_model_error(predictor_type)
            else:
                with st.spinner("⚡ Generating prediction…"):
                    result, heatmap = run_prediction(
                        predictor, predictor_type, img,
                        with_gradcam=settings["show_gradcam"],
                    )

                if result is None:
                    st.error(f"❌ Prediction failed: {heatmap}")
                else:
                    st.markdown(
                        '<div class="glass-card">',
                        unsafe_allow_html=True,
                    )
                    render_result(result, heatmap, img, settings)
                    st.markdown("</div>", unsafe_allow_html=True)

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
