#!/usr/bin/env python3
"""
TruthLens — app/theme.py
========================
Design system for the Streamlit frontend: tokens, stylesheet, and the small
HTML fragments that make up the interface.

Notes on choices made here, because the previous stylesheet made the opposite
ones and they caused real problems:

  * No ``@import url(fonts.googleapis.com)``. A webfont fetched at render time
    is an external network dependency that fails silently offline and shifts
    the layout when it does. The stack below is resolved locally on every
    platform.
  * No infinite animations. The verdict badge used to pulse forever, which is
    a distraction at best and a problem for motion-sensitive users. Motion is
    limited to a single entrance transition, and that is disabled under
    ``prefers-reduced-motion``.
  * Colours come from tokens, and green/red are reserved exclusively for the
    REAL/FAKE verdict. Nothing decorative borrows the verdict palette, so a
    coloured element always means something.
  * Streamlit's own widgets are themed in ``.streamlit/config.toml`` rather
    than fought with ``!important`` here.
"""

from __future__ import annotations

from typing import Iterable

# ─── Tokens ──────────────────────────────────────────────────────────────────
INK_900 = "#0a0c14"   # page
INK_800 = "#0f1219"   # raised surface
INK_700 = "#161a26"   # card
INK_600 = "#1e2331"   # hover / inset
LINE    = "#252b3b"   # hairline border
LINE_HI = "#333a4d"   # emphasised border

TEXT    = "#e7e9f0"   # primary copy
TEXT_2  = "#a2abbf"   # secondary copy
TEXT_3  = "#6b7488"   # captions, disabled

ACCENT     = "#7c5cff"   # brand violet — interactive, never semantic
ACCENT_DIM = "#5b3fd6"
ACCENT_BG  = "rgba(124, 92, 255, 0.12)"

REAL     = "#34d27b"     # verdict green — reserved
REAL_BG  = "rgba(52, 210, 123, 0.10)"
FAKE     = "#ff5f6d"     # verdict red — reserved
FAKE_BG  = "rgba(255, 95, 109, 0.10)"
CAUTION  = "#f5a524"     # low confidence / near-chance warnings
CAUTION_BG = "rgba(245, 165, 36, 0.10)"

FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    'Ubuntu, "Liberation Sans", Arial, sans-serif'
)
MONO = (
    '"SF Mono", "JetBrains Mono", "Fira Code", "Cascadia Mono", Consolas, '
    '"DejaVu Sans Mono", monospace'
)

R_SM, R_MD, R_LG = "6px", "10px", "14px"


def stylesheet() -> str:
    """The full <style> block. Injected once per rerun by app.main()."""
    return f"""
<style>
  :root {{
    --ink-900:{INK_900}; --ink-800:{INK_800}; --ink-700:{INK_700};
    --ink-600:{INK_600}; --line:{LINE}; --line-hi:{LINE_HI};
    --text:{TEXT}; --text-2:{TEXT_2}; --text-3:{TEXT_3};
    --accent:{ACCENT}; --accent-dim:{ACCENT_DIM}; --accent-bg:{ACCENT_BG};
    --real:{REAL}; --real-bg:{REAL_BG}; --fake:{FAKE}; --fake-bg:{FAKE_BG};
    --caution:{CAUTION}; --caution-bg:{CAUTION_BG};
  }}

  html, body, [class*="css"] {{ font-family:{FONT}; }}

  .stApp {{ background:{INK_900}; }}
  .block-container {{ padding-top:1.6rem; padding-bottom:4rem; max-width:1180px; }}

  /* ── Masthead ───────────────────────────────────────────────────────── */
  .tl-masthead {{
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
    padding-bottom:14px; margin-bottom:22px;
    border-bottom:1px solid {LINE};
  }}
  .tl-wordmark {{
    font-size:1.55rem; font-weight:700; letter-spacing:-0.02em; color:{TEXT};
  }}
  .tl-wordmark span {{ color:{ACCENT}; }}
  .tl-tagline {{ font-size:0.9rem; color:{TEXT_3}; }}
  .tl-masthead-spacer {{ flex:1 1 auto; }}

  /* ── Surfaces ───────────────────────────────────────────────────────── */
  .tl-card {{
    background:{INK_700}; border:1px solid {LINE}; border-radius:{R_LG};
    padding:20px 22px; margin-bottom:16px;
  }}
  .tl-card.tl-flush {{ padding:16px 18px; }}
  .tl-card h4 {{
    margin:0 0 10px; font-size:0.95rem; font-weight:650; color:{TEXT};
  }}
  .tl-card p {{ color:{TEXT_2}; font-size:0.88rem; line-height:1.65; margin:0 0 10px; }}
  .tl-card p:last-child {{ margin-bottom:0; }}
  .tl-card ul {{ color:{TEXT_2}; font-size:0.88rem; line-height:1.7; margin:0 0 10px 1.1rem; padding:0; }}
  .tl-card code {{
    font-family:{MONO}; font-size:0.82em; background:{INK_600};
    padding:1px 5px; border-radius:4px; color:{TEXT};
  }}

  .tl-eyebrow {{
    font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
    color:{TEXT_3}; margin:0 0 10px;
  }}

  /* ── Verdict ────────────────────────────────────────────────────────── */
  .tl-verdict {{
    border-radius:{R_LG}; padding:22px 24px; border:1px solid; margin-bottom:16px;
    animation:tl-rise 240ms ease-out;
  }}
  .tl-verdict.real {{ border-color:{REAL}; background:{REAL_BG}; }}
  .tl-verdict.fake {{ border-color:{FAKE}; background:{FAKE_BG}; }}
  .tl-verdict.uncertain {{ border-color:{CAUTION}; background:{CAUTION_BG}; }}
  .tl-verdict-label {{
    font-size:1.7rem; font-weight:750; letter-spacing:-0.02em; line-height:1.1;
    display:flex; align-items:center; gap:10px;
  }}
  .tl-verdict.real .tl-verdict-label {{ color:{REAL}; }}
  .tl-verdict.fake .tl-verdict-label {{ color:{FAKE}; }}
  .tl-verdict.uncertain .tl-verdict-label {{ color:{CAUTION}; }}
  .tl-verdict-sub {{ margin-top:8px; font-size:0.85rem; color:{TEXT_2}; }}
  .tl-verdict-sub b {{ color:{TEXT}; font-variant-numeric:tabular-nums; }}

  @keyframes tl-rise {{
    from {{ opacity:0; transform:translateY(4px); }}
    to   {{ opacity:1; transform:none; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    .tl-verdict {{ animation:none; }}
  }}

  /* ── Probability split bar ──────────────────────────────────────────── */
  .tl-split {{
    display:flex; height:10px; border-radius:999px; overflow:hidden;
    background:{INK_600}; margin:6px 0 8px;
  }}
  .tl-split-real {{ background:{REAL}; }}
  .tl-split-fake {{ background:{FAKE}; }}
  .tl-split-legend {{
    display:flex; justify-content:space-between;
    font-size:0.78rem; color:{TEXT_3}; font-variant-numeric:tabular-nums;
  }}
  .tl-split-legend b {{ font-weight:650; }}

  /* ── Stat grid ──────────────────────────────────────────────────────── */
  .tl-stats {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .tl-stat {{
    flex:1 1 82px; background:{INK_800}; border:1px solid {LINE};
    border-radius:{R_MD}; padding:10px 12px;
  }}
  .tl-stat-val {{
    font-size:1.15rem; font-weight:700; color:{TEXT};
    font-variant-numeric:tabular-nums; line-height:1.2;
  }}
  .tl-stat-lbl {{
    font-size:0.68rem; letter-spacing:0.06em; text-transform:uppercase;
    color:{TEXT_3}; margin-top:3px;
  }}

  /* ── Key/value rows ─────────────────────────────────────────────────── */
  .tl-kv {{
    display:flex; justify-content:space-between; gap:12px;
    padding:7px 0; border-bottom:1px solid {LINE}; font-size:0.84rem;
  }}
  .tl-kv:last-child {{ border-bottom:none; }}
  .tl-kv-k {{ color:{TEXT_3}; }}
  .tl-kv-v {{ color:{TEXT}; font-weight:600; font-variant-numeric:tabular-nums; }}
  .tl-kv-v.mono {{ font-family:{MONO}; font-size:0.8rem; font-weight:500; }}

  /* ── Reasoning list ─────────────────────────────────────────────────── */
  .tl-reason {{
    display:flex; gap:10px; align-items:flex-start;
    padding:9px 12px; margin-bottom:6px;
    background:{INK_800}; border:1px solid {LINE};
    border-left:2px solid {ACCENT}; border-radius:0 {R_SM} {R_SM} 0;
    font-size:0.85rem; color:{TEXT_2}; line-height:1.55;
  }}

  /* ── Chips ──────────────────────────────────────────────────────────── */
  .tl-chip {{
    display:inline-flex; align-items:center; gap:6px;
    padding:4px 11px; border-radius:999px; font-size:0.76rem; font-weight:600;
    background:{ACCENT_BG}; border:1px solid {ACCENT}; color:{ACCENT};
  }}
  .tl-chip.neutral {{
    background:{INK_600}; border-color:{LINE_HI}; color:{TEXT_2};
    font-family:{MONO}; font-weight:500;
  }}

  /* ── Notes ──────────────────────────────────────────────────────────── */
  .tl-note {{
    border-radius:{R_MD}; padding:11px 14px; margin:10px 0;
    font-size:0.83rem; line-height:1.6; border:1px solid;
  }}
  .tl-note.info    {{ background:{INK_800};   border-color:{LINE};    color:{TEXT_2}; }}
  .tl-note.caution {{ background:{CAUTION_BG};border-color:{CAUTION}; color:#f3d9a6; }}
  .tl-note.error   {{ background:{FAKE_BG};   border-color:{FAKE};    color:#ffc3c8; }}
  .tl-note b {{ color:{TEXT}; }}
  .tl-note code {{ font-family:{MONO}; font-size:0.8em; }}

  /* ── Empty state ────────────────────────────────────────────────────── */
  .tl-empty {{
    border:1px dashed {LINE_HI}; border-radius:{R_LG};
    padding:44px 24px; text-align:center; background:{INK_800};
  }}
  .tl-empty-title {{ color:{TEXT_2}; font-size:0.95rem; font-weight:600; }}
  .tl-empty-sub   {{ color:{TEXT_3}; font-size:0.8rem; margin-top:6px; }}

  /* ── Streamlit widget nudges (theme itself lives in .streamlit/config.toml) */
  [data-testid="stFileUploaderDropzone"] {{
    background:{INK_800}; border:1px dashed {LINE_HI}; border-radius:{R_LG};
  }}
  [data-testid="stFileUploaderDropzone"]:hover {{ border-color:{ACCENT}; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{
    background:transparent; border:none; border-radius:0;
    padding:9px 15px; font-size:0.87rem; font-weight:550; color:{TEXT_3};
  }}
  .stTabs [aria-selected="true"] {{
    color:{TEXT}; box-shadow:inset 0 -2px 0 {ACCENT};
  }}

  .stButton > button {{
    border-radius:{R_MD}; font-weight:600; font-size:0.88rem;
    border:1px solid {LINE_HI}; transition:border-color 140ms, background 140ms;
  }}
  .stButton > button[kind="primary"] {{
    background:{ACCENT}; border-color:{ACCENT}; color:#fff;
  }}
  .stButton > button[kind="primary"]:hover {{
    background:{ACCENT_DIM}; border-color:{ACCENT_DIM};
  }}

  [data-testid="stSidebar"] {{ background:{INK_800}; border-right:1px solid {LINE}; }}
  [data-testid="stSidebar"] .block-container {{ padding-top:1.2rem; }}
  [data-testid="stSidebar"] hr {{ margin:14px 0; border-color:{LINE}; }}

  hr {{ border:none; border-top:1px solid {LINE}; margin:22px 0; }}
  #MainMenu, footer {{ visibility:hidden; }}

  [data-testid="stImage"] img {{ border-radius:{R_MD}; }}
  .stDataFrame {{ border:1px solid {LINE}; border-radius:{R_MD}; }}
</style>
"""


# ─── HTML fragment builders ──────────────────────────────────────────────────
def masthead(right_html: str = "") -> str:
    return (
        '<div class="tl-masthead">'
        '<div class="tl-wordmark">Truth<span>Lens</span></div>'
        '<div class="tl-tagline">AI-generated image detection</div>'
        '<div class="tl-masthead-spacer"></div>'
        f"{right_html}"
        "</div>"
    )


def eyebrow(text: str) -> str:
    return f'<div class="tl-eyebrow">{text}</div>'


def note(body: str, kind: str = "info") -> str:
    """kind: info | caution | error."""
    return f'<div class="tl-note {kind}">{body}</div>'


def chip(text: str, neutral: bool = False) -> str:
    cls = "tl-chip neutral" if neutral else "tl-chip"
    return f'<span class="{cls}">{text}</span>'


def verdict(label: str, is_fake: bool, subtitle: str,
            uncertain: bool = False) -> str:
    tone = "uncertain" if uncertain else ("fake" if is_fake else "real")
    icon = ("&#9888;" if uncertain else
            "&#129302;" if is_fake else "&#10003;")
    return (
        f'<div class="tl-verdict {tone}">'
        f'  <div class="tl-verdict-label"><span>{icon}</span><span>{label}</span></div>'
        f'  <div class="tl-verdict-sub">{subtitle}</div>'
        f"</div>"
    )


def split_bar(real_pct: float, fake_pct: float) -> str:
    return (
        '<div class="tl-split">'
        f'  <div class="tl-split-real" style="width:{max(real_pct, 0):.2f}%"></div>'
        f'  <div class="tl-split-fake" style="width:{max(fake_pct, 0):.2f}%"></div>'
        "</div>"
        '<div class="tl-split-legend">'
        f'  <span>real <b style="color:{REAL}">{real_pct:.1f}%</b></span>'
        f'  <span><b style="color:{FAKE}">{fake_pct:.1f}%</b> ai-generated</span>'
        "</div>"
    )


def stats(pairs: Iterable[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="tl-stat"><div class="tl-stat-val">{v}</div>'
        f'<div class="tl-stat-lbl">{k}</div></div>'
        for k, v in pairs
    )
    return f'<div class="tl-stats">{cells}</div>'


def kv_rows(pairs: Iterable[tuple[str, str]], mono: bool = False) -> str:
    cls = "tl-kv-v mono" if mono else "tl-kv-v"
    return "".join(
        f'<div class="tl-kv"><span class="tl-kv-k">{k}</span>'
        f'<span class="{cls}">{v}</span></div>'
        for k, v in pairs
    )


def reasons(items: Iterable[str]) -> str:
    return "".join(f'<div class="tl-reason">{it}</div>' for it in items)


def empty_state(title: str, sub: str, icon: str = "&#128247;") -> str:
    return (
        '<div class="tl-empty">'
        f'  <div style="font-size:2rem; opacity:0.5; margin-bottom:10px;">{icon}</div>'
        f'  <div class="tl-empty-title">{title}</div>'
        f'  <div class="tl-empty-sub">{sub}</div>'
        "</div>"
    )
