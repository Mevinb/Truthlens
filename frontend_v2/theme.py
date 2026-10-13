"""Visual system and small HTML fragments for the TruthLens redesign."""

from __future__ import annotations

import html
from typing import Iterable, Mapping


LIGHT = {
    "canvas": "#F4F6F8",
    "paper": "#FCFDFE",
    "paper_alt": "#E9EDF2",
    "ink": "#17202A",
    "muted": "#66727F",
    "faint": "#65717D",
    "line": "#D5DBE2",
    "line_strong": "#B8C1CB",
    "accent": "#3157C8",
    "accent_hover": "#2446AB",
    "accent_soft": "#E7ECFF",
    "real": "#087A55",
    "real_soft": "#E0F3EA",
    "fake": "#C13C4C",
    "fake_soft": "#F9E6E9",
    "caution": "#946200",
    "caution_soft": "#FFF1CE",
    "button_text": "#F9FAFF",
}

DARK = {
    "canvas": "#12171D",
    "paper": "#192028",
    "paper_alt": "#222B35",
    "ink": "#EDF1F5",
    "muted": "#B2BDC7",
    "faint": "#8995A1",
    "line": "#303A45",
    "line_strong": "#44505D",
    "accent": "#7391FF",
    "accent_hover": "#91A7FF",
    "accent_soft": "#26345C",
    "real": "#54C99A",
    "real_soft": "#17392F",
    "fake": "#F07B88",
    "fake_soft": "#45272E",
    "caution": "#E6B955",
    "caution_soft": "#3C321F",
    "button_text": "#101722",
}


def safe(value: object) -> str:
    """Escape runtime text and remove typographic separators from UI copy."""
    text = str(value).translate({0x2014: ord("-"), 0x2013: ord("-"), 0x00B7: ord("/")})
    return html.escape(text)


def stylesheet(dark: bool = False) -> str:
    c = DARK if dark else LIGHT
    return f"""
<style>
  :root {{
    color-scheme: {"dark" if dark else "light"};
    --canvas:{c["canvas"]}; --paper:{c["paper"]};
    --paper-alt:{c["paper_alt"]}; --ink:{c["ink"]};
    --muted:{c["muted"]}; --faint:{c["faint"]};
    --line:{c["line"]}; --line-strong:{c["line_strong"]};
    --accent:{c["accent"]}; --accent-hover:{c["accent_hover"]};
    --accent-soft:{c["accent_soft"]}; --real:{c["real"]};
    --real-soft:{c["real_soft"]}; --fake:{c["fake"]};
    --fake-soft:{c["fake_soft"]}; --caution:{c["caution"]};
    --caution-soft:{c["caution_soft"]};
    --button-text:{c["button_text"]};
    --radius:14px;
    --shadow:0 18px 48px rgba(35, 55, 78, {"0.22" if dark else "0.09"});
    --font-body:"Avenir Next", Avenir, "Segoe UI", Helvetica, Arial, sans-serif;
    --font-display:"Arial Narrow", "Avenir Next Condensed", "Roboto Condensed", sans-serif;
    --font-mono:"SFMono-Regular", Consolas, "Liberation Mono", monospace;
  }}

  html, body, [class*="css"] {{ font-family:var(--font-body); }}
  html {{ background:var(--canvas); }}
  .stApp {{ background:var(--canvas); color:var(--ink); }}
  .block-container {{ max-width:1280px; padding:1.25rem 2.25rem 4rem; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  [data-testid="stToolbar"], #MainMenu, footer {{ visibility:hidden; }}
  [data-testid="stSidebar"] {{ display:none; }}

  h1, h2, h3 {{ color:var(--ink); font-family:var(--font-display); }}
  p {{ color:var(--muted); }}
  a {{ color:var(--accent); }}
  code {{
    border:1px solid var(--line); border-radius:5px; background:var(--paper-alt);
    color:var(--ink); padding:1px 5px; font-family:var(--font-mono);
  }}

  .tl-header {{
    display:flex; align-items:center; min-height:58px; gap:13px;
  }}
  .tl-mark {{
    display:grid; place-items:center; width:34px; height:34px;
    border-radius:10px; background:var(--ink); color:var(--paper);
    font-family:var(--font-display); font-size:0.72rem; font-weight:800;
    letter-spacing:-0.03em;
  }}
  .tl-wordmark {{
    color:var(--ink); font-family:var(--font-display); font-size:1.28rem;
    font-weight:800; letter-spacing:-0.035em; line-height:1;
  }}
  .tl-wordmark-sub {{ color:var(--faint); font-size:0.72rem; margin-top:3px; }}

  .tl-training {{
    display:inline-flex; align-items:center; min-height:34px; padding:7px 11px;
    border:1px solid var(--line); border-radius:10px; background:var(--paper);
    color:var(--muted); font-size:0.76rem; line-height:1.25;
  }}
  .tl-training strong {{ color:var(--ink); font-weight:700; margin-right:5px; }}

  .stTabs [data-baseweb="tab-list"] {{
    gap:24px; border-bottom:1px solid var(--line); margin-bottom:20px;
  }}
  .stTabs [data-baseweb="tab"] {{
    height:44px; padding:0 1px; border:none; border-radius:0;
    background:transparent; color:var(--muted); font-weight:650;
    font-size:0.88rem;
  }}
  .stTabs [aria-selected="true"] {{
    color:var(--ink); box-shadow:inset 0 -2px 0 var(--accent);
  }}

  .st-key-workspace {{
    padding:clamp(20px, 4vw, 46px); border:1px solid var(--line);
    border-radius:var(--radius); background:var(--paper); box-shadow:var(--shadow);
  }}
  .tl-hero-title {{
    max-width:11ch; margin:0; color:var(--ink); font-family:var(--font-display);
    font-size:clamp(2.5rem, 5.1vw, 4.9rem); font-weight:800;
    letter-spacing:-0.055em; line-height:0.93;
  }}
  .tl-hero-copy {{
    max-width:43ch; margin:18px 0 0; color:var(--muted); font-size:1rem;
    line-height:1.65;
  }}
  .tl-trust-note {{
    max-width:46ch; margin-top:25px; padding:15px 16px;
    border-left:3px solid var(--accent); background:var(--accent-soft);
    color:var(--ink); font-size:0.83rem; line-height:1.55;
  }}
  .tl-workspace-label {{
    margin:1px 0 9px; color:var(--ink); font-size:0.88rem; font-weight:750;
  }}
  .tl-workspace-help {{ margin:0 0 12px; color:var(--muted); font-size:0.78rem; }}
  .st-key-workspace [data-testid="stImage"] img {{
    max-height:242px; object-fit:cover; border-radius:var(--radius);
  }}
  [data-testid="stFileUploaderDropzone"] {{
    min-height:124px; border:1px dashed var(--line-strong);
    border-radius:var(--radius); background:var(--canvas); color:var(--ink);
  }}
  [data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--accent); }}
  [data-testid="stFileUploaderDropzone"] button {{
    border-color:var(--accent); background:var(--accent); color:var(--button-text);
  }}
  [data-testid="stFileUploaderDropzoneInstructions"] span {{ color:var(--ink); }}
  [data-testid="stFileUploaderDropzoneInstructions"] small {{ color:var(--muted); }}

  .tl-section-head {{ margin:38px 0 16px; }}
  .tl-section-head h2 {{
    margin:0; font-size:clamp(1.7rem, 3vw, 2.45rem); font-weight:800;
    letter-spacing:-0.045em; line-height:1;
  }}
  .tl-section-head p {{ max-width:62ch; margin:10px 0 0; font-size:0.88rem; line-height:1.6; }}

  .tl-sample-label {{
    display:flex; justify-content:space-between; gap:12px; margin:-6px 0 10px;
    color:var(--muted); font-size:0.76rem;
  }}
  .tl-sample-label b {{ color:var(--ink); font-weight:700; }}
  .st-key-samples [data-testid="stImage"] img {{
    aspect-ratio:4/3; object-fit:cover; border-radius:var(--radius);
    border:1px solid var(--line);
  }}

  .tl-verdict {{
    position:relative; overflow:hidden; padding:25px 27px; margin-top:28px;
    border:1px solid var(--line); border-radius:var(--radius); background:var(--paper);
  }}
  .tl-verdict::before {{
    content:""; position:absolute; inset:0 auto 0 0; width:5px; background:var(--tone);
  }}
  .tl-verdict.real {{ --tone:var(--real); --tone-soft:var(--real-soft); }}
  .tl-verdict.fake {{ --tone:var(--fake); --tone-soft:var(--fake-soft); }}
  .tl-verdict.uncertain {{ --tone:var(--caution); --tone-soft:var(--caution-soft); }}
  .tl-verdict-state {{ color:var(--muted); font-size:0.78rem; font-weight:650; }}
  .tl-verdict-label {{
    margin-top:8px; color:var(--tone); font-family:var(--font-display);
    font-size:clamp(2rem, 4vw, 3.7rem); font-weight:850;
    letter-spacing:-0.05em; line-height:0.95;
  }}
  .tl-verdict-sub {{ max-width:70ch; margin-top:12px; color:var(--muted); font-size:0.86rem; line-height:1.55; }}
  .tl-verdict-sub b {{ color:var(--ink); }}
  .tl-verdict-confidence {{
    display:inline-block; margin-top:18px; padding:7px 10px;
    border-radius:8px; background:var(--tone-soft); color:var(--tone);
    font-family:var(--font-mono); font-size:0.78rem; font-weight:700;
  }}

  .tl-probabilities {{
    display:grid; grid-template-columns:1fr 1fr; gap:1px; overflow:hidden;
    margin:0 0 18px; border:1px solid var(--line); border-radius:var(--radius);
    background:var(--line);
  }}
  .tl-prob {{ padding:18px; background:var(--paper); }}
  .tl-prob-label {{ color:var(--muted); font-size:0.75rem; }}
  .tl-prob-value {{
    margin-top:3px; color:var(--ink); font-family:var(--font-display);
    font-size:2rem; font-weight:800; letter-spacing:-0.04em;
    font-variant-numeric:tabular-nums;
  }}
  .tl-prob.real .tl-prob-value {{ color:var(--real); }}
  .tl-prob.fake .tl-prob-value {{ color:var(--fake); }}

  .tl-signal-list {{ margin:0 0 18px; border-top:1px solid var(--line); }}
  .tl-signal {{
    display:grid; grid-template-columns:10px 1fr; gap:12px;
    padding:13px 0; border-bottom:1px solid var(--line);
    color:var(--muted); font-size:0.84rem; line-height:1.55;
  }}
  .tl-signal-index {{ color:var(--accent); font-family:var(--font-mono); font-size:0.72rem; }}

  [class*="st-key-image_"] [data-testid="stImage"] img {{
    width:100%; max-height:520px; object-fit:contain; border:1px solid var(--line);
    border-radius:var(--radius); background:var(--paper-alt);
  }}
  .tl-file-meta {{ margin:7px 0 0; color:var(--faint); font-size:0.72rem; font-family:var(--font-mono); }}

  .tl-callout {{
    padding:13px 15px; margin:10px 0; border:1px solid var(--line);
    border-radius:10px; background:var(--paper-alt); color:var(--muted);
    font-size:0.81rem; line-height:1.6;
  }}
  .tl-callout strong {{ color:var(--ink); }}
  .tl-callout.caution {{ border-color:var(--caution); background:var(--caution-soft); color:var(--ink); }}
  .tl-callout.error {{ border-color:var(--fake); background:var(--fake-soft); color:var(--ink); }}

  .tl-kv-grid {{
    display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:1px;
    overflow:hidden; border:1px solid var(--line); border-radius:var(--radius);
    background:var(--line);
  }}
  .tl-kv {{ min-height:74px; padding:14px 15px; background:var(--paper); }}
  .tl-kv-key {{ color:var(--muted); font-size:0.71rem; line-height:1.35; }}
  .tl-kv-value {{
    margin-top:6px; overflow-wrap:anywhere; color:var(--ink); font-size:0.88rem;
    font-weight:700; line-height:1.35; font-variant-numeric:tabular-nums;
  }}
  .tl-kv-grid.mono .tl-kv-value {{ font-family:var(--font-mono); font-size:0.75rem; font-weight:550; }}

  .tl-empty {{
    margin-top:24px; padding:36px 24px; border:1px dashed var(--line-strong);
    border-radius:var(--radius); background:var(--paper); text-align:center;
  }}
  .tl-empty-mark {{
    width:34px; height:34px; margin:0 auto 13px; border:7px solid var(--line-strong);
    border-radius:50%; box-shadow:inset 0 0 0 4px var(--paper);
  }}
  .tl-empty h3 {{ margin:0; font-size:1.2rem; font-weight:800; letter-spacing:-0.025em; }}
  .tl-empty p {{ margin:7px auto 0; max-width:44ch; font-size:0.8rem; line-height:1.55; }}

  .tl-skeleton {{
    display:grid; grid-template-columns:1.15fr 0.85fr; gap:18px; margin-top:26px;
  }}
  .tl-skeleton-block {{
    min-height:225px; border-radius:var(--radius); background:var(--paper-alt);
    transform:translateZ(0); overflow:hidden; position:relative;
  }}
  .tl-skeleton-block::after {{
    content:""; position:absolute; inset:0; transform:translateX(-100%);
    background:linear-gradient(90deg, transparent, rgba(255,255,255,0.34), transparent);
    animation:tl-shimmer 1.35s ease-in-out infinite;
  }}
  @keyframes tl-shimmer {{ to {{ transform:translateX(100%); }} }}

  .tl-benchmark-lead {{
    display:grid; grid-template-columns:minmax(0, 1.4fr) minmax(220px, .6fr);
    gap:24px; padding:28px; border:1px solid var(--line); border-radius:var(--radius);
    background:var(--paper);
  }}
  .tl-benchmark-lead h2 {{
    margin:0; max-width:13ch; font-size:clamp(2.25rem, 5vw, 4.2rem);
    font-weight:850; letter-spacing:-0.055em; line-height:0.95;
  }}
  .tl-benchmark-lead p {{ max-width:58ch; margin:15px 0 0; font-size:0.88rem; line-height:1.65; }}
  .tl-headline-metric {{ align-self:end; padding-left:20px; border-left:1px solid var(--line); }}
  .tl-headline-metric b {{
    display:block; color:var(--accent); font-family:var(--font-display);
    font-size:3.1rem; letter-spacing:-0.05em; line-height:1;
  }}
  .tl-headline-metric span {{ display:block; margin-top:7px; color:var(--muted); font-size:0.76rem; line-height:1.45; }}

  .tl-model-row {{
    display:grid; grid-template-columns:minmax(190px,1.5fr) repeat(4,minmax(70px,.55fr));
    gap:14px; align-items:center; padding:15px 4px; border-bottom:1px solid var(--line);
    color:var(--muted); font-size:0.78rem;
  }}
  .tl-model-row.header {{ color:var(--faint); font-size:0.69rem; font-weight:700; }}
  .tl-model-name {{ color:var(--ink); font-weight:700; }}
  .tl-model-value {{ font-family:var(--font-mono); font-variant-numeric:tabular-nums; }}

  .tl-method-grid {{
    display:grid; grid-template-columns:1.2fr .8fr; gap:22px; margin-top:4px;
  }}
  .tl-method-copy {{ padding:28px; border:1px solid var(--line); border-radius:var(--radius); background:var(--paper); }}
  .tl-method-copy h2 {{ margin:0 0 14px; font-size:2rem; font-weight:850; letter-spacing:-0.04em; }}
  .tl-method-copy h3 {{ margin:28px 0 9px; font-size:1.05rem; font-family:var(--font-body); font-weight:750; }}
  .tl-method-copy p, .tl-method-copy li {{ color:var(--muted); font-size:0.86rem; line-height:1.7; }}
  .tl-method-copy ul {{ margin:0; padding-left:1.1rem; }}

  [data-testid="stPopover"] > button, .stButton > button,
  [data-testid="stDownloadButton"] > button {{
    min-height:40px; border:1px solid var(--line-strong); border-radius:10px;
    background:var(--paper); color:var(--ink); font-weight:700; font-size:0.82rem;
    transition:transform 120ms ease, border-color 120ms ease, background 120ms ease;
  }}
  [data-testid="stPopover"] > button:hover, .stButton > button:hover,
  [data-testid="stDownloadButton"] > button:hover {{ border-color:var(--accent); color:var(--accent); }}
  [data-testid="stPopover"] > button:active, .stButton > button:active,
  [data-testid="stDownloadButton"] > button:active {{ transform:translateY(1px); }}
  .stButton > button[kind="primary"] {{ background:var(--accent); border-color:var(--accent); color:var(--button-text); }}
  .stButton > button[kind="primary"]:hover {{ background:var(--accent-hover); border-color:var(--accent-hover); color:var(--button-text); }}
  button:focus-visible, input:focus-visible, [tabindex]:focus-visible {{ outline:3px solid var(--accent) !important; outline-offset:2px; }}

  [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    border-color:var(--line-strong); border-radius:10px; background:var(--paper); color:var(--ink);
  }}
  input::placeholder {{ color:var(--muted); opacity:1; }}
  [data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; }}
  [data-testid="stProgress"] > div > div {{ background:var(--accent); }}
  [data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:var(--radius); background:var(--paper); }}
  hr {{ border:none; border-top:1px solid var(--line); margin:30px 0; }}

  @media (max-width: 767px) {{
    .block-container {{ padding:0.8rem 1rem 3rem; }}
    .tl-training {{ display:none; }}
    .st-key-workspace {{ padding:20px; }}
    .tl-hero-title {{ max-width:12ch; font-size:2.9rem; }}
    .tl-benchmark-lead, .tl-method-grid, .tl-skeleton {{ grid-template-columns:1fr; }}
    .tl-headline-metric {{ padding:18px 0 0; border-left:0; border-top:1px solid var(--line); }}
    .tl-model-row {{ grid-template-columns:1.3fr repeat(2,.7fr); }}
    .tl-model-row > :nth-child(4), .tl-model-row > :nth-child(5) {{ display:none; }}
    .tl-kv-grid {{ grid-template-columns:1fr; }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration:0.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; }}
  }}
</style>
"""


def header(training: bool = True) -> str:
    status = (
        '<div class="tl-training"><strong>Training in progress.</strong>'
        " Results use the latest saved checkpoint.</div>"
        if training
        else ""
    )
    return (
        '<div class="tl-header">'
        '<div class="tl-mark">TL</div>'
        '<div><div class="tl-wordmark">TruthLens</div>'
        '<div class="tl-wordmark-sub">Image authenticity review</div></div>'
        '<div style="flex:1"></div>'
        f"{status}</div>"
    )


def hero_intro() -> str:
    return (
        '<h1 class="tl-hero-title">Check what you see.</h1>'
        '<p class="tl-hero-copy">Upload an image for a model verdict, confidence, '
        "provenance checks, and visual evidence.</p>"
        '<div class="tl-trust-note"><strong>Screening signal, not proof.</strong> '
        "Training is still in progress, and unfamiliar generators or edits can fool the detector.</div>"
    )


def section_head(title: str, body: str = "") -> str:
    description = f"<p>{safe(body)}</p>" if body else ""
    return f'<div class="tl-section-head"><h2>{safe(title)}</h2>{description}</div>'


def callout(body: str, kind: str = "info", allow_html: bool = False) -> str:
    content = body if allow_html else safe(body)
    return f'<div class="tl-callout {kind}">{content}</div>'


def empty_state(title: str, body: str) -> str:
    return (
        '<div class="tl-empty"><div class="tl-empty-mark"></div>'
        f"<h3>{safe(title)}</h3><p>{safe(body)}</p></div>"
    )


def verdict(label: str, confidence: float, subtitle: str, tone: str) -> str:
    return (
        f'<div class="tl-verdict {tone}">'
        '<div class="tl-verdict-state">Model verdict</div>'
        f'<div class="tl-verdict-label">{safe(label)}</div>'
        f'<div class="tl-verdict-sub">{subtitle}</div>'
        f'<div class="tl-verdict-confidence">{confidence:.1f}% confidence</div>'
        "</div>"
    )


def probabilities(real: float, fake: float) -> str:
    return (
        '<div class="tl-probabilities">'
        '<div class="tl-prob real"><div class="tl-prob-label">Real photograph</div>'
        f'<div class="tl-prob-value">{real:.1f}%</div></div>'
        '<div class="tl-prob fake"><div class="tl-prob-label">AI generated</div>'
        f'<div class="tl-prob-value">{fake:.1f}%</div></div></div>'
    )


def signals(items: Iterable[str]) -> str:
    rows = "".join(
        '<div class="tl-signal">'
        f'<div class="tl-signal-index">{index:02d}</div><div>{safe(item)}</div></div>'
        for index, item in enumerate(items, start=1)
    )
    return f'<div class="tl-signal-list">{rows}</div>'


def kv_grid(values: Mapping[str, object] | Iterable[tuple[str, object]], mono: bool = False) -> str:
    pairs = values.items() if isinstance(values, Mapping) else values
    cells = "".join(
        '<div class="tl-kv">'
        f'<div class="tl-kv-key">{safe(key)}</div>'
        f'<div class="tl-kv-value">{safe(value)}</div></div>'
        for key, value in pairs
    )
    cls = "tl-kv-grid mono" if mono else "tl-kv-grid"
    return f'<div class="{cls}">{cells}</div>'


def skeleton() -> str:
    return (
        '<div class="tl-skeleton" aria-label="Analyzing image">'
        '<div class="tl-skeleton-block"></div>'
        '<div class="tl-skeleton-block"></div></div>'
    )


def model_rows(rows: Iterable[Mapping[str, object]]) -> str:
    head = (
        '<div class="tl-model-row header"><div>Model</div><div>Accuracy</div>'
        "<div>AUC</div><div>Real recall</div><div>Fake recall</div></div>"
    )
    body = "".join(
        '<div class="tl-model-row">'
        f'<div class="tl-model-name">{safe(row.get("Model", ""))}</div>'
        f'<div class="tl-model-value">{float(row.get("Accuracy") or 0):.1f}%</div>'
        f'<div class="tl-model-value">{float(row.get("AUC") or 0):.3f}</div>'
        f'<div class="tl-model-value">{float(row.get("Real acc") or 0):.1f}%</div>'
        f'<div class="tl-model-value">{float(row.get("Fake acc") or 0):.1f}%</div>'
        "</div>"
        for row in rows
    )
    return f"{head}{body}"
