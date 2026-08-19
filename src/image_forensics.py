#!/usr/bin/env python3
"""Container, metadata, watermark, and steganography-oriented image checks.

These checks are intentionally kept separate from the trained REAL/FAKE model.
Some findings (for example an explicit ``Software=Stable Diffusion`` tag or a
C2PA manifest) can be useful provenance evidence. Others, especially visible
watermark and least-significant-bit statistics, are heuristics and are not
reliable enough to change the model verdict.

The analyser accepts original encoded bytes whenever possible. Decoding an
image to RGB before this stage destroys container chunks, EXIF/XMP records,
content credentials, alpha-channel payloads, and trailing data.
"""

from __future__ import annotations

import io
import hashlib
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image


AI_TERMS = {
    "ai generated", "aigenerated", "artificial intelligence", "automatic1111",
    "comfyui", "dall-e", "dalle", "diffusion", "firefly", "flux", "fooocus",
    "generative fill", "gpt image", "gpt-image", "invokeai", "leonardo",
    "midjourney", "negative prompt", "novelai", "openai", "prompt",
    "sampler", "seed", "stable diffusion", "steps", "text-to-image",
}
EDITOR_TERMS = {
    "adobe", "affinity", "canva", "capture one", "darktable", "gimp",
    "google photos", "imagemagick", "lightroom", "photoshop", "snapseed",
}
WATERMARK_TERMS = {
    "c2pa", "contentauth", "content credentials", "digimarc", "imwatermark",
    "invisible-watermark", "synthid", "trustmark", "watermark",
}
CAMERA_KEYS = {
    "make", "model", "lensmake", "lensmodel", "datetimeoriginal",
    "exposuretime", "fnumber", "isospeedratings", "photographicsensitivity",
    "focallength",
}
METADATA_KEYS_OF_INTEREST = {
    "software", "artist", "copyright", "comment", "description", "parameters",
    "prompt", "workflow", "source", "generator", "creation time", "xml:com.adobe.xmp",
}


def _source_bytes(source: Any) -> Optional[bytes]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        try:
            return Path(source).read_bytes()
        except OSError:
            return None
    if hasattr(source, "getvalue"):
        try:
            return bytes(source.getvalue())
        except Exception:
            return None
    return None


def _open_image(source: Any, raw: Optional[bytes]) -> Image.Image:
    if raw is not None:
        return Image.open(io.BytesIO(raw))
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        return Image.fromarray(source)
    if isinstance(source, (str, Path)):
        return Image.open(source)
    raise TypeError(f"Unsupported image source: {type(source)!r}")


def _safe_text(value: Any, limit: int = 500) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif isinstance(value, (tuple, list)):
        value = ", ".join(_safe_text(v, 100) for v in value)
    text = str(value).replace("\x00", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _contains_any(text: str, terms: Iterable[str]) -> List[str]:
    low = text.casefold()
    return sorted({term for term in terms if term in low})


def _metadata(img: Image.Image, raw: Optional[bytes]) -> Dict[str, Any]:
    fields: Dict[str, str] = {}
    exif_present = False
    gps_present = False

    try:
        exif = img.getexif()
        exif_present = bool(exif)
        for tag_id, value in exif.items():
            key = ExifTags.TAGS.get(tag_id, str(tag_id))
            if key == "GPSInfo":
                gps_present = True
                fields[key] = "[present; values hidden]"
            else:
                fields[key] = _safe_text(value)
    except Exception:
        pass

    for key, value in (getattr(img, "info", {}) or {}).items():
        if key in {"exif", "icc_profile"}:
            continue
        if isinstance(value, (str, bytes, int, float, tuple, list)):
            fields.setdefault(str(key), _safe_text(value))

    searchable = "\n".join(f"{k}: {v}" for k, v in fields.items())
    ai_matches = _contains_any(searchable, AI_TERMS)
    editor_matches = _contains_any(searchable, EDITOR_TERMS)
    watermark_matches = _contains_any(searchable, WATERMARK_TERMS)
    camera_fields = sorted(
        key for key in fields if key.casefold().replace(" ", "") in CAMERA_KEYS
    )

    # XMP sometimes survives only as an opaque byte string in a JPEG APP1 or
    # PNG iTXt chunk and is not exposed through Pillow's ``info`` dictionary.
    raw_low = raw.lower() if raw else b""
    xmp_present = bool(
        "xmp" in {k.casefold() for k in fields}
        or b"http://ns.adobe.com/xap/1.0/" in raw_low
        or b"<x:xmpmeta" in raw_low
    )

    display_fields = {}
    for key, value in fields.items():
        low = key.casefold()
        if (
            low.replace(" ", "") in CAMERA_KEYS
            or low in METADATA_KEYS_OF_INTEREST
            or _contains_any(f"{key}: {value}", AI_TERMS | EDITOR_TERMS | WATERMARK_TERMS)
        ):
            display_fields[key] = value
    # If there are only a few fields, showing all of them is more useful than
    # an empty filtered list.
    if len(fields) <= 16:
        display_fields = fields

    return {
        "present": bool(fields),
        "field_count": len(fields),
        "fields": dict(sorted(display_fields.items())[:40]),
        "exif_present": exif_present,
        "xmp_present": xmp_present,
        "gps_present": gps_present,
        "camera_fields": camera_fields,
        "ai_terms": ai_matches,
        "editor_terms": editor_matches,
        "watermark_terms": watermark_matches,
    }


def _png_chunks(raw: bytes) -> Tuple[List[str], int]:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return [], 0
    names: List[str] = []
    pos = 8
    end = 0
    while pos + 12 <= len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        kind = raw[pos + 4:pos + 8].decode("latin-1", errors="replace")
        next_pos = pos + 12 + length
        if next_pos > len(raw):
            break
        names.append(kind)
        pos = next_pos
        if kind == "IEND":
            end = pos
            break
    return names, max(0, len(raw) - end) if end else 0


def _jpeg_markers_and_trailing(raw: bytes) -> Tuple[List[str], int]:
    if not raw.startswith(b"\xff\xd8"):
        return [], 0
    markers = set()
    pos = 2
    # Parse marker segments only up to start-of-scan. Entropy-coded JPEG bytes
    # can contain arbitrary 0xE0..0xEF values, so scanning every byte would
    # invent APP markers that are not actually present.
    while pos + 1 < len(raw):
        if raw[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(raw) and raw[pos] == 0xFF:
            pos += 1
        if pos >= len(raw):
            break
        marker = raw[pos]
        pos += 1
        if marker == 0xDA:  # SOS: compressed pixel stream begins
            break
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(raw):
            break
        length = struct.unpack(">H", raw[pos:pos + 2])[0]
        if length < 2 or pos + length > len(raw):
            break
        if 0xE0 <= marker <= 0xEF:
            markers.add(f"APP{marker - 0xE0}")
        pos += length
    eoi = raw.rfind(b"\xff\xd9")
    trailing = max(0, len(raw) - (eoi + 2)) if eoi >= 0 else 0
    return sorted(markers), trailing


def _container(img: Image.Image, raw: Optional[bytes]) -> Dict[str, Any]:
    chunks: List[str] = []
    markers: List[str] = []
    trailing = 0
    if raw:
        chunks, trailing = _png_chunks(raw)
        if not chunks:
            markers, trailing = _jpeg_markers_and_trailing(raw)
    info = getattr(img, "info", {}) or {}
    return {
        "format": img.format or "unknown",
        "mode": img.mode,
        "size": list(img.size),
        "png_chunks": chunks,
        "jpeg_app_markers": markers,
        "has_alpha": "A" in img.getbands(),
        "has_icc_profile": bool(info.get("icc_profile")),
        "animated": bool(getattr(img, "is_animated", False)),
        "frame_count": int(getattr(img, "n_frames", 1)),
        "trailing_bytes": trailing,
    }


def _content_credentials(raw: Optional[bytes], metadata: Dict[str, Any]) -> Dict[str, Any]:
    low = raw.lower() if raw else b""
    c2pa = any(token in low for token in (b"c2pa", b"contentauth", b"content credentials"))
    jumbf = any(token in low for token in (b"jumb", b"jumd", b"c2ma"))
    manifest_ref = any(
        token in low
        for token in (b"c2pa_manifest", b"dcterms:provenance", b"manifeststore")
    )
    return {
        "detected": bool(c2pa or jumbf or manifest_ref),
        "c2pa_marker": c2pa,
        "jumbf_marker": jumbf,
        "manifest_reference": manifest_ref,
        "note": (
            "A marker was found, but cryptographic trust was not validated."
            if c2pa or jumbf or manifest_ref
            else "No C2PA/JUMBF marker found. Absence does not imply the image is authentic."
        ),
    }


def _region_name(x: int, y: int, w: int, h: int, iw: int, ih: int) -> str:
    cx, cy = x + w / 2, y + h / 2
    vertical = "top" if cy < ih / 3 else "bottom" if cy > ih * 2 / 3 else "middle"
    horizontal = "left" if cx < iw / 3 else "right" if cx > iw * 2 / 3 else "center"
    return f"{vertical}-{horizontal}"


def _visible_watermark(img: Image.Image) -> Dict[str, Any]:
    """Find possible small text/logo overlays near borders.

    This is not OCR. It detects clusters of high-contrast, text-shaped connected
    components in the outer band of an image and reports where they occur.
    """
    rgb = np.asarray(img.convert("RGB"))
    ih, iw = rgb.shape[:2]
    scale = min(1.0, 1200.0 / max(iw, ih))
    if scale < 1:
        work = cv2.resize(rgb, (round(iw * scale), round(ih * scale)))
    else:
        work = rgb
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        in_outer_band = x < w * .22 or x + cw > w * .78 or y < h * .22 or y + ch > h * .78
        text_shape = 3 <= ch <= max(12, h * .12) and 2 <= cw and .12 <= cw / max(ch, 1) <= 18
        if in_outer_band and text_shape and 8 <= area <= w * h * .035:
            components.append((x, y, cw, ch))

    # Aggregate components into coarse regions. A lone edge is common; a cluster
    # of several glyph/logo-like components is the useful signal.
    region_counts: Dict[str, int] = {}
    for x, y, cw, ch in components:
        name = _region_name(x, y, cw, ch, w, h)
        region_counts[name] = region_counts.get(name, 0) + 1
    likely_regions = sorted(k for k, count in region_counts.items() if count >= 3)
    cluster_strength = max(region_counts.values(), default=0)
    score = min(100.0, cluster_strength * 12.0 + len(likely_regions) * 8.0)

    alpha_overlay = False
    if "A" in img.getbands():
        alpha = np.asarray(img.getchannel("A"))
        alpha_overlay = bool(np.any((alpha > 0) & (alpha < 255)))
        if alpha_overlay:
            score = min(100.0, score + 25.0)

    return {
        "suspected": score >= 55,
        "score": round(score, 1),
        "regions": likely_regions,
        "text_like_components": len(components),
        "partial_alpha_present": alpha_overlay,
        "method": "border text/logo component heuristic; no OCR",
    }


def _binary_entropy(bits: np.ndarray) -> float:
    p = float(bits.mean())
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _invisible_watermark(img: Image.Image, raw: Optional[bytes]) -> Dict[str, Any]:
    """Report weak signs associated with embedded/invisible payloads.

    Provider watermarks such as SynthID generally require the provider's
    detector or a secret model/key. This function can detect explicit marker
    strings, hidden colour under transparent pixels, and unusual LSB statistics;
    it cannot verify a proprietary watermark.
    """
    rgba = np.asarray(img.convert("RGBA"))
    rgb = rgba[:, :, :3]
    indicators: List[str] = []

    explicit = []
    if raw:
        explicit = _contains_any(
            raw.decode("latin-1", errors="ignore"), WATERMARK_TERMS
        )
        if explicit:
            indicators.append("embedded watermark identifier: " + ", ".join(explicit))

    alpha = rgba[:, :, 3]
    transparent = alpha == 0
    hidden_rgb_ratio = 0.0
    if np.any(transparent):
        hidden_rgb_ratio = float(np.mean(np.any(rgb[transparent] != 0, axis=1)))
        if hidden_rgb_ratio > 0.05:
            indicators.append("non-zero RGB data exists under fully transparent pixels")

    # Sample large images to bound work while preserving the parity statistics.
    flat = rgb.reshape(-1, 3)
    if len(flat) > 1_000_000:
        flat = flat[:: max(1, len(flat) // 1_000_000)]
    entropies = [_binary_entropy((flat[:, c] & 1).astype(np.uint8)) for c in range(3)]
    lsb_entropy = float(np.mean(entropies))

    spatial = rgb[:: max(1, rgb.shape[0] // 700), :: max(1, rgb.shape[1] // 700)]
    bits = spatial & 1
    transitions = []
    if bits.shape[1] > 1:
        transitions.append(float(np.mean(bits[:, 1:] != bits[:, :-1])))
    if bits.shape[0] > 1:
        transitions.append(float(np.mean(bits[1:, :] != bits[:-1, :])))
    transition_rate = float(np.mean(transitions)) if transitions else 0.0

    # High entropy and near-random transitions are compatible with LSB payloads,
    # but also with sensor noise and JPEG decoding. Keep the score deliberately
    # weak unless an explicit marker or hidden-alpha payload exists.
    statistical_score = max(0.0, (lsb_entropy - .985) * 900)
    statistical_score += max(0.0, .018 - abs(transition_rate - .5)) * 700
    score = min(45.0, statistical_score)
    if explicit:
        score = max(score, 90.0)
    if hidden_rgb_ratio > .05:
        score = max(score, min(85.0, 55.0 + hidden_rgb_ratio * 30))
    if score >= 30 and not explicit and hidden_rgb_ratio <= .05:
        indicators.append("LSB bit planes are unusually close to random")

    return {
        "suspected": score >= 55,
        "score": round(score, 1),
        "explicit_markers": explicit,
        "lsb_entropy": round(lsb_entropy, 4),
        "lsb_transition_rate": round(transition_rate, 4),
        "hidden_rgb_under_alpha_ratio": round(hidden_rgb_ratio, 4),
        "indicators": indicators,
        "note": (
            "A proprietary invisible watermark cannot be confirmed without its "
            "provider-specific detector or verification key."
        ),
    }


def _fingerprints(img: Image.Image, raw: Optional[bytes]) -> Dict[str, Any]:
    """Stable identifiers for matching against a known-image/hash database."""
    gray = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return {
        "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "dhash64": f"{value:016x}",
        "note": (
            "SHA-256 matches exact files; dHash can match visually similar copies "
            "against an external known-image database. No database lookup is performed."
        ),
    }


def analyze_image_provenance(source: Any) -> Dict[str, Any]:
    """Run all non-model provenance checks and return a JSON-safe report."""
    raw = _source_bytes(source)
    try:
        img = _open_image(source, raw)
        img.load()
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    metadata = _metadata(img, raw)
    container = _container(img, raw)
    credentials = _content_credentials(raw, metadata)
    visible = _visible_watermark(img)
    invisible = _invisible_watermark(img, raw)
    fingerprints = _fingerprints(img, raw)

    signals: List[Dict[str, str]] = []
    if metadata["ai_terms"]:
        signals.append({
            "level": "strong",
            "kind": "metadata",
            "title": "AI-generation terms in metadata",
            "detail": ", ".join(metadata["ai_terms"]),
        })
    if metadata["editor_terms"]:
        signals.append({
            "level": "info",
            "kind": "metadata",
            "title": "Editing software metadata",
            "detail": ", ".join(metadata["editor_terms"]),
        })
    if metadata["camera_fields"]:
        signals.append({
            "level": "info",
            "kind": "metadata",
            "title": "Camera-style EXIF fields present",
            "detail": ", ".join(metadata["camera_fields"]),
        })
    if credentials["detected"]:
        signals.append({
            "level": "strong",
            "kind": "credentials",
            "title": "Content-credential container marker",
            "detail": "C2PA/JUMBF-related bytes were found; signature trust is not verified.",
        })
    if visible["suspected"]:
        signals.append({
            "level": "weak",
            "kind": "visible_watermark",
            "title": "Possible visible text/logo watermark",
            "detail": ", ".join(visible["regions"]) or "outer image band",
        })
    if invisible["suspected"]:
        signals.append({
            "level": "weak" if not invisible["explicit_markers"] else "strong",
            "kind": "invisible_watermark",
            "title": "Possible embedded/invisible watermark",
            "detail": "; ".join(invisible["indicators"]) or "statistical anomaly",
        })
    if container["trailing_bytes"]:
        signals.append({
            "level": "weak",
            "kind": "container",
            "title": "Data appended after the image end marker",
            "detail": f"{container['trailing_bytes']} trailing bytes",
        })

    strongest = (
        "strong evidence found" if any(s["level"] == "strong" for s in signals)
        else "heuristic indicators found" if signals
        else "no provenance indicator found"
    )
    return {
        "available": True,
        "summary": strongest,
        "signals": signals,
        "metadata": metadata,
        "content_credentials": credentials,
        "visible_watermark": visible,
        "invisible_watermark": invisible,
        "container": container,
        "fingerprints": fingerprints,
        "affects_verdict": False,
    }
