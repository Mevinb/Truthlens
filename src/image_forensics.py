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
import json
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
    "chatgpt", "comfyui", "dall-e", "dalle", "dall·e", "deepmind", "diffusion", "firefly",
    "flux", "fooocus", "gemini", "generative fill", "google ai", "google imagen",
    "gpt image", "gpt-image", "gpt-image-1", "gpt-image-2", "imagen", "imagen 2",
    "imagen 3", "invokeai", "leonardo", "midjourney", "negative prompt",
    "novelai", "openai", "prompt", "sampler", "seed", "stable diffusion", "steps",
    "synthid", "com.google.synthid", "text-to-image", "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
}
EDITOR_TERMS = {
    "adobe", "affinity", "canva", "capture one", "darktable", "gimp",
    "google photos", "imagemagick", "lightroom", "photoshop", "snapseed",
}
WATERMARK_TERMS = {
    "c2pa", "contentauth", "content credentials", "digimarc", "imwatermark",
    "invisible-watermark", "synthid", "google-synthid", "synthid_version",
    "trustmark", "watermark", "c2ma", "c2cs", "c2as", "jumbf",
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


def _flat_text(value: Any, limit: int = 500) -> str:
    """Coerce a manifest/metadata value to plain searchable text.

    C2PA fields such as ``claim_generator`` are usually strings, but some
    manifests store them as objects (e.g. ``{"name": ..., ...}``). String
    concatenation with such a value raises ``TypeError: can only concatenate
    str (not "dict") to str`` and would fail the whole prediction, so every
    value entering a ``+``/``join`` text corpus goes through here.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (int, float, bool)):
        return str(value)[:limit]
    if isinstance(value, dict):
        for key in ("name", "title", "value", "label", "text"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()[:limit]
        try:
            return json.dumps(value, ensure_ascii=False)[:limit]
        except Exception:                                        # noqa: BLE001
            return str(value)[:limit]
    if isinstance(value, (list, tuple)):
        return ", ".join(_flat_text(v, 100) for v in value)[:limit]
    return _safe_text(value, limit)


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


def _webp_chunks(raw: bytes) -> Tuple[List[str], int]:
    """Parse WebP container chunks and return (chunk_names, trailing_bytes)."""
    if not (raw.startswith(b"RIFF") and len(raw) >= 12 and raw[8:12] == b"WEBP"):
        return [], 0
    names: List[str] = []
    pos = 12
    last_valid_end = 12
    while pos + 8 <= len(raw):
        chunk_type = raw[pos:pos + 4].decode("latin-1", errors="replace").strip()
        chunk_len = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if not all(c.isalnum() or c in (" ", "_", "-") for c in chunk_type):
            break
        if pos + 8 + chunk_len > len(raw):
            break
        names.append(chunk_type)
        pos += 8 + chunk_len
        if chunk_len % 2 == 1 and pos < len(raw) and raw[pos] == 0:
            pos += 1
        last_valid_end = pos
    trailing = max(0, len(raw) - last_valid_end)
    return names, trailing


def _container(img: Image.Image, raw: Optional[bytes]) -> Dict[str, Any]:
    chunks: List[str] = []
    webp_chunks: List[str] = []
    markers: List[str] = []
    trailing = 0
    if raw:
        chunks, trailing = _png_chunks(raw)
        if not chunks:
            webp_chunks, trailing = _webp_chunks(raw)
        if not chunks and not webp_chunks:
            markers, trailing = _jpeg_markers_and_trailing(raw)
    info = getattr(img, "info", {}) or {}
    fmt = (img.format or "").upper()
    return {
        "format": img.format or "unknown",
        "mode": img.mode,
        "size": list(img.size),
        "png_chunks": chunks,
        "webp_chunks": webp_chunks,
        "jpeg_app_markers": markers,
        "has_alpha": "A" in img.getbands(),
        "has_icc_profile": bool(info.get("icc_profile")),
        "animated": bool(getattr(img, "is_animated", False)),
        "frame_count": int(getattr(img, "n_frames", 1)),
        "trailing_bytes": trailing,
    }


def _extract_xmp(img: Image.Image, raw: Optional[bytes]) -> Dict[str, Any]:
    """Extract and parse XMP packet into structured metadata."""
    xmp_str = ""
    info = getattr(img, "info", {}) or {}
    for key in ("XML:com.adobe.xmp", "xmp", "XMP"):
        val = info.get(key)
        if isinstance(val, (str, bytes)):
            xmp_str = val.decode("utf-8", errors="ignore") if isinstance(val, bytes) else val
            if xmp_str:
                break

    if not xmp_str and raw:
        # Check standard XMP header
        xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
        idx = raw.find(xmp_header)
        if idx != -1:
            start = idx + len(xmp_header)
            end = raw.find(b"</x:xmpmeta>", start)
            if end != -1:
                xmp_str = raw[start:end + len(b"</x:xmpmeta>")].decode("utf-8", errors="ignore")
            else:
                end_pkt = raw.find(b"<?xpacket end=", start)
                if end_pkt != -1:
                    xmp_str = raw[start:end_pkt + 20].decode("utf-8", errors="ignore")
        if not xmp_str:
            # Check WebP RIFF XMP chunk
            if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
                xmp_idx = raw.find(b"XMP ")
                if xmp_idx != -1 and xmp_idx + 8 < len(raw):
                    xmp_len = struct.unpack("<I", raw[xmp_idx + 4:xmp_idx + 8])[0]
                    chunk_slice = raw[xmp_idx + 8: xmp_idx + 8 + xmp_len]
                    xmp_str = chunk_slice.decode("utf-8", errors="ignore")
        if not xmp_str:
            start = raw.find(b"<x:xmpmeta")
            if start != -1:
                end = raw.find(b"</x:xmpmeta>", start)
                if end != -1:
                    xmp_str = raw[start:end + len(b"</x:xmpmeta>")].decode("utf-8", errors="ignore")

    if not xmp_str:
        return {
            "present": False,
            "raw_length": 0,
            "digital_source_type": None,
            "credit": None,
            "creator": None,
            "software": None,
            "source": None,
            "text": "",
        }

    digital_source_type = None
    credit = None
    creator = None
    software = None
    source = None

    dst_match = re.search(r'(?:DigitalSourceType|digitalSourceType)["\'>\s]*([^<"\'\s>]+)', xmp_str, re.IGNORECASE)
    if dst_match:
        digital_source_type = dst_match.group(1).strip()

    credit_match = re.search(r'(?:Credit|credit|photoshop:Credit|Iptc4xmpExt:Credit)["\'>\s]*([^<"\'>]+)', xmp_str, re.IGNORECASE)
    if credit_match:
        credit = credit_match.group(1).strip()

    creator_match = re.search(r'(?:Creator|creator|dc:creator|CreatorTool|xmp:CreatorTool)["\'>\s]*([^<"\'>]+)', xmp_str, re.IGNORECASE)
    if creator_match:
        creator = creator_match.group(1).strip()

    software_match = re.search(r'(?:Software|software|xmp:CreatorTool)["\'>\s]*([^<"\'>]+)', xmp_str, re.IGNORECASE)
    if software_match:
        software = software_match.group(1).strip()

    source_match = re.search(r'(?:Source|source|dc:source|photoshop:Source)["\'>\s]*([^<"\'>]+)', xmp_str, re.IGNORECASE)
    if source_match:
        source = source_match.group(1).strip()

    return {
        "present": True,
        "raw_length": len(xmp_str),
        "digital_source_type": digital_source_type,
        "credit": credit,
        "creator": creator,
        "software": software,
        "source": source,
        "text": xmp_str[:2500],
    }


def _extract_cbor_text_strings(data: bytes) -> List[str]:
    """Extract UTF-8 text strings directly from CBOR data stream without external dependencies."""
    strings: List[str] = []
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        # Major type 3 (text string): high 3 bits are 011 -> 0x60
        if (b & 0xE0) == 0x60:
            val = b & 0x1F
            i += 1
            str_len = None
            if val < 24:
                str_len = val
            elif val == 24 and i < n:
                str_len = data[i]
                i += 1
            elif val == 25 and i + 1 < n:
                str_len = struct.unpack(">H", data[i:i + 2])[0]
                i += 2
            elif val == 26 and i + 3 < n:
                str_len = struct.unpack(">I", data[i:i + 4])[0]
                i += 4
            if str_len is not None and 0 < str_len <= 1500 and i + str_len <= n:
                try:
                    s = data[i:i + str_len].decode("utf-8")
                    if s and all(c.isprintable() or c.isspace() for c in s):
                        strings.append(s)
                except UnicodeDecodeError:
                    pass
                i += str_len
                continue
        i += 1
    return strings


def _c2pa_from_sdk(raw: bytes) -> Optional[Dict[str, Any]]:
    """Try to parse C2PA manifest using official c2pa-python SDK when available."""
    try:
        import c2pa
        reader = c2pa.Reader(io.BytesIO(raw))
        manifest_str = reader.json()
        if not manifest_str:
            return None
        data = json.loads(manifest_str)
        active_id = data.get("active_manifest")
        manifests = data.get("manifests", {})
        active = manifests.get(active_id) or data
        claim_generator = active.get("claim_generator") or data.get("claim_generator")
        title = active.get("title")

        software_agent = None
        action = None
        digital_source_type = None

        assertions = active.get("assertions", [])
        for assertion in assertions:
            label = str(assertion.get("label", ""))
            adata = assertion.get("data", {})
            if "actions" in label or label == "c2pa.actions":
                actions_list = adata.get("actions", [])
                if isinstance(actions_list, list) and actions_list:
                    act_item = actions_list[0]
                    if isinstance(act_item, dict):
                        action = act_item.get("action")
                        if not software_agent:
                            software_agent = act_item.get("softwareAgent")
            if "digital_source_type" in label or "digitalSourceType" in str(adata):
                if isinstance(adata, dict):
                    digital_source_type = adata.get("digitalSourceType") or adata.get("digital_source_type")

        return {
            "claim_generator": _flat_text(claim_generator) or None,
            "software_agent": _flat_text(software_agent) or None,
            "action": _flat_text(action) or None,
            "digital_source_type": _flat_text(digital_source_type) or None,
            "title": _flat_text(title) or None,
            "sdk_verified": True,
        }
    except Exception:
        return None


def _extract_cbor_value(data: bytes, key: str) -> Optional[str]:
    """Find key in CBOR stream and extract the immediate text string value."""
    k_bytes = key.encode("utf-8")
    idx = 0
    while True:
        pos = data.find(k_bytes, idx)
        if pos == -1:
            return None
        after = pos + len(k_bytes)
        if after < len(data):
            b = data[after]
            if (b & 0xE0) == 0x60:  # Major type 3: text string
                val = b & 0x1F
                i = after + 1
                str_len = None
                if val < 24:
                    str_len = val
                elif val == 24 and i < len(data):
                    str_len = data[i]
                    i += 1
                elif val == 25 and i + 1 < len(data):
                    str_len = struct.unpack(">H", data[i:i + 2])[0]
                    i += 2
                elif val == 26 and i + 3 < len(data):
                    str_len = struct.unpack(">I", data[i:i + 4])[0]
                    i += 4
                if str_len is not None and 0 < str_len <= 1500 and i + str_len <= len(data):
                    try:
                        return data[i:i + str_len].decode("utf-8")
                    except UnicodeDecodeError:
                        pass
        idx = pos + 1


def _content_credentials(raw: Optional[bytes], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Parse C2PA JUMBF container and extract claims / software agents."""
    if not raw:
        return {
            "detected": False,
            "c2pa_marker": False,
            "jumbf_marker": False,
            "manifest_reference": False,
            "claim_generator": None,
            "software_agent": None,
            "action": None,
            "digital_source_type": None,
            "is_ai_generated": False,
            "note": "No C2PA/JUMBF marker found. Absence does not imply the image is authentic.",
        }

    # 1. Try SDK parser first if supported
    sdk_res = _c2pa_from_sdk(raw)

    low = raw.lower()
    c2pa_marker = any(token in low for token in (b"c2pa", b"contentauth", b"content credentials"))
    jumbf_marker = any(token in low for token in (b"jumb", b"jumd", b"c2ma", b"c2cs", b"c2as", b"catx"))
    manifest_ref = any(token in low for token in (b"c2pa_manifest", b"dcterms:provenance", b"manifeststore"))

    detected = bool(sdk_res or c2pa_marker or jumbf_marker or manifest_ref)
    if not detected:
        return {
            "detected": False,
            "c2pa_marker": False,
            "jumbf_marker": False,
            "manifest_reference": False,
            "claim_generator": None,
            "software_agent": None,
            "action": None,
            "digital_source_type": None,
            "is_ai_generated": False,
            "note": "No C2PA/JUMBF marker found. Absence does not imply the image is authentic.",
        }

    claim_generator = sdk_res.get("claim_generator") if sdk_res else None
    software_agent = sdk_res.get("software_agent") if sdk_res else None
    action = sdk_res.get("action") if sdk_res else None
    digital_source_type = sdk_res.get("digital_source_type") if sdk_res else None

    # 2. Extract CBOR key-value matches directly
    if not claim_generator:
        claim_generator = _extract_cbor_value(raw, "claim_generator") or _extract_cbor_value(raw, "claimGenerator")
    if not software_agent:
        software_agent = _extract_cbor_value(raw, "softwareAgent") or _extract_cbor_value(raw, "software_agent")
    if not action:
        action = _extract_cbor_value(raw, "action") or _extract_cbor_value(raw, "c2pa.action")
    if not digital_source_type:
        digital_source_type = _extract_cbor_value(raw, "digitalSourceType") or _extract_cbor_value(raw, "digital_source_type")

    # 3. String & regex scanning with UTF-8 decoding (preventing latin-1 corruption of special chars like ·)
    raw_utf8 = raw.decode("utf-8", errors="ignore")
    raw_utf8_low = raw_utf8.lower()
    raw_latin1 = raw.decode("latin-1", errors="ignore")

    if not claim_generator:
        gen_match = re.search(r'["\']?(?:claim_generator|claimGenerator)["\']?\s*[:=]?\s*["\']?([^"\'\,\}\s]+)', raw_utf8, re.IGNORECASE)
        if gen_match:
            claim_generator = gen_match.group(1).strip()
        elif "openai" in raw_utf8_low and ("c2pa" in raw_utf8_low or "jumb" in raw_utf8_low):
            claim_generator = "OpenAI"
        elif "google" in raw_utf8_low and ("c2pa" in raw_utf8_low or "jumb" in raw_utf8_low):
            claim_generator = "Google"

    if not software_agent:
        sa_match = re.search(r'["\']?(?:softwareAgent|software_agent)["\']?\s*[:=]?\s*["\']?([^"\'\,\}\r\n]+)', raw_utf8, re.IGNORECASE)
        if sa_match:
            candidate = sa_match.group(1).strip().strip('"\'')
            if candidate and len(candidate) <= 80:
                software_agent = candidate
        if not software_agent:
            if "gpt-image-2" in raw_utf8_low:
                software_agent = "ChatGPT (GPT-Image-2)"
            elif "gpt-image-1" in raw_utf8_low:
                software_agent = "ChatGPT (GPT-Image-1)"
            elif "chatgpt" in raw_utf8_low:
                software_agent = "ChatGPT"
            elif any(d in raw_utf8_low or d in raw_latin1.lower() for d in ("dall-e", "dall·e", "dalle")):
                software_agent = "DALL-E 3"
            elif "gemini" in raw_utf8_low or "imagen" in raw_utf8_low:
                software_agent = "Google Imagen 3 (Gemini)"

    if not action:
        act_match = re.search(r'["\']?(?:c2pa\.action|action)["\']?\s*[:=]?\s*["\']?([^"\'\,\}\s]+)', raw_utf8, re.IGNORECASE)
        if act_match:
            action = act_match.group(1).strip()

    if not digital_source_type:
        if "trainedalgorithmicmedia" in raw_utf8_low or "trainedalgorithmicmedia" in raw_latin1.lower():
            digital_source_type = "trainedAlgorithmicMedia"
        elif "compositewithtrainedalgorithmicmedia" in raw_utf8_low or "compositewithtrainedalgorithmicmedia" in raw_latin1.lower():
            digital_source_type = "compositeWithTrainedAlgorithmicMedia"

    is_ai = bool(
        digital_source_type
        or (software_agent and any(term in _flat_text(software_agent).lower() for term in ("chatgpt", "dall-e", "dalle", "dall·e", "gpt-image", "firefly", "midjourney", "imagen", "gemini")))
        or (claim_generator and any(term in _flat_text(claim_generator).lower() for term in ("openai", "chatgpt", "google", "dall-e", "dall·e", "dalle")))
        or (sdk_res and (sdk_res.get("claim_generator") or sdk_res.get("software_agent")))
    )

    return {
        "detected": True,
        "c2pa_marker": c2pa_marker,
        "jumbf_marker": jumbf_marker,
        "manifest_reference": manifest_ref,
        "claim_generator": claim_generator,
        "software_agent": software_agent,
        "action": action,
        "digital_source_type": digital_source_type,
        "is_ai_generated": is_ai,
        "sdk_verified": bool(sdk_res),
        "note": "C2PA/JUMBF content credential markers and claims identified.",
    }


def _detect_gemini(metadata: Dict[str, Any], xmp: Dict[str, Any], c2pa: Dict[str, Any], invisible: Dict[str, Any], raw: Optional[bytes]) -> Dict[str, Any]:
    """Detect Google Gemini / Imagen specific watermarks and provenance metadata."""
    evidence = []
    confidence = 0.0
    watermark_type = None
    model_name = "Imagen 3 / Gemini"

    raw_low = raw.lower() if raw else b""
    raw_utf8 = raw.decode("utf-8", errors="ignore").lower() if raw else ""
    fields_low = {str(k).lower(): _flat_text(v).lower() for k, v in metadata.get("fields", {}).items()}
    text_corpus = (
        "\n".join(f"{k}: {_flat_text(v)}" for k, v in metadata.get("fields", {}).items())
        + "\n" + _flat_text(xmp.get("text", ""), 2500)
        + "\n" + _flat_text(c2pa.get("claim_generator"), 500)
        + "\n" + _flat_text(c2pa.get("software_agent"), 500)
        + "\n" + raw_utf8
    ).lower()

    # Determine specific model variant
    if "imagen 3" in text_corpus:
        model_name = "Google Imagen 3 (Gemini)"
    elif "imagen 2" in text_corpus:
        model_name = "Google Imagen 2"
    elif "gemini" in text_corpus:
        model_name = "Google Gemini"

    # 1. SynthID watermarking detection
    synthid_present = (
        "synthid" in text_corpus
        or "synthid" in invisible.get("explicit_markers", [])
        or b"synthid" in raw_low
        or "com.google.synthid" in text_corpus
    )
    if synthid_present:
        evidence.append("Google DeepMind SynthID watermark signature identified")
        confidence = max(confidence, 0.98)
        watermark_type = "Google SynthID Invisible Watermark"

    # 2. IPTC DigitalSourceType = trainedAlgorithmicMedia + Google Credit / Source
    dst = (xmp.get("digital_source_type") or c2pa.get("digital_source_type") or "").lower()
    credit = (xmp.get("credit") or fields_low.get("credit") or "").lower()
    source = (xmp.get("source") or fields_low.get("source") or "").lower()
    software = (xmp.get("software") or fields_low.get("software") or "").lower()
    creator = (xmp.get("creator") or fields_low.get("creator") or fields_low.get("artist") or "").lower()

    is_google_ai_credit = any(term in credit for term in ("google ai", "generated with google", "google", "deepmind"))
    is_google_software = any(term in software for term in ("gemini", "imagen", "google", "deepmind"))
    is_google_source = any(term in source or term in creator for term in ("google", "deepmind", "gemini", "imagen"))

    is_algorithmic = "trainedalgorithmicmedia" in dst or "compositewithtrainedalgorithmicmedia" in dst

    if is_algorithmic and (is_google_ai_credit or is_google_software or is_google_source):
        credit_display = xmp.get("credit") or metadata.get("fields", {}).get("Credit") or "Google AI"
        evidence.append(f"IPTC DigitalSourceType is trainedAlgorithmicMedia with Google credit ('{credit_display}')")
        confidence = max(confidence, 0.99)
        watermark_type = watermark_type or "Google AI Metadata (IPTC Standard)"
    elif is_google_ai_credit:
        credit_val = xmp.get("credit") or metadata.get("fields", {}).get("Credit")
        evidence.append(f"Credit tag indicates Google AI generation: '{credit_val}'")
        confidence = max(confidence, 0.95)
        watermark_type = watermark_type or "Google AI Credit Metadata"

    # 3. Google C2PA manifest
    if c2pa.get("detected"):
        cg = _flat_text(c2pa.get("claim_generator"), 500).lower()
        sa = _flat_text(c2pa.get("software_agent"), 500).lower()
        if any(t in cg for t in ("google", "deepmind", "imagen", "gemini")) or any(t in sa for t in ("gemini", "imagen", "google")):
            evidence.append(f"C2PA manifest confirms Google provenance (Claim: {c2pa.get('claim_generator')}, Agent: {c2pa.get('software_agent')})")
            confidence = max(confidence, 0.99)
            watermark_type = "Google C2PA Content Credentials"

    # 4. Explicit Gemini / Imagen tags in metadata
    if any(term in text_corpus for term in ("gemini", "imagen 2", "imagen 3", "google imagen", "deepmind imagen")):
        if not evidence:
            evidence.append("Metadata specifies Gemini/Imagen model family")
            confidence = max(confidence, 0.90)
            watermark_type = watermark_type or "Google Gemini Model Metadata"

    return {
        "detected": bool(evidence),
        "confidence": confidence,
        "evidence": evidence,
        "watermark_type": watermark_type or "None",
        "model_name": model_name,
    }


def _detect_chatgpt(metadata: Dict[str, Any], xmp: Dict[str, Any], c2pa: Dict[str, Any], raw: Optional[bytes]) -> Dict[str, Any]:
    """Detect OpenAI / ChatGPT (DALL-E 3 / GPT-Image) specific C2PA credentials & metadata."""
    evidence = []
    confidence = 0.0
    watermark_type = None
    model_name = "DALL-E 3 (ChatGPT)"

    raw_low = raw.lower() if raw else b""
    raw_utf8 = raw.decode("utf-8", errors="ignore").lower() if raw else ""
    fields_low = {str(k).lower(): _flat_text(v).lower() for k, v in metadata.get("fields", {}).items()}
    text_corpus = (
        "\n".join(f"{k}: {_flat_text(v)}" for k, v in metadata.get("fields", {}).items())
        + "\n" + _flat_text(xmp.get("text", ""), 2500)
        + "\n" + _flat_text(c2pa.get("claim_generator"), 500)
        + "\n" + _flat_text(c2pa.get("software_agent"), 500)
        + "\n" + raw_utf8
    ).lower()

    # Determine modern model family
    if "gpt-image-2" in text_corpus:
        model_name = "ChatGPT (GPT-Image-2)"
    elif "gpt-image-1" in text_corpus or "gpt image 1" in text_corpus:
        model_name = "ChatGPT (GPT-Image-1)"
    elif "dall-e 3" in text_corpus or "dall·e 3" in text_corpus or "dalle 3" in text_corpus or "dalle3" in text_corpus:
        model_name = "ChatGPT (DALL-E 3)"
    elif "chatgpt" in text_corpus:
        model_name = "ChatGPT"

    # 1. C2PA Credentials from OpenAI / ChatGPT
    if c2pa.get("detected"):
        cg = _flat_text(c2pa.get("claim_generator"), 500).lower()
        sa = _flat_text(c2pa.get("software_agent"), 500).lower()
        has_openai_cg = any(term in cg for term in ("openai", "chatgpt", "dall-e", "dalle", "dall·e", "gpt-image"))
        has_openai_sa = any(term in sa for term in ("chatgpt", "dall-e", "dalle", "dall·e", "openai", "gpt-image"))
        if has_openai_cg or has_openai_sa:
            agent = c2pa.get('software_agent') or c2pa.get('claim_generator') or 'ChatGPT'
            evidence.append(f"C2PA Content Credentials identify OpenAI agent ({agent})")
            confidence = max(confidence, 0.99)
            watermark_type = "C2PA Content Credentials (OpenAI)"
        elif c2pa.get("is_ai_generated") and any(t in text_corpus for t in ("chatgpt", "openai", "dall-e", "dalle", "dall·e", "gpt-image")):
            evidence.append("C2PA AI credential coupled with OpenAI indicators")
            confidence = max(confidence, 0.97)
            watermark_type = "C2PA Content Credentials (OpenAI)"

    # 2. EXIF / IPTC / XMP tags for DALL-E, GPT-Image, or ChatGPT
    software = fields_low.get("software", "") or (xmp.get("software") or "").lower()
    if any(term in software for term in ("dall·e", "dall-e", "dalle", "chatgpt", "openai", "gpt-image")):
        soft_val = metadata.get('fields', {}).get('Software') or xmp.get('software') or "OpenAI Software"
        evidence.append(f"Software metadata tag: '{soft_val}'")
        confidence = max(confidence, 0.99)
        watermark_type = watermark_type or "OpenAI EXIF Metadata"

    # 3. Artist, Copyright, or Description tags
    for tag_key in ("artist", "copyright", "description", "comment", "usercomment"):
        val = fields_low.get(tag_key, "")
        if val and any(t in val for t in ("openai", "dall-e", "dall·e", "chatgpt")):
            evidence.append(f"Metadata {tag_key.capitalize()} tag confirms OpenAI provenance: '{metadata.get('fields', {}).get(tag_key.capitalize()) or val}'")
            confidence = max(confidence, 0.98)
            watermark_type = watermark_type or "OpenAI Metadata Assertion"

    # 4. Prompt / Generation parameters
    if "openai" in text_corpus and any(term in text_corpus for term in ("dall-e", "dall·e", "prompt", "generation", "gpt-image")):
        if not evidence:
            evidence.append("OpenAI generation indicators in metadata")
            confidence = max(confidence, 0.92)
            watermark_type = watermark_type or "OpenAI Model Metadata"

    return {
        "detected": bool(evidence),
        "confidence": confidence,
        "evidence": evidence,
        "watermark_type": watermark_type or "None",
        "model_name": model_name,
    }


def _ai_attribution(metadata: Dict[str, Any], xmp: Dict[str, Any], c2pa: Dict[str, Any], gemini: Dict[str, Any], chatgpt: Dict[str, Any]) -> Dict[str, Any]:
    """Provide consolidated AI provider & model provenance attribution."""
    if gemini.get("detected"):
        return {
            "detected": True,
            "provider": "Google Gemini",
            "model_name": gemini.get("model_name", "Imagen 3 / Gemini"),
            "watermark_detected": True,
            "watermark_type": gemini.get("watermark_type", "SynthID & AI Metadata"),
            "confidence": gemini.get("confidence", 0.99),
            "evidence": gemini.get("evidence", []),
            "digital_source_type": xmp.get("digital_source_type") or c2pa.get("digital_source_type"),
            "c2pa_claim_generator": c2pa.get("claim_generator"),
        }
    if chatgpt.get("detected"):
        return {
            "detected": True,
            "provider": "OpenAI ChatGPT",
            "model_name": chatgpt.get("model_name", "DALL-E 3"),
            "watermark_detected": True,
            "watermark_type": chatgpt.get("watermark_type", "C2PA Content Credentials"),
            "confidence": chatgpt.get("confidence", 0.99),
            "evidence": chatgpt.get("evidence", []),
            "digital_source_type": xmp.get("digital_source_type") or c2pa.get("digital_source_type"),
            "c2pa_claim_generator": c2pa.get("claim_generator"),
        }

    ai_terms = metadata.get("ai_terms", [])
    if ai_terms:
        provider = "Stable Diffusion" if any(t in ai_terms for t in ("stable diffusion", "automatic1111", "comfyui", "flux")) else \
                   "Midjourney" if "midjourney" in ai_terms else \
                   "Adobe Firefly" if "firefly" in ai_terms else "AI Generator"
        return {
            "detected": True,
            "provider": provider,
            "model_name": provider,
            "watermark_detected": False,
            "watermark_type": "Generator Metadata Parameters",
            "confidence": 0.95,
            "evidence": [f"AI terms in metadata: {', '.join(ai_terms)}"],
            "digital_source_type": xmp.get("digital_source_type") or c2pa.get("digital_source_type"),
            "c2pa_claim_generator": c2pa.get("claim_generator"),
        }

    if c2pa.get("is_ai_generated"):
        return {
            "detected": True,
            "provider": c2pa.get("claim_generator") or "AI Certified",
            "model_name": c2pa.get("software_agent") or "Generative AI",
            "watermark_detected": True,
            "watermark_type": "C2PA DigitalSourceType",
            "confidence": 0.98,
            "evidence": ["C2PA Content Credentials indicate synthetic media"],
            "digital_source_type": c2pa.get("digital_source_type"),
            "c2pa_claim_generator": c2pa.get("claim_generator"),
        }

    return {
        "detected": False,
        "provider": "None",
        "model_name": "None",
        "watermark_detected": False,
        "watermark_type": "None",
        "confidence": 0.0,
        "evidence": [],
        "digital_source_type": None,
        "c2pa_claim_generator": None,
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
    detector or a secret model/key. This function detects explicit marker
    strings, hidden colour under transparent pixels, wavelet high-frequency
    anomalies, and unusual LSB statistics.
    """
    rgba = np.asarray(img.convert("RGBA"))
    rgb = rgba[:, :, :3]
    indicators: List[str] = []

    explicit = []
    if raw:
        text_utf8 = raw.decode("utf-8", errors="ignore")
        explicit = _contains_any(text_utf8, WATERMARK_TERMS)
        if not explicit:
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

    # 2D Haar Wavelet high-frequency diagonal dispersion check
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gh, gw = gray.shape
    gh_even = gh - (gh % 2)
    gw_even = gw - (gw % 2)
    diagonal_ratio = 0.0
    wavelet_anomaly = False
    if gh_even >= 64 and gw_even >= 64:
        sub = gray[:gh_even, :gw_even].astype(np.float32)
        row_diff = (sub[:, 0::2] - sub[:, 1::2]) * 0.5
        row_avg  = (sub[:, 0::2] + sub[:, 1::2]) * 0.5
        lh = (row_avg[0::2, :] - row_avg[1::2, :]) * 0.5
        hl = (row_diff[0::2, :] + row_diff[1::2, :]) * 0.5
        hh = (row_diff[0::2, :] - row_diff[1::2, :]) * 0.5
        hh_var = float(np.var(hh))
        hl_var = float(np.var(hl))
        lh_var = float(np.var(lh))
        total_hf_var = hh_var + hl_var + lh_var
        if total_hf_var > 1e-4:
            diagonal_ratio = float(hh_var / total_hf_var)
            if diagonal_ratio > 0.34:
                wavelet_anomaly = True
                indicators.append(f"wavelet high-frequency dispersion anomaly ({diagonal_ratio:.3f})")

    # High entropy and near-random transitions are compatible with LSB payloads,
    # but also with sensor noise and JPEG decoding. Keep the score deliberately
    # weak unless an explicit marker or hidden-alpha payload exists.
    statistical_score = max(0.0, (lsb_entropy - .985) * 900)
    statistical_score += max(0.0, .018 - abs(transition_rate - .5)) * 700
    if wavelet_anomaly:
        statistical_score += 15.0
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
        "diagonal_wavelet_ratio": round(diagonal_ratio, 4),
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
    xmp = _extract_xmp(img, raw)
    credentials = _content_credentials(raw, metadata)
    visible = _visible_watermark(img)
    invisible = _invisible_watermark(img, raw)
    fingerprints = _fingerprints(img, raw)

    gemini = _detect_gemini(metadata, xmp, credentials, invisible, raw)
    chatgpt = _detect_chatgpt(metadata, xmp, credentials, raw)
    attribution = _ai_attribution(metadata, xmp, credentials, gemini, chatgpt)

    signals: List[Dict[str, str]] = []
    if gemini["detected"]:
        signals.append({
            "level": "strong",
            "kind": "watermark",
            "title": "Google Gemini / SynthID provenance detected",
            "detail": "; ".join(gemini["evidence"]),
        })
    if chatgpt["detected"]:
        signals.append({
            "level": "strong",
            "kind": "watermark",
            "title": "OpenAI / ChatGPT provenance detected",
            "detail": "; ".join(chatgpt["evidence"]),
        })
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
    if credentials["detected"] and not (gemini["detected"] or chatgpt["detected"]):
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
    if invisible["suspected"] and not gemini["detected"]:
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
        "xmp": xmp,
        "content_credentials": credentials,
        "gemini_provenance": gemini,
        "chatgpt_provenance": chatgpt,
        "ai_attribution": attribution,
        "visible_watermark": visible,
        "invisible_watermark": invisible,
        "container": container,
        "fingerprints": fingerprints,
        "affects_verdict": False,
    }
