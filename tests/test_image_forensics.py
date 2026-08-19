import io
import json

from PIL import Image, PngImagePlugin


def _png_bytes(metadata=None, append=b""):
    buf = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    for key, value in (metadata or {}).items():
        info.add_text(key, value)
    Image.new("RGB", (96, 64), (30, 90, 140)).save(
        buf, format="PNG", pnginfo=info
    )
    return buf.getvalue() + append


def test_ai_generator_metadata_is_reported_as_a_strong_signal():
    from src.image_forensics import analyze_image_provenance

    report = analyze_image_provenance(
        _png_bytes({"Software": "ComfyUI", "parameters": "prompt: lighthouse; seed: 42"})
    )

    assert report["available"]
    assert "comfyui" in report["metadata"]["ai_terms"]
    assert any(
        signal["kind"] == "metadata" and signal["level"] == "strong"
        for signal in report["signals"]
    )
    assert report["affects_verdict"] is False


def test_camera_exif_is_preserved_without_claiming_authenticity():
    from src.image_forensics import analyze_image_provenance

    exif = Image.Exif()
    exif[271] = "Example Camera Corp"  # Make
    exif[272] = "Model One"            # Model
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), "gray").save(buf, format="JPEG", exif=exif)

    report = analyze_image_provenance(buf.getvalue())

    assert report["metadata"]["exif_present"]
    assert {"Make", "Model"}.issubset(report["metadata"]["camera_fields"])
    assert not report["metadata"]["ai_terms"]


def test_explicit_invisible_watermark_identifier_is_detected():
    from src.image_forensics import analyze_image_provenance

    report = analyze_image_provenance(
        _png_bytes({"Provenance": "Google SynthID watermark present"})
    )

    invisible = report["invisible_watermark"]
    assert invisible["suspected"]
    assert "synthid" in invisible["explicit_markers"]
    assert invisible["score"] >= 90


def test_png_trailing_payload_is_reported():
    from src.image_forensics import analyze_image_provenance

    report = analyze_image_provenance(_png_bytes(append=b"hidden payload"))

    assert report["container"]["trailing_bytes"] == len(b"hidden payload")
    assert any(signal["kind"] == "container" for signal in report["signals"])


def test_plain_image_returns_a_complete_json_safe_report():
    from src.image_forensics import analyze_image_provenance

    report = analyze_image_provenance(_png_bytes())

    assert report["available"]
    assert "visible_watermark" in report
    assert "invisible_watermark" in report
    assert "content_credentials" in report
    assert len(report["fingerprints"]["sha256"]) == 64
    assert len(report["fingerprints"]["dhash64"]) == 16
    json.dumps(report)
