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


def test_gemini_synthid_and_google_ai_metadata_detection():
    from src.image_forensics import analyze_image_provenance

    # Simulated Google Gemini image with SynthID marker and Google AI attribution
    report = analyze_image_provenance(
        _png_bytes({
            "Credit": "Generated with Google AI",
            "Provenance": "Google DeepMind SynthID digital watermark embedded",
            "Software": "Google Gemini Imagen 3",
        })
    )

    assert report["available"]
    gemini = report["gemini_provenance"]
    assert gemini["detected"] is True
    assert gemini["confidence"] >= 0.95
    assert "SynthID" in gemini["watermark_type"] or "Google AI" in gemini["watermark_type"]

    attr = report["ai_attribution"]
    assert attr["detected"] is True
    assert attr["provider"] == "Google Gemini"
    assert attr["watermark_detected"] is True
    assert any("Google" in ev for ev in attr["evidence"])
    assert any(s["kind"] == "watermark" and "Google Gemini" in s["title"] for s in report["signals"])


def test_gemini_xmp_iptc_digitalsourcetype_detection():
    from src.image_forensics import analyze_image_provenance

    # Simulated XMP packet with IPTC trainedAlgorithmicMedia and Google credit
    xmp_packet = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" '
        b'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
        b'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">'
        b'<Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>'
        b'<photoshop:Credit>Generated with Google AI</photoshop:Credit>'
        b'</rdf:Description></rdf:RDF></x:xmpmeta>'
    )
    raw = _png_bytes(append=xmp_packet)
    report = analyze_image_provenance(raw)

    assert report["available"]
    assert report["xmp"]["present"] is True
    assert "trainedalgorithmicmedia" in (report["xmp"]["digital_source_type"] or "").lower()
    assert report["gemini_provenance"]["detected"] is True
    assert report["ai_attribution"]["provider"] == "Google Gemini"


def test_chatgpt_c2pa_content_credentials_detection():
    from src.image_forensics import analyze_image_provenance

    # Simulated C2PA JUMBF container assertion bytes for OpenAI ChatGPT / DALL-E 3
    c2pa_payload = (
        b"c2pa_manifest\x00\x00jumb\x00\x00jumd\x00\x00c2ma"
        b'{"claim_generator":"OpenAI/ChatGPT","softwareAgent":"ChatGPT DALL\xc2\xb7E 3","action":"c2pa.created"}'
    )
    raw = _png_bytes(append=c2pa_payload)
    report = analyze_image_provenance(raw)

    assert report["available"]
    c2pa = report["content_credentials"]
    assert c2pa["detected"] is True
    assert c2pa["is_ai_generated"] is True

    chatgpt = report["chatgpt_provenance"]
    assert chatgpt["detected"] is True
    assert chatgpt["confidence"] >= 0.95
    assert "C2PA" in chatgpt["watermark_type"]

    attr = report["ai_attribution"]
    assert attr["detected"] is True
    assert attr["provider"] == "OpenAI ChatGPT"
    assert attr["watermark_detected"] is True
    assert any("OpenAI" in ev or "ChatGPT" in ev for ev in attr["evidence"])
    assert any(s["kind"] == "watermark" and "OpenAI / ChatGPT" in s["title"] for s in report["signals"])


def test_chatgpt_dalle_exif_software_detection():
    from src.image_forensics import analyze_image_provenance

    exif = Image.Exif()
    exif[305] = "DALL·E 3 by OpenAI"  # Software tag
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 150, 200)).save(buf, format="JPEG", exif=exif)

    report = analyze_image_provenance(buf.getvalue())
    assert report["available"]
    chatgpt = report["chatgpt_provenance"]
    assert chatgpt["detected"] is True
    assert report["ai_attribution"]["provider"] == "OpenAI ChatGPT"


def test_camera_image_has_no_ai_attribution():
    from src.image_forensics import analyze_image_provenance

    exif = Image.Exif()
    exif[271] = "Sony"
    exif[272] = "ILCE-7M4"
    exif[305] = "ILCE-7M4 v2.00"
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="JPEG", exif=exif)

    report = analyze_image_provenance(buf.getvalue())
    assert report["available"]
    assert report["gemini_provenance"]["detected"] is False
    assert report["chatgpt_provenance"]["detected"] is False
    assert report["ai_attribution"]["detected"] is False
    assert report["ai_attribution"]["provider"] == "None"
