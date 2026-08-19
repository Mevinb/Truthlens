"""Guards on per-row provenance in the collector.

A mislabelled corpus is the worst failure available here: it trains and
evaluates cleanly, reports a plausible number, and is wrong. The three
behaviours that could produce one silently are pinned below — an excluded class
that must stay excluded, a generator name that must land on its existing
spelling, and a quarantined row that must not look like a leak.
"""

import json

import pytest

pytest.importorskip("PIL")


def _src(**kw):
    """A So-Fake-shaped source, with only the fields under test spelled out."""
    from dataset.collect_modern_generators import Source

    base = dict(
        key="s", repo_id="r/s", label="fake", generator="?", method="parquet",
        license="test", url="http://x",
        where={"label": ["REAL", "FULL_SYNTHETIC"]},
        label_column="authenticity",
        label_map={"REAL": "real", "FAKE": "fake"},
        generator_column="source_name",
        generator_map={"GPT-image-2": "gpt-image-2", "Reddit": "camera/reddit",
                       "Imagen4": "imagen-4"},
        holdout_when={"source_name": ["Imagen4"]},
    )
    base.update(kw)
    return Source(**base)


def _row(label, authenticity, source_name, **extra):
    return {"label": label, "authenticity": authenticity,
            "source_name": source_name, **extra}


# ─── Row filtering ────────────────────────────────────────────────────────────
def test_tampered_rows_are_excluded_by_name():
    """A locally retouched real photograph is not a generated image.

    So-Fake-OOD ships TAMPERED as a third class. Its `authenticity` field says
    FAKE, so anything keying on that alone would sweep 34,000 mostly-real images
    into the fake class and teach the model to reject genuine photos that had
    been edited. The exclusion is by explicit class name so it is visible in the
    source definition rather than implied.
    """
    from dataset.collect_modern_generators import _row_matches

    src = _src()
    assert _row_matches(src, _row("REAL", "REAL", "Reddit"), {})
    assert _row_matches(src, _row("FULL_SYNTHETIC", "FAKE", "GPT-image-2"), {})
    assert not _row_matches(src, _row("TAMPERED", "FAKE", "openai"), {})


def test_where_not_drops_listed_values():
    from dataset.collect_modern_generators import _row_matches

    src = _src(where={}, where_not={"source_name": ["Imagen4"]})
    assert not _row_matches(src, _row("FULL_SYNTHETIC", "FAKE", "Imagen4"), {})
    assert _row_matches(src, _row("FULL_SYNTHETIC", "FAKE", "GPT-image-2"), {})


def test_classlabel_ints_are_matched_by_name_not_position():
    """Filtering on the integer would break when upstream reorders its classes."""
    from dataset.collect_modern_generators import _row_value

    class Feat:
        names = ["real", "fake"]

    assert _row_value({"label": 1}, {"label": Feat()}, "label") == "fake"
    assert _row_value({"label": 0}, {"label": Feat()}, "label") == "real"
    # Out of range must not raise — it degrades to the raw value.
    assert _row_value({"label": 7}, {"label": Feat()}, "label") == "7"


# ─── Per-row provenance ───────────────────────────────────────────────────────
def test_a_known_generator_lands_on_its_existing_corpus_spelling():
    """The single most load-bearing mapping in the source definition.

    The corpus already holds gpt-image-2 from two repositories and scores 12% on
    the unseen one. Ingesting a third source as "GPT-image-2" would register a
    *new* generator, so the held-out-source tier would lose its subject and the
    fix would be invisible in the numbers it was meant to move.
    """
    from dataset.collect_modern_generators import _row_overrides

    ov = _row_overrides(_src(), _row("FULL_SYNTHETIC", "FAKE", "GPT-image-2"), {})
    assert ov["generator"] == "gpt-image-2"
    assert ov["label"] == "fake"
    assert "holdout" not in ov


def test_an_unmapped_generator_is_skipped_loudly(capsys):
    """Passing it through would invent a generator that may already exist."""
    from dataset.collect_modern_generators import _row_overrides

    assert _row_overrides(_src(), _row("FULL_SYNTHETIC", "FAKE", "BrandNew"),
                          {}) is None
    assert "BrandNew" in capsys.readouterr().err


def test_reals_and_fakes_come_off_the_same_stream_with_the_right_labels():
    """One pass has to serve both classes; a 135GB repo cannot be scanned twice."""
    from dataset.collect_modern_generators import _row_overrides

    src = _src()
    assert _row_overrides(src, _row("REAL", "REAL", "Reddit"), {})["label"] == "real"
    assert _row_overrides(
        src, _row("FULL_SYNTHETIC", "FAKE", "GPT-image-2"), {})["label"] == "fake"


def test_rendition_is_recorded_per_row():
    from dataset.collect_modern_generators import _row_overrides

    src = _src(rendition_column="source_type",
               rendition_map={"generator": "pristine", "platform": "wild"})
    fake = _row_overrides(src, _row("FULL_SYNTHETIC", "FAKE", "GPT-image-2",
                                    source_type="generator"), {})
    real = _row_overrides(src, _row("REAL", "REAL", "Reddit",
                                    source_type="platform"), {})
    assert (fake["rendition"], real["rendition"]) == ("pristine", "wild")


# ─── Quarantine ───────────────────────────────────────────────────────────────
def test_a_quarantined_row_gets_a_distinct_source_key(tmp_path, monkeypatch):
    """Otherwise the evaluation cannot use it, and says so.

    A held-out image sharing its source_key with training rows is, to the
    protocol, an image whose provenance the model has already seen. Held-out
    generators from a multi-generator repo have to arrive under their own key or
    the tier that measures them reads as contaminated.
    """
    import io

    from PIL import Image

    from dataset import collect_modern_generators as C

    monkeypatch.setattr(C, "OUTPUT_DIR", tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (7, 90, 140)).save(buf, format="JPEG")
    raw = buf.getvalue()

    manifest = C.ManifestWriter(tmp_path / "manifest.jsonl")
    src = _src()
    trained = C.store(raw, src, set(), manifest,
                      {"label": "fake", "generator": "gpt-image-2"})
    held = C.store(_recolour(), src, set(), manifest,
                   {"label": "fake", "generator": "imagen-4", "holdout": True})
    manifest.close()

    rows = [json.loads(l) for l in
            (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert held == "holdout"
    assert trained != "holdout"
    keys = {r["generator"]: r["source_key"] for r in rows}
    assert keys["gpt-image-2"] == "s"
    assert keys["imagen-4"] == "s_holdout", "quarantined rows need their own key"


def _recolour():
    """A second distinct image, so the content-hash dedupe does not eat it."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()
