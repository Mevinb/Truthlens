import csv

from dataset.download_architecture_dataset import (
    choose_split,
    existing_state,
    normalize_architecture,
)


def test_real_label_overrides_architecture_metadata():
    assert normalize_architecture("GAN", label=0) == "real"


def test_architecture_aliases_are_normalized():
    assert normalize_architecture("latent diffusion", label=1) == "LatDiff"
    assert normalize_architecture("pixel diffusion", label=1) == "PixDiff"


def test_split_boundaries_follow_80_10_10_ratio():
    assert choose_split(15_999, 20_000) == "train"
    assert choose_split(16_000, 20_000) == "val"
    assert choose_split(18_000, 20_000) == "test"


def test_existing_state_retains_source_ids_for_resume(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("architecture", "source_id"))
        writer.writeheader()
        writer.writerow({"architecture": "GAN", "source_id": "model/image.png"})

    counts, source_ids = existing_state(manifest)

    assert counts["GAN"] == 1
    assert source_ids == {"model/image.png"}
