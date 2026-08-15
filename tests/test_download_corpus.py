import csv

from PIL import Image

from dataset.download_corpus import architecture_for, existing_hashes, image_hash, split_for


def test_split_for_is_deterministic_and_balanced():
    assert split_for(0, 100) == "train"
    assert split_for(80, 100) == "val"
    assert split_for(90, 100) == "test"


def test_real_samples_are_not_assigned_a_generator_family():
    assert architecture_for({"architecture": "GAN"}, "real", "community_forensics") == "real"


def test_architecture_metadata_is_normalized():
    assert architecture_for({"architecture": "latdiff"}, "fake", "community_forensics") == "LatDiff"
    assert architecture_for({}, "fake", "highres_mixed") == "other"


def test_existing_hashes_support_resume(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sha256", "source", "architecture", "label"))
        writer.writeheader()
        writer.writerow({"sha256": "abc", "source": "cifake", "architecture": "other", "label": "fake"})

    hashes, counts = existing_hashes(manifest)

    assert hashes == {"abc"}
    assert counts[("cifake", "other", "fake")] == 1


def test_image_hash_is_stable():
    image = Image.new("RGB", (2, 2), "white")
    assert image_hash(image) == image_hash(image.copy())
