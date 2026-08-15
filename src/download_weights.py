"""
TruthLens — Automated Weight Downloader from Hugging Face Hub.
Downloads trained model checkpoints to local models/ directory if missing.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "MevinBenty/truthlens-models"
LOCAL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_all_weights(repo_id: str = REPO_ID, local_dir: Path = LOCAL_MODELS_DIR) -> Path:
    """Download all trained checkpoints from Hugging Face Hub."""
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model checkpoints from {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        ignore_patterns=[".git*", "*.md"],
    )
    print("All checkpoints downloaded and ready.")
    return local_dir


if __name__ == "__main__":
    download_all_weights()
