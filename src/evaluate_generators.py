#!/usr/bin/env python3
"""Evaluate a detector separately on each generator directory."""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import CNNPredictor
from src.utils import Config


def evaluate(fake_root: Path, model_name: str) -> dict[str, float]:
    predictor = CNNPredictor(Config(cnn_model_name=model_name))
    results = {}
    generator_dirs = [path for path in fake_root.iterdir() if path.is_dir()]
    if not generator_dirs:
        generator_dirs = [fake_root]

    for generator_dir in sorted(generator_dirs):
        files = [
            path for path in generator_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        if not files:
            continue
        predictions = [
            predictor.predict(path)["label_index"]
            for path in files
        ]
        results[generator_dir.name] = {
            "images": len(files),
            "fake_recall": round(float(np.mean(predictions)), 4),
            "accuracy": round(float(accuracy_score(np.ones(len(files)), predictions)), 4),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--model-name", default="resnet18_gemini.pth")
    args = parser.parse_args()
    for name, metrics in evaluate(args.fake_root, args.model_name).items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
