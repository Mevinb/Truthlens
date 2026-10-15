#!/usr/bin/env python3
"""Smoke-test the Streamlit app's inference path without launching a server.

Imports app.app for real (Streamlit runs headless outside a script context) and
calls load_predictor / run_prediction on genuine test images for every model in
MODEL_OPTIONS, asserting the documented result keys are present.
"""
import glob
import sys
import warnings

warnings.filterwarnings("ignore")

from PIL import Image

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import app.app as A

REQUIRED = {"prediction", "confidence", "probabilities", "emoji",
            "explanation", "likely_generator"}


def find_sample_images() -> tuple[Path, Path]:
    real_candidates = sorted((ROOT_DIR / "datasets/prepared/multires/test/real").glob("highres_*"))
    fake_candidates = sorted((ROOT_DIR / "datasets/prepared/multires/test/fake").glob("highres_*"))

    if real_candidates and fake_candidates:
        return real_candidates[0], fake_candidates[0]

    # Fallback to demo samples shipped with app
    demo_dir = ROOT_DIR / "app" / "demo_samples"
    demo_real = demo_dir / "real_camera_sample.jpg"
    demo_fake = demo_dir / "chatgpt_c2pa_sample.png"
    if not demo_fake.exists():
        demo_fake = demo_dir / "gemini_synthid_sample.png"

    if demo_real.exists() and demo_fake.exists():
        return demo_real, demo_fake

    raise FileNotFoundError(
        "No test images found in datasets/prepared/multires/test/ or app/demo_samples/"
    )


def main() -> int:
    try:
        real_path, fake_path = find_sample_images()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for label, path in (("real", real_path), ("fake", fake_path)):
        img = Image.open(path).convert("RGB")
        for display in A.MODEL_OPTIONS:
            model_type, err = A.load_predictor(display)
            if model_type is None:
                print(f"  LOAD FAIL  {display:<30s} {err}")
                failures += 1
                continue
            result, heatmap = A.run_prediction(model_type, img, with_gradcam=True)
            if result is None:
                print(f"  PRED FAIL  {display:<30s} {heatmap}")
                failures += 1
                continue
            missing = REQUIRED - set(result)
            forensic = [k for k in ("spectral_analysis", "ela_metrics") if k in result]
            hm = "heatmap" if heatmap is not None else "no-heatmap"
            flag = "" if not missing else f"  MISSING {sorted(missing)}"
            if missing:
                failures += 1
            print(f"  {label:<4s} {display:<30s} -> {result['prediction']:<4s} "
                  f"{result['confidence']:5.1f}%  {hm:<11s} {'+'.join(forensic) or 'none'}{flag}")

    print(f"\n  {'FAILURES: ' + str(failures) if failures else 'ALL APP PATHS OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

