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

import app.app as A

REQUIRED = {"prediction", "confidence", "probabilities", "emoji",
            "explanation", "likely_generator"}

real = sorted(glob.glob("datasets/prepared/multires/test/real/highres_*"))[0]
fake = sorted(glob.glob("datasets/prepared/multires/test/fake/highres_*"))[0]

failures = 0
for label, path in (("real", real), ("fake", fake)):
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
sys.exit(1 if failures else 0)
