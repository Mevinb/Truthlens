"""Thin HTTP layer over TruthLens' existing, shared inference module."""

from __future__ import annotations

import base64
import io
import math
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import (  # noqa: E402
    CNN_TYPES,
    GRADCAM_TYPES,
    MODEL_BLURB,
    MODEL_OPTIONS,
    benchmark_rows,
    load_predictor,
    model_scores,
    resolution_rows,
    run_prediction,
    tier_for_size,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
INFERENCE_LOCK = Lock()
logger = logging.getLogger(__name__)

app = FastAPI(title="TruthLens API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _heatmap_data_url(heatmap: Any) -> str | None:
    if heatmap is None:
        return None
    array = np.asarray(heatmap)
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _predict(data: bytes, model_type: str, with_gradcam: bool) -> tuple[dict[str, Any], str | None]:
    with INFERENCE_LOCK if model_type in CNN_TYPES else nullcontext():
        loaded_model, error = load_predictor(model_type)
        if error or loaded_model is None:
            raise RuntimeError(error or "The selected model could not be loaded.")
        result, heatmap = run_prediction(loaded_model, data, with_gradcam)
        if result is None:
            raise RuntimeError(str(heatmap or "Inference did not return a result."))
        return _jsonable(result), _heatmap_data_url(heatmap)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "truthlens-inference"}


@app.get("/models")
def models() -> dict[str, Any]:
    items = []
    for label, key in MODEL_OPTIONS.items():
        items.append(
            {
                "key": key,
                "label": label,
                "blurb": MODEL_BLURB.get(key, ""),
                "supports_gradcam": key in GRADCAM_TYPES,
                "scores": _jsonable(model_scores(key)),
            }
        )
    return {"models": items}


@app.get("/benchmarks")
def benchmarks() -> dict[str, Any]:
    return {
        "models": _jsonable(benchmark_rows()),
        "resolutions": _jsonable(resolution_rows()),
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_type: str = Form("swin_improved_ep6"),
    with_gradcam: bool = Form(False),
) -> dict[str, Any]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WebP, or BMP image.")
    if model_type not in MODEL_OPTIONS.values():
        raise HTTPException(status_code=400, detail="Unknown model selection.")
    if with_gradcam and model_type not in GRADCAM_TYPES:
        with_gradcam = False

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 25 MB or smaller.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from exc

    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Images may contain at most 40 million pixels.")

    try:
        result, heatmap = await run_in_threadpool(_predict, data, model_type, with_gradcam)
    except RuntimeError as exc:
        logger.exception("Inference failed for model %s", model_type)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check the Python service logs for details.",
        ) from exc

    return {
        "filename": file.filename or "upload",
        "width": width,
        "height": height,
        "resolution_tier": tier_for_size(width, height),
        "model_type": model_type,
        "result": result,
        "heatmap": heatmap,
    }
