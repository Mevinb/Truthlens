#!/usr/bin/env python3
"""
TruthLens — Post-Transform Shortcut Probe
==========================================
Answers a question ``diag_corpus_audit.py`` cannot: of the confounds present in
the corpus *files*, which ones still reach the model after augmentation?

Why the file audit is not enough
--------------------------------
``diag_corpus_audit.py`` reads width, height and container format off disk. It
found the corpus's worst leak that way — 768-1400px short side ran 0.82 fake,
1-2MP ran 0.99 fake, because every modern generator emits 1024px+ while the
Flickr photo sets are capped near 768. But a property of a *file* is only a
shortcut if it survives the transform pipeline into the tensor. ``NativeScaleCrop``
takes 1:1 crops that look identical whether they came from a 600px photo or a
1536px generation, so for half the training views source resolution is simply
gone. Reporting it as a leak anyway is how you end up rebuilding a corpus to fix
something augmentation already handled — or, worse, trusting an augmentation that
does not actually work.

What this measures
------------------
Run the real train transform over a sample of both classes, reduce each output
tensor to a handful of cheap low-level statistics, and fit logistic regression on
those statistics alone. The features carry no semantics — no objects, no
composition, no anatomy — only brightness, colour spread, high-frequency energy
and blockiness. So the balanced accuracy of that model is a floor on how well the
classes can be told apart *without looking at the image content*:

* **~50%** — low-level shortcuts are dead. Whatever the CNN learns, it has to
  come from structure.
* **65-80%** — a real shortcut survives. The CNN will find it first, because it
  is far cheaper than the generator fingerprint, and it will not transfer.
* **>85%** — the corpus is separable on noise statistics alone and any headline
  accuracy from it is fiction.

``--compare-legacy`` runs the same probe twice, once with
``NativeScaleCrop.frame_short_range`` disabled, which is exactly the pipeline as
it stood before the whole-frame branch was equalised. The difference between the
two numbers is what that change bought, measured rather than argued.

This is a diagnostic, not a gate. A probe accuracy of 60% does not mean the CNN
will score 60%; it means 60% is available for free. Read it next to
``diag_eval_all.py``'s ``internet`` and ``holdout`` rows, which measure whether
the model took the free option.

Usage
-----
    python diag_shortcut_probe.py datasets/prepared/modern_v2 datasets/prepared/multires
    python diag_shortcut_probe.py datasets/prepared/modern_v2 --compare-legacy
    python diag_shortcut_probe.py datasets/prepared/modern_v2 --per-class 500 --views 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "results" / "metrics"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

FEATURE_NAMES = (
    "mean_lum", "std_lum", "hp_std", "hf_ratio", "blockiness",
    "chroma_std", "sat_mean", "grad_mean", "clip_low", "clip_high",
)


def features(tensor) -> np.ndarray:
    """Low-level statistics of one post-transform tensor. No semantics.

    Deliberately the kind of thing a first-layer filter bank computes for free —
    if these separate the classes, the network never has to learn anything else.
    """
    arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    # Undo ImageNet normalisation so the numbers mean something in pixel space.
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    rgb = np.clip(arr * std + mean, 0.0, 1.0)
    lum = rgb.mean(axis=0)

    # High-pass residual: fine texture and sensor/decoder noise. Separable
    # [1 2 1]/4 blur done by edge-padded slicing — a np.convolve per row would
    # be 448 Python-level calls per image, and this runs over thousands.
    blur = lum
    for axis in (0, 1):
        pad = np.pad(blur, [(1, 1) if a == axis else (0, 0) for a in (0, 1)],
                     mode="edge")
        lo = pad[:-2] if axis == 0 else pad[:, :-2]
        mid = pad[1:-1] if axis == 0 else pad[:, 1:-1]
        hi = pad[2:] if axis == 0 else pad[:, 2:]
        blur = 0.25 * lo + 0.5 * mid + 0.25 * hi
    hp = lum - blur

    # Radial FFT energy above 0.75 Nyquist — how much fine detail survived
    # whatever resampling the pipeline applied.
    spec = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean())))
    cy, cx = np.array(spec.shape) // 2
    yy, xx = np.mgrid[:spec.shape[0], :spec.shape[1]]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(spec.shape) / 2)
    hf_ratio = spec[radius > 0.75].sum() / max(spec.sum(), 1e-9)

    # Blockiness: JPEG puts discontinuities on an 8px grid. Compares the mean
    # absolute step across block boundaries with the step just inside them, so a
    # naturally busy image does not read as blocky.
    d = np.abs(np.diff(lum, axis=1))
    on_grid = d[:, 7::8].mean() if d.shape[1] > 8 else 0.0
    off_grid = d.mean()
    blockiness = float(on_grid / max(off_grid, 1e-9))

    grad = np.hypot(*np.gradient(lum))
    return np.array([
        float(lum.mean()), float(lum.std()), float(hp.std()), float(hf_ratio),
        blockiness, float(rgb.std(axis=0).mean()),
        float((rgb.max(axis=0) - rgb.min(axis=0)).mean()), float(grad.mean()),
        float((lum < 0.02).mean()), float((lum > 0.98).mean()),
    ], dtype=np.float64)


def sample_files(roots: List[Path], split: str, per_class: int
                 ) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {"real": [], "fake": []}
    for label in ("real", "fake"):
        pool: List[Path] = []
        for root in roots:
            d = root / split / label
            if d.exists():
                pool += [p for p in sorted(d.iterdir())
                         if p.suffix.lower() in EXTS]
        # Stride, not truncate: the first N of a sorted listing is one
        # generator's filename prefix, not a sample of the class.
        step = max(1, len(pool) // per_class) if per_class else 1
        out[label] = pool[::step][:per_class]
    return out


def extract(files: Dict[str, List[Path]], transform, views: int
            ) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    from tqdm import tqdm

    X, y = [], []
    todo = [(p, 1 if label == "fake" else 0)
            for label, paths in files.items() for p in paths]
    for path, target in tqdm(todo, desc="  probing", ncols=70, unit="img",
                             leave=False):
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception:                                          # noqa: BLE001
            continue
        for _ in range(views):
            try:
                X.append(features(transform(img)))
                y.append(target)
            except Exception:                                      # noqa: BLE001
                break
    return np.array(X), np.array(y)


def probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Fit on half, score on the other half. Balanced accuracy is the headline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
    pred = (prob >= 0.5).astype(int)

    weights = dict(sorted(
        zip(FEATURE_NAMES, (float(w) for w in clf.coef_[0])),
        key=lambda kv: -abs(kv[1])))
    return {
        "n": int(len(y)),
        "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
        "auc": round(float(roc_auc_score(yte, prob)), 4),
        "top_features": {k: round(v, 3) for k, v in list(weights.items())[:5]},
    }


def verdict(bal: float) -> str:
    if bal < 0.58:
        return "clean — low-level statistics carry almost no label information"
    if bal < 0.68:
        return "mild — a weak shortcut survives; watch internet/holdout"
    if bal < 0.85:
        return "LEAKING — a real shortcut survives augmentation and the CNN " \
               "will prefer it"
    return "SEVERE — the classes separate on noise statistics alone"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--views", type=int, default=2,
                    help="augmented views per image (the transform is random)")
    ap.add_argument("--compare-legacy", action="store_true",
                    help="also probe with the whole-frame equaliser disabled")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT_DIR))
    from src.preprocessing import get_transforms

    roots = [ROOT_DIR / r for r in args.roots]
    files = sample_files(roots, args.split, args.per_class)
    if not files["real"] or not files["fake"]:
        sys.exit(f"Need both classes under {args.split}/ — found "
                 f"real={len(files['real'])} fake={len(files['fake'])}")

    print("=" * 74)
    print("  POST-TRANSFORM SHORTCUT PROBE")
    print(f"  {', '.join(r.name for r in roots)} · split={args.split}")
    print(f"  real {len(files['real'])} · fake {len(files['fake'])} "
          f"· {args.views} view(s) each")
    print("=" * 74)

    variants: List[Tuple[str, object]] = [
        ("current", get_transforms("train", 224, modern=True))]
    if args.compare_legacy:
        legacy = get_transforms("train", 224, modern=True)
        for step in getattr(legacy, "transforms", []):
            if hasattr(step, "frame_short_range"):
                step.frame_short_range = None      # pre-fix whole-frame branch
        variants.append(("legacy (whole-frame equaliser off)", legacy))

    payload = {"roots": [r.name for r in roots], "split": args.split,
               "per_class": args.per_class, "views": args.views, "variants": {}}

    for name, transform in variants:
        X, y = extract(files, transform, args.views)
        if len(set(y.tolist())) < 2:
            print(f"  {name}: could not build both classes")
            continue
        res = probe(X, y)
        payload["variants"][name] = res
        print(f"\n  {name}")
        print(f"    n={res['n']}  balanced accuracy {res['balanced_accuracy']:.3f}"
              f"  AUC {res['auc']:.3f}")
        print(f"    {verdict(res['balanced_accuracy'])}")
        print("    strongest features: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in res["top_features"].items()))

    if len(payload["variants"]) == 2:
        cur, leg = (payload["variants"][k]["balanced_accuracy"]
                    for k in ("current",
                              "legacy (whole-frame equaliser off)"))
        print("\n" + "-" * 74)
        print(f"  whole-frame equaliser removed {leg - cur:+.3f} balanced "
              f"accuracy from the free shortcut ({leg:.3f} → {cur:.3f})")

    print("=" * 74)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"shortcut_probe_{args.split}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
