#!/usr/bin/env python3
"""
TruthLens — src/train_head.py
==============================
Fit linear heads on the cached frozen-CLIP features from :mod:`src.features_clip`.

What this is for
----------------
Everything expensive already happened at extraction time, so a head fits in
seconds and the whole point is that experiments become cheap enough to run
honestly. Every design choice below trades headline accuracy for a number that
survives contact with an unseen source or generator.

One head per view, and they are reported separately
---------------------------------------------------
The four cached views are not interchangeable evidence. Three are 1:1 native
crops, which are scale-blind — a 224px crop taken at source scale looks the same
whether it came from a 600px photo or a 4000px one, so the source's resolution
cannot be read off them. The fourth is the whole frame, and it *can* be: showing
a 4000px original at 224px averages 18x where a 300px original averages 1.3x, and
"how smooth is this" then reads out resolution, which the corpus audit measures as
this corpus's strongest leak.

Averaging all four into one number would hide that. Fitting one head per view and
printing each one's off-tier drop exposes it, and the ``native`` ensemble below
gives a headline that does not depend on the leaking view at all. If ``frame``
turns out strong in-tier and collapses off-tier, that is the shortcut showing
itself, and it is better to publish that than to average it away.

Calibration and thresholds come from validation, never from the test tier
------------------------------------------------------------------------
A threshold picked on the same rows it is scored against is an upper bound on
what calibration *could* buy, not a claim about a deployed system.
:func:`src.eval_protocol.tier_metrics` already distinguishes these and calls the
first ``oracle``; this module fits the Platt scaler and the fixed-FPR threshold on
the validation split only, then applies them unchanged everywhere else. The gap
between the two is the part most easily fooled by accident, so it is reported
rather than smoothed over.

Why logistic regression and not something stronger
--------------------------------------------------
A stronger head on frozen features fits the corpus's remaining confounds, which
is the failure this whole phase exists to avoid. A linear probe measures how much
of the task is already linearly available in CLIP's representation, and that is
the quantity worth knowing. ``C`` is chosen on validation balanced accuracy
across a coarse grid, because leaving it at the default is an unstated choice
rather than no choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import ROOT_DIR, get_logger

logger = get_logger(__name__)

DEFAULT_CACHE = ROOT_DIR / "features" / "clip_vitl14"
DEFAULT_OUT = ROOT_DIR / "models" / "clip_linear"
METRICS_DIR = ROOT_DIR / "results" / "metrics"

# Grid is coarse on purpose: frozen L2-normalised features put the useful range
# within a couple of orders of magnitude, and a fine grid on 768 dimensions
# overfits the validation split it is chosen on.
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

TARGET_FPR = 0.05


# ─── Cache loading ────────────────────────────────────────────────────────────
@dataclass
class Split:
    """One split's features, one matrix per view, plus the shared row index."""

    name: str
    rows: List[dict]
    features: Dict[str, np.ndarray]
    native: np.ndarray                      # (n, n_views) bool
    readable: np.ndarray                    # (n,) bool
    views: List[str] = field(default_factory=list)

    @property
    def y(self) -> np.ndarray:
        return np.array([1 if r["label"] == "fake" else 0 for r in self.rows])


def load_split(cache: Path, split: str,
               views: Optional[Sequence[str]] = None) -> Split:
    index_path = cache / f"{split}__index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"no feature cache for split {split!r} at {index_path}; "
            f"run: python -m src.features_clip")
    idx = json.loads(index_path.read_text())
    names = [v["name"] for v in idx["views"]]
    want = list(views) if views else names
    missing = [v for v in want if v not in names]
    if missing:
        raise KeyError(f"views {missing} not in cache for {split} (has {names})")

    feats = {v: np.load(cache / f"{split}__{v}.npy").astype(np.float32)
             for v in want}
    native = np.asarray(idx["native_scale"], dtype=bool)
    readable = np.asarray(idx["readable"], dtype=bool)

    # An unreadable image cached as a zero vector would train the head on a
    # meaningless point and score it as a confident something. Drop them here so
    # no downstream count silently includes them.
    keep = np.flatnonzero(readable)
    rows = [idx["rows"][i] for i in keep]
    if len(keep) != len(idx["rows"]):
        logger.warning("%s: dropping %d unreadable rows",
                       split, len(idx["rows"]) - len(keep))
    return Split(split, rows,
                 {v: feats[v][keep] for v in want},
                 native[keep][:, [names.index(v) for v in want]],
                 readable[keep], want)


# ─── Heads ────────────────────────────────────────────────────────────────────
@dataclass
class Head:
    """A fitted per-view head plus the calibration fitted on validation."""

    view: str
    C: float
    model: object
    calibrator: object
    leaks_resolution: bool
    val_balanced_acc: float

    def p_fake(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.decision_function(X).reshape(-1, 1)
        return self.calibrator.predict_proba(raw)[:, 1]


def fit_head(view: str, train: Split, val: Split,
             leaks_resolution: bool) -> Head:
    """Fit one view's head, choosing ``C`` on validation balanced accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    Xtr, ytr = train.features[view], train.y
    Xva, yva = val.features[view], val.y

    best: Tuple[float, float, object] = (-1.0, C_GRID[0], None)
    for C in C_GRID:
        # class_weight balanced because the corpus is not, and an unbalanced
        # head reaches a good-looking accuracy by preferring the larger class.
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        m.fit(Xtr, ytr)
        score = balanced_accuracy_score(yva, m.predict(Xva))
        logger.info("  %-10s C=%-6g val balanced acc %.4f", view, C, score)
        if score > best[0]:
            best = (score, C, m)
    val_ba, C, model = best

    # Platt scaling on the validation split's raw margins. Fitting the
    # calibrator on train would calibrate to margins the head has already
    # overfitted, which is how a confidently wrong probability gets made.
    from sklearn.linear_model import LogisticRegression as Platt
    cal = Platt(max_iter=3000)
    cal.fit(model.decision_function(Xva).reshape(-1, 1), yva)

    logger.info("  %-10s chose C=%g (val balanced acc %.4f)%s",
                view, C, val_ba, "  [leaks resolution]" if leaks_resolution else "")
    return Head(view, C, model, cal, leaks_resolution, float(val_ba))


# ─── Ensembles ────────────────────────────────────────────────────────────────
def ensemble_p(heads: Sequence[Head], split: Split,
               only_native: bool = False) -> np.ndarray:
    """Mean calibrated probability over the chosen heads.

    A mean of calibrated probabilities rather than of raw margins, because the
    margins of differently-scaled heads are not commensurable and averaging them
    silently weights whichever head happens to be most confident.
    """
    use = [h for h in heads if not (only_native and h.leaks_resolution)]
    if not use:
        raise ValueError("no heads left after filtering")
    return np.mean([h.p_fake(split.features[h.view]) for h in use], axis=0)


# ─── Reporting ────────────────────────────────────────────────────────────────
def _group_of(row: dict) -> str:
    """Group a row by what it is evidence about: its generator."""
    return row.get("generator", "?")


def report(name: str, split: Split, y: np.ndarray, p: np.ndarray,
           applied_thr: Optional[float]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    from src.eval_protocol import at_threshold, threshold_at_fpr

    both = (y == 0).any() and (y == 1).any()
    pred_05 = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "auc": round(float(roc_auc_score(y, p)), 4) if both else None,
        "accuracy_at_0.5": round(float(accuracy_score(y, pred_05)), 4),
        "f1_at_0.5": round(float(f1_score(y, pred_05, zero_division=0)), 4),
        "balanced_acc_at_0.5": round(
            float(balanced_accuracy_score(y, pred_05)), 4),
        "real_acc_at_0.5": round(float((pred_05[y == 0] == 0).mean()), 4)
        if (y == 0).any() else None,
        "fake_acc_at_0.5": round(float((pred_05[y == 1] == 1).mean()), 4)
        if (y == 1).any() else None,
    }
    if both:
        oracle = threshold_at_fpr(y, p, TARGET_FPR)
        out["oracle_at_fpr"] = ({"target_fpr": TARGET_FPR,
                                 **at_threshold(y, p, oracle)}
                                if np.isfinite(oracle) else
                                {"target_fpr": TARGET_FPR, "reachable": False})
    if applied_thr is not None and np.isfinite(applied_thr):
        out["applied"] = at_threshold(y, p, applied_thr)

    # Per-generator, because a mean over generators hides the one that fails.
    by_gen: Dict[str, dict] = {}
    for g in sorted({_group_of(r) for r in split.rows}):
        i = [k for k, r in enumerate(split.rows) if _group_of(r) == g]
        yg, pg = y[i], p[i]
        thr = applied_thr if applied_thr is not None else 0.5
        hit = (pg >= thr) if yg[0] == 1 else (pg < thr)
        by_gen[g] = {"n": len(i),
                     "label": "fake" if yg[0] == 1 else "real",
                     "accuracy": round(float(hit.mean()), 4),
                     "mean_p_fake": round(float(pg.mean()), 4)}
    out["by_generator"] = by_gen
    return out


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def print_block(title: str, m: dict) -> None:
    print(f"\n  {title}")
    print(f"    n={m['n']} ({m['n_real']} real / {m['n_fake']} fake)  "
          f"AUC {m['auc'] if m['auc'] is not None else '—'}  "
          f"balanced acc @0.5 {_pct(m['balanced_acc_at_0.5'])}")
    if "applied" in m:
        a = m["applied"]
        print(f"    applied thr {a['threshold']}: "
              f"real acc {_pct(a['real_acc'])}  "
              f"fake recall {_pct(a['fake_recall'])}  "
              f"balanced {_pct(a['balanced_acc'])}")
    o = m.get("oracle_at_fpr")
    if o and o.get("reachable", True) and "balanced_acc" in o:
        print(f"    oracle @{int(o['target_fpr'] * 100)}% FPR: "
              f"fake recall {_pct(o['fake_recall'])}  "
              f"balanced {_pct(o['balanced_acc'])}   [upper bound, not served]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit linear heads on cached CLIP features")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--views", nargs="+", default=None)
    ap.add_argument("--eval-splits", nargs="+", default=["test", "holdout"])
    ap.add_argument("--tag", default="clip_linear",
                    help="name for the metrics json")
    args = ap.parse_args(argv)

    train = load_split(args.cache, "train", args.views)
    val = load_split(args.cache, "val", args.views)
    views = train.views
    idx_meta = json.loads((args.cache / "train__index.json").read_text())
    leaks = {v["name"]: v["leaks_resolution"] for v in idx_meta["views"]}

    logger.info("train %d rows, val %d rows, views %s",
                len(train.rows), len(val.rows), views)
    logger.info("backbone %s/%s, reencode q%s",
                idx_meta["model"], idx_meta["pretrained"],
                idx_meta["reencode_quality"])

    heads = [fit_head(v, train, val, leaks.get(v, False)) for v in views]

    # Thresholds are fitted here, on validation, and then never refitted.
    from src.eval_protocol import threshold_at_fpr
    thresholds: Dict[str, float] = {}
    for h in heads:
        thresholds[h.view] = threshold_at_fpr(val.y, h.p_fake(val.features[h.view]),
                                              TARGET_FPR)
    for label, only_native in (("ensemble_all", False), ("ensemble_native", True)):
        thresholds[label] = threshold_at_fpr(
            val.y, ensemble_p(heads, val, only_native), TARGET_FPR)

    results: Dict[str, dict] = {
        "backbone": {"model": idx_meta["model"],
                     "pretrained": idx_meta["pretrained"],
                     "embed_dim": idx_meta["embed_dim"],
                     "reencode_quality": idx_meta["reencode_quality"]},
        "views": {v: {"leaks_resolution": leaks.get(v, False),
                      "C": next(h.C for h in heads if h.view == v),
                      "val_balanced_acc": next(h.val_balanced_acc
                                               for h in heads if h.view == v)}
                  for v in views},
        "thresholds_from_val": {k: (None if not np.isfinite(v) else round(v, 4))
                                for k, v in thresholds.items()},
        "target_fpr": TARGET_FPR,
        "splits": {},
    }

    for split_name in args.eval_splits:
        try:
            sp = load_split(args.cache, split_name, views)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            continue
        print(f"\n{'=' * 72}\n{split_name.upper()}  ({len(sp.rows)} rows)\n{'=' * 72}")
        block: Dict[str, dict] = {}
        for h in heads:
            m = report(h.view, sp, sp.y, h.p_fake(sp.features[h.view]),
                       thresholds[h.view])
            m["leaks_resolution"] = h.leaks_resolution
            block[h.view] = m
            print_block(f"{h.view}"
                        + ("   [leaks resolution — off-tier drop expected]"
                           if h.leaks_resolution else ""), m)
        for label, only_native in (("ensemble_all", False),
                                   ("ensemble_native", True)):
            m = report(label, sp, sp.y, ensemble_p(heads, sp, only_native),
                       thresholds[label])
            block[label] = m
            print_block(label + ("   [headline: no leaking view]"
                                 if only_native else ""), m)
        results["splits"][split_name] = block

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dest = METRICS_DIR / f"{args.tag}.json"
    dest.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT_DIR)}")

    args.out.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"heads": [{"view": h.view, "C": h.C, "model": h.model,
                            "calibrator": h.calibrator,
                            "leaks_resolution": h.leaks_resolution}
                           for h in heads],
                 "thresholds": thresholds,
                 "backbone": results["backbone"],
                 "target_fpr": TARGET_FPR},
                args.out / "heads.joblib")
    head_path = (args.out / "heads.joblib").resolve()
    try:
        shown = head_path.relative_to(ROOT_DIR)
    except ValueError:
        shown = head_path
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
