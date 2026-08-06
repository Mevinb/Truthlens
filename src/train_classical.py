#!/usr/bin/env python3
"""
TruthLens — src/train_classical.py
====================================
Train and evaluate 6 classical ML models on HOG + LBP features.

Models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. SVM (RBF kernel)
  5. k-Nearest Neighbours
  6. Gaussian Naive Bayes

Features: HOG (1764-d) + LBP (64-d) = 1828-dimensional vector per image

Usage:
    python src/train_classical.py
    python src/train_classical.py --max-samples 5000  # quick test run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.preprocessing import build_feature_dataset
from src.utils import (
    Config,
    get_logger,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    print_classification_report,
    save_metrics,
    save_sklearn_model,
    seed_everything,
)

logger = get_logger(__name__, log_file=Path("results/logs/train_classical.log"))


# ─── Model Definitions ────────────────────────────────────────────────────────
def get_model_zoo(cfg: Config) -> Dict[str, Pipeline]:
    """Return all classical ML pipelines, each with StandardScaler pre-processing."""
    rs = cfg.random_state

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=200, random_state=rs)),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs",
                random_state=rs, n_jobs=cfg.n_jobs,
            )),
        ]),

        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                max_depth=20, min_samples_split=10,
                min_samples_leaf=5, random_state=rs,
            )),
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5,
                n_jobs=cfg.n_jobs, random_state=rs,
            )),
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=200, random_state=rs)),
            ("clf",    SVC(
                C=10.0, kernel="rbf", gamma="scale",
                probability=True, random_state=rs,
            )),
        ]),

        "k-NN": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=150, random_state=rs)),
            ("clf",    KNeighborsClassifier(
                n_neighbors=7, metric="euclidean", n_jobs=cfg.n_jobs,
            )),
        ]),

        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GaussianNB()),
        ]),
    }


# ─── Train & Evaluate One Model ───────────────────────────────────────────────
def train_and_evaluate(
    name:     str,
    pipeline: Pipeline,
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    cfg:      Config,
) -> Dict[str, Any]:
    """Fit, cross-validate, and test a single model. Return metrics dict."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )

    logger.info(f"\n{'─'*55}")
    logger.info(f"  Training: {name}")
    logger.info(f"{'─'*55}")

    # ── Cross-validation on training set ──
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    cv_start = time.time()
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=skf,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=min(cfg.n_jobs, cfg.cv_folds),
        return_train_score=True,
    )
    cv_time = time.time() - cv_start

    cv_acc_mean = cv_results["test_accuracy"].mean()
    cv_acc_std  = cv_results["test_accuracy"].std()
    logger.info(
        f"  CV Accuracy ({cfg.cv_folds}-fold): "
        f"{cv_acc_mean*100:.2f}% ± {cv_acc_std*100:.2f}%  "
        f"[{cv_time:.1f}s]"
    )

    # ── Final fit on full training set ──
    fit_start = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - fit_start
    logger.info(f"  Fit time: {fit_time:.1f}s")

    # ── Test set evaluation ──
    y_pred  = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    class_names = cfg.class_names
    report_dict = print_classification_report(y_test, y_pred, class_names, name)

    # ── Plots ──
    safe_name = name.replace(" ", "_").lower()
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        title=f"{name} — Confusion Matrix",
        save_path=cfg.results_dir / "plots" / f"cm_{safe_name}.png",
    )
    auc = plot_roc_curve(
        y_test, y_score,
        title=f"{name} — ROC Curve",
        save_path=cfg.results_dir / "plots" / f"roc_{safe_name}.png",
    )

    metrics = {
        "model":    name,
        "accuracy": round(acc,  4),
        "precision":round(prec, 4),
        "recall":   round(rec,  4),
        "f1":       round(f1,   4),
        "auc":      round(auc,  4),
        "cv_accuracy_mean": round(float(cv_acc_mean), 4),
        "cv_accuracy_std":  round(float(cv_acc_std),  4),
        "fit_time_sec":     round(fit_time, 2),
        "classification_report": report_dict,
    }

    logger.info(
        f"  Test  → Acc: {acc*100:.2f}%  F1: {f1:.4f}  AUC: {auc:.4f}"
    )
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
def train_all(cfg: Config, max_samples: int = 0) -> Dict[str, Dict]:
    """Train all classical ML models and save results."""
    seed_everything(cfg.random_state)

    train_dir = cfg.data_dir / "train"
    test_dir  = cfg.data_dir / "test"

    if not train_dir.exists():
        raise RuntimeError(
            "Dataset not found. Run:\n"
            "  python dataset/download_cifake.py --method kaggle"
        )

    # ── Feature Extraction ──
    logger.info("\n[Step 1/3] Extracting HOG + LBP features from train set ...")
    kw = {"max_samples_per_class": max_samples} if max_samples > 0 else {}
    X_train, y_train = build_feature_dataset(train_dir, **kw)

    logger.info("[Step 2/3] Extracting features from test set ...")
    X_test, y_test = build_feature_dataset(test_dir, **kw)

    logger.info(
        f"  Feature shape → train: {X_train.shape}, test: {X_test.shape}"
    )

    # ── Train Models ──
    logger.info("\n[Step 3/3] Training classical models ...")
    model_zoo    = get_model_zoo(cfg)
    all_metrics: Dict[str, Dict] = {}

    for name, pipeline in tqdm(model_zoo.items(), desc="Models", ncols=80):
        metrics = train_and_evaluate(
            name, pipeline, X_train, y_train, X_test, y_test, cfg
        )
        all_metrics[name] = metrics

        # Save model
        safe_name  = name.replace(" ", "_").lower()
        model_path = cfg.models_dir / f"{safe_name}.pkl"
        save_sklearn_model(pipeline, model_path)
        logger.info(f"  Model saved → {model_path.name}")

        # Save per-model metrics
        save_metrics(
            metrics,
            cfg.results_dir / "metrics" / f"{safe_name}_metrics.json",
        )

    # ── Comparison Plot ──
    comparison = {
        name: {k: v for k, v in m.items() if k in ("accuracy", "f1", "auc")}
        for name, m in all_metrics.items()
    }
    plot_model_comparison(
        comparison,
        save_path=cfg.results_dir / "plots" / "model_comparison.png",
    )

    # ── Summary Table ──
    logger.info("\n" + "="*65)
    logger.info("  Classical ML — Final Results Summary")
    logger.info("="*65)
    header = f"  {'Model':<22} {'Acc':>7} {'F1':>7} {'AUC':>7}  CV Acc"
    logger.info(header)
    logger.info("  " + "-"*60)
    for name, m in all_metrics.items():
        logger.info(
            f"  {name:<22} {m['accuracy']*100:>6.2f}%  "
            f"{m['f1']:>6.4f}  {m['auc']:>6.4f}  "
            f"{m['cv_accuracy_mean']*100:.2f}%"
        )
    logger.info("="*65)

    save_metrics(all_metrics, cfg.results_dir / "metrics" / "all_classical_results.json")
    return all_metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train classical ML models for TruthLens.")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max images per class (0 = use all). Use e.g. 2000 for a quick test.",
    )
    args = parser.parse_args()

    cfg = Config()
    train_all(cfg, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
