#!/usr/bin/env python3
"""
TruthLens — src/evaluate.py
=============================
Comprehensive evaluation of all trained models (CNN + Classical ML).

Outputs per model:
  • Classification report (precision, recall, F1 per class)
  • Confusion matrix (saved as PNG)
  • ROC curve (saved as PNG)
  • JSON metrics file

Combined output:
  • Model comparison bar chart
  • Summary leaderboard table

Usage:
    python src/evaluate.py --model all
    python src/evaluate.py --model resnet18
    python src/evaluate.py --model svm
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocessing import CIFAKEDataset, build_feature_dataset, get_transforms
from src.train import build_model
from src.utils import (
    Config,
    get_device,
    get_logger,
    load_sklearn_model,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    print_classification_report,
    save_metrics,
)

logger = get_logger(__name__, log_file=Path("results/logs/evaluate.log"))


# ─── CNN Evaluation ───────────────────────────────────────────────────────────
def evaluate_cnn(cfg: Config) -> Dict[str, Any]:
    """Load the best ResNet18 checkpoint and evaluate on the test set."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    ckpt_path = cfg.models_dir / cfg.cnn_model_name
    if not ckpt_path.exists():
        logger.error(f"CNN checkpoint not found: {ckpt_path}")
        logger.error("Run  `python src/train.py`  first.")
        return {}

    device = get_device()
    model  = build_model(cfg.num_classes).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    logger.info(
        f"  Loaded CNN checkpoint (epoch {checkpoint.get('epoch','?')}, "
        f"val_acc={checkpoint.get('val_acc', 0):.2f}%)"
    )

    test_dir = cfg.data_dir / "test"
    if not test_dir.exists():
        logger.error("Test directory not found.")
        return {}

    dataset = CIFAKEDataset(test_dir, split="test", img_size=cfg.img_size)
    loader  = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="CNN Inference", ncols=80):
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]   # prob of FAKE class
            preds  = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred  = np.concatenate(all_preds)
    y_true  = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    report = print_classification_report(y_true, y_pred, cfg.class_names, "ResNet18")

    plot_confusion_matrix(
        y_true, y_pred, cfg.class_names,
        title="ResNet18 — Confusion Matrix",
        save_path=cfg.results_dir / "plots" / "cm_resnet18.png",
    )
    auc = plot_roc_curve(
        y_true, y_score,
        title="ResNet18 — ROC Curve",
        save_path=cfg.results_dir / "plots" / "roc_resnet18.png",
    )

    metrics = {
        "model":     "ResNet18",
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "auc":       round(auc,  4),
        "classification_report": report,
    }
    save_metrics(metrics, cfg.results_dir / "metrics" / "resnet18_metrics.json")
    logger.info(f"  ResNet18 → Acc: {acc*100:.2f}%  F1: {f1:.4f}  AUC: {auc:.4f}")
    return metrics


# ─── Classical Model Evaluation ───────────────────────────────────────────────
def evaluate_classical_model(
    name:     str,
    model_path: Path,
    cfg: Config,
) -> Dict[str, Any]:
    """Load and evaluate a single saved sklearn model."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if not model_path.exists():
        logger.warning(f"  Model not found: {model_path} — skipping.")
        return {}

    pipeline = load_sklearn_model(model_path)
    test_dir = cfg.data_dir / "test"
    if not test_dir.exists():
        logger.error("Test directory not found.")
        return {}

    logger.info(f"\n  Evaluating: {name}")
    X_test, y_test = build_feature_dataset(test_dir)

    y_pred  = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    report = print_classification_report(y_test, y_pred, cfg.class_names, name)
    safe   = name.replace(" ", "_").lower()

    plot_confusion_matrix(
        y_test, y_pred, cfg.class_names,
        title=f"{name} — Confusion Matrix",
        save_path=cfg.results_dir / "plots" / f"cm_{safe}.png",
    )
    auc = plot_roc_curve(
        y_test, y_score,
        title=f"{name} — ROC Curve",
        save_path=cfg.results_dir / "plots" / f"roc_{safe}.png",
    )

    metrics = {
        "model":     name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "auc":       round(auc,  4),
        "classification_report": report,
    }
    save_metrics(metrics, cfg.results_dir / "metrics" / f"{safe}_metrics.json")
    return metrics


# ─── Evaluate All ─────────────────────────────────────────────────────────────
def evaluate_all(cfg: Config) -> Dict[str, Dict]:
    """Evaluate every trained model and produce a comparison report."""
    all_results: Dict[str, Dict] = {}

    # CNN
    logger.info("\n[1/7] Evaluating ResNet18 (CNN) ...")
    cnn_metrics = evaluate_cnn(cfg)
    if cnn_metrics:
        all_results["ResNet18"] = cnn_metrics

    # Classical models
    classical_models = {
        "Logistic Regression": cfg.models_dir / "logistic_regression.pkl",
        "Decision Tree":       cfg.models_dir / "decision_tree.pkl",
        "Random Forest":       cfg.models_dir / "random_forest.pkl",
        "SVM":                 cfg.models_dir / "svm.pkl",
        "k-NN":                cfg.models_dir / "k-nn.pkl",
        "Naive Bayes":         cfg.models_dir / "naive_bayes.pkl",
    }

    for i, (name, path) in enumerate(classical_models.items(), start=2):
        logger.info(f"\n[{i}/7] Evaluating {name} ...")
        m = evaluate_classical_model(name, path, cfg)
        if m:
            all_results[name] = m

    if not all_results:
        logger.warning("No model results found. Train models first.")
        return {}

    # Comparison plot
    comparison = {
        n: {k: v for k, v in m.items() if k in ("accuracy", "f1", "auc")}
        for n, m in all_results.items()
    }
    plot_model_comparison(
        comparison,
        save_path=cfg.results_dir / "plots" / "full_model_comparison.png",
    )

    # Summary leaderboard
    logger.info("\n" + "="*70)
    logger.info("  TruthLens — Final Leaderboard")
    logger.info("="*70)
    logger.info(f"  {'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    logger.info("  " + "-"*60)

    ranked = sorted(all_results.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
    for rank, (name, m) in enumerate(ranked, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"  {rank}."
        logger.info(
            f"  {medal} {name:<23} {m['accuracy']*100:>7.2f}%  "
            f"{m['f1']:>7.4f}  {m['auc']:>7.4f}"
        )

    save_metrics(all_results, cfg.results_dir / "metrics" / "full_evaluation.json")
    return all_results


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate TruthLens models.")
    parser.add_argument(
        "--model",
        choices=["all", "resnet18", "svm", "rf", "lr", "dt", "knn", "nb"],
        default="all",
        help="Which model to evaluate.",
    )
    parser.add_argument("--data-dir", type=Path, default=Config().data_dir)
    parser.add_argument("--model-name", default=Config().cnn_model_name)
    args = parser.parse_args()
    cfg  = Config(data_dir=args.data_dir, cnn_model_name=args.model_name)

    if args.model == "all":
        evaluate_all(cfg)
    elif args.model == "resnet18":
        evaluate_cnn(cfg)
    else:
        name_map = {
            "svm": ("SVM",                 cfg.models_dir / "svm.pkl"),
            "rf":  ("Random Forest",       cfg.models_dir / "random_forest.pkl"),
            "lr":  ("Logistic Regression", cfg.models_dir / "logistic_regression.pkl"),
            "dt":  ("Decision Tree",       cfg.models_dir / "decision_tree.pkl"),
            "knn": ("k-NN",                cfg.models_dir / "k-nn.pkl"),
            "nb":  ("Naive Bayes",         cfg.models_dir / "naive_bayes.pkl"),
        }
        name, path = name_map[args.model]
        evaluate_classical_model(name, path, cfg)


if __name__ == "__main__":
    main()
