"""Evaluation for the explicit positive class, Diseased (1)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data_loader import LABELS


def score_predictions(
    labels: np.ndarray, predicted: np.ndarray, scores: np.ndarray
) -> dict:
    """Return metrics, per-class results and curves for a binary partition."""
    labels = np.asarray(labels)
    cm = confusion_matrix(labels, predicted, labels=list(LABELS))
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return {
        "positive_class": "Diseased",
        "positive_label": 1,
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "weighted_f1": f1_score(labels, predicted, average="weighted"),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "confusion_matrix": cm.tolist(),
        "class_distribution": {"Healthy": int(tn + fp), "Diseased": int(fn + tp)},
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "class_error_rate": {
            "Healthy": float(fp / (tn + fp)),
            "Diseased": float(fn / (fn + tp)),
        },
        "classification_report": classification_report(
            labels,
            predicted,
            labels=list(LABELS),
            target_names=list(LABELS.values()),
            output_dict=True,
            zero_division=0,
        ),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "precision_recall_curve": {
            "recall": recall.tolist(),
            "precision": precision.tolist(),
        },
    }
