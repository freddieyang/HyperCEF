"""Binary ranking and threshold metrics used by HyperCEF."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold: float = 0.5, top_k: int = 50) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    predictions = probabilities >= threshold
    order = np.argsort(-probabilities)[: min(top_k, len(labels))]
    top_labels = labels[order]
    positives = int(labels.sum())
    return {
        "AUC": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "AUPRC": float(average_precision_score(labels, probabilities)) if positives else float("nan"),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "PrecisionAt50": float(top_labels.mean()) if len(top_labels) else float("nan"),
        "RecallAt50": float(top_labels.sum() / positives) if positives else float("nan"),
    }
