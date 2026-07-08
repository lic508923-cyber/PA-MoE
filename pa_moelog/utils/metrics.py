"""Binary anomaly detection metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover - fallback only used when sklearn is absent.
    average_precision_score = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def compute_binary_metrics(y_true: Any, y_score: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Compute threshold metrics plus AUROC/AUPRC when labels allow it."""

    true = _to_numpy(y_true).astype(int).reshape(-1)
    score = _to_numpy(y_score).astype(float).reshape(-1)
    pred = (score >= threshold).astype(int)

    if true.size == 0:
        raise ValueError("Cannot compute metrics on an empty input.")

    if precision_score is not None:
        precision = float(precision_score(true, pred, zero_division=0))
        recall = float(recall_score(true, pred, zero_division=0))
        f1 = float(f1_score(true, pred, zero_division=0))
        cm = confusion_matrix(true, pred, labels=[0, 1]).astype(int).tolist()
    else:
        tp = int(((true == 1) & (pred == 1)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        fn = int(((true == 1) & (pred == 0)).sum())
        tn = int(((true == 0) & (pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        cm = [[tn, fp], [fn, tp]]

    auroc = None
    auprc = None
    if np.unique(true).size > 1:
        if roc_auc_score is not None:
            auroc = float(roc_auc_score(true, score))
        if average_precision_score is not None:
            auprc = float(average_precision_score(true, score))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": cm,
    }
