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


def select_best_f1_threshold(y_true: Any, y_score: Any) -> dict[str, float]:
    """Select the exact score threshold that maximizes validation F1.

    Every distinct score is evaluated with tied scores entering together.  If
    several thresholds have the same F1, the highest threshold is retained so
    the selected operating point predicts fewer anomalies.
    """

    true = _to_numpy(y_true).astype(int).reshape(-1)
    score = _to_numpy(y_score).astype(float).reshape(-1)
    if true.size == 0 or true.size != score.size:
        raise ValueError("y_true and y_score must be non-empty and have equal length.")
    if not np.isin(true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_true = true[order]
    group_ends = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
    true_positives = np.cumsum(sorted_true)[group_ends].astype(float)
    predicted_positives = (group_ends + 1).astype(float)
    positives = float(sorted_true.sum())
    precision = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives),
        where=predicted_positives > 0,
    )
    recall = true_positives / positives if positives else np.zeros_like(true_positives)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    return {
        "threshold": float(sorted_score[group_ends[best]]),
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def compute_binary_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
    fixed_recall: float = 0.95,
) -> dict[str, Any]:
    """Compute threshold metrics plus AUROC/AUPRC when labels allow it."""

    true = _to_numpy(y_true).astype(int).reshape(-1)
    score = _to_numpy(y_score).astype(float).reshape(-1)
    pred = (score >= threshold).astype(int)

    if true.size == 0:
        raise ValueError("Cannot compute metrics on an empty input.")
    if not 0.0 <= fixed_recall <= 1.0:
        raise ValueError("fixed_recall must be between 0 and 1.")

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

    negatives = int((true == 0).sum())
    positives = int((true == 1).sum())
    fpr = float(cm[0][1] / negatives) if negatives else None
    fpr_at_fixed_recall = None
    if positives and negatives:
        # Evaluating each distinct score also handles tied predictions correctly.
        candidate_thresholds = np.concatenate(([np.inf], np.unique(score)[::-1], [-np.inf]))
        feasible_fprs = []
        for candidate in candidate_thresholds:
            candidate_pred = score >= candidate
            candidate_recall = float(((true == 1) & candidate_pred).sum() / positives)
            if candidate_recall >= fixed_recall:
                feasible_fprs.append(float(((true == 0) & candidate_pred).sum() / negatives))
        if feasible_fprs:
            fpr_at_fixed_recall = min(feasible_fprs)

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
        "fpr": fpr,
        "fixed_recall": fixed_recall,
        "fpr_at_fixed_recall": fpr_at_fixed_recall,
        "confusion_matrix": cm,
    }
