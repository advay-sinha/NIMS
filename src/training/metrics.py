"""Classification metrics for Engine A.

Purpose
-------
Compute the metric suite the project mandates (CLAUDE.md > Machine Learning
Standards: "Never report accuracy alone") for both binary and multiclass
intrusion detection: precision, recall, F1, ROC-AUC, false-positive rate and the
confusion matrix. All values are JSON-serialisable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def false_positive_rate(confusion: "Any", labels: list[int]) -> dict[str, Any]:
    """Compute per-class and averaged false-positive rate from a confusion matrix.

    For class ``i``: ``FP = column_i_sum - TP_i`` and
    ``TN = total - row_i_sum - column_i_sum + TP_i``; ``FPR = FP / (FP + TN)``.

    Parameters
    ----------
    confusion:
        Confusion matrix (rows = true, columns = predicted), ordered by
        ``labels``.
    labels:
        Class labels in matrix order.

    Returns
    -------
    dict
        ``{"per_class": {label: fpr}, "macro": float, "binary": float | None}``.
    """
    import numpy as np

    cm = np.asarray(confusion, dtype=float)
    total = cm.sum()
    per_class: dict[str, float] = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        per_class[str(label)] = float(fp / denom) if denom > 0 else 0.0

    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    # For binary problems the positive class (highest label) FPR is the headline.
    binary = per_class[str(labels[-1])] if len(labels) == 2 else None
    return {"per_class": per_class, "macro": macro, "binary": binary}


def _roc_auc(
    y_true: "Any",
    y_proba: "Any | None",
    labels: list[int],
    average: str,
) -> float | None:
    """Compute ROC-AUC for binary or multiclass, returning ``None`` on failure.

    ``labels`` must be the COMPLETE class list the model was fitted on so that
    the columns of ``y_proba`` align with the class set (otherwise multiclass
    ROC-AUC fails whenever a split is missing a class).
    """
    if y_proba is None:
        return None
    from sklearn.metrics import roc_auc_score

    try:
        if len(labels) == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1], labels=labels))
        return float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average=average, labels=labels
            )
        )
    except (ValueError, IndexError) as exc:
        logger.warning("ROC-AUC could not be computed: %s", exc)
        return None


def classification_metrics(
    y_true: "Any",
    y_pred: "Any",
    y_proba: "Any | None" = None,
    average: str = "weighted",
    classes: "Any | None" = None,
) -> dict[str, Any]:
    """Compute the full classification metric suite.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    y_proba:
        Predicted class probabilities (for ROC-AUC), or ``None``.
    average:
        Averaging strategy for multiclass precision/recall/F1.
    classes:
        The complete label set the model was fitted on (e.g. the fitted
        ``LabelEncoder`` / classifier ``classes_``). When provided, the
        confusion matrix and ROC-AUC are computed over this full label set
        rather than only the labels present in this split. Falls back to the
        observed union when ``None``.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1, ROC-AUC, FPR, confusion matrix, labels
        and class count.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if classes is not None:
        labels = [int(v) for v in classes]
    else:
        labels = sorted(
            int(v)
            for v in np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics: dict[str, Any] = {
        "n_classes": len(labels),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "roc_auc": _roc_auc(y_true, y_proba, labels, average),
        "false_positive_rate": false_positive_rate(cm, labels),
        "confusion_matrix": cm.tolist(),
        "average": average,
    }
    return metrics
