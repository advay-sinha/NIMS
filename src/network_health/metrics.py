"""Network-health model metrics.

Purpose
-------
Score the anomaly baseline in both regimes: supervised metrics when labels
exist, anomaly-score statistics otherwise. Pure computation — no I/O.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def labeled_metrics(y_true: Any, y_pred: Any, scores: Any | None) -> dict[str, Any]:
    """Supervised anomaly metrics (labels: 0 = healthy, 1 = anomalous).

    Parameters
    ----------
    y_true, y_pred:
        True and predicted anomaly labels.
    scores:
        Continuous anomaly scores (higher = more anomalous), for ROC-AUC;
        omitted when unavailable or when only one class is present.

    Returns
    -------
    dict
        precision, recall, f1, roc_auc (nullable), confusion matrix and
        support counts.
    """
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", pos_label=1,
        zero_division=0,
    )
    roc_auc = None
    if scores is not None and len(np.unique(np.asarray(y_true))) > 1:
        roc_auc = float(roc_auc_score(y_true, scores))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "mode": "labeled",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(len(np.asarray(y_true))),
        "n_anomalous_true": int(np.asarray(y_true).sum()),
        "n_anomalous_predicted": int(np.asarray(y_pred).sum()),
    }


def unlabeled_metrics(
    scores: Any, y_pred: Any, threshold: float
) -> dict[str, Any]:
    """Anomaly-score statistics for unlabeled telemetry.

    Parameters
    ----------
    scores:
        Continuous anomaly scores (higher = more anomalous).
    y_pred:
        Thresholded anomaly flags.
    threshold:
        The score threshold applied.

    Returns
    -------
    dict
        Score distribution summary, anomaly rate and the threshold used.
    """
    import numpy as np

    values = np.asarray(scores, dtype=float)
    return {
        "mode": "unlabeled",
        "n_samples": int(len(values)),
        "anomaly_rate": float(np.asarray(y_pred).mean()) if len(values) else 0.0,
        "threshold": float(threshold),
        "score_distribution": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        },
    }
