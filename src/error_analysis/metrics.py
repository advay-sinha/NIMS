"""Error-analysis frame builders (pure computation, no I/O).

Purpose
-------
Turn ``(y_true, y_pred, probabilities)`` into the tabular artefacts of the
error-analysis subsystem: confusion matrix, per-class metrics, hardest-class
ranking and misclassified examples. Kept free of file I/O and configuration
so every function is directly unit-testable.

Inputs
------
Label vectors (any hashable labels), an optional probability matrix aligned
with ``labels``, and the complete class-label list (so classes absent from a
split still appear with zero support).

Outputs
-------
pandas DataFrames, one per artefact.

Limitations
-----------
No plots and no ROC/PR curves (later phase).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def confusion_frame(y_true: Any, y_pred: Any, labels: Sequence[Any]) -> "Any":
    """Confusion matrix with true labels as rows, predicted labels as columns.

    Parameters
    ----------
    y_true, y_pred:
        Label vectors.
    labels:
        Complete ordered class-label list.

    Returns
    -------
    pandas.DataFrame
        Indexed by true label (named ``true_label``); one column per
        predicted label.
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    frame = pd.DataFrame(matrix, index=list(labels), columns=list(labels))
    frame.index.name = "true_label"
    return frame


def class_metrics_frame(y_true: Any, y_pred: Any, labels: Sequence[Any]) -> "Any":
    """Per-class precision/recall/F1 with error counts.

    Parameters
    ----------
    y_true, y_pred:
        Label vectors.
    labels:
        Complete ordered class-label list.

    Returns
    -------
    pandas.DataFrame
        Columns ``class_label``, ``support``, ``precision``, ``recall``,
        ``f1_score``, ``false_positives``, ``false_negatives`` — one row per
        class, in ``labels`` order.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    confusion = confusion_frame(y_true, y_pred, labels).to_numpy()
    true_positives = np.diag(confusion)
    false_positives = confusion.sum(axis=0) - true_positives
    false_negatives = confusion.sum(axis=1) - true_positives

    return pd.DataFrame(
        {
            "class_label": list(labels),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positives": false_positives.astype(int),
            "false_negatives": false_negatives.astype(int),
        }
    )


def hardest_classes_frame(class_metrics: "Any") -> "Any":
    """Rank observed classes from hardest (lowest F1) to easiest.

    Classes with zero support are excluded: they were never seen in the
    split, so an F1 of 0 would rank them "hardest" without evidence.

    Parameters
    ----------
    class_metrics:
        Output of :func:`class_metrics_frame`.

    Returns
    -------
    pandas.DataFrame
        Same columns plus ``rank`` (1 = hardest), sorted ascending by F1
        (support descending, then label, break ties deterministically).
    """
    frame = class_metrics[class_metrics["support"] > 0].copy()
    frame["class_label"] = frame["class_label"].astype(str)
    frame = frame.sort_values(
        ["f1_score", "support", "class_label"], ascending=[True, False, True]
    ).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame


def misclassified_frame(
    y_true: Any,
    y_pred: Any,
    confidence: Any | None,
    *,
    max_examples: int,
    x: Any | None = None,
    include_feature_values: bool = False,
) -> "Any":
    """Misclassified rows, most confident errors first.

    Parameters
    ----------
    y_true, y_pred:
        Label vectors (positionally aligned).
    confidence:
        Predicted-class probability per row, or ``None`` when the model
        exposes no probabilities (the column is then empty, never a failure).
    max_examples:
        Upper bound on returned rows. When confidences exist, the highest-
        confidence errors are kept (the most damaging mistakes); otherwise
        rows keep split order.
    x:
        Optional feature matrix (positionally aligned with the labels).
    include_feature_values:
        When true (and ``x`` is given), append the feature columns.

    Returns
    -------
    pandas.DataFrame
        Columns ``row_index``, ``true_label``, ``predicted_label``,
        ``confidence`` (+ feature columns when configured).
    """
    import numpy as np
    import pandas as pd

    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(true_arr)),
            "true_label": true_arr,
            "predicted_label": pred_arr,
            "confidence": (
                np.asarray(confidence) if confidence is not None else pd.NA
            ),
        }
    )
    errors = frame[frame["true_label"] != frame["predicted_label"]]
    if confidence is not None:
        errors = errors.sort_values(
            ["confidence", "row_index"], ascending=[False, True]
        )
    errors = errors.head(int(max_examples)).reset_index(drop=True)

    if include_feature_values and x is not None:
        features = x.iloc[errors["row_index"].to_numpy()].reset_index(drop=True)
        errors = pd.concat([errors, features], axis=1)
    return errors


def binary_error_frames(
    misclassified: "Any", labels: Sequence[Any]
) -> tuple["Any", "Any"] | None:
    """Split binary misclassifications into false positives/negatives.

    The positive class is the second (highest) label — the ``attack`` class
    in the normal(0)/attack(1) encoding used by the binary datasets.

    Parameters
    ----------
    misclassified:
        Output of :func:`misclassified_frame`.
    labels:
        Complete class-label list; must have exactly two entries.

    Returns
    -------
    tuple | None
        ``(false_positive_frame, false_negative_frame)`` for binary tasks,
        ``None`` otherwise.
    """
    if len(labels) != 2:
        return None
    positive = labels[-1]
    false_positives = misclassified[
        misclassified["predicted_label"] == positive
    ].reset_index(drop=True)
    false_negatives = misclassified[
        misclassified["true_label"] == positive
    ].reset_index(drop=True)
    return false_positives, false_negatives
