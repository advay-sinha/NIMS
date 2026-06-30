"""Tests for src.training.metrics."""

from __future__ import annotations

import numpy as np

from src.training import metrics


def test_binary_metrics_perfect_prediction() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    result = metrics.classification_metrics(y_true, y_pred)
    assert result["n_classes"] == 2
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == 1.0
    assert result["false_positive_rate"]["binary"] == 0.0


def test_roc_auc_with_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    result = metrics.classification_metrics(y_true, y_pred, proba)
    assert result["roc_auc"] == 1.0


def test_no_proba_yields_none_roc_auc() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = metrics.classification_metrics(y_true, y_pred, None)
    assert result["roc_auc"] is None


def test_multiclass_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    result = metrics.classification_metrics(y_true, y_pred, average="macro")
    assert result["n_classes"] == 3
    assert 0.0 <= result["f1"] <= 1.0
    assert len(result["confusion_matrix"]) == 3
    assert "macro" in result["false_positive_rate"]


def test_false_positive_rate_computation() -> None:
    # 2 classes; one false positive for class 1.
    cm = [[2, 1], [0, 2]]
    fpr = metrics.false_positive_rate(cm, [0, 1])
    # class 1: FP=1, TN=2 -> 1/3
    assert np.isclose(fpr["per_class"]["1"], 1 / 3)
    assert fpr["binary"] == fpr["per_class"]["1"]


def test_confusion_matrix_shape_matches_classes() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    result = metrics.classification_metrics(y_true, y_pred)
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (2, 2)


def test_full_classes_used_when_split_missing_a_class() -> None:
    # Model fitted on 3 classes; this split only contains classes 0 and 1.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    # 3-column probabilities (class 2 never appears in this split).
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]]
    )
    result = metrics.classification_metrics(
        y_true, y_pred, proba, average="macro", classes=[0, 1, 2]
    )
    # Confusion matrix spans the full 3-class set despite the missing class.
    assert result["n_classes"] == 3
    assert np.array(result["confusion_matrix"]).shape == (3, 3)


def test_classes_default_falls_back_to_observed() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    result = metrics.classification_metrics(y_true, y_pred, classes=None)
    assert result["labels"] == [0, 1]
