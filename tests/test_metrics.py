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


def _proba(n_rows: int, n_cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n_rows, n_cols))
    return p / p.sum(axis=1, keepdims=True)


def test_missing_class_roc_auc_emits_no_warning_and_no_nan() -> None:
    """Regression: fitted class absent from a split must be skipped, not warned
    about by sklearn (UndefinedMetricWarning) or turned into NaN (macro)."""
    import warnings

    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y_pred = y_true.copy()
    proba = _proba(8, 4)  # 4 fitted classes; class 3 never appears
    for average in ("weighted", "macro"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            result = metrics.classification_metrics(
                y_true, y_pred, proba, average=average, classes=[0, 1, 2, 3]
            )
        assert result["roc_auc"] is not None
        assert np.isfinite(result["roc_auc"])


def test_missing_class_macro_auc_averages_computable_classes() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    proba = _proba(6, 4, seed=1)
    result = metrics.classification_metrics(
        y_true, y_true, proba, average="macro", classes=[0, 1, 2, 3]
    )
    from sklearn.metrics import roc_auc_score

    expected = np.mean(
        [roc_auc_score((y_true == c).astype(int), proba[:, c]) for c in (0, 1, 2)]
    )
    assert np.isclose(result["roc_auc"], expected)


def test_all_classes_present_matches_sklearn_weighted() -> None:
    """The skip-aware implementation must be a no-op when nothing is missing."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, size=60)
    proba = _proba(60, 3, seed=3)
    result = metrics.classification_metrics(
        y_true, y_true, proba, average="weighted", classes=[0, 1, 2]
    )
    expected = roc_auc_score(
        y_true, proba, multi_class="ovr", average="weighted", labels=[0, 1, 2]
    )
    assert np.isclose(result["roc_auc"], expected)


def test_single_class_y_true_yields_none_roc_auc() -> None:
    """One-vs-rest AUC is undefined with a single observed class."""
    y_true = np.array([1, 1, 1, 1])
    proba = _proba(4, 3, seed=4)
    result = metrics.classification_metrics(
        y_true, y_true, proba, average="weighted", classes=[0, 1, 2]
    )
    assert result["roc_auc"] is None
