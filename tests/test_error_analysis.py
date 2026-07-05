"""Tests for src.error_analysis (metrics, analyzer, artefacts, reporting)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.error_analysis.analyzer import (
    analyze_model,
    analyze_predictions,
    maybe_analyze_after_training,
)
from src.error_analysis.metrics import (
    class_metrics_frame,
    confusion_frame,
    hardest_classes_frame,
    misclassified_frame,
)
from src.error_analysis.reporting import error_analysis_summary

# A small, fully hand-checkable multiclass case:
# class 0: 3 samples, all correct.
# class 1: 3 samples, 2 correct, 1 predicted as 2.
# class 2: 2 samples, 1 correct, 1 predicted as 0.
_TRUE = np.array([0, 0, 0, 1, 1, 1, 2, 2])
_PRED = np.array([0, 0, 0, 1, 1, 2, 2, 0])
_LABELS = [0, 1, 2]


def _proba(pred: np.ndarray, confidence: float = 0.9) -> np.ndarray:
    proba = np.full((len(pred), 3), (1 - confidence) / 2)
    proba[np.arange(len(pred)), pred] = confidence
    return proba


def test_confusion_matrix_true_rows_predicted_columns() -> None:
    frame = confusion_frame(_TRUE, _PRED, _LABELS)
    assert frame.index.name == "true_label"
    assert list(frame.index) == _LABELS and list(frame.columns) == _LABELS
    assert frame.loc[0, 0] == 3          # all class-0 correct
    assert frame.loc[1, 2] == 1          # one 1 -> 2 error
    assert frame.loc[2, 0] == 1          # one 2 -> 0 error
    assert frame.to_numpy().sum() == len(_TRUE)


def test_class_metrics_correctness() -> None:
    frame = class_metrics_frame(_TRUE, _PRED, _LABELS).set_index("class_label")
    assert list(frame.columns) == [
        "support", "precision", "recall", "f1_score",
        "false_positives", "false_negatives",
    ]
    # Class 0: 3/3 recalled, but one class-2 sample predicted as 0.
    assert frame.loc[0, "support"] == 3
    assert frame.loc[0, "recall"] == pytest.approx(1.0)
    assert frame.loc[0, "precision"] == pytest.approx(3 / 4)
    assert frame.loc[0, "false_positives"] == 1
    assert frame.loc[0, "false_negatives"] == 0
    # Class 1: 2/3 recalled, no false positives.
    assert frame.loc[1, "recall"] == pytest.approx(2 / 3)
    assert frame.loc[1, "precision"] == pytest.approx(1.0)
    assert frame.loc[1, "false_negatives"] == 1
    # Class 2: 1/2 recalled, one false positive (the 1 -> 2 error).
    assert frame.loc[2, "false_positives"] == 1
    assert frame.loc[2, "false_negatives"] == 1


def test_hardest_classes_sorted_by_lowest_f1() -> None:
    hardest = hardest_classes_frame(class_metrics_frame(_TRUE, _PRED, _LABELS))
    assert list(hardest["rank"]) == [1, 2, 3]
    assert hardest["f1_score"].is_monotonic_increasing
    # Class 2 (F1 = 2/3 * ... lowest) must rank hardest.
    assert hardest.iloc[0]["class_label"] == "2"


def test_hardest_classes_exclude_zero_support() -> None:
    metrics = class_metrics_frame(_TRUE, _PRED, [0, 1, 2, 99])
    hardest = hardest_classes_frame(metrics)
    assert "99" not in set(hardest["class_label"])
    assert len(hardest) == 3


def test_misclassified_examples_creation() -> None:
    frame = misclassified_frame(
        _TRUE, _PRED, _proba(_PRED).max(axis=1), max_examples=1000
    )
    assert list(frame.columns) == [
        "row_index", "true_label", "predicted_label", "confidence",
    ]
    assert len(frame) == 2
    assert set(frame["row_index"]) == {5, 7}
    assert (frame["true_label"] != frame["predicted_label"]).all()


def test_misclassified_cap_keeps_highest_confidence() -> None:
    confidence = np.linspace(0.1, 0.9, len(_PRED))
    frame = misclassified_frame(_TRUE, _PRED, confidence, max_examples=1)
    assert len(frame) == 1
    assert frame.iloc[0]["row_index"] == 7  # higher confidence of the 2 errors


def test_missing_probabilities_write_empty_confidence() -> None:
    result = analyze_predictions(_TRUE, _PRED, None, _LABELS)
    assert result.misclassified["confidence"].isna().all()
    assert result.summary["n_misclassified"] == 2


def test_no_misclassifications() -> None:
    result = analyze_predictions(_TRUE, _TRUE, _proba(_TRUE), _LABELS)
    assert result.summary["n_misclassified"] == 0
    assert result.summary["accuracy"] == pytest.approx(1.0)
    assert len(result.misclassified) == 0
    assert result.hardest_classes["f1_score"].min() == pytest.approx(1.0)


def test_binary_classification_fp_fn_frames() -> None:
    true = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])  # one FP (row 1), one FN (row 3)
    result = analyze_predictions(true, pred, None, [0, 1])
    assert result.false_positives is not None
    assert list(result.false_positives["row_index"]) == [1]
    assert list(result.false_negatives["row_index"]) == [3]


def test_multiclass_has_no_fp_fn_frames() -> None:
    result = analyze_predictions(_TRUE, _PRED, None, _LABELS)
    assert result.false_positives is None
    assert result.false_negatives is None


def test_summary_statistics() -> None:
    from sklearn.metrics import f1_score

    result = analyze_predictions(_TRUE, _PRED, _proba(_PRED), _LABELS)
    assert result.summary["n_samples"] == 8
    assert result.summary["n_classes"] == 3
    assert result.summary["accuracy"] == pytest.approx(6 / 8)
    assert result.summary["macro_f1"] == pytest.approx(
        f1_score(_TRUE, _PRED, average="macro")
    )
    assert result.summary["weighted_f1"] == pytest.approx(
        f1_score(_TRUE, _PRED, average="weighted")
    )


# ------------------------------------------------- persistence + end-to-end


@pytest.fixture()
def fitted_model_xy() -> tuple:
    from src.models.registry import build_model

    rng = np.random.default_rng(11)
    n = 150
    y = pd.Series(np.arange(n) % 3, name="label")
    x = pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.4, n),
            "f2": rng.normal(size=n),
        }
    )
    model = build_model(
        "xgboost", {"gpu": False, "params": {"n_estimators": 10}}, use_gpu=False,
        seed=42,
    )
    model.fit(x, y)
    return model, x, y


def _config(**error_analysis: object) -> dict:
    return {"error_analysis": {"enabled": True, "split": "test", **error_analysis}}


def test_artifact_persistence(fitted_model_xy, tmp_path: Path) -> None:
    model, x, y = fitted_model_xy
    paths, result = analyze_model(
        model, x, y,
        experiment_id="demo_xgboost_20260101T000000", dataset_id="demo",
        config=_config(max_misclassified_examples=50),
        output_root=tmp_path,
    )
    out_dir = tmp_path / "demo_xgboost_20260101T000000"
    for name in (
        "metadata", "confusion_matrix", "class_metrics", "hardest_classes",
        "misclassified_examples",
    ):
        assert paths[name].is_file() and paths[name].parent == out_dir

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in (
        "experiment_id", "dataset", "model_type", "timestamp", "n_samples",
        "n_classes", "n_misclassified", "accuracy", "macro_f1", "weighted_f1",
    ):
        assert key in metadata
    assert metadata["model_type"] == "xgboost"
    assert metadata["n_samples"] == len(x)
    assert metadata["n_misclassified"] == result.summary["n_misclassified"]

    confusion = pd.read_csv(out_dir / "confusion_matrix.csv", index_col=0)
    assert confusion.to_numpy().sum() == len(x)
    assert len(pd.read_csv(out_dir / "class_metrics.csv")) == 3
    assert len(pd.read_csv(out_dir / "misclassified_examples.csv")) <= 50


def test_feature_values_included_when_configured(fitted_model_xy, tmp_path: Path) -> None:
    model, x, y = fitted_model_xy
    paths, _ = analyze_model(
        model, x, y,
        experiment_id="withfeat", dataset_id="demo",
        config=_config(include_feature_values=True),
        output_root=tmp_path,
    )
    frame = pd.read_csv(paths["misclassified_examples"])
    assert {"f1", "f2"}.issubset(frame.columns)


def test_reporting_summary_contains_headline(fitted_model_xy, tmp_path: Path) -> None:
    model, x, y = fitted_model_xy
    _, result = analyze_model(
        model, x, y, experiment_id="rep", dataset_id="demo",
        config=_config(), output_root=tmp_path,
    )
    summary = error_analysis_summary(result, tmp_path / "rep")
    assert "## Error Analysis" in summary
    assert "Accuracy" in summary and "Macro F1" in summary
    assert "hardest classes" in summary
    assert str(tmp_path / "rep") in summary


# ------------------------------------------------------------- trainer hook


class _Result:
    experiment_id = "demo_xgboost_20260101T000000"
    dataset_id = "demo"
    model_name = "xgboost"


class _Paths:
    def __init__(self, root: Path) -> None:
        self.error_analysis_dir = root


def test_hook_disabled_is_noop(fitted_model_xy, tmp_path: Path) -> None:
    model, x, y = fitted_model_xy
    out = maybe_analyze_after_training(
        _Result(), model, {"test": (x, y)},
        {"error_analysis": {"enabled": False}}, _Paths(tmp_path),
    )
    assert out is None
    assert not any(tmp_path.iterdir())


def test_hook_generates_artifacts(fitted_model_xy, tmp_path: Path) -> None:
    model, x, y = fitted_model_xy
    out = maybe_analyze_after_training(
        _Result(), model, {"test": (x, y)}, _config(), _Paths(tmp_path)
    )
    assert out is not None
    assert (tmp_path / _Result.experiment_id / "confusion_matrix.csv").is_file()


def test_hook_skips_missing_split(fitted_model_xy, tmp_path: Path) -> None:
    model, x, y = fitted_model_xy
    out = maybe_analyze_after_training(
        _Result(), model, {"train": (x, y)}, _config(), _Paths(tmp_path)
    )
    assert out is None


def test_hook_never_raises(fitted_model_xy, tmp_path: Path) -> None:
    """A crashing model must be logged, not propagated (run already saved)."""

    class _Broken:
        name = "xgboost"
        is_supervised = True

        def predict(self, x):  # noqa: ANN001, ANN201
            raise RuntimeError("boom")

    _, x, y = fitted_model_xy
    out = maybe_analyze_after_training(
        _Result(), _Broken(), {"test": (x, y)}, _config(), _Paths(tmp_path)
    )
    assert out is None
