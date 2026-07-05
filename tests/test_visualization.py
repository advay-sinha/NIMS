"""Tests for src.visualization (plots, runner, artefacts, reporting)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_hardest_classes,
    plot_misclassification_pairs,
)
from src.visualization.reporting import visualization_summary
from src.visualization.runner import generate_visualizations

_EXPERIMENT = "demo_xgboost_20260101T000000"


def _confusion(n_classes: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, 50, size=(n_classes, n_classes))
    frame = pd.DataFrame(
        matrix, index=range(n_classes), columns=[str(c) for c in range(n_classes)]
    )
    frame.index.name = "true_label"
    return frame


def _hardest(n_classes: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_label": [str(c) for c in range(n_classes)],
            "support": np.arange(1, n_classes + 1) * 10,
            "precision": np.linspace(0.5, 0.99, n_classes),
            "recall": np.linspace(0.5, 0.99, n_classes),
            "f1_score": np.linspace(0.5, 0.99, n_classes),
            "false_positives": np.ones(n_classes, dtype=int),
            "false_negatives": np.ones(n_classes, dtype=int),
            "rank": np.arange(1, n_classes + 1),
        }
    )


def _importance(n_features: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [f"f{i}" for i in range(n_features)],
            "mean_abs_shap": np.linspace(2.0, 0.1, n_features),
            "std_abs_shap": np.full(n_features, 0.1),
            "percentage_contribution": np.full(n_features, 100 / n_features),
            "cumulative_percentage": np.linspace(0, 100, n_features),
            "rank": np.arange(1, n_features + 1),
        }
    )


def _misclassified(n_rows: int, n_classes: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    true = rng.integers(0, n_classes, size=n_rows)
    pred = (true + rng.integers(1, n_classes, size=n_rows)) % n_classes
    return pd.DataFrame(
        {
            "row_index": np.arange(n_rows),
            "true_label": true,
            "predicted_label": pred,
            "confidence": rng.random(n_rows),
        }
    )


def _sources(
    root: Path,
    *,
    n_classes: int = 3,
    n_misclassified: int = 40,
    with_explainability: bool = True,
) -> tuple[Path, Path]:
    error_dir = root / "error_analysis" / _EXPERIMENT
    error_dir.mkdir(parents=True)
    _confusion(n_classes).to_csv(error_dir / "confusion_matrix.csv")
    _hardest(n_classes).to_csv(error_dir / "hardest_classes.csv", index=False)
    _misclassified(n_misclassified, n_classes).to_csv(
        error_dir / "misclassified_examples.csv", index=False
    )
    explain_root = root / "explainability"
    if with_explainability:
        explain_dir = explain_root / _EXPERIMENT
        explain_dir.mkdir(parents=True)
        _importance(25).to_csv(
            explain_dir / "global_feature_importance.csv", index=False
        )
    return root / "error_analysis", explain_root


def _run(root: Path, **source_kwargs: object) -> dict:
    error_root, explain_root = _sources(root, **source_kwargs)
    return generate_visualizations(
        _EXPERIMENT,
        dataset_id="demo",
        model_type="xgboost",
        error_analysis_dir=error_root,
        explainability_dir=explain_root,
        output_root=root / "visualizations",
        config={"visualization": {"dpi": 72}},
    )


# ----------------------------------------------------------- plot renderers


def test_confusion_matrix_plot_binary(tmp_path: Path) -> None:
    path = tmp_path / "cm.png"
    plot_confusion_matrix(_confusion(2), path, dpi=72)
    assert path.is_file() and path.stat().st_size > 0


def test_confusion_matrix_plot_multiclass_many_labels(tmp_path: Path) -> None:
    path = tmp_path / "cm40.png"
    plot_confusion_matrix(_confusion(40), path, dpi=72)  # annotations disabled
    assert path.is_file() and path.stat().st_size > 0


def test_confusion_matrix_normalized(tmp_path: Path) -> None:
    frame = _confusion(3)
    frame.iloc[2] = 0  # zero-support row must not divide by zero
    path = tmp_path / "cmn.png"
    plot_confusion_matrix(frame, path, normalized=True, dpi=72)
    assert path.is_file() and path.stat().st_size > 0


def test_feature_importance_plot_caps_at_top_n(tmp_path: Path) -> None:
    path = tmp_path / "fi.png"
    plot_feature_importance(_importance(25), path, top_n=20, dpi=72)
    assert path.is_file() and path.stat().st_size > 0


def test_feature_importance_plot_fewer_than_top_n(tmp_path: Path) -> None:
    path = tmp_path / "fi3.png"
    plot_feature_importance(_importance(3), path, top_n=20, dpi=72)
    assert path.is_file() and path.stat().st_size > 0


def test_hardest_classes_plot_binary(tmp_path: Path) -> None:
    path = tmp_path / "hc.png"
    plot_hardest_classes(_hardest(2), path, top_n=10, dpi=72)
    assert path.is_file() and path.stat().st_size > 0


def test_misclassification_pairs_plot(tmp_path: Path) -> None:
    path = tmp_path / "mp.png"
    plot_misclassification_pairs(_misclassified(60), path, top_n=20, dpi=72)
    assert path.is_file() and path.stat().st_size > 0


def test_misclassification_pairs_rejects_empty() -> None:
    with pytest.raises(ValueError, match="No misclassified examples"):
        plot_misclassification_pairs(
            _misclassified(0), Path("unused.png"), top_n=20, dpi=72
        )


# ------------------------------------------------------------------- runner


def test_runner_generates_all_plots_multiclass(tmp_path: Path) -> None:
    result = _run(tmp_path, n_classes=5)
    # Multiclass: normalized view is primary; counts kept as companion.
    assert set(result["plots"]) == {
        "confusion_matrix", "confusion_matrix_counts",
        "feature_importance_top20", "hardest_classes_top10",
        "misclassification_pairs_top20",
    }
    assert result["skipped"] == {}
    for path in result["plots"].values():
        assert path.is_file() and path.stat().st_size > 0


def test_runner_binary_classification(tmp_path: Path) -> None:
    result = _run(tmp_path, n_classes=2, n_misclassified=4)
    # Binary keeps counts primary + normalized companion.
    assert "confusion_matrix" in result["plots"]
    assert "confusion_matrix_normalized" in result["plots"]
    assert "misclassification_pairs_top20" in result["plots"]


def test_confusion_plot_decodes_class_names(tmp_path: Path) -> None:
    from src.visualization.plots import _decode_labels

    assert _decode_labels([0, 1], ["normal", "attack"]) == ["normal", "attack"]
    # Incomplete/mismatched mappings keep the raw ids instead of failing.
    assert _decode_labels([0, 5], ["normal", "attack"]) == ["0", "5"]
    assert _decode_labels(["a", "b"], ["normal", "attack"]) == ["a", "b"]
    path = tmp_path / "named.png"
    plot_confusion_matrix(_confusion(2), path, dpi=72,
                          class_names=["normal", "attack"])
    assert path.is_file() and path.stat().st_size > 0


def test_load_label_classes_from_encoding_report(tmp_path: Path) -> None:
    from src.visualization.artifacts import load_label_classes

    report_dir = tmp_path / "demo"
    report_dir.mkdir()
    (report_dir / "encoding_report.json").write_text(
        json.dumps({"label_classes": ["normal", "attack"]}), encoding="utf-8"
    )
    assert load_label_classes(tmp_path, "demo") == ["normal", "attack"]
    assert load_label_classes(tmp_path, "missing") is None


def test_runner_skips_missing_explainability(tmp_path: Path) -> None:
    result = _run(tmp_path, with_explainability=False)
    assert "feature_importance_top20" not in result["plots"]
    assert "source artefact missing" in result["skipped"]["feature_importance_top20"]
    assert "confusion_matrix" in result["plots"]  # others unaffected


def test_runner_no_misclassifications_skips_pairs_plot(tmp_path: Path) -> None:
    result = _run(tmp_path, n_misclassified=0)
    assert "misclassification_pairs_top20" not in result["plots"]
    assert result["skipped"]["misclassification_pairs_top20"] == (
        "no misclassified examples"
    )
    metadata = json.loads(result["metadata"].read_text(encoding="utf-8"))
    assert "misclassification_pairs_top20" in metadata["plots_skipped"]


def test_metadata_persistence(tmp_path: Path) -> None:
    result = _run(tmp_path)
    metadata = json.loads(result["metadata"].read_text(encoding="utf-8"))
    for key in (
        "experiment_id", "dataset", "model_type", "timestamp",
        "plots_generated", "plots_skipped", "source_artifacts_used",
    ):
        assert key in metadata
    assert metadata["experiment_id"] == _EXPERIMENT
    assert metadata["dataset"] == "demo"
    assert len(metadata["plots_generated"]) == 5
    assert len(metadata["source_artifacts_used"]) == 4


def test_reporting_summary(tmp_path: Path) -> None:
    result = _run(tmp_path, with_explainability=False)
    summary = visualization_summary(result)
    assert "## Visualizations" in summary
    assert "confusion_matrix.png" in summary
    assert "feature_importance_top20" in summary  # listed as skipped
    assert str(result["metadata"].parent) in summary


# ------------------------------------------------------------------- script


def test_script_reports_missing_experiment(tmp_path: Path, monkeypatch) -> None:
    from scripts.run_visualizations import main

    monkeypatch.chdir(tmp_path)  # no experiments here
    exit_code = main(["--dataset", "nope", "--model", "xgboost"])
    assert exit_code == 1
