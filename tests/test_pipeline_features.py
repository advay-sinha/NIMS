"""Tests for src.features.pipeline (orchestration + no leakage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from src.features import pipeline
from src.utils.io import load_artifact, read_json, read_parquet, write_parquet


def _processed_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    informative = y + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "informative": informative,
            "redundant": informative * 2.0 + 1.0,  # ~perfectly correlated
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "const": np.ones(n),                   # zero variance
            "label": y,
        }
    )


def _config(method: str = "mutual_information", k: int | None = 2, pca: bool = False) -> dict:
    return {
        "project": {"seed": 42},
        "data": {"active_datasets": ["demo"]},
        "features": {
            "random_seed": 42,
            "variance": {"enabled": True, "threshold": 0.0},
            "correlation": {"enabled": True, "method": "pearson", "threshold": 0.95},
            "selection": {"enabled": True, "method": method, "number_of_features": k},
            "importance": {"n_estimators": 20, "max_depth": None},
            "rfe": {"n_estimators": 20, "step": 1},
            "dimensionality": {"pca": {"enabled": pca, "explained_variance": 0.95}},
            "reports": {"enabled": True},
            "artifacts": {"save": True},
            "io": {"format": "parquet", "compression": "snappy"},
        },
    }


@pytest.fixture()
def paths(make_paths: Callable[[dict], Any]):
    return make_paths({})


def _write_processed(paths: Any, seed_offsets=(0, 1, 2)) -> None:
    """Write demo train/validation/test processed parquet files."""
    proc_dir = Path(paths.processed_out_dir) / "demo"
    for name, off in zip(("train", "validation", "test"), seed_offsets):
        write_parquet(_processed_frame(seed=off), proc_dir / f"{name}.parquet")


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline, "load_dataset_config", lambda did: {"label_column": "label"}
    )


def test_pipeline_produces_all_outputs(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    result = pipeline.run_feature_pipeline("demo", _config(), paths)

    expected = {
        "train", "validation", "test",
        "feature_report", "feature_metadata", "selected_features", "removed_features",
        "feature_selector",
    }
    assert expected <= set(result.output_paths)
    for path in result.output_paths.values():
        assert Path(path).exists()


def test_variance_and_correlation_removal_recorded(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    result = pipeline.run_feature_pipeline("demo", _config(), paths)

    report = read_json(result.output_paths["feature_report"])
    assert report["original_feature_count"] == 5
    assert "const" in report["removed_by_variance"]
    # One of the correlated pair (informative/redundant) is dropped.
    assert len(report["removed_by_correlation"]) == 1


def test_no_leakage_val_test_match_train_columns(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    result = pipeline.run_feature_pipeline("demo", _config(k=2), paths)

    train = read_parquet(result.output_paths["train"])
    val = read_parquet(result.output_paths["validation"])
    test = read_parquet(result.output_paths["test"])
    # Identical schema across splits: selector fit on train, only transforms others.
    assert list(train.columns) == list(val.columns) == list(test.columns)
    assert "label" in train.columns
    # k=2 selected features + label.
    assert train.shape[1] == 3


def test_pipeline_is_deterministic(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    r1 = pipeline.run_feature_pipeline("demo", _config(), paths)
    sel1 = read_json(r1.output_paths["selected_features"])["selected"]
    r2 = pipeline.run_feature_pipeline("demo", _config(), paths)
    sel2 = read_json(r2.output_paths["selected_features"])["selected"]
    assert sel1 == sel2


def test_feature_selector_artifact_roundtrips(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    result = pipeline.run_feature_pipeline("demo", _config(), paths)
    selector = load_artifact(result.output_paths["feature_selector"])
    train = read_parquet(result.output_paths["train"])
    expected = [c for c in train.columns if c != "label"]
    assert selector.selected_features == expected


def test_pca_path(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    result = pipeline.run_feature_pipeline("demo", _config(pca=True), paths)
    assert result.pca_enabled
    assert "pca" in result.output_paths

    train = read_parquet(result.output_paths["train"])
    assert any(c.startswith("pc_") for c in train.columns)
    assert "label" in train.columns
    report = read_json(result.output_paths["feature_report"])
    assert report["pca"] is not None
    assert report["pca"]["n_components"] >= 1


def test_selection_disabled_keeps_post_correlation_features(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    cfg = _config()
    cfg["features"]["selection"]["enabled"] = False
    result = pipeline.run_feature_pipeline("demo", cfg, paths)
    report = read_json(result.output_paths["feature_report"])
    # const removed by variance, one correlated removed -> 3 features retained.
    assert report["retained_feature_count"] == 3


def test_missing_processed_data_raises(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    with pytest.raises(FileNotFoundError):
        pipeline.run_feature_pipeline("demo", _config(), paths)


# --------------------------------------------------------------------------- #
# Regression: string provenance/metadata columns reaching feature selection   #
# ("could not convert string to float: 'train'")                              #
# --------------------------------------------------------------------------- #
def _frame_with_provenance(n: int = 200, seed: int = 0) -> pd.DataFrame:
    frame = _processed_frame(n, seed)
    frame["split"] = "train"          # string provenance -> the original crash
    frame["attack_cat"] = "dos"        # non-numeric secondary label / metadata
    frame["free_text"] = "x"           # non-numeric, not a known provenance name
    return frame


def _write_processed_with_provenance(paths: Any) -> None:
    proc_dir = Path(paths.processed_out_dir) / "demo"
    for name, off in zip(("train", "validation", "test"), (0, 1, 2)):
        write_parquet(_frame_with_provenance(seed=off), proc_dir / f"{name}.parquet")


@pytest.mark.parametrize("method", ["mutual_information", "anova", "chi_square"])
def test_string_columns_do_not_break_selection(monkeypatch, paths, method) -> None:
    _patch(monkeypatch)
    _write_processed_with_provenance(paths)
    # Must NOT raise "could not convert string to float: 'train'".
    result = pipeline.run_feature_pipeline("demo", _config(method, k=2), paths)

    report = read_json(result.output_paths["feature_report"])
    assert "split" in report["excluded_columns"]["provenance"]
    assert "attack_cat" in report["excluded_columns"]["provenance"]
    assert "free_text" in report["excluded_columns"]["non_numeric"]


def test_transformed_outputs_have_no_string_columns(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed_with_provenance(paths)
    result = pipeline.run_feature_pipeline("demo", _config(), paths)

    train = read_parquet(result.output_paths["train"])
    numeric = set(train.select_dtypes(include=[np.number]).columns)
    non_numeric = [c for c in train.columns if c not in numeric]
    # Only the integer label may be non-float; no string columns remain.
    assert "split" not in train.columns
    assert "attack_cat" not in train.columns
    assert "free_text" not in train.columns
    assert non_numeric == [] or non_numeric == ["label"]


def test_extra_exclude_columns_config(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_processed(paths)
    cfg = _config()
    cfg["features"]["exclude_columns"] = ["noise1"]
    result = pipeline.run_feature_pipeline("demo", cfg, paths)
    report = read_json(result.output_paths["feature_report"])
    assert "noise1" in report["excluded_columns"]["provenance"]
    train = read_parquet(result.output_paths["train"])
    assert "noise1" not in train.columns
