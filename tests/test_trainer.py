"""Tests for src.training.trainer (orchestration)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from src.training import trainer
from src.utils.io import read_json, write_parquet


def _feature_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.2, n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "label": y,
        }
    )


def _config() -> dict:
    return {
        "project": {"seed": 42},
        "training": {
            "random_seed": 42,
            "active_model": "xgboost",
            "models": ["xgboost"],
            "use_gpu": False,
            "min_train_rows": 50,   # test fixtures use ~120 rows
            "evaluation": {"average": "weighted"},
        },
        "models": {
            "xgboost": {"gpu": False, "params": {"n_estimators": 8, "max_depth": 3}},
        },
    }


@pytest.fixture()
def paths(make_paths: Callable[[dict], Any]):
    return make_paths({})


def _write_features(paths: Any) -> None:
    feat_dir = Path(paths.features_out_dir) / "demo"
    for name, off in zip(("train", "validation", "test"), (0, 1, 2)):
        write_parquet(_feature_frame(seed=off), feat_dir / f"{name}.parquet")


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trainer, "load_dataset_config", lambda did: {"label_column": "label"}
    )


def test_train_model_produces_artifacts(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_features(paths)
    result = trainer.train_model("demo", "xgboost", _config(), paths)

    assert {"model", "metrics", "manifest"} <= set(result.output_paths)
    for path in result.output_paths.values():
        assert Path(path).exists()
    assert result.output_dir.is_dir()


def test_metrics_cover_all_splits(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_features(paths)
    result = trainer.train_model("demo", "xgboost", _config(), paths)
    assert {"train", "validation", "test"} <= set(result.metrics)
    val = result.metrics["validation"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                "false_positive_rate", "confusion_matrix"):
        assert key in val


def test_manifest_records_provenance(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_features(paths)
    result = trainer.train_model("demo", "xgboost", _config(), paths)
    manifest = read_json(result.output_paths["manifest"])
    assert manifest["seed"] == 42
    assert manifest["model_name"] == "xgboost"
    assert "hardware" in manifest
    assert manifest["timings"]["train_seconds"] >= 0.0
    assert manifest["timings"]["model_size_bytes"] > 0


def test_unknown_model_raises(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_features(paths)
    with pytest.raises(KeyError):
        trainer.train_model("demo", "not_configured", _config(), paths)


def test_missing_features_raises(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    with pytest.raises(FileNotFoundError):
        trainer.train_model("demo", "xgboost", _config(), paths)


def test_isolation_forest_runs_end_to_end(monkeypatch, paths) -> None:
    _patch(monkeypatch)
    _write_features(paths)
    cfg = _config()
    cfg["models"]["isolation_forest"] = {"gpu": False, "params": {"n_estimators": 10}}
    result = trainer.train_model("demo", "isolation_forest", cfg, paths)
    assert result.metrics["validation"]["n_classes"] >= 1


def test_subset_training_matrix_is_rejected(monkeypatch, paths) -> None:
    """Regression: a tiny training matrix (the LightGBM 20-row bug) must abort."""
    _patch(monkeypatch)
    feat_dir = Path(paths.features_out_dir) / "demo"
    for name in ("train", "validation", "test"):
        write_parquet(_feature_frame(n=20), feat_dir / f"{name}.parquet")
    cfg = _config()
    cfg["training"]["min_train_rows"] = 1000  # default production guard
    with pytest.raises(AssertionError):
        trainer.train_model("demo", "xgboost", cfg, paths)


def test_full_feature_matrix_passed_to_fit(monkeypatch, paths) -> None:
    """The model must receive the full split (no sampling/debug slice)."""
    _patch(monkeypatch)
    _write_features(paths)  # 120 rows, 3 features
    cfg = _config()
    result = trainer.train_model("demo", "xgboost", cfg, paths)
    # Round-trip the confusion matrix support to confirm all rows were evaluated.
    cm = result.metrics["train"]["confusion_matrix"]
    total = sum(sum(row) for row in cm)
    assert total == 120
