"""Tests for src.training.experiment_index."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.training.experiment_index import (
    INDEX_COLUMNS,
    INDEX_FILENAME,
    append_index_row,
    index_row_from_manifest,
    rebuild_index,
)
from src.utils.io import write_json


def _manifest(run: str = "a", deep: bool = False, **overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"n_estimators": 400, "learning_rate": 0.1}
    fitted: dict[str, Any] = {}
    if deep:
        params = {
            "hidden_layers": [64, 32],
            "dropout": 0.2,
            "training": {"batch_size": 4096, "learning_rate": 0.002, "epochs": 50},
        }
        fitted = {"epochs_trained": 42}
    manifest = {
        "experiment_id": f"demo_model_{run}",
        "dataset_id": "demo",
        "model_name": "model",
        "created_at": f"2026-07-03T0{run}:00:00+00:00",
        "seed": 42,
        "model": {"device": "cuda", "fitted_params": fitted},
        "config_snapshot": {"model": {"params": params}},
        "hardware": {"device": "cuda"},
        "timings": {"train_seconds": 12.5},
        "metrics": {
            "train": {"accuracy": 0.99, "f1": 0.99, "roc_auc": 0.999},
            "test": {"accuracy": 0.95, "f1": 0.94, "roc_auc": 0.98},
        },
    }
    manifest.update(overrides)
    return manifest


def test_row_uses_test_split_and_classical_params() -> None:
    row = index_row_from_manifest(_manifest())
    assert row["run_id"] == "demo_model_a"
    assert row["accuracy"] == 0.95
    assert row["f1"] == 0.94
    assert row["roc_auc"] == 0.98
    assert row["hardware"] == "cuda"
    assert row["best_epoch"] is None
    assert "learning_rate=0.1" in row["key_hyperparameters"]
    assert set(row) == set(INDEX_COLUMNS)


def test_row_deep_model_lifts_training_block() -> None:
    row = index_row_from_manifest(_manifest(deep=True))
    assert row["best_epoch"] == 42
    hp = row["key_hyperparameters"]
    assert "batch_size=4096" in hp and "epochs=50" in hp and "dropout=0.2" in hp
    assert "hidden_layers" not in hp  # non-scalar params are omitted


def test_row_falls_back_to_train_split() -> None:
    manifest = _manifest()
    manifest["metrics"] = {"train": {"accuracy": 0.9, "f1": 0.9, "roc_auc": 0.9}}
    assert index_row_from_manifest(manifest)["accuracy"] == 0.9


def test_append_creates_header_once(tmp_path: Path) -> None:
    append_index_row(_manifest(run="a"), tmp_path)
    index_path = append_index_row(_manifest(run="b"), tmp_path)
    assert index_path == tmp_path / INDEX_FILENAME
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == ["demo_model_a", "demo_model_b"]


def test_rebuild_backfills_and_sorts(tmp_path: Path) -> None:
    for run in ("b", "a"):
        run_dir = tmp_path / "demo" / "model" / f"demo_model_{run}"
        write_json(_manifest(run=run), run_dir / "manifest.json")
    index_path = rebuild_index(tmp_path)
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == ["demo_model_a", "demo_model_b"]
    assert list(rows[0]) == INDEX_COLUMNS


def test_rebuild_empty_root_writes_header_only(tmp_path: Path) -> None:
    index_path = rebuild_index(tmp_path)
    with index_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
