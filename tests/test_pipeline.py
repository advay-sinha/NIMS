"""Tests for src.data.pipeline (orchestration + no leakage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from src.data import pipeline
from src.data.base import RawDataset
from src.data.splitting import train_val_test_split
from src.utils.io import load_artifact, read_parquet


def _frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "duration": rng.normal(size=n),
            "proto": rng.choice(["tcp", "udp"], size=n),
            "bytes": rng.integers(0, 1000, size=n).astype(float),
            "label": ["attack"] * (n // 2) + ["normal"] * (n // 2),
        }
    )


def _config() -> dict[str, Any]:
    return {
        "project": {"seed": 42},
        "data": {
            "active_datasets": ["demo"],
            "split": {
                "train_size": 0.70,
                "val_size": 0.15,
                "test_size": 0.15,
                "stratify": True,
                "shuffle": True,
            },
            "cleaning": {
                "drop_duplicates": True,
                "drop_constant_columns": False,
                "replace_inf": True,
                "numeric_impute": "median",
                "categorical_impute": "most_frequent",
            },
            "encoding": {
                "categorical_strategy": "onehot",
                "handle_unknown": "ignore",
                "encode_labels": True,
            },
            "scaling": {"numeric_strategy": "standard"},
            "io": {"processed_format": "parquet", "compression": "snappy"},
        },
    }


def _patch(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, raw_dir: Path) -> None:
    dataset_config = {
        "id": "demo",
        "name": "Demo",
        "engine": "A",
        "label_column": "label",
        "categorical_columns": ["proto"],
    }

    class _Loader:
        def __init__(self, cfg: Any, paths: Any) -> None:
            pass

        def load_raw(self) -> RawDataset:
            return RawDataset(
                frame=frame, label_column="label", categorical_columns=("proto",)
            )

        def raw_dir(self) -> Path:
            return raw_dir

    monkeypatch.setattr(pipeline, "load_dataset_config", lambda did: dataset_config)
    monkeypatch.setattr(pipeline, "get_loader_cls", lambda did: _Loader)


@pytest.fixture()
def paths(make_paths: Callable[[dict], Any]):
    return make_paths({})


def test_run_pipeline_produces_all_outputs(monkeypatch, paths, tmp_path) -> None:
    frame = _frame()
    _patch(monkeypatch, frame, tmp_path)
    result = pipeline.run_pipeline("demo", _config(), paths)

    expected = {
        "train", "validation", "test",
        "encoder", "scaler", "label_encoder",
        "cleaning_report", "encoding_report", "scaling_report", "split_report",
        "manifest",
    }
    assert expected <= set(result.output_paths)
    for path in result.output_paths.values():
        assert Path(path).exists()


def test_processed_files_are_readable_and_labelled(monkeypatch, paths, tmp_path) -> None:
    frame = _frame()
    _patch(monkeypatch, frame, tmp_path)
    result = pipeline.run_pipeline("demo", _config(), paths)

    train = read_parquet(result.output_paths["train"])
    assert "label" in train.columns
    assert train.isna().sum().sum() == 0
    # Encoded label is integer {0, 1}.
    assert set(train["label"].unique()) <= {0, 1}
    # One-hot expanded proto -> indicator columns present.
    assert any(c.startswith("proto") for c in train.columns)


def test_no_data_leakage_scaler_fit_on_train_only(monkeypatch, paths, tmp_path) -> None:
    frame = _frame()
    _patch(monkeypatch, frame, tmp_path)
    cfg = _config()
    result = pipeline.run_pipeline("demo", cfg, paths)

    scaler = load_artifact(result.output_paths["scaler"])
    # Reproduce the deterministic split and confirm the scaler's learned mean
    # matches the TRAIN partition only — never the full dataset.
    x = frame.drop(columns=["label"])
    y = frame["label"]
    x_train, *_ = train_val_test_split(x, y, cfg["data"]["split"], 42)
    idx = list(scaler.columns).index("duration")
    assert np.isclose(scaler.scaler.mean_[idx], x_train["duration"].mean())
    assert not np.isclose(scaler.scaler.mean_[idx], x["duration"].mean())


def test_manifest_records_provenance(monkeypatch, paths, tmp_path) -> None:
    frame = _frame()
    _patch(monkeypatch, frame, tmp_path)
    result = pipeline.run_pipeline("demo", _config(), paths)

    from src.utils.io import read_json

    manifest = read_json(result.output_paths["manifest"])
    assert manifest["seed"] == 42
    assert manifest["encoder"]["strategy"] == "onehot"
    assert manifest["scaler"]["strategy"] == "standard"
    assert manifest["split_ratios"]["train"] == 0.70
    assert "fingerprint" in manifest
    assert manifest["label_encoder"]["column"] == "label"


def test_run_pipeline_fails_fast_on_validation_error(monkeypatch, paths, tmp_path) -> None:
    frame = _frame()
    frame["label"] = np.nan  # entirely-null label -> validation error
    _patch(monkeypatch, frame, tmp_path)
    with pytest.raises(RuntimeError):
        pipeline.run_pipeline("demo", _config(), paths)


# --------------------------------------------------------------------------- #
# Phase 2.1 — memory optimization                                             #
# --------------------------------------------------------------------------- #
def test_downcast_numeric_reduces_dtypes() -> None:
    df = pd.DataFrame(
        {
            "f": np.array([1.0, 2.0, 3.0], dtype="float64"),
            "i": np.array([1, 2, 3], dtype="int64"),
            "b": [True, False, True],
            "s": ["x", "y", "z"],
        }
    )
    out, changed = pipeline.downcast_numeric(df, enabled=True)
    assert str(out["f"].dtype) == "float32"
    assert out["i"].dtype.itemsize < 8          # int64 -> smaller int
    assert out["b"].dtype == bool               # bool untouched
    assert out["s"].dtype == df["s"].dtype       # non-numeric untouched
    assert "f" in changed and "i" in changed


def test_downcast_numeric_disabled_is_noop() -> None:
    df = pd.DataFrame({"f": np.array([1.0], dtype="float64")})
    out, changed = pipeline.downcast_numeric(df, enabled=False)
    assert out is df
    assert changed == {}


def test_memory_profile_written(monkeypatch, paths, tmp_path) -> None:
    from src.utils.io import read_json

    frame = _frame()
    _patch(monkeypatch, frame, tmp_path)
    result = pipeline.run_pipeline("demo", _config(), paths)

    assert "memory_profile" in result.output_paths
    profile = read_json(result.output_paths["memory_profile"])
    recorded = {s["stage"] for s in profile["stages"]}
    assert {"cleaning", "splitting", "encoding", "scaling", "serialization"} <= recorded
    assert "peak_mb" in profile
    assert profile["downcast_dtypes"]  # outputs were downcast


def test_processed_outputs_are_downcast(monkeypatch, paths, tmp_path) -> None:
    frame = _frame()
    _patch(monkeypatch, frame, tmp_path)
    result = pipeline.run_pipeline("demo", _config(), paths)

    train = read_parquet(result.output_paths["train"])
    # No float64 survives downcasting; scaled/one-hot columns become float32.
    assert not any(str(dt) == "float64" for dt in train.dtypes)
