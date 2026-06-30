"""Tests for src.utils.io."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import io


def test_write_then_read_json_roundtrip(tmp_artifact_path: Path) -> None:
    payload = {"dataset": "nsl_kdd", "rows": 125973}
    written = io.write_json(payload, tmp_artifact_path)
    assert written.exists()
    assert io.read_json(written) == payload


def test_write_json_creates_parent_directories(tmp_artifact_path: Path) -> None:
    assert not tmp_artifact_path.parent.exists()
    io.write_json({"k": "v"}, tmp_artifact_path)
    assert tmp_artifact_path.parent.is_dir()


def test_write_then_read_parquet_roundtrip(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    target = tmp_path / "nested" / "df.parquet"
    written = io.write_parquet(df, target)
    assert written.exists()
    pd.testing.assert_frame_equal(io.read_parquet(written), df)


def test_save_then_load_artifact_roundtrip(tmp_path: Path) -> None:
    payload = {"weights": [1, 2, 3], "strategy": "standard"}
    target = tmp_path / "nested" / "obj.joblib"
    written = io.save_artifact(payload, target)
    assert written.exists()
    assert io.load_artifact(written) == payload


def test_read_parquet_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io.read_parquet(tmp_path / "missing.parquet")


def test_load_artifact_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io.load_artifact(tmp_path / "missing.joblib")
