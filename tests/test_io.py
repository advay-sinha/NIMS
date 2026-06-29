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


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_read_parquet_not_yet_implemented(tmp_path: Path) -> None:
    io.read_parquet(tmp_path / "x.parquet")


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_save_artifact_not_yet_implemented(tmp_path: Path) -> None:
    io.save_artifact(object(), tmp_path / "x.joblib")
