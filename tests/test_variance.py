"""Tests for src.features.variance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import variance


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "informative": [1.0, 2.0, 3.0, 4.0, 5.0],
            "near_const": [1.0, 1.0, 1.0, 1.0, 1.01],
            "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )


def test_removes_constant_feature(frame: pd.DataFrame) -> None:
    result = variance.fit_variance_threshold(frame, threshold=0.0)
    assert "const" in result.removed
    assert "informative" in result.kept
    assert "const" not in result.kept


def test_threshold_removes_near_constant(frame: pd.DataFrame) -> None:
    result = variance.fit_variance_threshold(frame, threshold=0.01)
    assert "near_const" in result.removed
    assert "const" in result.removed
    assert result.kept == ["informative"]


def test_records_variances(frame: pd.DataFrame) -> None:
    result = variance.fit_variance_threshold(frame, threshold=0.0)
    assert set(result.variances) == {"informative", "near_const", "const"}
    assert result.variances["const"] == 0.0


def test_to_dict_is_serialisable(frame: pd.DataFrame) -> None:
    result = variance.fit_variance_threshold(frame)
    data = result.to_dict()
    assert data["threshold"] == 0.0
    assert isinstance(data["kept"], list)
