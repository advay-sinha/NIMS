"""Tests for src.features.correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import correlation


@pytest.fixture()
def frame() -> pd.DataFrame:
    base = np.arange(50, dtype=float)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": base,
            "b": base * 2.0 + 1.0,        # perfectly correlated with a
            "c": rng.normal(size=50),     # independent
        }
    )


def test_removes_one_of_correlated_pair(frame: pd.DataFrame) -> None:
    result = correlation.fit_correlation_filter(frame, threshold=0.95, method="pearson")
    assert len(result.removed) == 1
    assert result.removed[0] in {"a", "b"}
    assert "c" in result.kept


def test_records_pairs(frame: pd.DataFrame) -> None:
    result = correlation.fit_correlation_filter(frame, threshold=0.95)
    assert any(
        {p["feature_a"], p["feature_b"]} == {"a", "b"} for p in result.pairs
    )
    assert result.pairs[0]["correlation"] >= 0.95


def test_high_threshold_keeps_all(frame: pd.DataFrame) -> None:
    result = correlation.fit_correlation_filter(frame, threshold=1.01)
    assert result.removed == []
    assert len(result.kept) == 3


def test_spearman_method(frame: pd.DataFrame) -> None:
    result = correlation.fit_correlation_filter(frame, threshold=0.95, method="spearman")
    assert result.method == "spearman"
    assert len(result.removed) == 1


def test_unknown_method_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        correlation.fit_correlation_filter(frame, method="kendall")
