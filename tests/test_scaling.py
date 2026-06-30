"""Tests for src.data.scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import scaling


@pytest.fixture()
def numeric_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [0.0, 5.0, 10.0, 15.0], "b": [1.0, 1.0, 2.0, 2.0]})


def test_standard_scaler_zero_mean_unit_std(numeric_frame: pd.DataFrame) -> None:
    fitted = scaling.fit_scaler(numeric_frame, ["a", "b"], {"numeric_strategy": "standard"})
    out = scaling.apply_scaler(fitted, numeric_frame)
    assert np.allclose(out["a"].mean(), 0.0, atol=1e-9)
    assert np.allclose(out["a"].std(ddof=0), 1.0, atol=1e-9)


def test_minmax_scaler_bounds(numeric_frame: pd.DataFrame) -> None:
    fitted = scaling.fit_scaler(numeric_frame, ["a"], {"numeric_strategy": "minmax"})
    out = scaling.apply_scaler(fitted, numeric_frame)
    assert out["a"].min() == 0.0
    assert out["a"].max() == 1.0


def test_none_strategy_is_identity(numeric_frame: pd.DataFrame) -> None:
    fitted = scaling.fit_scaler(numeric_frame, ["a"], {"numeric_strategy": "none"})
    assert fitted.scaler is None
    out = scaling.apply_scaler(fitted, numeric_frame)
    # No-op returns the original object (no defensive copy).
    assert out is numeric_frame


def test_scaler_only_touches_named_columns(numeric_frame: pd.DataFrame) -> None:
    fitted = scaling.fit_scaler(numeric_frame, ["a"], {"numeric_strategy": "standard"})
    out = scaling.apply_scaler(fitted, numeric_frame)
    # 'b' not named -> passes through unchanged.
    pd.testing.assert_series_equal(out["b"], numeric_frame["b"])


def test_unknown_strategy_raises(numeric_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        scaling.fit_scaler(numeric_frame, ["a"], {"numeric_strategy": "bogus"})


def test_apply_scaler_does_not_mutate_input(numeric_frame: pd.DataFrame) -> None:
    before = numeric_frame.copy(deep=True)
    fitted = scaling.fit_scaler(numeric_frame, ["a"], {"numeric_strategy": "robust"})
    scaling.apply_scaler(fitted, numeric_frame)
    pd.testing.assert_frame_equal(numeric_frame, before)
