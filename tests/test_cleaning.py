"""Tests for src.data.cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import cleaning


@pytest.fixture()
def dirty_frame() -> pd.DataFrame:
    """A frame exercising duplicates, infinities, missing and a constant col."""
    return pd.DataFrame(
        {
            "num": [1.0, 1.0, np.inf, np.nan, 5.0],
            "cat": ["a", "a", "b", None, "a"],
            "const": [7, 7, 7, 7, 7],
        }
    )


def test_replace_infinities_counts_and_replaces() -> None:
    frame = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 2.0]})
    out, n, cols = cleaning.replace_infinities(frame)
    assert n == 2
    assert cols == ["x"]
    assert not np.isinf(out["x"]).any()


def test_drop_duplicate_rows_keep_first() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, removed = cleaning.drop_duplicate_rows(frame, keep="first")
    assert removed == 1
    assert len(out) == 2


def test_drop_constant_columns() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "c": [9, 9, 9]})
    out, dropped = cleaning.drop_constant_columns(frame)
    assert dropped == ["c"]
    assert list(out.columns) == ["a"]


def test_impute_missing_numeric_median_and_categorical_mode() -> None:
    frame = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", None, "a"]})
    out, modified = cleaning.impute_missing(
        frame, {"numeric_impute": "median", "categorical_impute": "most_frequent"}
    )
    assert out["num"].isna().sum() == 0
    assert out["cat"].isna().sum() == 0
    assert out.loc[1, "num"] == 2.0  # median of [1, 3]
    assert out.loc[1, "cat"] == "a"  # mode
    assert set(modified) == {"num", "cat"}


def test_clip_outliers_iqr() -> None:
    frame = pd.DataFrame({"x": [10, 11, 12, 13, 1000]})
    out, n_clipped, cols = cleaning.clip_outliers(
        frame, {"enabled": True, "factor": 1.5}
    )
    assert n_clipped == 1
    assert cols == ["x"]
    assert out["x"].max() < 1000


def test_clip_outliers_disabled_is_noop() -> None:
    frame = pd.DataFrame({"x": [1, 2, 1000]})
    out, n_clipped, cols = cleaning.clip_outliers(frame, {"enabled": False})
    assert n_clipped == 0
    assert cols == []
    assert out["x"].max() == 1000


def test_clean_dataset_does_not_mutate_raw(dirty_frame: pd.DataFrame) -> None:
    before = dirty_frame.copy(deep=True)
    cleaning.clean_dataset(dirty_frame, {})
    pd.testing.assert_frame_equal(dirty_frame, before)


def test_clean_dataset_full_report(dirty_frame: pd.DataFrame) -> None:
    config = {
        "drop_duplicates": True,
        "drop_constant_columns": True,
        "replace_inf": True,
        "numeric_impute": "median",
        "categorical_impute": "most_frequent",
    }
    cleaned, report = cleaning.clean_dataset(dirty_frame, config)

    assert report.rows_before == 5
    assert report.duplicates_removed == 1
    assert report.infinities_replaced == 1
    assert "const" in report.columns_dropped       # constant column removed
    assert cleaned.isna().sum().sum() == 0          # all imputed
    assert report.missing_after == 0
    assert report.elapsed_seconds >= 0.0
    # Report serialises cleanly to JSON-friendly primitives.
    assert isinstance(report.to_dict(), dict)
