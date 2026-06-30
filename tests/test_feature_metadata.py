"""Tests for src.features.metadata (feature/target separation + column filtering)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import metadata


def test_split_xy_separates_label() -> None:
    frame = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
    x, y = metadata.split_xy(frame, "label")
    assert list(x.columns) == ["a"]
    assert y.name == "label"


def test_split_xy_missing_label_raises() -> None:
    with pytest.raises(KeyError):
        metadata.split_xy(pd.DataFrame({"a": [1]}), "label")


def test_select_feature_columns_drops_provenance_and_non_numeric() -> None:
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "i": [1, 2],
            "split": ["train", "train"],     # provenance (the reported bug)
            "attack_cat": ["dos", "normal"],  # secondary label / metadata
            "weird": ["x", "y"],              # non-numeric, not provenance
        }
    )
    out, excluded = metadata.select_feature_columns(df)
    assert list(out.columns) == ["num", "i"]
    assert "split" in excluded["provenance"]
    assert "attack_cat" in excluded["provenance"]
    assert "weird" in excluded["non_numeric"]


def test_select_feature_columns_is_case_insensitive() -> None:
    df = pd.DataFrame({"a": [1.0], "SPLIT": ["train"], "Source_File": ["d.csv"]})
    out, excluded = metadata.select_feature_columns(df)
    assert list(out.columns) == ["a"]
    assert {"SPLIT", "Source_File"} <= set(excluded["provenance"])


def test_select_feature_columns_extra_exclude() -> None:
    df = pd.DataFrame({"a": [1, 2], "drop_me": [3, 4]})
    out, excluded = metadata.select_feature_columns(df, extra_exclude=["drop_me"])
    assert list(out.columns) == ["a"]
    assert "drop_me" in excluded["provenance"]


def test_select_feature_columns_all_numeric_no_exclusion() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    out, excluded = metadata.select_feature_columns(df)
    assert list(out.columns) == ["a", "b"]
    assert excluded["provenance"] == []
    assert excluded["non_numeric"] == []
