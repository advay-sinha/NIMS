"""Tests for src.features.selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import selection


@pytest.fixture()
def xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    x = pd.DataFrame(
        {
            "informative": y + rng.normal(0, 0.1, n),   # strongly predictive
            "weak": y * 0.2 + rng.normal(0, 1.0, n),     # weakly predictive
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
        }
    )
    return x, pd.Series(y, name="label")


def _config(method: str, k: int | None = 2) -> dict:
    return {
        "selection": {"method": method, "number_of_features": k},
        "importance": {"n_estimators": 20, "max_depth": None},
        "rfe": {"n_estimators": 20, "step": 1},
    }


@pytest.mark.parametrize("method", ["mutual_information", "anova", "chi_square"])
def test_selection_picks_informative_feature(method: str, xy) -> None:
    x, y = xy
    result = selection.select_features(x, y, _config(method, k=2), seed=42)
    assert result.method == method
    assert result.n_selected == 2
    assert "informative" in result.selected
    assert len(result.ranking) == x.shape[1]


def test_tree_importance_selection(xy) -> None:
    x, y = xy
    result = selection.select_features(x, y, _config("tree_importance", k=2), seed=42)
    assert "informative" in result.selected


def test_rfe_selection(xy) -> None:
    x, y = xy
    result = selection.select_features(x, y, _config("rfe", k=2), seed=42)
    assert result.method == "rfe"
    assert result.n_selected == 2
    assert "informative" in result.selected


def test_selection_is_deterministic(xy) -> None:
    x, y = xy
    a = selection.select_features(x, y, _config("mutual_information", k=3), seed=7)
    b = selection.select_features(x, y, _config("mutual_information", k=3), seed=7)
    assert a.selected == b.selected


def test_number_of_features_none_keeps_all_ranked(xy) -> None:
    x, y = xy
    result = selection.select_features(x, y, _config("anova", k=None), seed=42)
    assert result.n_selected == x.shape[1]


def test_unknown_method_raises(xy) -> None:
    x, y = xy
    with pytest.raises(ValueError):
        selection.select_features(x, y, _config("bogus"), seed=42)


def test_feature_selector_transform_picks_columns(xy) -> None:
    x, _ = xy
    selector = selection.FeatureSelector(
        selected_features=["informative", "noise1"], method="anova"
    )
    out = selector.transform(x)
    assert list(out.columns) == ["informative", "noise1"]


def test_feature_selector_ignores_absent_columns(xy) -> None:
    x, _ = xy
    selector = selection.FeatureSelector(
        selected_features=["informative", "missing"], method="anova"
    )
    out = selector.transform(x)
    assert list(out.columns) == ["informative"]
