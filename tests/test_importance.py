"""Tests for src.features.importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import importance


@pytest.fixture()
def xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    x = pd.DataFrame(
        {
            "informative": y + rng.normal(0, 0.1, n),
            "noise": rng.normal(size=n),
        }
    )
    return x, pd.Series(y, name="label")


def test_build_random_forest_uses_config_and_seed() -> None:
    model = importance.build_random_forest({"n_estimators": 50, "max_depth": 5}, seed=42)
    assert model.n_estimators == 50
    assert model.max_depth == 5
    assert model.random_state == 42


def test_compute_tree_importance(xy) -> None:
    x, y = xy
    result = importance.compute_tree_importance(
        x, y, {"n_estimators": 25}, seed=42
    )
    assert set(result.importances) == {"informative", "noise"}
    # Importances sum to ~1 (RandomForest impurity importances).
    assert np.isclose(sum(result.importances.values()), 1.0, atol=1e-6)
    # The informative feature ranks first.
    assert result.ranking[0] == "informative"


def test_importance_is_deterministic(xy) -> None:
    x, y = xy
    a = importance.compute_tree_importance(x, y, {"n_estimators": 25}, seed=1)
    b = importance.compute_tree_importance(x, y, {"n_estimators": 25}, seed=1)
    assert a.importances == b.importances


def test_to_dict_serialisable(xy) -> None:
    x, y = xy
    result = importance.compute_tree_importance(x, y, {"n_estimators": 10}, seed=0)
    data = result.to_dict()
    assert "importances" in data and "ranking" in data
