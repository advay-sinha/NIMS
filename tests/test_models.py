"""Tests for src.models (base, registry, concrete models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import registry
from src.models.base import BaseModel


@pytest.fixture()
def xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 2, size=n)
    x = pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.1, n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    return x, pd.Series(y, name="label")


def _cfg(params: dict) -> dict:
    return {"gpu": False, "params": params}


def test_registry_lists_and_resolves() -> None:
    assert set(registry.available_models()) >= {"xgboost", "lightgbm", "isolation_forest"}
    assert registry.get_model_cls("xgboost").__name__ == "XGBoostModel"


def test_registry_unknown_raises() -> None:
    with pytest.raises(KeyError):
        registry.get_model_cls("does_not_exist")


def test_build_model_respects_gpu_flag(xy) -> None:
    model = registry.build_model("xgboost", _cfg({"n_estimators": 5}), use_gpu=True, seed=1)
    # Model-level gpu=False forces CPU regardless of global preference.
    assert model.use_gpu is False


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_supervised_model_fit_predict(name: str, xy) -> None:
    x, y = xy
    params = {"n_estimators": 5} if name == "xgboost" else {"n_estimators": 5, "num_leaves": 7}
    model = registry.build_model(name, _cfg(params), use_gpu=False, seed=42)
    model.fit(x, y)
    preds = model.predict(x)
    assert len(preds) == len(y)
    proba = model.predict_proba(x)
    assert proba is not None and proba.shape[0] == len(y)
    assert model.device == "cpu"
    assert model.is_supervised is True


def test_isolation_forest_is_unsupervised(xy) -> None:
    x, y = xy
    model = registry.build_model(
        "isolation_forest", _cfg({"n_estimators": 10}), use_gpu=False, seed=42
    )
    model.fit(x)  # y ignored
    preds = model.predict(x)
    assert set(np.unique(preds)) <= {0, 1}
    assert model.predict_proba(x) is None
    assert model.is_supervised is False


def test_model_save_load_roundtrip(xy, tmp_path: Path) -> None:
    x, y = xy
    model = registry.build_model("xgboost", _cfg({"n_estimators": 5}), use_gpu=False, seed=42)
    model.fit(x, y)
    path = model.save(tmp_path / "model.joblib")
    assert path.exists()

    loaded = BaseModel.load(path)
    assert np.array_equal(loaded.predict(x), model.predict(x))


def test_describe_reports_metadata(xy) -> None:
    model = registry.build_model("lightgbm", _cfg({"n_estimators": 5}), use_gpu=False, seed=1)
    desc = model.describe()
    assert desc["name"] == "lightgbm"
    assert desc["supervised"] is True
    assert "params" in desc
