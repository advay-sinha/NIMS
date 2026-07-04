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


def test_isolation_forest_explicit_n_jobs_from_config(xy) -> None:
    """Regression: config-provided n_jobs must not collide with a hardcoded one."""
    x, _ = xy
    model = registry.build_model(
        "isolation_forest", _cfg({"n_estimators": 10, "n_jobs": 1}), use_gpu=False, seed=42
    )
    model.fit(x)  # would raise TypeError if n_jobs were passed twice
    assert model.model.n_jobs == 1


def test_isolation_forest_defaults_n_jobs_when_unset(xy) -> None:
    x, _ = xy
    model = registry.build_model(
        "isolation_forest", _cfg({"n_estimators": 10}), use_gpu=False, seed=42
    )
    model.fit(x)
    assert model.model.n_jobs == -1


def test_isolation_forest_accepts_training_yaml_params(xy) -> None:
    """Regression: the full isolation_forest params block from configs/training.yaml
    (which includes n_jobs and random_state) must construct without duplicate
    keyword errors, and every user-configured hyperparameter must be preserved."""
    import yaml

    x, _ = xy
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        model_cfg = yaml.safe_load(fh)["models"]["isolation_forest"]

    model = registry.build_model("isolation_forest", model_cfg, use_gpu=False, seed=42)
    model.fit(x)

    fitted = model.model.get_params()
    for key, value in model_cfg["params"].items():
        assert fitted[key] == value, f"hyperparameter {key!r} not preserved"


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
    assert desc["fitted_params"] is None  # not fitted yet


def test_lightgbm_fitted_params_record_effective_device(xy) -> None:
    """Regression: manifests recorded the *configured* device (gpu) while the
    estimator actually trained on cpu after fallback. The final constructed
    parameter dict must be captured for provenance."""
    x, y = xy
    model = registry.build_model(
        "lightgbm",
        _cfg({"n_estimators": 5, "num_leaves": 7, "device": "gpu", "max_bin": 63}),
        use_gpu=False,  # forces the CPU path regardless of configured device
        seed=42,
    )
    model.fit(x, y)
    assert model.fitted_params is not None
    assert model.fitted_params["device"] == "cpu"
    assert model.fitted_params["max_bin"] == 63
    # Configured params stay untouched (both views are persisted).
    assert model.params["device"] == "gpu"
    assert model.describe()["fitted_params"]["device"] == "cpu"


@pytest.mark.parametrize("name", ["xgboost", "isolation_forest"])
def test_fitted_params_captured_after_fit(name: str, xy) -> None:
    x, y = xy
    model = registry.build_model(name, _cfg({"n_estimators": 5}), use_gpu=False, seed=42)
    model.fit(x, y)
    assert model.fitted_params is not None
    assert model.fitted_params["n_estimators"] == 5


# ------------------------------------------ XGBoost objective reconciliation
#
# Regression for the UNSW-NB15 failure: configs/training.yaml sets
# objective=multi:softprob for every dataset, but the XGBoost sklearn wrapper
# only injects num_class when it observes more than two classes. On the binary
# UNSW-NB15 target the core booster then aborted with
# "value 0 for Parameter num_class should be greater equal to 1".


def _xgboost_yaml_params() -> dict:
    """The real xgboost params block from configs/training.yaml (shrunk trees)."""
    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        params = dict(yaml.safe_load(fh)["models"]["xgboost"]["params"])
    params["n_estimators"] = 5  # keep the test fast; objective path unchanged
    return params


def _class_data(n_classes: int, n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    y = pd.Series(np.arange(n) % n_classes, name="label")
    x = pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.1, n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    return x, y


@pytest.mark.parametrize(
    ("dataset_like", "n_classes"),
    [("unsw_nb15", 2), ("cicids2017", 15), ("nsl_kdd", 40)],
)
def test_xgboost_yaml_objective_fits_every_dataset_cardinality(
    dataset_like: str, n_classes: int
) -> None:
    """The production objective config must fit binary and multiclass targets."""
    x, y = _class_data(n_classes)
    model = registry.build_model(
        "xgboost", _cfg(_xgboost_yaml_params()), use_gpu=False, seed=42
    )
    model.fit(x, y, x, y)  # config early_stopping_rounds needs an eval set
    assert model.model.n_classes_ == n_classes
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), n_classes)


def test_xgboost_binary_target_switches_to_binary_objective() -> None:
    x, y = _class_data(2)
    model = registry.build_model(
        "xgboost", _cfg(_xgboost_yaml_params()), use_gpu=False, seed=42
    )
    model.fit(x, y, x, y)
    assert model.fitted_params["objective"] == "binary:logistic"
    assert model.fitted_params["eval_metric"] == "logloss"
    # Configured params stay untouched (provenance keeps both views).
    assert model.params["objective"] == "multi:softprob"
    assert model.params["eval_metric"] == "mlogloss"


def test_xgboost_multiclass_target_keeps_configured_objective() -> None:
    x, y = _class_data(15)
    model = registry.build_model(
        "xgboost", _cfg(_xgboost_yaml_params()), use_gpu=False, seed=42
    )
    model.fit(x, y, x, y)
    assert model.fitted_params["objective"] == "multi:softprob"
    assert model.fitted_params["eval_metric"] == "mlogloss"


def test_reconcile_multiclass_objective_unit() -> None:
    from src.models.xgboost_model import _reconcile_multiclass_objective

    # Binary target + multiclass objective: switched, metrics mapped (list too).
    params = {"objective": "multi:softmax", "eval_metric": ["mlogloss", "auc"]}
    _reconcile_multiclass_objective(params, 2)
    assert params["objective"] == "binary:logistic"
    assert params["eval_metric"] == ["logloss", "auc"]

    # Multiclass target: untouched.
    params = {"objective": "multi:softprob", "eval_metric": "mlogloss"}
    _reconcile_multiclass_objective(params, 40)
    assert params == {"objective": "multi:softprob", "eval_metric": "mlogloss"}

    # Non-multiclass objective: untouched regardless of class count.
    params = {"objective": "binary:logistic"}
    _reconcile_multiclass_objective(params, 2)
    assert params == {"objective": "binary:logistic"}
