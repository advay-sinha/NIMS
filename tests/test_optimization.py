"""Tests for src.optimization (search spaces, objective, runner, artefacts)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.objective import build_objective, compute_metric
from src.optimization.runner import run_optimization
from src.optimization.search_spaces import (
    OptimizationError,
    suggest_params,
    supported_models,
)

_XGB_FIXED = {
    "max_depth": 5, "learning_rate": 0.1, "n_estimators": 100,
    "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
    "reg_alpha": 0.01,
}


def _fixed_trial(params: dict):
    import optuna

    return optuna.trial.FixedTrial(params)


@pytest.fixture()
def splits() -> dict:
    """Small 3-class train/validation splits (no test split needed)."""
    rng = np.random.default_rng(21)

    def _xy(n: int, seed_shift: int) -> tuple[pd.DataFrame, pd.Series]:
        y = pd.Series(np.arange(n) % 3, name="label")
        x = pd.DataFrame(
            {
                "f1": y + rng.normal(0, 0.3, n),
                "f2": rng.normal(size=n),
            }
        )
        return x, y

    return {"train": _xy(150, 0), "validation": _xy(60, 1)}


def _config(**optimization: object) -> dict:
    return {
        "training": {"use_gpu": False, "random_seed": 42},
        "models": {
            "xgboost": {"gpu": False, "params": {"n_jobs": -1}},
            "lightgbm": {"gpu": False, "params": {}},
        },
        "optimization": {
            "enabled": True, "n_trials": 3, "metric": "f1_weighted",
            "direction": "maximize", "sampler": "random", "seed": 42,
            "train_final_best_model": False,
            "models": {"xgboost": {"enabled": True}},
            **optimization,
        },
    }


class _Paths:
    def __init__(self, root: Path) -> None:
        self.optimization_dir = root / "optimization"
        self.features_out_dir = root / "features"
        self.experiments_dir = root / "experiments"


# ------------------------------------------------------------- search spaces


def test_xgboost_search_space_shape() -> None:
    params = suggest_params(_fixed_trial(_XGB_FIXED), "xgboost")
    assert set(params) == {
        "max_depth", "learning_rate", "n_estimators", "subsample",
        "colsample_bytree", "reg_lambda", "reg_alpha",
    }
    assert params["max_depth"] == 5 and params["reg_lambda"] == 1.0


def test_lightgbm_search_space_shape() -> None:
    fixed = {
        "num_leaves": 63, "max_depth": 8, "learning_rate": 0.1,
        "n_estimators": 200, "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_lambda": 1.0, "reg_alpha": 0.01, "min_child_samples": 20,
    }
    params = suggest_params(_fixed_trial(fixed), "lightgbm")
    assert params["num_leaves"] == 63
    assert params["min_child_samples"] == 20
    assert len(params) == 9


def test_mlp_search_space_is_config_shaped() -> None:
    fixed = {
        "hidden_dim": 256, "n_layers": 3, "dropout": 0.2,
        "learning_rate": 0.002, "batch_size": 2048, "weight_decay": 1e-4,
        "patience": 5,
    }
    params = suggest_params(_fixed_trial(fixed), "mlp")
    assert params["hidden_layers"] == [256, 128, 64]
    assert params["dropout"] == 0.2
    assert params["training"]["batch_size"] == 2048
    assert params["training"]["early_stopping"]["patience"] == 5


def test_unsupported_model_rejected() -> None:
    assert supported_models() == ["lightgbm", "mlp", "xgboost"]
    with pytest.raises(OptimizationError, match="No search space"):
        suggest_params(_fixed_trial({}), "cnn")


# ----------------------------------------------------------------- objective


def test_objective_returns_numeric_metric(splits) -> None:
    objective = build_objective(
        "xgboost", {"gpu": False, "params": {}}, splits,
        metric="f1_weighted", use_gpu=False, seed=42,
    )
    value = objective(_fixed_trial(_XGB_FIXED))
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_objective_requires_validation_split(splits) -> None:
    with pytest.raises(OptimizationError, match="validation split"):
        build_objective(
            "xgboost", {"gpu": False, "params": {}},
            {"train": splits["train"]},
            metric="f1_weighted", use_gpu=False, seed=42,
        )


def test_compute_metric_variants() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    assert compute_metric(y, pred, proba, "accuracy", [0, 1]) == pytest.approx(0.75)
    assert 0 < compute_metric(y, pred, proba, "f1_macro", [0, 1]) <= 1
    assert 0 < compute_metric(y, pred, proba, "roc_auc", [0, 1]) <= 1
    with pytest.raises(OptimizationError, match="Unknown metric"):
        compute_metric(y, pred, proba, "nope", [0, 1])
    with pytest.raises(OptimizationError, match="requires predicted"):
        compute_metric(y, pred, None, "roc_auc", [0, 1])


# -------------------------------------------------------------------- runner


def test_runner_persists_all_artifacts(splits, tmp_path: Path) -> None:
    result = run_optimization(
        "demo", "xgboost", _config(), _Paths(tmp_path), splits=splits
    )
    out_dir = tmp_path / "optimization" / result.study_id
    for name in ("metadata.json", "trials.csv", "best_params.json",
                 "best_trial.json", "optimization_summary.md"):
        assert (out_dir / name).is_file()

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in ("study_id", "dataset", "model_type", "metric", "direction",
                "n_trials_requested", "n_trials_completed", "seed",
                "timestamp", "best_value"):
        assert key in metadata
    assert metadata["n_trials_requested"] == 3
    assert metadata["n_trials_completed"] == 3
    assert metadata["best_value"] == pytest.approx(result.best_value)

    trials = pd.read_csv(out_dir / "trials.csv")
    assert len(trials) == 3
    assert {"trial_number", "state", "value", "duration_seconds"}.issubset(
        trials.columns
    )
    assert "param_max_depth" in trials.columns
    assert (trials["state"] == "COMPLETE").all()

    best_params = json.loads(
        (out_dir / "best_params.json").read_text(encoding="utf-8")
    )
    assert set(best_params) == {
        "max_depth", "learning_rate", "n_estimators", "subsample",
        "colsample_bytree", "reg_lambda", "reg_alpha",
    }
    assert result.final_experiment_id is None  # train_final_best_model: false


def test_runner_deterministic_with_fixed_seed(splits, tmp_path: Path) -> None:
    results = [
        run_optimization(
            "demo", "xgboost", _config(), _Paths(tmp_path / run), splits=splits
        )
        for run in ("a", "b")
    ]
    frames = [
        pd.read_csv(
            _Paths(tmp_path / run).optimization_dir / r.study_id / "trials.csv"
        )
        for run, r in zip(("a", "b"), results)
    ]
    param_cols = [c for c in frames[0].columns if c.startswith("param_")]
    pd.testing.assert_frame_equal(frames[0][param_cols], frames[1][param_cols])
    assert results[0].best_value == pytest.approx(results[1].best_value)


def test_runner_handles_failed_trials_gracefully(splits, tmp_path: Path,
                                                 monkeypatch) -> None:
    """One crashing trial must be recorded as FAILED, not end the study."""
    import src.optimization.objective as objective_module

    original = objective_module.suggest_params
    calls = {"n": 0}

    def _flaky(trial, model_name):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original(trial, model_name)

    monkeypatch.setattr(objective_module, "suggest_params", _flaky)
    result = run_optimization(
        "demo", "xgboost", _config(), _Paths(tmp_path), splits=splits
    )
    trials = pd.read_csv(
        tmp_path / "optimization" / result.study_id / "trials.csv"
    )
    assert sorted(trials["state"]) == ["COMPLETE", "COMPLETE", "FAIL"]
    assert result.n_trials_completed == 2


def test_runner_rejects_unsupported_and_disabled_models(splits, tmp_path: Path) -> None:
    with pytest.raises(OptimizationError, match="No search space"):
        run_optimization("demo", "cnn", _config(), _Paths(tmp_path), splits=splits)
    disabled = _config(models={"xgboost": {"enabled": False}})
    with pytest.raises(OptimizationError, match="disabled"):
        run_optimization("demo", "xgboost", disabled, _Paths(tmp_path),
                         splits=splits)


def test_final_training_flag_invokes_trainer_with_provenance(
    splits, tmp_path: Path, monkeypatch
) -> None:
    import src.training.trainer as trainer_module

    captured: dict = {}

    class _FakeResult:
        experiment_id = "demo_xgboost_20260101T000000"

    def _fake_train(dataset_id, model_name, config, paths, provenance=None):
        captured["params"] = config["models"][model_name]["params"]
        captured["provenance"] = provenance
        return _FakeResult()

    monkeypatch.setattr(trainer_module, "train_model", _fake_train)
    result = run_optimization(
        "demo", "xgboost", _config(train_final_best_model=True),
        _Paths(tmp_path), splits=splits,
    )
    assert result.final_experiment_id == "demo_xgboost_20260101T000000"
    assert captured["provenance"]["source"] == "optimization"
    assert captured["provenance"]["study_id"] == result.study_id
    assert captured["provenance"]["best_trial_number"] == result.best_trial_number
    # Best params merged onto the model config (existing keys preserved).
    assert captured["params"]["max_depth"] == result.best_params["max_depth"]
    assert captured["params"]["n_jobs"] == -1


def test_trainer_manifest_records_provenance(tmp_path: Path) -> None:
    from src.training.experiment import build_manifest, create_experiment

    experiment = create_experiment("demo", "xgboost", tmp_path)
    manifest = build_manifest(
        experiment=experiment, model_description={}, config_snapshot={},
        hardware={}, metrics={}, timings={}, artefacts={}, seed=42,
    )
    manifest["provenance"] = {"source": "optimization", "study_id": "s1"}
    assert manifest["provenance"]["study_id"] == "s1"


# -------------------------------------------------------------------- script


def test_script_argument_parsing() -> None:
    from scripts.run_optimization import _overrides, build_parser

    args = build_parser().parse_args(
        ["--dataset", "unsw_nb15", "--model", "xgboost", "--n-trials", "10",
         "--metric", "f1_macro", "--sampler", "random", "--seed", "7",
         "--no-final-train"]
    )
    assert args.dataset == "unsw_nb15" and args.model == "xgboost"
    overrides = _overrides(args)
    assert overrides == {
        "n_trials": 10, "metric": "f1_macro", "sampler": "random",
        "seed": 7, "train_final_best_model": False,
    }
