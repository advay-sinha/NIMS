"""Hyperparameter search spaces per model.

Purpose
-------
Define one conservative, deterministic search space per supported model.
Spaces return parameter dictionaries in the same shape the model's config
``params`` block uses, so they deep-merge straight into the existing model
configuration — no second registry, no parallel parameter plumbing.

Inputs
------
An Optuna trial (or ``FixedTrial``) to draw suggestions from.

Outputs
-------
A ``params``-shaped dict (nested for deep models).

Limitations
-----------
XGBoost, LightGBM and MLP only; CNN/LSTM/Transformer/Isolation Forest are a
later phase. Ranges are intentionally tight to keep trial runtime manageable.
"""

from __future__ import annotations

from typing import Any


class OptimizationError(RuntimeError):
    """Raised when an optimization request cannot be fulfilled."""


def _xgboost_space(trial: Any) -> dict[str, Any]:
    """Conservative XGBoost space around the validated baseline config."""
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
    }


def _lightgbm_space(trial: Any) -> dict[str, Any]:
    """Conservative LightGBM space; reg_lambda floor keeps the multiclass
    softmax divergence (see configs/training.yaml) out of the space."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }


def _mlp_space(trial: Any) -> dict[str, Any]:
    """MLP space over architecture width/depth and the training block.

    Returned in the nested ``params`` shape of configs/deep_learning.yaml
    (``hidden_layers``/``dropout`` at the top level, optimiser settings under
    ``training``), so it deep-merges onto the configured defaults.
    """
    first_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    return {
        "hidden_layers": [max(first_dim // (2 ** i), 32) for i in range(n_layers)],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "training": {
            "learning_rate": trial.suggest_float(
                "learning_rate", 5e-4, 5e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [1024, 2048, 4096]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            ),
            "early_stopping": {
                "patience": trial.suggest_int("patience", 3, 7),
            },
        },
    }


_SPACES = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "mlp": _mlp_space,
}


def supported_models() -> list[str]:
    """Return the model names with a defined search space."""
    return sorted(_SPACES)


def suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Draw one parameter set for ``model_name`` from its search space.

    Parameters
    ----------
    trial:
        Optuna trial (or ``FixedTrial`` when replaying known parameters).
    model_name:
        Registered model id.

    Returns
    -------
    dict
        ``params``-shaped parameter dict, mergeable onto the model config.

    Raises
    ------
    OptimizationError
        When no search space exists for ``model_name``.
    """
    space = _SPACES.get(model_name)
    if space is None:
        raise OptimizationError(
            f"No search space for model '{model_name}'. "
            f"Supported: {', '.join(supported_models())}."
        )
    return space(trial)
