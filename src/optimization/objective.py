"""Optuna objective construction.

Purpose
-------
Turn the existing model-building path into an Optuna objective: each trial
merges its suggested parameters onto the configured model block, builds the
model through :func:`src.models.registry.build_model` (the trainer's own
factory), fits on the training split and scores on the VALIDATION split —
never the test split, which stays untouched for the final benchmark.

Inputs
------
The model's configuration block and the pre-loaded feature splits (loaded
once per study, not per trial).

Outputs
-------
A callable ``objective(trial) -> float``.

Limitations
-----------
Single-objective only; failed trials raise and are caught by the study's
``catch`` so the search continues.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from src.optimization.search_spaces import OptimizationError, suggest_params

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = ("f1_weighted", "f1_macro", "accuracy", "roc_auc")


def compute_metric(
    y_true: Any, y_pred: Any, proba: Any | None, metric: str, labels: Any
) -> float:
    """Compute one named validation metric.

    Parameters
    ----------
    y_true, y_pred:
        Validation labels and predictions.
    proba:
        Probability matrix (required for ``roc_auc``).
    metric:
        One of ``f1_weighted``, ``f1_macro``, ``accuracy``, ``roc_auc``.
    labels:
        Complete fitted class list (keeps multiclass scores defined when a
        class is absent from the validation split).

    Returns
    -------
    float

    Raises
    ------
    OptimizationError
        For unknown metrics, or ``roc_auc`` without probabilities.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    labels = list(labels)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric in ("f1_weighted", "f1_macro"):
        return float(
            f1_score(y_true, y_pred, labels=labels,
                     average=metric.split("_", 1)[1], zero_division=0)
        )
    if metric == "roc_auc":
        if proba is None:
            raise OptimizationError(
                "Metric 'roc_auc' requires predicted probabilities."
            )
        multiclass = {"multi_class": "ovr", "labels": labels} if len(labels) > 2 else {}
        scores = proba[:, 1] if len(labels) == 2 else proba
        return float(
            roc_auc_score(y_true, scores, average="weighted", **multiclass)
        )
    raise OptimizationError(
        f"Unknown metric '{metric}'. Supported: {', '.join(_SUPPORTED_METRICS)}."
    )


def build_objective(
    model_name: str,
    model_cfg: Mapping[str, Any],
    splits: Mapping[str, tuple[Any, Any]],
    *,
    metric: str,
    use_gpu: bool,
    seed: int,
) -> Callable[[Any], float]:
    """Build the Optuna objective for one (model, dataset) study.

    Parameters
    ----------
    model_name:
        Registered model id.
    model_cfg:
        The model's configuration block (``models.<name>``) — suggested
        parameters are deep-merged onto its ``params``.
    splits:
        Pre-loaded feature splits; ``train`` and ``validation`` are required.
    metric:
        Validation metric to optimize (see :func:`compute_metric`).
    use_gpu:
        GPU preference, negotiated per model exactly as in training.
    seed:
        Reproducibility seed injected into every trial's model.

    Returns
    -------
    Callable
        ``objective(trial) -> float`` (the validation metric).

    Raises
    ------
    OptimizationError
        When the validation split is missing.
    """
    from src.models.registry import build_model
    from src.utils.config import deep_merge

    if "validation" not in splits:
        raise OptimizationError(
            "Hyperparameter optimization requires a validation split; "
            "the test split is never used for tuning."
        )
    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]

    def objective(trial: Any) -> float:
        suggested = suggest_params(trial, model_name)
        cfg = deep_merge(dict(model_cfg), {"params": suggested})
        model = build_model(model_name, cfg, use_gpu, seed)
        model.fit(x_train, y_train, x_val, y_val)
        y_pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        value = compute_metric(y_val, y_pred, proba, metric, model.classes_)
        logger.info("Trial %d: %s=%.6f", trial.number, metric, value)
        return value

    return objective
