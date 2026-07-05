"""Optimization orchestration.

Purpose
-------
Run one Optuna study for a (dataset, model) pair: seeded sampler, splits
loaded once, every trial scored on the validation split, artefacts persisted
under ``outputs/optimization/<study_id>/`` and — when configured — one final
model trained with the best parameters through the standard experiment
tracking system (its manifest records the optimization provenance).

Inputs
------
Dataset/model identity, the effective merged configuration and resolved
paths.

Outputs
-------
An :class:`OptimizationResult`.

Limitations
-----------
Optimization never runs during normal training; it is invoked explicitly via
``scripts/run_optimization.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.optimization.objective import build_objective
from src.optimization.search_spaces import (
    OptimizationError,
    suggest_params,
    supported_models,
)

logger = logging.getLogger(__name__)

_SAMPLERS = ("tpe", "random")


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of one optimization study."""

    study_id: str
    best_value: float
    best_trial_number: int
    best_params: dict[str, Any]
    n_trials_completed: int
    artifact_paths: dict[str, Path]
    final_experiment_id: str | None


def _make_sampler(name: str, seed: int) -> Any:
    """Build a seeded Optuna sampler (reproducible by construction)."""
    import optuna

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise OptimizationError(
        f"Unknown sampler '{name}'. Supported: {', '.join(_SAMPLERS)}."
    )


def run_optimization(
    dataset_id: str,
    model_name: str,
    config: Mapping[str, Any],
    paths: Any,
    *,
    splits: Mapping[str, tuple[Any, Any]] | None = None,
) -> OptimizationResult:
    """Run one hyperparameter study and (optionally) train the final model.

    Parameters
    ----------
    dataset_id, model_name:
        What to tune. The model needs a search space AND must be enabled in
        the ``optimization.models`` configuration.
    config:
        Effective merged configuration (``optimization`` block + model
        configs). CLI overrides are merged before calling.
    paths:
        Resolved :class:`src.utils.paths.Paths`.
    splits:
        Pre-loaded feature splits (used by tests); loaded from the dataset's
        feature artefacts when omitted.

    Returns
    -------
    OptimizationResult

    Raises
    ------
    OptimizationError
        For unsupported/disabled models, unknown samplers/metrics, a missing
        validation split, or when every trial failed.
    """
    import optuna

    from src.optimization.artifacts import write_optimization_artifacts
    from src.optimization.reporting import optimization_summary

    cfg = dict(config.get("optimization") or {})
    if model_name not in supported_models():
        raise OptimizationError(
            f"No search space for model '{model_name}'. "
            f"Supported: {', '.join(supported_models())}."
        )
    model_flags = dict(cfg.get("models") or {}).get(model_name, {})
    if not dict(model_flags).get("enabled", True):
        raise OptimizationError(
            f"Optimization for '{model_name}' is disabled in configs/optimization.yaml."
        )
    models_cfg = config.get("models", {})
    if model_name not in models_cfg:
        raise OptimizationError(f"No configuration for model '{model_name}'.")

    metric = str(cfg.get("metric", "f1_weighted"))
    direction = str(cfg.get("direction", "maximize"))
    n_trials = int(cfg.get("n_trials", 20))
    timeout = cfg.get("timeout")
    seed = int(cfg.get("seed", 42))
    sampler_name = str(cfg.get("sampler", "tpe"))
    use_gpu = bool(config.get("training", {}).get("use_gpu", True))

    if splits is None:
        splits = _load_dataset_splits(dataset_id, paths)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    study_id = f"{dataset_id}_{model_name}_optuna_{stamp}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_id,
        direction=direction,
        sampler=_make_sampler(sampler_name, seed),
        storage=cfg.get("study_storage"),
    )
    objective = build_objective(
        model_name, models_cfg[model_name], splits,
        metric=metric, use_gpu=use_gpu, seed=seed,
    )
    logger.info(
        "Study %s: %d trial(s), sampler=%s, metric=%s (%s).",
        study_id, n_trials, sampler_name, metric, direction,
    )
    # catch=(Exception,) records failing trials as FAILED and continues.
    study.optimize(
        objective, n_trials=n_trials,
        timeout=float(timeout) if timeout else None,
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise OptimizationError(
            f"All {len(study.trials)} trial(s) failed; see the logs for the "
            f"underlying errors."
        )

    # Rebuild the config-shaped best params by replaying the best trial's raw
    # suggestions through the search space (single source of structure).
    best_params = suggest_params(
        optuna.trial.FixedTrial(study.best_trial.params), model_name
    )

    final_experiment_id = _train_final_model(
        dataset_id, model_name, config, paths, cfg, study, best_params, metric
    ) if cfg.get("train_final_best_model", True) else None

    summary = optimization_summary(
        study, study_id=study_id, metric=metric, best_params=best_params,
        artifact_dir=Path(paths.optimization_dir) / study_id,
        final_experiment_id=final_experiment_id,
    )
    artifact_paths = write_optimization_artifacts(
        study, study_id=study_id, dataset_id=dataset_id, model_name=model_name,
        metric=metric, direction=direction, n_trials_requested=n_trials,
        seed=seed, best_params=best_params,
        output_root=Path(paths.optimization_dir), summary_md=summary,
    )
    return OptimizationResult(
        study_id=study_id,
        best_value=float(study.best_value),
        best_trial_number=int(study.best_trial.number),
        best_params=best_params,
        n_trials_completed=len(completed),
        artifact_paths=artifact_paths,
        final_experiment_id=final_experiment_id,
    )


def _load_dataset_splits(dataset_id: str, paths: Any) -> dict[str, tuple[Any, Any]]:
    """Load the dataset's feature splits exactly as the trainer does."""
    from src.training.trainer import _load_splits
    from src.utils.config import load_dataset_config

    label_column = load_dataset_config(dataset_id).get("label_column")
    feat_dir = Path(paths.features_out_dir) / dataset_id
    return _load_splits(feat_dir, str(label_column))


def _train_final_model(
    dataset_id: str,
    model_name: str,
    config: Mapping[str, Any],
    paths: Any,
    cfg: Mapping[str, Any],
    study: Any,
    best_params: Mapping[str, Any],
    metric: str,
) -> str:
    """Train the best-params model through the standard experiment pipeline."""
    from src.training.trainer import train_model
    from src.utils.config import deep_merge

    final_config = deep_merge(
        dict(config), {"models": {model_name: {"params": dict(best_params)}}}
    )
    provenance = {
        "source": "optimization",
        "study_id": study.study_name,
        "best_trial_number": int(study.best_trial.number),
        "metric": metric,
        "best_validation_value": float(study.best_value),
    }
    logger.info("Training final model with the best parameters from %s.",
                study.study_name)
    result = train_model(
        dataset_id, model_name, final_config, paths, provenance=provenance
    )
    return result.experiment_id
