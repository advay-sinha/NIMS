"""Explainability orchestration.

Purpose
-------
Connect a completed training run to the explainability backends: sample the
configured split deterministically, resolve the backend from the registry,
and persist the artefacts. Also provides the trainer's post-training hook,
which is configuration-gated and can never fail a training run whose
artefacts are already persisted.

Inputs
------
A fitted model wrapper, the loaded feature splits, the effective merged
configuration and resolved paths.

Outputs
-------
Artefact paths under ``outputs/explainability/<experiment_id>/`` (see
:mod:`src.explainability.artifacts`), or ``None`` when disabled/skipped.

Limitations
-----------
Backends exist for XGBoost only; other models are skipped with an INFO log.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from src.explainability.base import ExplainabilityError
from src.explainability.registry import available_explainers, get_explainer

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SAMPLES = 2000


def explain_model(
    model: Any,
    x: Any,
    *,
    experiment_id: str,
    dataset_id: str,
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Path]:
    """Explain a fitted model on ``x`` and persist the artefacts.

    Parameters
    ----------
    model:
        Fitted :class:`src.models.base.BaseModel` wrapper.
    x:
        Feature matrix of the split to explain (sampled down to the
        configured ``max_samples`` with the reproducibility seed).
    experiment_id, dataset_id:
        Experiment identity for the artefact directory and metadata.
    config:
        Effective merged configuration (``explainability`` block +
        reproducibility seed).
    output_root:
        Explainability output root (``paths.explainability_dir``).

    Returns
    -------
    dict[str, Path]
        Artefact name -> written path.

    Raises
    ------
    ExplainabilityError
        When no backend supports the model or the input is invalid.
    """
    from src.explainability.artifacts import write_explanation_artifacts

    cfg = dict(config.get("explainability") or {})
    max_samples = int(cfg.get("max_samples", _DEFAULT_MAX_SAMPLES))
    seed = config.get("training", {}).get("random_seed")
    if seed is None:
        seed = config.get("project", {}).get("seed", 42)
    seed = int(seed)

    sample = x
    if len(x) > max_samples:
        sample = x.sample(n=max_samples, random_state=seed)
        logger.info(
            "Explaining a seeded sample of %d/%d rows (max_samples=%d, seed=%d).",
            len(sample), len(x), max_samples, seed,
        )

    model_name = str(getattr(model, "name", type(model).__name__))
    explainer = get_explainer(model_name)
    result = explainer.explain(model, sample)
    return write_explanation_artifacts(
        result,
        sample,
        experiment_id=experiment_id,
        model_type=model_name,
        dataset_id=dataset_id,
        output_root=Path(output_root),
        plot_config=cfg.get("plot"),
        global_config=cfg.get("global_importance"),
        local_config=cfg.get("local_explanations"),
        seed=seed,
    )


def maybe_explain_after_training(
    result: Any,
    model: Any,
    splits: Mapping[str, tuple[Any, Any]],
    config: Mapping[str, Any],
    paths: Any,
) -> dict[str, Path] | None:
    """Trainer hook: explain a just-completed run when configured to.

    Never raises: the training run's artefacts are already persisted when
    this executes, so explainability failures are logged (with stack trace)
    and swallowed rather than failing a successful run.

    Parameters
    ----------
    result:
        :class:`src.training.trainer.TrainingResult` of the completed run.
    model:
        The fitted model wrapper from the same run.
    splits:
        Loaded feature splits ``{name: (X, y)}``.
    config:
        Effective merged configuration.
    paths:
        Resolved :class:`src.utils.paths.Paths`.

    Returns
    -------
    dict[str, Path] | None
        Artefact paths, or ``None`` when disabled, unsupported or failed.
    """
    cfg = dict(config.get("explainability") or {})
    if not cfg.get("enabled", False):
        logger.debug("Explainability disabled by configuration; skipping.")
        return None

    if result.model_name not in available_explainers():
        logger.info(
            "No explainability backend for model '%s' (supported: %s); skipping.",
            result.model_name, ", ".join(available_explainers()),
        )
        return None

    split = str(cfg.get("split", "test"))
    if split not in splits:
        logger.warning(
            "Explainability split '%s' not available for %s; skipping.",
            split, result.experiment_id,
        )
        return None

    x, _y = splits[split]
    try:
        return explain_model(
            model,
            x,
            experiment_id=result.experiment_id,
            dataset_id=result.dataset_id,
            config=config,
            output_root=Path(paths.explainability_dir),
        )
    except ExplainabilityError as exc:
        logger.error(
            "Explainability skipped for %s: %s", result.experiment_id, exc
        )
        return None
    except Exception:  # noqa: BLE001 - post-run enrichment must not fail the run
        logger.exception(
            "Explainability failed for %s; the training run itself is "
            "unaffected.", result.experiment_id,
        )
        return None
