"""Error-analysis orchestration.

Purpose
-------
Compose the metric builders into one analysis of a fitted model's predictions
on a split, and provide the two entry surfaces: :func:`analyze_model` (used
by the post-hoc script) and :func:`maybe_analyze_after_training` (the
configuration-gated trainer hook, which can never fail a training run).

Inputs
------
A fitted model wrapper and an ``(X, y)`` split, or raw prediction vectors.

Outputs
-------
An :class:`ErrorAnalysisResult` (frames + summary), persisted via
:mod:`src.error_analysis.artifacts`.

Limitations
-----------
Analyses one split per invocation (the configured split, default ``test``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.error_analysis.metrics import (
    binary_error_frames,
    class_metrics_frame,
    confusion_frame,
    hardest_classes_frame,
    misclassified_frame,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_EXAMPLES = 1000


@dataclass(frozen=True)
class ErrorAnalysisResult:
    """All frames and summary statistics of one error analysis.

    ``false_positives``/``false_negatives`` are populated for binary tasks
    only (``None`` otherwise). ``summary`` holds the scalar statistics that
    feed ``metadata.json`` and the reporting helper.
    """

    confusion: Any
    class_metrics: Any
    hardest_classes: Any
    misclassified: Any
    false_positives: Any | None
    false_negatives: Any | None
    summary: dict[str, Any]


def analyze_predictions(
    y_true: Any,
    y_pred: Any,
    proba: Any | None,
    labels: Sequence[Any],
    *,
    max_misclassified: int = _DEFAULT_MAX_EXAMPLES,
    x: Any | None = None,
    include_feature_values: bool = False,
) -> ErrorAnalysisResult:
    """Analyse one set of predictions.

    Parameters
    ----------
    y_true, y_pred:
        Label vectors (positionally aligned).
    proba:
        Probability matrix aligned with ``labels`` columns, or ``None``;
        confidences then stay empty rather than failing.
    labels:
        Complete ordered class-label list (e.g. the model's fitted classes).
    max_misclassified:
        Cap on persisted misclassified examples.
    x:
        Optional feature matrix for ``include_feature_values``.
    include_feature_values:
        Append feature columns to the misclassified examples.

    Returns
    -------
    ErrorAnalysisResult
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(labels)
    confidence = None
    if proba is not None:
        confidence = np.asarray(proba).max(axis=1)

    class_metrics = class_metrics_frame(y_true, y_pred, labels)
    misclassified = misclassified_frame(
        y_true, y_pred, confidence,
        max_examples=max_misclassified,
        x=x, include_feature_values=include_feature_values,
    )
    binary = binary_error_frames(misclassified, labels)

    n_samples = int(len(np.asarray(y_true)))
    n_misclassified = int((np.asarray(y_true) != np.asarray(y_pred)).sum())
    summary = {
        "n_samples": n_samples,
        "n_classes": len(labels),
        "n_misclassified": n_misclassified,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }
    return ErrorAnalysisResult(
        confusion=confusion_frame(y_true, y_pred, labels),
        class_metrics=class_metrics,
        hardest_classes=hardest_classes_frame(class_metrics),
        misclassified=misclassified,
        false_positives=binary[0] if binary else None,
        false_negatives=binary[1] if binary else None,
        summary=summary,
    )


def analyze_model(
    model: Any,
    x: Any,
    y: Any,
    *,
    experiment_id: str,
    dataset_id: str,
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Path]:
    """Predict with a fitted model, analyse the errors, persist the artefacts.

    Parameters
    ----------
    model:
        Fitted :class:`src.models.base.BaseModel` wrapper.
    x, y:
        Feature matrix and true labels of the split to analyse.
    experiment_id, dataset_id:
        Experiment identity for the artefact directory and metadata.
    config:
        Effective merged configuration (``error_analysis`` block).
    output_root:
        Error-analysis output root (``paths.error_analysis_dir``).

    Returns
    -------
    tuple[dict[str, Path], ErrorAnalysisResult]
        Artefact name -> written path, plus the analysis itself (for
        reporting without re-predicting).
    """
    from src.error_analysis.artifacts import write_error_analysis_artifacts

    cfg = dict(config.get("error_analysis") or {})

    y_pred = model.predict(x)
    proba = model.predict_proba(x) if getattr(model, "is_supervised", True) else None
    labels = getattr(model, "classes_", None)
    if labels is None:
        import numpy as np

        labels = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))

    result = analyze_predictions(
        y, y_pred, proba, list(labels),
        max_misclassified=int(
            cfg.get("max_misclassified_examples", _DEFAULT_MAX_EXAMPLES)
        ),
        x=x,
        include_feature_values=bool(cfg.get("include_feature_values", False)),
    )
    paths = write_error_analysis_artifacts(
        result,
        experiment_id=experiment_id,
        model_type=str(getattr(model, "name", type(model).__name__)),
        dataset_id=dataset_id,
        output_root=Path(output_root),
    )
    return paths, result


def maybe_analyze_after_training(
    result: Any,
    model: Any,
    splits: Mapping[str, tuple[Any, Any]],
    config: Mapping[str, Any],
    paths: Any,
) -> dict[str, Path] | None:
    """Trainer hook: run error analysis on a just-completed run when enabled.

    Never raises: the training run's artefacts are already persisted when
    this executes, so failures are logged (with stack trace) and swallowed.

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
        Artefact paths, or ``None`` when disabled, unavailable or failed.
    """
    cfg = dict(config.get("error_analysis") or {})
    if not cfg.get("enabled", False):
        logger.debug("Error analysis disabled by configuration; skipping.")
        return None

    split = str(cfg.get("split", "test"))
    if split not in splits:
        logger.warning(
            "Error-analysis split '%s' not available for %s; skipping.",
            split, result.experiment_id,
        )
        return None

    x, y = splits[split]
    try:
        artefacts, _analysis = analyze_model(
            model, x, y,
            experiment_id=result.experiment_id,
            dataset_id=result.dataset_id,
            config=config,
            output_root=Path(paths.error_analysis_dir),
        )
        return artefacts
    except Exception:  # noqa: BLE001 - post-run enrichment must not fail the run
        logger.exception(
            "Error analysis failed for %s; the training run itself is "
            "unaffected.", result.experiment_id,
        )
        return None
