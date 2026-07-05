"""Visualization orchestration.

Purpose
-------
Generate every configured plot for one completed experiment from the
persisted error-analysis and explainability artefacts. Nothing upstream is
recomputed: a missing or empty source turns its plot into a ``plots_skipped``
metadata entry (with a warning) instead of a failure.

Inputs
------
Experiment identity, the upstream artefact roots and the ``visualization``
configuration block.

Outputs
-------
PNG plots + ``metadata.json`` under
``outputs/visualizations/<experiment_id>/``.

Limitations
-----------
Static plots only; no dashboard, no ROC/PR curves (later phases).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from src.visualization.artifacts import (
    load_label_classes,
    load_source_csv,
    write_visualization_metadata,
)
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_hardest_classes,
    plot_misclassification_pairs,
)

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "top_n_features": 20,
    "top_n_hardest_classes": 10,
    "top_n_misclassification_pairs": 20,
    "dpi": 150,
    "image_format": "png",
}


def generate_visualizations(
    experiment_id: str,
    *,
    dataset_id: str,
    model_type: str,
    error_analysis_dir: Path,
    explainability_dir: Path,
    output_root: Path,
    config: Mapping[str, Any],
    preprocessing_dir: Path | None = None,
) -> dict[str, Any]:
    """Render all visualizations for one experiment.

    Parameters
    ----------
    experiment_id:
        The run to visualize.
    dataset_id, model_type:
        Experiment identity for the metadata.
    error_analysis_dir, explainability_dir:
        Roots of the upstream artefact trees.
    output_root:
        Visualization output root (``paths.visualizations_dir``).
    config:
        Effective merged configuration (``visualization`` block).
    preprocessing_dir:
        Optional preprocessing output root; when given, decoded class names
        from the dataset's encoding report replace numeric ids on the
        confusion-matrix axes.

    Returns
    -------
    dict
        ``{"metadata": Path, "plots": {name: Path}, "skipped": {name: reason}}``.
    """
    from src.utils.paths import ensure_dir

    cfg = {**_DEFAULTS, **dict(config.get("visualization") or {})}
    fmt = str(cfg["image_format"])
    dpi = int(cfg["dpi"])
    out_dir = ensure_dir(Path(output_root) / experiment_id)

    error_dir = Path(error_analysis_dir) / experiment_id
    explain_dir = Path(explainability_dir) / experiment_id
    plots: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    sources: list[Path] = []

    # --- confusion matrix ----------------------------------------------------
    # Multiclass: the primary plot is row-normalized (raw counts hide
    # minority-class errors); the counts view is kept as a companion.
    # Binary: counts stay primary (both cells are readable), normalized second.
    confusion_source = error_dir / "confusion_matrix.csv"
    confusion = load_source_csv(confusion_source, index_col=0)
    if confusion is None:
        skipped["confusion_matrix"] = f"source artefact missing: {confusion_source}"
        skipped["confusion_matrix_normalized"] = (
            f"source artefact missing: {confusion_source}"
        )
    else:
        sources.append(confusion_source)
        class_names = (
            load_label_classes(preprocessing_dir, dataset_id)
            if preprocessing_dir is not None else None
        )
        multiclass = len(confusion.index) > 2
        variants = (
            (("confusion_matrix", True), ("confusion_matrix_counts", False))
            if multiclass
            else (("confusion_matrix", False), ("confusion_matrix_normalized", True))
        )
        for name, normalized in variants:
            path = out_dir / f"{name}.{fmt}"
            plot_confusion_matrix(
                confusion, path, normalized=normalized, dpi=dpi,
                class_names=class_names,
            )
            plots[name] = path

    # --- feature importance --------------------------------------------------
    importance_source = explain_dir / "global_feature_importance.csv"
    importance = load_source_csv(importance_source)
    if importance is None:
        skipped["feature_importance_top20"] = (
            f"source artefact missing: {importance_source}"
        )
    else:
        sources.append(importance_source)
        path = out_dir / f"feature_importance_top20.{fmt}"
        plot_feature_importance(
            importance, path, top_n=int(cfg["top_n_features"]), dpi=dpi
        )
        plots["feature_importance_top20"] = path

    # --- hardest classes ------------------------------------------------------
    hardest_source = error_dir / "hardest_classes.csv"
    hardest = load_source_csv(hardest_source)
    if hardest is None:
        skipped["hardest_classes_top10"] = (
            f"source artefact missing: {hardest_source}"
        )
    else:
        sources.append(hardest_source)
        path = out_dir / f"hardest_classes_top10.{fmt}"
        plot_hardest_classes(
            hardest, path, top_n=int(cfg["top_n_hardest_classes"]), dpi=dpi
        )
        plots["hardest_classes_top10"] = path

    # --- misclassification pairs ---------------------------------------------
    errors_source = error_dir / "misclassified_examples.csv"
    errors = load_source_csv(errors_source)
    if errors is None:
        skipped["misclassification_pairs_top20"] = (
            f"source artefact missing: {errors_source}"
        )
    elif len(errors) == 0:
        sources.append(errors_source)
        skipped["misclassification_pairs_top20"] = "no misclassified examples"
        logger.warning(
            "No misclassifications for %s; pairs plot skipped.", experiment_id
        )
    else:
        sources.append(errors_source)
        path = out_dir / f"misclassification_pairs_top20.{fmt}"
        plot_misclassification_pairs(
            errors, path,
            top_n=int(cfg["top_n_misclassification_pairs"]), dpi=dpi,
        )
        plots["misclassification_pairs_top20"] = path

    metadata_path = write_visualization_metadata(
        out_dir,
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        model_type=model_type,
        plots_generated=[p.name for p in plots.values()],
        plots_skipped=skipped,
        source_artifacts_used=sources,
    )
    logger.info(
        "Visualizations for %s: %d generated, %d skipped -> %s",
        experiment_id, len(plots), len(skipped), out_dir,
    )
    return {"metadata": metadata_path, "plots": plots, "skipped": skipped}
