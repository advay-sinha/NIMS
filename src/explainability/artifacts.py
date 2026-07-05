"""Explainability artefact persistence.

Purpose
-------
Write one experiment's explanation to disk in a fixed, tool-friendly layout::

    outputs/explainability/<experiment_id>/
        metadata.json                  # provenance: run, model, dataset, versions
        feature_importance.csv         # feature, mean_abs_shap, rank
        shap_values.pkl                # full SHAP arrays for downstream analysis
        global_summary.png             # SHAP global summary plot
        global_feature_importance.csv  # extended global stats (see below)
        local/sample_NNNN.csv          # per-sample explanations (see below)

Inputs
------
An :class:`src.explainability.base.ExplanationResult` plus experiment
identity, and the explained sample matrix (feature values for the summary
plot and the local explanations).

Outputs
-------
Mapping of artefact name -> written path.

Multiclass aggregation (documented design decision)
---------------------------------------------------
SHAP values arrive normalised as ``(n_samples, n_features, n_outputs)``.
Multi-output (multiclass) explanations are AGGREGATED ACROSS OUTPUTS rather
than written per class: one local file per sample, where ``abs_contribution``
is the mean |SHAP| over class outputs and ``shap_contribution`` is the mean
signed SHAP over class outputs. With a single output (binary models) both
reduce exactly to the signed SHAP value and its magnitude. Per-class local
files were rejected because 40-class NSL-KDD would multiply the file count
by the class cardinality without aiding triage.

Limitations
-----------
Only the global summary plot is rendered; per-sample plots are a later phase.
An empty sample matrix persists metadata and raw values only (frames/plots
are skipped with a warning).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_PLOT_DEFAULTS = {"max_display": 20, "dpi": 150}


def feature_importance_frame(result: Any) -> "Any":
    """Rank features by global importance (mean |SHAP| over samples/outputs).

    Parameters
    ----------
    result:
        :class:`ExplanationResult` to aggregate.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``mean_abs_shap``, ``rank`` (1 = most
        important), sorted by importance; name breaks ties deterministically.
    """
    import numpy as np
    import pandas as pd

    mean_abs = np.abs(result.values).mean(axis=(0, 2))
    frame = pd.DataFrame(
        {"feature": result.feature_names, "mean_abs_shap": mean_abs}
    ).sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame


def global_importance_frame(result: Any) -> "Any":
    """Extended global importance statistics per feature.

    Parameters
    ----------
    result:
        :class:`ExplanationResult` to aggregate.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``mean_abs_shap``, ``std_abs_shap``,
        ``percentage_contribution``, ``cumulative_percentage``, ``rank``,
        sorted by importance (rank 1 first). Percentages are of the total
        mean-|SHAP| mass, so they sum to 100 and the cumulative column ends
        at 100 (0 when every SHAP value is exactly zero).
    """
    import numpy as np
    import pandas as pd

    per_sample = np.abs(result.values).mean(axis=2)  # (n_samples, n_features)
    mean_abs = per_sample.mean(axis=0)
    std_abs = per_sample.std(axis=0)
    total = float(mean_abs.sum())
    percentage = mean_abs / total * 100.0 if total > 0 else np.zeros_like(mean_abs)

    frame = pd.DataFrame(
        {
            "feature": result.feature_names,
            "mean_abs_shap": mean_abs,
            "std_abs_shap": std_abs,
            "percentage_contribution": percentage,
        }
    ).sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    frame["cumulative_percentage"] = frame["percentage_contribution"].cumsum()
    frame["rank"] = frame.index + 1
    return frame


def local_explanation_frame(result: Any, x: Any, row: int) -> "Any":
    """Per-feature explanation of one sample (outputs aggregated; see module).

    Parameters
    ----------
    result:
        :class:`ExplanationResult` holding the SHAP values.
    x:
        The explained sample matrix (row-aligned with ``result.values``).
    row:
        Positional row index of the sample to explain.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``feature_value``, ``shap_contribution``,
        ``abs_contribution``, ``contribution_rank`` (1 = strongest), sorted
        by absolute contribution.
    """
    import numpy as np
    import pandas as pd

    sample_values = result.values[row]  # (n_features, n_outputs)
    frame = pd.DataFrame(
        {
            "feature": result.feature_names,
            "feature_value": x.iloc[row].to_numpy(),
            "shap_contribution": sample_values.mean(axis=1),
            "abs_contribution": np.abs(sample_values).mean(axis=1),
        }
    ).sort_values(
        ["abs_contribution", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    frame["contribution_rank"] = frame.index + 1
    return frame


def write_local_explanations(
    result: Any,
    x: Any,
    out_dir: Path,
    *,
    max_samples: int,
    seed: int,
) -> Path | None:
    """Write per-sample explanation CSVs to ``<out_dir>/local/``.

    Samples are chosen deterministically: a generator seeded with ``seed``
    draws ``max_samples`` distinct rows (all rows when fewer exist), written
    in ascending row order as ``sample_0001.csv``, ``sample_0002.csv``, ...

    Parameters
    ----------
    result, x:
        Explanation and its row-aligned sample matrix.
    out_dir:
        Experiment artefact directory (the ``local`` subfolder is created).
    max_samples:
        Upper bound on the number of local files.
    seed:
        Reproducibility seed for sample selection.

    Returns
    -------
    Path | None
        The ``local`` directory, or ``None`` when there is nothing to write.
    """
    import numpy as np

    from src.utils.paths import ensure_dir

    count = min(int(max_samples), result.n_samples)
    if count <= 0:
        logger.warning("No samples available for local explanations; skipping.")
        return None

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(result.n_samples, size=count, replace=False))
    local_dir = ensure_dir(Path(out_dir) / "local")
    for position, row in enumerate(rows, start=1):
        local_explanation_frame(result, x, int(row)).to_csv(
            local_dir / f"sample_{position:04d}.csv", index=False
        )
    logger.info("Wrote %d local explanation file(s) to %s.", count, local_dir)
    return local_dir


def write_explanation_artifacts(
    result: Any,
    x: Any,
    *,
    experiment_id: str,
    model_type: str,
    dataset_id: str,
    output_root: Path,
    plot_config: Mapping[str, Any] | None = None,
    global_config: Mapping[str, Any] | None = None,
    local_config: Mapping[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Persist all explainability artefacts for one experiment.

    Parameters
    ----------
    result:
        :class:`ExplanationResult` to persist.
    x:
        The explained sample matrix (feature values colour the summary plot).
    experiment_id, model_type, dataset_id:
        Experiment identity recorded into ``metadata.json``.
    output_root:
        Explainability root directory; artefacts land in
        ``<output_root>/<experiment_id>/``.
    plot_config:
        Optional ``{max_display, dpi}`` overrides for the summary plot.
    global_config:
        ``{enabled}`` block for ``global_feature_importance.csv``
        (defaults to enabled).
    local_config:
        ``{enabled, max_samples}`` block for the per-sample explanations
        under ``local/`` (defaults to enabled, 5 samples).
    seed:
        Reproducibility seed for deterministic local sample selection.

    Returns
    -------
    dict[str, Path]
        Artefact name -> written path (``metadata``, ``shap_values``, and —
        when samples exist / sections are enabled — ``feature_importance``,
        ``global_summary``, ``global_feature_importance``,
        ``local_explanations``).
    """
    import joblib
    import shap

    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(output_root) / experiment_id)
    plot_cfg = {**_PLOT_DEFAULTS, **dict(plot_config or {})}

    metadata = {
        "experiment_id": experiment_id,
        "model_type": model_type,
        "dataset": dataset_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples_explained": result.n_samples,
        "n_features": len(result.feature_names),
        "class_labels": result.class_labels,
        "shap_version": shap.__version__,
    }
    metadata_path = write_json(metadata, out_dir / "metadata.json")

    values_path = out_dir / "shap_values.pkl"
    joblib.dump(
        {
            "values": result.values,
            "base_values": result.base_values,
            "feature_names": result.feature_names,
            "class_labels": result.class_labels,
        },
        values_path,
    )
    paths: dict[str, Path] = {"metadata": metadata_path, "shap_values": values_path}

    if result.n_samples == 0:
        logger.warning(
            "Explanation for %s has no samples; importance tables, plot and "
            "local explanations skipped.", experiment_id,
        )
        return paths

    importance_path = out_dir / "feature_importance.csv"
    feature_importance_frame(result).to_csv(importance_path, index=False)
    paths["feature_importance"] = importance_path

    global_cfg = dict(global_config or {"enabled": True})
    if global_cfg.get("enabled", True):
        global_path = out_dir / "global_feature_importance.csv"
        global_importance_frame(result).to_csv(global_path, index=False)
        paths["global_feature_importance"] = global_path

    local_cfg = dict(local_config or {})
    if local_cfg.get("enabled", True):
        local_dir = write_local_explanations(
            result, x, out_dir,
            max_samples=int(local_cfg.get("max_samples", 5)),
            seed=seed,
        )
        if local_dir is not None:
            paths["local_explanations"] = local_dir

    plot_path = out_dir / "global_summary.png"
    _write_summary_plot(result, x, plot_path, plot_cfg)
    paths["global_summary"] = plot_path

    logger.info(
        "Explainability artefacts written to %s (%d samples, %d features).",
        out_dir, result.n_samples, len(result.feature_names),
    )
    return paths


def _write_summary_plot(
    result: Any, x: Any, path: Path, plot_cfg: Mapping[str, Any]
) -> None:
    """Render the SHAP global summary plot to ``path`` (headless backend)."""
    import matplotlib

    matplotlib.use("Agg")  # never require a display
    import matplotlib.pyplot as plt
    import shap

    values = result.values
    try:
        if values.shape[2] > 1:  # multiclass: per-class mean-|SHAP| bars
            shap.summary_plot(
                [values[:, :, i] for i in range(values.shape[2])],
                x,
                feature_names=result.feature_names,
                class_names=result.class_labels,
                plot_type="bar",
                max_display=int(plot_cfg["max_display"]),
                show=False,
            )
        else:  # single output: beeswarm coloured by feature value
            shap.summary_plot(
                values[:, :, 0],
                x,
                feature_names=result.feature_names,
                max_display=int(plot_cfg["max_display"]),
                show=False,
            )
        plt.gcf().suptitle("SHAP global feature importance", fontsize=11)
        plt.savefig(path, dpi=int(plot_cfg["dpi"]), bbox_inches="tight")
    finally:
        plt.close("all")
