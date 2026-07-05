"""Visualization source loading and metadata persistence.

Purpose
-------
Locate and load the upstream CSV artefacts (error analysis, explainability)
for one experiment, and persist the visualization run's ``metadata.json``.
Missing sources are reported as ``None`` — the runner records them as skipped
plots instead of failing.

Inputs
------
Experiment id + the error-analysis / explainability output roots.

Outputs
-------
DataFrames per source; a ``metadata.json`` describing what was generated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def load_source_csv(path: Path, **read_kwargs: Any) -> "Any | None":
    """Load one upstream CSV, returning ``None`` when it does not exist.

    Parameters
    ----------
    path:
        CSV to read.
    **read_kwargs:
        Passed to :func:`pandas.read_csv` (e.g. ``index_col``).

    Returns
    -------
    pandas.DataFrame | None
    """
    import pandas as pd

    if not Path(path).is_file():
        logger.warning("Source artefact missing: %s", path)
        return None
    return pd.read_csv(path, **read_kwargs)


def load_label_classes(preprocessing_dir: Path, dataset_id: str) -> list[str] | None:
    """Return decoded class names in encoded-id order, when recorded.

    The preprocessing pipeline persists the fitted label encoder's classes in
    ``<preprocessing_dir>/<dataset>/encoding_report.json`` (``label_classes``,
    index-aligned with the encoded integer ids used everywhere downstream).

    Parameters
    ----------
    preprocessing_dir:
        Preprocessing output root (``paths.preprocessing_dir``).
    dataset_id:
        Dataset identifier.

    Returns
    -------
    list[str] | None
        Decoded class names, or ``None`` when the report is absent.
    """
    import json

    report_path = Path(preprocessing_dir) / dataset_id / "encoding_report.json"
    if not report_path.is_file():
        logger.warning("No encoding report for %s; plots keep numeric class ids.",
                       dataset_id)
        return None
    try:
        classes = json.loads(report_path.read_text(encoding="utf-8")).get(
            "label_classes"
        )
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable encoding report %s: %s", report_path, exc)
        return None
    return [str(c) for c in classes] if classes else None


def write_visualization_metadata(
    out_dir: Path,
    *,
    experiment_id: str,
    dataset_id: str,
    model_type: str,
    plots_generated: Sequence[str],
    plots_skipped: Mapping[str, str],
    source_artifacts_used: Sequence[Path],
) -> Path:
    """Persist ``metadata.json`` for one visualization run.

    Parameters
    ----------
    out_dir:
        Experiment visualization directory.
    experiment_id, dataset_id, model_type:
        Experiment identity.
    plots_generated:
        Filenames of the rendered plots.
    plots_skipped:
        Plot name -> reason it was skipped.
    source_artifacts_used:
        Upstream artefact paths that were actually read.

    Returns
    -------
    Path
        The written ``metadata.json``.
    """
    from src.utils.io import write_json

    metadata = {
        "experiment_id": experiment_id,
        "dataset": dataset_id,
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plots_generated": sorted(plots_generated),
        "plots_skipped": dict(sorted(plots_skipped.items())),
        "source_artifacts_used": sorted(str(p) for p in source_artifacts_used),
    }
    return write_json(metadata, Path(out_dir) / "metadata.json")
