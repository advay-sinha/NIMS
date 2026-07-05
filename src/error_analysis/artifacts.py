"""Error-analysis artefact persistence.

Purpose
-------
Write one experiment's error analysis to disk in a fixed layout::

    outputs/error_analysis/<experiment_id>/
        metadata.json                # identity + summary statistics
        confusion_matrix.csv         # true labels x predicted labels
        class_metrics.csv            # per-class precision/recall/F1/errors
        hardest_classes.csv          # classes ranked by lowest F1
        misclassified_examples.csv   # row_index, labels, confidence
        false_positive_examples.csv  # binary tasks only
        false_negative_examples.csv  # binary tasks only

Inputs
------
An :class:`src.error_analysis.analyzer.ErrorAnalysisResult` and experiment
identity.

Outputs
-------
Mapping of artefact name -> written path.

Limitations
-----------
No plots (later phase). Binary FP/FN files are omitted for multiclass tasks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_error_analysis_artifacts(
    result: Any,
    *,
    experiment_id: str,
    model_type: str,
    dataset_id: str,
    output_root: Path,
) -> dict[str, Path]:
    """Persist all error-analysis artefacts for one experiment.

    Parameters
    ----------
    result:
        :class:`ErrorAnalysisResult` to persist.
    experiment_id, model_type, dataset_id:
        Experiment identity recorded into ``metadata.json``.
    output_root:
        Error-analysis root directory; artefacts land in
        ``<output_root>/<experiment_id>/``.

    Returns
    -------
    dict[str, Path]
        Artefact name -> written path.
    """
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(output_root) / experiment_id)

    metadata = {
        "experiment_id": experiment_id,
        "dataset": dataset_id,
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result.summary,
    }
    paths: dict[str, Path] = {
        "metadata": write_json(metadata, out_dir / "metadata.json")
    }

    result.confusion.to_csv(out_dir / "confusion_matrix.csv")
    paths["confusion_matrix"] = out_dir / "confusion_matrix.csv"

    tables = {
        "class_metrics": result.class_metrics,
        "hardest_classes": result.hardest_classes,
        "misclassified_examples": result.misclassified,
        "false_positive_examples": result.false_positives,
        "false_negative_examples": result.false_negatives,
    }
    for name, frame in tables.items():
        if frame is None:
            continue
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    logger.info(
        "Error-analysis artefacts written to %s (%d/%d misclassified).",
        out_dir, result.summary["n_misclassified"], result.summary["n_samples"],
    )
    return paths
