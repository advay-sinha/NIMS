"""End-to-end data pipeline orchestration.

Purpose
-------
Compose the Phase 1 stages into one reproducible flow per dataset, honouring
the documented order so that encoders and scalers never see validation/test
data (no leakage):

    load_raw -> validate -> clean -> SPLIT -> fit(encode,scale) on train
             -> apply to val/test -> statistics -> metadata -> persist

This is the single orchestration surface; scripts call into it rather than
re-implementing the sequence (CLAUDE.md > Repository Principles: no duplicated
logic).

Inputs
------
- A dataset id, the merged config and resolved paths.

Outputs
-------
- Persisted processed splits, fitted transforms, statistics and metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.data.base import DatasetSplit

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run for one dataset.

    Attributes
    ----------
    dataset_id:
        Processed dataset id.
    split:
        The produced :class:`DatasetSplit`.
    output_paths:
        Mapping of artefact name -> written path.
    validation_passed:
        Whether validation produced no errors.
    """

    dataset_id: str
    split: DatasetSplit | None
    output_paths: dict[str, Path]
    validation_passed: bool


def run_pipeline(
    dataset_id: str,
    config: Mapping[str, Any],
    paths: Any,
) -> PipelineResult:
    """Run the full data pipeline for a single dataset.

    Steps
    -----
    1. Resolve loader via :mod:`src.data.registry` and ``load_raw``.
    2. Validate (:mod:`src.data.validation`); abort on errors.
    3. Clean (:mod:`src.data.cleaning`).
    4. Split (:mod:`src.data.splitting`) — before fitting transforms.
    5. Fit encoder + scaler on train; apply to all splits.
    6. Compute statistics; build + save metadata.
    7. Persist processed splits and fitted transforms.

    Parameters
    ----------
    dataset_id:
        Registered dataset identifier.
    config:
        Effective merged configuration.
    paths:
        Resolved :class:`src.utils.paths.Paths`.

    Returns
    -------
    PipelineResult

    Raises
    ------
    RuntimeError
        If validation fails with errors (fail-fast; do not persist).
    """
    logger.info("Starting data pipeline for dataset '%s'", dataset_id)
    # TODO(data-engineer): orchestrate the seven steps above, logging timing
    #   per stage via src.utils.timer.Timer. Fail fast on validation errors.
    raise NotImplementedError


def run_all(config: Mapping[str, Any], paths: Any) -> dict[str, PipelineResult]:
    """Run the pipeline for every dataset in ``data.active_datasets``.

    Parameters
    ----------
    config:
        Effective merged configuration.
    paths:
        Resolved paths.

    Returns
    -------
    dict
        dataset_id -> :class:`PipelineResult`.
    """
    # TODO(data-engineer): iterate config["data"]["active_datasets"], calling
    #   run_pipeline; collect results; continue-or-abort policy via config.
    raise NotImplementedError
