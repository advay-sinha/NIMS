"""Dataset metadata generation and persistence.

Purpose
-------
Build a :class:`src.data.schema.DatasetMetadata` record from a processed
dataset and persist it (Phase 1 outputs: metadata + data report). Metadata
makes each processed artefact self-describing and reproducible.

Inputs
------
- A processed ``pandas.DataFrame`` and its dataset config.
- The split sizes and source files used.

Outputs
-------
- A persisted ``metadata.json`` under ``paths.metadata_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from src.data.schema import DatasetMetadata

logger = logging.getLogger(__name__)


def build_metadata(
    frame: "Any",
    dataset_config: Mapping[str, Any],
    split_sizes: Mapping[str, int],
    source_files: list[str],
) -> DatasetMetadata:
    """Construct a :class:`DatasetMetadata` from a processed dataset.

    Parameters
    ----------
    frame:
        Processed feature DataFrame.
    dataset_config:
        The ``dataset`` config block.
    split_sizes:
        Row counts per split.
    source_files:
        Raw files the dataset was derived from.

    Returns
    -------
    DatasetMetadata
    """
    # TODO(data-engineer): infer FeatureMetadata per column (kind from config /
    #   dtype), populate shape, label column and config snapshot.
    raise NotImplementedError


def save_metadata(metadata: DatasetMetadata, metadata_dir: str | Path) -> Path:
    """Persist metadata to ``<metadata_dir>/<dataset_id>.metadata.json``.

    Parameters
    ----------
    metadata:
        Metadata record to persist.
    metadata_dir:
        Target directory (created if missing).

    Returns
    -------
    Path
        The written file path.
    """
    # TODO(data-engineer): use src.utils.io.write_json(metadata.to_dict(), ...).
    raise NotImplementedError
