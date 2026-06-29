"""Dataset fingerprinting.

Purpose
-------
Produce a compact, reproducible fingerprint of each raw dataset so that data
provenance and change-detection are possible across runs (Phase 1A). A
fingerprint records identity, shape, a content checksum and generation
metadata.

A fingerprint contains:
    - dataset name
    - row count
    - column count
    - SHA256 checksum (over the raw source data files)
    - schema version
    - generation timestamp (UTC, ISO 8601)
    - source path

This module is READ-ONLY with respect to raw data.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Keys under a dataset config's ``files`` mapping that hold loadable data
# (as opposed to auxiliary dictionaries / reference PDFs).
DATA_FILE_KEYS: tuple[str, ...] = ("train", "test", "data")

_CHUNK_SIZE: int = 1024 * 1024  # 1 MiB streaming reads for large captures.


def resolve_data_files(dataset_config: Mapping[str, Any]) -> list[str]:
    """Return the list of data-file names used to load a dataset.

    Handles the three config shapes in use:
      - ``files`` as a list (CICIDS2017 captures),
      - ``files`` as a dict with ``train`` / ``test`` (NSL-KDD, UNSW-NB15),
      - ``files`` as a dict with ``data`` (SNMP).

    Parameters
    ----------
    dataset_config:
        The ``dataset`` config block.

    Returns
    -------
    list[str]
        File names (relative to the dataset's raw directory).
    """
    files = dataset_config.get("files")
    if isinstance(files, list):
        return [str(f) for f in files]
    if isinstance(files, Mapping):
        return [str(files[k]) for k in DATA_FILE_KEYS if k in files]
    return []


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA256 of a file, read in streaming chunks.

    Parameters
    ----------
    path:
        File to hash.

    Returns
    -------
    str
        Lower-case hex digest.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Cannot checksum missing file: {resolved}")
    digest = hashlib.sha256()
    with open(resolved, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_checksum(paths: list[Path]) -> str:
    """Return one SHA256 over the per-file digests (order-independent).

    Each file is hashed individually; the sorted digests are then hashed
    together so the result is stable regardless of file ordering.

    Parameters
    ----------
    paths:
        Files to combine.

    Returns
    -------
    str
        Combined hex digest (``""`` for an empty list).
    """
    if not paths:
        return ""
    per_file = sorted(sha256_file(p) for p in paths)
    combined = hashlib.sha256()
    for digest in per_file:
        combined.update(digest.encode("ascii"))
    return combined.hexdigest()


def build_fingerprint(
    dataset_config: Mapping[str, Any],
    raw_dir: Path,
    n_rows: int,
    n_features: int,
) -> dict[str, Any]:
    """Assemble a fingerprint dict for one dataset.

    Parameters
    ----------
    dataset_config:
        The ``dataset`` config block.
    raw_dir:
        Resolved read-only raw directory for the dataset.
    n_rows, n_features:
        Shape of the loaded dataset.

    Returns
    -------
    dict
        JSON-serialisable fingerprint.
    """
    file_names = resolve_data_files(dataset_config)
    file_paths = [raw_dir / name for name in file_names]

    per_file = {
        name: sha256_file(path)
        for name, path in zip(file_names, file_paths)
        if path.is_file()
    }

    fingerprint = {
        "dataset_name": dataset_config.get("name"),
        "dataset_id": dataset_config.get("id"),
        "row_count": int(n_rows),
        "column_count": int(n_features),
        "sha256": combined_checksum([p for p in file_paths if p.is_file()]),
        "schema_version": str(dataset_config.get("schema_version", "1.0")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(raw_dir),
        "source_files": per_file,
    }
    logger.info(
        "[%s] fingerprint sha256=%s rows=%s cols=%s",
        fingerprint["dataset_id"],
        fingerprint["sha256"][:12] or "<none>",
        n_rows,
        n_features,
    )
    return fingerprint
