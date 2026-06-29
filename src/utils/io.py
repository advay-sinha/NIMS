"""Typed IO helpers.

Purpose
-------
Provide a small, consistent surface for reading and writing the artefact
formats used across the pipeline (parquet, csv, json, yaml, joblib). Keeping
IO in one place enforces consistent compression / encoding and eases future
backend swaps.

Inputs / Outputs
----------------
Paths in, in-memory objects out (and vice versa). Parent directories are
created on write.

Examples
--------
>>> from src.utils.io import write_parquet, read_json
>>> write_parquet(df, paths.processed_dir / "nsl_kdd.parquet")  # doctest: +SKIP

Limitations
-----------
Heavy dependencies (pandas, pyarrow) are imported lazily inside functions to
keep ``import src.utils`` cheap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of ``path`` if needed and return ``Path``."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_parquet(path: str | Path) -> "Any":
    """Read a parquet file into a pandas DataFrame.

    Parameters
    ----------
    path:
        Source ``.parquet`` file.

    Returns
    -------
    pandas.DataFrame
    """
    # TODO(data-engineer): import pandas lazily; pd.read_parquet(path).
    raise NotImplementedError


def write_parquet(
    df: "Any",
    path: str | Path,
    compression: str = "snappy",
) -> Path:
    """Write a DataFrame to parquet, creating parent directories.

    Parameters
    ----------
    df:
        pandas DataFrame to persist.
    path:
        Destination ``.parquet`` file.
    compression:
        Parquet codec (default ``"snappy"``).

    Returns
    -------
    Path
        The path written.
    """
    target = _ensure_parent(path)
    # TODO(data-engineer): df.to_parquet(target, compression=compression).
    raise NotImplementedError


def read_csv(path: str | Path, **kwargs: Any) -> "Any":
    """Read a CSV into a pandas DataFrame.

    Parameters
    ----------
    path:
        Source ``.csv`` file.
    **kwargs:
        Forwarded to ``pandas.read_csv`` (e.g. ``header``, ``names``,
        ``chunksize``).

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV file not found: {resolved}")
    return pd.read_csv(resolved, **kwargs)


def read_json(path: str | Path) -> Any:
    """Read a JSON file into a python object."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise ``obj`` to a JSON file, creating parent directories.

    Parameters
    ----------
    obj:
        JSON-serialisable object.
    path:
        Destination ``.json`` file.
    indent:
        Pretty-print indentation.

    Returns
    -------
    Path
    """
    target = _ensure_parent(path)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str)
    return target


def write_yaml(obj: Any, path: str | Path) -> Path:
    """Serialise ``obj`` to a YAML file, creating parent directories."""
    target = _ensure_parent(path)
    # TODO(data-engineer): import yaml lazily; yaml.safe_dump(obj, ...).
    raise NotImplementedError


def save_artifact(obj: Any, path: str | Path) -> Path:
    """Persist an arbitrary python object via joblib (e.g. a fitted scaler).

    Parameters
    ----------
    obj:
        Object to pickle (encoder, scaler, ...).
    path:
        Destination ``.joblib`` file.

    Returns
    -------
    Path
    """
    target = _ensure_parent(path)
    # TODO(data-engineer): import joblib lazily; joblib.dump(obj, target).
    raise NotImplementedError


def load_artifact(path: str | Path) -> Any:
    """Load a joblib-serialised object (inverse of :func:`save_artifact`)."""
    # TODO(data-engineer): import joblib lazily; return joblib.load(path).
    raise NotImplementedError
