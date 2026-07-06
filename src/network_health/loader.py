"""Telemetry CSV loading.

Purpose
-------
Read network-health telemetry from a configured location — one CSV file or a
directory of CSVs — with optional device filtering. Raw files are read-only;
nothing here mutates or rewrites them.

Limitations
-----------
CSV only; live SNMP polling is a later phase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def load_telemetry(
    source_path: str | Path,
    *,
    device_column: str = "device_id",
    device_filter: Sequence[str] | None = None,
) -> "Any":
    """Load telemetry rows from a CSV file or a directory of CSVs.

    Parameters
    ----------
    source_path:
        A ``.csv`` file, or a directory whose ``*.csv`` files are read in
        sorted-name order and concatenated.
    device_column:
        Column used by ``device_filter``.
    device_filter:
        Optional device ids to keep (all devices when omitted).

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        When the path does not exist or a directory contains no CSVs.
    """
    import pandas as pd

    source = Path(source_path)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under {source}.")
    else:
        raise FileNotFoundError(f"Telemetry source not found: {source}")

    frames = [pd.read_csv(path) for path in files]
    frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d telemetry row(s) from %d file(s) under %s.",
        len(frame), len(files), source,
    )

    if device_filter:
        wanted = {str(d) for d in device_filter}
        if device_column not in frame.columns:
            raise KeyError(
                f"Cannot filter by '{device_column}': column not present."
            )
        before = len(frame)
        frame = frame[frame[device_column].astype(str).isin(wanted)]
        frame = frame.reset_index(drop=True)
        logger.info(
            "Device filter %s kept %d/%d row(s).", sorted(wanted), len(frame),
            before,
        )
    return frame
