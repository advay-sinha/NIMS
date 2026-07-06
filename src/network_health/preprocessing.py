"""Telemetry preprocessing.

Purpose
-------
Turn validated raw telemetry into model-ready splits::

    sort by time -> deduplicate -> fill missing -> counter deltas & rates
    -> clip impossible values -> optional resampling -> CHRONOLOGICAL split

Counter columns are converted to per-second rates from per-series deltas;
negative deltas (counter resets/wraps) are clipped to zero. Splits are by
time — never random — so no future information leaks into training.

Outputs
-------
A :class:`PreprocessingResult` (splits + manifest), persisted by
:mod:`src.network_health.artifacts`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from src.network_health.schema import TelemetrySchema

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Preprocessed chronological splits plus the reproducibility manifest."""

    splits: dict[str, Any]
    manifest: dict[str, Any]


def compute_counter_rates(
    frame: "Any", schema: TelemetrySchema, *, clip_resets: bool = True
) -> "Any":
    """Add ``<counter>_delta`` and ``<counter>_rate`` columns per series.

    Deltas are per-(device, interface) first differences; rates divide by
    the elapsed seconds between readings. The first reading of every series
    has no predecessor — delta and rate are 0. Negative deltas (counter
    reset/wrap) are clipped to 0 when ``clip_resets``.

    Parameters
    ----------
    frame:
        Telemetry sorted by series and time, with a parsed datetime
        timestamp column.
    schema:
        Column roles.
    clip_resets:
        Clip negative counter deltas to zero.

    Returns
    -------
    pandas.DataFrame
        Copy with the delta/rate columns appended.
    """
    result = frame.copy(deep=False)
    groups = result.groupby(schema.series_columns, sort=False)
    seconds = (
        groups[schema.timestamp_column].diff().dt.total_seconds()
    )
    for column in schema.counter_columns:
        if column not in result.columns:
            continue
        delta = groups[column].diff()
        if clip_resets:
            delta = delta.clip(lower=0)
        rate = (delta / seconds).where(seconds > 0)
        result[f"{column}_delta"] = delta.fillna(0.0)
        result[f"{column}_rate"] = rate.fillna(0.0)
    return result


def chronological_split(
    frame: "Any", schema: TelemetrySchema, fractions: Mapping[str, float]
) -> dict[str, Any]:
    """Split by global time cutoffs (train earliest, test latest).

    Parameters
    ----------
    frame:
        Preprocessed telemetry with a parsed timestamp column.
    schema:
        Column roles.
    fractions:
        ``{train, validation, test}`` fractions (normalised over their sum).

    Returns
    -------
    dict
        ``{split_name: DataFrame}`` — every row in ``train`` precedes every
        row in ``validation``, which precedes every row in ``test``.
    """
    total = sum(float(fractions.get(k, 0)) for k in ("train", "validation", "test"))
    train_q = float(fractions.get("train", 0.7)) / total
    val_q = train_q + float(fractions.get("validation", 0.15)) / total

    times = frame[schema.timestamp_column]
    t_train = times.quantile(train_q)
    t_val = times.quantile(val_q)
    splits = {
        "train": frame[times <= t_train],
        "validation": frame[(times > t_train) & (times <= t_val)],
        "test": frame[times > t_val],
    }
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


def preprocess_telemetry(
    frame: "Any",
    schema: TelemetrySchema,
    preprocessing_config: Mapping[str, Any],
    dataset_id: str,
) -> PreprocessingResult:
    """Run the full preprocessing sequence on raw telemetry.

    Parameters
    ----------
    frame:
        Loaded (raw) telemetry rows.
    schema:
        The dataset's configured schema.
    preprocessing_config:
        The ``network_health.preprocessing`` block.
    dataset_id:
        Identity recorded in the manifest.

    Returns
    -------
    PreprocessingResult
    """
    import pandas as pd

    n_raw = len(frame)
    work = frame.copy(deep=False)
    work[schema.timestamp_column] = pd.to_datetime(
        work[schema.timestamp_column], errors="coerce", format="mixed"
    )
    work = work.dropna(subset=[schema.timestamp_column])
    work = work.sort_values(
        [*schema.series_columns, schema.timestamp_column]
    )

    # One reading per (timestamp, device, interface).
    duplicate_subset = list(
        preprocessing_config.get("duplicate_subset")
        or [schema.timestamp_column, *schema.series_columns]
    )
    n_duplicates = int(work.duplicated(subset=duplicate_subset).sum())
    work = work.drop_duplicates(subset=duplicate_subset, keep="first")

    # Missing numeric values: forward-fill per series, back-fill the leading
    # gap, drop rows that remain incomplete.
    numeric = [c for c in schema.numeric_columns if c in work.columns]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    groups = work.groupby(schema.series_columns, sort=False)
    work[numeric] = groups[numeric].ffill()
    work[numeric] = work.groupby(schema.series_columns, sort=False)[numeric].bfill()
    n_incomplete = int(work[numeric].isna().any(axis=1).sum())
    work = work.dropna(subset=numeric)

    # Counter deltas + per-second rates (resets clipped by configuration).
    clip_resets = (
        str(preprocessing_config.get("counter_reset_handling", "clip_zero"))
        == "clip_zero"
    )
    work = compute_counter_rates(work, schema, clip_resets=clip_resets)

    # Clip impossible values.
    for column in schema.non_negative_columns:
        if column in work.columns:
            work[column] = work[column].clip(lower=0)
    for column, (low, high) in schema.bounded_columns.items():
        if column in work.columns:
            work[column] = work[column].clip(lower=low, upper=high)

    # Optional per-series resampling.
    rule = preprocessing_config.get("resample")
    if rule:
        work = _resample(work, schema, str(rule))

    splits = chronological_split(
        work, schema, dict(preprocessing_config.get("split") or {})
    )
    manifest = {
        "dataset_id": dataset_id,
        "n_raw_rows": n_raw,
        "n_rows_after_preprocessing": int(len(work)),
        "n_duplicates_removed": n_duplicates,
        "n_incomplete_rows_dropped": n_incomplete,
        "counter_reset_handling": "clip_zero" if clip_resets else "keep",
        "resample": rule,
        "split_rows": {name: int(len(part)) for name, part in splits.items()},
        "split_time_ranges": {
            name: {
                "start": str(part[schema.timestamp_column].min()),
                "end": str(part[schema.timestamp_column].max()),
            }
            for name, part in splits.items() if len(part)
        },
        "config": dict(preprocessing_config),
    }
    logger.info(
        "Preprocessed '%s': %d -> %d row(s); splits %s.",
        dataset_id, n_raw, len(work), manifest["split_rows"],
    )
    return PreprocessingResult(splits=splits, manifest=manifest)


def _resample(frame: "Any", schema: TelemetrySchema, rule: str) -> "Any":
    """Resample every (device, interface) series to a fixed interval."""
    import pandas as pd

    numeric = [
        c for c in frame.columns
        if c not in (*schema.series_columns, schema.timestamp_column)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    pieces = []
    for keys, part in frame.groupby(schema.series_columns, sort=False):
        resampled = (
            part.set_index(schema.timestamp_column)[numeric]
            .resample(rule).mean().dropna(how="all").reset_index()
        )
        for column, value in zip(schema.series_columns, keys):
            resampled[column] = value
        pieces.append(resampled)
    return pd.concat(pieces, ignore_index=True).sort_values(
        [*schema.series_columns, schema.timestamp_column]
    )
