"""Data cleaning stage.

Purpose
-------
Transform a validated raw DataFrame into a cleaned DataFrame driven entirely by
the ``data.cleaning`` configuration (CLAUDE.md > Repository Principles: "Never
hardcode hyperparameters; everything from configuration files"). Supported
steps, each independently toggleable:

    - explicit column dropping,
    - ``+/-inf`` -> NaN replacement,
    - duplicate-row removal (configurable keep policy),
    - constant (zero-variance) column removal,
    - all-null row removal,
    - missing-value imputation (numeric + categorical strategies),
    - IQR-based outlier clipping,
    - datatype normalisation (numeric downcasting).

Cleaning operates on a COPY; raw inputs are never mutated in place
(CLAUDE.md > Dataset Rules: "Nothing should overwrite raw data"). A
:class:`CleaningReport` records before/after metrics for auditability.

Inputs
------
- A raw ``pandas.DataFrame``.
- The ``data.cleaning`` config block.

Outputs
-------
- A cleaned ``pandas.DataFrame`` (new object) and a :class:`CleaningReport`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from src.utils.timer import Timer

logger = logging.getLogger(__name__)

# Numeric imputation strategies that compute a fill value from the column.
_NUMERIC_AGG_STRATEGIES = ("median", "mean")


@dataclass
class CleaningReport:
    """Audit record for a single cleaning run.

    Attributes
    ----------
    rows_before, rows_after:
        Row counts before and after cleaning.
    columns_before, columns_after:
        Column counts before and after cleaning.
    duplicates_removed:
        Number of duplicate rows dropped.
    infinities_replaced:
        Number of ``+/-inf`` cells converted to NaN.
    missing_before, missing_after:
        Total missing cells before and after imputation.
    rows_dropped_all_null:
        Rows removed because every value was null.
    outliers_clipped:
        Number of numeric cells clipped by the IQR rule.
    columns_dropped:
        Column names removed (explicit drop + constant columns).
    columns_modified:
        Column names whose values changed (imputed / clipped / dtype-cast).
    elapsed_seconds:
        Wall-clock duration of the cleaning stage.
    """

    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    duplicates_removed: int = 0
    infinities_replaced: int = 0
    missing_before: int = 0
    missing_after: int = 0
    rows_dropped_all_null: int = 0
    outliers_clipped: int = 0
    columns_dropped: list[str] = field(default_factory=list)
    columns_modified: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Individual cleaning steps                                                   #
# --------------------------------------------------------------------------- #
def drop_columns(frame: "Any", columns: Sequence[str]) -> tuple["Any", list[str]]:
    """Drop the configured columns that are present.

    Returns the (copy) frame and the list of columns actually dropped.
    """
    present = [c for c in columns if c in frame.columns]
    if not present:
        return frame, []
    logger.debug("Dropping configured columns: %s", present)
    return frame.drop(columns=present), present


def replace_infinities(frame: "Any") -> tuple["Any", int, list[str]]:
    """Replace ``+/-inf`` with NaN (common in CICIDS2017 flow ratios).

    Returns the frame, the number of cells replaced and the affected columns.
    """
    import numpy as np

    numeric = frame.select_dtypes(include=[np.number])
    mask = numeric.apply(lambda s: np.isinf(s.to_numpy())).sum()
    affected = [str(c) for c, n in mask.items() if n > 0]
    total = int(mask.sum())
    if total:
        frame = frame.replace([np.inf, -np.inf], np.nan)
        logger.debug("Replaced %d infinite values in %s", total, affected)
    return frame, total, affected


def drop_duplicate_rows(
    frame: "Any", keep: "str | bool" = "first"
) -> tuple["Any", int]:
    """Remove exact duplicate rows per the configured keep policy.

    ``keep`` is ``"first"`` | ``"last"`` | ``"none"`` (drop every duplicate).
    """
    keep_arg: "str | bool" = False if keep == "none" else keep
    before = len(frame)
    deduped = frame.drop_duplicates(keep=keep_arg)
    removed = before - len(deduped)
    if removed:
        logger.debug("Dropped %d duplicate rows (keep=%s)", removed, keep)
    return deduped, removed


def drop_constant_columns(frame: "Any") -> tuple["Any", list[str]]:
    """Remove zero-variance (constant) columns; they carry no signal."""
    constant = [str(c) for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
    if constant:
        logger.debug("Dropping constant columns: %s", constant)
        frame = frame.drop(columns=constant)
    return frame, constant


def drop_all_null_rows(frame: "Any") -> tuple["Any", int]:
    """Drop rows where every value is null."""
    before = len(frame)
    cleaned = frame.dropna(how="all")
    removed = before - len(cleaned)
    if removed:
        logger.debug("Dropped %d all-null rows", removed)
    return cleaned, removed


def impute_missing(
    frame: "Any", cleaning_config: Mapping[str, Any]
) -> tuple["Any", list[str]]:
    """Impute missing values per the configured numeric/categorical strategy.

    Numeric columns use ``numeric_impute`` (median | mean | zero | drop | none);
    categorical columns use ``categorical_impute`` (most_frequent | constant |
    drop | none). ``drop`` removes rows with any missing value in the relevant
    column group. Returns the frame and the list of columns whose values changed.
    """
    import numpy as np

    modified: list[str] = []
    numeric_cols = list(frame.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in frame.columns if c not in numeric_cols]

    numeric_strategy = str(cleaning_config.get("numeric_impute", "median"))
    categorical_strategy = str(cleaning_config.get("categorical_impute", "most_frequent"))
    fill_value = cleaning_config.get("categorical_fill_value", "__missing__")

    frame = _impute_numeric(frame, numeric_cols, numeric_strategy, modified)
    frame = _impute_categorical(
        frame, categorical_cols, categorical_strategy, fill_value, modified
    )
    return frame, modified


def _impute_numeric(
    frame: "Any",
    columns: Sequence[str],
    strategy: str,
    modified: list[str],
) -> "Any":
    """Apply the numeric imputation strategy in place on a copy."""
    if strategy == "none":
        return frame
    if strategy == "drop":
        subset = [c for c in columns if frame[c].isna().any()]
        return frame.dropna(subset=subset) if subset else frame

    for col in columns:
        if not frame[col].isna().any():
            continue
        if strategy in _NUMERIC_AGG_STRATEGIES:
            fill = getattr(frame[col], strategy)()
        elif strategy == "zero":
            fill = 0
        else:
            raise ValueError(f"Unknown numeric_impute strategy: {strategy!r}")
        frame[col] = frame[col].fillna(fill)
        modified.append(str(col))
    return frame


def _impute_categorical(
    frame: "Any",
    columns: Sequence[str],
    strategy: str,
    fill_value: Any,
    modified: list[str],
) -> "Any":
    """Apply the categorical imputation strategy in place on a copy."""
    if strategy == "none":
        return frame
    if strategy == "drop":
        subset = [c for c in columns if frame[c].isna().any()]
        return frame.dropna(subset=subset) if subset else frame

    for col in columns:
        if not frame[col].isna().any():
            continue
        if strategy == "most_frequent":
            modes = frame[col].mode(dropna=True)
            fill = modes.iloc[0] if len(modes) else fill_value
        elif strategy == "constant":
            fill = fill_value
        else:
            raise ValueError(f"Unknown categorical_impute strategy: {strategy!r}")
        frame[col] = frame[col].fillna(fill)
        modified.append(str(col))
    return frame


def clip_outliers(
    frame: "Any", outlier_config: Mapping[str, Any]
) -> tuple["Any", int, list[str]]:
    """Clip numeric outliers using the IQR rule when enabled.

    Values beyond ``Q1 - factor*IQR`` / ``Q3 + factor*IQR`` are clipped to those
    bounds. Returns the frame, the number of clipped cells and affected columns.
    """
    import numpy as np

    if not outlier_config.get("enabled", False):
        return frame, 0, []

    factor = float(outlier_config.get("factor", 1.5))
    configured = outlier_config.get("columns") or []
    numeric_cols = list(frame.select_dtypes(include=[np.number]).columns)
    target_cols = [c for c in configured if c in numeric_cols] or numeric_cols

    total_clipped = 0
    affected: list[str] = []
    for col in target_cols:
        series = frame[col]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or not np.isfinite(iqr):
            continue
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_clipped = int(((series < lower) | (series > upper)).sum())
        if n_clipped:
            frame[col] = series.clip(lower=lower, upper=upper)
            total_clipped += n_clipped
            affected.append(str(col))
    if total_clipped:
        logger.debug("Clipped %d outlier cells in %s", total_clipped, affected)
    return frame, total_clipped, affected


def normalize_dtypes(
    frame: "Any", normalize_config: Mapping[str, Any]
) -> tuple["Any", list[str]]:
    """Optionally downcast numeric columns to save memory.

    Returns the frame and the list of columns whose dtype changed.
    """
    import pandas as pd

    if not normalize_config.get("enabled", False):
        return frame, []
    if not normalize_config.get("downcast_numeric", True):
        return frame, []

    changed: list[str] = []
    for col in frame.select_dtypes(include=["integer"]).columns:
        new = pd.to_numeric(frame[col], downcast="integer")
        if new.dtype != frame[col].dtype:
            frame[col] = new
            changed.append(str(col))
    for col in frame.select_dtypes(include=["floating"]).columns:
        new = pd.to_numeric(frame[col], downcast="float")
        if new.dtype != frame[col].dtype:
            frame[col] = new
            changed.append(str(col))
    if changed:
        logger.debug("Downcast dtypes for %s", changed)
    return frame, changed


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _total_missing(frame: "Any") -> int:
    """Return the total number of missing cells in ``frame``."""
    return int(frame.isna().sum().sum())


def clean_dataset(
    frame: "Any", cleaning_config: Mapping[str, Any]
) -> tuple["Any", CleaningReport]:
    """Run the full cleaning stage, returning the cleaned copy and a report.

    Order
    -----
    drop columns -> replace infinities -> drop duplicates -> drop constant
    columns -> drop all-null rows -> impute missing -> clip outliers ->
    normalise dtypes. Each step is guarded by its config flag.

    Parameters
    ----------
    frame:
        Raw (validated) DataFrame. Never mutated in place.
    cleaning_config:
        The ``data.cleaning`` config block.

    Returns
    -------
    tuple[pandas.DataFrame, CleaningReport]
    """
    report = CleaningReport(
        rows_before=int(len(frame)),
        columns_before=int(frame.shape[1]),
        missing_before=_total_missing(frame),
    )
    modified: set[str] = set()

    with Timer("cleaning", log_level=logging.DEBUG) as timer:
        # Shallow copy: shares column blocks with ``frame``. Every mutating step
        # below either rebinds ``work`` to a new frame (drop/replace/dropna) or
        # assigns a single column, which Copy-on-Write isolates from ``frame``.
        # The raw input is therefore never modified, without an upfront deep copy.
        work = frame.copy(deep=False)

        work, dropped_explicit = drop_columns(
            work, cleaning_config.get("drop_columns", []) or []
        )
        report.columns_dropped.extend(dropped_explicit)

        if cleaning_config.get("replace_inf", True):
            work, n_inf, inf_cols = replace_infinities(work)
            report.infinities_replaced = n_inf
            modified.update(inf_cols)

        if cleaning_config.get("drop_duplicates", True):
            work, removed = drop_duplicate_rows(
                work, cleaning_config.get("duplicate_keep", "first")
            )
            report.duplicates_removed = removed

        if cleaning_config.get("drop_constant_columns", True):
            work, constant_cols = drop_constant_columns(work)
            report.columns_dropped.extend(constant_cols)

        if cleaning_config.get("drop_rows_all_null", True):
            work, n_rows = drop_all_null_rows(work)
            report.rows_dropped_all_null = n_rows

        work, imputed_cols = impute_missing(work, cleaning_config)
        modified.update(imputed_cols)

        work, n_clipped, clipped_cols = clip_outliers(
            work, cleaning_config.get("outlier_clip", {}) or {}
        )
        report.outliers_clipped = n_clipped
        modified.update(clipped_cols)

        work, dtype_cols = normalize_dtypes(
            work, cleaning_config.get("normalize_dtypes", {}) or {}
        )
        modified.update(dtype_cols)

    report.rows_after = int(len(work))
    report.columns_after = int(work.shape[1])
    report.missing_after = _total_missing(work)
    report.columns_modified = sorted(modified)
    report.elapsed_seconds = round(timer.elapsed, 6)

    logger.info(
        "Cleaning: rows %d->%d, cols %d->%d, dupes=%d, inf=%d, missing %d->%d",
        report.rows_before,
        report.rows_after,
        report.columns_before,
        report.columns_after,
        report.duplicates_removed,
        report.infinities_replaced,
        report.missing_before,
        report.missing_after,
    )
    return work, report
