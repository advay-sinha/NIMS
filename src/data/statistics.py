"""Descriptive dataset statistics — the single home for profiling.

Purpose
-------
All statistical profiling of a dataset lives here so that
:mod:`src.data.validation` orchestrates and validates without duplicating
computation (CLAUDE.md > Repository Principles: "Avoid duplicated code").

Responsibilities (Phase 1A):
    - dataset statistics (shape)
    - missing values
    - duplicates
    - memory usage
    - label distribution + class imbalance
    - numerical summaries
    - categorical summaries
    - infinite-value accounting

All functions are pure, read-only and return JSON-serialisable values (numpy
scalars are coerced to native python). Inputs are never mutated.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Primitive profiling helpers                                                 #
# --------------------------------------------------------------------------- #
def count_missing(frame: "Any") -> dict[str, int]:
    """Return per-column counts of missing (NaN/None) values (non-zero only)."""
    return {str(col): int(n) for col, n in frame.isna().sum().items() if n > 0}


def count_infinite(frame: "Any") -> dict[str, int]:
    """Return per-column counts of ``+/-inf`` for numeric columns (non-zero only)."""
    import numpy as np

    numeric = frame.select_dtypes(include=[np.number])
    counts: dict[str, int] = {}
    for col in numeric.columns:
        n = int(np.isinf(numeric[col].to_numpy()).sum())
        if n > 0:
            counts[str(col)] = n
    return counts


def count_duplicates(frame: "Any") -> int:
    """Return the number of exact duplicate rows."""
    return int(frame.duplicated().sum())


def memory_usage_bytes(frame: "Any") -> int:
    """Return the deep memory footprint of ``frame`` in bytes."""
    return int(frame.memory_usage(deep=True).sum())


def column_dtypes(frame: "Any") -> dict[str, str]:
    """Return a mapping of column name -> dtype string."""
    return {str(col): str(dtype) for col, dtype in frame.dtypes.items()}


# --------------------------------------------------------------------------- #
# Label statistics                                                            #
# --------------------------------------------------------------------------- #
def label_distribution(frame: "Any", label_column: str | None) -> dict[str, int]:
    """Return class counts for ``label_column`` (empty if unsupervised/absent)."""
    if label_column is None or label_column not in frame.columns:
        return {}
    counts = frame[label_column].value_counts(dropna=False)
    return {str(label): int(n) for label, n in counts.items()}


def class_imbalance(distribution: Mapping[str, int]) -> dict[str, Any]:
    """Summarise class imbalance from a label distribution.

    Returns
    -------
    dict
        Keys: ``n_classes``, ``majority_class``, ``minority_class``,
        ``majority_count``, ``minority_count``, ``imbalance_ratio``
        (majority/minority), ``proportions``.
    """
    if not distribution:
        return {
            "n_classes": 0,
            "majority_class": None,
            "minority_class": None,
            "majority_count": None,
            "minority_count": None,
            "imbalance_ratio": None,
            "proportions": {},
        }

    total = sum(distribution.values())
    majority = max(distribution, key=lambda k: distribution[k])
    minority = min(distribution, key=lambda k: distribution[k])
    minority_count = distribution[minority]
    ratio = (
        float(distribution[majority]) / float(minority_count)
        if minority_count > 0
        else None
    )
    return {
        "n_classes": len(distribution),
        "majority_class": majority,
        "minority_class": minority,
        "majority_count": int(distribution[majority]),
        "minority_count": int(minority_count),
        "imbalance_ratio": ratio,
        "proportions": {k: v / total for k, v in distribution.items()},
    }


# --------------------------------------------------------------------------- #
# Per-column summaries                                                        #
# --------------------------------------------------------------------------- #
def numerical_summary(frame: "Any") -> dict[str, dict[str, float]]:
    """Return summary statistics for numeric columns.

    Each column maps to count, mean, std, min, quartiles and max.
    """
    import numpy as np

    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return {}

    import math

    # ±inf in raw flow features makes std/quantile math emit benign warnings.
    with np.errstate(invalid="ignore", over="ignore"):
        described = numeric.describe().to_dict()
    summary: dict[str, dict[str, float]] = {}
    for col, stats in described.items():
        summary[str(col)] = {
            # Coerce NaN/±inf to None so the result is valid JSON.
            key: (float(val) if isinstance(val, (int, float)) and math.isfinite(val) else None)
            for key, val in stats.items()
        }
    return summary


def categorical_summary(
    frame: "Any",
    categorical_columns: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Return summary statistics for categorical columns.

    A column is summarised if it is object/category/bool dtype OR explicitly
    listed in ``categorical_columns``.

    Each column maps to count, unique, top value, top frequency and missing.
    """
    import numpy as np

    candidates = set(frame.select_dtypes(exclude=[np.number]).columns)
    candidates.update(c for c in categorical_columns if c in frame.columns)

    summary: dict[str, dict[str, Any]] = {}
    for col in candidates:
        series = frame[col]
        value_counts = series.value_counts(dropna=True)
        top = str(value_counts.index[0]) if len(value_counts) else None
        top_freq = int(value_counts.iloc[0]) if len(value_counts) else 0
        summary[str(col)] = {
            "count": int(series.notna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "top": top,
            "top_freq": top_freq,
            "missing": int(series.isna().sum()),
        }
    return summary


# --------------------------------------------------------------------------- #
# Aggregate                                                                   #
# --------------------------------------------------------------------------- #
def dataset_statistics(
    frame: "Any",
    label_column: str | None = None,
    categorical_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Aggregate every profiling metric into one JSON-serialisable dict.

    Parameters
    ----------
    frame:
        Input DataFrame.
    label_column:
        Target column name, or ``None`` for unsupervised data.
    categorical_columns:
        Columns to force-treat as categorical for the categorical summary.

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_features``, ``memory_usage_bytes``,
        ``memory_usage_mb``, ``duplicate_rows``, ``total_missing``,
        ``missing_values``, ``infinite_values``, ``dtypes``,
        ``label_distribution``, ``class_imbalance``, ``numerical_summary``,
        ``categorical_summary``.
    """
    missing = count_missing(frame)
    distribution = label_distribution(frame, label_column)
    mem_bytes = memory_usage_bytes(frame)

    stats = {
        "n_rows": int(len(frame)),
        "n_features": int(frame.shape[1]),
        "memory_usage_bytes": mem_bytes,
        "memory_usage_mb": round(mem_bytes / 1024**2, 4),
        "duplicate_rows": count_duplicates(frame),
        "total_missing": int(sum(missing.values())),
        "missing_values": missing,
        "infinite_values": count_infinite(frame),
        "dtypes": column_dtypes(frame),
        "label_distribution": distribution,
        "class_imbalance": class_imbalance(distribution),
        "numerical_summary": numerical_summary(frame),
        "categorical_summary": categorical_summary(frame, categorical_columns),
    }
    logger.debug(
        "Computed statistics: rows=%s features=%s missing=%s duplicates=%s",
        stats["n_rows"],
        stats["n_features"],
        stats["total_missing"],
        stats["duplicate_rows"],
    )
    return stats
