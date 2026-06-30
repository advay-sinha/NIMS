"""Correlation-based feature filtering.

Purpose
-------
Detect highly correlated (redundant) feature pairs and drop one of each pair.
Supports Pearson and Spearman correlation. Fit on training data only; the
kept-column list is applied unchanged to validation/test.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_VALID_METHODS = ("pearson", "spearman")


@dataclass
class CorrelationResult:
    """Outcome of correlation filtering.

    Attributes
    ----------
    method:
        ``"pearson"`` | ``"spearman"``.
    threshold:
        Absolute correlation at/above which a pair is considered redundant.
    kept:
        Feature names retained.
    removed:
        Feature names removed (one per redundant pair).
    pairs:
        Highly correlated pairs as ``{feature_a, feature_b, correlation}``.
    """

    method: str
    threshold: float
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


def fit_correlation_filter(
    x_train: "Any",
    threshold: float = 0.95,
    method: str = "pearson",
) -> CorrelationResult:
    """Identify and mark redundant features by correlation on training data.

    A column is removed when it correlates (``|corr| >= threshold``) with any
    earlier-ordered column (upper-triangle scan), a deterministic recipe.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    threshold:
        Absolute correlation cut-off.
    method:
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    CorrelationResult

    Raises
    ------
    ValueError
        If ``method`` is not supported.
    """
    import numpy as np

    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown correlation method {method!r}; expected {_VALID_METHODS}.")

    numeric = x_train.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        cols = [str(c) for c in x_train.columns]
        return CorrelationResult(method=method, threshold=float(threshold), kept=cols)

    corr = numeric.corr(method=method).abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)

    removed: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for col in upper.columns:
        high_rows = upper.index[upper[col] >= threshold]
        for row in high_rows:
            pairs.append(
                {
                    "feature_a": str(row),
                    "feature_b": str(col),
                    "correlation": float(upper.loc[row, col]),
                }
            )
        if len(high_rows) > 0:
            removed.add(str(col))

    kept = [str(c) for c in x_train.columns if str(c) not in removed]
    logger.info(
        "Correlation filter (%s, threshold=%s): kept %d, removed %d (%d pair(s)).",
        method, threshold, len(kept), len(removed), len(pairs),
    )
    return CorrelationResult(
        method=method,
        threshold=float(threshold),
        kept=kept,
        removed=sorted(removed),
        pairs=pairs,
    )
