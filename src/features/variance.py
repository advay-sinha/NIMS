"""Variance-threshold feature filtering.

Purpose
-------
Remove constant / near-constant features, which carry no signal and can break
downstream scoring. Fit on training data only; the resulting kept-column list
is applied unchanged to validation/test.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VarianceResult:
    """Outcome of variance-threshold filtering.

    Attributes
    ----------
    threshold:
        Variance threshold applied (features with ``var <= threshold`` removed).
    kept:
        Feature names retained.
    removed:
        Feature names removed.
    variances:
        Per-column variance (computed on training data).
    """

    threshold: float
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    variances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


def fit_variance_threshold(x_train: "Any", threshold: float = 0.0) -> VarianceResult:
    """Identify low-variance features on the training matrix.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    threshold:
        Features with variance ``<= threshold`` are removed.

    Returns
    -------
    VarianceResult
    """
    variances = x_train.var(numeric_only=True)
    removed = [str(c) for c in variances.index if variances[c] <= threshold]
    removed_set = set(removed)
    kept = [str(c) for c in x_train.columns if str(c) not in removed_set]
    logger.info(
        "Variance filter (threshold=%s): kept %d, removed %d feature(s).",
        threshold, len(kept), len(removed),
    )
    return VarianceResult(
        threshold=float(threshold),
        kept=kept,
        removed=removed,
        variances={str(c): float(v) for c, v in variances.items()},
    )
