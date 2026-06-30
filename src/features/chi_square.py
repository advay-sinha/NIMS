"""Chi-square feature scoring.

Purpose
-------
Score features by the chi-square statistic against the (classification) target.
Chi-square requires non-negative inputs, so features are min-max scaled to
``[0, 1]`` for scoring ONLY (the underlying data is never modified). Computed on
training data only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_chi_square(x_train: "Any", y_train: "Any") -> dict[str, float]:
    """Return per-feature chi-square scores.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    y_train:
        Training target vector.

    Returns
    -------
    dict[str, float]
        ``{feature: chi2_score}`` (non-finite scores coerced to ``0.0``).
    """
    import math

    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import MinMaxScaler

    # chi2 needs non-negative features; scale a copy for scoring only.
    scaled = MinMaxScaler().fit_transform(x_train)
    scores, _ = chi2(scaled, y_train)
    result = {
        str(col): (float(s) if math.isfinite(s) else 0.0)
        for col, s in zip(x_train.columns, scores)
    }
    logger.info("Computed chi-square scores for %d feature(s).", len(result))
    return result
