"""ANOVA F-test feature scoring.

Purpose
-------
Score features by the one-way ANOVA F-statistic against the (classification)
target. Pure scoring; computed on training data only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_anova(x_train: "Any", y_train: "Any") -> dict[str, float]:
    """Return per-feature ANOVA F-scores.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    y_train:
        Training target vector.

    Returns
    -------
    dict[str, float]
        ``{feature: f_score}`` (non-finite scores coerced to ``0.0``;
        constant features legitimately produce ``NaN``).
    """
    import math
    import warnings

    from sklearn.feature_selection import f_classif

    with warnings.catch_warnings():
        # Constant features yield 0/0 -> NaN F-values; handled below.
        warnings.simplefilter("ignore")
        scores, _ = f_classif(x_train, y_train)
    result = {
        str(col): (float(s) if math.isfinite(s) else 0.0)
        for col, s in zip(x_train.columns, scores)
    }
    logger.info("Computed ANOVA F-scores for %d feature(s).", len(result))
    return result
