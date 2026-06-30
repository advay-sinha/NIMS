"""Mutual-information feature scoring.

Purpose
-------
Score features by mutual information with the (classification) target. Pure
scoring; selection of the top-k is handled in :mod:`src.features.selection`.
Computed on training data only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_mutual_information(
    x_train: "Any",
    y_train: "Any",
    seed: int = 42,
) -> dict[str, float]:
    """Return per-feature mutual information with the target.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    y_train:
        Training target vector.
    seed:
        Random seed for the estimator (mutual information uses a randomised
        nearest-neighbour estimate).

    Returns
    -------
    dict[str, float]
        ``{feature: mutual_information}``.
    """
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(x_train, y_train, random_state=seed)
    result = {str(col): float(s) for col, s in zip(x_train.columns, scores)}
    logger.info("Computed mutual information for %d feature(s).", len(result))
    return result
