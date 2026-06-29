"""Reproducible train / validation / test splitting.

Purpose
-------
Partition a cleaned dataset into train / validation / test sets reproducibly
(CLAUDE.md > Machine Learning Standards: "Never train without validation").
Splitting happens BEFORE encoder / scaler fitting so those transforms see only
training data (no leakage).

Inputs
------
- Feature matrix ``X`` and optional target ``y``.
- The ``data.split`` config block and a seed.

Outputs
-------
- Six partitions ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def _validate_ratios(train: float, val: float, test: float) -> None:
    """Ensure split ratios are valid and sum to ~1.0.

    Raises
    ------
    ValueError
        If any ratio is out of (0, 1) or they do not sum to 1.0.
    """
    if not all(0.0 < r < 1.0 for r in (train, val, test)):
        raise ValueError("Each split ratio must lie in the open interval (0, 1).")
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def train_val_test_split(
    x: "Any",
    y: "Any | None",
    split_config: Mapping[str, Any],
    seed: int,
) -> tuple["Any", "Any", "Any", "Any | None", "Any | None", "Any | None"]:
    """Split features (and optional target) into three partitions.

    Performed as two successive splits: first hold out the test set, then carve
    validation out of the remainder, so proportions are exact and stratified
    when configured.

    Parameters
    ----------
    x:
        Feature matrix.
    y:
        Target vector, or ``None`` for unsupervised datasets (SNMP).
    split_config:
        The ``data.split`` config block (sizes, stratify, shuffle).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(x_train, x_val, x_test, y_train, y_val, y_test)``; the ``y_*`` are
        ``None`` when ``y`` is ``None``.
    """
    _validate_ratios(
        float(split_config["train_size"]),
        float(split_config["val_size"]),
        float(split_config["test_size"]),
    )
    # TODO(data-engineer): two-stage sklearn train_test_split honouring
    #   stratify (only when y is not None) and shuffle; return six partitions.
    raise NotImplementedError
