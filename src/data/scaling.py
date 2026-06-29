"""Numerical scaling stage.

Purpose
-------
Scale numerical features reproducibly. The scaler is FIT ON TRAINING DATA ONLY
and reused everywhere else (CLAUDE.md > skills/preprocessing: "Scaling — fit
only on training data. Reuse scaler for validation and inference"). Fitted
scalers are persisted via :mod:`src.utils.io`.

Inputs
------
- Train / val / test feature DataFrames.
- The ``data.scaling`` config block.

Outputs
-------
- Scaled DataFrames and a fitted, serialisable scaler object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass
class FittedScaler:
    """Container for a fitted numerical scaler and its metadata.

    Attributes
    ----------
    strategy:
        ``"standard"`` | ``"minmax"`` | ``"robust"`` | ``"none"``.
    columns:
        Numeric columns the scaler was fit on.
    scaler:
        The underlying fitted transformer (e.g. sklearn ``StandardScaler``).
    """

    strategy: str
    columns: tuple[str, ...]
    scaler: Any


def fit_scaler(
    x_train: "Any",
    numeric_columns: Sequence[str],
    scaling_config: Mapping[str, Any],
) -> FittedScaler:
    """Fit a numerical scaler on training features only.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    numeric_columns:
        Columns to scale.
    scaling_config:
        The ``data.scaling`` config block (strategy).

    Returns
    -------
    FittedScaler
    """
    # TODO(data-engineer): select scaler per strategy ("none" => identity);
    #   fit on x_train[numeric_columns].
    raise NotImplementedError


def apply_scaler(scaler: FittedScaler, x: "Any") -> "Any":
    """Transform features with a previously fitted scaler.

    Parameters
    ----------
    scaler:
        Result of :func:`fit_scaler`.
    x:
        Feature DataFrame (train, val, test or inference).

    Returns
    -------
    pandas.DataFrame
        Scaled copy; non-numeric columns passed through unchanged.
    """
    # TODO(data-engineer): transform numeric cols in place on a copy.
    raise NotImplementedError
