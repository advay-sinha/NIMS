"""Categorical encoding stage.

Purpose
-------
Encode categorical columns reproducibly. The encoder is FIT ON TRAINING DATA
ONLY and reused for validation / test / inference to prevent leakage
(CLAUDE.md > skills/preprocessing: "Use reproducible encoding";
feature_engineering: "Avoid data leakage"). Fitted encoders are persisted via
:mod:`src.utils.io` so inference uses identical transforms.

Inputs
------
- Train / val / test DataFrames.
- The ``data.encoding`` config block and the list of categorical columns.

Outputs
-------
- Encoded DataFrames and a fitted, serialisable encoder object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass
class FittedEncoder:
    """Container for a fitted categorical encoder and its metadata.

    Attributes
    ----------
    strategy:
        ``"onehot"`` | ``"ordinal"`` | ``"target"``.
    columns:
        Categorical columns the encoder was fit on.
    encoder:
        The underlying fitted transformer (e.g. sklearn ``OneHotEncoder``).
    """

    strategy: str
    columns: tuple[str, ...]
    encoder: Any


def fit_encoder(
    x_train: "Any",
    categorical_columns: Sequence[str],
    encoding_config: Mapping[str, Any],
) -> FittedEncoder:
    """Fit a categorical encoder on training features only.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    categorical_columns:
        Columns to encode.
    encoding_config:
        The ``data.encoding`` config block (strategy, handle_unknown).

    Returns
    -------
    FittedEncoder
    """
    # TODO(data-engineer): build the encoder per strategy with
    #   handle_unknown=ignore; fit on x_train[categorical_columns].
    raise NotImplementedError


def apply_encoder(encoder: FittedEncoder, x: "Any") -> "Any":
    """Transform features with a previously fitted encoder.

    Parameters
    ----------
    encoder:
        Result of :func:`fit_encoder`.
    x:
        Feature DataFrame (train, val, test or inference).

    Returns
    -------
    pandas.DataFrame
        Encoded features with original numeric columns preserved.
    """
    # TODO(data-engineer): transform categorical cols, concat with numeric;
    #   keep deterministic column ordering.
    raise NotImplementedError
