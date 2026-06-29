"""Data cleaning stage.

Purpose
-------
Transform a validated raw DataFrame into a cleaned DataFrame: handle missing
values, duplicates, constant columns and infinities. Strategy is driven by the
``data.cleaning`` configuration and documented (CLAUDE.md > Dataset Rules:
"Missing Values — document strategy"). Raw inputs are never mutated in place;
a cleaned copy is returned.

Inputs
------
- A raw ``pandas.DataFrame``.
- The ``data.cleaning`` config block.

Outputs
-------
- A cleaned ``pandas.DataFrame`` (new object).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def replace_infinities(frame: "Any") -> "Any":
    """Replace ``+/-inf`` with NaN so they flow into imputation.

    Common in CICIDS2017 flow-ratio features.

    Parameters
    ----------
    frame:
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Copy with infinities replaced by NaN.
    """
    # TODO(data-engineer): frame.replace([np.inf, -np.inf], np.nan) on a copy.
    raise NotImplementedError


def drop_duplicate_rows(frame: "Any") -> "Any":
    """Return a copy with exact duplicate rows removed."""
    # TODO(data-engineer): frame.drop_duplicates(); log the count removed.
    raise NotImplementedError


def drop_constant_columns(frame: "Any") -> "Any":
    """Return a copy with zero-variance (constant) columns removed.

    Constant columns carry no signal and can break some scalers.
    """
    # TODO(data-engineer): drop columns with nunique() <= 1; log dropped names.
    raise NotImplementedError


def impute_missing(frame: "Any", cleaning_config: Mapping[str, Any]) -> "Any":
    """Impute missing values per the configured strategy.

    Numeric columns use ``numeric_impute`` (median/mean/zero/drop);
    categorical columns use ``categorical_impute`` (most_frequent).

    Parameters
    ----------
    frame:
        Input DataFrame (post infinity replacement).
    cleaning_config:
        The ``data.cleaning`` config block.

    Returns
    -------
    pandas.DataFrame
    """
    # TODO(data-engineer): split numeric/categorical, apply configured
    #   imputation; document chosen strategy in metadata.
    raise NotImplementedError


def clean_dataset(frame: "Any", cleaning_config: Mapping[str, Any]) -> "Any":
    """Run the full cleaning stage in order.

    Order: replace infinities -> drop duplicates -> drop constant columns ->
    impute missing.

    Parameters
    ----------
    frame:
        Raw (validated) DataFrame.
    cleaning_config:
        The ``data.cleaning`` config block.

    Returns
    -------
    pandas.DataFrame
        Cleaned copy.
    """
    # TODO(data-engineer): compose the steps above guarded by config flags.
    raise NotImplementedError
