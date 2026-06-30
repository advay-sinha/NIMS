"""Numerical scaling stage.

Purpose
-------
Scale numerical features reproducibly. The scaler is FIT ON TRAINING DATA ONLY
and reused everywhere else (CLAUDE.md > Machine Learning Standards; skills/
preprocessing: "Scaling — fit only on training data. Reuse scaler for
validation and inference"). Fitted scalers are persisted via
:mod:`src.utils.io`.

Supported strategies: ``standard`` | ``minmax`` | ``robust`` | ``none``.

Inputs
------
- Train / val / test feature DataFrames and the ``data.scaling`` config block.

Outputs
-------
- Scaled DataFrames plus a fitted, serialisable scaler and a
  :class:`ScalingReport`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_SCALER_FACTORIES = {
    "standard": "StandardScaler",
    "minmax": "MinMaxScaler",
    "robust": "RobustScaler",
}


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
        The underlying fitted transformer, or ``None`` for the identity
        (``"none"``) strategy.
    """

    strategy: str
    columns: tuple[str, ...]
    scaler: Any


@dataclass
class ScalingReport:
    """Audit record for the scaling stage."""

    strategy: str
    scaled_columns: list[str] = field(default_factory=list)
    n_scaled_features: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


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
        Columns to scale (only those present in ``x_train`` are used).
    scaling_config:
        The ``data.scaling`` config block (``numeric_strategy``).

    Returns
    -------
    FittedScaler

    Raises
    ------
    ValueError
        If the configured strategy is unknown.
    """
    strategy = str(scaling_config.get("numeric_strategy", "standard"))
    columns = tuple(c for c in numeric_columns if c in x_train.columns)

    if strategy == "none" or not columns:
        logger.info("Scaling strategy '%s' is a no-op (%d cols).", strategy, len(columns))
        return FittedScaler(strategy=strategy, columns=columns, scaler=None)

    if strategy not in _SCALER_FACTORIES:
        raise ValueError(
            f"Unknown numeric_strategy {strategy!r}; "
            f"expected {tuple(_SCALER_FACTORIES) + ('none',)}."
        )

    scaler = _build_scaler(strategy)
    scaler.fit(x_train[list(columns)])
    logger.info("Fitted %s scaler on %d column(s).", strategy, len(columns))
    return FittedScaler(strategy=strategy, columns=columns, scaler=scaler)


def _build_scaler(strategy: str) -> "Any":
    """Construct an unfitted sklearn scaler for ``strategy``."""
    import sklearn.preprocessing as skp

    return getattr(skp, _SCALER_FACTORIES[strategy])()


def apply_scaler(scaler: FittedScaler, x: "Any") -> "Any":
    """Transform features with a previously fitted scaler.

    Non-target numeric columns named in the scaler are transformed; all other
    columns pass through unchanged.

    Parameters
    ----------
    scaler:
        Result of :func:`fit_scaler`.
    x:
        Feature DataFrame (train, val, test or inference).

    Returns
    -------
    pandas.DataFrame
        Scaled copy.
    """
    # No-op strategy: nothing is mutated, so return the original object.
    if scaler.scaler is None or not scaler.columns:
        return x
    columns = [c for c in scaler.columns if c in x.columns]
    if not columns:
        return x
    # Shallow copy shares unscaled column blocks with ``x``; assigning the
    # scaled columns replaces only those blocks (Copy-on-Write), so the large
    # passthrough columns are never duplicated.
    result = x.copy(deep=False)
    result[columns] = scaler.scaler.transform(x[columns])
    return result
