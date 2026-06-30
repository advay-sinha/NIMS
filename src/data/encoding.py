"""Categorical encoding stage.

Purpose
-------
Encode categorical features and the target label reproducibly. Encoders are FIT
ON TRAINING DATA ONLY and reused for validation / test / inference to prevent
leakage (CLAUDE.md > Machine Learning Standards; feature_engineering: "Avoid
data leakage"). Fitted encoders are persisted via :mod:`src.utils.io` so
inference uses identical transforms.

Supported feature strategies:
    - ``onehot``  : sklearn ``OneHotEncoder`` (handle_unknown=ignore),
    - ``ordinal`` : sklearn ``OrdinalEncoder`` (unknown -> -1).

The target column is encoded separately with a label encoder (labels only).

Inputs
------
- Train / val / test feature DataFrames and target Series.
- The ``data.encoding`` config block and the list of categorical columns.

Outputs
-------
- Encoded DataFrames / arrays plus fitted, serialisable encoder objects and a
  :class:`EncodingReport`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = ("onehot", "ordinal")


@dataclass
class FittedEncoder:
    """Container for a fitted categorical feature encoder and its metadata.

    Attributes
    ----------
    strategy:
        ``"onehot"`` | ``"ordinal"``.
    columns:
        Categorical columns the encoder was fit on.
    encoder:
        The underlying fitted sklearn transformer.
    feature_names_out:
        Output feature names produced by the encoder (deterministic order).
    """

    strategy: str
    columns: tuple[str, ...]
    encoder: Any
    feature_names_out: tuple[str, ...]


@dataclass
class FittedLabelEncoder:
    """Container for a fitted label encoder (target column only).

    Unknown labels seen at transform time map to ``-1`` (safe, never raises).
    """

    column: str
    encoder: Any
    classes: tuple[str, ...]


@dataclass
class EncodingReport:
    """Audit record for the encoding stage."""

    strategy: str
    encoded_columns: list[str] = field(default_factory=list)
    n_input_categorical: int = 0
    n_output_features: int = 0
    label_column: str | None = None
    label_classes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Feature encoding                                                            #
# --------------------------------------------------------------------------- #
def fit_encoder(
    x_train: "Any",
    categorical_columns: Sequence[str],
    encoding_config: Mapping[str, Any],
) -> FittedEncoder:
    """Fit a categorical feature encoder on training features only.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    categorical_columns:
        Columns to encode (only those present in ``x_train`` are used).
    encoding_config:
        The ``data.encoding`` config block (``categorical_strategy``,
        ``handle_unknown``).

    Returns
    -------
    FittedEncoder

    Raises
    ------
    ValueError
        If the configured strategy is unknown.
    """
    strategy = str(encoding_config.get("categorical_strategy", "onehot"))
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown categorical_strategy {strategy!r}; expected {_VALID_STRATEGIES}."
        )

    columns = tuple(c for c in categorical_columns if c in x_train.columns)
    if not columns:
        logger.info("No categorical columns to encode; encoder is a no-op.")
        return FittedEncoder(strategy=strategy, columns=(), encoder=None,
                             feature_names_out=())

    encoder = _build_encoder(strategy, encoding_config)
    encoder.fit(x_train[list(columns)])
    feature_names = tuple(str(c) for c in encoder.get_feature_names_out(columns))
    logger.info(
        "Fitted %s encoder on %d column(s) -> %d output feature(s).",
        strategy,
        len(columns),
        len(feature_names),
    )
    return FittedEncoder(
        strategy=strategy,
        columns=columns,
        encoder=encoder,
        feature_names_out=feature_names,
    )


def _build_encoder(strategy: str, encoding_config: Mapping[str, Any]) -> "Any":
    """Construct an unfitted sklearn encoder for ``strategy``."""
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    handle_unknown = str(encoding_config.get("handle_unknown", "ignore"))
    if strategy == "onehot":
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    # ordinal: map unseen categories to a sentinel rather than raising.
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def apply_encoder(encoder: FittedEncoder, x: "Any") -> "Any":
    """Transform features with a previously fitted encoder.

    Categorical columns are replaced by their encoded representation; all other
    (numeric) columns pass through unchanged. Column ordering is deterministic.

    Parameters
    ----------
    encoder:
        Result of :func:`fit_encoder`.
    x:
        Feature DataFrame (train, val, test or inference).

    Returns
    -------
    pandas.DataFrame
        Encoded features.
    """
    import pandas as pd

    # No categoricals -> nothing to transform; return the original object. The
    # caller never mutates it in place (Copy-on-Write protects downstream
    # stages), so an explicit copy here would needlessly duplicate the frame.
    if not encoder.columns:
        return x

    columns = list(encoder.columns)
    passthrough = x.drop(columns=columns)
    transformed = encoder.encoder.transform(x[columns])

    if encoder.strategy == "onehot":
        encoded = pd.DataFrame(
            transformed,
            columns=list(encoder.feature_names_out),
            index=x.index,
        )
    else:  # ordinal: one column per input column, original names preserved.
        encoded = pd.DataFrame(transformed, columns=columns, index=x.index)

    # Single concat builds the final frame once; release the operands promptly.
    result = pd.concat([passthrough, encoded], axis=1)
    del passthrough, encoded, transformed
    return result


# --------------------------------------------------------------------------- #
# Label encoding                                                              #
# --------------------------------------------------------------------------- #
def fit_label_encoder(y_train: "Any", column: str | None = None) -> FittedLabelEncoder:
    """Fit a label encoder on the training target only.

    Parameters
    ----------
    y_train:
        Training target Series.
    column:
        Name of the target column (for reporting); defaults to ``y_train.name``.

    Returns
    -------
    FittedLabelEncoder
    """
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(y_train.astype(str))
    classes = tuple(str(c) for c in encoder.classes_)
    name = column if column is not None else getattr(y_train, "name", None)
    logger.info("Fitted label encoder on '%s' with %d class(es).", name, len(classes))
    return FittedLabelEncoder(column=str(name), encoder=encoder, classes=classes)


def apply_label_encoder(label_encoder: FittedLabelEncoder, y: "Any") -> "Any":
    """Encode a target Series, mapping unknown labels to ``-1`` safely.

    Parameters
    ----------
    label_encoder:
        Result of :func:`fit_label_encoder`.
    y:
        Target Series to encode.

    Returns
    -------
    numpy.ndarray
        Integer-encoded labels (unseen classes -> ``-1``).
    """
    import numpy as np

    known = set(label_encoder.classes)
    values = y.astype(str)
    mask = values.isin(known)
    encoded = np.full(len(values), -1, dtype=np.int64)
    if mask.any():
        encoded[mask.to_numpy()] = label_encoder.encoder.transform(values[mask])
    n_unknown = int((~mask).sum())
    if n_unknown:
        logger.warning("Label encoder saw %d unknown label(s) -> -1.", n_unknown)
    return encoded
