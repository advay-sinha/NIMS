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
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class SplitReport:
    """Audit record for the train / validation / test split."""

    train_size: float
    val_size: float
    test_size: float
    stratified: bool
    shuffle: bool
    seed: int
    n_total: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


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
    from sklearn.model_selection import train_test_split

    train_size = float(split_config["train_size"])
    val_size = float(split_config["val_size"])
    test_size = float(split_config["test_size"])
    _validate_ratios(train_size, val_size, test_size)

    shuffle = bool(split_config.get("shuffle", True))
    stratify_enabled = bool(split_config.get("stratify", False)) and y is not None
    if stratify_enabled and not shuffle:
        raise ValueError("Stratified splitting requires shuffle=True.")

    # Stage 1: hold out the test partition.
    strat1 = y if stratify_enabled else None
    if y is None:
        x_rest, x_test = train_test_split(
            x, test_size=test_size, random_state=seed, shuffle=shuffle
        )
        y_rest = y_test = None
    else:
        x_rest, x_test, y_rest, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, shuffle=shuffle,
            stratify=strat1,
        )

    # Stage 2: carve validation out of the remainder (relative proportion).
    val_relative = val_size / (train_size + val_size)
    strat2 = y_rest if stratify_enabled else None
    if y is None:
        x_train, x_val = train_test_split(
            x_rest, test_size=val_relative, random_state=seed, shuffle=shuffle
        )
        y_train = y_val = None
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x_rest, y_rest, test_size=val_relative, random_state=seed,
            shuffle=shuffle, stratify=strat2,
        )

    logger.info(
        "Split %d rows -> train=%d val=%d test=%d (stratified=%s, seed=%d)",
        len(x), len(x_train), len(x_val), len(x_test), stratify_enabled, seed,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_split_report(
    y_train: "Any | None",
    y_val: "Any | None",
    y_test: "Any | None",
    split_config: Mapping[str, Any],
    seed: int,
) -> SplitReport:
    """Assemble a :class:`SplitReport` describing the produced partitions.

    Parameters
    ----------
    y_train, y_val, y_test:
        Target partitions (``None`` for unsupervised datasets).
    split_config:
        The ``data.split`` config block.
    seed:
        Seed used for the split.

    Returns
    -------
    SplitReport
    """
    report = SplitReport(
        train_size=float(split_config["train_size"]),
        val_size=float(split_config["val_size"]),
        test_size=float(split_config["test_size"]),
        stratified=bool(split_config.get("stratify", False)) and y_train is not None,
        shuffle=bool(split_config.get("shuffle", True)),
        seed=int(seed),
        n_train=0 if y_train is None else int(len(y_train)),
        n_val=0 if y_val is None else int(len(y_val)),
        n_test=0 if y_test is None else int(len(y_test)),
    )
    report.n_total = report.n_train + report.n_val + report.n_test
    report.class_distribution = {
        "train": _distribution(y_train),
        "validation": _distribution(y_val),
        "test": _distribution(y_test),
    }
    return report


def _distribution(y: "Any | None") -> dict[str, int]:
    """Return a ``{class: count}`` mapping for a target partition."""
    if y is None:
        return {}
    counts = y.value_counts(dropna=False)
    return {str(label): int(n) for label, n in counts.items()}
