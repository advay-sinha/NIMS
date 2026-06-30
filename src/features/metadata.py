"""Feature/target separation and shared feature metadata.

Purpose
-------
Small, dependency-light helpers shared by the feature-engineering stages:
splitting a processed frame into features/target and the final
:class:`FeatureMetadata` describing what was retained.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Provenance / metadata columns added during ingestion (e.g. the split tag,
# CICIDS source-file tag) or secondary label columns. They are never features
# and must be excluded before variance/correlation/selection. Matched
# case-insensitively against the exact column name.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "split",
    "source",
    "source_file",
    "dataset",
    "origin",
    "attack_cat",
)


def split_xy(frame: "Any", label_column: str) -> tuple["Any", "Any"]:
    """Split a processed frame into a feature matrix and target vector.

    Parameters
    ----------
    frame:
        Processed DataFrame containing features and the label column.
    label_column:
        Name of the target column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        ``(X, y)``.

    Raises
    ------
    KeyError
        If ``label_column`` is absent from ``frame``.
    """
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' not found in processed frame.")
    y = frame[label_column]
    x = frame.drop(columns=[label_column])
    return x, y


def select_feature_columns(
    x: "Any",
    extra_exclude: Iterable[str] = (),
) -> tuple["Any", dict[str, list[str]]]:
    """Reduce a feature matrix to numeric feature columns only.

    Drops provenance/metadata columns (by name, case-insensitive) and any
    remaining non-numeric columns, so downstream statistical selection only ever
    receives numeric features (labels/metadata are never part of ``X``).

    Parameters
    ----------
    x:
        Feature DataFrame (label already removed via :func:`split_xy`).
    extra_exclude:
        Additional column names to exclude (e.g. ``features.exclude_columns``).

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, list[str]]]
        The numeric-only feature frame and a mapping describing what was
        excluded: ``{"provenance": [...], "non_numeric": [...]}``.
    """
    import numpy as np

    exclude_names = {name.lower() for name in PROVENANCE_COLUMNS}
    exclude_names.update(str(c).lower() for c in extra_exclude)

    provenance = [c for c in x.columns if str(c).lower() in exclude_names]
    remaining = x.drop(columns=provenance) if provenance else x

    numeric_cols = list(remaining.select_dtypes(include=[np.number]).columns)
    numeric_set = set(numeric_cols)
    non_numeric = [str(c) for c in remaining.columns if c not in numeric_set]

    excluded = {
        "provenance": [str(c) for c in provenance],
        "non_numeric": non_numeric,
    }
    return remaining[numeric_cols], excluded


@dataclass
class FeatureMetadata:
    """Summary of the feature set before and after engineering.

    Attributes
    ----------
    dataset_id:
        Dataset identifier.
    label_column:
        Target column name.
    original_features:
        Feature names present in the processed input.
    retained_features:
        Feature names retained after selection (pre-PCA), or the PCA component
        names when PCA is enabled.
    pca_components:
        Number of PCA components, or ``None`` when PCA is disabled.
    """

    dataset_id: str
    label_column: str
    original_features: list[str] = field(default_factory=list)
    retained_features: list[str] = field(default_factory=list)
    pca_components: int | None = None

    @property
    def n_original(self) -> int:
        """Number of original features."""
        return len(self.original_features)

    @property
    def n_retained(self) -> int:
        """Number of retained features (post selection / PCA)."""
        return len(self.retained_features)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        data = asdict(self)
        data["n_original"] = self.n_original
        data["n_retained"] = self.n_retained
        return data
