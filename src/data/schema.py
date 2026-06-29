"""Dataset and feature metadata models.

Purpose
-------
Typed containers describing a dataset's schema and provenance. Persisted
alongside processed data so every artefact is self-describing and
reproducible (Phase 1 outputs: metadata, validation report).

These are plain dataclasses with no heavy dependencies; they are serialised to
JSON via :mod:`src.utils.io`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureKind(str, Enum):
    """Coarse feature category used to drive encoding / scaling decisions."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    LABEL = "label"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class FeatureMetadata:
    """Description of a single column.

    Attributes
    ----------
    name:
        Column name (post whitespace-normalisation).
    kind:
        Feature category (:class:`FeatureKind`).
    dtype:
        Pandas/numpy dtype string.
    n_missing:
        Count of missing values observed during profiling.
    n_unique:
        Count of distinct values.
    notes:
        Free-form provenance / engineering rationale.
    """

    name: str
    kind: FeatureKind
    dtype: str | None = None
    n_missing: int | None = None
    n_unique: int | None = None
    notes: str | None = None


@dataclass
class DatasetMetadata:
    """Self-describing metadata for a processed dataset.

    Attributes
    ----------
    dataset_id:
        Stable identifier (e.g. ``"nsl_kdd"``).
    name:
        Human-readable name.
    engine:
        ``"A"`` (cyber) or ``"B"`` (network health).
    n_rows / n_features:
        Shape after preprocessing.
    label_column:
        Target column name, or ``None`` for unsupervised telemetry.
    features:
        Per-column metadata.
    split_sizes:
        Row counts per split, e.g. ``{"train": .., "val": .., "test": ..}``.
    source_files:
        Raw files the dataset was built from (for provenance).
    config_snapshot:
        Snapshot of the dataset config used (reproducibility).
    """

    dataset_id: str
    name: str
    engine: str
    n_rows: int = 0
    n_features: int = 0
    label_column: str | None = None
    features: list[FeatureMetadata] = field(default_factory=list)
    split_sizes: dict[str, int] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Enum values are coerced to their string form so the result is safe to
        pass directly to :func:`src.utils.io.write_json`.

        Returns
        -------
        dict
        """
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "engine": self.engine,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "label_column": self.label_column,
            "features": [
                {
                    "name": f.name,
                    "kind": f.kind.value,
                    "dtype": f.dtype,
                    "n_missing": f.n_missing,
                    "n_unique": f.n_unique,
                    "notes": f.notes,
                }
                for f in self.features
            ],
            "split_sizes": dict(self.split_sizes),
            "source_files": list(self.source_files),
            "config_snapshot": dict(self.config_snapshot),
        }


def infer_feature_kind(
    column: str,
    dtype: str,
    *,
    label_column: str | None = None,
    categorical_columns: tuple[str, ...] = (),
    identifier_columns: tuple[str, ...] = (),
    timestamp_columns: tuple[str, ...] = (),
) -> FeatureKind:
    """Classify a column into a :class:`FeatureKind`.

    Resolution order: explicit roles (label, identifier, timestamp,
    categorical) take precedence over dtype-based inference.

    Parameters
    ----------
    column:
        Column name.
    dtype:
        Pandas/numpy dtype string for the column.
    label_column, categorical_columns, identifier_columns, timestamp_columns:
        Known role assignments from the dataset config.

    Returns
    -------
    FeatureKind
    """
    if label_column is not None and column == label_column:
        return FeatureKind.LABEL
    if column in identifier_columns:
        return FeatureKind.IDENTIFIER
    if column in timestamp_columns:
        return FeatureKind.TIMESTAMP
    if column in categorical_columns:
        return FeatureKind.CATEGORICAL
    if "datetime" in dtype:
        return FeatureKind.TIMESTAMP
    if dtype == "bool":
        return FeatureKind.BINARY
    if dtype.startswith(("int", "float", "uint")):
        return FeatureKind.NUMERIC
    return FeatureKind.CATEGORICAL
