"""Pre-fit feature-matrix audit.

Purpose
-------
Verify — before every model fit — that the loaded training matrix is exactly
the feature-engineered output: same columns (names AND order) as recorded by
feature engineering, numeric dtypes only, no missing values, no duplicated
columns, and identical column layout across train/validation/test. Catches
silent drift between Layer 2 (feature engineering) and Layer 3 (training)
without hardcoding any dataset assumption: the expected feature list is read
from the per-dataset feature-engineering artefacts.

Inputs
------
The feature directory (``outputs/features/<dataset>/``) and the loaded
``(X, y)`` splits.

Outputs
-------
``None`` on success (with an INFO summary per split); raises
:class:`FeatureAuditError` describing the first violation found.

Limitations
-----------
When neither ``feature_metadata.json`` nor ``selected_features.json`` exists
(e.g. synthetic fixtures), the expected-column check is skipped; structural
checks (dtypes, NaNs, duplicates, cross-split consistency) still run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class FeatureAuditError(ValueError):
    """Raised when the training matrix does not match the engineered features."""


def load_expected_features(feat_dir: Path) -> list[str] | None:
    """Return the final engineered feature names for a dataset, if recorded.

    Prefers ``feature_metadata.json`` (``retained_features`` holds the PCA
    component names when PCA is enabled, so it always reflects the actual
    columns of the persisted splits) and falls back to
    ``selected_features.json`` (``selected``).

    The recorded names are passed through the same canonical provenance
    filter (:func:`src.features.metadata.filter_provenance_names`) that the
    trainer applies to the loaded matrix, so the audit's expected list and
    the training matrix are derived from one predicate. This also repairs
    stale artefacts written before the feature pipeline excluded provenance
    columns (e.g. an encoded ``source_file`` recorded as a feature): such
    columns are never model features, and any dropped name is logged.

    Parameters
    ----------
    feat_dir:
        Per-dataset feature-engineering output directory.

    Returns
    -------
    list[str] | None
        Expected model-feature names in order, or ``None`` when no artefact
        exists.
    """
    from src.utils.io import read_json

    recorded: list[str] | None = None
    metadata_path = feat_dir / "feature_metadata.json"
    if metadata_path.is_file():
        retained = read_json(metadata_path).get("retained_features")
        if retained:
            recorded = [str(name) for name in retained]

    if recorded is None:
        selected_path = feat_dir / "selected_features.json"
        if selected_path.is_file():
            selected = read_json(selected_path).get("selected")
            if selected:
                recorded = [str(name) for name in selected]

    if recorded is None:
        return None

    from src.features.metadata import filter_provenance_names

    expected = filter_provenance_names(recorded)
    dropped = sorted(set(recorded) - set(expected))
    if dropped:
        logger.warning(
            "Feature artefacts in %s record provenance column(s) %s as "
            "features (stale metadata from an older feature-engineering run); "
            "excluded from the expected model features. Re-run feature "
            "engineering to refresh the artefacts.", feat_dir, dropped,
        )
    return expected


def audit_features(
    splits: Mapping[str, tuple["Any", "Any"]],
    expected_features: list[str] | None,
    dataset_id: str,
    model_name: str,
) -> None:
    """Audit every split's feature matrix before fitting.

    Parameters
    ----------
    splits:
        Mapping of split name -> ``(X, y)``.
    expected_features:
        Feature names (in order) recorded by feature engineering, or ``None``
        to skip the expected-column comparison.
    dataset_id, model_name:
        Context for error messages and logs.

    Raises
    ------
    FeatureAuditError
        On any column-name/order mismatch, non-numeric dtype, missing value or
        duplicated column.
    """
    context = f"[{dataset_id}/{model_name}]"
    train_columns = [str(c) for c in splits["train"][0].columns]

    if expected_features is not None and train_columns != expected_features:
        raise FeatureAuditError(
            f"{context} X_train columns do not match the feature-engineering "
            f"output (expected {len(expected_features)}, got "
            f"{len(train_columns)}). missing={sorted(set(expected_features) - set(train_columns))} "
            f"unexpected={sorted(set(train_columns) - set(expected_features))} "
            f"order_mismatch={set(train_columns) == set(expected_features)}"
        )

    for split_name, (x, _y) in splits.items():
        _audit_split(x, train_columns, split_name, context)
        logger.info(
            "%s feature audit OK: %s shape=%s columns=%d dtypes=numeric "
            "missing=0 duplicates=0",
            context, split_name, tuple(x.shape), x.shape[1],
        )


def _audit_split(
    x: "Any", train_columns: list[str], split_name: str, context: str
) -> None:
    """Run the structural checks on one split's feature matrix."""
    from pandas.api.types import is_numeric_dtype

    columns = [str(c) for c in x.columns]
    if columns != train_columns:
        raise FeatureAuditError(
            f"{context} {split_name} columns/order differ from train: "
            f"{columns[:5]}... vs {train_columns[:5]}..."
        )

    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise FeatureAuditError(
            f"{context} {split_name} has duplicated feature columns: {duplicates}"
        )

    non_numeric = [
        str(c) for c, dtype in x.dtypes.items() if not is_numeric_dtype(dtype)
    ]
    if non_numeric:
        raise FeatureAuditError(
            f"{context} {split_name} has non-numeric feature columns: {non_numeric}"
        )

    missing = x.isna().sum()
    with_missing = {str(c): int(n) for c, n in missing.items() if n > 0}
    if with_missing:
        raise FeatureAuditError(
            f"{context} {split_name} has missing values: {with_missing}"
        )
