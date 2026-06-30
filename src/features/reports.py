"""Feature-engineering report builders.

Purpose
-------
Render the stage results into the JSON reports the pipeline persists:
``feature_report.json``, ``feature_metadata.json``, ``selected_features.json``
and ``removed_features.json``. Pure functions returning JSON-serialisable dicts.
"""

from __future__ import annotations

from typing import Any

from src.features.correlation import CorrelationResult
from src.features.dimensionality import FittedPCA
from src.features.metadata import FeatureMetadata
from src.features.selection import SelectionResult
from src.features.variance import VarianceResult


def build_feature_report(
    dataset_id: str,
    original_features: list[str],
    variance: VarianceResult | None,
    correlation: CorrelationResult | None,
    selection: SelectionResult | None,
    pca: FittedPCA | None,
    retained_features: list[str],
    excluded_columns: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Assemble the headline feature report.

    Contains original count, excluded provenance/non-numeric columns, removals
    by variance/correlation, selected features with ranking + importance,
    retained count and PCA information.
    """
    removed_variance = variance.removed if variance else []
    removed_correlation = correlation.removed if correlation else []
    return {
        "dataset_id": dataset_id,
        "original_feature_count": len(original_features),
        "excluded_columns": excluded_columns or {"provenance": [], "non_numeric": []},
        "removed_by_variance": removed_variance,
        "removed_by_variance_count": len(removed_variance),
        "removed_by_correlation": removed_correlation,
        "removed_by_correlation_count": len(removed_correlation),
        "selection_method": selection.method if selection else None,
        "selected_features": selection.selected if selection else retained_features,
        "ranking": selection.ranking if selection else [],
        "importance": selection.scores if selection else {},
        "retained_feature_count": len(retained_features),
        "pca": pca.to_dict() if pca else None,
    }


def build_feature_metadata(metadata: FeatureMetadata) -> dict[str, Any]:
    """Return the feature metadata mapping."""
    return metadata.to_dict()


def build_selected_features(
    selection: SelectionResult | None,
    retained_features: list[str],
) -> dict[str, Any]:
    """Return the selected-features report (method, ranking, scores)."""
    if selection is None:
        return {"method": None, "selected": retained_features, "ranking": [], "scores": {}}
    return {
        "method": selection.method,
        "selected": selection.selected,
        "n_selected": selection.n_selected,
        "ranking": selection.ranking,
        "scores": selection.scores,
    }


def build_removed_features(
    variance: VarianceResult | None,
    correlation: CorrelationResult | None,
) -> dict[str, Any]:
    """Return the removed-features report grouped by stage."""
    return {
        "variance": {
            "removed": variance.removed if variance else [],
            "threshold": variance.threshold if variance else None,
        },
        "correlation": {
            "removed": correlation.removed if correlation else [],
            "threshold": correlation.threshold if correlation else None,
            "method": correlation.method if correlation else None,
            "pairs": correlation.pairs if correlation else [],
        },
    }
