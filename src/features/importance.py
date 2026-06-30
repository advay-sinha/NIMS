"""Tree-based feature importance.

Purpose
-------
Score features by RandomForest impurity-based importance. The estimator is
defined here but fitting happens only when the pipeline is executed by the
human operator (Human-in-the-Loop). Computed on training data only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class ImportanceResult:
    """Tree-based importance scores and the derived ranking.

    Attributes
    ----------
    importances:
        ``{feature: importance}``.
    ranking:
        Feature names ordered by descending importance.
    """

    importances: dict[str, float] = field(default_factory=dict)
    ranking: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


def build_random_forest(config: Mapping[str, Any], seed: int) -> "Any":
    """Construct (but do not fit) a RandomForest classifier from config.

    Parameters
    ----------
    config:
        The ``features.importance`` block (``n_estimators``, ``max_depth``).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 200)),
        max_depth=config.get("max_depth"),
        random_state=seed,
        n_jobs=-1,
    )


def compute_tree_importance(
    x_train: "Any",
    y_train: "Any",
    config: Mapping[str, Any],
    seed: int = 42,
) -> ImportanceResult:
    """Fit a RandomForest and return impurity-based feature importances.

    Parameters
    ----------
    x_train:
        Training feature DataFrame.
    y_train:
        Training target vector.
    config:
        The ``features.importance`` config block.
    seed:
        Random seed.

    Returns
    -------
    ImportanceResult
    """
    model = build_random_forest(config, seed)
    model.fit(x_train, y_train)
    importances = {
        str(col): float(imp)
        for col, imp in zip(x_train.columns, model.feature_importances_)
    }
    ranking = sorted(importances, key=lambda c: importances[c], reverse=True)
    logger.info("Computed RandomForest importance for %d feature(s).", len(importances))
    return ImportanceResult(importances=importances, ranking=ranking)
