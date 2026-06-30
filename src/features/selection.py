"""Statistical / model-based feature selection.

Purpose
-------
Turn per-feature scores (mutual information, chi-square, ANOVA, tree importance)
or Recursive Feature Elimination into a concrete, ordered list of selected
features. Selection is FIT ON TRAINING DATA ONLY and captured in a
:class:`FeatureSelector` artefact that simply re-selects the same columns on
validation/test (no leakage).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.features.anova import compute_anova
from src.features.chi_square import compute_chi_square
from src.features.importance import compute_tree_importance
from src.features.mutual_information import compute_mutual_information

logger = logging.getLogger(__name__)

_SCORE_METHODS = ("mutual_information", "chi_square", "anova", "tree_importance")
_VALID_METHODS = _SCORE_METHODS + ("rfe",)


@dataclass
class SelectionResult:
    """Outcome of statistical / model-based selection.

    Attributes
    ----------
    method:
        Selection method used.
    scores:
        ``{feature: score}`` (higher is better).
    ranking:
        Feature names ordered best-first.
    selected:
        Feature names retained (top-k or RFE support).
    """

    method: str
    scores: dict[str, float] = field(default_factory=dict)
    ranking: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        """Number of selected features."""
        return len(self.selected)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        data = asdict(self)
        data["n_selected"] = self.n_selected
        return data


@dataclass
class FeatureSelector:
    """Serialisable artefact reproducing the column selection on new data.

    Attributes
    ----------
    selected_features:
        Ordered feature names to retain (variance/correlation-removed columns
        are simply absent from this list).
    method:
        The statistical selection method that produced the list.
    """

    selected_features: list[str]
    method: str

    def transform(self, x: "Any") -> "Any":
        """Return ``x`` reduced to the selected feature columns (in order)."""
        cols = [c for c in self.selected_features if c in x.columns]
        return x[cols]


def _top_k(scores: Mapping[str, float], k: int | None) -> tuple[list[str], list[str]]:
    """Rank features by descending score; return ``(ranking, selected)``."""
    ranking = sorted(scores, key=lambda c: scores[c], reverse=True)
    selected = ranking[:k] if k else list(ranking)
    return ranking, selected


def select_features(
    x_train: "Any",
    y_train: "Any",
    features_config: Mapping[str, Any],
    seed: int = 42,
) -> SelectionResult:
    """Select features on the training data per the configured method.

    Parameters
    ----------
    x_train:
        Training feature DataFrame (already variance/correlation filtered).
    y_train:
        Training target vector.
    features_config:
        The ``features`` config block.
    seed:
        Random seed.

    Returns
    -------
    SelectionResult

    Raises
    ------
    ValueError
        If the configured selection method is unknown.
    """
    selection_cfg = features_config.get("selection", {})
    method = str(selection_cfg.get("method", "mutual_information"))
    k = selection_cfg.get("number_of_features")
    k = int(k) if k else None

    if method == "rfe":
        return _select_rfe(x_train, y_train, features_config, seed, k)
    if method not in _SCORE_METHODS:
        raise ValueError(f"Unknown selection method {method!r}; expected {_VALID_METHODS}.")

    if method == "mutual_information":
        scores = compute_mutual_information(x_train, y_train, seed)
    elif method == "chi_square":
        scores = compute_chi_square(x_train, y_train)
    elif method == "anova":
        scores = compute_anova(x_train, y_train)
    else:  # tree_importance
        scores = compute_tree_importance(
            x_train, y_train, features_config.get("importance", {}), seed
        ).importances

    ranking, selected = _top_k(scores, k)
    logger.info("Selection '%s': retained %d of %d feature(s).",
                method, len(selected), x_train.shape[1])
    return SelectionResult(method=method, scores=scores, ranking=ranking, selected=selected)


def _select_rfe(
    x_train: "Any",
    y_train: "Any",
    features_config: Mapping[str, Any],
    seed: int,
    k: int | None,
) -> SelectionResult:
    """Recursive Feature Elimination with a RandomForest estimator."""
    from sklearn.feature_selection import RFE

    from src.features.importance import build_random_forest

    rfe_cfg = features_config.get("rfe", {})
    n_features = k if k else max(1, x_train.shape[1] // 2)
    estimator = build_random_forest(
        {"n_estimators": rfe_cfg.get("n_estimators", 100)}, seed
    )
    selector = RFE(
        estimator,
        n_features_to_select=n_features,
        step=int(rfe_cfg.get("step", 1)),
    )
    selector.fit(x_train, y_train)

    columns = [str(c) for c in x_train.columns]
    rank_by_col = dict(zip(columns, selector.ranking_))
    # RFE ranking: 1 = selected; smaller is better. Convert to a "higher=better"
    # score so the report ordering is consistent across methods.
    scores = {col: float(1.0 / rank_by_col[col]) for col in columns}
    ranking = sorted(columns, key=lambda c: rank_by_col[c])
    selected = [col for col, keep in zip(columns, selector.support_) if keep]
    logger.info("RFE: retained %d of %d feature(s).", len(selected), len(columns))
    return SelectionResult(method="rfe", scores=scores, ranking=ranking, selected=selected)
