"""Optional PCA dimensionality reduction.

Purpose
-------
Reduce the selected feature set to principal components preserving a
configurable fraction of explained variance. PCA is FIT ON TRAINING DATA ONLY
and reused to transform validation/test (no leakage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class FittedPCA:
    """A fitted PCA transform and its metadata.

    Attributes
    ----------
    pca:
        The underlying fitted sklearn ``PCA``.
    input_columns:
        Feature names the PCA was fit on (transform input order).
    component_names:
        Output component column names (``pc_1`` ...).
    n_components:
        Number of retained components.
    explained_variance_ratio:
        Per-component explained variance ratio.
    cumulative_variance:
        Total explained variance across retained components.
    """

    pca: Any
    input_columns: list[str]
    component_names: list[str]
    n_components: int
    explained_variance_ratio: list[float] = field(default_factory=list)
    cumulative_variance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (excludes the estimator)."""
        return {
            "n_components": self.n_components,
            "input_columns": self.input_columns,
            "component_names": self.component_names,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": self.cumulative_variance,
        }


def fit_pca(
    x_train: "Any",
    pca_config: Mapping[str, Any],
    seed: int = 42,
) -> FittedPCA | None:
    """Fit PCA on training features when enabled.

    Parameters
    ----------
    x_train:
        Training feature DataFrame (post selection).
    pca_config:
        The ``features.dimensionality.pca`` config block (``enabled``,
        ``explained_variance``).
    seed:
        Random seed (used by randomised SVD solvers).

    Returns
    -------
    FittedPCA | None
        ``None`` when PCA is disabled.
    """
    if not pca_config.get("enabled", False):
        return None

    from sklearn.decomposition import PCA

    explained_variance = float(pca_config.get("explained_variance", 0.95))
    pca = PCA(n_components=explained_variance, random_state=seed)
    pca.fit(x_train)

    n_components = int(pca.n_components_)
    names = [f"pc_{i + 1}" for i in range(n_components)]
    ratios = [float(v) for v in pca.explained_variance_ratio_]
    logger.info(
        "PCA: %d component(s) retain %.4f of variance (target %.2f).",
        n_components, sum(ratios), explained_variance,
    )
    return FittedPCA(
        pca=pca,
        input_columns=[str(c) for c in x_train.columns],
        component_names=names,
        n_components=n_components,
        explained_variance_ratio=ratios,
        cumulative_variance=float(sum(ratios)),
    )


def apply_pca(fitted: FittedPCA, x: "Any") -> "Any":
    """Project ``x`` onto the fitted principal components.

    Parameters
    ----------
    fitted:
        Result of :func:`fit_pca`.
    x:
        Feature DataFrame to transform.

    Returns
    -------
    pandas.DataFrame
        Component scores with ``component_names`` columns, indexed like ``x``.
    """
    import pandas as pd

    transformed = fitted.pca.transform(x[fitted.input_columns])
    return pd.DataFrame(transformed, columns=fitted.component_names, index=x.index)
