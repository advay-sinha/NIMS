"""Explainability backend registry.

Purpose
-------
Resolve a model name to its explainability backend, mirroring
:mod:`src.models.registry`. Unsupported models fail with an explicit error
listing what is available, so callers can either surface or skip cleanly.
"""

from __future__ import annotations

from src.explainability.base import BaseExplainer, ExplainabilityError
from src.explainability.shap_xgboost import XGBoostShapExplainer

_EXPLAINERS: dict[str, type[BaseExplainer]] = {
    "xgboost": XGBoostShapExplainer,
}


def available_explainers() -> list[str]:
    """Return the model names with an implemented explainability backend."""
    return sorted(_EXPLAINERS)


def get_explainer(model_name: str) -> BaseExplainer:
    """Instantiate the explainability backend for a model name.

    Parameters
    ----------
    model_name:
        Registered model id (e.g. ``"xgboost"``).

    Returns
    -------
    BaseExplainer

    Raises
    ------
    ExplainabilityError
        When no backend exists for ``model_name``.
    """
    explainer_cls = _EXPLAINERS.get(model_name)
    if explainer_cls is None:
        raise ExplainabilityError(
            f"No explainability backend for model '{model_name}'. "
            f"Supported: {', '.join(available_explainers())}."
        )
    return explainer_cls()
