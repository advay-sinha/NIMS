"""SHAP explainability backend for XGBoost.

Purpose
-------
Explain fitted :class:`src.models.xgboost_model.XGBoostModel` instances with
``shap.TreeExplainer`` (exact tree SHAP — no sampling approximation, no model
re-training). Supports binary and multiclass classifiers.

Inputs
------
A fitted XGBoost model wrapper and a pandas feature matrix.

Outputs
-------
:class:`src.explainability.base.ExplanationResult` with values shaped
``(n_samples, n_features, n_outputs)``.

Limitations
-----------
XGBoost only; other model families need their own backend (registry).
"""

from __future__ import annotations

import logging
from typing import Any

from src.explainability.base import (
    BaseExplainer,
    ExplainabilityError,
    ExplanationResult,
    resolve_feature_names,
)

logger = logging.getLogger(__name__)


class XGBoostShapExplainer(BaseExplainer):
    """Tree-SHAP explainer for the XGBoost model wrapper."""

    name = "xgboost"

    def explain(self, model: Any, x: Any) -> ExplanationResult:
        """Compute exact tree-SHAP values for ``model`` on ``x``.

        Parameters
        ----------
        model:
            Fitted :class:`src.models.xgboost_model.XGBoostModel`.
        x:
            Feature matrix (pandas DataFrame) whose columns are the model's
            training features, in training order.

        Returns
        -------
        ExplanationResult

        Raises
        ------
        ExplainabilityError
            When ``model`` is not a fitted XGBoost wrapper, ``x`` lacks
            feature names, or shap is not installed.
        """
        import numpy as np

        try:
            import shap
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ExplainabilityError(
                "The 'shap' package is not installed; install it to enable "
                "explainability (pip install shap)."
            ) from exc

        if getattr(model, "name", None) != self.name:
            raise ExplainabilityError(
                f"XGBoostShapExplainer cannot explain model "
                f"'{getattr(model, 'name', type(model).__name__)}'."
            )
        if getattr(model, "model", None) is None:
            raise ExplainabilityError("Model is not fitted; nothing to explain.")

        feature_names = resolve_feature_names(x)
        logger.info(
            "Computing tree-SHAP values: %d sample(s) x %d feature(s).",
            len(x), len(feature_names),
        )
        explainer = shap.TreeExplainer(model.model)
        values = np.asarray(explainer.shap_values(x))
        if values.ndim == 2:  # binary: single margin output
            values = values[:, :, np.newaxis]
        base_values = np.atleast_1d(np.asarray(explainer.expected_value))

        n_outputs = values.shape[2]
        fitted = getattr(model, "classes_", None)
        fitted_classes = [] if fitted is None else list(fitted)
        if len(fitted_classes) == n_outputs:
            class_labels = [str(c) for c in fitted_classes]
        else:  # e.g. binary logistic: 2 classes, 1 margin output
            class_labels = [f"output_{i}" for i in range(n_outputs)]

        return ExplanationResult(
            values=values,
            base_values=base_values,
            feature_names=feature_names,
            class_labels=class_labels,
            n_samples=int(values.shape[0]),
        )
