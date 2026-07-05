"""Abstract explainability interface.

Purpose
-------
Define the single interface every explainability backend implements, mirroring
:mod:`src.models.base`: the runner and artefact writers are backend-agnostic,
and adding a new backend (LightGBM, deep models) never changes them.

Inputs
------
A fitted :class:`src.models.base.BaseModel` wrapper and a pandas feature
matrix whose columns are the model's feature names.

Outputs
-------
An :class:`ExplanationResult` with SHAP values normalised to a single 3-D
layout ``(n_samples, n_features, n_outputs)`` regardless of backend or class
count.

Limitations
-----------
Explainers operate on already-fitted models; they never train or mutate them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ExplainabilityError(RuntimeError):
    """Raised when an explanation cannot be produced for a model/input."""


@dataclass(frozen=True)
class ExplanationResult:
    """SHAP values for one model on one sample matrix.

    Attributes
    ----------
    values:
        ``numpy.ndarray`` of shape ``(n_samples, n_features, n_outputs)``.
        ``n_outputs`` is the class count for multiclass models and ``1`` for
        binary models (single margin output).
    base_values:
        Expected model output per output dimension, shape ``(n_outputs,)``.
    feature_names:
        Feature names in matrix column order.
    class_labels:
        One label per output dimension (fitted class labels when they map
        one-to-one onto outputs).
    n_samples:
        Number of rows explained.
    """

    values: Any
    base_values: Any
    feature_names: list[str]
    class_labels: list[str]
    n_samples: int


def resolve_feature_names(x: Any) -> list[str]:
    """Return the feature names of a matrix, failing loudly when absent.

    Parameters
    ----------
    x:
        Feature matrix; must expose named columns (pandas DataFrame).

    Returns
    -------
    list[str]
        Column names in order.

    Raises
    ------
    ExplainabilityError
        When ``x`` has no named columns (e.g. a bare numpy array) — SHAP
        artefacts without feature names are meaningless.
    """
    columns = getattr(x, "columns", None)
    if columns is None or len(columns) == 0:
        raise ExplainabilityError(
            "Feature matrix has no named columns; explainability requires a "
            "pandas DataFrame with the model's feature names."
        )
    return [str(c) for c in columns]


class BaseExplainer(ABC):
    """Abstract base for all explainability backends."""

    name: str = "base"

    @abstractmethod
    def explain(self, model: Any, x: Any) -> ExplanationResult:
        """Compute SHAP values for ``model`` on the sample matrix ``x``.

        Parameters
        ----------
        model:
            Fitted :class:`src.models.base.BaseModel` wrapper.
        x:
            Feature matrix (pandas DataFrame) to explain.

        Returns
        -------
        ExplanationResult

        Raises
        ------
        ExplainabilityError
            When the model is unsupported/unfitted or the input is invalid.
        """
        raise NotImplementedError
