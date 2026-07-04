"""Model registry.

Purpose
-------
Map a model id to its concrete class and build it from configuration, so callers
select models by config string rather than importing a specific class
(decoupling per CLAUDE.md > Repository Principles).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Type

from src.models.base import BaseModel
from src.models.isolation_forest import IsolationForestModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# Single source of truth mapping classical model id -> class. Deep-learning
# models live in src/models/deep_learning/registry.py and are merged lazily
# (importing them pulls in PyTorch, which classical paths should not pay for).
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "isolation_forest": IsolationForestModel,
}


def _full_registry() -> dict[str, Type[BaseModel]]:
    """Return the classical registry merged with the deep-learning one."""
    from src.models.deep_learning.registry import DEEP_MODEL_REGISTRY

    return {**MODEL_REGISTRY, **DEEP_MODEL_REGISTRY}


def get_model_cls(name: str) -> Type[BaseModel]:
    """Return the model class registered for ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name in MODEL_REGISTRY:  # fast path avoids importing torch
        return MODEL_REGISTRY[name]
    registry = _full_registry()
    if name not in registry:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(registry)}")
    return registry[name]


def available_models() -> list[str]:
    """Return the sorted list of registered model ids (classical + deep)."""
    return sorted(_full_registry())


def build_model(
    name: str,
    model_config: Mapping[str, Any],
    use_gpu: bool,
    seed: int,
) -> BaseModel:
    """Construct a model from its configuration block.

    Parameters
    ----------
    name:
        Registered model id.
    model_config:
        The ``models.<name>`` config block (``params``, ``gpu``).
    use_gpu:
        Global GPU preference (ANDed with the model's own ``gpu`` flag).
    seed:
        Reproducibility seed.

    Returns
    -------
    BaseModel
    """
    cls = get_model_cls(name)
    model_use_gpu = bool(use_gpu and model_config.get("gpu", True))
    return cls(params=model_config.get("params", {}), use_gpu=model_use_gpu, seed=seed)
