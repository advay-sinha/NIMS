"""Dataset loader registry.

Purpose
-------
Map a dataset id to its concrete loader class so callers select datasets by
configuration string, never by importing a specific loader (decoupling per
CLAUDE.md > Repository Principles).

Examples
--------
>>> from src.data.registry import get_loader_cls
>>> loader_cls = get_loader_cls("nsl_kdd")     # doctest: +SKIP
>>> loader = loader_cls(dataset_cfg, paths)     # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import Type

from src.data.base import BaseDatasetLoader
from src.data.loaders.cicids2017 import CICIDS2017Loader
from src.data.loaders.nsl_kdd import NSLKDDLoader
from src.data.loaders.snmp import SNMPLoader
from src.data.loaders.unsw_nb15 import UNSWNB15Loader

logger = logging.getLogger(__name__)

# Single source of truth mapping dataset id -> loader class.
DATASET_REGISTRY: dict[str, Type[BaseDatasetLoader]] = {
    "nsl_kdd": NSLKDDLoader,
    "unsw_nb15": UNSWNB15Loader,
    "cicids2017": CICIDS2017Loader,
    "snmp": SNMPLoader,
}


def get_loader_cls(dataset_id: str) -> Type[BaseDatasetLoader]:
    """Return the loader class registered for ``dataset_id``.

    Parameters
    ----------
    dataset_id:
        Registered dataset identifier.

    Returns
    -------
    Type[BaseDatasetLoader]

    Raises
    ------
    KeyError
        If ``dataset_id`` is not registered.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. "
            f"Registered: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[dataset_id]


def available_datasets() -> list[str]:
    """Return the sorted list of registered dataset ids."""
    return sorted(DATASET_REGISTRY)
