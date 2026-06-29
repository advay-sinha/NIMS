"""Configuration loading and merging.

Purpose
-------
Provide a single, reproducible entry point for reading the YAML configuration
under ``configs/``. The repository forbids hardcoded paths, dataset names and
hyperparameters (CLAUDE.md > Repository Principles); every runtime value must
be sourced through this module.

Inputs
------
- A root config file (default ``configs/config.yaml``) that lists included
  files under ``defaults`` and is deep-merged with them.
- Optional per-dataset config files under ``configs/datasets/``.
- Optional override mappings (e.g. parsed CLI flags).

Outputs
-------
- A nested ``dict`` representing the effective configuration.

Examples
--------
>>> cfg = load_config()  # doctest: +SKIP
>>> seed = cfg["project"]["seed"]  # doctest: +SKIP

Limitations
-----------
Schema validation is intentionally lightweight in Phase 1. A stricter,
dataclass-backed schema is planned. TODO(data-engineer): add schema models.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

logger = logging.getLogger(__name__)

# Repository-relative default configuration locations.
CONFIG_DIR: Path = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"
DATASET_CONFIG_DIR: Path = CONFIG_DIR / "datasets"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a single YAML file into a dictionary.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict
        Parsed contents, or an empty dict for an empty file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Configuration root must be a mapping, got {type(loaded).__name__}: "
            f"{resolved}"
        )
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs.

    Nested mappings are merged key-by-key; non-mapping values in ``override``
    replace those in ``base``. Used to compose ``config.yaml`` with its
    ``defaults`` includes and with CLI overrides.

    Parameters
    ----------
    base:
        Lower-priority mapping.
    override:
        Higher-priority mapping whose values win on conflict.

    Returns
    -------
    dict
        A new, deep-merged mapping.
    """
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and compose the effective NetSentinel configuration.

    Resolution order (lowest to highest priority):
        1. Files listed under the root config's ``defaults`` key.
        2. The root config file itself.
        3. ``overrides`` (e.g. parsed CLI flags).

    Parameters
    ----------
    path:
        Root configuration file. Defaults to ``configs/config.yaml``.
    overrides:
        Optional mapping deep-merged last.

    Returns
    -------
    dict
        The effective configuration.
    """
    logger.debug("Loading configuration from %s", path)
    root_path = Path(path)
    root = load_yaml(root_path)

    # Resolve and merge `defaults` includes first (in declared order), so that
    # values in the root file override them.
    effective: dict[str, Any] = {}
    for include in root.get("defaults", []) or []:
        include_path = (root_path.parent / include).resolve()
        effective = deep_merge(effective, load_yaml(include_path))

    # The root file's own keys (minus the consumed `defaults` list) win next.
    root_without_defaults = {k: v for k, v in root.items() if k != "defaults"}
    effective = deep_merge(effective, root_without_defaults)

    if overrides:
        effective = deep_merge(effective, overrides)

    return effective


def load_dataset_config(dataset_id: str) -> dict[str, Any]:
    """Load a single dataset configuration by id.

    Parameters
    ----------
    dataset_id:
        Identifier matching a file ``configs/datasets/<dataset_id>.yaml``
        (e.g. ``"nsl_kdd"``).

    Returns
    -------
    dict
        The dataset configuration (the contents under the ``dataset`` key).

    Raises
    ------
    FileNotFoundError
        If no config file exists for ``dataset_id``.
    """
    config_path = DATASET_CONFIG_DIR / f"{dataset_id}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No dataset config for '{dataset_id}' (expected {config_path})."
        )
    raw = load_yaml(config_path)
    dataset_block = raw.get("dataset")
    if not isinstance(dataset_block, dict):
        raise ValueError(
            f"Dataset config '{config_path}' is missing a top-level 'dataset' block."
        )
    return dataset_block


def get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Safely read a nested value using a dotted path.

    Parameters
    ----------
    config:
        Configuration mapping.
    dotted_key:
        Dotted path, e.g. ``"data.split.train_size"``.
    default:
        Returned when any segment is missing.

    Returns
    -------
    Any
        The resolved value or ``default``.
    """
    node: Any = config
    for segment in dotted_key.split("."):
        if not isinstance(node, MutableMapping) or segment not in node:
            return default
        node = node[segment]
    return node
