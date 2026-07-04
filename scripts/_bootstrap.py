"""Shared CLI bootstrap for scripts.

Purpose
-------
Remove duplication across entry points (CLAUDE.md > Repository Principles:
"Avoid duplicated code"). Centralises config loading, logging setup, seeding
and path resolution so every script starts from an identical, reproducible
state.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

# DLL load-order guard: with torch 2.6.0+cu124 and pyarrow 24.0 on Windows,
# importing torch before pyarrow.dataset makes the pyarrow import crash the
# process with an access violation (observed: every training run on
# 2026-07-03 died at the features parquet read with no traceback). Loading
# pyarrow's native libraries first is safe in both orders, so every entry
# point preloads it before anything (seeding, hardware detection) imports
# torch.
try:  # pragma: no cover - depends on the installed environment
    import pyarrow.dataset  # noqa: F401  (import for DLL side effect only)
except ImportError:
    pass

from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logging_utils import setup_logging
from src.utils.paths import Paths
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class RuntimeContext:
    """Initialised runtime state shared by scripts."""

    config: dict[str, Any]
    paths: Paths


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach arguments common to every script (config path, log level, seed)."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to root config (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override root log level (e.g. DEBUG, INFO).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the reproducibility seed.",
    )


def bootstrap(args: argparse.Namespace) -> RuntimeContext:
    """Load config, configure logging, seed RNGs and resolve paths.

    Parameters
    ----------
    args:
        Parsed CLI namespace (must include common arguments).

    Returns
    -------
    RuntimeContext
    """
    config_path = getattr(args, "config", None) or DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    paths = Paths.from_config(config)

    setup_logging(
        config=config,
        log_dir=paths.logs_dir,
        level=getattr(args, "log_level", None),
    )

    seed = getattr(args, "seed", None)
    if seed is None:
        seed = config.get("project", {}).get("seed", 42)
    set_global_seed(int(seed))

    logger.info("Bootstrap complete (config=%s, seed=%s)", config_path, seed)
    return RuntimeContext(config=config, paths=paths)
