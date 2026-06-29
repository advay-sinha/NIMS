"""Logging configuration for NetSentinel.

Purpose
-------
Centralise logging setup so that no module ever calls ``print`` (CLAUDE.md >
Logging Rules). Console output is human-readable (optionally via ``rich``);
a rotating file handler captures DEBUG-level detail for reproducibility.

Inputs
------
- The ``logging`` block from the merged configuration (``configs/logging.yaml``).
- A resolved log directory (from ``src.utils.paths``).

Outputs
-------
- A configured root logger; module loggers obtained via ``get_logger``.

Examples
--------
>>> from src.utils.logging_utils import setup_logging, get_logger
>>> setup_logging(config)            # doctest: +SKIP
>>> log = get_logger(__name__)       # doctest: +SKIP
>>> log.info("pipeline started")     # doctest: +SKIP

Limitations
-----------
``setup_logging`` is idempotent-by-intent: repeated calls must not duplicate
handlers. TODO(data-engineer): guard against double configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

# Module-level logger used for setup diagnostics only.
logger = logging.getLogger(__name__)

DEFAULT_LEVEL: int = logging.INFO


def setup_logging(
    config: Mapping[str, Any] | None = None,
    log_dir: str | Path | None = None,
    level: int | str | None = None,
) -> logging.Logger:
    """Configure the root logger from a logging configuration block.

    Parameters
    ----------
    config:
        Effective configuration containing a ``logging`` section. When ``None``
        a minimal console-only default is applied.
    log_dir:
        Directory for the rotating log file. Created if missing.
    level:
        Optional override of the root level (e.g. ``"DEBUG"`` from a CLI flag).

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    log_config: Mapping[str, Any] = {}
    if config is not None:
        log_config = config.get("logging", {}) or {}

    root = logging.getLogger()

    # Resolve the effective root level (explicit override wins).
    resolved_level = _coerce_level(level if level is not None else log_config.get("level"))
    root.setLevel(resolved_level)

    # Idempotency: clear handlers this function previously installed so repeated
    # setup calls (e.g. tests, re-entrant scripts) never duplicate output.
    for handler in list(root.handlers):
        if getattr(handler, "_netsentinel", False):
            root.removeHandler(handler)

    formatters = log_config.get("formatters", {})
    handlers = log_config.get("handlers", {})

    _add_console_handler(root, handlers.get("console", {}), formatters)
    _add_file_handler(root, handlers.get("file", {}), formatters, log_dir)

    logger.debug("Logging configured at level %s", logging.getLevelName(resolved_level))
    return root


def _coerce_level(level: int | str | None) -> int:
    """Normalise a level given as int, name or ``None`` to a logging int."""
    if level is None:
        return DEFAULT_LEVEL
    if isinstance(level, int):
        return level
    return logging.getLevelName(str(level).upper())  # type: ignore[return-value]


def _build_formatter(spec: Mapping[str, Any] | None) -> logging.Formatter:
    """Build a ``logging.Formatter`` from a formatter spec mapping."""
    spec = spec or {}
    return logging.Formatter(
        fmt=spec.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
        datefmt=spec.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )


def _add_console_handler(
    root: logging.Logger,
    spec: Mapping[str, Any],
    formatters: Mapping[str, Any],
) -> None:
    """Attach a console handler (RichHandler when available and requested)."""
    if not spec.get("enabled", True):
        return

    formatter = _build_formatter(formatters.get(spec.get("formatter", "console")))
    handler: logging.Handler

    if spec.get("use_rich", False):
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False)
        except ImportError:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    handler.setLevel(_coerce_level(spec.get("level")))
    handler._netsentinel = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def _add_file_handler(
    root: logging.Logger,
    spec: Mapping[str, Any],
    formatters: Mapping[str, Any],
    log_dir: str | Path | None,
) -> None:
    """Attach a rotating file handler under ``log_dir`` when enabled."""
    if not spec.get("enabled", False) or log_dir is None:
        return

    from logging.handlers import RotatingFileHandler

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / spec.get("filename", "netsentinel.log")

    handler = RotatingFileHandler(
        filename,
        maxBytes=int(spec.get("max_bytes", 10_485_760)),
        backupCount=int(spec.get("backup_count", 5)),
        encoding="utf-8",
    )
    handler.setFormatter(_build_formatter(formatters.get(spec.get("formatter", "detailed"))))
    handler.setLevel(_coerce_level(spec.get("level")))
    handler._netsentinel = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Thin wrapper over ``logging.getLogger`` to keep a single import surface and
    allow future enrichment (e.g. context injection) without touching callers.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
