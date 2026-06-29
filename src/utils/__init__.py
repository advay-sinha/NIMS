"""Cross-cutting utilities for NetSentinel.

Modules
-------
config
    Load, merge and validate YAML configuration.
logging_utils
    Configure logging from ``configs/logging.yaml`` (never use ``print``).
seed
    Deterministic seeding of python / numpy / torch for reproducibility.
paths
    Resolve repository paths from configuration.
io
    Typed read/write helpers (parquet, json, yaml, joblib).
gpu
    Device detection with automatic CPU fallback.
timer
    Lightweight timing context manager / decorator.

Nothing here imports heavy ML frameworks at module import time; optional
dependencies (torch) are imported lazily inside functions.
"""

from __future__ import annotations
