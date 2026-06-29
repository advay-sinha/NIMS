"""Concrete per-dataset loaders.

Each loader subclasses :class:`src.data.base.BaseDatasetLoader` and implements
only dataset-specific reading / labelling. They are wired to ids in
:mod:`src.data.registry`.

- :class:`NSLKDDLoader`     — NSL-KDD (Engine A)
- :class:`UNSWNB15Loader`   — UNSW-NB15 (Engine A)
- :class:`CICIDS2017Loader` — CICIDS2017 (Engine A)
- :class:`SNMPLoader`       — SNMP 2016 telemetry (Engine B)
"""

from __future__ import annotations

from src.data.loaders.cicids2017 import CICIDS2017Loader
from src.data.loaders.nsl_kdd import NSLKDDLoader
from src.data.loaders.snmp import SNMPLoader
from src.data.loaders.unsw_nb15 import UNSWNB15Loader

__all__ = [
    "NSLKDDLoader",
    "UNSWNB15Loader",
    "CICIDS2017Loader",
    "SNMPLoader",
]
