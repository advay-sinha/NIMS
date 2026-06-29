"""NetSentinel — AI-powered Network Intrusion Detection & Health Prediction.

Top-level package. Sub-packages are organised by architectural layer
(see CLAUDE.md > Project Architecture):

- ``src.data``   : Layer 1-2 — data ingestion, validation, preprocessing.
- ``src.utils``  : cross-cutting utilities (config, logging, seeding, IO).

Later phases add ``src.features``, ``src.models``, ``src.evaluation`` and
``src.api`` layers. Layers communicate only through defined interfaces.
"""

__version__ = "1.0.0"
