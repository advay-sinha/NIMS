"""Data layer (Layer 1-2) for NetSentinel.

Implements reproducible dataset ingestion, validation, cleaning, encoding,
scaling and splitting behind a single loader interface (Phase 1 goal:
"Every dataset can be loaded using one interface").

Modules
-------
base
    Abstract loader interface and split containers.
schema
    Feature / dataset metadata dataclasses.
registry
    Maps dataset ids to loader classes.
loaders
    Concrete per-dataset loaders (NSL-KDD, UNSW-NB15, CICIDS2017, SNMP).
validation
    Integrity and schema checks producing validation reports.
cleaning
    Missing-value, duplicate and infinity handling.
encoding
    Reproducible categorical encoding (fit on train only).
scaling
    Numerical scaling (fit on train only).
splitting
    Reproducible train/validation/test partitioning.
statistics
    Descriptive dataset statistics.
metadata
    Generation and persistence of dataset metadata.
pipeline
    Orchestrates the full Raw -> Processed flow per dataset.

Design rules (CLAUDE.md): never modify raw data; fit transforms on train only;
persist every stage; document every assumption.
"""

from __future__ import annotations
