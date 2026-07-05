"""Error analysis for NetSentinel experiments.

Analyses a fitted model's predictions on a split: confusion structure,
per-class metrics, hardest classes and misclassified examples. Works entirely
from existing experiment outputs (model + feature splits) — nothing is
retrained — and persists artefacts per experiment under
``outputs/error_analysis/<experiment_id>/``.
"""
