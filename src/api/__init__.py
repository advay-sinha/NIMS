"""NetSentinel inference API (FastAPI).

Serves the production models promoted in the model registry: batch CSV/JSON
inference through the SAME saved preprocessing and feature-engineering
artefacts the training pipeline produced. Nothing is fitted, trained or
recomputed at inference time.
"""
