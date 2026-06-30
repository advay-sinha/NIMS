"""Feature engineering layer (Layer 2) for NetSentinel.

Implements a reproducible, configuration-driven feature-engineering pipeline
that runs identically across datasets. Every selector is FIT ON TRAINING DATA
ONLY; validation/test sets are transformed with the fitted artefacts (no
leakage).

Stage order (see :mod:`src.features.pipeline`):
    variance -> correlation -> statistical selection -> optional PCA

Modules
-------
metadata
    Feature/target separation and shared metadata containers.
variance
    Variance-threshold filtering (constant / near-constant removal).
correlation
    Pearson / Spearman correlation filtering of redundant pairs.
mutual_information, chi_square, anova
    Univariate statistical scoring functions.
importance
    RandomForest-based feature importance.
selection
    Orchestrates statistical / model-based selection into a FeatureSelector.
dimensionality
    Optional PCA dimensionality reduction.
reports
    Builds the JSON feature reports.
pipeline
    Single orchestration surface, called by scripts.
"""
