"""Hyperparameter optimization for NetSentinel models (Optuna backend).

Tunes existing registered models over conservative search spaces, evaluating
every trial on the validation split through the same model-building path the
trainer uses. Studies persist their artefacts under
``outputs/optimization/<study_id>/``; the optional final model trains through
the standard experiment tracking system with optimization provenance recorded
in its manifest.
"""
