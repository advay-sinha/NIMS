"""Training layer (Layer 3) for NetSentinel Engine A.

Orchestrates reproducible model training and evaluation on the feature-
engineered datasets, recording metrics, timing, hardware and a per-run
experiment manifest (CLAUDE.md > Experiment Rules: no experiment is overwritten).

Modules
-------
metrics
    Classification metrics (precision, recall, F1, ROC-AUC, FPR, confusion).
experiment
    Unique experiment ids and output directories.
trainer
    Single orchestration surface, called by scripts.
"""
