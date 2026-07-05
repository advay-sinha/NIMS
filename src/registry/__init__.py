"""Model registry for NetSentinel.

A lightweight, file-based record of completed experiments and their promotion
state: every registered model, the automatically selected best candidate per
dataset, and the explicit production assignments. Built entirely from the
persisted experiment manifests — no model files are moved or copied, nothing
is retrained, no database is involved. The resolver is the lookup surface a
future inference service (FastAPI) will use.
"""
