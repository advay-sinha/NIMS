"""Model explainability for NetSentinel (SHAP backends).

Explains trained Engine A models: which features drive each prediction and
how important every feature is globally. Backends implement
:class:`src.explainability.base.BaseExplainer` and are resolved through
:mod:`src.explainability.registry`; artefacts are persisted per experiment
under ``outputs/explainability/<experiment_id>/``.
"""
