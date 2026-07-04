"""Engine A model layer (Layer 3) for NetSentinel.

Cyber intrusion-detection models behind one interface (:class:`BaseModel`):
gradient-boosted classifiers (XGBoost, LightGBM) and an unsupervised anomaly
detector (Isolation Forest). GPU usage is negotiated through
:mod:`src.utils.hardware`; hyperparameters come from configuration only.

Modules
-------
base
    Abstract model interface + serialisation.
registry
    Maps model ids to classes and builds them from config.
xgboost_model, lightgbm_model, isolation_forest
    Concrete model wrappers.
"""
