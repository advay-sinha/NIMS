# NIMS — Network Intrusion Monitoring System

> A modular, scalable, AI-powered Network Intrusion Monitoring System designed to detect, classify, and analyze malicious network traffic using machine learning and deep learning.

---

# Overview

NIMS is an end-to-end cybersecurity project focused on building a production-ready Network Intrusion Monitoring System capable of identifying cyber attacks across multiple benchmark datasets.

The project combines robust data engineering, machine learning, deep learning, and software engineering principles to create a reproducible and extensible intrusion detection pipeline.

Rather than being a collection of experiments or notebooks, NIMS is being developed as a complete software system with configurable pipelines, automated validation, experiment tracking, and deployment-ready architecture.

---

# Objectives

The primary goals of NIMS are to:

- Develop a unified preprocessing pipeline for multiple intrusion detection datasets.
- Detect and classify malicious network traffic with high accuracy.
- Compare traditional machine learning algorithms with modern deep learning architectures.
- Support binary, multiclass, and anomaly detection tasks.
- Build reproducible training and evaluation pipelines.
- Create a scalable system suitable for future real-time deployment.

---

# Supported Datasets

Current support includes:

- NSL-KDD
- UNSW-NB15
- CIC-IDS2017

The architecture is designed to support additional intrusion detection datasets in future releases.

---

# Current Progress

## ✅ Completed

### Project Foundation

- Modular project architecture
- Configuration-driven pipeline
- Logging framework
- Utility modules
- Automated testing framework
- Dataset registry

### Data Engineering

- Dataset loaders
- Dataset validation
- Schema verification
- Statistical profiling
- Dataset fingerprinting
- Dataset audit generation
- Automated validation reports

### Data Preprocessing

- Configuration-driven cleaning (duplicates, infinities, missing values, outlier clipping, dtype normalization)
- Train-only categorical encoding (one-hot / ordinal) with safe unknown-category handling
- Train-only feature scaling (standard / min-max / robust)
- Reproducible, stratified train/validation/test splitting
- End-to-end preprocessing orchestrator with no data leakage
- Persisted processed datasets, fitted encoder/scaler artifacts, and a reproducibility manifest
- Per-stage reports (cleaning, encoding, scaling, split)

### Feature Engineering

- Variance-threshold filtering (constant / near-constant removal)
- Correlation filtering (Pearson / Spearman) of redundant feature pairs
- Statistical selection: mutual information, chi-square, ANOVA F-test
- Tree-based (RandomForest) feature importance
- Recursive Feature Elimination (RFE)
- Optional PCA with configurable explained variance
- Train-only fitting with no leakage; serialized selector + PCA artifacts
- Per-dataset reports (feature report, metadata, selected/removed features)

### Engine A — Baseline Models (Layer 3)

- Centralized GPU/hardware detection (`src/utils/hardware.py`) with automatic CUDA selection and CPU fallback (no training-based probes)
- Model interface + registry: XGBoost (GPU), LightGBM (attempts GPU, falls back to CPU with a warning), Isolation Forest (anomaly)
- Reproducible, configuration-driven training orchestrator (no leakage; train-only fitting) with a defensive minimum-rows guard against accidental data subsets
- Full metric suite: precision, recall, F1, ROC-AUC (multiclass computed over the complete fitted label set), false-positive rate, confusion matrix
- Experiment tracking with unique, never-overwritten run directories
- Per-run artifacts: serialized model, metrics, and a manifest (config snapshot, hardware, timings, model size)
- Isolation Forest config hardening: config-provided `n_jobs` no longer collides with the wrapper default (regression-tested against `configs/training.yaml`)

### Model Diagnostics

- First NSL-KDD training runs completed for XGBoost (test F1 ≈ 0.992) and LightGBM (test F1 ≈ 0.682)
- Root-cause diagnostic for the LightGBM gap: severe underfitting traced to the OpenCL GPU tree learner combined with `max_bin: 63` (see `outputs/reports/nsl_kdd_lightgbm_vs_xgboost_diagnostic.md`)

### Software Quality

- Unit testing
- Configuration validation
- Dataset integrity verification
- Reproducibility support

---

## 🚧 Current Phase

The Engine A baseline training framework is implemented and verified by tests.
Models are trained manually (Human-in-the-Loop). Development is now moving
toward model evaluation, comparison, and tuning.

This includes:

- Manual training runs on NSL-KDD, UNSW-NB15, CICIDS2017
- Cross-model comparison and metric reporting
- Hyperparameter tuning
- Explainability (SHAP / feature importance)

Immediate next step (manual, HITL): re-run LightGBM on NSL-KDD on CPU with
default binning to confirm the GPU/`max_bin` diagnosis from the diagnostic
report before any hyperparameter tuning.

---

# Planned Features

## Machine Learning

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## Deep Learning

- Multi-Layer Perceptron (MLP)
- LSTM
- CNN
- Autoencoder-based anomaly detection
- Transformer-based architectures

## Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Cross-validation
- Statistical model comparison

## Explainability

- SHAP
- Feature importance
- Error analysis
- Prediction visualization

## Deployment

- FastAPI inference service
- REST API
- Docker support
- Real-time monitoring
- Model serving
- Logging and monitoring

---

# Project Structure

```text
NIMS/
│
├── configs/
├── datasets/
├── notebooks/
├── outputs/
├── scripts/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── pyproject.toml
└── README.md
```

---

# Usage

All entry points are configuration-driven and read from `configs/`. Run them as
modules from the repository root.

Validate raw datasets:

```bash
python -m scripts.validate_datasets --all
python -m scripts.validate_datasets --dataset nsl_kdd
```

Audit datasets (statistics, fingerprints, Markdown audit report):

```bash
python -m scripts.run_audit --all
```

Run the preprocessing pipeline (clean → split → encode → scale → persist):

```bash
python -m scripts.run_preprocessing --dataset nsl_kdd
python -m scripts.run_preprocessing --all
```

Preprocessing outputs are written per dataset under:

```text
outputs/preprocessing/<id>/{cleaning,encoding,scaling,split}_report.json
outputs/preprocessing/<id>/preprocessing_manifest.json
outputs/processed/<id>/{train,validation,test}.parquet
outputs/artifacts/<id>/{encoder,scaler,label_encoder}.joblib
```

Run feature engineering (variance → correlation → selection → optional PCA):

```bash
python -m scripts.run_feature_engineering --dataset nsl_kdd
python -m scripts.run_feature_engineering --all
```

Feature-engineering outputs are written per dataset under:

```text
outputs/features/<id>/{train,validation,test}.parquet
outputs/features/<id>/{feature_report,feature_metadata,selected_features,removed_features}.json
outputs/artifacts/<id>/{feature_selector,pca}.joblib
```

Train Engine A models (GPU is auto-detected; falls back to CPU):

```bash
python -m scripts.train_model --dataset nsl_kdd --model xgboost
python -m scripts.train_model --dataset nsl_kdd --all-models
python -m scripts.train_model --all-datasets --all-models
```

Each run writes an isolated, never-overwritten experiment:

```text
outputs/experiments/<id>/<model>/<run_id>/{model.joblib,metrics.json,manifest.json}
```

Run the test suite:

```bash
pytest
```

---

# Design Principles

NIMS is built around the following principles:

- Modular architecture
- Configuration over hardcoded values
- Reproducible experiments
- Test-driven development
- Separation of data engineering and model training
- Scalable software design
- Production-oriented implementation

---

# Roadmap

- ✅ Project architecture
- ✅ Dataset ingestion
- ✅ Dataset validation
- ✅ Dataset auditing
- ✅ Data preprocessing
- ✅ Feature engineering
- ✅ Engine A baseline model framework (XGBoost, LightGBM, Isolation Forest)
- 🚧 Model evaluation & tuning
- ⏳ Deep learning models
- ⏳ Hyperparameter optimization
- ⏳ Model explainability
- ⏳ Model deployment
- ⏳ Real-time monitoring dashboard

---

# Current Status

**Current Development Stage:** Engine A Model Training & Evaluation

The data engineering, preprocessing, and feature-engineering layers are complete, and the Engine A baseline training framework (GPU-aware XGBoost, LightGBM, and Isolation Forest with reproducible experiment tracking and a full metric suite) is implemented and tested. Per the Human-in-the-Loop policy, models are trained manually; development is now progressing toward training runs, cross-model evaluation, and tuning.

---

# License

This project is intended for educational, research, and experimental purposes.