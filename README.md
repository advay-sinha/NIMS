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

### Software Quality

- Unit testing
- Configuration validation
- Dataset integrity verification
- Reproducibility support

---

## 🚧 Current Phase

Development is now focused on feature engineering ahead of model training.

This includes:

- Domain-specific feature derivation
- Feature selection and importance analysis
- Correlation and redundancy reduction
- Feature-set versioning

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
- 🚧 Feature engineering
- ⏳ Machine learning models
- ⏳ Deep learning models
- ⏳ Hyperparameter optimization
- ⏳ Model explainability
- ⏳ Model deployment
- ⏳ Real-time monitoring dashboard

---

# Current Status

**Current Development Stage:** Feature Engineering

The data engineering foundation and the configuration-driven preprocessing pipeline are complete, including dataset ingestion, validation, profiling, fingerprinting, auditing, cleaning, encoding, scaling, and reproducible splitting with persisted artifacts. Development is now progressing toward feature engineering to prepare feature sets for downstream machine learning and deep learning workflows.

---

# License

This project is intended for educational, research, and experimental purposes.