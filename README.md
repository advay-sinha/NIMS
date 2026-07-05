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
- Pre-fit feature audit before every model fit: column names, order, dtypes, missing values and duplicates verified against the feature-engineering artifacts (`src/training/feature_audit.py`)
- Full metric suite: precision, recall, F1, ROC-AUC, false-positive rate, confusion matrix
- Multiclass ROC-AUC computed one-vs-rest over the complete fitted label set; classes absent from a split (undefined one-vs-rest AUC) are skipped mathematically instead of producing sklearn `UndefinedMetricWarning`s or NaN averages
- Experiment tracking with unique, never-overwritten run directories
- Searchable experiment index (`outputs/experiments/experiment_index.csv`): every run appends one summary row (timestamp, dataset, model, run id, train time, epochs, test accuracy/F1/ROC-AUC, hardware, key hyperparameters); `scripts/build_experiment_index.py` rebuilds it from the manifests
- Per-run artifacts: serialized model, metrics, and a manifest (config snapshot, hardware, timings, model size)
- Parameter provenance: manifests record both the configured hyperparameters and the estimator's final parameter dictionary captured at fit time (`fitted_params`), so device fallbacks and library defaults are always visible
- Isolation Forest config hardening: config-provided `n_jobs` no longer collides with the wrapper default (regression-tested against `configs/training.yaml`)

### Model Validation & Diagnostics

- Training runs completed on NSL-KDD, UNSW-NB15 and CICIDS2017 (XGBoost test F1 ≈ 0.992–0.999; LightGBM multiclass runs underperform, see below)
- Aggregated benchmark report generator (`scripts/generate_validation_report.py`) producing `outputs/reports/model_validation_report.md` from the run manifests: per-model summaries, cross-model comparison tables, timings, model sizes and train/validation/test metrics
- Root-cause diagnosis of the LightGBM multiclass gap, evidenced from the saved boosters: unregularized leaf outputs under 40-class softmax (`reg_lambda=0`, near-zero hessian sums) make validation loss diverge and boosting collapse after ~30 of 400 iterations, aggravated by the GPU-oriented `max_bin: 63` remaining active on CPU runs; recommended parameter changes are documented in the validation report (pending manual re-training)

### Engine B — Deep Learning Framework (Layer 3)

- PyTorch model family integrated behind the same model interface as Engine A: MLP, 1D-CNN, LSTM and Transformer over tabular feature vectors (`src/models/deep_learning/`)
- Shared training engine (`src/models/deep_learning/base.py`): train/validation loops, early stopping with in-memory best-model restore, learning-rate scheduling (plateau/cosine/step), mixed precision (`torch.amp`), gradient accumulation and clipping, optional checkpointing, per-epoch logging (loss, learning rate, time, GPU memory) and deterministic seeding
- Fully configuration-driven via `configs/deep_learning.yaml` (shared training block + per-model architecture parameters)
- CUDA selected automatically through the central hardware module with CPU fallback; pinned DataLoaders and non-blocking transfers on GPU
- Reuses the existing trainer, metric suite, experiment tracking and manifests — deep models train through the same entry point and produce identical run artifacts

### Explainability (SHAP)

- Modular explainability package (`src/explainability/`): backend interface, registry, artefact persistence and orchestration, mirroring the model registry design
- XGBoost backend using exact tree-SHAP (`shap.TreeExplainer`) with binary and multiclass support; values normalised to a single `(samples, features, outputs)` layout
- Per-experiment artefacts under `outputs/explainability/<experiment_id>/`: `metadata.json` (run identity, sample count, SHAP version), `feature_importance.csv` (feature, mean |SHAP|, rank), `shap_values.pkl` (full arrays) and `global_summary.png`
- Extended global importance (`global_feature_importance.csv`: mean/std |SHAP|, percentage and cumulative contribution per feature) and deterministic per-sample explanations (`local/sample_NNNN.csv`: feature value, signed and absolute SHAP contribution, rank); multiclass outputs are aggregated across classes, binary models keep exact signed values
- Configuration-driven (`configs/explainability.yaml`): enable/disable, explained split and a seeded sample cap without touching training code; supported models are explained automatically after training, and `scripts/run_explainability.py` explains any completed run post-hoc

### Software Quality

- Unit testing
- Configuration validation
- Dataset integrity verification
- Reproducibility support

---

## 🚧 Current Phase

The Engine A training pipeline has been audited and stabilized: metric
correctness (ROC-AUC on splits with missing classes), parameter provenance in
manifests, and a mandatory pre-fit feature audit are in place, and the
cross-model benchmark report is generated from the run manifests. Models are
trained manually on local hardware.

Remaining in this phase:

- Extend explainability to LightGBM and the deep models
- Per-class error analysis on UNSW-NB15 from the stored confusion matrices
- Hyperparameter tuning

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

- Multi-Layer Perceptron (MLP) — framework implemented
- LSTM — framework implemented
- CNN (1D) — framework implemented
- Transformer encoder — framework implemented
- Autoencoder-based anomaly detection — planned

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

Train models (GPU is auto-detected; falls back to CPU). Classical (Engine A)
and deep-learning models share the same entry point and output format:

```bash
python -m scripts.train_model --dataset nsl_kdd --model xgboost
python -m scripts.train_model --dataset nsl_kdd --model mlp
python -m scripts.train_model --dataset nsl_kdd --model transformer
python -m scripts.train_model --dataset nsl_kdd --all-models
python -m scripts.train_model --all-datasets --all-models
```

Deep-learning hyperparameters (batch size, epochs, optimizer, scheduler,
early stopping, mixed precision, gradient clipping/accumulation) live in
`configs/deep_learning.yaml`.

Each run writes an isolated, never-overwritten experiment:

```text
outputs/experiments/<id>/<model>/<run_id>/{model.joblib,metrics.json,manifest.json}
```

Generate the cross-model validation report from the recorded experiments:

```bash
python -m scripts.generate_validation_report
python -m scripts.generate_validation_report --analysis outputs/reports/model_validation_analysis.md
```

The report is written to `outputs/reports/model_validation_report.md`.

Rebuild the searchable experiment index (new runs append automatically):

```bash
python -m scripts.build_experiment_index
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
- ✅ Training-pipeline validation & benchmark reporting
- ✅ Deep-learning model framework (MLP, CNN, LSTM, Transformer)
- 🚧 Model evaluation & tuning
- ⏳ Deep-learning training runs & comparison
- ⏳ Hyperparameter optimization
- ⏳ Model explainability
- ⏳ Model deployment
- ⏳ Real-time monitoring dashboard

---

# Current Status

**Current Development Stage:** Deep-Learning Baselines & Model Evaluation

The data engineering, preprocessing, and feature-engineering layers are complete. The Engine A baseline framework (GPU-aware XGBoost, LightGBM, Isolation Forest) has been audited end-to-end — metric correctness, parameter provenance, and pre-fit feature validation — with benchmark results aggregated into a cross-model validation report. The deep-learning framework (MLP, 1D-CNN, LSTM, Transformer behind the same model interface, with mixed precision, early stopping, and configuration-driven training) is implemented and tested. Models are trained manually on local hardware; the next milestone is running and benchmarking the deep-learning baselines alongside the classical models.

---

# License

This project is intended for educational, research, and experimental purposes.