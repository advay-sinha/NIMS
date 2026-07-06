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

- Full benchmark completed: 7 models × 3 datasets, all trained manually on local hardware. XGBoost leads every dataset (test F1 0.9925 on NSL-KDD, 0.9244 on UNSW-NB15, 0.9988 on CICIDS2017); LightGBM is statistically adjacent; the deep models are consistent baselines 0.7–1.4 pp behind at 1–2 orders of magnitude more training time
- Aggregated benchmark report generator (`scripts/generate_validation_report.py`) producing `outputs/reports/model_validation_report.md` from the run manifests: executive summary, best-model and ranking tables, an overall cross-dataset model ranking, classical-vs-deep comparison, efficiency analysis, derived key findings, reproducibility notes and per-model detail tables
- LightGBM multiclass divergence root-caused from the saved boosters (unregularized leaf outputs under 40-class softmax) and fixed via `reg_lambda: 1.0` — NSL-KDD F1 0.775 → 0.9923, CICIDS2017 0.742 → 0.9987; the LightGBM OpenCL GPU limitation on CICIDS2017 (single-precision histograms) is documented in `configs/training.yaml` with a validated `gpu_use_dp` workaround
- XGBoost binary-target objective reconciliation: an explicitly configured multiclass objective on a two-class dataset (UNSW-NB15) is switched to the equivalent binary objective automatically, with regression tests across all dataset cardinalities

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

### Error Analysis

- Modular error-analysis package (`src/error_analysis/`): pure metric builders, an analyzer, artefact persistence and a Markdown reporting helper
- Per-experiment artefacts under `outputs/error_analysis/<experiment_id>/`: labelled confusion matrix, per-class metrics (support, precision, recall, F1, FP/FN counts), hardest classes ranked by lowest F1, and misclassified examples (highest-confidence errors first, capped); binary tasks additionally get false-positive/false-negative example files
- Configuration-driven (`configs/error_analysis.yaml`): enable/disable, analysed split, example cap and optional feature-value inclusion; runs automatically after training and post-hoc via `scripts/run_error_analysis.py`

### Visualization

- Modular visualization package (`src/visualization/`) rendering production plots from persisted artefacts only — predictions, SHAP values and training are never recomputed
- Per-experiment plots under `outputs/visualizations/<experiment_id>/`: confusion matrix (counts + row-normalized, annotations auto-disabled beyond 25 classes), top-20 SHAP feature importances, hardest classes with support annotations, and most common true→predicted misclassification pairs, plus a `metadata.json` recording generated/skipped plots and source artefacts
- Configuration-driven (`configs/visualization.yaml`): top-N limits, DPI and format; missing upstream artefacts or zero misclassifications skip the affected plot with a recorded reason instead of failing; post-hoc via `scripts/run_visualizations.py`
- Confusion-matrix axes show decoded class names (from the preprocessing encoding report) instead of numeric ids; multiclass runs use the row-normalized view as the primary plot (raw counts hide minority-class errors) with counts kept as a companion

### Hyperparameter Optimization

- Optuna-backed optimization package (`src/optimization/`): per-model search spaces, a validation-split objective built on the existing model registry, seeded TPE/random samplers for reproducible studies
- Supported models: XGBoost, LightGBM, MLP — conservative search spaces (depth/learning-rate/estimators/regularisation for the boosted trees; width/depth/dropout/optimiser settings for the MLP); trials never touch the test split
- Per-study artefacts under `outputs/optimization/<study_id>/`: `metadata.json`, `trials.csv` (state, value, duration, expanded params), `best_params.json`, `best_trial.json` and `optimization_summary.md`; failed trials are recorded and the study continues
- Optional final training of the best parameters through the standard experiment pipeline — the manifest records the optimization provenance (study id, best trial, validation value)
- Invoked explicitly via `scripts/run_optimization.py` (never during normal training); config defaults in `configs/optimization.yaml`

### Model Registry

- File-based model registry (`src/registry/`) built entirely from experiment manifests — no database, no model copies, no recomputed metrics
- `outputs/registry/`: `registry.json` (every registered run with scalar metrics, artefact references and lifecycle status), `best_per_dataset.json` (automatic best candidate by the configured metric) and `production.json` (explicit promotions with reasons)
- Rebuilds are idempotent (`scripts/build_model_registry.py` preserves registration timestamps, tags and statuses; `production.json` stays authoritative); promotion via `scripts/promote_model.py` picks the best registered candidate when no experiment id is pinned
- `resolve_model(dataset, stage)` returns the serving model's paths, metrics and preprocessing/feature artefact references — the lookup surface for the upcoming inference service
- Policy in `configs/registry.yaml`: selection metric/direction, whether optimized runs are eligible, whether test metrics are required

### Inference API

- FastAPI service (`src/api/`) serving the registry-promoted production models for batch inference; start with `python -m scripts.run_api` or `uvicorn src.api.app:app`
- Endpoints: `GET /health`, `GET /models` (production assignments with metrics), `POST /predict/{dataset}` (CSV upload) and `POST /predict-json/{dataset}` (JSON rows)
- Inference replays the training pipeline's saved transforms — the fitted feature encoder and scaler from preprocessing, then alignment to the canonical feature-engineering column list — before predicting; nothing is fitted at inference time, and predictions are decoded back to original label names via the saved label encoder
- Model bundles are resolved through the registry (no hardcoded paths) and cached in memory per (dataset, stage); requests with missing columns, invalid CSV, oversized batches or unknown datasets get clean HTTP errors without stack traces
- Configuration in `configs/api.yaml`: host/port, served stage, row cap, probability output, cache toggle

### Engine B — Network Health Foundation

- Modular telemetry subsystem (`src/network_health/`) over CSV-based SNMP/MIB metrics with a fully configurable schema (`configs/network_health.yaml`) — column roles (counters, gauges, status), required columns and value bounds are never hardcoded
- Schema validation (required columns, timestamp parseability, numeric payload, duplicates, missing values, impossible negatives, counter monotonicity, per-device/interface coverage) persisting JSON + Markdown reports
- Leakage-free preprocessing: per-series counter deltas and per-second rates (resets clipped), value clipping, optional resampling and strictly **chronological** train/validation/test splits
- Health feature engineering: canonical traffic/error/discard rates, configurable rolling mean/std/max windows, lags and status-change indicators — all causal, per (device, interface) series
- Isolation Forest baseline: trains on healthy rows when labels exist (supervised metrics: precision/recall/F1/ROC-AUC/confusion matrix), unsupervised with a quantile threshold otherwise (score distribution, anomaly rate); experiments persist model/metrics/manifest like Engine A
- Config-driven dataset adapters (`src/network_health/adapters.py`, `dataset_registry.py`): a single alias-based mapping engine converts raw/public telemetry — SNMP-MIB 2016 counter dumps, LCORE-D core-network exports or already-canonical CSVs — into the one canonical telemetry schema the pipeline consumes; column aliases, constant device/interface ids, label maps and unknown-column preservation are all configurable per dataset in `configs/network_health.yaml`, no vendor naming is hardcoded. Missing raw sources fail with a clear error (no fabricated data); a `--inspect` mode reports inferred timestamp/label/metric columns for unknown schemas, and adapter runs persist `adapter_report.{json,md}`
- Scripts: `prepare_network_health_dataset` (raw → canonical CSV), `validate_network_health`, `run_network_health_preprocessing`, `train_network_health_model` (also writes `outputs/network_health/reports/network_health_report.md`) — all resolvable by registered dataset id (`--dataset`); verified end-to-end on synthetic telemetry (`datasets/samples/network_health_synthetic.csv`) with injected interface degradation — recall 1.0, ROC-AUC 0.97

### Engine C — Network Configuration Intelligence (Offline)

- Modular, **offline and read-only** configuration subsystem (`src/network_config/`) that turns saved device command outputs into a structured inventory and a derived topology — no live device access, SNMP polling or remediation (those are later, gated phases with no code path in this phase)
- Typed models (`models.py`) for device, interface, VLAN, trunk, PoE, neighbor, MAC entry and STP state, aggregated into a per-device snapshot and a network inventory
- Tolerant parsers (`parsers.py`) for `show interface status`, `show vlan brief`, `show interfaces trunk`, `show lldp/cdp neighbors`, `show mac address-table`, `show power inline`, `show spanning-tree` and running-config identity — header-position slicing preserves fields with internal spaces (e.g. `Gig 0/1`, powered-device names), missing columns and VLAN ranges are handled, and a missing input file is warned about and skipped
- Inventory builder (`inventory.py`) merges parsed outputs, enriches interfaces with PoE state and derives access/trunk ports, MAC presence, unused/down ports and STP state counts; configuration-driven filenames (`configs/network_config.yaml`)
- Topology construction (`topology.py`) from the parsed inventory: high-confidence LLDP/CDP edges (deduplicated across protocols; reversed duplicate links collapse to one bidirectional edge with interface/hostname normalization), plus **conservative** MAC-table and STP hints represented as low/medium-confidence signals and warnings — MAC-derived adjacency is never claimed as certain
- Topology warnings: LLDP/CDP protocol mismatch, unidirectional neighbor, trunk with no discovered neighbor, access port exceeding a configurable MAC threshold, a MAC learned on multiple interfaces (loop risk), STP blocking on an access port and trunk ports missing STP data — each with severity, category and evidence, and deterministic ids
- YAML-driven rule engine (`rules.py`, `findings.py`) that evaluates the inventory and topology against configurable rules (`configs/network_rules.yaml`) and emits structured, detection-only findings — VLAN (disallowed access VLAN, trunk missing/unauthorized VLAN, native VLAN mismatch), port-state (unused-but-enabled ports, access ports with too many MACs), PoE (disabled on a port whose description implies a powered device), STP/loop-risk (blocking on an access port, MAC on multiple interfaces) and topology (trunk without a discovered neighbor). Rules, severities, thresholds, expected VLANs/keywords and suppression (by rule id, device/interface or tag) all live in YAML; findings carry deterministic ids and "candidate/possible" wording for inferred issues, and topology-dependent rules skip gracefully when topology is unavailable
- Per-snapshot artifacts under `outputs/network_config/<snapshot_id>/`: `inventory.json`, `metadata.json`, `network_config_report.md` (with topology and rule-findings summary sections), one CSV per object type (`interfaces`, `vlans`, `trunks`, `neighbors`, `mac_table`, `poe_status`, `stp_state`), the topology set (`topology.json`, `topology_nodes.csv`, `topology_edges.csv`, `topology_warnings.csv`) and the rule set (`findings.json`, `findings.csv`, `rule_summary.json`)
- Entry point `python -m scripts.analyze_network_config`: inventory always runs, topology and rules run automatically (`--skip-topology` / `--skip-rules` to opt out, `--rules-config` for a custom rule file); verified on synthetic saved outputs (`datasets/samples/network_config/`)

### Software Quality

- Unit testing
- Configuration validation
- Dataset integrity verification
- Reproducibility support

---

## 🚧 Current Phase

The full intrusion-detection stack is operational end-to-end: benchmarked
models on all three datasets, explainability, error analysis, visualization,
hyperparameter optimization, a model registry with promoted production models
per dataset, and a batch inference API serving them. Models are trained
manually on local hardware.

Engine B network-health prediction and Engine C offline network-configuration
analysis are underway alongside it. Next up:

- Extend explainability backends to LightGBM and the deep models
- Engine B network-health prediction over SNMP telemetry (LSTM autoencoder,
  live polling)
- Engine C human-approved remediation planning on top of the rule findings
  (still offline / read-only; no execution)
- Correlation engine combining cyber, network-health and configuration signals
- Monitoring dashboard

---

# Planned Features

## Models

- Logistic Regression, Decision Tree, Random Forest, CatBoost (classical additions)
- Autoencoder-based anomaly detection
- Engine B network-health models (SNMP/MIB telemetry)

## Evaluation

- PR-AUC
- Cross-validation
- Statistical model comparison

## Deployment

- Docker support
- Real-time monitoring
- Streaming/near-real-time inference
- Monitoring dashboard

---

# Project Structure

```text
NIMS/
│
├── configs/            # YAML configuration (one file per subsystem)
├── datasets/           # raw vendor datasets (read-only) + samples/
├── notebooks/
├── outputs/            # all generated artifacts (experiments, reports, registry, ...)
├── scripts/            # CLI entry points
├── src/
│   ├── data/           # loaders, validation, cleaning, encoding, scaling, splitting
│   ├── features/       # variance/correlation/statistical selection, PCA
│   ├── models/         # model wrappers + registry (classical & deep_learning/)
│   ├── training/       # trainer, metrics, experiment tracking, feature audit, reporting
│   ├── explainability/ # SHAP backends and artifacts
│   ├── error_analysis/ # confusion/per-class metrics, misclassified examples
│   ├── visualization/  # plots rendered from persisted artifacts
│   ├── optimization/   # Optuna search spaces, objective, studies
│   ├── registry/       # model registry, promotion, resolver
│   ├── api/            # FastAPI batch inference service
│   ├── network_health/ # Engine B telemetry: adapters, validation, features, baseline
│   ├── network_config/ # Engine C offline config parsing, inventory, reporting
│   └── utils/          # config, paths, hardware, io, logging, seed
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

Generate post-hoc analysis artifacts for a completed run (each also runs
automatically after training where configured):

```bash
python -m scripts.run_explainability --dataset unsw_nb15 --model xgboost
python -m scripts.run_error_analysis --dataset unsw_nb15 --model xgboost
python -m scripts.run_visualizations --dataset unsw_nb15 --model xgboost
```

Tune hyperparameters (validation-split objective; optionally trains the best
parameters as a normal experiment with optimization provenance):

```bash
python -m scripts.run_optimization --dataset unsw_nb15 --model xgboost --n-trials 20
```

Build the model registry, promote a production model and resolve it:

```bash
python -m scripts.build_model_registry
python -m scripts.promote_model --dataset unsw_nb15 --model xgboost --reason "Best validated baseline"
python -m scripts.resolve_model --dataset unsw_nb15 --json
```

Serve the promoted production models for batch inference:

```bash
python -m scripts.run_api          # or: uvicorn src.api.app:app
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models
curl -X POST http://127.0.0.1:8000/predict/unsw_nb15 -F "file=@datasets/samples/unsw_nb15_sample.csv"
```

Prepare and analyze network-health (Engine B) telemetry. Datasets are
registered in `configs/network_health.yaml`; adapters convert raw sources to
the canonical schema before validation, preprocessing and training:

```bash
python -m scripts.prepare_network_health_dataset --dataset snmp_mib_2016 --inspect
python -m scripts.prepare_network_health_dataset --dataset snmp_mib_2016
python -m scripts.validate_network_health --dataset snmp_mib_2016
python -m scripts.run_network_health_preprocessing --dataset snmp_mib_2016
python -m scripts.train_network_health_model --dataset snmp_mib_2016 --model isolation_forest
```

The bundled `synthetic` dataset runs the same chain without any raw files
(`--dataset synthetic`). Network-health artifacts are written under
`outputs/network_health/` (validation, adapter reports, processed splits,
features, experiments, report).

Analyze saved network-device command outputs offline (Engine C, read-only)
into a structured inventory:

```bash
python -m scripts.analyze_network_config \
    --input-dir datasets/samples/network_config --snapshot-id sample_offline
```

Outputs are written under `outputs/network_config/<snapshot_id>/`
(`inventory.json`, `metadata.json`, `network_config_report.md`, one CSV per
object type, the topology set `topology.json` / `topology_nodes.csv` /
`topology_edges.csv` / `topology_warnings.csv`, and the rule set
`findings.json` / `findings.csv` / `rule_summary.json`). Topology and rule
evaluation run automatically; pass `--skip-topology` and/or `--skip-rules` to
opt out, and `--rules-config` to point at a custom rule file. Input filenames
and topology thresholds live in `configs/network_config.yaml`; rule definitions,
severities, thresholds and suppression live in `configs/network_rules.yaml`.
Missing inputs are reported and skipped.

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
- ✅ Dataset ingestion, validation and auditing
- ✅ Data preprocessing
- ✅ Feature engineering
- ✅ Engine A baseline model framework (XGBoost, LightGBM, Isolation Forest)
- ✅ Deep-learning model framework (MLP, CNN, LSTM, Transformer)
- ✅ Full cross-model benchmark on all three datasets
- ✅ Model explainability (SHAP: global + per-sample)
- ✅ Error analysis (per-class metrics, hardest classes, misclassified examples)
- ✅ Visualization artifacts
- ✅ Hyperparameter optimization (Optuna)
- ✅ Model registry with production promotion
- ✅ Batch inference API (FastAPI)
- 🚧 Engine B network-health prediction (foundation complete: config-driven
  dataset adapters, telemetry validation, chronological preprocessing, health
  features, Isolation Forest baseline; LSTM autoencoder and live SNMP polling
  next)
- 🚧 Engine C network configuration intelligence (offline, read-only:
  command-output parsing, typed inventory, LLDP/CDP/MAC/STP topology and a
  YAML-driven rule engine producing structured findings; remediation planning
  next)
- ⏳ Correlation engine (cyber + network health + configuration)
- ⏳ Real-time monitoring dashboard
- ⏳ Docker / deployment hardening

---

# Current Status

**Current Development Stage:** Serving & Network-Health Expansion

The intrusion-detection stack is complete end-to-end. All seven models are benchmarked on NSL-KDD, UNSW-NB15 and CICIDS2017 (trained manually on local hardware), with XGBoost promoted as the production model for every dataset through the file-based model registry. Each completed experiment carries SHAP explanations, per-class error analysis and rendered visualizations, and the FastAPI service performs batch inference by replaying the saved preprocessing and feature-engineering transforms — validated against raw UNSW-NB15 samples. Engine B (network-health prediction over SNMP telemetry) has a working foundation with dataset adapters, and Engine C has an offline, read-only network-configuration parser producing a structured inventory, a derived LLDP/CDP/MAC/STP topology and a YAML-driven rule engine that emits structured configuration findings. The next milestones extend both engines — network-health LSTM modeling and Engine C's configuration rule engine and remediation planning — ahead of the correlation engine and monitoring dashboard.

---

# License

This project is intended for educational, research, and experimental purposes.