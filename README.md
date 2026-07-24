# NetSentinel (NIMS)

**AI-powered Network Intrusion Detection, Health Prediction & Configuration Intelligence Platform**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Tests](https://img.shields.io/badge/tests-70%2B%20modules-brightgreen)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

NetSentinel is a production-grade network security and operations platform that detects cyber attacks, predicts network degradation, analyzes device configurations, and correlates all three domains into unified, explainable incidents. It is engineered as an industrial software system — configuration-driven, fully tested, reproducible — not a collection of notebooks.

---

## Capabilities

| Component | What it does |
|---|---|
| **Engine A — Intrusion Detection** | Detects and classifies malicious traffic with 7 benchmarked models (XGBoost, LightGBM, Isolation Forest, MLP, CNN, LSTM, Transformer) across NSL-KDD, UNSW-NB15 and CIC-IDS2017 |
| **Engine B — Network Health** | Predicts interface/device degradation from SNMP/MIB telemetry using leakage-free chronological pipelines and anomaly models |
| **Engine C — Configuration Intelligence** | Parses device command output (Cisco IOS & Huawei VRP) into typed inventory, topology, YAML-rule findings and dry-run remediation plans with rollback and audit logging |
| **Correlation Engine** | Fuses Engine A/B/C signals and industrial syslog evidence into deterministic, evidence-backed incidents with severity and confidence scoring |
| **Live Ingestion (read-only)** | Polls and receives real device telemetry — Hirschmann SNMPv3 / traps / SSH config retrieval, Sophos Firewall syslog, Sophos Central SIEM API, Huawei VRP config retrieval — strictly read-only, credential-safe, offline-first |
| **Interfaces** | React web console, FastAPI batch-inference API, Streamlit operations dashboard |

---

## Benchmark Results

All models trained manually on local hardware (NVIDIA RTX 4060, CUDA) through the reproducible experiment pipeline. Best production model per dataset (test split):

| Dataset | Production Model | F1 | Notes |
|---|---|---|---|
| NSL-KDD | XGBoost | **0.9925** | LightGBM statistically adjacent (0.9923) |
| UNSW-NB15 | XGBoost | **0.9244** | Binary objective auto-reconciled |
| CIC-IDS2017 | XGBoost | **0.9988** | LightGBM 0.9987 |

Deep models (MLP/CNN/LSTM/Transformer) land 0.7–1.4 pp behind the boosted trees at 1–2 orders of magnitude more training time. Every run records precision, recall, F1, ROC-AUC, FPR, confusion matrix, timings and hardware; completed experiments carry SHAP explanations, per-class error analysis and rendered visualizations. Engine B validated on telemetry with injected degradation: recall 1.0, ROC-AUC 0.97.

---

## Architecture

Five isolated layers communicating only through persisted, versioned artifacts:

```
  Data ──▶ Features ──▶ Training ──▶ Evaluation ──▶ Serving
   │                        │             │             │
   ▼                        ▼             ▼             ▼
 validation,          experiments,   SHAP, error    registry,
 cleaning,            manifests,     analysis,      FastAPI,
 encoding, splits     metrics        visualization  dashboards

  Engine A (cyber)   Engine B (health)   Engine C (config)   Syslog
        └──────────────┴────────┬────────┴─────────────────────┘
                                ▼
                       Correlation Engine ──▶ Unified Incidents
                                ▼
              Web Console · Streaming Monitor · Reports
```

Design invariants enforced across the codebase:

- **Configuration over code** — every path, threshold, hyperparameter and rule lives in `configs/*.yaml`; nothing site-specific is hardcoded.
- **No data leakage** — encoders, scalers and selectors fit on train only; Engine B splits are strictly chronological; a pre-fit feature audit guards every model fit.
- **Artifact-driven integration** — downstream layers (correlation, dashboards, API) only read persisted artifacts; they never recompute or mutate upstream results.
- **Reproducibility** — seeded runs, config snapshots, dataset fingerprints, never-overwritten experiment directories, searchable experiment index.
- **Safety by construction** — detection is separated from action; remediation is dry-run with mandatory rollback/verification; the live layer is read-only and gated (see [Safety Model](#safety-model)).

---

## Project Structure

```
netsen/
├── configs/               # YAML configuration — one file per subsystem
├── datasets/              # raw datasets (read-only) + bundled offline samples
├── docs/                  # engine docs, safety audit, live-integration guide, project report
├── outputs/               # all generated artifacts (experiments, registry, reports, exports)
├── scripts/               # 40+ CLI entry points (python -m scripts.<name>)
├── src/
│   ├── data/              # loaders, validation, audit, cleaning, encoding, scaling, splits
│   ├── features/          # variance/correlation filters, MI/chi²/ANOVA, RFE, PCA
│   ├── models/            # classical + deep-learning wrappers behind one interface
│   ├── training/          # trainer, metrics, experiment tracking, feature audit
│   ├── explainability/    # SHAP backends and artifacts
│   ├── error_analysis/    # per-class metrics, hardest classes, misclassified examples
│   ├── visualization/     # plots rendered from persisted artifacts
│   ├── optimization/      # Optuna search spaces, objectives, studies
│   ├── registry/          # model registry, promotion, production resolution
│   ├── api/               # FastAPI batch-inference service
│   ├── network_health/    # Engine B: schema, adapters, chronological features, models
│   ├── network_config/    # Engine C: parsers, inventory, topology, rules, remediation
│   │   └── vendors/       # vendor parser packs (Huawei VRP)
│   ├── correlation/       # cross-engine + syslog incident correlation
│   ├── syslog_ingestion/  # industrial switch syslog → events, features, findings
│   ├── live_logging/      # read-only live adapters (SNMPv3, traps, syslog, SSH, API)
│   ├── streaming/         # replay engine + live-monitor state
│   ├── dashboard/         # Streamlit operations viewer
│   ├── ml_workflow/       # workflow planner console
│   └── utils/             # config, hardware detection, logging, io, seed
├── tests/                 # 70+ test modules — no live hardware required
├── webapp/
│   ├── server/            # Node/Express read-only artifact API (port 8050)
│   └── frontend/          # React (Vite) console (dev port 5175)
├── pyproject.toml         # build, Black/isort/Ruff/mypy/pytest configuration
└── requirements.txt
```

---

## Installation

**Prerequisites:** Python 3.11+ · Node.js 18+ (web console) · NVIDIA GPU with CUDA (optional — CPU fallback is automatic)

```bash
git clone <repository-url> netsentinel
cd netsentinel

# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 2. PyTorch — install the build matching your CUDA version (see pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Web console dependencies
cd webapp/server   && npm install && cd ../..
cd webapp/frontend && npm install && cd ../..
```

**Datasets** (Engine A training only — every other subsystem ships with offline samples): place NSL-KDD, UNSW-NB15 and CIC-IDS2017 under `datasets/` as registered in `configs/data.yaml`. Raw data is never modified or committed.

Verify the installation:

```bash
pytest                                        # full offline test suite
python -m scripts.validate_engine_c_safety    # static safety audit
```

---

## Quick Start

Prepare every artifact the frontend needs in one command, then launch the console:

```bash
python -m scripts.prepare_full_demo --skip-training   # reuse existing models
# or: python -m scripts.prepare_full_demo --dry-run   # inspect the plan first

# Terminal 1 — artifact API on http://127.0.0.1:8050
cd webapp/server && npm start

# Terminal 2 — React console on http://127.0.0.1:5175
cd webapp/frontend && npm run dev
```

The console provides an executive overview, live monitoring, training/validation explorer, per-engine sections, correlation incidents, history and a safety/audit page. For production, `npm run build` in `webapp/frontend` lets the API server serve the built frontend directly on port 8050.

---

## Core Workflows

### Engine A — intrusion detection pipeline

```bash
python -m scripts.validate_datasets --all           # schema + integrity checks
python -m scripts.run_preprocessing --dataset nsl_kdd
python -m scripts.run_feature_engineering --dataset nsl_kdd
python -m scripts.train_model --dataset nsl_kdd --model xgboost   # GPU auto-detected
python -m scripts.generate_validation_report        # cross-model benchmark report
python -m scripts.run_optimization --dataset nsl_kdd --model xgboost --n-trials 20
```

Explainability, error analysis and visualization run automatically after training and post-hoc via `run_explainability` / `run_error_analysis` / `run_visualizations`.

### Model registry & inference API

```bash
python -m scripts.build_model_registry
python -m scripts.promote_model --dataset unsw_nb15 --model xgboost --reason "Best validated"
python -m scripts.run_api                           # FastAPI on :8000
curl -X POST http://127.0.0.1:8000/predict/unsw_nb15 -F "file=@samples.csv"
```

Inference replays the saved preprocessing and feature transforms — nothing is fitted at serve time.

### Engine B — network health

```bash
python -m scripts.prepare_network_health_dataset --dataset snmp_mib_2016
python -m scripts.validate_network_health --dataset snmp_mib_2016
python -m scripts.run_network_health_preprocessing --dataset snmp_mib_2016
python -m scripts.train_network_health_model --dataset snmp_mib_2016 --model isolation_forest
```

Config-driven adapters map any raw telemetry export (SNMP-MIB 2016, LCORE-D, canonical CSV) onto one schema; the bundled `synthetic` dataset runs the full chain with no downloads.

### Engine C — configuration intelligence

```bash
# Cisco-style snapshot                       # Huawei VRP snapshot
python -m scripts.analyze_network_config \
    --input-dir datasets/samples/network_config --snapshot-id assessment_01
python -m scripts.analyze_network_config --vendor huawei \
    --input-dir <saved-display-outputs> --snapshot-id huawei_01

python -m scripts.dry_run_network_actions --snapshot-id assessment_01
python -m scripts.compare_network_snapshots --before before_01 --after after_01
python -m scripts.generate_network_config_report --snapshot-id assessment_01
python -m scripts.export_network_config_dashboard --snapshot-id assessment_01
```

One command produces inventory, topology (LLDP/CDP/MAC/STP), rule findings and a dry-run remediation plan in which every action carries a rollback, a verification step and a risk level. **No command is ever executed against a device.**

### Correlation, syslog & streaming

```bash
python -m scripts.ingest_switch_syslog --input-dir datasets/raw/syslog --run-id run_01
python -m scripts.run_correlation --engine-c-snapshot assessment_01 \
    --engine-b-dataset synthetic --engine-a-dataset unsw_nb15 --syslog-run latest
python -m scripts.run_streaming_demo --no-sleep     # replay into the live monitor
```

### Live ingestion (read-only)

Every source defaults to offline/mock mode and contacts nothing. Live mode requires per-source `enabled: true`, `mode: live`, `read_only: true` and credentials from environment variables — never from files or CLI:

```bash
python -m scripts.check_live_readiness --source hirschmann_snmp   # never connects
python -m scripts.run_live_logger --source hirschmann_snmp --live --run-once
```

Supported sources: `hirschmann_snmp` (SNMPv3 GET, profile-restricted OIDs), `hirschmann_traps`, `hirschmann_config` (read-only SSH, host-key verified, command allowlist), `sophos_firewall_syslog` (UDP receiver with source allowlist), `sophos_central` (SIEM API). Huawei VRP switches are supported via SNMPv3 polling and read-only `display` retrieval (`scripts/retrieve_huawei_config.py`), feeding Engine C directly. Setup per source: [`docs/live_integration_setup.md`](docs/live_integration_setup.md).

---

## Safety Model

The platform is operationally conservative by construction:

- **Read-only everywhere** — no SNMP SET, no configuration mode, no write command, no remediation execution path exists in the codebase; a static audit (`validate_engine_c_safety`) enforces it.
- **Detection ≠ action** — remediation plans are generated artifacts; the dry-run executor validates them and writes an append-only `action_audit_log.jsonl` with `executed=false` on every record.
- **Credential hygiene** — secrets come only from environment variables and are never logged, persisted, echoed to the frontend, or accepted via CLI; local target files are gitignored.
- **Cautious claims** — inferred issues are worded `candidate`/`possible`; verification returns `unknown` over false confidence; unreliable device clocks demote incident confidence.

---

## Quality & Testing

- **70+ pytest modules** covering parsing, pipelines, rules, remediation, execution refusal, correlation, live-adapter mocks, redaction and safety gates — no test touches live hardware.
- **Static tooling** via `pyproject.toml`: Black, isort, Ruff (pycodestyle/pyflakes/bugbear/pydocstyle), mypy with `disallow_untyped_defs`.
- **Structured logging** throughout (`configs/logging.yaml`); no `print()` in production code; exceptions are logged with context, never suppressed.

```bash
pytest                    # run everything
ruff check src scripts    # lint
mypy                      # type-check
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/NetSentinel_Project_Report.docx`](docs/NetSentinel_Project_Report.docx) | Complete technical report: architecture, stack, pipelines, module implementation |
| [`docs/engine_c_network_config.md`](docs/engine_c_network_config.md) | Engine C workflow reference |
| [`docs/engine_c_safety_audit.md`](docs/engine_c_safety_audit.md) | Safety model and audit evidence |
| [`docs/engine_c_integration_handoff.md`](docs/engine_c_integration_handoff.md) | Consuming Engine C artifacts downstream |
| [`docs/live_integration_setup.md`](docs/live_integration_setup.md) | Per-source live ingestion setup |

---

## License

This project is intended for educational, research and authorized operational use.
