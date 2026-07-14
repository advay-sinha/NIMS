"""Pure-Python readiness checks for the offline demo (no Streamlit imports).

Every function reads already-persisted artefacts and answers "is this section
ready for the frontend?" — it never runs a pipeline, contacts a device or
imports Streamlit. The dashboard readiness checker calls the existing
Streamlit-free dashboard loaders so it validates exactly what the frontend will.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Core Engine C assessment artefacts (produced by analyze_network_config).
ENGINE_C_CORE = (
    "inventory.json",
    "findings.json",
    "remediation_plan.json",
)
# Dashboard views (produced by export_network_config_dashboard).
ENGINE_C_DASHBOARD = (
    "dashboard/dashboard_summary.json",
    "dashboard/findings_view.json",
    "dashboard/topology_view.json",
    "dashboard/remediation_view.json",
)
# Everything the frontend needs for a usable assessment.
ENGINE_C_REQUIRED = ENGINE_C_CORE + ENGINE_C_DASHBOARD

CORRELATION_REQUIRED = (
    "signals.json", "incidents.json", "correlation_summary.json",
    "correlation_report.md",
)

STREAMING_REQUIRED = ("current/current_state.json", "stream_summary.json")


def _exists_all(root: Path, names) -> tuple[bool, list[str], list[str]]:
    present, missing = [], []
    for name in names:
        (present if (root / name).is_file() else missing).append(name)
    return (not missing), present, missing


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------- Engine C
def engine_c_ready(network_config_dir: str | Path, snapshot: str,
                   include_dashboard: bool = True) -> dict[str, Any]:
    """Whether the Engine C artefacts for ``snapshot`` are complete.

    ``include_dashboard=False`` checks only the core assessment artefacts
    (before the dashboard export stage has run).
    """
    root = Path(network_config_dir) / snapshot
    required = ENGINE_C_REQUIRED if include_dashboard else ENGINE_C_CORE
    ready, present, missing = _exists_all(root, required)
    return {"ready": ready and root.is_dir(), "snapshot": snapshot,
            "present": present, "missing": missing, "path": str(root)}


# --------------------------------------------------------------- Engine A
def engine_a_ready(registry_dir: str | Path, datasets) -> dict[str, Any]:
    """Per-dataset production-model presence from the registry."""
    production = _read_json(Path(registry_dir) / "production.json")
    production = production if isinstance(production, dict) else {}
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        entry = production.get(dataset)
        ready = isinstance(entry, dict) and bool(entry.get("experiment_id"))
        per_dataset[dataset] = {
            "ready": ready,
            "model_type": (entry or {}).get("model_type") if ready else None,
            "experiment_id": (entry or {}).get("experiment_id") if ready else None,
        }
    return {"ready": all(d["ready"] for d in per_dataset.values()) and bool(datasets),
            "datasets": per_dataset,
            "ready_datasets": [d for d, v in per_dataset.items() if v["ready"]],
            "missing_datasets": [d for d, v in per_dataset.items() if not v["ready"]]}


# --------------------------------------------------------------- Engine B
def engine_b_ready(network_health_dir: str | Path, dataset: str) -> dict[str, Any]:
    """Whether a usable Engine B experiment exists for ``dataset``."""
    exp_root = Path(network_health_dir) / "experiments" / dataset
    runs = []
    if exp_root.is_dir():
        runs = sorted(p for p in exp_root.glob("*/*")
                      if (p / "metrics.json").is_file())
    report = Path(network_health_dir) / "reports" / "network_health_report.md"
    return {"ready": bool(runs), "dataset": dataset,
            "experiment": runs[-1].name if runs else None,
            "report_available": report.is_file()}


# --------------------------------------------------------------- syslog
def syslog_ready(outputs_dir: str | Path, run: str) -> dict[str, Any]:
    """Resolve the syslog ingestion run (``latest`` supported); optional."""
    from src.correlation.syslog_loader import resolve_run_id

    syslog_dir = Path(outputs_dir) / "syslog_ingestion"
    run_id = resolve_run_id(syslog_dir, run)
    return {"ready": run_id is not None, "run_id": run_id,
            "path": str(syslog_dir / run_id) if run_id else None,
            "optional": True}


# --------------------------------------------------------------- correlation
def correlation_ready(correlation_dir: str | Path, correlation_id: str
                      ) -> dict[str, Any]:
    root = Path(correlation_dir) / correlation_id
    ready, present, missing = _exists_all(root, CORRELATION_REQUIRED)
    return {"ready": ready and root.is_dir(), "correlation_id": correlation_id,
            "present": present, "missing": missing, "path": str(root)}


# --------------------------------------------------------------- streaming
def streaming_ready(streaming_dir: str | Path) -> dict[str, Any]:
    root = Path(streaming_dir)
    ready, present, missing = _exists_all(root, STREAMING_REQUIRED)
    events = root / "events.jsonl"
    return {"ready": ready, "present": present, "missing": missing,
            "events_log_available": events.is_file(), "path": str(root)}


# --------------------------------------------------------------- dashboard
def dashboard_readiness(paths, snapshot: str, correlation_id: str,
                        syslog_run: str) -> dict[str, Any]:
    """Validate every frontend section via the existing (Streamlit-free) loaders.

    Returns a per-section availability map plus an overall flag and the safety
    banner check. Imports no Streamlit.
    """
    from src.dashboard import loader as dash
    from src.streaming.artifacts import load_current_state

    engine_c = dash.load_engine_c_dashboard(paths.network_config_dir, snapshot)
    correlation = dash.load_correlation(paths.correlation_dir, correlation_id)
    engine_a = dash.load_engine_a(paths.registry_dir, paths.reports_dir,
                                  paths.error_analysis_dir,
                                  paths.visualizations_dir, paths.experiments_dir)
    engine_b = dash.load_engine_b(paths.network_health_dir)
    syslog_dir = Path(paths.outputs_dir) / "syslog_ingestion"
    syslog_run_id = dash.latest_syslog_run(syslog_dir) if syslog_run == "latest" \
        else syslog_run
    syslog = dash.load_syslog_run(syslog_dir, syslog_run_id) if syslog_run_id else \
        {"available": False}
    current_state = load_current_state(Path(paths.outputs_dir) / "streaming"
                                       / "current")

    views = engine_c.get("views", {}) or {}
    incidents = correlation.get("incidents", []) or []
    safety = (current_state.get("safety") or {}) if isinstance(current_state, dict) \
        else {}
    safety_ok = bool(safety.get("offline_only")) and \
        bool(safety.get("no_command_execution"))

    sections = {
        "Executive Overview": bool(engine_c.get("available")
                                   or correlation.get("available")),
        "Correlated Incidents": bool(correlation.get("available")),
        "Engine A Cybersecurity": bool(engine_a.get("available")),
        "Engine B Network Health": bool(engine_b.get("available")),
        "Engine C Assessment": bool(engine_c.get("available")),
        "Topology": bool(views.get("topology_view")),
        "Remediation / Dry-Run Audit": bool(views.get("remediation_view")),
        "Syslog Activity": bool(syslog.get("available")),
        "Streaming / Live Monitor": bool(
            isinstance(current_state, dict)
            and current_state.get("available", True)
            and current_state.get("total_events") is not None),
        "Safety / Audit": safety_ok,
    }
    optional_sections = {"Syslog Activity"}
    missing_required = [name for name, ok in sections.items()
                        if not ok and name not in optional_sections]

    return {
        "ready": not missing_required,
        "sections": sections,
        "optional_sections": sorted(optional_sections),
        "missing_required_sections": missing_required,
        "safety_banner_ok": safety_ok,
        "incident_count": len(incidents),
        "syslog_available": bool(syslog.get("available")),
        "clock_integrity_warning": any(
            i.get("time_reliability", "reliable") != "reliable"
            for i in incidents),
    }
