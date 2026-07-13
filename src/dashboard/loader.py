"""Read-only artefact loaders for the monitoring dashboard (pure Python).

No Streamlit dependency. Every loader tolerates absent folders/files, returns
``available: False`` with a helpful message when an artefact set is missing, and
never recomputes, trains, infers, polls, captures or executes anything — it only
reads what other phases already wrote to disk.
"""

from __future__ import annotations

import json
import logging
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Files written by scripts.export_network_config_dashboard (2 optional).
_ENGINE_C_VIEWS: tuple[str, ...] = (
    "dashboard_summary", "inventory_view", "topology_view", "findings_view",
    "remediation_view", "action_audit_view", "risk_timeline",
    "device_health_cards", "export_metadata")
_ENGINE_C_OPTIONAL: tuple[str, ...] = ("diff_view", "verification_view")

# Directory names under network_config/ that are not device snapshots.
_NON_SNAPSHOT_DIRS: frozenset[str] = frozenset({"diffs"})


def _read_json(path: Path) -> Any:
    """Read JSON tolerantly; return ``None`` on any problem (never raises)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def load_dashboard_config(path: str | Path) -> dict[str, Any]:
    """Load ``configs/dashboard.yaml`` (tolerant: returns {} when absent)."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        import yaml
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError) as exc:
        logger.warning("could not read dashboard config %s: %s", p, exc)
        return {}


# ------------------------------------------------------------ source listings


def list_engine_c_snapshots(network_config_dir: str | Path) -> list[str]:
    """List Engine C snapshot ids (dirs), ignoring non-snapshot dirs like diffs."""
    root = Path(network_config_dir)
    if not root.is_dir():
        return []
    snaps = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in _NON_SNAPSHOT_DIRS:
            continue
        if (child / "dashboard").is_dir() or (child / "inventory.json").is_file():
            snaps.append(child.name)
    return snaps


def list_correlation_runs(correlation_dir: str | Path) -> list[str]:
    """List correlation run ids (dirs that contain a summary or incidents file)."""
    root = Path(correlation_dir)
    if not root.is_dir():
        return []
    runs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if ((child / "correlation_summary.json").is_file()
                or (child / "incidents.json").is_file()):
            runs.append(child.name)
    return runs


# ------------------------------------------------- friendly labels / latest


def humanize_timestamp(value: Any) -> Optional[str]:
    """Format an ISO timestamp as ``YYYY-MM-DD HH:MM UTC`` (tolerant)."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        from datetime import datetime
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _snapshot_timestamp(network_config_dir: Path, snapshot: str) -> Optional[str]:
    """Best-effort assessment timestamp for a snapshot from its metadata."""
    candidates = (
        ("dashboard/export_metadata.json", "generated_at"),
        ("dashboard/dashboard_summary.json", "generated_at"),
        ("metadata.json", "generated_at"),
        ("metadata.json", "timestamp"),
        ("metadata.json", "created_at"),
    )
    for rel, key in candidates:
        data = _read_json(network_config_dir / snapshot / rel)
        if isinstance(data, dict) and data.get(key):
            return str(data[key])
    return None


def _correlation_timestamp(correlation_dir: Path, run: str) -> Optional[str]:
    data = _read_json(correlation_dir / run / "correlation_summary.json")
    if isinstance(data, dict) and data.get("timestamp"):
        return str(data["timestamp"])
    return None


def _labeled(ids: list[str], timestamp_fn: Callable[[str], Optional[str]],
             kind: str) -> list[dict[str, Any]]:
    """Attach friendly labels + timestamps to ids, newest first, latest marked.

    ``kind`` is the human noun (e.g. ``"Assessment Run"``) used when no metadata
    timestamp is available so the raw id is never shown as a bare string.
    """
    items = []
    for run_id in ids:
        ts = timestamp_fn(run_id)
        human = humanize_timestamp(ts)
        label = f"{kind} · {human}" if human else f"{kind} · {run_id}"
        items.append({"id": run_id, "timestamp": ts, "human": human,
                      "label": label, "is_latest": False})
    # Newest first: sort by timestamp desc (missing sorts last), then id desc.
    items.sort(key=lambda i: (i["timestamp"] or "", i["id"]), reverse=True)
    if items:
        items[0]["is_latest"] = True
        items[0]["label"] += " (latest)"
    return items


def labeled_snapshots(network_config_dir: str | Path) -> list[dict[str, Any]]:
    """Assessment runs (Engine C snapshots) with friendly labels, newest first."""
    root = Path(network_config_dir)
    return _labeled(list_engine_c_snapshots(root),
                    lambda s: _snapshot_timestamp(root, s), "Assessment Run")


def labeled_correlation_runs(correlation_dir: str | Path) -> list[dict[str, Any]]:
    """Incident runs (correlation runs) with friendly labels, newest first."""
    root = Path(correlation_dir)
    return _labeled(list_correlation_runs(root),
                    lambda r: _correlation_timestamp(root, r), "Incident Run")


def latest_snapshot(network_config_dir: str | Path) -> Optional[str]:
    """Id of the most recent assessment run, or ``None`` when none exist."""
    items = labeled_snapshots(network_config_dir)
    return items[0]["id"] if items else None


def latest_correlation_run(correlation_dir: str | Path) -> Optional[str]:
    """Id of the most recent incident run, or ``None`` when none exist."""
    items = labeled_correlation_runs(correlation_dir)
    return items[0]["id"] if items else None


def resolve_default(items: list[dict[str, Any]],
                    configured: Optional[str] = None) -> Optional[str]:
    """Pick the default selection.

    The latest available run always wins (per the assessor UX: "default to the
    latest available run"). ``configured`` is only a fallback used when nothing
    is discoverable on disk, so the app can still reference it in empty-state
    guidance.
    """
    if items:
        return items[0]["id"]            # newest first -> latest
    return configured


# --------------------------------------------------------------- Engine C


def load_engine_c_dashboard(network_config_dir: str | Path,
                            snapshot_id: str) -> dict[str, Any]:
    """Load the Engine C dashboard-export views for one snapshot."""
    dash_dir = Path(network_config_dir) / snapshot_id / "dashboard"
    if not dash_dir.is_dir():
        return {
            "available": False, "snapshot_id": snapshot_id,
            "views": {}, "present": [], "missing": list(_ENGINE_C_VIEWS),
            "dry_run_executed_count": 0,
            "message": ("Dashboard exports not found. Run:\n"
                        "python -m scripts.export_network_config_dashboard "
                        f"--snapshot-id {snapshot_id}")}

    views: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    for name in _ENGINE_C_VIEWS:
        data = _read_json(dash_dir / f"{name}.json")
        if data is None:
            missing.append(name)
        else:
            views[name] = data
            present.append(name)
    for name in _ENGINE_C_OPTIONAL:
        data = _read_json(dash_dir / f"{name}.json")
        if data is not None:
            views[name] = data
            present.append(name)

    audit = views.get("action_audit_view") or {}
    executed = int(audit.get("executed_count", 0) or 0)
    message = None
    if missing:
        message = ("Some Engine C dashboard views are missing "
                   f"({', '.join(missing)}). Re-run: "
                   "python -m scripts.export_network_config_dashboard "
                   f"--snapshot-id {snapshot_id}")
    return {
        "available": bool(present), "snapshot_id": snapshot_id,
        "dashboard_dir": str(dash_dir), "views": views,
        "present": present, "missing": missing,
        "dry_run_executed_count": executed, "message": message}


# --------------------------------------------------------------- correlation


def load_correlation(correlation_dir: str | Path, correlation_id: str,
                     snapshot_hint: str = "<snapshot_id>") -> dict[str, Any]:
    """Load one correlation run's incidents, signals, summary and report path."""
    run_dir = Path(correlation_dir) / correlation_id
    summary = _read_json(run_dir / "correlation_summary.json")
    incidents = _read_json(run_dir / "incidents.json")
    signals = _read_json(run_dir / "signals.json")
    report = run_dir / "correlation_report.md"

    if summary is None and incidents is None:
        return {
            "available": False, "correlation_id": correlation_id,
            "incidents": [], "signals": [], "summary": {}, "report_path": None,
            "message": ("Correlation output not found. Run:\n"
                        "python -m scripts.run_correlation --engine-c-snapshot "
                        f"{snapshot_hint} --engine-b-dataset synthetic "
                        "--engine-a-dataset unsw_nb15 --correlation-id "
                        f"{correlation_id}")}
    return {
        "available": True, "correlation_id": correlation_id,
        "incidents": incidents if isinstance(incidents, list) else [],
        "signals": signals if isinstance(signals, list) else [],
        "summary": summary if isinstance(summary, dict) else {},
        "report_path": str(report) if report.is_file() else None,
        "message": None}


# --------------------------------------------------------------- Engine A


def load_engine_a(registry_dir: str | Path, reports_dir: str | Path,
                  error_analysis_dir: str | Path,
                  visualizations_dir: str | Path,
                  experiments_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Load Engine A production models and the paths of related artefacts."""
    production = _read_json(Path(registry_dir) / "production.json") or {}
    best = _read_json(Path(registry_dir) / "best_per_dataset.json") or {}

    models: list[dict[str, Any]] = []
    for dataset, entry in (production.items()
                           if isinstance(production, dict) else []):
        if not isinstance(entry, dict):
            continue
        best_entry = best.get(dataset) if isinstance(best, dict) else None
        test_f1 = (best_entry or {}).get("value") if isinstance(best_entry, dict) \
            else None
        roc = _engine_a_roc(experiments_dir, dataset, entry) \
            if experiments_dir else None
        models.append({
            "dataset": dataset,
            "model_type": entry.get("model_type"),
            "experiment_id": entry.get("experiment_id"),
            "test_f1": test_f1,
            "roc_auc": roc,
            "promoted_at": entry.get("promoted_at")})

    report = Path(reports_dir) / "model_validation_report.md"
    return {
        "available": bool(models),
        "production_model_count": len(models),
        "models": sorted(models, key=lambda m: str(m["dataset"])),
        "validation_report_path": str(report) if report.is_file() else None,
        "latest_error_analysis": _latest_dir(error_analysis_dir),
        "latest_visualization": _latest_dir(visualizations_dir),
        "message": None if models else "No Engine A production models registered."}


def _engine_a_roc(experiments_dir: str | Path, dataset: str,
                  entry: dict[str, Any]) -> Optional[float]:
    exp_id = entry.get("experiment_id")
    model = entry.get("model_type")
    if not (exp_id and model):
        return None
    metrics = _read_json(
        Path(experiments_dir) / dataset / model / exp_id / "metrics.json")
    test = metrics.get("test") if isinstance(metrics, dict) else None
    return test.get("roc_auc") if isinstance(test, dict) else None


# --------------------------------------------------------------- Engine B


def load_engine_b(network_health_dir: str | Path) -> dict[str, Any]:
    """Load the latest network-health experiment metrics per dataset."""
    root = Path(network_health_dir)
    exp_root = root / "experiments"
    datasets: list[dict[str, Any]] = []
    if exp_root.is_dir():
        for dataset_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
            entry = _latest_engine_b_experiment(dataset_dir)
            if entry:
                datasets.append(entry)

    report = root / "reports" / "network_health_report.md"
    anomaly = None
    if datasets:
        top = max(datasets, key=lambda d: d.get("anomaly_rate") or 0.0)
        anomaly = f"{top['dataset']}: {top['anomaly_rate']:.1%} anomalous (test)"
    return {
        "available": bool(datasets),
        "datasets": datasets,
        "anomaly_status": anomaly,
        "report_path": str(report) if report.is_file() else None,
        "message": None if datasets
        else "No Engine B network-health experiments found."}


def _latest_engine_b_experiment(dataset_dir: Path) -> Optional[dict[str, Any]]:
    runs = sorted((p for p in dataset_dir.glob("*/*")
                   if (p / "metrics.json").is_file()), key=lambda p: p.name)
    if not runs:
        return None
    run = runs[-1]
    metrics = _read_json(run / "metrics.json") or {}
    manifest = _read_json(run / "manifest.json") or {}
    test = metrics.get("test") if isinstance(metrics, dict) else {}
    test = test if isinstance(test, dict) else {}
    n_samples = int(test.get("n_samples", 0) or 0)
    n_pred = int(test.get("n_anomalous_predicted", 0) or 0)
    return {
        "dataset": dataset_dir.name,
        "experiment_id": manifest.get("experiment_id", run.name),
        "model_name": manifest.get("model_name"),
        "labeled": bool(manifest.get("labeled", False)),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1": test.get("f1"),
        "roc_auc": test.get("roc_auc"),
        "n_samples": n_samples,
        "n_anomalous_predicted": n_pred,
        "anomaly_rate": (n_pred / n_samples) if n_samples else 0.0}


# --------------------------------------------------------------- overview


def compute_overview(engine_c: dict[str, Any], correlation: dict[str, Any],
                     engine_a: dict[str, Any], engine_b: dict[str, Any]
                     ) -> dict[str, Any]:
    """Compute the top-level overview cards from the loaded artefact sets."""
    incidents = correlation.get("incidents", []) or []
    high_critical = sum(1 for i in incidents
                        if str(i.get("severity", "")).lower() in ("high", "critical"))
    ec_summary = (engine_c.get("views", {}) or {}).get("dashboard_summary") or {}
    return {
        "total_incidents": len(incidents),
        "high_critical_incidents": high_critical,
        "engine_c_findings": int(ec_summary.get("finding_count", 0) or 0),
        "remediation_actions_planned":
            int(ec_summary.get("remediation_action_count", 0) or 0),
        "dry_run_executed_count": int(engine_c.get("dry_run_executed_count", 0) or 0),
        "engine_b_anomaly_status": engine_b.get("anomaly_status") or "unavailable",
        "engine_a_production_models": int(engine_a.get("production_model_count", 0)),
    }


# --------------------------------------------------------- executive summary


def build_executive_summary(engine_c: dict[str, Any], correlation: dict[str, Any],
                            engine_b: dict[str, Any], engine_a: dict[str, Any]
                            ) -> dict[str, Any]:
    """Assessor-focused roll-up: status, critical incidents, causes, actions.

    Pure aggregation of already-loaded artefacts — no recomputation, no IO.
    """
    incidents = correlation.get("incidents", []) or []
    critical = [i for i in incidents
                if str(i.get("severity", "")).lower() in ("critical", "high")]
    focus = critical or incidents
    ec_summary = (engine_c.get("views", {}) or {}).get("dashboard_summary") or {}
    findings = int(ec_summary.get("finding_count", 0) or 0)

    status_level, status_text = _network_status(critical, findings, engine_b)

    devices: list[str] = []
    for inc in focus:
        for dev in inc.get("affected_devices", []) or []:
            if dev not in devices:
                devices.append(dev)
    for dev in ec_summary.get("top_risk_devices", []) or []:
        name = dev.get("device") if isinstance(dev, dict) else None
        if name and name not in devices:
            devices.append(name)

    causes: list[str] = []
    for inc in focus:
        hyp = inc.get("root_cause_hypothesis")
        if hyp and hyp not in causes:
            causes.append(hyp)

    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for inc in focus:
        for action in inc.get("recommended_actions", []) or []:
            title = action.get("title", "")
            if title and title not in seen:
                seen.add(title)
                actions.append({"title": title, "detail": action.get("detail", ""),
                                "owner": action.get("owner", "network")})

    return {
        "network_status": status_text,
        "network_status_level": status_level,
        "total_incidents": len(incidents),
        "critical_incident_count": len(critical),
        "critical_incidents": [
            {"incident_id": i.get("incident_id"), "severity": i.get("severity"),
             "title": i.get("title"), "rule_id": i.get("rule_id"),
             "devices": i.get("affected_devices", [])}
            for i in critical],
        "affected_devices": devices[:12],
        "likely_root_causes": causes[:5],
        "recommended_actions": actions[:6],
        "safety_status": {
            "offline": True,
            "no_command_execution": True,
            "no_live_device_access": True,
            "dry_run_executed": int(engine_c.get("dry_run_executed_count", 0) or 0),
        },
    }


def _network_status(critical: list[dict[str, Any]], findings: int,
                    engine_b: dict[str, Any]) -> tuple[str, str]:
    if critical:
        return ("attention",
                f"Attention required — {len(critical)} high/critical "
                "incident(s) correlated across engines.")
    anomaly = engine_b.get("anomaly_status")
    anomalous = bool(engine_b.get("available")) and bool(anomaly) \
        and "0.0%" not in str(anomaly)
    if findings or anomalous:
        return ("monitor",
                "Monitoring — configuration findings or network-health "
                "anomalies are present, but no critical incidents.")
    return ("stable",
            "Stable — no critical incidents detected in the selected runs.")


def describe_artifact_sources(engine_c: dict[str, Any], correlation: dict[str, Any],
                              dirs: dict[str, Any]) -> dict[str, Any]:
    """Raw ids and paths for the 'Advanced Artifact Sources' expander."""
    return {
        "assessment_run_id": engine_c.get("snapshot_id"),
        "engine_c_dashboard_dir": engine_c.get("dashboard_dir"),
        "engine_c_available": bool(engine_c.get("available")),
        "incident_run_id": correlation.get("correlation_id"),
        "correlation_report_path": correlation.get("report_path"),
        "correlation_available": bool(correlation.get("available")),
        "network_config_dir": str(dirs.get("network_config_dir", "")),
        "correlation_dir": str(dirs.get("correlation_dir", "")),
    }


# --------------------------------------------------------------- helpers


def _latest_dir(directory: str | Path) -> Optional[str]:
    root = Path(directory)
    if not root.is_dir():
        return None
    subdirs = sorted((p for p in root.iterdir() if p.is_dir()),
                     key=lambda p: p.name)
    return str(subdirs[-1]) if subdirs else None
