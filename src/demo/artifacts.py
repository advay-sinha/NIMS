"""Persistence of a demo-preparation run.

Writes the per-run artefacts under ``outputs/demo/<demo_run_id>/`` and updates
``outputs/demo/latest.json``. Pure serialisation of an already-computed plan /
result — nothing here executes a stage or mutates a source artefact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.demo.models import DemoConfig, StageResult
from src.utils.io import read_json, write_json

logger = logging.getLogger(__name__)


def collect_metrics(config: DemoConfig, paths) -> dict[str, Any]:
    """Read the produced correlation/streaming summaries for the report."""
    metrics: dict[str, Any] = {}

    corr = _safe_read(Path(paths.correlation_dir) / config.correlation_id
                      / "correlation_summary.json")
    if corr:
        metrics["correlation"] = {
            "total_signals": corr.get("total_signals"),
            "total_incidents": corr.get("total_incidents"),
            "incidents_with_syslog_evidence":
                corr.get("incidents_with_syslog_evidence"),
            "signals_by_engine": corr.get("signals_by_engine"),
            "incidents_by_rule": corr.get("incidents_by_rule"),
            "syslog_source": corr.get("syslog_source"),
        }

    stream = _safe_read(Path(paths.outputs_dir) / "streaming"
                        / "stream_summary.json")
    if stream:
        metrics["streaming"] = {
            "total_events": stream.get("total_events"),
            "active_incident_count": stream.get("active_incident_count"),
            "syslog_event_count": stream.get("syslog_event_count"),
            "clock_reliability_status": stream.get("clock_reliability_status"),
        }
    return metrics


def _safe_read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _generated_artifacts(stages: list[StageResult]) -> list[str]:
    """Distinct, existing artefact paths recorded across all stages."""
    seen: list[str] = []
    for stage in stages:
        for art in stage.artifacts:
            if art and Path(art).exists() and art not in seen:
                seen.append(art)
    return seen


def write_demo_run(config: DemoConfig, stages: list[StageResult],
                   readiness_roll: dict[str, Any], metrics: dict[str, Any],
                   demo_run_id: str, out_root: str | Path) -> dict[str, str]:
    """Persist every artefact for one demo-preparation run."""
    from src.demo.reporting import build_demo_report

    run_dir = Path(out_root) / demo_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    warnings = {s.name: s.warnings for s in stages if s.warnings}
    dashboard = readiness_roll.get("dashboard", {})

    write_json(config.to_dict(), run_dir / "demo_config.json")
    write_json([s.to_dict() for s in stages], run_dir / "steps.json")
    write_json([c.to_dict() for s in stages for c in s.commands],
               run_dir / "commands.json")
    write_json(_generated_artifacts(stages), run_dir / "generated_artifacts.json")
    write_json(warnings, run_dir / "warnings.json")

    readiness_out = {
        "demo_run_id": demo_run_id,
        "all_required_ok": readiness_roll.get("all_required_ok", False),
        "dry_run": config.dry_run,
        "stages": {s.name: s.status for s in stages},
        "dashboard": dashboard,
        "metrics": metrics,
    }
    write_json(readiness_out, run_dir / "readiness.json")

    report_path = run_dir / "demo_report.md"
    report_path.write_text(
        build_demo_report(config, stages, dashboard, metrics, demo_run_id),
        encoding="utf-8")
    paths["report"] = str(report_path)

    # latest pointer
    latest = {
        "demo_run_id": demo_run_id,
        "run_dir": str(run_dir),
        "all_required_ok": readiness_roll.get("all_required_ok", False),
        "dry_run": config.dry_run,
        "correlation_id": config.correlation_id,
        "engine_c_snapshot": config.engine_c_snapshot,
        "report": str(report_path),
    }
    write_json(latest, Path(out_root) / "latest.json")
    paths["run_dir"] = str(run_dir)
    paths["latest"] = str(Path(out_root) / "latest.json")
    logger.info("Demo run '%s' artefacts written to %s (offline; no device "
                "access, no command execution beyond the approved allowlist).",
                demo_run_id, run_dir)
    return paths
