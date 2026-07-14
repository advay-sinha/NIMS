"""Markdown report for a demo-preparation run.

Pure string building from an already-computed plan/result. The report answers
"is the frontend ready, and what will it show?" and always restates the offline,
no-command-execution safety posture.
"""

from __future__ import annotations

from typing import Any

from src.demo.models import DemoConfig, StageResult

_STATUS_ICON = {
    "success": "OK", "reused_existing": "REUSED", "skipped": "SKIP",
    "failed": "FAIL", "pending": "-", "running": "…",
}


def build_demo_report(config: DemoConfig, stages: list[StageResult],
                      dashboard: dict[str, Any], metrics: dict[str, Any],
                      demo_run_id: str) -> str:
    """Render the demo-readiness report."""
    lines: list[str] = []
    ok = all(s.ok for s in stages if s.required)
    headline = ("Demo data prepared successfully"
                if ok and not config.dry_run
                else ("Dry-run plan (nothing executed)" if config.dry_run
                      else "Demo preparation INCOMPLETE — see failures below"))

    lines += [f"# Full-Demo Readiness — `{demo_run_id}`", "",
              f"> **{headline}.** Offline preparation only — no device was "
              "contacted, no packets captured, no remediation executed. Only "
              "approved local commands ran.", ""]

    # Stage table
    lines += ["## Stages", "",
              "| Stage | Status | Reused | Elapsed (s) |", "|---|---|---|---|"]
    for s in stages:
        icon = _STATUS_ICON.get(s.status, s.status)
        lines.append(f"| {s.title} | {icon} | {'yes' if s.reused else '-'} | "
                     f"{s.elapsed_seconds:.2f} |")
    lines.append("")

    # Engine readiness
    lines += _engine_a_section(stages)
    lines += _engine_b_section(stages)
    lines += _engine_c_section(config, stages)
    lines += _syslog_section(stages)
    lines += _correlation_streaming_section(metrics)
    lines += _frontend_section(dashboard)
    lines += _warnings_section(stages)

    # Launch instructions
    lines += ["## Launch the frontend", "",
              "```bash", "python -m scripts.run_dashboard", "```",
              f"Assessment: `{config.engine_c_snapshot}` · "
              f"Correlation: `{config.correlation_id}` · "
              "Streaming: `outputs/streaming/current/current_state.json`", ""]

    lines += ["## Safety", "",
              "- Offline preparation only; no device contact, no SSH/SNMP/syslog "
              "listener, no packet capture.",
              "- Only allowlisted local commands were executed (argument arrays, "
              "never a shell string).",
              "- Engine C dry-run actions remain `executed=false` / "
              "`dry_run_only=true`; no remediation was applied.",
              "- No raw dataset or source artefact was mutated.", ""]
    return "\n".join(lines)


def _find(stages: list[StageResult], name: str) -> StageResult | None:
    return next((s for s in stages if s.name == name), None)


def _engine_a_section(stages: list[StageResult]) -> list[str]:
    stage = _find(stages, "engine_a")
    lines = ["## Engine A — Cyber models", ""]
    if not stage:
        return lines + ["_n/a_", ""]
    ready = stage.details.get("readiness", {})
    trained = stage.details.get("train_datasets", [])
    lines += ["| Dataset | Ready | Model | Action |", "|---|---|---|---|"]
    for dataset, info in (ready.get("datasets", {}) or {}).items():
        action = "trained" if dataset in trained else "reused"
        lines.append(f"| {dataset} | {'yes' if info.get('ready') else 'no'} | "
                     f"{info.get('model_type') or '-'} | {action} |")
    lines.append("")
    return lines


def _engine_b_section(stages: list[StageResult]) -> list[str]:
    stage = _find(stages, "engine_b")
    lines = ["## Engine B — Network health", ""]
    if not stage:
        return lines + ["_n/a_", ""]
    ready = stage.details.get("readiness", {})
    lines += [f"- Dataset: `{ready.get('dataset')}` | experiment: "
              f"`{ready.get('experiment') or 'none'}` | "
              f"{'reused' if stage.reused else 'trained'} | report: "
              f"{ready.get('report_available')}", ""]
    return lines


def _engine_c_section(config: DemoConfig, stages: list[StageResult]) -> list[str]:
    stage = _find(stages, "engine_c_assessment")
    lines = ["## Engine C — Assessment", ""]
    if not stage:
        return lines + ["_n/a_", ""]
    ready = stage.details.get("readiness", {})
    lines += [f"- Snapshot: `{config.engine_c_snapshot}` "
              f"({'reused' if stage.reused else 'refreshed'})",
              f"- Missing artefacts: {ready.get('missing') or 'none'}",
              "- Dry-run actions: `executed=false`, `dry_run_only=true`.", ""]
    return lines


def _syslog_section(stages: list[StageResult]) -> list[str]:
    stage = _find(stages, "syslog")
    lines = ["## Syslog", ""]
    if not stage:
        return lines + ["_n/a_", ""]
    ready = stage.details.get("readiness", {})
    run_id = stage.details.get("run_id") or ready.get("run_id")
    lines += [f"- Status: {stage.status} | run: `{run_id or 'none'}` "
              "(optional unless --require-syslog).", ""]
    return lines


def _correlation_streaming_section(metrics: dict[str, Any]) -> list[str]:
    corr = metrics.get("correlation", {})
    stream = metrics.get("streaming", {})
    lines = ["## Correlation & streaming", ""]
    if corr:
        lines.append(
            f"- Correlation: {corr.get('total_signals')} signal(s) -> "
            f"{corr.get('total_incidents')} incident(s); "
            f"{corr.get('incidents_with_syslog_evidence')} with syslog evidence "
            f"(source: `{corr.get('syslog_source') or 'none'}`).")
    if stream:
        lines.append(
            f"- Streaming: {stream.get('total_events')} event(s); "
            f"{stream.get('active_incident_count')} active incident(s); "
            f"clock: {stream.get('clock_reliability_status')}.")
    if not (corr or stream):
        lines.append("_No correlation/streaming metrics available._")
    lines.append("")
    return lines


def _frontend_section(dashboard: dict[str, Any]) -> list[str]:
    lines = ["## Frontend readiness", ""]
    if not dashboard:
        return lines + ["_Frontend readiness not evaluated._", ""]
    lines += ["| Section | Available |", "|---|---|"]
    for name, ok in (dashboard.get("sections", {}) or {}).items():
        lines.append(f"| {name} | {'yes' if ok else 'no'} |")
    lines += ["",
              f"- Overall ready: **{dashboard.get('ready')}** | "
              f"safety banner: {dashboard.get('safety_banner_ok')} | "
              f"incidents: {dashboard.get('incident_count')}"]
    if dashboard.get("clock_integrity_warning"):
        lines.append("- ⚠ Clock integrity warning: some timestamps unreliable; "
                     "time-based correlation is approximate.")
    if dashboard.get("missing_required_sections"):
        lines.append(f"- Missing required sections: "
                     f"{dashboard['missing_required_sections']}")
    lines.append("")
    return lines


def _warnings_section(stages: list[StageResult]) -> list[str]:
    warns = [(s.title, w) for s in stages for w in s.warnings]
    if not warns:
        return []
    lines = ["## Warnings", ""]
    lines += [f"- **{title}**: {w}" for title, w in warns]
    lines.append("")
    return lines
