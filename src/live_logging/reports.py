"""Ingestion status & report generation (Phase 9).

Purpose
-------
Summarise an ingestion run into ``ingestion_status.json`` and a human-readable
``ingestion_report.md`` (spec Phase 9 > Proposed Outputs). Counts are computed
from the persisted normalized events so the report reflects what is actually on
disk. Reports never contain secrets — they read already-redacted events.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.live_logging.event_store import EventStore
from src.live_logging.models import IngestionStatus

logger = logging.getLogger(__name__)

STATUS_FILENAME = "ingestion_status.json"
REPORT_FILENAME = "ingestion_report.md"


def _counts(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_engine: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_vendor: dict[str, int] = {}
    for event in events:
        _inc(by_engine, event.get("engine_target", "unknown"))
        _inc(by_severity, event.get("severity", "unknown"))
        _inc(by_vendor, event.get("source_vendor", "unknown"))
    return {"engine": by_engine, "severity": by_severity, "vendor": by_vendor}


def enrich_status(status: IngestionStatus, output_dir: str | Path) -> IngestionStatus:
    """Fill the status rollups from persisted events (idempotent)."""
    events = EventStore(output_dir).read_normalized()
    counts = _counts(events)
    status.events_by_engine = counts["engine"]
    status.events_by_severity = counts["severity"]
    status.events_by_vendor = counts["vendor"]
    status.total_events = len(events)
    return status


def write_status(status: IngestionStatus, output_dir: str | Path) -> Path:
    """Write ``ingestion_status.json`` and return its path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / STATUS_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(status.to_dict(), fh, indent=2, sort_keys=True)
    logger.info("Wrote ingestion status -> %s", path)
    return path


def write_report(status: IngestionStatus, output_dir: str | Path) -> Path:
    """Write ``ingestion_report.md`` and return its path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / REPORT_FILENAME
    path.write_text(render_markdown(status), encoding="utf-8")
    logger.info("Wrote ingestion report -> %s", path)
    return path


def render_markdown(status: IngestionStatus) -> str:
    """Render the ingestion status as a Markdown report."""
    lines: list[str] = []
    lines.append("# Live Ingestion Report")
    lines.append("")
    lines.append(f"- Mode: `{status.mode}`")
    lines.append(f"- Read-only: `{status.read_only}`")
    lines.append(f"- Window: {status.started_at} → {status.finished_at}")
    lines.append(f"- Total normalized events: **{status.total_events}**")
    lines.append(f"- Healthy: **{'yes' if status.healthy else 'no'}**")
    lines.append("")
    lines.append("_Read-only live ingestion. NIMS collects and analyzes telemetry "
                 "only. No firewall or switch configuration is changed, and no "
                 "remediation command is executed._")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("| Source | Target | Status | Mode | Events | Attempts | Error |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- |")
    for s in status.sources:
        err = s.error_category or ""
        lines.append(
            f"| {s.source} | {s.engine_target} | {s.status} | {s.mode} | "
            f"{s.events} | {s.attempts} | {err} |"
        )
    lines.append("")

    lines.append("## Events by engine target")
    lines.append("")
    lines.extend(_count_table(status.events_by_engine))
    lines.append("")
    lines.append("## Events by severity")
    lines.append("")
    lines.extend(_count_table(status.events_by_severity))
    lines.append("")
    lines.append("## Events by vendor")
    lines.append("")
    lines.extend(_count_table(status.events_by_vendor))
    lines.append("")
    return "\n".join(lines)


def _count_table(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["_none_"]
    rows = ["| Key | Count |", "| --- | ---: |"]
    for key in sorted(counts, key=lambda k: (-counts[k], k)):
        rows.append(f"| {key} | {counts[key]} |")
    return rows


def _inc(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1
