"""Correlation-run artefact persistence.

Writes the correlation outputs under ``outputs/correlation/<correlation_id>/``::

    signals.json
    signals.csv
    incidents.json
    incidents.csv
    correlation_summary.json
    correlation_report.md

Pure serialisation of an already-computed
:class:`~src.correlation.engine.CorrelationResult` — nothing here recomputes
state, mutates an Engine A/B/C artefact, contacts a device or executes a command.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Sequence

from src.correlation.engine import CorrelationResult
from src.correlation.models import CorrelatedIncident, Signal
from src.correlation.reporting import build_report

logger = logging.getLogger(__name__)

_SIGNAL_FIELDS = [
    "signal_id", "engine", "source_artifact", "category", "source_type",
    "severity", "confidence", "confidence_label", "title", "description",
    "raw_reference", "timestamp", "time_start", "time_end", "device",
    "interface", "src_ip", "dst_ip", "vlan", "mac", "event_count",
    "clock_unreliable", "entity_confident", "aggregate", "tags",
]
_INCIDENT_FIELDS = [
    "incident_id", "rule_id", "title", "severity", "confidence", "status",
    "engines", "multi_engine", "aggregate_only", "affected_devices",
    "affected_interfaces", "related_vlans", "related_ips", "signal_count",
    "syslog_signal_count", "time_reliability", "entity_match_confidence",
    "root_cause_hypothesis", "tags",
]


def write_correlation(result: CorrelationResult, out_dir: Path) -> dict[str, Path]:
    """Persist signals, incidents, summary and report for one correlation run."""
    from src.utils.io import write_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["signals_json"] = write_json(
        [s.to_row() for s in result.signals], out / "signals.json")
    paths["signals_csv"] = _write_csv(
        [s.to_row() for s in result.signals], _SIGNAL_FIELDS, out / "signals.csv")

    paths["incidents_json"] = write_json(
        [i.to_dict() for i in result.incidents], out / "incidents.json")
    paths["incidents_csv"] = _write_csv(
        [i.to_row() for i in result.incidents], _INCIDENT_FIELDS,
        out / "incidents.csv")

    paths["summary"] = write_json(result.summary.to_dict(),
                                  out / "correlation_summary.json")

    report_path = out / "correlation_report.md"
    report_path.write_text(build_report(result), encoding="utf-8")
    paths["report"] = report_path

    logger.info("Correlation artefacts for '%s' written to %s "
                "(offline; no commands executed).", result.correlation_id, out)
    return paths


def _write_csv(rows: Sequence[dict[str, Any]], fields: list[str],
               path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


__all__ = ["write_correlation"]
