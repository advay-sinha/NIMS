"""Engine C Phase 8 — Batfish validation artefact persistence.

Writes the optional Batfish validation outputs under
``outputs/network_config/<snapshot_id>/batfish/``::

    batfish_summary.json
    parse_status.csv
    node_properties.csv
    interface_properties.csv
    l3_edges.csv
    undefined_references.csv
    batfish_findings.csv
    batfish_findings.json

All output is clearly marked as external validation evidence. Nothing here
contacts a device or executes a command.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any

from src.network_config.batfish_adapter import (
    SAFETY_NOTE,
    BatfishTableResult,
    BatfishValidationResult,
)

logger = logging.getLogger(__name__)

_TABLE_FILES = {
    "parse_status": "parse_status.csv",
    "node_properties": "node_properties.csv",
    "interface_properties": "interface_properties.csv",
    "l3_edges": "l3_edges.csv",
    "undefined_references": "undefined_references.csv",
}

_FINDING_FIELDS = ["finding_id", "category", "severity", "title", "device",
                   "interface", "evidence", "recommendation", "source",
                   "confidence"]


def write_batfish(result: BatfishValidationResult, out_dir: Path) -> dict[str, Path]:
    """Persist a Batfish validation result into ``<out_dir>/batfish/``."""
    from src.utils.io import write_json

    out = Path(out_dir) / "batfish"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["summary"] = write_json(_summary_payload(result),
                                  out / "batfish_summary.json")

    by_name = {t.name: t for t in result.tables}
    for name, filename in _TABLE_FILES.items():
        paths[name] = _write_table_csv(out / filename, name, by_name.get(name))

    paths["findings_json"] = write_json(
        [dataclasses.asdict(f) for f in result.findings],
        out / "batfish_findings.json")
    paths["findings_csv"] = _write_findings_csv(out / "batfish_findings.csv",
                                                result)
    logger.info("Batfish validation artefacts (status=%s) written to %s.",
                result.status, out)
    return paths


def _summary_payload(result: BatfishValidationResult) -> dict[str, Any]:
    return {
        "snapshot_id": result.snapshot_id,
        "status": result.status,
        "reason": result.reason,
        "timestamp": result.timestamp,
        "node_count": result.node_count,
        "interface_count": result.interface_count,
        "l3_edge_count": result.l3_edge_count,
        "undefined_reference_count": result.undefined_reference_count,
        "parse_status_summary": result.parse_status_summary,
        "tables": [
            {"name": t.name, "status": t.status, "row_count": t.row_count,
             "error": t.error}
            for t in result.tables
        ],
        "findings": [dataclasses.asdict(f) for f in result.findings],
        "safety_note": SAFETY_NOTE,
    }


def _write_table_csv(path: Path, name: str,
                     table: BatfishTableResult | None) -> Path:
    """Write one Batfish table; header-only status row when it has no data."""
    if table and table.rows:
        fieldnames = sorted({key for row in table.rows for key in row})
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in table.rows:
                writer.writerow({k: _cell(row.get(k)) for k in fieldnames})
        return path
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "status", "row_count", "error"])
        status = table.status if table else "absent"
        error = (table.error if table else "table not produced") or ""
        row_count = table.row_count if table else 0
        writer.writerow([name, status, row_count, error])
    return path


def _write_findings_csv(path: Path, result: BatfishValidationResult) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FINDING_FIELDS)
        writer.writeheader()
        for finding in result.findings:
            record = dataclasses.asdict(finding)
            writer.writerow({f: _cell(record.get(f)) for f in _FINDING_FIELDS})
    return path


def _cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return "" if value is None else value
