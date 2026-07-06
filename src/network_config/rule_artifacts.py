"""Engine C Phase 3 — rule-finding artefact persistence.

Writes rule-engine output alongside the inventory/topology in
``outputs/network_config/<snapshot_id>/``::

    findings.json
    findings.csv
    rule_summary.json

Detection artefacts only; no remediation is written or executed.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_FINDING_FIELDS = [
    "finding_id", "rule_id", "title", "severity", "category", "device",
    "interface", "vlan", "status", "confidence", "source", "tags",
    "evidence", "recommendation",
]


def write_findings(
    findings: Sequence[Any], summary: dict[str, Any], out_dir: Path
) -> dict[str, Path]:
    """Persist findings (JSON + CSV) and the rule summary."""
    from src.utils.io import write_json

    out = Path(out_dir)
    paths: dict[str, Path] = {}
    paths["findings_json"] = write_json(
        [dataclasses.asdict(f) for f in findings], out / "findings.json"
    )
    paths["findings_csv"] = _write_csv(out / "findings.csv", findings)
    paths["rule_summary"] = write_json(summary, out / "rule_summary.json")
    logger.info("Rule findings (%d) written to %s.", len(findings), out)
    return paths


def _write_csv(path: Path, findings: Sequence[Any]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FINDING_FIELDS)
        writer.writeheader()
        for finding in findings:
            record = dataclasses.asdict(finding)
            writer.writerow({f: _cell(record.get(f)) for f in _FINDING_FIELDS})
    return path


def _cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return "" if value is None else value
