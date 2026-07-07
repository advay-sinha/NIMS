"""Engine C Phase 5 — dry-run execution artefact persistence.

Writes the dry-run execution outputs alongside the inventory/remediation in
``outputs/network_config/<snapshot_id>/``::

    dry_run_execution.json
    dry_run_execution.csv
    action_audit_log.jsonl
    execution_summary.json

and appends a dry-run execution section to ``network_config_report.md`` when it
exists. Every artefact states plainly that no commands were executed.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any

from src.network_config.audit import build_audit_records, write_audit_log
from src.network_config.executor import (
    DryRunExecutionResult,
    ExecutionSummary,
)

logger = logging.getLogger(__name__)

_NO_EXEC = "No commands were executed. This is a dry-run validation only."

# Marker delimiting the appended section so re-runs replace (not duplicate) it.
_REPORT_MARKER = "<!-- dry-run-execution -->"

_CSV_FIELDS = [
    "execution_id", "action_id", "finding_id", "rule_id", "device",
    "interface", "action_type", "status", "risk_level",
    "requires_confirmation", "dry_run_only", "execution_mode", "executed",
    "would_execute", "requested_command_count", "reason",
]


def write_execution(
    result: DryRunExecutionResult, summary: ExecutionSummary, out_dir: Path
) -> dict[str, Path]:
    """Persist the dry-run execution result, audit log and summary."""
    from src.utils.io import write_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    audit_records = build_audit_records(result)
    paths["audit_log"] = write_audit_log(audit_records, out / "action_audit_log.jsonl")

    summary_dict = {
        **dataclasses.asdict(summary),
        "audit_log_path": str(paths["audit_log"]),
        "notice": _NO_EXEC,
    }
    paths["summary"] = write_json(summary_dict, out / "execution_summary.json")
    paths["execution_json"] = write_json(
        _result_payload(result, summary_dict), out / "dry_run_execution.json"
    )
    paths["execution_csv"] = _write_csv(out / "dry_run_execution.csv", result)
    _update_report(out, summary_dict)

    logger.info(
        "Dry-run execution (%d action(s)) written to %s — no commands executed.",
        len(result.records), out,
    )
    return paths


def _result_payload(
    result: DryRunExecutionResult, summary_dict: dict[str, Any]
) -> dict[str, Any]:
    return {
        "snapshot_id": result.snapshot_id,
        "generated_at": result.generated_at,
        "operator": result.operator,
        "source": result.source,
        "execution_mode": result.execution_mode,
        "executed": result.executed,
        "would_execute": result.would_execute,
        "do_not_execute": result.do_not_execute,
        "notice": _NO_EXEC,
        "summary": summary_dict,
        "records": [dataclasses.asdict(r) for r in result.records],
    }


def _write_csv(path: Path, result: DryRunExecutionResult) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in result.records:
            writer.writerow({
                "execution_id": record.execution_id,
                "action_id": record.action_id,
                "finding_id": record.finding_id or "",
                "rule_id": record.rule_id or "",
                "device": record.device or "",
                "interface": record.interface or "",
                "action_type": record.action_type,
                "status": record.status,
                "risk_level": record.risk_level,
                "requires_confirmation": str(record.requires_confirmation).lower(),
                "dry_run_only": str(record.dry_run_only).lower(),
                "execution_mode": record.execution_mode,
                "executed": str(record.executed).lower(),
                "would_execute": str(record.would_execute).lower(),
                "requested_command_count": len(record.requested_commands),
                "reason": record.reason or "",
            })
    return path


def _update_report(out_dir: Path, summary_dict: dict[str, Any]) -> None:
    """Append (or replace) the dry-run execution section in the report."""
    from src.network_config.reporting import execution_report_section

    report = out_dir / "network_config_report.md"
    if not report.is_file():
        return
    text = report.read_text(encoding="utf-8")
    marker = text.find(_REPORT_MARKER)
    if marker != -1:
        text = text[:marker].rstrip()
    section = "\n".join([_REPORT_MARKER, *execution_report_section(summary_dict)])
    report.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
