"""Engine C Phase 5 — dry-run audit logging.

Purpose
-------
Persist one JSONL audit record per processed remediation action. Each record
documents what a dry-run validated, blocked or skipped — never what it ran.
Every record carries ``executed = False``; nothing in this module contacts a
device or opens a network connection.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.network_config.executor import DryRunExecutionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditLogRecord:
    """One append-only audit entry for a dry-run action outcome."""

    timestamp: str
    snapshot_id: str
    execution_id: str
    action_id: str
    finding_id: Optional[str]
    device: Optional[str]
    interface: Optional[str]
    action_type: str
    status: str
    reason: Optional[str]
    dry_run_only: bool
    operator: str
    source: str
    executed: bool = False


def build_audit_records(
    result: DryRunExecutionResult,
) -> tuple[AuditLogRecord, ...]:
    """Derive the audit records for every action in a dry-run result."""
    return tuple(
        AuditLogRecord(
            timestamp=record.timestamp,
            snapshot_id=result.snapshot_id,
            execution_id=record.execution_id,
            action_id=record.action_id,
            finding_id=record.finding_id,
            device=record.device,
            interface=record.interface,
            action_type=record.action_type,
            status=record.status,
            reason=record.reason,
            dry_run_only=record.dry_run_only,
            operator=result.operator,
            source=result.source,
            executed=record.executed,
        )
        for record in result.records
    )


def write_audit_log(
    records: tuple[AuditLogRecord, ...], path: str | Path
) -> Path:
    """Write the audit records as JSON Lines (one object per line)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dataclasses.asdict(record)) + "\n")
    logger.info("Wrote %d dry-run audit record(s) to %s.", len(records), target)
    return target
