"""Engine C Phase 6 — snapshot-diff artefact persistence.

Writes the offline comparison outputs under
``outputs/network_config/diffs/<before>__to__<after>/``::

    snapshot_diff.json
    snapshot_diff.csv
    verification_results.json
    verification_results.csv
    diff_summary.json
    network_diff_report.md

Comparison only — no snapshot artefact is modified and no command is executed.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.network_config.diff import DiffRecord, SnapshotDiff
from src.network_config.verification import VerificationResult

logger = logging.getLogger(__name__)

_SAFETY_NOTE = "offline comparison only, no commands executed"

_ACTIVE_TYPES = {"added", "removed", "changed"}
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

_DIFF_FIELDS = [
    "diff_id", "category", "change_type", "device", "interface", "vlan",
    "field", "before_value", "after_value", "severity", "evidence",
]
_VERIFICATION_FIELDS = [
    "verification_id", "action_id", "finding_id", "rule_id", "device",
    "interface", "expected_outcome", "observed_outcome", "status", "evidence",
    "recommendation",
]


def build_diff_summary(
    diff: SnapshotDiff, verifications: Sequence[VerificationResult]
) -> dict[str, Any]:
    """Roll a diff + verification set up into the ``diff_summary.json`` payload."""
    active = [r for r in diff.records if r.change_type in _ACTIVE_TYPES]
    by_status = Counter(v.status for v in verifications)
    return {
        "before_snapshot_id": diff.before_snapshot_id,
        "after_snapshot_id": diff.after_snapshot_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_changes": len(active),
        "changes_by_category": dict(Counter(r.category for r in active)),
        "changes_by_type": dict(Counter(r.change_type for r in active)),
        "findings_new": diff.findings_new,
        "findings_resolved": diff.findings_resolved,
        "findings_persistent": diff.findings_persistent,
        "findings_changed_severity": diff.findings_changed_severity,
        "verification_total": len(verifications),
        "verification_passed": by_status.get("passed", 0),
        "verification_failed": by_status.get("failed", 0),
        "verification_unknown": by_status.get("unknown", 0),
        "verification_not_applicable": by_status.get("not_applicable", 0),
        "warnings": list(diff.warnings),
        "safety_note": _SAFETY_NOTE,
    }


def write_diff(
    diff: SnapshotDiff,
    verifications: Sequence[VerificationResult],
    out_dir: Path,
) -> dict[str, Path]:
    """Persist the diff, verification results, summary and Markdown report."""
    from src.utils.io import write_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = build_diff_summary(diff, verifications)
    paths: dict[str, Path] = {}

    paths["diff_json"] = write_json(_diff_payload(diff, summary),
                                    out / "snapshot_diff.json")
    paths["diff_csv"] = _write_csv(out / "snapshot_diff.csv", diff.records,
                                   _DIFF_FIELDS)
    paths["verification_json"] = write_json(
        [dataclasses.asdict(v) for v in verifications],
        out / "verification_results.json")
    paths["verification_csv"] = _write_csv(
        out / "verification_results.csv", verifications, _VERIFICATION_FIELDS)
    paths["summary"] = write_json(summary, out / "diff_summary.json")

    report_path = out / "network_diff_report.md"
    report_path.write_text(_report(diff, verifications, summary),
                           encoding="utf-8")
    paths["report"] = report_path

    logger.info(
        "Snapshot diff '%s -> %s' written to %s (%d change(s), no commands "
        "executed).", diff.before_snapshot_id, diff.after_snapshot_id, out,
        summary["total_changes"])
    return paths


def _diff_payload(diff: SnapshotDiff, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "before_snapshot_id": diff.before_snapshot_id,
        "after_snapshot_id": diff.after_snapshot_id,
        "generated_at": diff.generated_at,
        "safety_note": _SAFETY_NOTE,
        "summary": summary,
        "warnings": list(diff.warnings),
        "records": [dataclasses.asdict(r) for r in diff.records],
    }


def _write_csv(path: Path, rows: Sequence[Any], fields: Sequence[str]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            record = dataclasses.asdict(row)
            writer.writerow({f: ("" if record.get(f) is None else record[f])
                             for f in fields})
    return path


# ------------------------------------------------------------------- report


def _report(diff, verifications, summary) -> str:
    lines = [
        f"# Network Diff Report — {diff.before_snapshot_id} → "
        f"{diff.after_snapshot_id}",
        "",
        f"> **No commands were executed.** {_SAFETY_NOTE.capitalize()}.",
        "",
        f"- Before snapshot: `{diff.before_snapshot_id}`",
        f"- After snapshot: `{diff.after_snapshot_id}`",
        f"- Generated: {diff.generated_at}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Total changes | {summary['total_changes']} |",
        f"| Findings new | {summary['findings_new']} |",
        f"| Findings resolved | {summary['findings_resolved']} |",
        f"| Findings persistent | {summary['findings_persistent']} |",
        f"| Verification passed | {summary['verification_passed']} |",
        f"| Verification failed | {summary['verification_failed']} |",
        f"| Verification unknown | {summary['verification_unknown']} |",
        "",
    ]
    lines += _table(summary["changes_by_category"], "Category",
                    "## Changes by category")
    lines += _table(summary["changes_by_type"], "Change", "## Changes by type")

    top = _top_changes(diff.records)
    if top:
        lines += ["", "## Top high/critical changes", ""]
        lines += [
            f"- **{r.severity}/{r.category}** [{r.change_type}] "
            f"{r.device or 'n/a'}"
            + (f" {r.interface}" if r.interface else "")
            + (f": {r.evidence}" if r.evidence else "")
            for r in top
        ]

    lines += _findings_lines(diff.records)
    lines += _verification_lines(verifications)

    if diff.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in diff.warnings]
    lines.append("")
    return "\n".join(lines)


def _table(counts: dict[str, int], label: str, header: str) -> list[str]:
    if not counts:
        return []
    rows = ["", header, "", f"| {label} | Count |", "|---|---|"]
    rows += [f"| {k} | {v} |"
             for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return rows


def _top_changes(records: Sequence[DiffRecord], limit: int = 10) -> list[DiffRecord]:
    ranked = sorted(
        (r for r in records
         if r.change_type in _ACTIVE_TYPES
         and r.severity in {"critical", "high"}),
        key=lambda r: _SEVERITY_ORDER.get(r.severity, 9))
    return ranked[:limit]


def _findings_lines(records: Sequence[DiffRecord]) -> list[str]:
    buckets = {"added": [], "removed": [], "changed": []}
    for r in records:
        if r.category == "finding" and r.change_type in buckets:
            buckets[r.change_type].append(r)
    if not any(buckets.values()):
        return []
    labels = {"added": "New", "removed": "Resolved", "changed": "Severity-changed"}
    lines = ["", "## Findings", ""]
    for change_type, label in labels.items():
        for r in buckets[change_type]:
            lines.append(
                f"- **{label}** [{r.field}] {r.device or 'n/a'}"
                + (f" {r.interface}" if r.interface else "")
                + (f" ({r.before_value} → {r.after_value})"
                   if change_type == "changed" else ""))
    return lines


def _verification_lines(verifications: Sequence[VerificationResult]) -> list[str]:
    if not verifications:
        return []
    lines = ["", "## Remediation verification", "",
             "> Evidence-based and conservative; `unknown` when the after "
             "snapshot lacks data.", ""]
    for v in verifications:
        lines.append(
            f"- **{v.status.upper()}** [{v.rule_id}] {v.device or 'n/a'}"
            + (f" {v.interface}" if v.interface else "")
            + f" — expected: {v.expected_outcome}; observed: {v.observed_outcome}")
    return lines
