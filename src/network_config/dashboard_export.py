"""Engine C Phase 9 — dashboard-ready data export.

Purpose
-------
Transform the already-persisted Engine C artefacts (inventory, topology,
findings, remediation, dry-run audit, Batfish, and an optional snapshot diff)
into clean, stable, frontend-friendly JSON *views* for a future monitoring UI.

This phase builds **no UI**. It reads artefacts only — it never recomputes
inventory/topology/findings/remediation, never runs Batfish, never executes an
action, never contacts a device and never mutates an existing artefact. Every
view carries an explicit offline/no-execution safety note and the JSON shapes
are intentionally flat and stable for frontend consumption.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.network_config.findings import SEVERITY_ORDER
from src.network_config.intelligence import (
    ConfigIntelligence,
    DiffArtifacts,
    SnapshotArtifacts,
    build_intelligence,
)

logger = logging.getLogger(__name__)

EXPORT_VERSION = "1.0"
SAFETY_NOTE = ("Offline analysis only; no commands were executed. Remediation "
               "plans are dry-run and require explicit human confirmation.")

# Snapshot artefacts we can consume, for used/missing bookkeeping.
_SNAPSHOT_ARTIFACTS = [
    "inventory.json", "topology.json", "findings.json", "rule_summary.json",
    "remediation_plan.json", "remediation_summary.json",
    "dry_run_execution.json", "execution_summary.json",
    "config_intelligence_summary.json", "metadata.json",
    "batfish/batfish_summary.json",
]
_DIFF_ARTIFACTS = ["snapshot_diff.json", "verification_results.json",
                   "diff_summary.json"]


# --------------------------------------------------------------- utilities


def _devices(inventory: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(str((d.get("device") or {}).get("device_id", "unknown")), d)
            for d in inventory.get("devices") or []]


def _open_findings(artifacts: SnapshotArtifacts) -> list[dict[str, Any]]:
    return [f for f in artifacts.findings
            if str(f.get("status", "open")) == "open"]


def _group_count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return out


def _risk_by_device(intel: ConfigIntelligence) -> dict[str, dict[str, int]]:
    """device_id -> {max risk_score, finding_count} from scored findings."""
    out: dict[str, dict[str, int]] = {}
    for finding, risk in intel.ranked_findings:
        device = str(finding.get("device"))
        entry = out.setdefault(device, {"risk_score": 0, "finding_count": 0})
        entry["risk_score"] = max(entry["risk_score"], risk.risk_score)
        entry["finding_count"] += 1
    return out


def _read_metadata(artifacts: SnapshotArtifacts) -> dict[str, Any]:
    path = Path(artifacts.directory) / "metadata.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------- the export


def build_dashboard(
    artifacts: SnapshotArtifacts, diff: Optional[DiffArtifacts] = None
) -> dict[str, dict[str, Any]]:
    """Build every dashboard view as ``{output_key: view_dict}``."""
    intel = build_intelligence(artifacts, diff)
    metadata = _read_metadata(artifacts)
    risk_by_device = _risk_by_device(intel)
    now = datetime.now(timezone.utc).isoformat()

    views = {
        "dashboard_summary": _summary_view(artifacts, intel, diff, risk_by_device,
                                           now),
        "inventory_view": _inventory_view(artifacts),
        "topology_view": _topology_view(artifacts, risk_by_device),
        "findings_view": _findings_view(artifacts, intel),
        "remediation_view": _remediation_view(artifacts),
        "action_audit_view": _action_audit_view(artifacts),
        "risk_timeline": _risk_timeline(artifacts, diff, metadata, now),
        "device_health_cards": _device_health_cards(artifacts, risk_by_device),
        "export_metadata": _export_metadata(artifacts, diff, now),
    }
    if diff is not None:
        views["diff_view"] = _diff_view(diff)
        views["verification_view"] = _verification_view(diff)
    return views


def _summary_view(artifacts, intel, diff, risk_by_device, now) -> dict[str, Any]:
    s = intel.summary
    top_devices = sorted(
        ({"device": d, "risk_score": v["risk_score"],
          "finding_count": v["finding_count"]}
         for d, v in risk_by_device.items() if d and d != "None"),
        key=lambda e: (-e["risk_score"], -e["finding_count"]))[:5]
    return {
        "snapshot_id": artifacts.snapshot_id,
        "generated_at": now,
        "device_count": len(artifacts.inventory.get("devices") or []),
        "interface_count": s.total_interfaces,
        "vlan_count": s.total_vlans,
        "topology_edge_count": s.total_topology_edges,
        "finding_count": s.total_findings,
        "findings_by_severity": s.findings_by_severity,
        "findings_by_category": s.findings_by_category,
        "remediation_action_count": s.total_remediation_actions,
        "command_action_count": s.command_actions,
        "investigation_action_count": s.investigation_actions,
        "blocked_action_count": s.blocked_actions,
        "dry_run_available": s.dry_run_available,
        "batfish_available": artifacts.batfish is not None,
        "diff_available": diff is not None,
        "top_risk_devices": top_devices,
        "safety_note": SAFETY_NOTE,
    }


def _inventory_view(artifacts: SnapshotArtifacts) -> dict[str, Any]:
    devices, by = [], {"interfaces": {}, "vlans": {}, "trunks": {}, "poe": {},
                       "stp": {}}
    for device_id, device in _devices(artifacts.inventory):
        info = device.get("device") or {}
        devices.append({"device_id": device_id,
                        "hostname": info.get("hostname"),
                        "platform": info.get("platform"),
                        "management_ip": info.get("management_ip")})
        by["interfaces"][device_id] = [
            {"name": i.get("name"), "status": i.get("status"),
             "vlan": i.get("vlan"), "mode": i.get("mode"),
             "description": i.get("description"), "speed": i.get("speed"),
             "duplex": i.get("duplex"), "poe_enabled": i.get("poe_enabled"),
             "poe_state": i.get("poe_state")}
            for i in device.get("interfaces") or []]
        by["vlans"][device_id] = [
            {"vlan_id": v.get("vlan_id"), "name": v.get("name"),
             "status": v.get("status"), "ports": v.get("ports")}
            for v in device.get("vlans") or []]
        by["trunks"][device_id] = [
            {"interface": t.get("interface"),
             "allowed_vlans": t.get("allowed_vlans"),
             "native_vlan": t.get("native_vlan"),
             "trunking_status": t.get("trunking_status")}
            for t in device.get("trunks") or []]
        by["poe"][device_id] = [
            {"interface": p.get("interface"), "admin_state": p.get("admin_state"),
             "oper_state": p.get("oper_state"), "power_watts": p.get("power_watts"),
             "powered_device": p.get("powered_device")}
            for p in device.get("poe") or []]
        by["stp"][device_id] = [
            {"interface": s.get("interface"), "vlan": s.get("vlan"),
             "role": s.get("role"), "state": s.get("state")}
            for s in device.get("stp_states") or []]
    return {
        "devices": devices,
        "interfaces_by_device": by["interfaces"],
        "vlans_by_device": by["vlans"],
        "trunks_by_device": by["trunks"],
        "poe_by_device": by["poe"],
        "stp_by_device": by["stp"],
        "safety_note": SAFETY_NOTE,
    }


def _topology_view(artifacts, risk_by_device) -> dict[str, Any]:
    topo = artifacts.topology
    if not topo:
        return {"available": False, "reason": "no topology artefact",
                "nodes": [], "edges": [], "warnings": [],
                "safety_note": SAFETY_NOTE}
    warnings = topo.get("warnings") or []
    warn_by_iface: dict[tuple[str, str], int] = {}
    for w in warnings:
        key = (str(w.get("device")), str(w.get("interface")))
        warn_by_iface[key] = warn_by_iface.get(key, 0) + 1
    nodes = [
        {"id": n.get("node_id"), "label": n.get("hostname") or n.get("node_id"),
         "type": n.get("device_type"),
         "risk_score": risk_by_device.get(str(n.get("node_id")), {}).get(
             "risk_score", 0),
         "finding_count": risk_by_device.get(str(n.get("node_id")), {}).get(
             "finding_count", 0)}
        for n in topo.get("nodes") or []]
    edges = [
        {"source": e.get("local_device"), "target": e.get("remote_device"),
         "source_interface": e.get("local_interface"),
         "target_interface": e.get("remote_interface"),
         "protocol": e.get("discovery_protocol"),
         "confidence": e.get("confidence"),
         "warning_count": warn_by_iface.get(
             (str(e.get("local_device")), str(e.get("local_interface"))), 0)}
        for e in topo.get("edges") or []]
    return {"nodes": nodes, "edges": edges, "warnings": warnings,
            "safety_note": SAFETY_NOTE}


def _findings_view(artifacts, intel) -> dict[str, Any]:
    risk_by_id = {str(f.get("finding_id")): r.risk_score
                  for f, r in intel.ranked_findings}
    findings = []
    for finding in _open_findings(artifacts):
        enriched = dict(finding)
        enriched["risk_score"] = risk_by_id.get(str(finding.get("finding_id")), 0)
        findings.append(enriched)
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.get("severity"), 9),
                                 -int(f.get("risk_score", 0))))
    grouped_device: dict[str, list] = {}
    for finding in findings:
        grouped_device.setdefault(str(finding.get("device")), []).append(finding)
    return {
        "findings": findings,
        "grouped_by_severity": _group_count(findings, "severity"),
        "grouped_by_category": _group_count(findings, "category"),
        "grouped_by_device": grouped_device,
        "top_findings": findings[:10],
        "safety_note": SAFETY_NOTE,
    }


def _remediation_view(artifacts: SnapshotArtifacts) -> dict[str, Any]:
    plan = artifacts.remediation_plan
    if not plan:
        return {"available": False, "reason": "no remediation plan artefact",
                "actions": [], "human_confirmation_required": True,
                "dry_run_only": True, "safety_note": SAFETY_NOTE}
    actions = list(plan.get("actions") or [])
    grouped_device: dict[str, list] = {}
    grouped_risk: dict[str, list] = {}
    for action in actions:
        grouped_device.setdefault(str(action.get("device")), []).append(action)
        grouped_risk.setdefault(str(action.get("risk_level")), []).append(action)
    return {
        "actions": actions,
        "command_actions": [a for a in actions if a.get("commands")],
        "investigation_actions": [a for a in actions
                                  if a.get("action_type") == "investigation"],
        "blocked_actions": [a for a in actions
                            if str(a.get("status")) == "blocked"],
        "grouped_by_device": grouped_device,
        "grouped_by_risk": grouped_risk,
        "human_confirmation_required": True,
        "dry_run_only": True,
        "safety_note": SAFETY_NOTE,
    }


def _action_audit_view(artifacts: SnapshotArtifacts) -> dict[str, Any]:
    execution = artifacts.dry_run_execution
    summary = artifacts.execution_summary
    if not execution and not summary:
        return {"available": False,
                "reason": "no dry-run execution artefact",
                "safety_note": SAFETY_NOTE}
    records = list((execution or {}).get("records") or [])
    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(str(record.get("status")), []).append(record)
    return {
        "available": True,
        "execution_summary": summary or {},
        "records_by_status": grouped,
        "audit_records": records,
        "executed_count": sum(1 for r in records if r.get("executed")),
        "safety_note": SAFETY_NOTE,
    }


def _risk_timeline(artifacts, diff, metadata, now) -> dict[str, Any]:
    events: list[dict[str, Any]] = []

    def add(step, label, timestamp, source):
        if timestamp:
            events.append({"step": step, "label": label,
                           "timestamp": timestamp, "source": source})

    snapshot_ts = metadata.get("timestamp")
    add("snapshot_generated", "Snapshot inventory generated", snapshot_ts,
        "metadata.json")
    if artifacts.topology is not None:
        add("topology_built", "Topology built", snapshot_ts, "topology.json")
    if artifacts.findings:
        add("findings_generated", "Findings generated", snapshot_ts,
            "findings.json")
    if artifacts.remediation_summary:
        add("remediation_planned", "Remediation plan generated",
            artifacts.remediation_summary.get("timestamp"),
            "remediation_summary.json")
    if artifacts.execution_summary:
        add("dry_run_executed", "Dry-run execution validated",
            artifacts.execution_summary.get("timestamp"),
            "execution_summary.json")
    if diff and diff.diff_summary:
        add("diff_verified", "Snapshot diff / verification",
            diff.diff_summary.get("timestamp"), "diff_summary.json")

    events.sort(key=lambda e: e["timestamp"])
    return {
        "generated_at": now,
        "kind": "artifact_lifecycle",
        "note": "Artifact lifecycle timeline, not a live monitoring time series.",
        "events": events,
        "safety_note": SAFETY_NOTE,
    }


def _device_health_cards(artifacts, risk_by_device) -> dict[str, Any]:
    findings_by_device: dict[str, list] = {}
    for finding in _open_findings(artifacts):
        findings_by_device.setdefault(str(finding.get("device")), []).append(
            finding)
    actions_by_device: dict[str, int] = {}
    for action in ((artifacts.remediation_plan or {}).get("actions") or []):
        key = str(action.get("device"))
        actions_by_device[key] = actions_by_device.get(key, 0) + 1

    cards = []
    for device_id, device in _devices(artifacts.inventory):
        interfaces = device.get("interfaces") or []
        device_findings = findings_by_device.get(device_id, [])
        cards.append(_health_card(device_id, device, interfaces, device_findings,
                                  actions_by_device.get(device_id, 0)))
    return {"cards": cards, "safety_note": SAFETY_NOTE}


def _health_card(device_id, device, interfaces, findings, action_count
                 ) -> dict[str, Any]:
    info = device.get("device") or {}
    severities = [str(f.get("severity")) for f in findings]
    highest = min(severities, key=lambda s: SEVERITY_ORDER.get(s, 9)) \
        if severities else None
    stp_blocked = sum(1 for s in device.get("stp_states") or []
                      if str(s.get("state")).lower() == "blocking")
    status = _health_status(interfaces, severities)
    return {
        "device_id": device_id,
        "hostname": info.get("hostname"),
        "interface_count": len(interfaces),
        "trunk_count": len(device.get("trunks") or []),
        "access_port_count": sum(1 for i in interfaces
                                 if str(i.get("mode")) == "access"),
        "finding_count": len(findings),
        "highest_severity": highest,
        "remediation_action_count": action_count,
        "topology_neighbor_count": len(device.get("neighbors") or []),
        "poe_port_count": len(device.get("poe") or []),
        "stp_blocked_count": stp_blocked,
        "status": status,
    }


def _health_status(interfaces, severities) -> str:
    if not interfaces:
        return "unknown"
    if any(s in ("critical", "high") for s in severities):
        return "critical"
    if any(s in ("medium", "low") for s in severities):
        return "warning"
    if severities:
        return "warning"
    return "healthy"


def _diff_view(diff: DiffArtifacts) -> dict[str, Any]:
    summary = diff.diff_summary or {}
    records = (diff.snapshot_diff or {}).get("records") or []
    return {
        "before_snapshot_id": summary.get("before_snapshot_id"),
        "after_snapshot_id": summary.get("after_snapshot_id"),
        "total_changes": summary.get("total_changes", 0),
        "changes_by_category": summary.get("changes_by_category", {}),
        "changes_by_type": summary.get("changes_by_type", {}),
        "findings_new": summary.get("findings_new", 0),
        "findings_resolved": summary.get("findings_resolved", 0),
        "findings_persistent": summary.get("findings_persistent", 0),
        "records": records,
        "safety_note": SAFETY_NOTE,
    }


def _verification_view(diff: DiffArtifacts) -> dict[str, Any]:
    results = diff.verification_results
    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(str(result.get("status")), []).append(result)
    return {
        "results": results,
        "grouped_by_status": grouped,
        "passed": len(grouped.get("passed", [])),
        "failed": len(grouped.get("failed", [])),
        "unknown": len(grouped.get("unknown", [])),
        "not_applicable": len(grouped.get("not_applicable", [])),
        "safety_note": SAFETY_NOTE,
    }


def _export_metadata(artifacts, diff, now) -> dict[str, Any]:
    used, missing = [], []
    root = Path(artifacts.directory)
    for name in _SNAPSHOT_ARTIFACTS:
        (used if (root / name).is_file() else missing).append(name)
    if diff is not None:
        for name in _DIFF_ARTIFACTS:
            key = f"diffs/{diff.diff_id}/{name}"
            present = ((name == "snapshot_diff.json" and diff.snapshot_diff)
                       or (name == "verification_results.json"
                           and diff.verification_results is not None)
                       or (name == "diff_summary.json" and diff.diff_summary))
            (used if present else missing).append(key)
    return {
        "snapshot_id": artifacts.snapshot_id,
        "diff_id": diff.diff_id if diff else None,
        "generated_at": now,
        "source_artifacts_used": used,
        "source_artifacts_missing": missing,
        "export_version": EXPORT_VERSION,
        "safety_note": SAFETY_NOTE,
    }
