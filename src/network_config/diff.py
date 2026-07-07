"""Engine C Phase 6 — offline snapshot diffing.

Purpose
-------
Compare two already-persisted offline snapshots (``inventory.json`` and the
optional ``topology.json`` / ``findings.json``) and describe exactly what
changed. This is pure artefact comparison: it never contacts a device, never
opens a network connection and never executes a command.

Design
------
:func:`load_snapshot` reads a snapshot directory (inventory required, the rest
optional). :class:`SnapshotDiffer` produces a flat list of
:class:`DiffRecord` across interfaces, VLANs, trunks, PoE, STP, topology and
findings, plus findings-classification counts (new / resolved / persistent /
changed-severity). Keys are stable, business-level tuples so a changed
finding-id algorithm never masks a persistent finding.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- models


@dataclass(frozen=True)
class DiffRecord:
    """One detected difference between the before and after snapshots."""

    diff_id: str
    category: str            # interface/vlan/trunk/poe/stp/topology/finding/...
    change_type: str         # added/removed/changed/unchanged
    device: Optional[str]
    interface: Optional[str] = None
    vlan: Optional[str] = None
    field: Optional[str] = None
    before_value: Optional[str] = None
    after_value: Optional[str] = None
    severity: str = "info"   # critical/high/medium/low/info
    evidence: Optional[str] = None


@dataclass(frozen=True)
class SnapshotDiff:
    """The full set of differences between two snapshots (offline, read-only)."""

    before_snapshot_id: str
    after_snapshot_id: str
    generated_at: str
    records: tuple[DiffRecord, ...] = ()
    findings_new: int = 0
    findings_resolved: int = 0
    findings_persistent: int = 0
    findings_changed_severity: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotData:
    """Loaded artefacts for one snapshot (inventory required, rest optional)."""

    snapshot_id: str
    directory: str
    inventory: dict[str, Any]
    topology: Optional[dict[str, Any]] = None
    findings: Optional[list[dict[str, Any]]] = None
    remediation: Optional[dict[str, Any]] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ------------------------------------------------------------------- loading


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(directory: str | Path, snapshot_id: str | None = None) -> SnapshotData:
    """Load one snapshot directory. ``inventory.json`` is required.

    Missing optional artefacts (topology/findings/remediation) are recorded as
    warnings rather than errors.
    """
    root = Path(directory)
    inventory_path = root / "inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Required inventory.json not found in {root}")
    try:
        inventory = _read_json(inventory_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid inventory.json in {root}: {exc}") from exc
    if not isinstance(inventory, dict) or "devices" not in inventory:
        raise ValueError(f"inventory.json in {root} is missing 'devices'")

    warnings: list[str] = []
    topology = _load_optional(root / "topology.json", "topology", warnings)
    findings = _load_optional(root / "findings.json", "findings", warnings)
    remediation = _load_optional(root / "remediation_plan.json", "remediation",
                                 warnings)

    return SnapshotData(
        snapshot_id=snapshot_id or str(inventory.get("snapshot_id", root.name)),
        directory=str(root),
        inventory=inventory,
        topology=topology if isinstance(topology, dict) else None,
        findings=findings if isinstance(findings, list) else None,
        remediation=remediation if isinstance(remediation, dict) else None,
        warnings=tuple(warnings),
    )


def _load_optional(path: Path, label: str, warnings: list[str]) -> Any:
    if not path.is_file():
        warnings.append(f"optional {label} artefact missing: {path.name}")
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError as exc:
        warnings.append(f"could not parse {label} artefact {path.name}: {exc}")
        return None


# ---------------------------------------------------------------- index maps


def _devices(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return list(inventory.get("devices") or [])


def _device_id(device: dict[str, Any]) -> str:
    return str((device.get("device") or {}).get("device_id", "unknown"))


def _object_map(inventory: dict[str, Any], collection: str, key_field: str
                ) -> dict[tuple[str, str], dict[str, Any]]:
    """Map ``(device_id, obj[key_field]) -> obj`` for one device collection."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for device in _devices(inventory):
        did = _device_id(device)
        for obj in device.get(collection) or []:
            out[(did, str(obj.get(key_field)))] = obj
    return out


def interface_map(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return _object_map(inventory, "interfaces", "name")


def trunk_map(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return _object_map(inventory, "trunks", "interface")


def poe_map(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return _object_map(inventory, "poe", "interface")


def vlan_map(inventory: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return _object_map(inventory, "vlans", "vlan_id")


def stp_map(inventory: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for device in _devices(inventory):
        did = _device_id(device)
        for obj in device.get("stp_states") or []:
            out[(did, str(obj.get("interface")), str(obj.get("vlan")))] = obj
    return out


def finding_key(finding: dict[str, Any]) -> tuple:
    """Stable, id-independent key for a finding (rule + location + category)."""
    return (
        finding.get("rule_id"), finding.get("device"),
        finding.get("interface"), finding.get("vlan"), finding.get("category"),
    )


def open_findings(findings: Optional[list[dict[str, Any]]]) -> dict[tuple, dict]:
    """Map stable-key -> finding for the ``open`` findings only."""
    return {
        finding_key(f): f
        for f in (findings or [])
        if str(f.get("status", "open")) == "open"
    }


# ------------------------------------------------------------------- helpers


def _diff_id(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"DIFF-{digest[:8]}"


def _s(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return str(value)


# -------------------------------------------------------------- the differ


# (field, severity) pairs compared for a present-in-both interface / object.
_INTERFACE_FIELDS = (
    ("status", "medium"), ("vlan", "medium"), ("mode", "medium"),
    ("description", "low"), ("poe_enabled", "medium"), ("poe_state", "medium"),
)
_TRUNK_FIELDS = (
    ("allowed_vlans", "medium"), ("native_vlan", "high"),
    ("trunking_status", "medium"),
)
_POE_FIELDS = (("admin_state", "medium"), ("oper_state", "medium"))
_STP_FIELDS = (("state", "medium"), ("role", "low"))


class SnapshotDiffer:
    """Compute a :class:`SnapshotDiff` between two loaded snapshots."""

    def __init__(self, include_unchanged: bool = False):
        self.include_unchanged = include_unchanged

    def diff(self, before: SnapshotData, after: SnapshotData) -> SnapshotDiff:
        records: list[DiffRecord] = []
        warnings: list[str] = []

        records += self._diff_collection(
            "interface", interface_map(before.inventory),
            interface_map(after.inventory), _INTERFACE_FIELDS, key_len=2)
        records += self._diff_collection(
            "trunk", trunk_map(before.inventory), trunk_map(after.inventory),
            _TRUNK_FIELDS, key_len=2)
        records += self._diff_collection(
            "poe", poe_map(before.inventory), poe_map(after.inventory),
            _POE_FIELDS, key_len=2)
        records += self._diff_presence(
            "vlan", vlan_map(before.inventory), vlan_map(after.inventory))
        records += self._diff_collection(
            "stp", stp_map(before.inventory), stp_map(after.inventory),
            _STP_FIELDS, key_len=3)

        topo_records, topo_warn = self._diff_topology(before, after)
        records += topo_records
        warnings += topo_warn

        finding_records, counts = self._diff_findings(before, after)
        records += finding_records

        return SnapshotDiff(
            before_snapshot_id=before.snapshot_id,
            after_snapshot_id=after.snapshot_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            records=tuple(records),
            findings_new=counts["new"],
            findings_resolved=counts["resolved"],
            findings_persistent=counts["persistent"],
            findings_changed_severity=counts["changed_severity"],
            warnings=tuple(warnings),
        )

    # -- inventory collections ------------------------------------------------

    def _diff_collection(self, category, before_map, after_map, fields, key_len):
        records: list[DiffRecord] = []
        for key in sorted(set(before_map) | set(after_map), key=lambda k: tuple(map(str, k))):
            device = key[0]
            interface = key[1] if key_len >= 2 else None
            vlan = key[2] if key_len >= 3 else None
            before_obj, after_obj = before_map.get(key), after_map.get(key)
            if before_obj and not after_obj:
                records.append(self._presence(category, "removed", device,
                                              interface, vlan, "low"))
                continue
            if after_obj and not before_obj:
                records.append(self._presence(category, "added", device,
                                              interface, vlan, "info"))
                continue
            changed = False
            for name, severity in fields:
                rec = self._field_change(category, device, interface, vlan,
                                         name, before_obj.get(name),
                                         after_obj.get(name), severity)
                if rec is not None:
                    records.append(rec)
                    changed = True
            if not changed and self.include_unchanged:
                records.append(self._presence(category, "unchanged", device,
                                              interface, vlan, "info"))
        return records

    def _diff_presence(self, category, before_map, after_map):
        """Added/removed only (VLAN definitions)."""
        records: list[DiffRecord] = []
        for key in sorted(set(before_map) | set(after_map), key=lambda k: tuple(map(str, k))):
            device, ident = key
            if key in before_map and key not in after_map:
                records.append(self._presence(category, "removed", device,
                                              None, ident, "low"))
            elif key in after_map and key not in before_map:
                records.append(self._presence(category, "added", device, None,
                                              ident, "info"))
            elif self.include_unchanged:
                records.append(self._presence(category, "unchanged", device,
                                              None, ident, "info"))
        return records

    def _presence(self, category, change_type, device, interface, vlan, severity):
        return DiffRecord(
            diff_id=_diff_id(category, change_type, device, interface, vlan),
            category=category, change_type=change_type, device=device,
            interface=interface, vlan=vlan, field="presence",
            severity=severity if change_type != "unchanged" else "info",
            evidence=f"{category} {change_type}",
        )

    def _field_change(self, category, device, interface, vlan, name,
                      before_v, after_v, severity) -> Optional[DiffRecord]:
        if _s(before_v) == _s(after_v):
            return None
        return DiffRecord(
            diff_id=_diff_id(category, device, interface, vlan, name),
            category=category, change_type="changed", device=device,
            interface=interface, vlan=vlan, field=name,
            before_value=_s(before_v), after_value=_s(after_v),
            severity=severity, evidence=f"{name}: {_s(before_v)} -> {_s(after_v)}",
        )

    # -- topology -------------------------------------------------------------

    def _diff_topology(self, before, after):
        if before.topology is None or after.topology is None:
            warn = ("topology diff skipped: topology artefact missing in "
                    f"{'before' if before.topology is None else 'after'} snapshot")
            logger.warning(warn)
            return [], [warn]
        records: list[DiffRecord] = []
        records += self._diff_topo_nodes(before.topology, after.topology)
        records += self._diff_topo_edges(before.topology, after.topology)
        records += self._diff_topo_warnings(before.topology, after.topology)
        return records, []

    def _diff_topo_nodes(self, before, after):
        b = {str(n.get("node_id")): n for n in before.get("nodes") or []}
        a = {str(n.get("node_id")): n for n in after.get("nodes") or []}
        records = []
        for node_id in sorted(set(b) | set(a)):
            if node_id in b and node_id not in a:
                records.append(self._topo_rec("removed", node_id, None, "node",
                                              "low"))
            elif node_id in a and node_id not in b:
                records.append(self._topo_rec("added", node_id, None, "node",
                                              "info"))
        return records

    def _diff_topo_edges(self, before, after):
        def emap(topo):
            out = {}
            for e in topo.get("edges") or []:
                endpoints = tuple(sorted([
                    f"{e.get('local_device')}:{e.get('local_interface')}",
                    f"{e.get('remote_device')}:{e.get('remote_interface')}",
                ]))
                out[endpoints] = e
            return out

        b, a = emap(before), emap(after)
        records = []
        for key in sorted(set(b) | set(a)):
            label = " <-> ".join(key)
            if key in b and key not in a:
                records.append(self._topo_rec("removed", label, None, "edge",
                                              "medium"))
            elif key in a and key not in b:
                records.append(self._topo_rec("added", label, None, "edge",
                                              "medium"))
            else:
                for name in ("confidence", "discovery_protocol"):
                    if _s(b[key].get(name)) != _s(a[key].get(name)):
                        records.append(self._topo_rec(
                            "changed", label, name, "edge", "low",
                            before=b[key].get(name), after=a[key].get(name)))
        return records

    def _diff_topo_warnings(self, before, after):
        def wmap(topo):
            return {
                (str(w.get("category")), str(w.get("device")),
                 str(w.get("interface")), str(w.get("message"))): w
                for w in topo.get("warnings") or []
            }

        b, a = wmap(before), wmap(after)
        records = []
        for key in sorted(set(b) | set(a)):
            source = b.get(key) or a.get(key)
            device, interface = source.get("device"), source.get("interface")
            if key in b and key not in a:      # warning resolved
                records.append(DiffRecord(
                    diff_id=_diff_id("topology", "warning", "removed", key),
                    category="topology", change_type="removed", device=device,
                    interface=interface, field="warning",
                    before_value=source.get("message"),
                    severity="low", evidence="topology warning resolved"))
            elif key in a and key not in b:    # new warning
                records.append(DiffRecord(
                    diff_id=_diff_id("topology", "warning", "added", key),
                    category="topology", change_type="added", device=device,
                    interface=interface, field="warning",
                    after_value=source.get("message"),
                    severity=str(source.get("severity", "medium")),
                    evidence="new topology warning"))
        return records

    def _topo_rec(self, change_type, ident, name, kind, severity,
                  before=None, after=None):
        return DiffRecord(
            diff_id=_diff_id("topology", kind, change_type, ident, name),
            category="topology", change_type=change_type, device=ident,
            field=name or kind, before_value=_s(before), after_value=_s(after),
            severity=severity, evidence=f"topology {kind} {change_type}")

    # -- findings -------------------------------------------------------------

    def _diff_findings(self, before, after):
        counts = {"new": 0, "resolved": 0, "persistent": 0,
                  "changed_severity": 0}
        records: list[DiffRecord] = []
        if before.findings is None and after.findings is None:
            return records, counts
        b = open_findings(before.findings)
        a = open_findings(after.findings)
        for key in sorted(set(b) | set(a), key=lambda k: tuple(map(str, k))):
            bf, af = b.get(key), a.get(key)
            if bf and not af:
                counts["resolved"] += 1
                records.append(self._finding_rec("removed", bf, "low",
                                                 "finding resolved"))
            elif af and not bf:
                counts["new"] += 1
                records.append(self._finding_rec("added", af,
                                                 af.get("severity", "medium"),
                                                 "new finding"))
            else:
                counts["persistent"] += 1
                if _s(bf.get("severity")) != _s(af.get("severity")):
                    counts["changed_severity"] += 1
                    records.append(DiffRecord(
                        diff_id=_diff_id("finding", "severity", key),
                        category="finding", change_type="changed",
                        device=af.get("device"), interface=af.get("interface"),
                        vlan=af.get("vlan"), field="severity",
                        before_value=_s(bf.get("severity")),
                        after_value=_s(af.get("severity")),
                        severity=str(af.get("severity", "medium")),
                        evidence="finding severity changed"))
                elif self.include_unchanged:
                    records.append(self._finding_rec("unchanged", af, "info",
                                                     "finding persistent"))
        return records, counts

    def _finding_rec(self, change_type, finding, severity, evidence):
        return DiffRecord(
            diff_id=_diff_id("finding", change_type, finding.get("rule_id"),
                             finding.get("device"), finding.get("interface"),
                             finding.get("vlan")),
            category="finding", change_type=change_type,
            device=finding.get("device"), interface=finding.get("interface"),
            vlan=finding.get("vlan"), field=finding.get("rule_id"),
            before_value=finding.get("title") if change_type == "removed" else None,
            after_value=finding.get("title") if change_type == "added" else None,
            severity=str(severity), evidence=evidence)
