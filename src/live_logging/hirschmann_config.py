"""Hirschmann configuration snapshot loading (offline-first) (Phase 9).

Purpose
-------
Load saved Hirschmann running/startup configuration snapshots and parse them
into a structured, comparable form for :mod:`config_diff`. Offline snapshot
files are the default; live SSH/SCP retrieval is a later, approved phase (no
device connection is made here).

Parsing is intentionally line-oriented and vendor-tolerant: it extracts the
hostname, VLAN ids, a small set of global settings and per-interface
VLAN/mode/PoE state. Secret-bearing lines (e.g. SNMP community) are never
retained in structured output; redaction additionally masks them if raw text is
ever surfaced.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigSnapshot:
    """A parsed configuration snapshot for one device at one point in time."""

    device_id: str
    label: str
    config_hash: str
    hostname: str | None
    vlans: list[int] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)
    interfaces: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary form (no raw secret-bearing text)."""
        return {
            "device_id": self.device_id,
            "label": self.label,
            "config_hash": self.config_hash,
            "hostname": self.hostname,
            "vlans": self.vlans,
            "globals": self.globals,
            "interfaces": self.interfaces,
        }


_INT_RE = re.compile(r"^\s*interface\s+(?P<name>\S+)", re.IGNORECASE)
_VLAN_RE = re.compile(r"^\s*vlan\s+(?P<id>\d+)", re.IGNORECASE)


def parse_config_text(text: str, device_id: str, label: str) -> ConfigSnapshot:
    """Parse raw configuration text into a :class:`ConfigSnapshot`."""
    hostname: str | None = None
    vlans: set[int] = set()
    globals_: dict[str, Any] = {"stp_mode": None, "telnet_enabled": False, "ssh_enabled": False}
    interfaces: dict[str, dict[str, Any]] = {}
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped == "!":
            if not raw_line.startswith(" "):
                current = None
            continue

        int_match = _INT_RE.match(line)
        if int_match:
            current = int_match.group("name")
            interfaces.setdefault(
                current,
                {"mode": None, "access_vlan": None, "native_vlan": None,
                 "trunk_allowed": [], "poe_enabled": None, "description": None},
            )
            continue

        vlan_match = _VLAN_RE.match(line)
        if vlan_match and current is None:
            vlans.add(int(vlan_match.group("id")))
            continue

        low = stripped.lower()
        if current is None:
            if low.startswith("hostname "):
                hostname = stripped.split(None, 1)[1]
            elif low.startswith("spanning-tree mode"):
                globals_["stp_mode"] = stripped.split()[-1]
            elif "telnet server" in low:
                globals_["telnet_enabled"] = not low.startswith("no ") and "disable" not in low
            elif "ssh server" in low:
                globals_["ssh_enabled"] = not low.startswith("no ") and "disable" not in low
            continue

        _apply_interface_line(interfaces[current], low, stripped)

    config_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return ConfigSnapshot(
        device_id=device_id,
        label=label,
        config_hash=config_hash,
        hostname=hostname,
        vlans=sorted(vlans),
        globals=globals_,
        interfaces=interfaces,
    )


def _apply_interface_line(iface: dict[str, Any], low: str, original: str) -> None:
    if low.startswith("description "):
        iface["description"] = original.split(None, 1)[1]
    elif low.startswith("switchport mode "):
        iface["mode"] = low.split()[-1]
    elif low.startswith("switchport access vlan "):
        iface["access_vlan"] = _int(low.split()[-1])
    elif low.startswith("switchport trunk native vlan "):
        iface["native_vlan"] = _int(low.split()[-1])
    elif low.startswith("switchport trunk allowed vlan "):
        iface["trunk_allowed"] = _vlan_list(low.split()[-1])
    elif low in {"poe enabled", "poe enable"} or low == "power inline auto":
        iface["poe_enabled"] = True
    elif low in {"poe disabled", "no poe", "poe disable"} or low == "power inline never":
        iface["poe_enabled"] = False


def load_snapshot(path: str | Path, device_id: str | None = None) -> ConfigSnapshot:
    """Load and parse a snapshot file. Label is the file stem; device from name."""
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    label = resolved.stem
    device = device_id or _device_from_name(label)
    return parse_config_text(text, device_id=device, label=label)


def load_snapshots_dir(snapshot_dir: str | Path) -> dict[str, list[ConfigSnapshot]]:
    """Load every ``*.cfg`` snapshot under a directory, grouped by device id.

    Each device's list is sorted by label (which encodes the date), oldest
    first, so ``config_diff`` can compare consecutive snapshots.
    """
    base = Path(snapshot_dir)
    grouped: dict[str, list[ConfigSnapshot]] = {}
    if not base.is_dir():
        logger.warning("Config snapshot dir not found: %s", base)
        return grouped
    for path in sorted(base.glob("*.cfg")):
        snap = load_snapshot(path)
        grouped.setdefault(snap.device_id, []).append(snap)
    for snaps in grouped.values():
        snaps.sort(key=lambda s: s.label)
    return grouped


def _device_from_name(stem: str) -> str:
    # e.g. "switch_01_2026-07-02" -> "switch_01"
    match = re.match(r"(?P<dev>.+?)_\d{4}-\d{2}-\d{2}$", stem)
    return match.group("dev") if match else stem


def _int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vlan_list(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            lo, _, hi = token.partition("-")
            if lo.isdigit() and hi.isdigit():
                out.extend(range(int(lo), int(hi) + 1))
        elif token.isdigit():
            out.append(int(token))
    return out
