"""Deterministic configuration diffing → change events (Phase 9).

Purpose
-------
Compare two :class:`~src.live_logging.hirschmann_config.ConfigSnapshot` objects
and emit structured configuration-change source records (VLAN, port mode, native
VLAN, trunk VLANs, PoE, STP, management-protocol changes, and port add/remove).
Diffing is pure and deterministic: identical inputs always yield identical,
ordered events. This module only reads snapshots and describes differences — it
never applies a change.
"""

from __future__ import annotations

from typing import Any

from src.live_logging.hirschmann_config import ConfigSnapshot

SOURCE_KEY = "hirschmann_config"
VENDOR = "hirschmann"
PRODUCT = "hios_switch"


def _record(
    snapshot: ConfigSnapshot,
    subcategory: str,
    severity: str,
    message: str,
    interface: str | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corr = {k: v for k, v in {"interface": interface, "device_id": snapshot.device_id}.items() if v}
    ref_parts = [snapshot.device_id, subcategory, interface or "-", snapshot.config_hash]
    return {
        "source_vendor": VENDOR,
        "source_product": PRODUCT,
        "source_type": "config_snapshot",
        "source_key": SOURCE_KEY,
        "source_name": snapshot.device_id,
        "timestamp": None,
        "device_id": snapshot.device_id,
        "hostname": snapshot.hostname,
        "category": "configuration",
        "subcategory": subcategory,
        "severity": severity,
        "message": message,
        "correlation_keys": corr,
        "normalized_fields": dict(fields or {}),
        "raw_ref": ":".join(ref_parts),
        "raw_payload": {"change": subcategory, "interface": interface, **(fields or {})},
    }


def diff_configs(prev: ConfigSnapshot, curr: ConfigSnapshot) -> list[dict[str, Any]]:
    """Return ordered configuration-change source records (prev → curr)."""
    events: list[dict[str, Any]] = []

    # --- global settings ---
    if prev.globals.get("stp_mode") != curr.globals.get("stp_mode"):
        events.append(_record(
            curr, "stp_change", "medium",
            f"STP mode changed {prev.globals.get('stp_mode')} -> {curr.globals.get('stp_mode')}",
            fields={"from": prev.globals.get("stp_mode"), "to": curr.globals.get("stp_mode")},
        ))
    if not prev.globals.get("telnet_enabled") and curr.globals.get("telnet_enabled"):
        events.append(_record(
            curr, "mgmt_protocol_change", "high",
            "Insecure management protocol enabled: telnet server turned on",
            fields={"protocol": "telnet", "enabled": True},
        ))
    if prev.globals.get("ssh_enabled") and not curr.globals.get("ssh_enabled"):
        events.append(_record(
            curr, "mgmt_protocol_change", "high",
            "SSH server disabled — management may fall back to insecure access",
            fields={"protocol": "ssh", "enabled": False},
        ))

    # --- VLAN set changes ---
    added = sorted(set(curr.vlans) - set(prev.vlans))
    removed = sorted(set(prev.vlans) - set(curr.vlans))
    for vlan in added:
        events.append(_record(curr, "vlan_added", "low", f"VLAN {vlan} added", fields={"vlan_id": vlan}))
    for vlan in removed:
        events.append(_record(curr, "vlan_removed", "medium", f"VLAN {vlan} removed", fields={"vlan_id": vlan}))

    # --- per-interface changes ---
    prev_ifaces = prev.interfaces
    for name in sorted(curr.interfaces):
        new = curr.interfaces[name]
        old = prev_ifaces.get(name)
        if old is None:
            events.append(_record(curr, "port_added", "low", f"Interface {name} added", interface=name))
            continue
        events.extend(_interface_diff(curr, name, old, new))

    for name in sorted(set(prev_ifaces) - set(curr.interfaces)):
        events.append(_record(curr, "port_removed", "medium", f"Interface {name} removed", interface=name))

    return events


def _interface_diff(
    snapshot: ConfigSnapshot,
    name: str,
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if old.get("mode") != new.get("mode"):
        events.append(_record(
            snapshot, "port_mode_change", "medium",
            f"Port {name} mode changed {old.get('mode')} -> {new.get('mode')}",
            interface=name, fields={"from": old.get("mode"), "to": new.get("mode")},
        ))
    if old.get("access_vlan") != new.get("access_vlan"):
        events.append(_record(
            snapshot, "access_vlan_change", "medium",
            f"Access VLAN on {name} changed {old.get('access_vlan')} -> {new.get('access_vlan')}",
            interface=name,
            fields={"from": old.get("access_vlan"), "to": new.get("access_vlan"),
                    "vlan_id": new.get("access_vlan")},
        ))
    if old.get("native_vlan") != new.get("native_vlan"):
        events.append(_record(
            snapshot, "native_vlan_change", "medium",
            f"Native VLAN on {name} changed {old.get('native_vlan')} -> {new.get('native_vlan')}",
            interface=name, fields={"from": old.get("native_vlan"), "to": new.get("native_vlan")},
        ))
    if sorted(old.get("trunk_allowed") or []) != sorted(new.get("trunk_allowed") or []):
        events.append(_record(
            snapshot, "trunk_vlan_change", "medium",
            f"Trunk allowed VLANs on {name} changed",
            interface=name,
            fields={"from": old.get("trunk_allowed"), "to": new.get("trunk_allowed")},
        ))
    if old.get("poe_enabled") != new.get("poe_enabled"):
        events.append(_record(
            snapshot, "poe_change", "medium",
            f"PoE on {name} changed {old.get('poe_enabled')} -> {new.get('poe_enabled')}",
            interface=name, fields={"from": old.get("poe_enabled"), "to": new.get("poe_enabled")},
        ))
    return events


def diff_snapshot_series(snapshots: list[ConfigSnapshot]) -> list[dict[str, Any]]:
    """Diff each consecutive pair in a chronologically-ordered snapshot list."""
    events: list[dict[str, Any]] = []
    for prev, curr in zip(snapshots, snapshots[1:]):
        events.extend(diff_configs(prev, curr))
    return events
