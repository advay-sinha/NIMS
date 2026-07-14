"""Weak (threshold-derived) labels for Engine B syslog feature windows.

These are *weak* labels: there is no ground-truth attack/degradation annotation
in the source logs. Each label is a transparent function of a configurable
threshold applied to an aggregated window, so the whole pipeline stays
reproducible and auditable. Consumers must treat them as heuristics, never as
verified incidents (see the report's Limitations section).
"""

from __future__ import annotations

from typing import Any, Mapping

# Default thresholds mirror configs/syslog_ingestion.yaml so the module is
# usable without a config (e.g. in unit tests); config always wins at runtime.
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "port_flaps_per_hour_warning": 3,
    "port_flaps_per_hour_high": 6,
    "mac_moves_warning": 20,
    "mac_moves_high": 100,
    "snmp_auth_fail_warning_per_5min": 5,
    "snmp_auth_fail_high_per_5min": 20,
    "erps_events_warning_per_hour": 3,
    "poe_fault_high": True,
    "device_instability_events": 1,
}


def _level(value: float, warning: float, high: float) -> str:
    """Return ``"high"``/``"warning"``/``"none"`` for a value against thresholds."""
    if value >= high:
        return "high"
    if value >= warning:
        return "warning"
    return "none"


def compute_labels(
    features: Mapping[str, Any], thresholds: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Derive weak labels for one aggregated feature window.

    Parameters
    ----------
    features:
        A feature row produced by :mod:`src.syslog_ingestion.features`.
    thresholds:
        The ``thresholds`` config block; missing keys fall back to defaults.

    Returns
    -------
    dict
        Boolean labels plus their severity level and the per-hour / per-5min
        normalised rates used to decide them.
    """
    t = {**DEFAULT_THRESHOLDS, **(dict(thresholds) if thresholds else {})}
    window_minutes = float(features.get("window_minutes", 5) or 5)

    flaps_per_hour = float(features.get("port_flap_count", 0)) * (60.0 / window_minutes)
    snmp_per_5min = (float(features.get("snmp_auth_fail_count", 0))
                     * (5.0 / window_minutes))
    erps_per_hour = float(features.get("erps_event_count", 0)) * (60.0 / window_minutes)
    mac_moves = float(features.get("mac_move_total", 0) or 0)
    instability = (
        int(features.get("reboot_or_clock_event_count", 0))
        + int(features.get("power_fault_count", 0))
        + int(features.get("fan_fault_count", 0))
    )
    poe_faults = int(features.get("poe_fault_count", 0))

    degradation_level = _level(
        flaps_per_hour, t["port_flaps_per_hour_warning"], t["port_flaps_per_hour_high"]
    )
    loop_level = _level(mac_moves, t["mac_moves_warning"], t["mac_moves_high"])
    snmp_level = _level(
        snmp_per_5min,
        t["snmp_auth_fail_warning_per_5min"],
        t["snmp_auth_fail_high_per_5min"],
    )

    return {
        "degradation_label": degradation_level != "none",
        "degradation_level": degradation_level,
        "port_flaps_per_hour": round(flaps_per_hour, 3),
        "loop_risk_label": loop_level != "none",
        "loop_risk_level": loop_level,
        "snmp_attack_weak_label": snmp_level != "none",
        "snmp_attack_level": snmp_level,
        "snmp_auth_fail_per_5min": round(snmp_per_5min, 3),
        "device_instability_label": instability >= int(t["device_instability_events"]),
        "poe_fault_label": bool(poe_faults) and bool(t.get("poe_fault_high", True)),
        "erps_churn_label": erps_per_hour >= float(t["erps_events_warning_per_hour"]),
    }
