"""Hirschmann SNMP health-metric ingestion (offline-first) (Phase 9).

Purpose
-------
Turn saved Hirschmann SNMP metric readings (a JSON dump of per-device/interface
counters and gauges) into network-health source records. One reading can raise
several events (port down, high error rate, temperature, power supply, device
unreachable) according to configuration thresholds. Offline dumps are the
default; live SNMP polling is a later, approved phase (no SNMP socket is opened
here).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SOURCE_KEY = "hirschmann_snmp"
VENDOR = "hirschmann"
PRODUCT = "hios_switch"

# Default thresholds (overridable from configs/hirschmann_logging.yaml).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "in_errors_high": 100,
    "in_discards_high": 100,
    "utilization_high": 90,
    "temperature_high_c": 70,
    "temperature_critical_c": 80,
}


def _base_record(reading: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_vendor": VENDOR,
        "source_product": PRODUCT,
        "source_type": "snmp_poll",
        "source_key": SOURCE_KEY,
        "source_name": str(reading.get("device_id") or "hirschmann_switch"),
        "timestamp": reading.get("timestamp"),
        "device_id": reading.get("device_id"),
        "device_ip": reading.get("device_ip"),
        "hostname": reading.get("hostname") or reading.get("device_id"),
        "raw_payload": dict(reading),
    }


def _emit(base: dict[str, Any], **over: Any) -> dict[str, Any]:
    record = dict(base)
    record.update(over)
    return record


def heartbeat_event(reading: Mapping[str, Any]) -> dict[str, Any] | None:
    """Positive availability record for a reading that was polled successfully.

    Emitted by the live poller (not the offline parser) so a healthy device that
    trips no threshold still produces one visible ``poll_ok`` health event
    confirming the read reached the store. Returns ``None`` for unreachable
    readings (those already raise a ``device_unreachable`` event).
    """
    if str(reading.get("reachable", "true")).lower() in {"false", "0", "no"}:
        return None
    base = _base_record(reading)
    corr = {k: v for k, v in {"device_id": reading.get("device_id")}.items() if v}
    fields = {k: reading.get(k) for k in ("sysName", "sysUpTime") if reading.get(k) is not None}
    return _emit(base, category="availability", subcategory="poll_ok", severity="info",
                 correlation_keys=corr,
                 message=f"SNMP poll OK for {reading.get('hostname') or reading.get('device_id')}",
                 normalized_fields=fields,
                 raw_ref=f"{reading.get('device_id')}:poll_ok:{reading.get('timestamp')}")


def metric_events(
    reading: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Derive zero or more health source records from one SNMP reading."""
    th = {**DEFAULT_THRESHOLDS, **(dict(thresholds) if thresholds else {})}
    base = _base_record(reading)
    interface = reading.get("interface")
    corr = {k: v for k, v in {"interface": interface, "device_id": reading.get("device_id")}.items() if v}
    out: list[dict[str, Any]] = []

    if str(reading.get("reachable", "true")).lower() in {"false", "0", "no"}:
        out.append(
            _emit(base, category="availability", subcategory="device_unreachable",
                  severity="high", correlation_keys=corr,
                  message=f"Device {reading.get('device_id')} is unreachable",
                  raw_ref=f"{reading.get('device_id')}:unreachable:{reading.get('timestamp')}")
        )

    oper = str(reading.get("if_oper_status", "")).lower()
    if interface and oper in {"down", "2", "lowerlayerdown"}:
        out.append(
            _emit(base, category="interface", subcategory="port_down", severity="medium",
                  correlation_keys=corr,
                  message=f"Interface {interface} operational status is down",
                  normalized_fields={"if_oper_status": oper},
                  raw_ref=f"{reading.get('device_id')}:{interface}:port_down:{reading.get('timestamp')}")
        )

    in_errors = _num(reading.get("in_errors"))
    if interface and in_errors is not None and in_errors > th["in_errors_high"]:
        out.append(
            _emit(base, category="interface", subcategory="high_error_rate", severity="high",
                  correlation_keys=corr,
                  message=f"High input error count on {interface}: {int(in_errors)}",
                  normalized_fields={"in_errors": in_errors},
                  raw_ref=f"{reading.get('device_id')}:{interface}:in_errors:{reading.get('timestamp')}")
        )

    util = _num(reading.get("utilization") if reading.get("utilization") is not None else reading.get("bandwidth_utilization"))
    if interface and util is not None and util > th["utilization_high"]:
        out.append(
            _emit(base, category="interface", subcategory="high_utilization", severity="medium",
                  correlation_keys=corr,
                  message=f"High utilization on {interface}: {util}%",
                  normalized_fields={"utilization": util},
                  raw_ref=f"{reading.get('device_id')}:{interface}:utilization:{reading.get('timestamp')}")
        )

    temp = _num(reading.get("temperature_c"))
    if temp is not None and temp >= th["temperature_high_c"]:
        severity = "high" if temp >= th["temperature_critical_c"] else "medium"
        out.append(
            _emit(base, category="environment", subcategory="temperature", severity=severity,
                  correlation_keys={k: v for k, v in {"device_id": reading.get("device_id")}.items() if v},
                  message=f"Temperature {temp}C on {reading.get('device_id')}",
                  normalized_fields={"temperature_c": temp},
                  raw_ref=f"{reading.get('device_id')}:temperature:{reading.get('timestamp')}")
        )

    for psu in ("power_supply_1", "power_supply_2"):
        state = reading.get(psu)
        if state is not None and str(state).lower() not in {"ok", "up", "present", "1", "true"}:
            out.append(
                _emit(base, category="power", subcategory="power_supply", severity="high",
                      correlation_keys={k: v for k, v in {"device_id": reading.get("device_id")}.items() if v},
                      message=f"{psu.replace('_', ' ').title()} state '{state}' on {reading.get('device_id')}",
                      normalized_fields={psu: state},
                      raw_ref=f"{reading.get('device_id')}:{psu}:{reading.get('timestamp')}")
            )

    return out


def parse_snmp_metrics(
    readings: list[Mapping[str, Any]],
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Parse a list of SNMP readings into health source records."""
    records: list[dict[str, Any]] = []
    for reading in readings:
        records.extend(metric_events(reading, thresholds))
    return records


def read_offline(
    sample_path: str | Path,
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Read a saved SNMP metrics JSON dump into health source records."""
    path = Path(sample_path)
    if not path.is_file():
        logger.warning("Hirschmann SNMP sample not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("readings") or data.get("metrics") or []
    return parse_snmp_metrics(list(data), thresholds)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
