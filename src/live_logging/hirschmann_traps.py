"""Hirschmann SNMP trap ingestion (offline-first) (Phase 9).

Purpose
-------
Parse saved SNMP trap lines into network-health source records. A trap line is
``<timestamp> <device_ip> <trapName> key=value ...``. Offline trap samples are
the default; a live trap receiver is a later, approved phase (no UDP socket is
bound here).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SOURCE_KEY = "hirschmann_traps"
VENDOR = "hirschmann"
PRODUCT = "hios_switch"

_LINE = re.compile(
    r"^\s*(?P<ts>\S+)\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\s+(?P<trap>\w+)\s*(?P<rest>.*)$"
)
_KV = re.compile(r"(\w[\w.\-]*)=(?:\"([^\"]*)\"|(\S+))")

# trap name -> (category, subcategory, severity)
_TRAP_MAP: dict[str, tuple[str, str, str]] = {
    "linkdown": ("interface", "port_down", "medium"),
    "linkup": ("interface", "port_up", "info"),
    "coldstart": ("system", "restart", "medium"),
    "warmstart": ("system", "restart", "low"),
    "authenticationfailure": ("security", "auth_failure", "medium"),
    "hmtemperature": ("environment", "temperature", "high"),
    "hmpowersupply": ("power", "power_supply", "high"),
    "hmsignalcontact": ("environment", "signal_contact", "high"),
    "hmpoefault": ("power", "poe_fault", "high"),
    "topologychange": ("topology", "stp_change", "medium"),
    "risingalarm": ("environment", "rmon_alarm", "high"),
}


def parse_trap_line(line: str) -> dict[str, Any] | None:
    """Parse one trap line into a source record, or None if blank.

    Raises
    ------
    ValueError
        If a non-blank line does not match the trap grammar (recorded as a
        ``trap_parse_error`` sample by the caller).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = _LINE.match(stripped)
    if not match:
        raise ValueError(f"Unparsable trap line: {stripped[:80]!r}")
    trap = match.group("trap")
    fields = {
        m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3))
        for m in _KV.finditer(match.group("rest"))
    }
    category, subcategory, severity = _TRAP_MAP.get(
        trap.lower(), ("system", trap.lower(), "medium")
    )
    severity = fields.get("severity") or fields.get("status") or severity
    interface = fields.get("ifname") or fields.get("interface") or fields.get("ifindex")
    corr = {k: v for k, v in {"interface": interface, "device_id": fields.get("sysname")}.items() if v}
    message = f"Trap {trap} from {match.group('ip')}" + (f" on {interface}" if interface else "")
    return {
        "source_vendor": VENDOR,
        "source_product": PRODUCT,
        "source_type": "snmp_trap",
        "source_key": SOURCE_KEY,
        "source_name": fields.get("sysname") or match.group("ip"),
        "timestamp": match.group("ts"),
        "device_ip": match.group("ip"),
        "hostname": fields.get("sysname"),
        "device_id": fields.get("sysname"),
        "category": category,
        "subcategory": subcategory,
        "severity": severity,
        "message": message,
        "correlation_keys": corr,
        "normalized_fields": {"trap": trap, **fields},
        "raw_ref": f"{match.group('ip')}:{trap}:{match.group('ts')}",
        "raw_payload": {"trap": trap, "ip": match.group("ip"), **fields},
    }


def parse_traps(lines: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse many trap lines; returns (records, parse_error_samples)."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in lines:
        try:
            record = parse_trap_line(line)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if record:
            records.append(record)
    return records, errors


def read_offline(sample_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read and parse a saved trap log (records, parse_error_samples)."""
    path = Path(sample_path)
    if not path.is_file():
        logger.warning("Hirschmann trap sample not found: %s", path)
        return [], []
    with open(path, "r", encoding="utf-8") as fh:
        return parse_traps(fh.readlines())
