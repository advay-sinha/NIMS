"""Sophos Firewall syslog ingestion (offline-first) (Phase 9).

Purpose
-------
Parse saved Sophos Firewall syslog lines (``key="value"`` device logs with a
standard syslog header) into normalizer source records. Offline files are the
default; a live UDP/TCP receiver is a later, explicitly-approved phase and is
not implemented here (no socket is ever opened by this module).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SOURCE_KEY = "sophos_syslog"
VENDOR = "sophos"
PRODUCT = "sophos_firewall"

# Leading syslog header: optional <pri>, ISO or BSD timestamp, host token.
_HEADER = re.compile(
    r"^(?:<\d+>)?\s*"
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*|"
    r"[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})?\s*"
    r"(?P<host>\S+)?\s*"
)
# key="value" or key=value pairs.
_KV = re.compile(r'(\w[\w.\-]*)=(?:"([^"]*)"|(\S+))')

# Sophos log_subtype/log_type mapped to our categories.
_SUBTYPE_CATEGORY: dict[str, str] = {
    "denied": "firewall",
    "allowed": "firewall",
    "firewall": "firewall",
    "ips": "ips",
    "idp": "ips",
    "antivirus": "antivirus",
    "virus": "antivirus",
    "web filter": "web_filter",
    "webfilter": "web_filter",
    "sandstorm": "sandbox",
    "sandbox": "sandbox",
    "atp": "ips",
    "authentication": "authentication",
}


def _category(fields: dict[str, str]) -> str:
    for key in ("log_subtype", "log_type", "log_component"):
        value = fields.get(key, "").strip().lower()
        if value in _SUBTYPE_CATEGORY:
            return _SUBTYPE_CATEGORY[value]
    return "firewall"


def parse_syslog_line(line: str) -> dict[str, Any] | None:
    """Parse one Sophos syslog line into a source record, or None if unparsable.

    Raises
    ------
    ValueError
        If the line has a header but no recognisable key/value payload — the
        caller records this as a ``syslog_parse_error`` sample reference.
    """
    stripped = line.strip()
    if not stripped:
        return None
    header = _HEADER.match(stripped)
    payload = stripped[header.end():] if header else stripped
    fields = {
        m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3))
        for m in _KV.finditer(payload)
    }
    if not fields:
        raise ValueError(f"No key=value pairs in syslog line: {stripped[:80]!r}")

    ts = (header.group("ts") if header else None) or fields.get("date") or fields.get("timestamp")
    src_ip = fields.get("src_ip") or fields.get("source_ip") or fields.get("srcip")
    dst_ip = fields.get("dst_ip") or fields.get("dest_ip") or fields.get("dstip")
    message = fields.get("message") or fields.get("log_message") or _synth_message(fields)
    correlation_keys = {
        k: v
        for k, v in {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": fields.get("protocol") or fields.get("proto"),
        }.items()
        if v
    }
    return {
        "source_vendor": VENDOR,
        "source_product": PRODUCT,
        "source_type": "syslog",
        "source_key": SOURCE_KEY,
        "source_name": fields.get("device_name") or fields.get("device") or "sophos_firewall",
        "timestamp": ts,
        "category": _category(fields),
        "subcategory": fields.get("log_subtype"),
        "severity": fields.get("severity") or fields.get("priority") or fields.get("log_priority"),
        "message": message,
        "device_ip": fields.get("device_ip"),
        "hostname": fields.get("device_name") or fields.get("device"),
        "raw_ref": fields.get("log_id") or fields.get("fw_rule_id"),
        "correlation_keys": correlation_keys,
        "normalized_fields": {
            "rule_id": fields.get("fw_rule_id"),
            "action": fields.get("log_subtype") or fields.get("action"),
        },
        "raw_payload": dict(fields),
    }


def _synth_message(fields: dict[str, str]) -> str:
    action = fields.get("log_subtype") or fields.get("log_type") or "event"
    src = fields.get("src_ip") or "?"
    dst = fields.get("dst_ip") or "?"
    return f"Firewall {action}: {src} -> {dst}"


def parse_syslog_lines(lines: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse many lines; returns (records, parse_error_samples)."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in lines:
        try:
            record = parse_syslog_line(line)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if record:
            records.append(record)
    return records, errors


def read_offline(sample_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read and parse a saved Sophos syslog file (records, parse_error_samples)."""
    path = Path(sample_path)
    if not path.is_file():
        logger.warning("Sophos syslog sample not found: %s", path)
        return [], []
    with open(path, "r", encoding="utf-8") as fh:
        return parse_syslog_lines(fh.readlines())
