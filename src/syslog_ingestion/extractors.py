"""Mnemonic-specific extractors: enrich a parsed event with entities/hints.

Purpose
-------
Given a grammar-parsed :class:`SyslogEvent` (status ``generic``), attach the
structured entities (interface, VLAN, MAC, IP, flap count, ERPS ring/state,
username, community, ...), semantic tags and per-engine routing hints that the
event's ``FACILITY-MNEMONIC`` code implies.

Design
------
Each handler parses only its own message shape and returns
``(entities, tags, hints, fully_parsed)``. Unknown codes fall through to
``generic`` untouched — an unfamiliar message must never crash ingestion. The
status becomes ``parsed`` when a handler extracted everything it expected, or
``partially_parsed`` when the code matched but the message text did not.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from src.syslog_ingestion.models import (
    GENERIC,
    PARSED,
    PARTIALLY_PARSED,
    EngineHints,
    SyslogEntities,
    SyslogEvent,
)

# --- message regexes --------------------------------------------------------
_IFACE_RE = re.compile(r"interface\s+([A-Za-z]+\d[\d/]*)", re.IGNORECASE)
_STATE_RE = re.compile(r"changed state to (up|down)", re.IGNORECASE)
_MAC_RE = re.compile(r"([0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4})")
_VLAN_RE = re.compile(r"vlan\s+(\d+)", re.IGNORECASE)
_MOVED_TO_RE = re.compile(r"to port(?:\s+interface)?\s+([A-Za-z]+\d[\d/]*)",
                          re.IGNORECASE)
_TIMES_RE = re.compile(r"for\s+(\d+)\s+times", re.IGNORECASE)
_IP_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")
_COMMUNITY_RE = re.compile(r"Community\s+(\S+)\s+authentication failed", re.IGNORECASE)
_USER_RE = re.compile(r"User\s+(\S+)\s+authentication failed", re.IGNORECASE)
_ERPS_RING_RE = re.compile(r"ring\s+(\d+)", re.IGNORECASE)
_ERPS_TO_STATE_RE = re.compile(r"changed to (\w+)(?:\s+from\s+\w+)?", re.IGNORECASE)
_ERPS_PORT_STATE_RE = re.compile(r"changed to (\w+)\s+on ring", re.IGNORECASE)
_ERPS_PORT_IFACE_RE = re.compile(r"(?:ERPS )?port\s+([A-Za-z]+\d[\d/]*)",
                                 re.IGNORECASE)
_POWER_UNIT_RE = re.compile(r"Power\s+(\d+)", re.IGNORECASE)
_FAN_RE = re.compile(r"Fan\s+(\d+)", re.IGNORECASE)
_PROCESS_RE = re.compile(r"Process\s+(\S+)\s+(?:is|has)", re.IGNORECASE)
_TELNET_PROTO_RE = re.compile(r"(Telnet|SSH)\(", re.IGNORECASE)

# A handler returns (entity fields, tags, engine-hint fields, fully_parsed).
Handler = Callable[[str], tuple[dict, tuple[str, ...], dict, bool]]


def _iface(msg: str) -> str | None:
    m = _IFACE_RE.search(msg)
    return m.group(1) if m else None


# --- handlers ---------------------------------------------------------------
def _lineproto(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    iface = _iface(msg)
    state = _STATE_RE.search(msg)
    ent = {"interface_id": iface}
    link_tag = f"link_{state.group(1)}" if state else "link_state"
    tags = ("port_flap", "link_state", link_tag)
    hints = {"engine_b_health": True}
    return ent, tuple(dict.fromkeys(tags)), hints, iface is not None


def _mac_flap(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    mac = _MAC_RE.search(msg)
    vlan = _VLAN_RE.search(msg)
    dest = _MOVED_TO_RE.search(msg)
    times = _TIMES_RE.search(msg)
    ent = {
        "mac_address": mac.group(1) if mac else None,
        "vlan_id": vlan.group(1) if vlan else None,
        "interface_id": dest.group(1) if dest else None,
        "flap_count": int(times.group(1)) if times else None,
    }
    hints = {"engine_b_health": True, "engine_c_topology": True,
             "correlation_candidate": True}
    fully = mac is not None and vlan is not None
    return ent, ("mac_flap", "loop_risk"), hints, fully


def _arp_change(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    ip = _IP_RE.search(msg)
    ent = {"ip_address": ip.group(1) if ip else None}
    hints = {"engine_a_intrusion": True, "engine_c_security": True,
             "correlation_candidate": True}
    return (ent, ("arp_instability", "duplicate_ip_possible", "arp_spoof_possible"),
            hints, ip is not None)


def _snmp_community(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    community = _COMMUNITY_RE.search(msg)
    ent = {"community": community.group(1) if community else None}
    hints = {"engine_a_intrusion": True, "correlation_candidate": True}
    return (ent, ("snmp_auth_failed", "recon_possible", "brute_force_possible"),
            hints, community is not None)


def _snmp_user(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    user = _USER_RE.search(msg)
    ent = {"username": user.group(1) if user else None}
    hints = {"engine_a_intrusion": True, "correlation_candidate": True}
    return (ent, ("snmp_user_auth_failed", "management_auth_failed"),
            hints, user is not None)


def _poe_abnormal(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    iface = _iface(msg)
    short = "short" in msg.lower()
    tags = ("poe_fault",) + (("short_detected",) if short else ())
    hints = {"engine_c_poe": True, "correlation_candidate": True}
    return {"interface_id": iface}, tags, hints, iface is not None


def _poe_churn(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    iface = _iface(msg)
    hints = {"engine_c_poe": True, "engine_b_health": True}
    return {"interface_id": iface}, ("poe_power_churn",), hints, iface is not None


def _erps_ring(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    ring = _ERPS_RING_RE.search(msg)
    state = _ERPS_TO_STATE_RE.search(msg)
    ent = {"erps_ring": ring.group(1) if ring else None,
           "erps_state": state.group(1).lower() if state else None}
    hints = {"engine_c_topology": True, "engine_b_health": True,
             "correlation_candidate": True}
    return (ent, ("erps_churn", "ring_protection", "topology_instability"),
            hints, ring is not None)


def _erps_port(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    ring = _ERPS_RING_RE.search(msg)
    iface_m = _ERPS_PORT_IFACE_RE.search(msg)
    iface = iface_m.group(1) if iface_m else _iface(msg)
    state = _ERPS_PORT_STATE_RE.search(msg)
    ent = {"erps_ring": ring.group(1) if ring else None,
           "interface_id": iface,
           "erps_state": state.group(1).lower() if state else None}
    hints = {"engine_c_topology": True, "engine_b_health": True,
             "correlation_candidate": True}
    return (ent, ("erps_churn", "ring_protection", "topology_instability"),
            hints, ring is not None)


def _power(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    unit = _POWER_UNIT_RE.search(msg)
    abnormal = not re.search(r"normal|inserted|online", msg, re.IGNORECASE)
    tags = ("device_health",) + (("power_fault",) if abnormal else ())
    hints = {"engine_b_health": True, "correlation_candidate": abnormal}
    ent = {"power_unit": unit.group(1) if unit else None}
    return ent, tags, hints, unit is not None


def _fan(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    fan = _FAN_RE.search(msg)
    abnormal = not re.search(r"normal|inserted|online", msg, re.IGNORECASE)
    tags = ("device_health",) + (("fan_fault",) if abnormal else ())
    hints = {"engine_b_health": True, "correlation_candidate": abnormal}
    return {"fan_id": fan.group(1) if fan else None}, tags, hints, fan is not None


def _ha_unit(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    hints = {"engine_b_health": True}
    return {}, ("device_health", "reboot"), hints, True


def _clock(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    hints = {"engine_b_health": True}
    return {}, ("device_health", "clock_change"), hints, True


def _process(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    proc = _PROCESS_RE.search(msg)
    hints = {"engine_b_health": True}
    return ({"process_name": proc.group(1) if proc else None},
            ("device_health", "reboot"), hints, proc is not None)


def _telnet(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    ip = _IP_RE.search(msg)
    proto = _TELNET_PROTO_RE.search(msg)
    ent = {"ip_address": ip.group(1) if ip else None,
           "login_protocol": proto.group(1).lower() if proto else "telnet"}
    hints = {"engine_c_security": True}
    return ent, ("management_access", "insecure_telnet"), hints, True


def _web(msg: str) -> tuple[dict, tuple[str, ...], dict, bool]:
    ip = _IP_RE.search(msg)
    ent = {"ip_address": ip.group(1) if ip else None, "login_protocol": "http"}
    hints = {"engine_c_security": True}
    return ent, ("management_access",), hints, True


# --- dispatch ---------------------------------------------------------------
# Exact ``FACILITY-MNEMONIC`` code -> handler.
_EXACT: dict[str, Handler] = {
    "PORTMGR-LINEPROTO_UP": _lineproto,
    "PORTMGR-LINEPROTO_DOWN": _lineproto,
    "LINK-LINEPROTO_UP": _lineproto,
    "LINK-LINEPROTO_DOWN": _lineproto,
    "LINK-INTERFACE_UP": _lineproto,
    "LINK-INTERFACE_DOWN": _lineproto,
    "FDB-MAC_ADDR_FLAPPING_VLAN": _mac_flap,
    "ARP-MAC_CHANGE_TOO_FAST": _arp_change,
    "SNMP-COMMUNITY_AUTHOR_FAILED": _snmp_community,
    "SNMP-USER_AUTH_FAILED": _snmp_user,
    "POE-POWER_ABNORMAL": _poe_abnormal,
    "POE-POWER_ON": _poe_churn,
    "POE-POWER_OFF": _poe_churn,
    "ERPS-RING_STAT_CHG": _erps_ring,
    "ERPS-RING_PORT_BLK": _erps_port,
    "ERPS-RING_PORT_FWD": _erps_port,
    "HA-UNIT_STATE_NORMAL": _ha_unit,
    "CLKM-DEV_CLOCK_CHANGE": _clock,
}

# Facility-level fallbacks keyed by (facility, mnemonic-substring).
_BY_FACILITY_SUBSTR: tuple[tuple[str, str, Handler], ...] = (
    ("SYSMGMT", "PWR", _power),
    ("SYSMGMT", "POWER", _power),
    ("SYSMGMT", "FAN", _fan),
    ("HA", "UNIT_STATE", _ha_unit),
    ("CLKM", "CLOCK", _clock),
    ("DCM", "PROCESS", _process),
    ("TELNET", "LOGIN", _telnet),
    ("TELNET", "LOGOUT", _telnet),
    ("WEB", "LOGIN", _web),
)


def _resolve_handler(facility: str | None, mnemonic: str | None) -> Handler | None:
    """Find the handler for a code, exact match first then facility substrings."""
    if facility is None:
        return None
    code = f"{facility}-{mnemonic}" if mnemonic else facility
    if code in _EXACT:
        return _EXACT[code]
    mnem = mnemonic or ""
    for fac, substr, handler in _BY_FACILITY_SUBSTR:
        if facility == fac and substr in mnem:
            return handler
    return None


def enrich_event(event: SyslogEvent) -> SyslogEvent:
    """Return a copy of ``event`` enriched by its mnemonic-specific handler.

    Unknown codes are returned unchanged with ``parse_status = generic``.
    """
    handler = _resolve_handler(event.facility, event.mnemonic)
    if handler is None:
        return replace(event, parse_status=GENERIC)

    ent_fields, tags, hint_fields, fully = handler(event.message)
    entities = SyslogEntities(**{k: v for k, v in ent_fields.items() if v is not None})
    hints = EngineHints(**hint_fields)
    status = PARSED if fully else PARTIALLY_PARSED
    return replace(
        event,
        entities=entities,
        tags=tuple(dict.fromkeys(tags)),
        engine_hints=hints,
        parse_status=status,
    )
