"""Normalisation helpers for syslog-derived correlation signals (Phase 13).

Pure, deterministic helpers shared by the syslog loader and the correlation
rules:

* interface-name normalisation (``Gi0/1`` / ``GigabitEthernet0/1`` / ``0/1`` ->
  a canonical key) so evidence from different sources can be matched
  conservatively;
* syslog source-type classification from a Phase-12 finding;
* evidence-quality / confidence banding, including clock-reliability and
  incomplete-identity reductions;
* cross-source entity matching that reports *how* two signals matched
  (exact / normalized / uncertain / none) so incidents never overstate a shared
  root cause.

No IO, no device access, no execution.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# --- syslog signal source types (stable strings persisted into artefacts) ---
SYSLOG_SNMP_AUTH_ACTIVITY = "SYSLOG_SNMP_AUTH_ACTIVITY"
SYSLOG_PORT_FLAP = "SYSLOG_PORT_FLAP"
SYSLOG_MAC_FLAP = "SYSLOG_MAC_FLAP"
SYSLOG_DUPLICATE_IP = "SYSLOG_DUPLICATE_IP"
SYSLOG_ERPS_CHURN = "SYSLOG_ERPS_CHURN"
SYSLOG_POE_FAULT = "SYSLOG_POE_FAULT"
SYSLOG_CLOCK_UNRELIABLE = "SYSLOG_CLOCK_UNRELIABLE"
SYSLOG_MANAGEMENT_ACCESS = "SYSLOG_MANAGEMENT_ACCESS"
SYSLOG_HA_STATE_CHANGE = "SYSLOG_HA_STATE_CHANGE"
SYSLOG_GENERIC_WARNING = "SYSLOG_GENERIC_WARNING"

# Phase-12 finding rule_id -> Phase-13 syslog source type.
_RULE_TO_SOURCE_TYPE: dict[str, str] = {
    "SYS-SNMP-AUTHFAIL": SYSLOG_SNMP_AUTH_ACTIVITY,
    "SYS-PORT-FLAP": SYSLOG_PORT_FLAP,
    "SYS-MAC-FLAP": SYSLOG_MAC_FLAP,
    "SYS-ARP-FAST": SYSLOG_DUPLICATE_IP,
    "SYS-ERPS-CHURN": SYSLOG_ERPS_CHURN,
    "SYS-POE-FAULT": SYSLOG_POE_FAULT,
    "SYS-TELNET": SYSLOG_MANAGEMENT_ACCESS,
    "SYS-DEVICE": SYSLOG_HA_STATE_CHANGE,
}

# Evidence-quality band per source type (section 3 of the spec). "Explicit"
# device messages are high; inferred/aggregated evidence is medium; fallback or
# integrity-degrading evidence is low.
_QUALITY_BY_SOURCE_TYPE: dict[str, str] = {
    SYSLOG_SNMP_AUTH_ACTIVITY: "high",   # explicit authentication failures
    SYSLOG_DUPLICATE_IP: "high",         # explicit duplicate-IP message
    SYSLOG_POE_FAULT: "high",            # explicit PoE fault
    SYSLOG_PORT_FLAP: "high",            # explicit up/down transitions
    SYSLOG_ERPS_CHURN: "high",           # explicit ring state transition
    SYSLOG_HA_STATE_CHANGE: "high",      # explicit HA/device state
    SYSLOG_MAC_FLAP: "medium",           # inferred repeated MAC movement
    SYSLOG_MANAGEMENT_ACCESS: "medium",  # management access activity
    SYSLOG_GENERIC_WARNING: "low",       # generic parse fallback
    SYSLOG_CLOCK_UNRELIABLE: "low",      # timestamps not precisely ordered
}

_DEFAULT_QUALITY_CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.5}

# Interface long-form prefixes -> canonical short prefix.
_IFACE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("tengigabitethernet", "te"), ("tengige", "te"), ("tengig", "te"),
    ("gigabitethernet", "gi"), ("gigabit", "gi"), ("gige", "gi"), ("gig", "gi"),
    ("fastethernet", "fa"), ("ethernet", "eth"), ("tunnel", "tu"),
    ("port-channel", "po"), ("portchannel", "po"),
)
_IFACE_RE = re.compile(r"^\s*([a-zA-Z][a-zA-Z\-]*?)\s*([\d/\.:]+)\s*$")


def normalize_interface(name: Optional[str]) -> Optional[str]:
    """Canonicalise an interface name for conservative cross-source matching.

    ``GigabitEthernet0/1`` / ``Gi0/1`` / ``Gig 0/1`` -> ``gi0/1``; a bare
    ``0/1`` normalises to ``0/1``. Returns ``None`` for empty input and the
    lower-cased original when the shape is unrecognised (never guesses).
    """
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    match = _IFACE_RE.match(text)
    if not match:
        return text.lower().replace(" ", "")
    prefix, number = match.group(1).lower(), match.group(2)
    for long_form, short in _IFACE_PREFIXES:
        if prefix == long_form or prefix == short or long_form.startswith(prefix):
            return f"{short}{number}"
    if prefix in {"", "-"}:
        return number
    return f"{prefix}{number}"


def source_type_for_finding(finding: Mapping[str, Any]) -> str:
    """Classify a Phase-12 finding into a syslog source type."""
    rule_id = str(finding.get("rule_id", ""))
    if rule_id in _RULE_TO_SOURCE_TYPE:
        return _RULE_TO_SOURCE_TYPE[rule_id]
    tags = {str(t).lower() for t in finding.get("tags") or ()}
    if "insecure_telnet" in tags or "management_access" in tags:
        return SYSLOG_MANAGEMENT_ACCESS
    return SYSLOG_GENERIC_WARNING


def quality_band(source_type: str) -> str:
    """Return the evidence-quality band (high/medium/low) for a source type."""
    return _QUALITY_BY_SOURCE_TYPE.get(source_type, "medium")


def resolve_confidence(
    source_type: str,
    *,
    clock_unreliable: bool = False,
    entity_confident: bool = True,
    config: Mapping[str, Any] | None = None,
) -> tuple[float, str, list[str]]:
    """Return ``(confidence, band, notes)`` applying evidence-quality rules.

    Clock-unreliable evidence and incomplete device/interface identity both
    reduce confidence and are recorded as evidence-quality notes.
    """
    conf_map = _DEFAULT_QUALITY_CONFIDENCE
    if config:
        conf_map = {**conf_map, **(config.get("confidence_by_quality") or {})}
    band = quality_band(source_type)
    confidence = float(conf_map.get(band, 0.7))
    notes: list[str] = []

    if clock_unreliable:
        mult = float((config or {}).get(
            "unreliable_clock_confidence_multiplier", 0.6))
        confidence *= mult
        band = "low"
        notes.append("timestamps unreliable — chronology approximate")
    if not entity_confident:
        confidence *= 0.85
        if band == "high":
            band = "medium"
        notes.append("device/interface identity incomplete")

    return max(0.0, min(1.0, confidence)), band, notes


def match_entities(
    a_device: Optional[str], a_interface: Optional[str],
    b_device: Optional[str], b_interface: Optional[str],
    *, require_exact_device: bool = True, normalize: bool = True,
) -> str:
    """Classify how two signals' entities match.

    Returns one of ``exact`` / ``normalized`` / ``uncertain`` / ``none``. Never
    merges on vaguely-similar names: a normalised interface match still needs a
    device match (unless ``require_exact_device`` is disabled).
    """
    dev_a = (a_device or "").strip().lower() or None
    dev_b = (b_device or "").strip().lower() or None
    if_a_raw = (a_interface or "").strip() or None
    if_b_raw = (b_interface or "").strip() or None

    device_match = dev_a is not None and dev_a == dev_b
    if require_exact_device and not device_match:
        # Different or missing device -> cannot confidently claim same entity.
        if if_a_raw and if_b_raw and if_a_raw.lower() == if_b_raw.lower():
            return "uncertain"
        return "none"

    if if_a_raw and if_b_raw:
        if if_a_raw.lower() == if_b_raw.lower():
            return "exact" if device_match else "uncertain"
        if normalize and normalize_interface(if_a_raw) == normalize_interface(if_b_raw):
            return "normalized" if device_match else "uncertain"
        return "uncertain" if device_match else "none"

    # No interfaces to compare; fall back to the device relationship.
    if device_match:
        return "exact"
    return "none"
