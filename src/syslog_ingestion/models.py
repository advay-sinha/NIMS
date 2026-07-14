"""Typed models for offline industrial-switch syslog ingestion (Phase 12).

Purpose
-------
One frozen, JSON-serialisable :class:`SyslogEvent` is the single structured
contract that every downstream consumer (Engine B feature windows, Engine C
findings, streaming replay, reporting) reads. A syslog line flows through
preprocessing -> parsing -> mnemonic extraction and emerges as exactly one of
these objects; no dict-of-dicts ever leaks between stages.

The models here are *pure value objects* — no IO, no device access, no
execution. They describe saved log data only.

Inputs / Outputs
----------------
Constructed by :mod:`src.syslog_ingestion.parser`; serialised via
:meth:`SyslogEvent.to_dict` / :meth:`to_row`.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# --- severity ---------------------------------------------------------------
# The trailing digit of a ``%FACILITY-MNEMONIC-<n>`` code is the syslog level.
SEVERITY_LABELS: dict[int, str] = {
    0: "emergency",
    1: "alert",
    2: "critical",
    3: "error",
    4: "warning",
    5: "notice",
    6: "info",
    7: "debug",
}
UNKNOWN_SEVERITY = "unknown"

# Ordering (most severe first) for deterministic sorting / reporting.
SEVERITY_ORDER: dict[str, int] = {
    "emergency": 0, "alert": 1, "critical": 2, "error": 3,
    "warning": 4, "notice": 5, "info": 6, "debug": 7, UNKNOWN_SEVERITY: 8,
}

# --- parse status -----------------------------------------------------------
PARSED = "parsed"                    # grammar + a mnemonic-specific extractor ran
PARTIALLY_PARSED = "partially_parsed"  # grammar ok, extractor found only some fields
GENERIC = "generic"                 # grammar ok, no specific extractor for the code
FAILED = "failed"                   # grammar did not match (kept for audit)

# --- ansi color hints (captured before stripping) ---------------------------
ANSI_COLOR_HINTS: dict[str, str] = {
    "33": "yellow",   # warning hint
    "36": "cyan",     # ERPS / topology hint
    "37": "white",    # info hint
}


def severity_label(code: Optional[int]) -> str:
    """Map a numeric syslog level to its human label (``unknown`` if unmapped)."""
    if code is None:
        return UNKNOWN_SEVERITY
    return SEVERITY_LABELS.get(code, UNKNOWN_SEVERITY)


@dataclass(frozen=True)
class SyslogEntities:
    """Structured entities extracted from a single log message.

    Every field is optional: a given event only populates what its mnemonic
    exposes. Absent fields stay ``None`` so consumers never guess.
    """

    interface_id: Optional[str] = None
    vlan_id: Optional[str] = None
    mac_address: Optional[str] = None
    ip_address: Optional[str] = None
    port: Optional[str] = None
    flap_count: Optional[int] = None
    erps_ring: Optional[str] = None
    erps_state: Optional[str] = None
    username: Optional[str] = None
    community: Optional[str] = None
    process_name: Optional[str] = None
    fan_id: Optional[str] = None
    power_unit: Optional[str] = None
    login_protocol: Optional[str] = None

    def is_empty(self) -> bool:
        """True when no entity field was populated."""
        return all(v is None for v in asdict(self).values())


@dataclass(frozen=True)
class EngineHints:
    """Which downstream engines a given event is a candidate signal for."""

    engine_a_intrusion: bool = False
    engine_b_health: bool = False
    engine_c_config: bool = False
    engine_c_topology: bool = False
    engine_c_poe: bool = False
    engine_c_security: bool = False
    correlation_candidate: bool = False


@dataclass(frozen=True)
class SyslogEvent:
    """One parsed, structured industrial-switch syslog event."""

    event_id: str
    timestamp: Optional[str]                 # ISO local time (source tz), or None
    timezone: str                            # e.g. "Asia/Kolkata"
    clock_unreliable: bool                   # True for Jan 1 1970 boot-clock lines
    hostname: Optional[str]
    module: Optional[str]
    facility: Optional[str]
    mnemonic: Optional[str]
    severity_code: Optional[int]
    severity_label: str
    message: str
    original_line: str
    cleaned_line: str
    parse_status: str
    timestamp_utc: Optional[str] = None      # ISO UTC when derivable
    ansi_color_hint: Optional[str] = None
    duplicate_count: int = 1                 # exact-repeat weight (>=1)
    entities: SyslogEntities = field(default_factory=SyslogEntities)
    tags: tuple[str, ...] = ()
    engine_hints: EngineHints = field(default_factory=EngineHints)

    @property
    def code(self) -> Optional[str]:
        """The ``FACILITY-MNEMONIC`` code (facility only when no mnemonic)."""
        if self.facility is None:
            return None
        return f"{self.facility}-{self.mnemonic}" if self.mnemonic else self.facility

    @property
    def severity_rank(self) -> int:
        """Numeric rank for sorting (unknown sorts last)."""
        return SEVERITY_ORDER.get(self.severity_label, len(SEVERITY_ORDER))

    def to_dict(self) -> dict[str, Any]:
        """Full nested serialisation (entities/hints as sub-objects)."""
        data = asdict(self)
        data["code"] = self.code
        return data

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for CSV export (entities/hints flattened, prefixed)."""
        row: dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "timestamp_utc": self.timestamp_utc,
            "timezone": self.timezone,
            "clock_unreliable": self.clock_unreliable,
            "hostname": self.hostname,
            "module": self.module,
            "facility": self.facility,
            "mnemonic": self.mnemonic,
            "code": self.code,
            "severity_code": self.severity_code,
            "severity_label": self.severity_label,
            "duplicate_count": self.duplicate_count,
            "ansi_color_hint": self.ansi_color_hint,
            "parse_status": self.parse_status,
            "tags": ";".join(self.tags),
            "message": self.message,
        }
        for key, value in asdict(self.entities).items():
            row[f"ent_{key}"] = value
        for key, value in asdict(self.engine_hints).items():
            row[f"hint_{key}"] = value
        return row


def make_event_id(cleaned_line: str) -> str:
    """Deterministic id for an event from its (post-dedup unique) cleaned line."""
    digest = hashlib.sha1(cleaned_line.encode("utf-8")).hexdigest()[:12]
    return f"SYS-{digest}"
