"""Typed value objects for live logging ingestion (Phase 9).

Purpose
-------
Every ingested input (a Sophos alert, a syslog line, an SNMP metric, a trap, a
config change) is converted into exactly one :class:`NormalizedEvent` — the
single structured contract every downstream consumer (Engine A/B/C, correlation,
the React dashboard) reads. These models are *pure value objects*: no IO, no
device access, no execution.

Inputs / Outputs
----------------
Constructed by :mod:`src.live_logging.normalizer`; serialised via
:meth:`NormalizedEvent.to_dict`. Persisted by
:mod:`src.live_logging.event_store`.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# --- engine targets ---------------------------------------------------------
# The deterministic routing outcome of every normalized event.
ENGINE_CYBER = "cyber"
ENGINE_NETWORK_HEALTH = "network_health"
ENGINE_NETWORK_CONFIG = "network_config"
ENGINE_UNKNOWN = "unknown"

ENGINE_TARGETS: frozenset[str] = frozenset(
    {ENGINE_CYBER, ENGINE_NETWORK_HEALTH, ENGINE_NETWORK_CONFIG, ENGINE_UNKNOWN}
)

# --- severity ---------------------------------------------------------------
# A small, reserved severity vocabulary paired with a deterministic ordering
# (most severe first) for reporting.
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "unknown": 5,
}
UNKNOWN_SEVERITY = "unknown"

# Vendor severity synonyms mapped onto the reserved vocabulary.
_SEVERITY_ALIASES: dict[str, str] = {
    "critical": "critical",
    "crit": "critical",
    "emergency": "critical",
    "alert": "critical",
    "severe": "critical",
    "high": "high",
    "error": "high",
    "err": "high",
    "major": "high",
    "medium": "medium",
    "warning": "medium",
    "warn": "medium",
    "minor": "medium",
    "moderate": "medium",
    "low": "low",
    "notice": "low",
    "info": "info",
    "informational": "info",
    "debug": "info",
    "none": "info",
}


def normalize_severity(value: Any) -> str:
    """Map an arbitrary vendor severity onto the reserved vocabulary."""
    if value is None:
        return UNKNOWN_SEVERITY
    token = str(value).strip().lower()
    if not token:
        return UNKNOWN_SEVERITY
    return _SEVERITY_ALIASES.get(token, UNKNOWN_SEVERITY if token.isnumeric() else token)


def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_event_id(*parts: Any) -> str:
    """Deterministic short event id from its identifying parts.

    Stable across runs for identical inputs so re-ingesting a sample does not
    mint new ids (supports idempotent offline runs and checkpointing).
    """
    digest = hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode())
    return "evt_" + digest.hexdigest()[:16]


@dataclass(frozen=True)
class RawEvent:
    """An unmodified (but redacted) input record, kept for audit/traceability.

    ``payload`` is the redacted source record; secrets are masked before a
    :class:`RawEvent` is ever constructed (see :mod:`redaction`).
    """

    raw_ref: str
    source_vendor: str
    source_product: str
    source_type: str
    source_name: str
    received_at: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary form."""
        return asdict(self)


@dataclass(frozen=True)
class NormalizedEvent:
    """The shared, engine-agnostic event schema (see spec Phase 9)."""

    event_id: str
    timestamp: str
    source_vendor: str
    source_product: str
    source_type: str
    category: str
    severity: str
    message: str
    engine_target: str
    observed_at: str = ""
    source_name: str = ""
    device_id: Optional[str] = None
    device_ip: Optional[str] = None
    hostname: Optional[str] = None
    subcategory: Optional[str] = None
    raw_event_ref: Optional[str] = None
    normalized_fields: dict[str, Any] = field(default_factory=dict)
    correlation_keys: dict[str, Any] = field(default_factory=dict)

    # Fields the spec marks as required (used for validation in tests).
    REQUIRED_FIELDS: tuple[str, ...] = (
        "event_id",
        "timestamp",
        "source_vendor",
        "source_product",
        "source_type",
        "category",
        "severity",
        "message",
        "engine_target",
    )

    def __post_init__(self) -> None:
        if self.engine_target not in ENGINE_TARGETS:
            raise ValueError(
                f"engine_target must be one of {sorted(ENGINE_TARGETS)}, "
                f"got {self.engine_target!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary form (excludes the class constant)."""
        data = asdict(self)
        data.pop("REQUIRED_FIELDS", None)
        return data

    @property
    def severity_rank(self) -> int:
        """Sort key: lower is more severe."""
        return SEVERITY_ORDER.get(self.severity, SEVERITY_ORDER[UNKNOWN_SEVERITY])


@dataclass(frozen=True)
class Checkpoint:
    """Per-source cursor state so polling can resume safely."""

    source: str
    cursor: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary form."""
        return asdict(self)


# --- ingestion status -------------------------------------------------------
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DISABLED = "disabled"
STATUS_SKIPPED = "skipped"

# Failure categories (spec Phase 9 > Failure and Retry Strategy).
FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "api_auth_error",
        "api_rate_limit",
        "network_timeout",
        "syslog_parse_error",
        "snmp_timeout",
        "trap_parse_error",
        "config_diff_error",
        "checkpoint_write_error",
        "redaction_error",
        "unknown_error",
    }
)


@dataclass
class SourceStatus:
    """Per-source outcome of an ingestion run."""

    source: str
    engine_target: str
    status: str = STATUS_OK
    mode: str = "offline"
    events: int = 0
    raw_events: int = 0
    attempts: int = 1
    error_category: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary form."""
        return asdict(self)


@dataclass
class IngestionStatus:
    """Aggregate outcome of a whole ingestion run."""

    started_at: str
    finished_at: str
    mode: str
    read_only: bool
    total_events: int
    sources: list[SourceStatus] = field(default_factory=list)
    events_by_engine: dict[str, int] = field(default_factory=dict)
    events_by_severity: dict[str, int] = field(default_factory=dict)
    events_by_vendor: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary form."""
        data = asdict(self)
        data["sources"] = [s.to_dict() for s in self.sources]
        return data

    @property
    def healthy(self) -> bool:
        """True when no enabled source failed."""
        return all(s.status != STATUS_FAILED for s in self.sources)
