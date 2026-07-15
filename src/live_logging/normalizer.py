"""Convert vendor source records into the shared NormalizedEvent schema.

Purpose
-------
Vendor parsers (:mod:`sophos_api`, :mod:`sophos_syslog`, :mod:`hirschmann_snmp`,
:mod:`hirschmann_traps`, :mod:`config_diff`) each emit *source records* — plain
dicts describing one observation. This module is the single place that turns a
source record into a :class:`NormalizedEvent` (and its paired
:class:`RawEvent`), applying severity normalization, deterministic id
generation and deterministic routing.

Keeping the shared-schema construction here (and vendor interpretation in the
vendor modules) means every event, regardless of origin, is built identically.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.live_logging.models import (
    NormalizedEvent,
    RawEvent,
    make_event_id,
    normalize_severity,
    utc_now_iso,
)
from src.live_logging.routing import route

# Keys a source record must provide.
_REQUIRED_RECORD_KEYS: tuple[str, ...] = (
    "source_vendor",
    "source_product",
    "source_type",
    "source_key",
    "category",
    "message",
)


def build_normalized(
    record: Mapping[str, Any],
    routing: Mapping[str, str] | None = None,
) -> tuple[NormalizedEvent, RawEvent]:
    """Build a (NormalizedEvent, RawEvent) pair from a vendor source record.

    Parameters
    ----------
    record:
        Source record dict (see module docstring for the expected keys).
    routing:
        Optional engine-target routing map (source_key -> engine_target).

    Raises
    ------
    KeyError
        If a required record key is missing.
    """
    missing = [k for k in _REQUIRED_RECORD_KEYS if not record.get(k)]
    if missing:
        raise KeyError(f"Source record missing required keys: {missing}")

    source_key = str(record["source_key"])
    observed_at = str(record.get("observed_at") or utc_now_iso())
    timestamp = str(record.get("timestamp") or observed_at)
    raw_ref = str(record.get("raw_ref") or make_event_id(source_key, timestamp, record["message"]))
    engine_target = route(source_key, routing)

    event = NormalizedEvent(
        event_id=make_event_id(source_key, raw_ref, timestamp, record["message"]),
        timestamp=timestamp,
        observed_at=observed_at,
        source_vendor=str(record["source_vendor"]),
        source_product=str(record["source_product"]),
        source_type=str(record["source_type"]),
        source_name=str(record.get("source_name") or source_key),
        device_id=_opt(record.get("device_id")),
        device_ip=_opt(record.get("device_ip")),
        hostname=_opt(record.get("hostname")),
        category=str(record["category"]),
        subcategory=_opt(record.get("subcategory")),
        severity=normalize_severity(record.get("severity")),
        message=str(record["message"]),
        raw_event_ref=raw_ref,
        normalized_fields=dict(record.get("normalized_fields") or {}),
        engine_target=engine_target,
        correlation_keys=dict(record.get("correlation_keys") or {}),
    )

    raw = RawEvent(
        raw_ref=raw_ref,
        source_vendor=event.source_vendor,
        source_product=event.source_product,
        source_type=event.source_type,
        source_name=event.source_name,
        received_at=observed_at,
        payload=dict(record.get("raw_payload") or {}),
    )
    return event, raw


def build_batch(
    records: list[Mapping[str, Any]],
    routing: Mapping[str, str] | None = None,
) -> tuple[list[NormalizedEvent], list[RawEvent]]:
    """Normalize a batch of source records into parallel event lists."""
    events: list[NormalizedEvent] = []
    raws: list[RawEvent] = []
    for record in records:
        event, raw = build_normalized(record, routing)
        events.append(event)
        raws.append(raw)
    return events, raws


def _opt(value: Any) -> Any:
    """Coerce empty strings to None for optional identity fields."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
