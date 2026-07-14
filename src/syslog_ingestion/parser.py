"""Syslog grammar parsing: cleaned lines -> :class:`SyslogEvent` objects.

Purpose
-------
Parse the general Belden/DCN industrial-switch log grammar::

    <Mon DD YYYY HH:MM:SS> <hostname> [<module>] %<FACILITY>-<MNEMONIC>-<SEV>:<msg>

into typed events. Timestamps carry no timezone, so a configured default
(Asia/Kolkata / IST, UTC+05:30, no DST) is applied. ``Jan 1 1970`` boot-clock
lines are parsed but flagged ``clock_unreliable`` so feature windowing can
exclude them while reboot analysis keeps them.

Grammar failures never raise: the line is returned as a ``failed`` event for the
audit trail. Mnemonic-specific enrichment (entities/tags/engine hints) is
delegated to :mod:`src.syslog_ingestion.extractors`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional

from src.syslog_ingestion.extractors import enrich_event
from src.syslog_ingestion.models import (
    FAILED,
    SyslogEvent,
    make_event_id,
    severity_label,
)
from src.syslog_ingestion.preprocess import PreprocessedLine

logger = logging.getLogger(__name__)

# Timestamp prefix, e.g. "Jun 13 2026 04:44:05" (day may be space-padded).
_TS_RE = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{4}\s+\d{2}:\d{2}:\d{2})\s+(?P<rest>.*)$"
)
# host, optional module (a token not starting with '%'), then the %code.
_REST_RE = re.compile(
    r"^(?P<host>\S+)\s+(?:(?P<module>[^%\s]\S*)\s+)?(?P<code>%.*)$"
)
# %FACILITY-[MNEMONIC-]SEVERITY:message  (mnemonic optional, e.g. %SYSMGMT-4:)
_CODE_RE = re.compile(
    r"^%(?P<facility>[A-Z0-9]+)-(?:(?P<mnemonic>[A-Z0-9_]+)-)?"
    r"(?P<severity>\d):(?P<message>.*)$"
)

_TS_FORMAT = "%b %d %Y %H:%M:%S"

# Fixed UTC offsets for the timezone names we support (industrial sites are
# single-zone; IST has no DST so a fixed offset is correct and avoids depending
# on a tz database being installed on Windows).
_TZ_OFFSETS: dict[str, timedelta] = {
    "Asia/Kolkata": timedelta(hours=5, minutes=30),
    "IST": timedelta(hours=5, minutes=30),
    "UTC": timedelta(0),
}
_DEFAULT_TZ = "Asia/Kolkata"


def _resolve_offset(tz_name: str) -> timedelta:
    """Return the fixed UTC offset for a supported timezone name."""
    if tz_name in _TZ_OFFSETS:
        return _TZ_OFFSETS[tz_name]
    try:  # pragma: no cover - only when tzdata is installed
        from zoneinfo import ZoneInfo

        now = datetime(2026, 1, 1, tzinfo=ZoneInfo(tz_name))
        return now.utcoffset() or timedelta(0)
    except Exception:  # noqa: BLE001 - fall back to IST default
        logger.warning("Unknown timezone %s; defaulting to IST (+05:30)", tz_name)
        return _TZ_OFFSETS[_DEFAULT_TZ]


def parse_timestamp(
    ts_str: str, tz_name: str
) -> tuple[Optional[str], Optional[str], bool]:
    """Parse a log timestamp into ``(iso_local, iso_utc, clock_unreliable)``.

    ``clock_unreliable`` is True for pre-boot clock values (year < 2000, i.e.
    the ``Jan 1 1970`` lines emitted before NTP sync).
    """
    normalised = re.sub(r"\s+", " ", ts_str.strip())
    try:
        naive = datetime.strptime(normalised, _TS_FORMAT)
    except ValueError:
        return None, None, False

    offset = _resolve_offset(tz_name)
    local = naive.replace(tzinfo=timezone(offset))
    utc = local.astimezone(timezone.utc)
    clock_unreliable = naive.year < 2000
    return local.isoformat(), utc.isoformat(), clock_unreliable


def _default_tz(config: Mapping[str, object] | None) -> str:
    block = (config or {}).get("syslog_ingestion", {}) if config else {}
    return str(block.get("default_timezone", _DEFAULT_TZ))  # type: ignore[union-attr]


def parse_line(
    line: PreprocessedLine, config: Mapping[str, object] | None = None
) -> SyslogEvent:
    """Parse one preprocessed line into an enriched :class:`SyslogEvent`.

    A line that does not match the grammar becomes a ``failed`` event (kept for
    audit) rather than raising.
    """
    tz_name = _default_tz(config)
    cleaned = line.cleaned_line

    ts_match = _TS_RE.match(cleaned)
    if not ts_match:
        return _failed_event(line, tz_name, reason="no_timestamp")

    rest_match = _REST_RE.match(ts_match.group("rest"))
    if not rest_match:
        return _failed_event(line, tz_name, reason="no_host_code")

    code_match = _CODE_RE.match(rest_match.group("code"))
    if not code_match:
        return _failed_event(line, tz_name, reason="no_code")

    iso_local, iso_utc, clock_unreliable = parse_timestamp(
        ts_match.group("ts"), tz_name
    )
    severity_code = int(code_match.group("severity"))
    base = SyslogEvent(
        event_id=make_event_id(cleaned),
        timestamp=iso_local,
        timestamp_utc=iso_utc,
        timezone=tz_name,
        clock_unreliable=clock_unreliable,
        hostname=rest_match.group("host"),
        module=rest_match.group("module"),
        facility=code_match.group("facility"),
        mnemonic=code_match.group("mnemonic"),
        severity_code=severity_code,
        severity_label=severity_label(severity_code),
        message=code_match.group("message").strip(),
        original_line=line.original_line,
        cleaned_line=cleaned,
        parse_status="generic",
        ansi_color_hint=line.ansi_color_hint,
        duplicate_count=line.duplicate_count,
    )
    return enrich_event(base)


def _failed_event(
    line: PreprocessedLine, tz_name: str, *, reason: str
) -> SyslogEvent:
    """Build a ``failed`` event for a line that did not match the grammar."""
    return SyslogEvent(
        event_id=make_event_id(line.cleaned_line),
        timestamp=None,
        timestamp_utc=None,
        timezone=tz_name,
        clock_unreliable=False,
        hostname=None,
        module=None,
        facility=None,
        mnemonic=None,
        severity_code=None,
        severity_label="unknown",
        message=line.cleaned_line,
        original_line=line.original_line,
        cleaned_line=line.cleaned_line,
        parse_status=FAILED,
        ansi_color_hint=line.ansi_color_hint,
        duplicate_count=line.duplicate_count,
        tags=(f"parse_failed:{reason}",),
    )


def parse_lines(
    lines: list[PreprocessedLine], config: Mapping[str, object] | None = None
) -> list[SyslogEvent]:
    """Parse many preprocessed lines, preserving order."""
    return [parse_line(line, config) for line in lines]
