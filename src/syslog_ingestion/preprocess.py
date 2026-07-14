"""Preprocessing: turn raw terminal/log text into clean, dedup'd syslog lines.

Purpose
-------
Real captures arrive two ways: clean raw syslog exports, and PuTTY terminal
captures polluted with ANSI colour codes, ``---MORE---`` pager markers, command
echoes and ``show logging`` headers. This module normalises both into a list of
:class:`PreprocessedLine` records ready for grammar parsing, while preserving an
audit trail of everything it dropped or collapsed.

Order of operations per line (all configurable):
    1. capture the ANSI foreground-colour hint (before stripping)
    2. strip ANSI escape sequences
    3. remove inline ``---MORE---`` pager markers
    4. trim whitespace
    5. drop terminal/pager/echo noise (recorded, never silently lost)
Then, across the surviving lines:
    6. deduplicate exact repeats, accumulating a ``duplicate_count`` weight.

This module performs no IO beyond accepting an in-memory list of lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.syslog_ingestion.models import ANSI_COLOR_HINTS

# ANSI SGR escape sequence, e.g. "\x1b[1;33m" or "\x1b[0m".
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Foreground colour code (30-37) inside an SGR sequence, for the colour hint.
_ANSI_SGR_RE = re.compile(r"\x1b\[([0-9;]+)m")
# PuTTY pager marker; may prefix an otherwise-valid log line.
_MORE_RE = re.compile(r"-{2,}MORE-{2,}\s*")
# The discriminating marker of a real syslog line: a ``%FAC-MNEM-<sev>:`` code.
_CODE_MARKER_RE = re.compile(r"%[A-Z0-9]+-(?:[A-Z0-9_]+-)?\d:")

# Whole-line terminal/echo/header noise (matched against the cleaned line).
_NOISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("putty_header", re.compile(r"^=~=.*PuTTY log", re.IGNORECASE)),
    ("logging_header", re.compile(r"Logging source configurations", re.IGNORECASE)),
    ("logging_context", re.compile(r"The Context of logging file", re.IGNORECASE)),
    ("logging_level", re.compile(r"\bis enabled,\s*level:", re.IGNORECASE)),
    ("command_echo", re.compile(r"^\S*#")),        # HOST#sh logging
    ("bare_command", re.compile(r"^(sh|show|dis|display)\s", re.IGNORECASE)),
)


@dataclass(frozen=True)
class PreprocessedLine:
    """One surviving log line plus its audit metadata and dedup weight."""

    original_line: str
    cleaned_line: str
    ansi_color_hint: str | None = None
    duplicate_count: int = 1


@dataclass
class PreprocessResult:
    """Kept lines plus a full audit trail of drops and collapses."""

    kept: list[PreprocessedLine] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duplicate_lines_collapsed(self) -> int:
        """Total number of repeat lines folded away (sum of extra counts)."""
        return sum(int(d["duplicate_count"]) - 1 for d in self.duplicates)


def _flag(config: Mapping[str, Any], key: str, default: bool = True) -> bool:
    block = config.get("preprocessing", {}) if config else {}
    return bool(block.get(key, default))


def capture_color_hint(line: str) -> str | None:
    """Return the semantic ANSI colour hint (yellow/cyan/white) if present."""
    for match in _ANSI_SGR_RE.finditer(line):
        for code in match.group(1).split(";"):
            if code in ANSI_COLOR_HINTS:
                return ANSI_COLOR_HINTS[code]
    return None


def strip_ansi(line: str) -> str:
    """Remove all ANSI SGR escape sequences from ``line``."""
    return _ANSI_RE.sub("", line)


def _classify_noise(cleaned: str) -> str | None:
    """Return a drop reason if ``cleaned`` is terminal noise, else ``None``."""
    if not cleaned:
        return "empty"
    if _CODE_MARKER_RE.search(cleaned):
        return None  # a real syslog line always wins over noise heuristics
    for reason, pattern in _NOISE_PATTERNS:
        if pattern.search(cleaned):
            return reason
    return "no_syslog_marker"


def clean_line(line: str, config: Mapping[str, Any]) -> tuple[str, str | None]:
    """Clean one raw line; return ``(cleaned_line, ansi_color_hint)``."""
    hint: str | None = None
    if _flag(config, "capture_ansi_color_hint"):
        hint = capture_color_hint(line)
    cleaned = line
    if _flag(config, "strip_ansi"):
        cleaned = strip_ansi(cleaned)
    if _flag(config, "drop_pager_noise"):
        cleaned = _MORE_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned, hint


def preprocess_lines(
    raw_lines: list[str],
    config: Mapping[str, Any] | None = None,
    *,
    source: str | None = None,
) -> PreprocessResult:
    """Clean, filter and deduplicate a list of raw log lines.

    Parameters
    ----------
    raw_lines:
        Lines exactly as read from a file (newline-stripped or not).
    config:
        Effective configuration; the ``preprocessing`` and ``syslog_ingestion``
        blocks control ANSI/noise handling and deduplication.
    source:
        Optional source filename, recorded on drop records for auditing.

    Returns
    -------
    PreprocessResult
    """
    config = config or {}
    drop_prompt = _flag(config, "drop_prompt_echo")
    dedup = bool(
        (config.get("syslog_ingestion", {}) or {}).get(
            "deduplicate_exact_repeats", True
        )
    )

    result = PreprocessResult()
    seen: dict[str, PreprocessedLine] = {}

    for raw in raw_lines:
        raw = raw.rstrip("\n").rstrip("\r")
        cleaned, hint = clean_line(raw, config)
        reason = _classify_noise(cleaned)

        # Honour the drop_prompt_echo toggle: when disabled we still drop empty
        # and marker-less lines (they cannot be parsed) but keep echoes visible.
        if reason in {"command_echo", "bare_command"} and not drop_prompt:
            reason = None if _CODE_MARKER_RE.search(cleaned) else reason

        if reason is not None:
            result.dropped.append(
                {"source": source, "reason": reason,
                 "original_line": raw, "cleaned_line": cleaned}
            )
            continue

        if dedup and cleaned in seen:
            first = seen[cleaned]
            merged = PreprocessedLine(
                original_line=first.original_line,
                cleaned_line=first.cleaned_line,
                ansi_color_hint=first.ansi_color_hint or hint,
                duplicate_count=first.duplicate_count + 1,
            )
            seen[cleaned] = merged
            continue

        record = PreprocessedLine(
            original_line=raw, cleaned_line=cleaned, ansi_color_hint=hint,
        )
        seen[cleaned] = record
        result.kept.append(record)

    if dedup:
        # Reconcile kept records with their accumulated duplicate counts, and
        # surface every collapsed group in the audit trail.
        final: list[PreprocessedLine] = []
        for record in result.kept:
            merged = seen[record.cleaned_line]
            final.append(merged)
            if merged.duplicate_count > 1:
                result.duplicates.append(
                    {"source": source,
                     "cleaned_line": merged.cleaned_line,
                     "duplicate_count": merged.duplicate_count}
                )
        result.kept = final

    return result
