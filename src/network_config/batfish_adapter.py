"""Engine C Phase 8 — optional Batfish configuration-validation adapter.

Purpose
-------
Optionally cross-check saved device configurations against the Batfish
configuration model and surface *external validation evidence* (parse status,
node/interface properties, L3 edges, undefined references). This adapter is
**disabled by default** and strictly additive: it never replaces the offline
parsers, topology builder, rule engine, remediation planner or reports, and it
never accesses a live device or executes a command.

Optionality
-----------
``pybatfish`` is imported **lazily inside** :func:`run_batfish_validation` — the
whole of Engine C imports and runs without Batfish, Docker or pybatfish present.
When Batfish is unavailable the adapter degrades to a clearly-marked ``skipped``
result rather than raising (unless ``fail_if_unavailable`` is honoured by the
caller). Tests inject a fake session via ``session_factory`` and never require a
running Batfish service.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

SAFETY_NOTE = ("External configuration validation only; no device access and "
               "no commands were executed.")

# Status values for a validation run.
STATUS_DISABLED = "disabled"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_SUCCESS = "success"

# Logical question name -> pybatfish question attribute (camelCase as exposed
# on ``session.q``). Kept here so the set is config-driven, not hardcoded inline.
_QUESTIONS: dict[str, str] = {
    "parse_status": "fileParseStatus",
    "node_properties": "nodeProperties",
    "interface_properties": "interfaceProperties",
    "l3_edges": "layer3Edges",
    "undefined_references": "undefinedReferences",
}


class BatfishUnavailableError(RuntimeError):
    """Raised internally when pybatfish/Batfish cannot be used."""


# ------------------------------------------------------------------- models


@dataclass(frozen=True)
class BatfishTableResult:
    """The outcome of one Batfish question."""

    name: str
    status: str                          # success / failed / skipped
    row_count: int = 0
    rows: tuple[dict[str, Any], ...] = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class BatfishFinding:
    """A lightweight finding derived from Batfish output (external evidence)."""

    finding_id: str
    category: str
    severity: str
    title: str
    device: Optional[str]
    interface: Optional[str]
    evidence: Optional[str]
    recommendation: Optional[str]
    source: str = "batfish"
    confidence: str = "medium"


@dataclass(frozen=True)
class BatfishValidationResult:
    """The full result of an (optional) Batfish validation run."""

    snapshot_id: str
    status: str                          # disabled/skipped/failed/success
    reason: Optional[str] = None
    tables: tuple[BatfishTableResult, ...] = ()
    findings: tuple[BatfishFinding, ...] = ()
    node_count: int = 0
    interface_count: int = 0
    l3_edge_count: int = 0
    undefined_reference_count: int = 0
    parse_status_summary: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""
    safety_note: str = SAFETY_NOTE


# ------------------------------------------------------------------- config


def load_batfish_config(path: str | Path) -> dict[str, Any]:
    """Load ``configs/batfish.yaml`` (raises if the file is absent)."""
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Batfish config not found: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


# ------------------------------------------------------------------- runner


def run_batfish_validation(
    snapshot_id: str,
    config: Mapping[str, Any],
    snapshot_path: str | Path,
    *,
    session_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
) -> BatfishValidationResult:
    """Run the optional Batfish validation and return a structured result.

    ``session_factory`` (for tests) returns a Batfish-session-like object; when
    ``None`` the real ``pybatfish`` session is imported lazily. Any
    unavailability (no pybatfish, no service, missing snapshot, parse failure)
    yields a ``skipped``/``failed`` result rather than crashing Engine C — the
    caller decides whether ``fail_if_unavailable`` should turn that into an
    error exit.
    """
    now = datetime.now(timezone.utc).isoformat()
    global_cfg = dict(config.get("global") or {})
    if not global_cfg.get("enabled", False):
        return BatfishValidationResult(
            snapshot_id=snapshot_id, status=STATUS_DISABLED, timestamp=now,
            reason="Batfish validation is disabled in configuration.")

    try:
        session = _get_session(config, session_factory)
    except BatfishUnavailableError as exc:
        logger.warning("Batfish unavailable: %s", exc)
        return BatfishValidationResult(
            snapshot_id=snapshot_id, status=STATUS_SKIPPED, timestamp=now,
            reason=str(exc))

    path = Path(snapshot_path)
    if not path.is_dir():
        reason = f"Batfish snapshot directory not found: {path}"
        logger.warning("%s", reason)
        return BatfishValidationResult(
            snapshot_id=snapshot_id, status=STATUS_SKIPPED, timestamp=now,
            reason=reason)

    try:
        _init_snapshot(session, snapshot_id, config, path)
    except Exception as exc:                       # noqa: BLE001 - report, don't crash
        reason = f"Batfish snapshot initialisation failed: {exc}"
        logger.warning("%s", reason)
        return BatfishValidationResult(
            snapshot_id=snapshot_id, status=STATUS_FAILED, timestamp=now,
            reason=reason)

    tables = _run_questions(session, config)
    by_name = {t.name: t for t in tables}
    findings = _findings_from_tables(by_name)
    any_success = any(t.status == STATUS_SUCCESS for t in tables)
    status = STATUS_SUCCESS if any_success else STATUS_FAILED
    reason = None if any_success else "All Batfish questions failed."

    return BatfishValidationResult(
        snapshot_id=snapshot_id, status=status, reason=reason,
        tables=tuple(tables), findings=tuple(findings),
        node_count=_rows(by_name, "node_properties"),
        interface_count=_rows(by_name, "interface_properties"),
        l3_edge_count=_rows(by_name, "l3_edges"),
        undefined_reference_count=_rows(by_name, "undefined_references"),
        parse_status_summary=_parse_summary(by_name.get("parse_status")),
        timestamp=now)


def _get_session(config, session_factory):
    """Return a Batfish session (fake in tests, lazily-imported real otherwise)."""
    if session_factory is not None:
        return session_factory(config)
    try:
        from pybatfish.client.session import Session  # lazy: never at import time
    except ImportError as exc:
        raise BatfishUnavailableError(
            "pybatfish is not installed (Batfish validation is optional).") from exc
    conn = dict(config.get("connection") or {})
    try:
        return Session(host=str(conn.get("host", "localhost")))
    except Exception as exc:                        # noqa: BLE001
        raise BatfishUnavailableError(
            f"could not connect to the Batfish service: {exc}") from exc


def _init_snapshot(session, snapshot_id, config, path: Path) -> None:
    conn = dict(config.get("connection") or {})
    network = str(conn.get("network_name", "nims_engine_c"))
    prefix = str(conn.get("snapshot_name_prefix", "nims_snapshot"))
    session.set_network(network)
    session.init_snapshot(str(path), name=f"{prefix}_{snapshot_id}",
                          overwrite=True)


def _run_questions(session, config) -> list[BatfishTableResult]:
    questions_cfg = dict(config.get("questions") or {})
    results: list[BatfishTableResult] = []
    for name, attr in _QUESTIONS.items():
        if not questions_cfg.get(name, True):
            results.append(BatfishTableResult(name, STATUS_SKIPPED,
                                              error="disabled in configuration"))
            continue
        try:
            question = getattr(session.q, attr)
            frame = question().answer().frame()
            rows = _frame_to_rows(frame)
            results.append(BatfishTableResult(name, STATUS_SUCCESS, len(rows),
                                              tuple(rows)))
        except Exception as exc:                    # noqa: BLE001 - per-question isolation
            logger.warning("Batfish question '%s' failed: %s", name, exc)
            results.append(BatfishTableResult(name, STATUS_FAILED,
                                              error=str(exc)))
    return results


# ------------------------------------------------------------------- helpers


def _frame_to_rows(frame: Any) -> list[dict[str, Any]]:
    """Normalise a pandas DataFrame (or list of dicts) into JSON-safe rows."""
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        records = frame.to_dict(orient="records")
    else:
        records = list(frame)
    return [{str(k): _norm(v) for k, v in dict(r).items()} for r in records]


def _norm(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return str(value)


def _rows(by_name: dict[str, BatfishTableResult], name: str) -> int:
    table = by_name.get(name)
    return table.row_count if table else 0


def _get(row: Mapping[str, Any], *keys: str) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        value = lower.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def _first_node(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value) if value not in (None, "") else None


def _parse_summary(table: Optional[BatfishTableResult]) -> dict[str, int]:
    summary = {"passed": 0, "failed": 0, "partially_parsed": 0}
    if not table:
        return summary
    for row in table.rows:
        status = str(_get(row, "Status") or "").upper().replace(" ", "_")
        if status == "PASSED":
            summary["passed"] += 1
        elif status == "FAILED":
            summary["failed"] += 1
        elif status.startswith("PARTIAL"):
            summary["partially_parsed"] += 1
    return summary


def _bf_id(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"BF-{digest[:8]}"


def _bf_finding(category, severity, title, device, interface, evidence,
                recommendation, confidence) -> BatfishFinding:
    return BatfishFinding(
        finding_id=_bf_id(category, title, device, interface, evidence),
        category=category, severity=severity, title=title, device=device,
        interface=interface, evidence=evidence, recommendation=recommendation,
        confidence=confidence)


def _findings_from_tables(
    by_name: dict[str, BatfishTableResult]
) -> list[BatfishFinding]:
    """Derive cautious findings from parse-status and undefined-reference tables."""
    findings: list[BatfishFinding] = []

    parse_status = by_name.get("parse_status")
    if parse_status:
        for row in parse_status.rows:
            status = str(_get(row, "Status") or "").upper().replace(" ", "_")
            fname = _get(row, "File_Name", "File", "Filename")
            node = _first_node(_get(row, "Nodes", "Node"))
            if status == "FAILED":
                findings.append(_bf_finding(
                    "config", "high", "Configuration file failed to parse", node,
                    None, f"Batfish could not parse {fname}.",
                    "Review the configuration; Engine C parsing may be "
                    "incomplete for this file.", "high"))
            elif status.startswith("PARTIAL"):
                findings.append(_bf_finding(
                    "config", "medium", "Configuration file partially parsed",
                    node, None, f"Batfish partially parsed {fname}.",
                    "Some lines were not modelled; review for unsupported "
                    "constructs.", "medium"))

    undefined = by_name.get("undefined_references")
    if undefined:
        for row in undefined.rows:
            stype = _get(row, "Structure_Type", "Struct_Type")
            ref = _get(row, "Ref_Name", "Reference")
            fname = _get(row, "File_Name", "File")
            node = _first_node(_get(row, "Nodes", "Node"))
            findings.append(_bf_finding(
                "config", "medium", "Undefined reference in configuration", node,
                None, f"{stype} '{ref}' is referenced but undefined ({fname}).",
                "Define the referenced object or remove the stale reference.",
                "medium"))

    return findings
