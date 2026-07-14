"""Run orchestration and artefact writing for syslog ingestion.

Purpose
-------
Tie the offline pipeline together: read saved log files -> preprocess -> parse
-> enrich -> build Engine B feature windows and Engine C findings, then persist
every artefact under ``outputs/syslog_ingestion/<run_id>/``.

Strictly offline: the only inputs are local files and the only outputs are local
files. No device is contacted, no packet is captured, no command is executed.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.syslog_ingestion.features import (
    build_windows,
    chronological_split,
    summarize_features,
    summarize_weak_labels,
)
from src.syslog_ingestion.findings import (
    SyslogFinding,
    generate_findings,
    summarize_findings,
)
from src.syslog_ingestion.models import SyslogEvent
from src.syslog_ingestion.parser import parse_lines
from src.syslog_ingestion.preprocess import preprocess_lines
from src.utils.io import write_json

logger = logging.getLogger(__name__)

SAFETY_BANNER = ("offline saved-log mode; no device access; no packet capture; "
                 "no remediation")


@dataclass
class IngestRun:
    """In-memory result of one ingestion run, before/independent of writing."""

    run_id: str
    input_files: list[str]
    events: list[SyslogEvent]
    dropped: list[dict[str, Any]]
    duplicates: list[dict[str, Any]]
    windows: dict[int, list[dict[str, Any]]]        # window_minutes -> rows
    split_manifest: dict[str, Any]
    findings: list[SyslogFinding]
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_window_minutes(self) -> int:
        ingest = self.config.get("syslog_ingestion", {}) or {}
        return int(ingest.get("default_window_minutes", 5))


# --------------------------------------------------------------- input IO
def read_input_files(files: list[str] | None, directory: str | None
                     ) -> dict[str, list[str]]:
    """Read one or more local log files into ``{path: lines}``.

    Accepts an explicit file list and/or a directory (all regular files, sorted).
    Raises ``FileNotFoundError`` when nothing usable is found.
    """
    paths: list[Path] = []
    for f in files or []:
        paths.append(Path(f))
    if directory:
        d = Path(directory)
        if d.is_dir():
            paths.extend(sorted(p for p in d.iterdir() if p.is_file()))
    resolved = [p for p in paths if p.is_file()]
    if not resolved:
        raise FileNotFoundError(
            "No input log files found (checked "
            f"files={files!r}, dir={directory!r}).")

    contents: dict[str, list[str]] = {}
    for path in resolved:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            contents[str(path)] = fh.readlines()
    return contents


# --------------------------------------------------------------- pipeline
def ingest(
    contents: Mapping[str, list[str]],
    run_id: str,
    config: Mapping[str, Any],
    *,
    window_minutes: int | None = None,
    include_clock_unreliable: bool = False,
    host_holdout: str | None = None,
) -> IngestRun:
    """Run the offline pipeline over already-read file contents."""
    effective = _effective_config(config, include_clock_unreliable, host_holdout)
    ingest_cfg = effective.get("syslog_ingestion", {}) or {}
    primary = int(window_minutes or ingest_cfg.get("default_window_minutes", 5))
    extra = [int(m) for m in (ingest_cfg.get("additional_window_minutes") or [])]
    window_sizes = list(dict.fromkeys([primary, *extra]))

    events: list[SyslogEvent] = []
    dropped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for source, lines in contents.items():
        pre = preprocess_lines(lines, effective, source=source)
        dropped.extend(pre.dropped)
        duplicates.extend(pre.duplicates)
        events.extend(parse_lines(pre.kept, effective))

    windows: dict[int, list[dict[str, Any]]] = {}
    split_manifest: dict[str, Any] = {}
    for size in window_sizes:
        rows = build_windows(events, size, effective)
        if size == primary:
            rows, split_manifest = chronological_split(rows, effective)
        windows[size] = rows

    findings = generate_findings(events, effective)
    return IngestRun(
        run_id=run_id, input_files=list(contents.keys()), events=events,
        dropped=dropped, duplicates=duplicates, windows=windows,
        split_manifest=split_manifest, findings=findings, config=effective)


def _effective_config(config: Mapping[str, Any], include_clock_unreliable: bool,
                      host_holdout: str | None) -> dict[str, Any]:
    """Apply CLI overrides onto a copy of the config."""
    from copy import deepcopy

    eff = deepcopy(dict(config))
    if include_clock_unreliable:
        eff.setdefault("syslog_ingestion", {})[
            "drop_clock_unreliable_from_features"] = False
    if host_holdout:
        eff.setdefault("splitting", {})["host_holdout"] = host_holdout
    return eff


# --------------------------------------------------------------- summaries
def parser_summary(run: IngestRun) -> dict[str, Any]:
    """Build the parser-quality / coverage summary for ``parser_summary.json``."""
    events = run.events
    status = Counter(e.parse_status for e in events)
    severity = Counter(e.severity_label for e in events)
    facility = Counter(e.facility for e in events if e.facility)
    mnemonic = Counter(e.code for e in events if e.code)
    hosts = sorted({e.hostname for e in events if e.hostname})
    stamps = sorted(e.timestamp for e in events
                    if e.timestamp and not e.clock_unreliable)
    weighted = sum(max(1, int(e.duplicate_count)) for e in events)

    return {
        "run_id": run.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safety": SAFETY_BANNER,
        "input_files": run.input_files,
        "parsed_events": len(events),
        "weighted_events": weighted,
        "dropped_lines": len(run.dropped),
        "duplicate_lines_collapsed": sum(int(d["duplicate_count"]) - 1
                                         for d in run.duplicates),
        "parse_status": dict(status),
        "hosts": hosts,
        "time_range": {"first": stamps[0] if stamps else None,
                       "last": stamps[-1] if stamps else None},
        "severity_distribution": dict(severity),
        "top_facilities": dict(facility.most_common(15)),
        "top_mnemonics": dict(mnemonic.most_common(20)),
        "clock_unreliable_events": sum(1 for e in events if e.clock_unreliable),
    }


# --------------------------------------------------------------- writing
def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write a list of dict rows to CSV (empty file with no rows)."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_run(run: IngestRun, output_dir: str | Path) -> dict[str, str]:
    """Persist all artefacts for a run and return a map of key -> path."""
    from src.syslog_ingestion.reporting import build_report

    base = Path(output_dir)
    engine_b = base / "engine_b"
    engine_c = base / "engine_c"
    paths: dict[str, str] = {}

    summary = parser_summary(run)

    # --- core parsed artefacts ---
    write_json([e.to_dict() for e in run.events], base / "parsed_events.json")
    _write_csv([e.to_row() for e in run.events], base / "parsed_events.csv")
    write_json(summary, base / "parser_summary.json")
    _write_csv(run.dropped, base / "dropped_lines.csv")
    _write_csv([{"source": d.get("source"), "duplicate_count": d["duplicate_count"],
                 "cleaned_line": d["cleaned_line"]} for d in run.duplicates],
               base / "duplicate_lines.csv")
    paths["parsed_events_json"] = str(base / "parsed_events.json")
    paths["parser_summary"] = str(base / "parser_summary.json")

    # --- Engine B feature windows ---
    primary = run.primary_window_minutes
    feature_summary = {}
    weak_label_summary = {}
    for size, rows in run.windows.items():
        csv_path = engine_b / f"syslog_windows_{size}min.csv"
        _write_csv(rows, csv_path)
        if size == primary:
            feature_summary = summarize_features(rows)
            weak_label_summary = summarize_weak_labels(rows)
            paths["engine_b_windows"] = str(csv_path)
    write_json(run.split_manifest, engine_b / "split_manifest.json")
    write_json(feature_summary, engine_b / "feature_summary.json")
    write_json(weak_label_summary, engine_b / "weak_label_summary.json")

    # --- Engine C findings ---
    # TODO(Phase 13): syslog_findings.json is a stable, self-describing artefact
    # (finding_id/severity/category/device/interface/evidence/tags). A future
    # correlation loader can consume it directly as a fourth signal source
    # (alongside Engine A/B/C) without re-parsing raw logs.
    write_json([f.to_dict() for f in run.findings], engine_c / "syslog_findings.json")
    _write_csv([f.to_row() for f in run.findings], engine_c / "syslog_findings.csv")
    write_json(summarize_findings(run.findings), engine_c / "syslog_rule_summary.json")
    paths["engine_c_findings"] = str(engine_c / "syslog_findings.json")

    # --- report ---
    report_path = base / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(run, summary, feature_summary,
                                        weak_label_summary), encoding="utf-8")
    paths["report"] = str(report_path)

    return paths
