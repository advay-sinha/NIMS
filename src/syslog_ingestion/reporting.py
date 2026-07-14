"""Markdown report generation for a syslog ingestion run.

Renders the human-readable ``report.md`` from the in-memory run and its
summaries. Pure string building — no IO here (the caller writes the file).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from src.syslog_ingestion.findings import summarize_findings

if TYPE_CHECKING:  # avoid a runtime import cycle with artifacts
    from src.syslog_ingestion.artifacts import IngestRun


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _counts_table(title_headers: list[str], counter: dict[str, int],
                  limit: int | None = None) -> str:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit:
        items = items[:limit]
    return _table(title_headers, [[k, v] for k, v in items]) if items else "_None._"


def build_report(run: "IngestRun", summary: dict[str, Any],
                 feature_summary: dict[str, Any],
                 weak_label_summary: dict[str, Any]) -> str:
    """Assemble the full markdown report for a run."""
    events = run.events
    time_range = summary.get("time_range", {})
    finding_summary = summarize_findings(run.findings)

    lines: list[str] = []
    lines.append(f"# Industrial Switch Syslog Ingestion — `{run.run_id}`")
    lines.append("")
    lines.append("> Offline saved-log analysis. No device was contacted, "
                 "no packets captured, no command run.")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Parsed **{summary['parsed_events']}** events "
        f"(weighted **{summary['weighted_events']}**) from "
        f"**{len(run.input_files)}** file(s).")
    lines.append(f"- Hosts observed: {', '.join(summary['hosts']) or 'n/a'}.")
    lines.append(
        f"- Time range: {time_range.get('first') or 'n/a'} "
        f"→ {time_range.get('last') or 'n/a'}.")
    lines.append(
        f"- Dropped noise lines: {summary['dropped_lines']}; "
        f"duplicate lines collapsed: {summary['duplicate_lines_collapsed']}.")
    lines.append(
        f"- Engine C findings: **{finding_summary['total']}** "
        f"({finding_summary['by_severity']}).")
    lines.append("")

    # Input Files
    lines.append("## Input Files")
    lines.append("")
    for f in run.input_files:
        lines.append(f"- `{f}`")
    lines.append("")

    # Parsing Quality
    lines.append("## Parsing Quality")
    lines.append("")
    lines.append(_counts_table(["parse_status", "count"], summary["parse_status"]))
    lines.append("")
    lines.append(f"Clock-unreliable (boot-clock) events flagged and excluded "
                 f"from features by default: {summary['clock_unreliable_events']}.")
    lines.append("")

    # Timeline Coverage / Hosts
    lines.append("## Timeline Coverage")
    lines.append("")
    lines.append(f"First event: {time_range.get('first') or 'n/a'}  ")
    lines.append(f"Last event: {time_range.get('last') or 'n/a'}")
    lines.append("")
    lines.append("## Hosts Observed")
    lines.append("")
    host_counts = Counter(e.hostname for e in events if e.hostname)
    lines.append(_counts_table(["host", "events"], dict(host_counts)))
    lines.append("")

    # Severity / Facilities / Mnemonics
    lines.append("## Severity Distribution")
    lines.append("")
    lines.append(_counts_table(["severity", "count"],
                               summary["severity_distribution"]))
    lines.append("")
    lines.append("## Top Facilities / Mnemonics")
    lines.append("")
    lines.append("**Facilities**")
    lines.append("")
    lines.append(_counts_table(["facility", "count"], summary["top_facilities"]))
    lines.append("")
    lines.append("**Mnemonics**")
    lines.append("")
    lines.append(_counts_table(["code", "count"], summary["top_mnemonics"]))
    lines.append("")

    # Domain summaries
    lines.append("## Port Flap Summary")
    lines.append("")
    lines.append(_domain_line(events, "port_flap",
                              "port link-state changes (flaps)"))
    lines.append("")
    lines.append("## MAC Flap / Loop-Risk Summary")
    lines.append("")
    lines.append(_domain_line(events, "mac_flap",
                              "MAC-flap reports (possible L2 loop)"))
    lines.append("")
    lines.append("## SNMP / Auth Failure Summary")
    lines.append("")
    lines.append(_domain_line(events, "snmp_auth_failed",
                              "SNMP authorization failures"))
    lines.append("")
    lines.append("## PoE Fault Summary")
    lines.append("")
    lines.append(_domain_line(events, "poe_fault", "PoE fault events"))
    lines.append("")
    lines.append("## ERPS / Topology Summary")
    lines.append("")
    lines.append(_domain_line(events, "erps_churn", "ERPS ring state changes"))
    lines.append("")
    lines.append("## Device Health Events")
    lines.append("")
    lines.append(_domain_line(events, "device_health",
                              "device power/fan/reboot/clock events"))
    lines.append("")

    # Engine B / Engine C
    lines.append("## Engine B Feature Windows")
    lines.append("")
    lines.append(f"- Windows generated: {feature_summary.get('window_count', 0)} "
                 f"(scopes: {feature_summary.get('scopes', {})}).")
    lines.append(f"- Positive weak-label windows: "
                 f"{weak_label_summary.get('positive_windows', {})}.")
    lines.append("")
    lines.append("## Engine C Findings")
    lines.append("")
    if run.findings:
        rows = [[f.severity, f.rule_id, f.device or "-", f.interface or "-",
                 (f.title[:60])] for f in run.findings[:25]]
        lines.append(_table(["severity", "rule", "device", "interface", "title"],
                            rows))
    else:
        lines.append("_No findings crossed their thresholds._")
    lines.append("")

    # Safety / Limitations
    lines.append("## Safety Notes")
    lines.append("")
    lines.append("- Read-only, offline ingestion of saved log files.")
    lines.append("- No SSH/telnet/SNMP/live connection; no packet capture.")
    lines.append("- No remediation is planned or executed by this phase.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- No ground-truth attack labels exist in the source logs.")
    lines.append("- Weak labels are threshold-derived heuristics, not incidents.")
    lines.append("- Logs are offline saved files; state may be stale.")
    lines.append("- No live device contact, packet capture, or remediation.")
    lines.append("")
    return "\n".join(lines)


def _domain_line(events: list, tag: str, label: str) -> str:
    """One-line weighted count of events carrying ``tag``."""
    matched = [e for e in events if tag in e.tags]
    weighted = sum(max(1, int(e.duplicate_count)) for e in matched)
    hosts = sorted({e.hostname for e in matched if e.hostname})
    if not matched:
        return f"_No {label} observed._"
    return (f"- {weighted} {label} across {len(matched)} record(s) on "
            f"host(s): {', '.join(hosts)}.")
