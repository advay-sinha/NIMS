"""Operator-facing correlation report (Markdown).

Pure formatting of an already-computed
:class:`~src.correlation.engine.CorrelationResult`. Nothing here recomputes
state, reads an artefact, contacts a device or executes a command. The safety
section always states, explicitly, that correlation is offline/artefact-driven
and that no commands were executed.
"""

from __future__ import annotations

from src.correlation.engine import CorrelationResult
from src.correlation.models import (
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    CorrelatedIncident,
)
from src.correlation.rules import SINGLE_ENGINE_HIGH_RISK

_SAFETY_LINES = [
    "- Correlation is **offline and artefact-driven** — no packets were "
    "captured and no SNMP was polled.",
    "- **No commands were executed** and no device was contacted.",
    "- Remediation remains **human-confirmed and dry-run only** (Engine C never "
    "executes against a device).",
    "- Cross-engine wording is deliberately cautious "
    "(`possible`/`candidate`/`likely`); aggregate signals are marked and "
    "down-weighted.",
]


def build_report(result: CorrelationResult) -> str:
    """Render the full correlation report as a Markdown string."""
    s = result.summary
    lines = [
        f"# Correlation Report — {s.correlation_id}",
        "",
        f"> **No commands were executed.** {s.safety_note}",
        "",
        f"- Generated: {s.timestamp}",
        "",
    ]
    lines += _executive_summary(result)
    lines += _inputs_section(result)
    lines += _signal_summary(result)
    lines += _incidents_section(result)
    lines += _multi_engine_section(result)
    lines += _single_engine_section(result)
    lines += _hypotheses_section(result)
    lines += _actions_section(result)
    lines += _safety_section()
    lines += _appendix(result)
    lines.append("")
    return "\n".join(lines)


def _executive_summary(result: CorrelationResult) -> list[str]:
    s = result.summary
    top = result.incidents[0] if result.incidents else None
    return [
        "## Executive Summary", "",
        f"- Signals ingested: **{s.total_signals}** "
        f"({s.aggregate_signal_count} aggregate).",
        f"- Correlated incidents: **{s.total_incidents}** "
        f"({s.multi_engine_incident_count} multi-engine).",
        f"- Highest-severity incident: "
        f"**{top.severity if top else 'n/a'}**"
        + (f" — {top.title}" if top else "") + ".",
        f"- Rules that fired: "
        f"{', '.join(sorted(s.incidents_by_rule)) or 'none'}.",
        "",
    ]


def _inputs_section(result: CorrelationResult) -> list[str]:
    s = result.summary
    return [
        "## Inputs Used", "",
        "| Engine | Source |", "|---|---|",
        f"| Engine A (cyber) | {s.engine_a_source or 'not provided'} |",
        f"| Engine B (health) | {s.engine_b_source or 'not provided'} |",
        f"| Engine C (config) | {s.engine_c_source or 'not provided'} |",
        "",
    ]


def _signal_summary(result: CorrelationResult) -> list[str]:
    s = result.summary
    lines = ["## Signal Summary", "",
             "| Engine | Signals |", "|---|---|"]
    for engine, label in ((ENGINE_A, "Engine A"), (ENGINE_B, "Engine B"),
                          (ENGINE_C, "Engine C")):
        lines.append(f"| {label} | {s.signals_by_engine.get(engine, 0)} |")
    lines.append("")
    return lines


def _incidents_section(result: CorrelationResult) -> list[str]:
    lines = ["## Correlated Incidents", ""]
    if not result.incidents:
        return lines + ["_No incidents were generated._", ""]
    lines += ["| Severity | Conf. | Rule | Engines | Devices | Title |",
              "|---|---|---|---|---|---|"]
    for inc in result.incidents:
        lines.append(
            f"| {inc.severity} | {inc.confidence:.2f} | {inc.rule_id} | "
            f"{'+'.join(e.split('_')[-1] for e in inc.engines)} | "
            f"{', '.join(inc.affected_devices) or 'n/a'} | {inc.title} |")
    lines.append("")
    return lines


def _multi_engine_section(result: CorrelationResult) -> list[str]:
    multi = [i for i in result.incidents if i.multi_engine]
    lines = ["## Multi-Engine Incidents", ""]
    if not multi:
        return lines + ["_No multi-engine correlations were found._", ""]
    for inc in multi:
        lines += _incident_detail(inc)
    return lines


def _single_engine_section(result: CorrelationResult) -> list[str]:
    single = [i for i in result.incidents
              if i.rule_id == SINGLE_ENGINE_HIGH_RISK]
    lines = ["## Single-Engine High-Risk Items", ""]
    if not single:
        return lines + ["_No uncorrelated high-risk items._", ""]
    lines += ["| Severity | Engine | Device | Interface | Title |",
              "|---|---|---|---|---|"]
    for inc in single:
        engine = inc.engines[0] if inc.engines else "n/a"
        lines.append(
            f"| {inc.severity} | {engine} | "
            f"{', '.join(inc.affected_devices) or 'n/a'} | "
            f"{', '.join(inc.affected_interfaces) or '-'} | {inc.title} |")
    lines.append("")
    return lines


def _hypotheses_section(result: CorrelationResult) -> list[str]:
    lines = ["## Root-Cause Hypotheses", ""]
    correlated = [i for i in result.incidents
                  if i.rule_id != SINGLE_ENGINE_HIGH_RISK]
    if not correlated:
        return lines + ["_No cross-engine root-cause hypotheses._", ""]
    for inc in correlated:
        lines += [f"### {inc.title} ({inc.severity}, conf {inc.confidence:.2f})",
                  "", f"- {inc.root_cause_hypothesis}",
                  f"- Devices: {', '.join(inc.affected_devices) or 'n/a'}",
                  f"- Interfaces: {', '.join(inc.affected_interfaces) or 'n/a'}",
                  ""]
    return lines


def _actions_section(result: CorrelationResult) -> list[str]:
    lines = ["## Recommended Operator Actions", ""]
    seen: set[tuple[str, str]] = set()
    rows: list[str] = []
    for inc in result.incidents:
        for action in inc.recommended_actions:
            key = (inc.incident_id, action.title)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                f"| {inc.severity} | {action.owner} | {action.title} | "
                f"{action.detail} |")
    if not rows:
        return lines + ["_No recommended actions._", ""]
    lines += ["| Severity | Owner | Action | Detail |", "|---|---|---|---|"]
    lines += rows
    lines += ["",
              "_Every action requires explicit human confirmation and is "
              "dry-run only; nothing is executed automatically._", ""]
    return lines


def _incident_detail(inc: CorrelatedIncident) -> list[str]:
    lines = [
        f"### {inc.title}", "",
        f"- Incident: `{inc.incident_id}` | Rule: `{inc.rule_id}` | "
        f"Severity: **{inc.severity}** | Confidence: {inc.confidence:.2f}",
        f"- Engines: {', '.join(inc.engines)}",
        f"- Root cause: {inc.root_cause_hypothesis}",
        "- Evidence:",
    ]
    lines += [f"  - {e.summary}" for e in inc.evidence]
    if inc.scoring_factors:
        lines.append(f"- Scoring: {', '.join(inc.scoring_factors)}")
    lines.append("")
    return lines


def _safety_section() -> list[str]:
    return ["## Safety Notes", ""] + _SAFETY_LINES + [""]


def _appendix(result: CorrelationResult) -> list[str]:
    lines = ["## Artifact Appendix", "",
             "Correlation-run artefacts (this run):", "",
             "- `signals.json` / `signals.csv`",
             "- `incidents.json` / `incidents.csv`",
             "- `correlation_summary.json`",
             "- `correlation_report.md`", "",
             "Source artefacts consumed (read-only):", ""]
    sources = sorted({s.source_artifact for s in result.signals})
    lines += [f"- `{src}`" for src in sources] or ["- _none_"]
    lines.append("")
    return lines
