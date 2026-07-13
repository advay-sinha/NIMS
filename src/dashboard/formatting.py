"""Presentation helpers for the monitoring dashboard (pure Python).

No Streamlit dependency: this module only reshapes already-loaded artefact data
into simple, display-friendly structures (rows, counts, filters, banner text).
It performs no IO and mutates nothing.
"""

from __future__ import annotations

from typing import Any, Optional

SEVERITY_ORDER: dict[str, int] = {
    "critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# Shown on every page — the dashboard is a viewer, never an actuator.
OFFLINE_BANNER = (
    "Offline artifact analysis only — no commands were executed, no device was "
    "contacted, and no remediation was applied.")

SAFETY_STATEMENTS: tuple[str, ...] = (
    "Offline artifact analysis only; the dashboard reads persisted files.",
    "Remediation is dry-run only and requires explicit human confirmation.",
    "No live device access (no SSH, SNMP polling or packet capture).",
    "No command execution — the dashboard never actuates anything.",
    "Batfish validation is optional and disabled by default.",
    "Remediation remains plan-only / dry-run-only.",
)

SAFETY_VALIDATOR_COMMAND = "python -m scripts.validate_engine_c_safety"

# Plain-language empty-state guidance shown when no runs are available yet.
EMPTY_NO_ASSESSMENT = (
    "No assessment runs are available yet. An assessment run is produced when "
    "Engine C analyses saved device data. To create one, run:\n\n"
    "    python -m scripts.export_network_config_dashboard "
    "--snapshot-id sample_remediation")
EMPTY_NO_INCIDENT_RUN = (
    "No incident runs are available yet. An incident run correlates the "
    "cyber, network-health and configuration findings into unified incidents. "
    "To create one, run:\n\n"
    "    python -m scripts.run_correlation --engine-c-snapshot sample_remediation "
    "--engine-b-dataset synthetic --engine-a-dataset unsw_nb15 "
    "--correlation-id sample_correlation")

# Human-friendly wording for the network-status banner colour.
STATUS_LEVEL_LABEL: dict[str, str] = {
    "attention": "Attention required",
    "monitor": "Monitoring",
    "stable": "Stable",
}


def sort_by_severity(items: list[dict[str, Any]], key: str = "severity"
                     ) -> list[dict[str, Any]]:
    """Return ``items`` ordered most-severe first (stable)."""
    return sorted(items, key=lambda i: SEVERITY_ORDER.get(
        str(i.get(key, "info")).lower(), 9))


def filter_incidents(
    incidents: list[dict[str, Any]],
    severities: Optional[list[str]] = None,
    rules: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Filter incidents by severity and/or rule id (empty filter = keep all)."""
    out = incidents
    if severities:
        wanted = {s.lower() for s in severities}
        out = [i for i in out if str(i.get("severity", "")).lower() in wanted]
    if rules:
        rset = set(rules)
        out = [i for i in out if i.get("rule_id") in rset]
    return sort_by_severity(list(out))


def incident_rows(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten incidents into table rows (nested fields collapsed to strings)."""
    rows: list[dict[str, Any]] = []
    for inc in sort_by_severity(incidents):
        rows.append({
            "incident_id": inc.get("incident_id"),
            "severity": inc.get("severity"),
            "confidence": inc.get("confidence"),
            "rule": inc.get("rule_id"),
            "engines": ", ".join(inc.get("engines", [])),
            "devices": ", ".join(inc.get("affected_devices", [])) or "-",
            "title": inc.get("title"),
        })
    return rows


def signal_rows(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten signals into table rows."""
    return [{
        "signal_id": s.get("signal_id"),
        "engine": s.get("engine"),
        "severity": s.get("severity"),
        "category": s.get("category"),
        "device": s.get("device") or "-",
        "interface": s.get("interface") or "-",
        "aggregate": s.get("aggregate"),
        "title": s.get("title"),
    } for s in signals]


def unique_severities(incidents: list[dict[str, Any]]) -> list[str]:
    """Distinct severities present, ordered most-severe first."""
    present = {str(i.get("severity", "info")).lower() for i in incidents}
    return [s for s in ("critical", "high", "medium", "low", "info")
            if s in present]


def unique_rules(incidents: list[dict[str, Any]]) -> list[str]:
    """Distinct rule ids present, sorted."""
    return sorted({str(i.get("rule_id", "")) for i in incidents if i.get("rule_id")})


def count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count ``items`` grouped by a string field."""
    out: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return out


def fmt_metric(value: Any, digits: int = 3) -> str:
    """Format a numeric metric, or ``n/a`` when absent/non-numeric."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Format a 0-1 ratio as a percentage string."""
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


# ------------------------------------------------------------------ topology

# Risk-score -> node fill colour (mirrors the Engine C risk levels).
_RISK_FILL: tuple[tuple[int, str], ...] = (
    (65, "#e06666"),   # high/critical
    (40, "#f6b26b"),   # medium
    (1, "#ffd966"),    # low
)
_NODE_DEFAULT_FILL = "#d9d9d9"


def _dot_escape(text: Any) -> str:
    """Escape a value for use inside a DOT double-quoted string."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _risk_fill(risk_score: Any) -> str:
    try:
        score = int(risk_score or 0)
    except (TypeError, ValueError):
        score = 0
    for lower, colour in _RISK_FILL:
        if score >= lower:
            return colour
    return _NODE_DEFAULT_FILL


def topology_dot(view: dict[str, Any]) -> str:
    """Build an undirected Graphviz DOT string for the topology mesh.

    Nodes are coloured by risk score and edges are labelled with the discovery
    protocol (warning-bearing edges are drawn red). Rendered client-side by
    ``st.graphviz_chart`` — no extra Python dependency. Returns ``""`` when there
    are no nodes to draw.
    """
    nodes = view.get("nodes") or []
    edges = view.get("edges") or []
    if not nodes:
        return ""

    lines = ["graph topology {",
             "  graph [layout=neato, overlap=false, splines=true, "
             "bgcolor=transparent];",
             '  node [shape=box, style="rounded,filled", fontsize=10];',
             "  edge [fontsize=9];"]
    for node in nodes:
        node_id = _dot_escape(node.get("id", "unknown"))
        findings = int(node.get("finding_count", 0) or 0)
        label = node.get("label", node.get("id", "unknown"))
        suffix = f"\\n({findings} finding(s))" if findings else ""
        lines.append(
            f'  "{node_id}" [label="{_dot_escape(label)}{suffix}", '
            f'fillcolor="{_risk_fill(node.get("risk_score"))}"];')
    for edge in edges:
        source = _dot_escape(edge.get("source", "unknown"))
        target = _dot_escape(edge.get("target", "unknown"))
        protocol = _dot_escape(edge.get("protocol", ""))
        colour = "#cc0000" if int(edge.get("warning_count", 0) or 0) else "#888888"
        lines.append(
            f'  "{source}" -- "{target}" [label="{protocol}", '
            f'color="{colour}"];')
    lines.append("}")
    return "\n".join(lines)
