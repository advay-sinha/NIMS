"""Dashboard section renderers (Streamlit).

Each ``render_*`` function draws one section from already-loaded artefact data.
Streamlit is imported at module load, so this module is imported only by the
running app. Views render data only — they never load, recompute or execute.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dashboard import components as ui
from src.dashboard import formatting as fmt


def render_executive_overview(summary: dict[str, Any]) -> None:
    ui.section_title("Executive Overview",
                     "Plain-language assessment of the selected runs.")
    level = summary.get("network_status_level", "stable")
    status = summary.get("network_status", "")
    label = fmt.STATUS_LEVEL_LABEL.get(level, "Status")
    banner = {"attention": st.error, "monitor": st.warning}.get(level, st.success)
    banner(f"**Network status — {label}.** {status}")

    if summary.get("clock_integrity_warning"):
        st.warning("Some device timestamps are unreliable. Event ordering and "
                   "time-based correlation may be approximate.")

    ui.metric_cards([
        ("Correlated incidents", summary.get("total_incidents", 0)),
        ("Critical / high", summary.get("critical_incident_count", 0)),
        ("With syslog evidence", summary.get("incidents_with_syslog_evidence", 0)),
        ("Affected devices", len(summary.get("affected_devices", []))),
    ], per_row=4)

    st.markdown("**Critical incidents**")
    crit = summary.get("critical_incidents", [])
    if crit:
        ui.table([{"severity": c.get("severity"), "rule": c.get("rule_id"),
                   "devices": ", ".join(c.get("devices", [])) or "-",
                   "title": c.get("title")} for c in crit])
    else:
        st.success("No high or critical incidents in the selected runs.")

    st.markdown("**Affected devices**")
    devices = summary.get("affected_devices", [])
    st.write(", ".join(devices) if devices else "None identified.")

    st.markdown("**Likely root causes**")
    causes = summary.get("likely_root_causes", [])
    if causes:
        for cause in causes:
            st.markdown(f"- {cause}")
    else:
        st.caption("No cross-engine root-cause hypotheses for the selected runs.")

    st.markdown("**Recommended actions**")
    actions = summary.get("recommended_actions", [])
    if actions:
        for action in actions:
            st.markdown(f"- **{action['title']}** ({action['owner']}): "
                        f"{action['detail']}")
        st.caption("All actions require explicit human confirmation and are "
                   "dry-run only — the dashboard executes nothing.")
    else:
        st.caption("No recommended actions — review the detailed tabs.")

    safety = summary.get("safety_status", {})
    st.markdown("**Safety status**")
    st.info(
        f"Offline analysis only · no command execution · no live device access · "
        f"dry-run actions executed: {safety.get('dry_run_executed', 0)}.")


def render_overview(overview: dict[str, Any]) -> None:
    ui.section_title("Overview", "Cross-engine snapshot of the current artefacts.")
    ui.metric_cards([
        ("Correlated incidents", overview["total_incidents"]),
        ("High/critical", overview["high_critical_incidents"]),
        ("Engine C findings", overview["engine_c_findings"]),
        ("Remediation planned", overview["remediation_actions_planned"]),
        ("Dry-run executed", overview["dry_run_executed_count"]),
        ("Engine A models", overview["engine_a_production_models"]),
    ])
    st.caption(f"Engine B: {overview['engine_b_anomaly_status']}")
    if overview["dry_run_executed_count"] == 0:
        st.success("Dry-run executed count is 0 — nothing was applied to a device.")


def render_incidents(correlation: dict[str, Any]) -> None:
    ui.section_title("Correlated Incidents",
                     "Unified incidents from the Phase 9 correlation engine.")
    if not correlation.get("available"):
        ui.missing_notice(correlation.get("message"))
        return
    incidents = correlation.get("incidents", [])
    if not incidents:
        st.caption("No incidents in this correlation run.")
        return

    if any(i.get("time_reliability", "reliable") != "reliable" for i in incidents):
        st.warning("Some device timestamps are unreliable. Event ordering and "
                   "time-based correlation may be approximate.")
    with_syslog = sum(1 for i in incidents if i.get("syslog_signal_count", 0))
    if with_syslog:
        st.caption(f"{with_syslog} incident(s) include syslog evidence. "
                   "No commands were executed.")

    severities = st.multiselect("Severity", fmt.unique_severities(incidents),
                                default=fmt.unique_severities(incidents))
    rules = st.multiselect("Rule / source", fmt.unique_rules(incidents),
                           default=fmt.unique_rules(incidents))
    filtered = fmt.filter_incidents(incidents, severities, rules)
    ui.table(fmt.incident_rows(filtered),
             empty_msg="No incidents match the current filters.")

    for inc in filtered:
        with st.expander(f"[{inc.get('severity')}] {inc.get('title')} "
                         f"({inc.get('incident_id')})"):
            st.markdown(f"**Rule:** `{inc.get('rule_id')}` | "
                        f"**Confidence:** {inc.get('confidence')} | "
                        f"**Engines:** {', '.join(inc.get('engines', []))}")
            st.markdown(f"**Root-cause hypothesis:** "
                        f"{inc.get('root_cause_hypothesis', 'n/a')}")
            if inc.get("syslog_signal_count"):
                st.markdown(
                    f"**Syslog evidence:** {inc.get('syslog_signal_count')} "
                    f"signal(s) | **Entity match:** "
                    f"{inc.get('entity_match_confidence', 'n/a')} | "
                    f"**Time reliability:** {inc.get('time_reliability', 'n/a')}")
            evidence = inc.get("evidence", [])
            if evidence:
                st.markdown("**Evidence by source:**")
                for bundle in evidence:
                    st.markdown(f"- {bundle.get('summary')}")
            notes = inc.get("evidence_quality_notes", [])
            if notes:
                st.markdown("**Evidence quality / alternatives:**")
                for note in notes:
                    st.markdown(f"- {note}")
            st.markdown("**Recommended operator actions:**")
            for action in inc.get("recommended_actions", []):
                st.markdown(
                    f"- **{action.get('title')}** ({action.get('owner')}): "
                    f"{action.get('detail')}")
            safety = inc.get("safety_notes", [])
            if safety:
                st.info(" ".join(safety))


def render_engine_c(engine_c: dict[str, Any]) -> None:
    ui.section_title("Engine C — Configuration Intelligence",
                     "Offline configuration findings and dry-run remediation.")
    if not engine_c.get("available"):
        ui.missing_notice(engine_c.get("message"))
        return
    ui.missing_notice(engine_c.get("message"))       # partial-missing notice
    views = engine_c.get("views", {})

    cards = (views.get("device_health_cards") or {}).get("cards", [])
    st.markdown("**Device health**")
    ui.table(cards, empty_msg="No device health cards.")

    summary = views.get("dashboard_summary") or {}
    st.markdown("**Findings**")
    col1, col2 = st.columns(2)
    with col1:
        ui.counts_table(summary.get("findings_by_severity", {}), "Severity")
    with col2:
        ui.counts_table(summary.get("findings_by_category", {}), "Category")

    findings = (views.get("findings_view") or {}).get("findings", [])
    ui.table([{k: f.get(k) for k in
               ("severity", "rule_id", "device", "interface", "risk_score", "title")}
              for f in findings], empty_msg="No findings.")

    remediation = views.get("remediation_view") or {}
    st.markdown("**Remediation (dry-run only)** — grouped by risk")
    grouped = remediation.get("grouped_by_risk", {})
    for risk, actions in grouped.items():
        st.markdown(f"_Risk: {risk}_")
        ui.table([{k: a.get(k) for k in
                   ("action_id", "action_type", "device", "interface", "status")}
                  for a in actions], empty_msg="None.")

    audit = views.get("action_audit_view") or {}
    executed = engine_c.get("dry_run_executed_count", 0)
    st.markdown(f"**Dry-run / audit status:** executed count = {executed} "
                f"(available: {audit.get('available', False)})")
    st.success("No commands were executed — Engine C is offline and read-only.")


def render_topology(engine_c: dict[str, Any]) -> None:
    ui.section_title("Topology",
                     "Discovered mesh from Engine C (offline, read-only).")
    view = (engine_c.get("views", {}) or {}).get("topology_view")
    if not view:
        st.caption("No topology view available for this snapshot.")
        return

    dot = fmt.topology_dot(view)
    if dot:
        st.markdown("**Network mesh** — nodes coloured by risk; red edges carry "
                    "warnings")
        st.graphviz_chart(dot, use_container_width=True)
    else:
        st.caption("No nodes to draw for this snapshot.")

    st.markdown("**Nodes**")
    ui.table(view.get("nodes", []), empty_msg="No nodes.")
    st.markdown("**Edges**")
    ui.table(view.get("edges", []), empty_msg="No edges.")
    st.markdown("**Warnings**")
    ui.table(view.get("warnings", []), empty_msg="No topology warnings.")


def render_engine_b(engine_b: dict[str, Any]) -> None:
    ui.section_title("Engine B — Network Health",
                     "Latest network-health experiment metrics (read-only).")
    if not engine_b.get("available"):
        ui.missing_notice(engine_b.get("message"))
        return
    for ds in engine_b.get("datasets", []):
        st.markdown(f"**{ds['dataset']}** — model `{ds.get('model_name')}` "
                    f"(labeled: {ds.get('labeled')})")
        ui.metric_cards([
            ("Anomaly rate", fmt.fmt_pct(ds.get("anomaly_rate"))),
            ("Precision", fmt.fmt_metric(ds.get("precision"))),
            ("Recall", fmt.fmt_metric(ds.get("recall"))),
            ("F1", fmt.fmt_metric(ds.get("f1"))),
            ("ROC-AUC", fmt.fmt_metric(ds.get("roc_auc"))),
        ], per_row=5)
        st.caption(f"Test samples: {ds.get('n_samples')} | flagged anomalous: "
                   f"{ds.get('n_anomalous_predicted')}")
    if engine_b.get("report_path"):
        st.caption(f"Report: `{engine_b['report_path']}`")


def render_syslog(syslog: dict[str, Any]) -> None:
    """Read-only view of one industrial-switch syslog ingestion run."""
    ui.section_title("Industrial Syslog",
                     "Offline ingestion of saved switch logs (read-only).")
    if not syslog.get("available"):
        ui.missing_notice(syslog.get("message"))
        return

    summary = syslog.get("summary", {})
    time_range = summary.get("time_range", {}) or {}
    ui.metric_cards([
        ("Parsed events", summary.get("parsed_events", 0)),
        ("Weighted", summary.get("weighted_events", 0)),
        ("Dropped noise", summary.get("dropped_lines", 0)),
        ("Duplicates collapsed", summary.get("duplicate_lines_collapsed", 0)),
        ("Hosts", len(summary.get("hosts", []))),
    ], per_row=5)
    st.caption(f"Time range: {time_range.get('first') or 'n/a'} → "
               f"{time_range.get('last') or 'n/a'} | hosts: "
               f"{', '.join(summary.get('hosts', [])) or 'n/a'}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Events by severity**")
        ui.counts_table(summary.get("severity_distribution", {}), "Severity")
    with col2:
        st.markdown("**Top mnemonics**")
        ui.counts_table(summary.get("top_mnemonics", {}), "Code")

    st.markdown("**Top facilities**")
    ui.counts_table(summary.get("top_facilities", {}), "Facility")

    weak = (syslog.get("weak_label_summary", {}) or {}).get("positive_windows", {})
    st.markdown("**Engine B weak-label windows** (threshold heuristics)")
    ui.counts_table(weak, "Weak label")

    st.markdown("**Engine C findings**")
    findings = syslog.get("findings", [])
    ui.table([{k: f.get(k) for k in
               ("severity", "rule_id", "device", "interface", "title")}
              for f in findings], empty_msg="No syslog findings.")

    if syslog.get("windows_csv_path"):
        st.caption(f"Engine B windows: `{syslog['windows_csv_path']}`")
    if syslog.get("report_path"):
        st.caption(f"Report: `{syslog['report_path']}`")


def render_engine_a(engine_a: dict[str, Any]) -> None:
    ui.section_title("Engine A — Cybersecurity",
                     "Promoted intrusion-detection models (read-only).")
    if not engine_a.get("available"):
        ui.missing_notice(engine_a.get("message"))
        return
    rows = [{
        "dataset": m["dataset"], "model": m.get("model_type"),
        "test_f1": fmt.fmt_metric(m.get("test_f1")),
        "roc_auc": fmt.fmt_metric(m.get("roc_auc")),
        "experiment_id": m.get("experiment_id"),
    } for m in engine_a.get("models", [])]
    ui.table(rows, empty_msg="No production models.")
    if engine_a.get("validation_report_path"):
        st.caption(f"Validation report: `{engine_a['validation_report_path']}`")
    if engine_a.get("latest_error_analysis"):
        st.caption(f"Latest error analysis: `{engine_a['latest_error_analysis']}`")
    if engine_a.get("latest_visualization"):
        st.caption(f"Latest visualization: `{engine_a['latest_visualization']}`")


def render_live_monitor(current_state: dict[str, Any]) -> None:
    ui.section_title("Live Monitor",
                     "Near-real-time demo replay of persisted artefacts.")
    from src.streaming import formatting as sfmt

    st.info(sfmt.STREAM_SAFETY_BANNER)
    if not current_state.get("available"):
        ui.missing_notice(current_state.get("message"))
        return
    ui.metric_cards([
        ("Events", current_state.get("total_events", 0)),
        ("Active incidents", current_state.get("active_incident_count", 0)),
        ("Critical", current_state.get("critical_incident_count", 0)),
        ("Active devices", current_state.get("active_device_count", 0)),
    ])
    st.caption(f"Last event: {current_state.get('last_event_at', 'n/a')}")

    st.markdown("**Events by type**")
    ui.counts_table(current_state.get("events_by_type", {}), "Type")
    st.markdown("**Active incidents**")
    ui.table(current_state.get("active_incidents", []),
             empty_msg="No active incidents.")
    st.markdown("**Recent events**")
    ui.table(sfmt.event_rows(current_state.get("recent_events", [])),
             empty_msg="No events yet.")
    st.caption("Click **Rerun** to refresh after re-running the streaming demo.")


def render_ml_workflow_console() -> None:
    ui.section_title("Offline ML Workflow",
                     "Build an offline Engine A workflow command (read-only).")
    from src.ml_workflow import planner

    st.info("Offline Engine A datasets and models only — no live devices, "
            "traffic, SNMP, SSH, packet capture or remediation. The dashboard "
            "**builds the command for you to run in a terminal**; it does not "
            "execute anything.")
    datasets = st.multiselect("Datasets", list(planner.DATASETS),
                              default=["unsw_nb15"])
    models = st.multiselect("Models", list(planner.MODELS), default=["xgboost"])
    steps = st.multiselect("Steps", list(planner.STEP_ORDER),
                           default=["validate", "preprocess", "features", "train"])
    try:
        plan = planner.build_plan(datasets or ["all"], models or ["all"],
                                  steps or ["all"])
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.markdown(f"**Planned steps ({len(plan)})** — run these in a terminal:")
    st.code("\n".join(step.display for step in plan), language="bash")
    st.caption("After running, pick the new Assessment / Incident Run in the "
               "sidebar to load the freshly generated artefacts.")


def render_safety() -> None:
    ui.section_title("Safety / Audit", "The dashboard is a viewer, not an actuator.")
    for statement in fmt.SAFETY_STATEMENTS:
        st.markdown(f"- {statement}")
    st.markdown("**Verify the Engine C safety posture:**")
    st.code(fmt.SAFETY_VALIDATOR_COMMAND, language="bash")
