"""Streamlit entry point for the NIMS monitoring dashboard prototype.

Run with either::

    python -m scripts.run_dashboard
    streamlit run src/dashboard/app.py

The app is a **read-only viewer** over persisted artefacts. It never runs an
engine pipeline, trains, infers, polls SNMP, captures packets, contacts a device
or executes a command; every page shows the offline/no-execution banner.

Streamlit is imported inside :func:`main`, and the module only renders when
executed as a script, so importing it never fails in a Streamlit-free
environment (the loader/formatting logic lives in Streamlit-free modules).
"""

from __future__ import annotations

import sys
from pathlib import Path

# When launched via ``streamlit run src/dashboard/app.py`` the repo root is not
# on sys.path (only the script dir is). Add it so ``import src.*`` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    """Render the dashboard (imports Streamlit lazily)."""
    import streamlit as st

    from src.dashboard import formatting as fmt
    from src.dashboard import loader, views
    from src.utils.config import CONFIG_DIR, load_config
    from src.utils.paths import Paths

    st.set_page_config(page_title="NIMS Monitoring Dashboard", layout="wide")

    paths = Paths.from_config(load_config())
    dash_cfg = loader.load_dashboard_config(CONFIG_DIR / "dashboard.yaml")
    defaults = dash_cfg.get("dashboard", {}) or {}
    sections = dash_cfg.get("sections", {}) or {}

    st.title("NIMS / NetSentinel — Monitoring Dashboard")
    st.caption("Offline assessment view — artifact-driven, read-only.")
    views.ui.offline_banner()

    # ---- friendly run selectors (raw ids hidden behind friendly labels) ----
    assessments = loader.labeled_snapshots(paths.network_config_dir)
    incident_runs = loader.labeled_correlation_runs(paths.correlation_dir)
    snapshot_id = _select_run(
        st, "Assessment Run", assessments,
        loader.resolve_default(assessments,
                               defaults.get("default_engine_c_snapshot")),
        fmt.EMPTY_NO_ASSESSMENT)
    correlation_id = _select_run(
        st, "Incident Run", incident_runs,
        loader.resolve_default(incident_runs,
                               defaults.get("default_correlation_id")),
        fmt.EMPTY_NO_INCIDENT_RUN)

    # ---- read-only loads ----
    engine_c = loader.load_engine_c_dashboard(paths.network_config_dir,
                                              snapshot_id) if snapshot_id else \
        {"available": False, "snapshot_id": None, "views": {},
         "dry_run_executed_count": 0, "message": fmt.EMPTY_NO_ASSESSMENT}
    correlation = loader.load_correlation(
        paths.correlation_dir, correlation_id,
        snapshot_hint=snapshot_id or "sample_remediation") if correlation_id else \
        {"available": False, "correlation_id": None, "incidents": [],
         "signals": [], "summary": {}, "message": fmt.EMPTY_NO_INCIDENT_RUN}
    engine_a = loader.load_engine_a(paths.registry_dir, paths.reports_dir,
                                    paths.error_analysis_dir,
                                    paths.visualizations_dir, paths.experiments_dir)
    engine_b = loader.load_engine_b(paths.network_health_dir)
    syslog_dir = paths.outputs_dir / "syslog_ingestion"
    syslog_run = defaults.get("default_syslog_run") or \
        loader.latest_syslog_run(syslog_dir)
    syslog = loader.load_syslog_run(syslog_dir, syslog_run) if syslog_run else \
        {"available": False, "run_id": None,
         "message": "No industrial syslog run found. Run: "
                    "python -m scripts.ingest_switch_syslog "
                    "--input-dir datasets/raw/syslog --run-id "
                    "lw_terminal_syslog_sample"}
    overview = loader.compute_overview(engine_c, correlation, engine_a, engine_b)
    executive = loader.build_executive_summary(engine_c, correlation, engine_b,
                                               engine_a)

    from src.streaming.artifacts import load_current_state
    current_state = load_current_state(paths.root / "outputs" / "streaming"
                                       / "current")

    # ---- advanced (raw) artifact sources, tucked away ----
    _render_advanced_sources(st, loader.describe_artifact_sources(
        engine_c, correlation,
        {"network_config_dir": paths.network_config_dir,
         "correlation_dir": paths.correlation_dir}))

    # ---- enabled sections as tabs (Executive Overview first) ----
    tab_specs = [
        ("Executive Overview", "executive",
         lambda: views.render_executive_overview(executive)),
        ("Live Monitor", "live_monitor",
         lambda: views.render_live_monitor(current_state)),
        ("Incidents", "incidents", lambda: views.render_incidents(correlation)),
        ("Metrics", "overview", lambda: views.render_overview(overview)),
        ("Engine A", "engine_a", lambda: views.render_engine_a(engine_a)),
        ("Engine B", "engine_b", lambda: views.render_engine_b(engine_b)),
        ("Industrial Syslog", "syslog", lambda: views.render_syslog(syslog)),
        ("Engine C", "engine_c", lambda: views.render_engine_c(engine_c)),
        ("Topology", "topology", lambda: views.render_topology(engine_c)),
        ("ML Workflow", "ml_workflow", views.render_ml_workflow_console),
        ("Safety", "safety", views.render_safety),
    ]
    active = [(label, fn) for label, key, fn in tab_specs
              if sections.get(key, True)]
    tabs = st.tabs([label for label, _ in active])
    for tab, (_, render) in zip(tabs, active):
        with tab:
            views.ui.offline_banner()
            render()


def _select_run(st, label: str, items: list[dict], default_id: str | None,
                empty_guidance: str) -> str | None:
    """Sidebar run selector showing friendly labels; raw ids stay hidden."""
    if not items:
        st.sidebar.info(f"No {label.lower()} available yet.")
        st.sidebar.caption(empty_guidance)
        return default_id
    ids = [i["id"] for i in items]
    labels = {i["id"]: i["label"] for i in items}
    index = ids.index(default_id) if default_id in ids else 0
    return st.sidebar.selectbox(label, ids, index=index,
                                format_func=lambda i: labels.get(i, i))


def _render_advanced_sources(st, sources: dict) -> None:
    """Raw ids and artefact paths, hidden inside an expander for power users."""
    with st.sidebar.expander("Advanced Artifact Sources"):
        st.caption("Raw artefact identifiers and paths (read-only).")
        st.write({
            "Assessment run id": sources.get("assessment_run_id"),
            "Incident run id": sources.get("incident_run_id"),
            "Engine C dashboard dir": sources.get("engine_c_dashboard_dir"),
            "Correlation report": sources.get("correlation_report_path"),
            "network_config_dir": sources.get("network_config_dir"),
            "correlation_dir": sources.get("correlation_dir"),
        })


if __name__ == "__main__":
    main()
