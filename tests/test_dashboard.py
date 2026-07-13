"""Tests for the Phase 10 monitoring dashboard (loader/formatting/launcher).

These tests exercise the Streamlit-free logic only. They never import
``src.dashboard.app``/``views``/``components`` and never require Streamlit to be
installed, matching the constraint that pytest passes without Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard import formatting as fmt
from src.dashboard import loader


def _write(directory: Path, name: str, payload) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), "utf-8")


def _build_engine_c_dashboard(nc_dir: Path, snapshot: str, *, findings=3,
                              executed=0) -> None:
    dash = nc_dir / snapshot / "dashboard"
    _write(dash, "dashboard_summary.json", {
        "snapshot_id": snapshot, "device_count": 1, "interface_count": 7,
        "finding_count": findings, "findings_by_severity": {"medium": 1, "low": 2},
        "findings_by_category": {"port": 2, "stp": 1},
        "remediation_action_count": 3, "command_action_count": 2})
    _write(dash, "findings_view.json", {"findings": [
        {"severity": "medium", "rule_id": "STP", "device": "sw1",
         "interface": "Gi0/3", "risk_score": 55, "title": "stp"}]})
    _write(dash, "topology_view.json",
           {"nodes": [{"id": "sw1"}], "edges": [], "warnings": []})
    _write(dash, "remediation_view.json", {"grouped_by_risk": {"medium": []}})
    _write(dash, "device_health_cards.json",
           {"cards": [{"device_id": "sw1", "status": "warning"}]})
    _write(dash, "action_audit_view.json",
           {"available": True, "executed_count": executed})
    for name in ("inventory_view", "risk_timeline", "export_metadata"):
        _write(dash, f"{name}.json", {"ok": True})


# ------------------------------------------------------------- Engine C loader


def test_loader_reads_engine_c_dashboard(tmp_path: Path):
    nc = tmp_path / "network_config"
    _build_engine_c_dashboard(nc, "snap")
    data = loader.load_engine_c_dashboard(nc, "snap")
    assert data["available"] is True
    assert data["views"]["dashboard_summary"]["finding_count"] == 3
    assert "topology_view" in data["views"]
    assert data["missing"] == []
    assert data["dry_run_executed_count"] == 0


def test_loader_handles_missing_engine_c_snapshot(tmp_path: Path):
    data = loader.load_engine_c_dashboard(tmp_path / "network_config", "nope")
    assert data["available"] is False
    assert "export_network_config_dashboard" in data["message"]
    assert data["views"] == {}


def test_loader_partial_engine_c_views_do_not_crash(tmp_path: Path):
    nc = tmp_path / "network_config"
    dash = nc / "snap" / "dashboard"
    _write(dash, "dashboard_summary.json", {"finding_count": 1})
    data = loader.load_engine_c_dashboard(nc, "snap")
    assert data["available"] is True                 # some views present
    assert "findings_view" in data["missing"]
    assert data["message"] and "export_network_config_dashboard" in data["message"]


# ------------------------------------------------------------- correlation


def test_loader_reads_correlation(tmp_path: Path):
    run = tmp_path / "correlation" / "run1"
    _write(run, "correlation_summary.json",
           {"correlation_id": "run1", "total_incidents": 2})
    _write(run, "incidents.json",
           [{"incident_id": "INC1", "severity": "high", "rule_id": "DOS"}])
    _write(run, "signals.json", [{"signal_id": "SIG1", "engine": "engine_a"}])
    (run / "correlation_report.md").write_text("# report", "utf-8")
    data = loader.load_correlation(tmp_path / "correlation", "run1")
    assert data["available"] is True
    assert data["summary"]["total_incidents"] == 2
    assert data["incidents"][0]["incident_id"] == "INC1"
    assert data["report_path"] is not None


def test_loader_handles_missing_correlation(tmp_path: Path):
    data = loader.load_correlation(tmp_path / "correlation", "missing",
                                   snapshot_hint="snap")
    assert data["available"] is False
    assert "run_correlation" in data["message"]
    assert "--correlation-id missing" in data["message"]
    assert "snap" in data["message"]
    assert data["incidents"] == []


# ------------------------------------------------------------- Engine A / B


def test_registry_loader_reads_production_models(tmp_path: Path):
    reg = tmp_path / "registry"
    _write(reg, "production.json", {
        "nsl_kdd": {"experiment_id": "nsl_x", "model_type": "xgboost"},
        "unsw_nb15": {"experiment_id": "unsw_x", "model_type": "xgboost"}})
    _write(reg, "best_per_dataset.json", {
        "nsl_kdd": {"value": 0.9925}, "unsw_nb15": {"value": 0.9244}})
    exp = tmp_path / "experiments"
    _write(exp / "nsl_kdd" / "xgboost" / "nsl_x", "metrics.json",
           {"test": {"roc_auc": 0.999}})
    data = loader.load_engine_a(reg, tmp_path / "reports",
                                tmp_path / "error_analysis",
                                tmp_path / "visualizations", exp)
    assert data["available"] is True
    assert data["production_model_count"] == 2
    nsl = next(m for m in data["models"] if m["dataset"] == "nsl_kdd")
    assert nsl["test_f1"] == 0.9925
    assert nsl["roc_auc"] == 0.999


def test_engine_a_loader_missing_registry(tmp_path: Path):
    data = loader.load_engine_a(tmp_path / "registry", tmp_path / "reports",
                                tmp_path / "ea", tmp_path / "viz")
    assert data["available"] is False
    assert data["production_model_count"] == 0


def test_engine_b_loader_reads_latest_experiment(tmp_path: Path):
    nh = tmp_path / "network_health"
    run = nh / "experiments" / "synthetic" / "isolation_forest" / "run_2"
    _write(run, "metrics.json", {"test": {"n_samples": 100,
           "n_anomalous_predicted": 30, "precision": 0.7, "recall": 1.0,
           "f1": 0.82, "roc_auc": 0.97}})
    _write(run, "manifest.json", {"experiment_id": "run_2",
           "model_name": "isolation_forest", "labeled": True})
    data = loader.load_engine_b(nh)
    assert data["available"] is True
    ds = data["datasets"][0]
    assert ds["anomaly_rate"] == 0.3
    assert "synthetic" in data["anomaly_status"]


def test_engine_b_loader_missing(tmp_path: Path):
    data = loader.load_engine_b(tmp_path / "network_health")
    assert data["available"] is False
    assert data["datasets"] == []


# ------------------------------------------------------------- source listing


def test_source_listing_ignores_non_snapshot_dirs(tmp_path: Path):
    nc = tmp_path / "network_config"
    _build_engine_c_dashboard(nc, "sample_remediation")
    _write(nc / "sample_offline", "inventory.json", {"devices": []})
    (nc / "diffs" / "a__to__b").mkdir(parents=True)      # must be ignored
    snaps = loader.list_engine_c_snapshots(nc)
    assert "diffs" not in snaps
    assert set(snaps) == {"sample_remediation", "sample_offline"}


def test_correlation_run_listing(tmp_path: Path):
    corr = tmp_path / "correlation"
    _write(corr / "run1", "incidents.json", [])
    (corr / "empty").mkdir(parents=True)                 # no artefacts -> ignored
    runs = loader.list_correlation_runs(corr)
    assert runs == ["run1"]


def test_listings_tolerate_absent_root(tmp_path: Path):
    assert loader.list_engine_c_snapshots(tmp_path / "nope") == []
    assert loader.list_correlation_runs(tmp_path / "nope") == []


# ------------------------------------------------------------- overview


def test_overview_metrics_computed_correctly(tmp_path: Path):
    engine_c = {"views": {"dashboard_summary": {"finding_count": 3,
                "remediation_action_count": 2}}, "dry_run_executed_count": 0}
    correlation = {"incidents": [
        {"severity": "high"}, {"severity": "critical"}, {"severity": "low"}]}
    engine_a = {"production_model_count": 3}
    engine_b = {"anomaly_status": "synthetic: 30% anomalous"}
    overview = loader.compute_overview(engine_c, correlation, engine_a, engine_b)
    assert overview["total_incidents"] == 3
    assert overview["high_critical_incidents"] == 2
    assert overview["engine_c_findings"] == 3
    assert overview["remediation_actions_planned"] == 2
    assert overview["dry_run_executed_count"] == 0
    assert overview["engine_a_production_models"] == 3
    assert "synthetic" in overview["engine_b_anomaly_status"]


def test_overview_handles_all_missing():
    overview = loader.compute_overview(
        {"views": {}, "dry_run_executed_count": 0},
        {"incidents": []}, {}, {})
    assert overview["total_incidents"] == 0
    assert overview["engine_a_production_models"] == 0
    assert overview["engine_b_anomaly_status"] == "unavailable"


# ------------------------------------------------------------- formatting


def test_filter_incidents_by_severity_and_rule():
    incidents = [
        {"incident_id": "1", "severity": "high", "rule_id": "DOS"},
        {"incident_id": "2", "severity": "low", "rule_id": "EXPOSURE"},
        {"incident_id": "3", "severity": "critical", "rule_id": "DOS"}]
    hi = fmt.filter_incidents(incidents, severities=["high", "critical"])
    assert {i["incident_id"] for i in hi} == {"1", "3"}
    dos = fmt.filter_incidents(incidents, rules=["DOS"])
    assert {i["incident_id"] for i in dos} == {"1", "3"}
    # Sorted most-severe first.
    assert fmt.filter_incidents(incidents)[0]["severity"] == "critical"


def test_safety_banner_mentions_no_command_execution():
    assert "no command" in fmt.OFFLINE_BANNER.lower()
    joined = " ".join(fmt.SAFETY_STATEMENTS).lower()
    assert "no command execution" in joined
    assert "no live device access" in joined
    assert fmt.SAFETY_VALIDATOR_COMMAND == "python -m scripts.validate_engine_c_safety"


def test_topology_dot_renders_mesh():
    view = {
        "nodes": [
            {"id": "sw1", "label": "sw1", "risk_score": 55, "finding_count": 3},
            {"id": "router1", "label": "router1", "risk_score": 0,
             "finding_count": 0}],
        "edges": [
            {"source": "sw1", "target": "router1", "protocol": "cdp",
             "warning_count": 1}]}
    dot = fmt.topology_dot(view)
    assert dot.startswith("graph topology {")
    assert "layout=neato" in dot                     # mesh layout, not hierarchy
    assert '"sw1" -- "router1"' in dot                # undirected edge = mesh link
    assert "cdp" in dot
    assert "#cc0000" in dot                           # warning edge drawn red
    assert "(3 finding(s))" in dot


def test_topology_dot_empty_without_nodes():
    assert fmt.topology_dot({"nodes": [], "edges": []}) == ""


def test_topology_dot_escapes_quotes():
    dot = fmt.topology_dot({"nodes": [{"id": 'a"b', "label": 'a"b'}], "edges": []})
    assert '\\"' in dot                               # embedded quote escaped


def test_incident_rows_flatten_nested_fields():
    rows = fmt.incident_rows([
        {"incident_id": "1", "severity": "high", "rule_id": "DOS",
         "engines": ["engine_a", "engine_b"], "affected_devices": ["sw1"],
         "title": "t", "confidence": 0.6}])
    assert rows[0]["engines"] == "engine_a, engine_b"
    assert rows[0]["devices"] == "sw1"


# ------------------------------------------------- friendly labels / latest


def test_humanize_timestamp():
    assert loader.humanize_timestamp("2026-07-07T16:42:00+00:00") == \
        "2026-07-07 16:42 UTC"
    assert loader.humanize_timestamp(None) is None
    assert loader.humanize_timestamp("not-a-date") is None


def test_friendly_run_labels_from_metadata(tmp_path: Path):
    nc = tmp_path / "network_config"
    _write(nc / "run_a" / "dashboard", "export_metadata.json",
           {"generated_at": "2026-07-05T10:00:00+00:00"})
    _write(nc / "run_a", "inventory.json", {"devices": []})
    _write(nc / "run_b" / "dashboard", "export_metadata.json",
           {"generated_at": "2026-07-07T16:42:00+00:00"})
    _write(nc / "run_b", "inventory.json", {"devices": []})
    items = loader.labeled_snapshots(nc)
    assert [i["id"] for i in items] == ["run_b", "run_a"]     # newest first
    assert items[0]["is_latest"] is True
    assert items[0]["label"].startswith("Assessment Run ·")
    assert "2026-07-07 16:42 UTC" in items[0]["label"]
    assert "(latest)" in items[0]["label"]
    # Raw id is not surfaced as the label when a timestamp is available.
    assert "run_b" not in items[0]["label"]


def test_friendly_label_falls_back_to_id_without_timestamp(tmp_path: Path):
    nc = tmp_path / "network_config"
    _write(nc / "plain", "inventory.json", {"devices": []})
    items = loader.labeled_snapshots(nc)
    assert items[0]["human"] is None
    assert "plain" in items[0]["label"]


def test_latest_run_selection(tmp_path: Path):
    nc = tmp_path / "network_config"
    _write(nc / "old" / "dashboard", "export_metadata.json",
           {"generated_at": "2026-07-01T09:00:00+00:00"})
    _write(nc / "old", "inventory.json", {"devices": []})
    _write(nc / "new" / "dashboard", "export_metadata.json",
           {"generated_at": "2026-07-08T09:00:00+00:00"})
    _write(nc / "new", "inventory.json", {"devices": []})
    assert loader.latest_snapshot(nc) == "new"
    items = loader.labeled_snapshots(nc)
    # Latest wins even when an older run is the configured default.
    assert loader.resolve_default(items, "old") == "new"
    # Configured id is only a fallback when nothing is discoverable.
    assert loader.resolve_default([], "sample_remediation") == "sample_remediation"


def test_labeled_correlation_runs(tmp_path: Path):
    corr = tmp_path / "correlation"
    _write(corr / "r1", "correlation_summary.json",
           {"timestamp": "2026-07-06T12:00:00+00:00"})
    _write(corr / "r1", "incidents.json", [])
    _write(corr / "r2", "correlation_summary.json",
           {"timestamp": "2026-07-08T12:00:00+00:00"})
    _write(corr / "r2", "incidents.json", [])
    items = loader.labeled_correlation_runs(corr)
    assert items[0]["id"] == "r2"
    assert items[0]["label"].startswith("Incident Run ·")
    assert loader.latest_correlation_run(corr) == "r2"


# ------------------------------------------------- advanced artifact display


def test_describe_artifact_sources_exposes_raw_ids():
    engine_c = {"snapshot_id": "snap", "dashboard_dir": "/x/snap/dashboard",
                "available": True}
    correlation = {"correlation_id": "corr", "report_path": "/y/report.md",
                   "available": True}
    src = loader.describe_artifact_sources(
        engine_c, correlation,
        {"network_config_dir": "/nc", "correlation_dir": "/co"})
    assert src["assessment_run_id"] == "snap"
    assert src["incident_run_id"] == "corr"
    assert src["engine_c_dashboard_dir"] == "/x/snap/dashboard"
    assert src["correlation_report_path"] == "/y/report.md"
    assert src["network_config_dir"] == "/nc"
    assert src["correlation_dir"] == "/co"


# ------------------------------------------------- executive summary


def test_build_executive_summary_attention():
    engine_c = {"views": {"dashboard_summary": {
        "finding_count": 3, "top_risk_devices": [{"device": "sw9"}]}},
        "dry_run_executed_count": 0}
    correlation = {"incidents": [
        {"incident_id": "1", "severity": "high", "title": "saturation",
         "rule_id": "DOS_SATURATION", "affected_devices": ["sw1"],
         "root_cause_hypothesis": "possible attack-induced saturation",
         "recommended_actions": [{"title": "Investigate saturation",
                                  "detail": "check utilisation", "owner": "security"}]},
        {"incident_id": "2", "severity": "low", "title": "minor",
         "rule_id": "SINGLE", "affected_devices": []}]}
    engine_b = {"available": True, "anomaly_status": "synthetic: 12.8% anomalous"}
    summary = loader.build_executive_summary(engine_c, correlation, engine_b, {})
    assert summary["network_status_level"] == "attention"
    assert summary["critical_incident_count"] == 1
    assert "sw1" in summary["affected_devices"]
    assert "sw9" in summary["affected_devices"]
    assert "possible attack-induced saturation" in summary["likely_root_causes"]
    assert summary["recommended_actions"][0]["title"] == "Investigate saturation"
    assert summary["safety_status"]["dry_run_executed"] == 0
    assert summary["safety_status"]["no_command_execution"] is True


def test_build_executive_summary_stable_when_empty():
    summary = loader.build_executive_summary(
        {"views": {}, "dry_run_executed_count": 0}, {"incidents": []}, {}, {})
    assert summary["network_status_level"] == "stable"
    assert summary["critical_incidents"] == []
    assert summary["affected_devices"] == []


def test_empty_state_guidance_is_plain_language():
    assert "export_network_config_dashboard" in fmt.EMPTY_NO_ASSESSMENT
    assert "assessment run" in fmt.EMPTY_NO_ASSESSMENT.lower()
    assert "run_correlation" in fmt.EMPTY_NO_INCIDENT_RUN
    assert "incident run" in fmt.EMPTY_NO_INCIDENT_RUN.lower()


# ------------------------------------------------------------- launcher


def test_run_dashboard_missing_streamlit(monkeypatch, capsys):
    import scripts.run_dashboard as cli

    monkeypatch.setattr(cli, "streamlit_available", lambda: False)
    code = cli.main([])
    assert code == 1
    out = capsys.readouterr().out
    assert "pip install streamlit" in out
    assert "streamlit run src/dashboard/app.py" in out


def test_run_dashboard_launches_when_available(monkeypatch):
    import scripts.run_dashboard as cli

    launched: dict = {}

    def _fake_launch(extra):
        launched["extra"] = extra
        return 0

    monkeypatch.setattr(cli, "streamlit_available", lambda: True)
    monkeypatch.setattr(cli, "_launch", _fake_launch)
    assert cli.main(["--port", "8599"]) == 0
    assert launched["extra"] == ["--server.port", "8599"]
