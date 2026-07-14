"""Tests for the offline full-demo orchestrator.

Everything here is offline and mocked: no real training, no subprocess is
actually executed against the network, no socket, no sleep. Synthetic artefact
fixtures drive the readiness checks and planner.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.demo import artifacts, planner, readiness, runner
from src.demo.models import (
    FAILED,
    REUSED_EXISTING,
    SKIPPED,
    SUCCESS,
    DemoCommand,
)
from src.demo.runner import DisallowedCommandError, DemoRunner

DEMO_YAML = {
    "demo": {
        "engine_c_input_dir": "datasets/samples/network_config",
        "engine_c_snapshot": "demo_assessment",
        "engine_a_datasets": ["nsl_kdd", "unsw_nb15", "cicids2017"],
        "engine_a_model": "xgboost", "engine_b_dataset": "synthetic",
        "engine_b_model": "isolation_forest", "syslog_run": "latest",
        "correlation_id": "demo_full_correlation", "refresh_engine_c": True,
        "conditional_training": True, "launch_dashboard": False,
    },
    "syslog": {"fallback_fixture": None},
}


def _paths(tmp: Path) -> SimpleNamespace:
    out = tmp / "outputs"
    return SimpleNamespace(
        root=tmp, outputs_dir=out,
        registry_dir=out / "registry",
        network_config_dir=out / "network_config",
        network_health_dir=out / "network_health",
        correlation_dir=out / "correlation",
        reports_dir=out / "reports",
        error_analysis_dir=out / "error_analysis",
        visualizations_dir=out / "visualizations",
        experiments_dir=out / "experiments")


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), "utf-8")


def _production(paths, datasets=("nsl_kdd", "unsw_nb15", "cicids2017")) -> None:
    _write(paths.registry_dir / "production.json",
           {d: {"model_type": "xgboost", "experiment_id": f"{d}_xgb_1"}
            for d in datasets})


def _engine_b(paths, dataset="synthetic") -> None:
    run = paths.network_health_dir / "experiments" / dataset / "iforest" / "exp1"
    _write(run / "metrics.json", {"test": {"n_samples": 10,
                                           "n_anomalous_predicted": 1}})


def _engine_c(paths, snapshot="demo_assessment", dashboard=True) -> None:
    root = paths.network_config_dir / snapshot
    for name in ("inventory.json", "findings.json", "remediation_plan.json"):
        _write(root / name, [])
    if dashboard:
        for name in ("dashboard_summary", "findings_view", "topology_view",
                     "remediation_view"):
            _write(root / "dashboard" / f"{name}.json", {})


def _syslog(paths, run_id="lw_run") -> None:
    run = paths.outputs_dir / "syslog_ingestion" / run_id
    _write(run / "parser_summary.json", {"parsed_events": 10, "hosts": ["SW1"]})
    _write(run / "engine_c" / "syslog_findings.json", [])


def _correlation(paths, cid="demo_full_correlation") -> None:
    root = paths.correlation_dir / cid
    _write(root / "signals.json", [])
    _write(root / "incidents.json", [])
    _write(root / "correlation_summary.json", {"total_signals": 0,
                                               "total_incidents": 0})
    (root / "correlation_report.md").write_text("# report", "utf-8")


def _streaming(paths) -> None:
    _write(paths.outputs_dir / "streaming" / "current" / "current_state.json",
           {"total_events": 5, "safety": {"offline_only": True,
                                          "no_command_execution": True}})
    _write(paths.outputs_dir / "streaming" / "stream_summary.json",
           {"total_events": 5})


# --------------------------------------------------------------- config/plan
def test_resolve_config_defaults():
    config = planner.resolve_config(DEMO_YAML, {})
    assert config.engine_c_snapshot == "demo_assessment"
    assert config.engine_a_datasets == ("nsl_kdd", "unsw_nb15", "cicids2017")
    assert config.correlation_id == "demo_full_correlation"


def test_demo_plan_generation(tmp_path):
    paths = _paths(tmp_path)
    _production(paths)
    _engine_b(paths)
    config = planner.resolve_config(DEMO_YAML, {})
    stages = planner.build_plan(config, paths)
    names = [s.name for s in stages]
    assert names == ["env_validation", "engine_a", "engine_b",
                     "engine_c_assessment", "syslog", "engine_c_dashboard",
                     "correlation", "streaming", "frontend_readiness",
                     "final_report"]


def test_engine_c_assessment_command_generation(tmp_path):
    config = planner.resolve_config(DEMO_YAML, {})
    stages = planner.build_plan(config, _paths(tmp_path))
    ec = next(s for s in stages if s.name == "engine_c_assessment")
    modules = [c.module for c in ec.commands]
    assert "scripts.analyze_network_config" in modules
    assert "scripts.dry_run_network_actions" in modules
    assert "scripts.generate_network_config_report" in modules
    analyze = next(c for c in ec.commands
                   if c.module == "scripts.analyze_network_config")
    assert "--snapshot-id" in analyze.args and "demo_assessment" in analyze.args


def test_correlation_and_streaming_command_generation(tmp_path):
    paths = _paths(tmp_path)
    _production(paths)
    _syslog(paths)
    config = planner.resolve_config(DEMO_YAML, {})
    stages = planner.build_plan(config, paths)
    corr = next(s for s in stages if s.name == "correlation").commands[0]
    assert corr.module == "scripts.run_correlation"
    assert "--syslog-run" in corr.args      # syslog exists -> included
    stream = next(s for s in stages if s.name == "streaming").commands[0]
    assert stream.module == "scripts.run_streaming_demo"
    assert "--snapshot-id" in stream.args and "--no-sleep" in stream.args


# --------------------------------------------------------------- readiness
def test_engine_c_artifact_readiness(tmp_path):
    paths = _paths(tmp_path)
    _engine_c(paths, dashboard=False)
    core = readiness.engine_c_ready(paths.network_config_dir, "demo_assessment",
                                    include_dashboard=False)
    assert core["ready"]
    full = readiness.engine_c_ready(paths.network_config_dir, "demo_assessment",
                                    include_dashboard=True)
    assert not full["ready"] and "dashboard/findings_view.json" in full["missing"]


def test_engine_a_reuse(tmp_path):
    paths = _paths(tmp_path)
    _production(paths)
    ready = readiness.engine_a_ready(paths.registry_dir,
                                     ("nsl_kdd", "unsw_nb15", "cicids2017"))
    assert ready["ready"] and not ready["missing_datasets"]


def test_engine_b_reuse_and_missing(tmp_path):
    paths = _paths(tmp_path)
    assert not readiness.engine_b_ready(paths.network_health_dir,
                                        "synthetic")["ready"]
    _engine_b(paths)
    assert readiness.engine_b_ready(paths.network_health_dir,
                                    "synthetic")["ready"]


def test_syslog_latest_reuse(tmp_path):
    paths = _paths(tmp_path)
    _syslog(paths)
    ready = readiness.syslog_ready(paths.outputs_dir, "latest")
    assert ready["ready"] and ready["run_id"] == "lw_run"


# --------------------------------------------------------------- planner branches
def test_engine_a_missing_triggers_training_plan(tmp_path):
    paths = _paths(tmp_path)
    _production(paths, datasets=("nsl_kdd",))  # unsw_nb15/cicids2017 missing
    config = planner.resolve_config(DEMO_YAML, {})
    stage = planner._engine_a_stage(config, paths)
    modules = [c.module for c in stage.commands]
    assert "scripts.train_model" in modules and "scripts.promote_model" in modules
    assert stage.details["train_datasets"] == ["unsw_nb15", "cicids2017"]


def test_engine_a_force_training_plan(tmp_path):
    paths = _paths(tmp_path)
    _production(paths)  # all present
    config = planner.resolve_config(DEMO_YAML, {"force_train_engine_a": True})
    stage = planner._engine_a_stage(config, paths)
    assert any(c.module == "scripts.train_model" for c in stage.commands)


def test_engine_a_skip_training_missing_fails(tmp_path):
    paths = _paths(tmp_path)
    # no production.json -> all missing
    config = planner.resolve_config(DEMO_YAML, {"skip_training": True})
    stages = planner.build_plan(config, paths)
    r = DemoRunner(config, paths, tmp_path)
    stage = next(s for s in stages if s.name == "engine_a")
    r._run_stage(stage)
    assert stage.status == FAILED


def test_engine_b_missing_triggers_training(tmp_path):
    paths = _paths(tmp_path)
    config = planner.resolve_config(DEMO_YAML, {})
    stage = planner._engine_b_stage(config, paths)
    modules = [c.module for c in stage.commands]
    assert "scripts.train_network_health_model" in modules


def test_missing_optional_syslog_warns(tmp_path):
    paths = _paths(tmp_path)
    config = planner.resolve_config(DEMO_YAML, {})
    stage = planner._syslog_stage(config, paths)
    assert stage.warnings and not stage.required
    r = DemoRunner(config, paths, tmp_path)
    r._recheck_syslog(stage)
    assert stage.status == SKIPPED


def test_require_syslog_failure(tmp_path):
    paths = _paths(tmp_path)
    config = planner.resolve_config(DEMO_YAML, {"require_syslog": True})
    stage = planner._syslog_stage(config, paths)
    r = DemoRunner(config, paths, tmp_path)
    r._recheck_syslog(stage)
    assert stage.status == FAILED and stage.required


# --------------------------------------------------------------- runner/safety
def test_allowlist_enforcement():
    runner.assert_allowed(DemoCommand("scripts.run_correlation"))
    with pytest.raises(DisallowedCommandError):
        runner.assert_allowed(DemoCommand("os.system"))


def test_arbitrary_command_rejected():
    with pytest.raises(DisallowedCommandError):
        runner.assert_allowed(DemoCommand("subprocess"))
    with pytest.raises(DisallowedCommandError):
        runner.assert_allowed(DemoCommand("scripts.apply_network_action"))


def test_dry_run_executes_nothing(tmp_path, monkeypatch):
    import subprocess
    paths = _paths(tmp_path)
    _production(paths)
    _engine_b(paths)

    def _boom(*a, **k):
        raise AssertionError("subprocess must not run in dry-run")
    monkeypatch.setattr(subprocess, "run", _boom)

    config = planner.resolve_config(DEMO_YAML, {"dry_run": True})
    stages = planner.build_plan(config, paths)
    DemoRunner(config, paths, tmp_path).run(stages)
    cmd_stage = next(s for s in stages if s.name == "correlation")
    assert cmd_stage.status == SKIPPED  # planned, not executed


def test_launch_dashboard_only_after_readiness(tmp_path, monkeypatch):
    """--launch-dashboard in dry-run must not launch the dashboard."""
    calls = []
    monkeypatch.setattr(DemoRunner, "_exec",
                        lambda self, cmd: calls.append(cmd) or (0, ""))
    import scripts.prepare_full_demo as script

    monkeypatch.setattr(script, "bootstrap",
                        lambda args: SimpleNamespace(paths=_full_paths(tmp_path)))
    rc = script.main(["--dry-run", "--launch-dashboard"])
    assert rc == 0
    assert not any(c.module == "scripts.run_dashboard" for c in calls)


def _full_paths(tmp_path):
    paths = _paths(tmp_path)
    _production(paths)
    _engine_b(paths)
    _syslog(paths)
    paths.logs_dir = tmp_path / "logs"
    return paths


# --------------------------------------------------------------- integration
def test_full_reuse_run_all_ready(tmp_path, monkeypatch):
    """Every artefact pre-exists -> stages succeed with no real subprocess.

    Command stages (dashboard export / correlation / streaming / safety) always
    run a command; here ``_exec`` is a no-op because their outputs already exist,
    so the per-stage rechecks pass without executing anything real.
    """
    paths = _paths(tmp_path)
    (tmp_path / "datasets" / "samples" / "network_config").mkdir(parents=True)
    _production(paths)
    _engine_b(paths)
    _engine_c(paths)
    _syslog(paths)
    _correlation(paths)
    _streaming(paths)

    monkeypatch.setattr(DemoRunner, "_exec", lambda self, cmd: (0, ""))
    monkeypatch.setattr(
        readiness, "dashboard_readiness",
        lambda *a, **k: {"ready": True, "sections": {}, "safety_banner_ok": True,
                         "missing_required_sections": [], "incident_count": 0,
                         "syslog_available": True, "clock_integrity_warning": False,
                         "optional_sections": []})

    config = planner.resolve_config(DEMO_YAML, {"reuse_assessment": True})
    stages = planner.build_plan(config, paths)
    roll = DemoRunner(config, paths, tmp_path).run(stages)
    assert roll["all_required_ok"], {s.name: s.status for s in stages}


def test_artifact_manifest_and_latest_pointer(tmp_path):
    paths = _paths(tmp_path)
    _production(paths)
    config = planner.resolve_config(DEMO_YAML, {"dry_run": True})
    stages = planner.build_plan(config, paths)
    metrics = {"correlation": {"total_signals": 1}}
    out_root = paths.outputs_dir / "demo"
    written = artifacts.write_demo_run(
        config, stages, {"all_required_ok": True, "dashboard": {}}, metrics,
        "demo_test", out_root)
    assert Path(written["latest"]).is_file()
    run_dir = out_root / "demo_test"
    for name in ("demo_config.json", "steps.json", "commands.json",
                 "generated_artifacts.json", "warnings.json", "readiness.json",
                 "demo_report.md"):
        assert (run_dir / name).is_file()


def test_final_report_rendering(tmp_path):
    from src.demo.reporting import build_demo_report
    config = planner.resolve_config(DEMO_YAML, {})
    stages = planner.build_plan(config, _paths(tmp_path))
    report = build_demo_report(config, stages, {"sections": {}, "ready": True},
                               {"correlation": {"total_signals": 3}}, "demo_x")
    assert "Full-Demo Readiness" in report
    assert "Launch the frontend" in report and "Safety" in report


# --------------------------------------------------------------- guards
def test_no_live_device_or_capture_imports():
    banned = ("netmiko", "napalm", "paramiko", "pysnmp", "scapy", "pyshark",
              "telnetlib", "streamlit")
    pkg_dir = Path(planner.__file__).parent
    for py in pkg_dir.glob("*.py"):
        text = py.read_text("utf-8")
        for lib in banned:
            assert f"import {lib}" not in text, f"{py.name} imports {lib}"


def test_no_source_artifact_deletion():
    """The demo package must not delete/overwrite source artefacts."""
    pkg_dir = Path(planner.__file__).parent
    for py in pkg_dir.glob("*.py"):
        text = py.read_text("utf-8")
        for danger in ("shutil.rmtree", "os.remove", "os.unlink", ".unlink("):
            assert danger not in text, f"{py.name} contains {danger}"


def test_apply_action_not_in_allowlist():
    assert "scripts.apply_network_action" not in runner.ALLOWLIST
    assert all("apply" not in m for m in runner.ALLOWLIST)
