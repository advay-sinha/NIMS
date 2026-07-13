"""Tests for the Phase 11 offline streaming foundation.

Everything here is offline: events are replayed from local JSON fixtures, no
device is contacted, no pacing sleep is used, and nothing is executed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.streaming import models as m
from src.streaming import sources
from src.streaming.artifacts import load_current_state
from src.streaming.event_log import EventLog
from src.streaming.models import StreamEvent, event_id
from src.streaming.replay import order_events, replay
from src.streaming.runtime import run_stream
from src.streaming.state import MonitoringState


def _write(directory: Path, name: str, payload) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), "utf-8")


def _event(event_type=m.CONFIG_FINDING, engine=m.ENGINE_C, severity="high",
           entity_id="e1", title="t", **kw) -> StreamEvent:
    return StreamEvent(
        event_id=event_id(event_type, engine, entity_id, title),
        event_type=event_type, source_engine=engine, severity=severity,
        title=title, summary="s", entity_type="device", entity_id=entity_id, **kw)


# --------------------------------------------------------------- models


def test_event_id_deterministic_and_to_dict():
    a = _event()
    b = _event()
    assert a.event_id == b.event_id
    assert a.event_id.startswith("EVT-")
    d = a.to_dict()
    assert d["event_type"] == m.CONFIG_FINDING and "payload" in d
    assert "payload" not in a.to_row()


# --------------------------------------------------------------- event log


def test_event_log_append_and_read_roundtrip(tmp_path: Path):
    log = EventLog(tmp_path / "events.jsonl")
    log.reset()
    log.append(_event(entity_id="a"))
    log.append(_event(entity_id="b"))
    events = log.read_all()
    assert [e.entity_id for e in events] == ["a", "b"]


def test_event_log_reset_truncates(tmp_path: Path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append(_event(entity_id="a"))
    log.reset()
    assert log.read_all() == []


# --------------------------------------------------------------- sources


def _build_artifacts(tmp: Path) -> dict:
    # Correlation run
    corr = tmp / "correlation" / "run1"
    _write(corr, "correlation_summary.json",
           {"correlation_id": "run1", "timestamp": "2026-07-08T10:00:00+00:00"})
    _write(corr, "incidents.json", [
        {"incident_id": "INC1", "severity": "high", "title": "saturation",
         "rule_id": "DOS", "affected_devices": ["sw1"],
         "root_cause_hypothesis": "possible saturation"}])
    # Engine C dashboard
    dash = tmp / "network_config" / "snap" / "dashboard"
    _write(dash, "export_metadata.json", {"generated_at": "2026-07-08T09:00:00+00:00"})
    _write(dash, "findings_view.json", {"findings": [
        {"finding_id": "F1", "rule_id": "STP", "title": "stp blocking",
         "severity": "medium", "device": "sw1", "interface": "Gi0/3",
         "evidence": "blocking"}]})
    _write(dash, "topology_view.json", {"warnings": [
        {"warning_id": "TW1", "severity": "warning", "message": "loop risk",
         "device": "sw1", "interface": "Gi0/3", "evidence": "x"}]})
    _write(dash, "remediation_view.json", {"command_actions": [
        {"action_id": "A1", "title": "shutdown", "risk_level": "medium",
         "device": "sw1", "interface": "Gi0/7"}]})
    # Engine B experiment
    run = tmp / "network_health" / "experiments" / "synthetic" / "if" / "run_1"
    _write(run, "metrics.json", {"test": {"n_samples": 100,
           "n_anomalous_predicted": 30, "f1": 0.8}})
    _write(run, "manifest.json", {"experiment_id": "run_1", "model_name": "if"})
    return {
        "correlation_dir": tmp / "correlation",
        "network_config_dir": tmp / "network_config",
        "network_health_dir": tmp / "network_health",
        "registry_dir": tmp / "registry",
        "reports_dir": tmp / "reports",
        "error_analysis_dir": tmp / "error_analysis",
        "visualizations_dir": tmp / "visualizations",
        "experiments_dir": tmp / "experiments",
    }


def test_sources_from_correlation(tmp_path: Path):
    _build_artifacts(tmp_path)
    events = sources.events_from_correlation(tmp_path / "correlation", "run1")
    assert len(events) == 1
    assert events[0].event_type == m.CORRELATION_INCIDENT
    assert events[0].incident_id == "INC1"
    assert events[0].timestamp == "2026-07-08T10:00:00+00:00"


def test_sources_from_engine_c(tmp_path: Path):
    _build_artifacts(tmp_path)
    events = sources.events_from_engine_c(tmp_path / "network_config", "snap")
    types = {e.event_type for e in events}
    assert m.CONFIG_FINDING in types
    assert m.TOPOLOGY_WARNING in types
    assert m.REMEDIATION_PLAN in types


def test_sources_from_engine_b(tmp_path: Path):
    _build_artifacts(tmp_path)
    events = sources.events_from_engine_b(tmp_path / "network_health")
    assert len(events) == 1
    assert events[0].event_type == m.HEALTH_ANOMALY
    assert events[0].severity == "high"          # 30% >= 20%


def test_collect_events_includes_safety_and_system(tmp_path: Path):
    dirs = _build_artifacts(tmp_path)
    config = {"sources": {"correlation": {"enabled": True,
              "default_correlation_id": "run1"},
              "engine_c": {"enabled": True, "default_snapshot_id": "snap"},
              "engine_b": {"enabled": True}, "engine_a": {"enabled": False}}}
    events = sources.collect_events(config, dirs)
    types = {e.event_type for e in events}
    assert m.SAFETY_STATUS in types and m.SYSTEM_STATUS in types
    assert m.CORRELATION_INCIDENT in types


# --------------------------------------------------------------- state


def test_state_apply_and_snapshot():
    state = MonitoringState()
    state.apply(_event(event_type=m.CORRELATION_INCIDENT, engine=m.CORRELATION,
                       severity="high", entity_id="INC1", incident_id="INC1",
                       device_id="sw1", seq=0, emitted_at="t0"))
    state.apply(_event(event_type=m.CONFIG_FINDING, severity="low",
                       entity_id="F1", device_id="sw2", seq=1, emitted_at="t1"))
    snap = state.snapshot()
    assert snap["total_events"] == 2
    assert snap["active_incident_count"] == 1
    assert snap["critical_incident_count"] == 1
    assert set(snap["active_devices"]) == {"sw1", "sw2"}
    assert snap["safety"]["no_command_execution"] is True


# --------------------------------------------------------------- replay


def test_replay_orders_by_timestamp_and_stamps():
    events = [_event(entity_id="late", timestamp="2026-07-08T12:00:00+00:00"),
              _event(entity_id="early", timestamp="2026-07-08T08:00:00+00:00")]
    emitted: list[StreamEvent] = []
    count = replay(events, emitted.append, tick_seconds=0.0)
    assert count == 2
    assert [e.entity_id for e in emitted] == ["early", "late"]
    assert emitted[0].seq == 0 and emitted[1].seq == 1
    assert all(e.emitted_at for e in emitted)


def test_replay_respects_max_events():
    events = [_event(entity_id=str(i)) for i in range(5)]
    emitted: list[StreamEvent] = []
    count = replay(events, emitted.append, max_events=2)
    assert count == 2


def test_order_events_missing_timestamp_last():
    events = [_event(entity_id="none"),
              _event(entity_id="ts", timestamp="2026-07-08T08:00:00+00:00")]
    ordered = order_events(events)
    assert [e.entity_id for e in ordered] == ["ts", "none"]


# --------------------------------------------------------------- runtime


def test_run_stream_end_to_end(tmp_path: Path):
    dirs = _build_artifacts(tmp_path)
    dirs.update({
        "output_dir": tmp_path / "streaming",
        "current_state_dir": tmp_path / "streaming" / "current",
        "event_log_path": tmp_path / "streaming" / "events.jsonl"})
    config = {
        "streaming": {"tick_seconds": 0.0, "max_events": None, "loop": False},
        "sources": {"correlation": {"enabled": True,
                    "default_correlation_id": "run1"},
                    "engine_c": {"enabled": True, "default_snapshot_id": "snap"},
                    "engine_b": {"enabled": True}, "engine_a": {"enabled": False}},
        "dashboard": {"write_current_state": True, "write_summary": True}}
    result = run_stream(config, dirs, sleep_fn=None)   # no pacing
    assert result.events_emitted > 0
    assert (tmp_path / "streaming" / "events.jsonl").is_file()
    state = load_current_state(tmp_path / "streaming" / "current")
    assert state["available"] is True
    assert state["safety"]["offline_only"] is True
    assert state["total_events"] == result.events_emitted
    summary = json.loads(result.summary_path.read_text("utf-8"))
    assert "no device access" in summary["safety_note"].lower()


# --------------------------------------------------------------- formatting


def test_formatting_event_rows_and_labels():
    from src.streaming import formatting as fmt

    rows = fmt.event_rows([
        {"seq": 0, "event_type": m.CORRELATION_INCIDENT, "severity": "high",
         "source_engine": m.CORRELATION, "device_id": "sw1", "title": "t",
         "emitted_at": "t0"}])
    assert rows[0]["type"] == "Correlated incident"
    assert rows[0]["device"] == "sw1"
    assert "no command execution" in fmt.STREAM_SAFETY_BANNER.lower()


# --------------------------------------------------------------- CLI


class _FakeCtx:
    def __init__(self, tmp: Path):
        self.config = {}

        class _P:
            pass

        self.paths = _P()
        self.paths.root = tmp
        self.paths.correlation_dir = tmp / "correlation"
        self.paths.network_config_dir = tmp / "network_config"
        self.paths.network_health_dir = tmp / "network_health"
        self.paths.registry_dir = tmp / "registry"
        self.paths.reports_dir = tmp / "reports"
        self.paths.error_analysis_dir = tmp / "error_analysis"
        self.paths.visualizations_dir = tmp / "visualizations"
        self.paths.experiments_dir = tmp / "experiments"


def _streaming_config_file(tmp: Path, out: Path, *, unsafe=False) -> Path:
    import yaml
    cfg = {
        "streaming": {"enabled": True, "tick_seconds": 0.0,
                      "output_dir": str(out),
                      "current_state_dir": str(out / "current"),
                      "event_log_path": str(out / "events.jsonl")},
        "safety": {"allow_device_access": unsafe},
        "sources": {"correlation": {"enabled": True,
                    "default_correlation_id": "run1"},
                    "engine_c": {"enabled": True, "default_snapshot_id": "snap"},
                    "engine_b": {"enabled": True}, "engine_a": {"enabled": False}},
        "dashboard": {"write_current_state": True, "write_summary": True}}
    path = tmp / "streaming.yaml"
    path.write_text(yaml.safe_dump(cfg), "utf-8")
    return path


def test_cli_happy_path(tmp_path: Path, monkeypatch):
    import scripts.run_streaming_demo as cli

    _build_artifacts(tmp_path)
    out = tmp_path / "stream_out"
    cfg = _streaming_config_file(tmp_path, out)
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(tmp_path))
    code = cli.main(["--streaming-config", str(cfg), "--no-sleep"])
    assert code == 0
    assert (out / "current" / "current_state.json").is_file()
    assert (out / "events.jsonl").is_file()


def test_cli_refuses_unsafe_flags(tmp_path: Path, monkeypatch):
    import scripts.run_streaming_demo as cli

    out = tmp_path / "stream_out"
    cfg = _streaming_config_file(tmp_path, out, unsafe=True)
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(tmp_path))
    assert cli.main(["--streaming-config", str(cfg), "--no-sleep"]) == 1
    assert not (out / "events.jsonl").exists()
