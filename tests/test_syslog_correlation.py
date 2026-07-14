"""Tests for Phase 13 syslog-enhanced correlation.

Everything here is offline: small synthetic syslog artefacts are written to a
tmp dir, normalised into signals, correlated into incidents and replayed as
stream events. No device is contacted, no socket is opened, nothing is executed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.correlation.engine import correlate
from src.correlation.models import ENGINE_A, ENGINE_B, ENGINE_C, SYSLOG, Signal
from src.correlation.rules import (
    CLOCK_INTEGRITY_RISK,
    DUPLICATE_IP_CONFLICT,
    LOOP_OR_REDUNDANCY_INSTABILITY,
    MANAGEMENT_ACCESS_EXPOSURE,
    POE_ENDPOINT_FAILURE,
    PORT_INSTABILITY,
    SINGLE_SYSLOG_HIGH_RISK,
    SYSLOG_SNMP_AUTH_CAMPAIGN,
)
from src.correlation.signal_normalization import (
    SYSLOG_PORT_FLAP,
    match_entities,
    normalize_interface,
    resolve_confidence,
    source_type_for_finding,
)
from src.correlation.syslog_loader import (
    load_syslog_artifacts,
    load_syslog_signals,
    resolve_run_id,
)
from src.utils.config import load_yaml

CONFIG = load_yaml("configs/correlation.yaml")


# --------------------------------------------------------------- fixtures
def _finding(rule_id, severity, device=None, interface=None, vlan=None,
             evidence="evidence", event_count=1, details=None, tags=(),
             category="syslog", first="2026-01-01T00:00:00+05:30",
             last="2026-01-01T01:00:00+05:30", fid=None):
    return {
        "finding_id": fid or f"SYSF-{rule_id}-{device}-{interface}",
        "rule_id": rule_id, "title": f"{rule_id} on {device}", "severity": severity,
        "category": category, "device": device, "interface": interface,
        "vlan": vlan, "evidence": evidence, "recommendation": "check it",
        "confidence": "high", "event_count": event_count,
        "first_seen": first, "last_seen": last, "tags": list(tags),
        "details": details or {},
    }


def _make_run(root: Path, run_id: str, findings, *, parsed=100, generic=0,
              clock_unreliable=0, events=None, weak_labels=None):
    run_dir = root / run_id
    (run_dir / "engine_c").mkdir(parents=True, exist_ok=True)
    (run_dir / "engine_b").mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id, "parsed_events": parsed, "weighted_events": parsed + 5,
        "input_files": [f"{run_id}.log"], "hosts": ["SW1"],
        "time_range": {"first": "2026-01-01T00:00:00+05:30",
                       "last": "2026-01-02T00:00:00+05:30"},
        "severity_distribution": {"error": 10}, "top_facilities": {"PORTMGR": 5},
        "top_mnemonics": {"PORTMGR-LINEPROTO_UP": 5},
        "parse_status": {"parsed": parsed - generic, "generic": generic},
        "clock_unreliable_events": clock_unreliable,
        "safety": "offline saved-log mode; no device access",
    }
    (run_dir / "parser_summary.json").write_text(json.dumps(summary), "utf-8")
    (run_dir / "engine_c" / "syslog_findings.json").write_text(
        json.dumps(findings), "utf-8")
    (run_dir / "engine_c" / "syslog_rule_summary.json").write_text(
        json.dumps({"total": len(findings)}), "utf-8")
    if events is not None:
        (run_dir / "parsed_events.json").write_text(json.dumps(events), "utf-8")
    (run_dir / "engine_b" / "weak_label_summary.json").write_text(
        json.dumps(weak_labels or {"positive_windows": {}, "window_count": 0}),
        "utf-8")
    (run_dir / "engine_b" / "feature_summary.json").write_text(
        json.dumps({"window_count": 0}), "utf-8")
    return run_dir


# --------------------------------------------------------------- discovery
def test_latest_run_discovery(tmp_path):
    _make_run(tmp_path, "run_a", [])
    _make_run(tmp_path, "run_b", [])
    import os
    os.utime(tmp_path / "run_b" / "parser_summary.json", (10**9 + 100, 10**9 + 100))
    assert resolve_run_id(tmp_path, "latest") == "run_b"


def test_explicit_run_loading(tmp_path):
    _make_run(tmp_path, "run_x", [_finding("SYS-POE-FAULT", "high", "SW1", "gi0/1")])
    data = load_syslog_artifacts(tmp_path, "run_x")
    assert data["available"] and data["run_id"] == "run_x"
    assert data["event_count"] == 100


def test_missing_run_graceful(tmp_path):
    data = load_syslog_artifacts(tmp_path, "nope")
    assert data["available"] is False
    assert data["warnings"] and "ingest_switch_syslog" in data["warnings"][0]


def test_require_syslog_failure_behaviour(tmp_path):
    lr, meta = load_syslog_signals(tmp_path, "missing", CONFIG)
    assert lr.signals == []
    assert meta["syslog_signals_loaded"] == 0


# --------------------------------------------------------------- normalization
def test_event_to_signal_normalization(tmp_path):
    _make_run(tmp_path, "r", [_finding("SYS-PORT-FLAP", "high", "SW1", "gi0/1",
                                        event_count=8)])
    lr, _ = load_syslog_signals(tmp_path, "r", CONFIG)
    sig = next(s for s in lr.signals if s.source_type == SYSLOG_PORT_FLAP)
    assert sig.engine == SYSLOG and sig.device == "SW1" and sig.interface == "gi0/1"
    assert sig.event_count == 8 and sig.confidence_label == "high"


def test_deterministic_signal_ids(tmp_path):
    _make_run(tmp_path, "r1", [_finding("SYS-POE-FAULT", "high", "SW1", "gi0/2")])
    # Content-addressed: reloading the same run yields identical signal ids.
    id1 = [s.signal_id for s in load_syslog_signals(tmp_path, "r1", CONFIG)[0].signals]
    id2 = [s.signal_id for s in load_syslog_signals(tmp_path, "r1", CONFIG)[0].signals]
    assert id1 == id2 and id1 and id1[0].startswith("SIG-")


def test_duplicate_count_preserved(tmp_path):
    _make_run(tmp_path, "r", [_finding("SYS-SNMP-AUTHFAIL", "high", "SW1",
                                       event_count=42)])
    lr, _ = load_syslog_signals(tmp_path, "r", CONFIG)
    snmp = next(s for s in lr.signals if "snmp" in (s.source_type or "").lower())
    assert snmp.event_count == 42


def test_clock_unreliable_confidence_reduction():
    hi, band_hi, _ = resolve_confidence("SYSLOG_PORT_FLAP", clock_unreliable=False)
    lo, band_lo, notes = resolve_confidence("SYSLOG_PORT_FLAP",
                                            clock_unreliable=True)
    assert lo < hi and band_lo == "low" and notes


def test_generic_parse_fallback(tmp_path):
    events = [{"parse_status": "generic", "hostname": "SW1", "code": "X-Y",
               "duplicate_count": 3, "timestamp": "2026-01-01T00:00:00+05:30"}]
    _make_run(tmp_path, "r", [], generic=1, events=events)
    lr, meta = load_syslog_signals(tmp_path, "r", CONFIG)
    generic = [s for s in lr.signals if s.source_type == "SYSLOG_GENERIC_WARNING"]
    assert generic and generic[0].event_count == 3
    assert generic[0].confidence_label == "low" and generic[0].severity == "info"


def test_interface_normalization():
    assert normalize_interface("GigabitEthernet0/1") == "gi0/1"
    assert normalize_interface("Gi0/1") == "gi0/1"
    assert normalize_interface("Gig 0/1") == "gi0/1"
    assert normalize_interface("TenGigabitEthernet0/28") == "te0/28"
    assert normalize_interface(None) is None


def test_source_type_classification():
    assert source_type_for_finding({"rule_id": "SYS-MAC-FLAP"}) == "SYSLOG_MAC_FLAP"
    assert source_type_for_finding({"rule_id": "???"}) == "SYSLOG_GENERIC_WARNING"


# --------------------------------------------------------------- entity match
def test_exact_entity_match():
    assert match_entities("SW1", "Gi0/1", "SW1", "Gi0/1") == "exact"


def test_normalized_entity_match():
    assert match_entities("SW1", "GigabitEthernet0/1", "SW1", "Gi0/1") == "normalized"


def test_uncertain_entity_match():
    # same interface text but no device -> cannot be sure it's the same entity
    assert match_entities(None, "Gi0/1", None, "Gi0/1") == "uncertain"


def test_false_match_prevented():
    assert match_entities("SW1", "Gi0/1", "SW2", "Gi0/2") == "none"


# --------------------------------------------------------------- correlation
def _correlate(findings, config=CONFIG, extra_signals=None, **run_kw):
    """Helper: load syslog signals from synthetic findings and correlate."""
    import tempfile
    root = Path(tempfile.mkdtemp())
    _make_run(root, "run", findings, **run_kw)
    lr, meta = load_syslog_signals(root, "run", config)
    signals = list(lr.signals) + list(extra_signals or [])
    return correlate(signals, config, "cid", {SYSLOG: "run"}, meta)


def test_snmp_auth_campaign_correlation():
    res = _correlate([_finding("SYS-SNMP-AUTHFAIL", "high", "SW1",
                               event_count=50)])
    assert any(i.rule_id == SYSLOG_SNMP_AUTH_CAMPAIGN for i in res.incidents)


def test_port_instability_correlation():
    res = _correlate([_finding("SYS-PORT-FLAP", "high", "SW1", "gi0/1",
                               event_count=8)])
    inc = next(i for i in res.incidents if i.rule_id == PORT_INSTABILITY)
    assert "SW1" in inc.affected_devices


def test_loop_correlation_multi_source():
    res = _correlate([
        _finding("SYS-MAC-FLAP", "critical", "SW1", "te0/28", event_count=5,
                 details={"max_moves": 120}),
        _finding("SYS-ERPS-CHURN", "medium", "SW1", event_count=4),
    ])
    inc = next(i for i in res.incidents
               if i.rule_id == LOOP_OR_REDUNDANCY_INSTABILITY)
    assert inc.severity == "critical"  # multi-source allows critical


def test_loop_single_source_capped():
    res = _correlate([_finding("SYS-MAC-FLAP", "critical", "SW1", "te0/28",
                               event_count=5, details={"max_moves": 120})])
    inc = next(i for i in res.incidents
               if i.rule_id == LOOP_OR_REDUNDANCY_INSTABILITY)
    assert inc.severity == "high"  # single-source capped below critical


def test_duplicate_ip_correlation():
    res = _correlate([_finding("SYS-ARP-FAST", "high", "SW1",
                               details={"ip_address": "192.168.1.9"})])
    inc = next(i for i in res.incidents if i.rule_id == DUPLICATE_IP_CONFLICT)
    assert inc.severity in ("medium", "high")


def test_poe_correlation():
    res = _correlate([_finding("SYS-POE-FAULT", "high", "SW1", "gi0/7")])
    assert any(i.rule_id == POE_ENDPOINT_FAILURE for i in res.incidents)


def test_management_exposure_correlation():
    res = _correlate([_finding("SYS-TELNET", "medium", "SW1")])
    assert any(i.rule_id == MANAGEMENT_ACCESS_EXPOSURE for i in res.incidents)


def test_clock_integrity_incident():
    res = _correlate([], clock_unreliable=5)
    assert any(i.rule_id == CLOCK_INTEGRITY_RISK for i in res.incidents)
    inc = next(i for i in res.incidents if i.rule_id == CLOCK_INTEGRITY_RISK)
    assert inc.time_reliability in ("approximate", "unreliable")


def test_single_syslog_high_risk_fallback():
    # A device-health finding that no cross rule consumes, at high severity.
    res = _correlate([_finding("SYS-DEVICE", "high", "SW9")])
    assert any(i.rule_id == SINGLE_SYSLOG_HIGH_RISK for i in res.incidents)


def test_duplicate_incident_suppression():
    res = _correlate([_finding("SYS-POE-FAULT", "high", "SW1", "gi0/7")])
    poe = [i for i in res.incidents if i.rule_id == POE_ENDPOINT_FAILURE]
    ids = [i.incident_id for i in poe]
    assert len(ids) == len(set(ids))  # no duplicate incident ids


# --------------------------------------------------------------- summary/report
def test_summary_syslog_fields():
    res = _correlate([_finding("SYS-PORT-FLAP", "high", "SW1", "gi0/1",
                               event_count=8)], clock_unreliable=2)
    s = res.summary
    assert s.syslog_signals_loaded >= 1
    assert s.syslog_findings_loaded == 1
    assert s.clock_unreliable_count == 2
    assert s.incidents_with_syslog_evidence >= 1
    assert SYSLOG in s.signals_by_engine


def test_report_has_syslog_section():
    from src.correlation.reporting import build_report
    res = _correlate([_finding("SYS-PORT-FLAP", "high", "SW1", "gi0/1",
                               event_count=8)], clock_unreliable=3)
    report = build_report(res)
    assert "Syslog Evidence Overview" in report
    assert "Time Reliability Notes" in report


def test_cross_source_entity_match_in_incident():
    """Engine C interface finding on the same normalised interface corroborates."""
    engine_c = Signal(
        signal_id="SIG-ec1", engine=ENGINE_C,
        source_artifact="network_config/x/findings.json", category="port",
        severity="medium", confidence=0.7, title="High errors on Gi0/1",
        description="errors", raw_reference="F1", device="SW1",
        interface="GigabitEthernet0/1", tags=("port",))
    res = _correlate([_finding("SYS-PORT-FLAP", "high", "SW1", "gi0/1",
                               event_count=8)], extra_signals=[engine_c])
    inc = next(i for i in res.incidents if i.rule_id == PORT_INSTABILITY)
    assert ENGINE_C in inc.engines
    assert inc.entity_match_confidence in ("exact", "normalized")


# --------------------------------------------------------------- backward-compat
def test_non_syslog_correlation_unchanged():
    """A run with no syslog keeps the historical {A,B,C} summary shape."""
    a = Signal(signal_id="SIG-a", engine=ENGINE_A, source_artifact="x",
               category="intrusion", severity="medium", confidence=0.6,
               title="cyber", description="d", raw_reference="r", aggregate=True,
               tags=("attack",))
    res = correlate([a], CONFIG, "cid", {ENGINE_A: "x"})
    assert set(res.summary.signals_by_engine) == {ENGINE_A, ENGINE_B, ENGINE_C}
    assert res.summary.incidents_with_syslog_evidence == 0


# --------------------------------------------------------------- streaming
def test_streaming_source_emits_syslog_events(tmp_path):
    from src.streaming import models as sm
    from src.streaming.syslog_source import events_from_syslog
    _make_run(tmp_path, "run", [
        _finding("SYS-SNMP-AUTHFAIL", "high", "SW1", category="security"),
        _finding("SYS-PORT-FLAP", "medium", "SW1", "gi0/1", category="interface"),
        _finding("SYS-MAC-FLAP", "high", "SW1", "te0/1", category="loop"),
    ], clock_unreliable=4,
        weak_labels={"positive_windows": {"degradation_label": 6}})
    events = events_from_syslog(tmp_path, "run")
    assert all(e.source_engine == sm.SYSLOG for e in events)
    types = {e.event_type for e in events}
    assert {sm.CYBER_ALERT, sm.CONFIG_FINDING, sm.TOPOLOGY_WARNING,
            sm.SAFETY_STATUS, sm.SYSTEM_STATUS}.issubset(types)


def test_monitoring_state_tracks_syslog(tmp_path):
    from src.streaming.state import MonitoringState
    from src.streaming.syslog_source import events_from_syslog
    _make_run(tmp_path, "run", [
        _finding("SYS-SNMP-AUTHFAIL", "high", "SW1", category="security"),
        _finding("SYS-PORT-FLAP", "medium", "SW1", "gi0/1", category="interface"),
        _finding("SYS-MAC-FLAP", "high", "SW1", "te0/1", category="loop"),
    ], clock_unreliable=4)
    state = MonitoringState()
    for e in events_from_syslog(tmp_path, "run"):
        state.apply(e)
    snap = state.snapshot()["syslog"]
    assert snap["syslog_event_count"] >= 4
    assert snap["port_instability_count"] >= 1
    assert snap["loop_redundancy_candidates"] >= 1
    assert snap["management_auth_activity"] >= 1
    assert snap["clock_reliability_status"] == "degraded"


# --------------------------------------------------------------- dashboard
def test_dashboard_reads_syslog_enhanced_incidents(tmp_path):
    from src.dashboard import loader
    res = _correlate([_finding("SYS-PORT-FLAP", "high", "SW1", "gi0/1",
                               event_count=8)])
    from src.correlation.artifacts import write_correlation
    out = tmp_path / "corr"
    write_correlation(res, out)
    incidents = json.loads((out / "incidents.json").read_text("utf-8"))
    assert incidents and "time_reliability" in incidents[0]
    assert "syslog_signal_count" in incidents[0]

    # executive summary tolerates the enhanced incidents
    summary = loader.build_executive_summary(
        {"views": {}}, {"incidents": incidents}, {}, {})
    assert "incidents_with_syslog_evidence" in summary


def test_dashboard_missing_syslog_does_not_crash(tmp_path):
    from src.dashboard import loader
    data = loader.load_syslog_run(tmp_path / "syslog", "nope")
    assert data["available"] is False and data["message"]


# --------------------------------------------------------------- safety
def test_no_live_device_libraries_in_correlation():
    """Guard: correlation + syslog modules must not import live-device clients."""
    banned = ("netmiko", "napalm", "paramiko", "pysnmp", "scapy", "pyshark",
              "telnetlib")
    import src.correlation as corr
    pkg_dir = Path(corr.__file__).parent
    for py in pkg_dir.glob("*.py"):
        text = py.read_text("utf-8")
        for lib in banned:
            assert f"import {lib}" not in text, f"{py.name} imports {lib}"
