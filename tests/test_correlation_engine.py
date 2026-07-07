"""Tests for the Phase 9 correlation engine (offline, artefact-driven).

Covers signal loading from each engine's artefacts, the deterministic
correlation rules, severity/confidence scoring, artefact persistence, the
report, and the CLI. No test requires a live device, a running engine pipeline
or any network access.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.correlation.artifacts import write_correlation
from src.correlation.engine import SAFETY_NOTE, correlate
from src.correlation.loader import (
    load_engine_a_signals,
    load_engine_b_signals,
    load_engine_c_signals,
)
from src.correlation.models import (
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    Signal,
    incident_id,
    signal_id,
)
from src.correlation.reporting import build_report
from src.correlation.rules import (
    CONFIG_EXPOSURE,
    DOS_SATURATION,
    LINK_DEGRADATION,
    SINGLE_ENGINE_HIGH_RISK,
    VLAN_POLICY_RISK,
)
from src.utils.config import CONFIG_DIR, load_yaml


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def config() -> dict:
    """The real shipped correlation config (kept authoritative for tests)."""
    return load_yaml(CONFIG_DIR / "correlation.yaml")


def _sig(engine, *, category="config", severity="high", confidence=0.8,
         device=None, interface=None, vlan=None, aggregate=False, tags=(),
         title=None) -> Signal:
    title = title or f"{engine}:{category}"
    return Signal(
        signal_id=signal_id(engine, "src", category, device, interface, title),
        engine=engine, source_artifact=f"{engine}/src.json", category=category,
        severity=severity, confidence=confidence, title=title, description="d",
        raw_reference="ref", device=device, interface=interface, vlan=vlan,
        aggregate=aggregate, tags=tuple(tags))


def _attack(**kw):
    kw.setdefault("tags", ("attack", "dos", "aggregate"))
    kw.setdefault("aggregate", True)
    kw.setdefault("severity", "medium")
    return _sig(ENGINE_A, category="intrusion", **kw)


def _health(**kw):
    kw.setdefault("tags", ("anomaly", "degradation", "aggregate"))
    kw.setdefault("aggregate", True)
    kw.setdefault("severity", "high")
    return _sig(ENGINE_B, category="network_health", **kw)


def _cfg_finding(rule_id, category="port", **kw):
    kw.setdefault("tags", (rule_id,))
    return _sig(ENGINE_C, category=category, **kw)


def _write(directory: Path, name: str, payload) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), "utf-8")


# --------------------------------------------------------- Engine C loading


def test_engine_c_signal_loading_from_findings(tmp_path: Path, config):
    snap = tmp_path / "snap"
    _write(snap, "findings.json", [
        {"finding_id": "F1", "rule_id": "UNUSED_PORT_ADMIN_UP", "title": "unused",
         "severity": "low", "category": "port", "device": "sw1",
         "interface": "Gi0/7", "vlan": None, "confidence": "low",
         "evidence": "admin up", "recommendation": "shut", "tags": ["port"]}])
    result = load_engine_c_signals(snap, "snap", config)
    assert len(result.signals) == 1
    sig = result.signals[0]
    assert sig.engine == ENGINE_C
    assert sig.device == "sw1" and sig.interface == "Gi0/7"
    assert "UNUSED_PORT_ADMIN_UP" in sig.tags
    assert sig.confidence == pytest.approx(0.55)      # low -> 0.55


def test_engine_c_topology_warning_signal(tmp_path: Path, config):
    snap = tmp_path / "snap"
    _write(snap, "topology.json", {"warnings": [
        {"warning_id": "TW004", "severity": "warning", "category": "stp",
         "message": "STP blocking on access port", "device": "sw1",
         "interface": "Gi0/3", "evidence": "blocking"}]})
    result = load_engine_c_signals(snap, "snap", config)
    warn = [s for s in result.signals if s.category == "topology"]
    assert len(warn) == 1
    assert warn[0].severity == "medium"               # warning -> medium
    assert warn[0].interface == "Gi0/3"


def test_engine_c_device_card_signal_only_when_unhealthy(tmp_path: Path, config):
    snap = tmp_path / "snap"
    _write(snap / "dashboard", "device_health_cards.json", {"cards": [
        {"device_id": "sw1", "status": "warning", "finding_count": 3,
         "highest_severity": "medium"},
        {"device_id": "sw2", "status": "healthy", "finding_count": 0}]})
    result = load_engine_c_signals(snap, "snap", config)
    cards = [s for s in result.signals if s.category == "device_health"]
    assert [c.device for c in cards] == ["sw1"]        # healthy is dropped
    assert cards[0].severity == "medium"


# --------------------------------------------------------- Engine B loading


def _write_nh_experiment(nh_dir: Path, dataset: str, run: str, n_pred: int,
                         n_samples: int = 100) -> None:
    run_dir = nh_dir / "experiments" / dataset / "isolation_forest" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"test": {"n_samples": n_samples, "n_anomalous_predicted": n_pred,
                        "f1": 0.84, "roc_auc": 0.97}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), "utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(
        {"experiment_id": run, "model_name": "isolation_forest",
         "created_at": "2026-07-06T19:07:39+00:00"}), "utf-8")


def test_engine_b_aggregate_signal_loading(tmp_path: Path, config):
    nh = tmp_path / "nh"
    _write_nh_experiment(nh, "synthetic", "synthetic_if_20260706T190738", n_pred=30)
    result = load_engine_b_signals(nh, "synthetic", config)
    assert len(result.signals) == 1
    sig = result.signals[0]
    assert sig.engine == ENGINE_B and sig.aggregate is True
    assert sig.severity == "high"                     # 30/100 = 0.3 >= 0.2
    assert result.source == "synthetic_if_20260706T190738"


def test_engine_b_picks_latest_run(tmp_path: Path, config):
    nh = tmp_path / "nh"
    _write_nh_experiment(nh, "synthetic", "synthetic_if_20260706T100000", n_pred=1)
    _write_nh_experiment(nh, "synthetic", "synthetic_if_20260706T200000", n_pred=30)
    result = load_engine_b_signals(nh, "synthetic", config)
    assert result.source == "synthetic_if_20260706T200000"   # newest wins


def test_engine_b_missing_dataset_warns(tmp_path: Path, config):
    result = load_engine_b_signals(tmp_path / "nh", "nope", config)
    assert result.signals == []
    assert result.warnings


# --------------------------------------------------------- Engine A loading


def _write_engine_a(exp_dir: Path, reg_dir: Path, dataset: str, run: str) -> None:
    run_dir = exp_dir / dataset / "xgboost" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(
        {"test": {"f1": 0.92, "roc_auc": 0.99}}), "utf-8")
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "best_per_dataset.json").write_text(json.dumps(
        {dataset: {"experiment_id": run, "model_type": "xgboost"}}), "utf-8")


def test_engine_a_aggregate_signal_loading(tmp_path: Path, config):
    exp, reg, ea = tmp_path / "exp", tmp_path / "reg", tmp_path / "ea"
    _write_engine_a(exp, reg, "unsw_nb15", "unsw_nb15_xgboost_20260704T150017")
    result = load_engine_a_signals(exp, reg, ea, "unsw_nb15", config)
    assert len(result.signals) == 1
    sig = result.signals[0]
    assert sig.engine == ENGINE_A and sig.aggregate is True
    assert "attack" in sig.tags and "dos" in sig.tags   # dataset families tagged
    assert result.source == "unsw_nb15_xgboost_20260704T150017"


def test_engine_a_missing_dataset_warns(tmp_path: Path, config):
    result = load_engine_a_signals(tmp_path / "exp", tmp_path / "reg",
                                   tmp_path / "ea", "nope", config)
    assert result.signals == []
    assert result.warnings


# --------------------------------------------------------------- rules


def test_dos_saturation_rule(config):
    signals = [_attack(), _health(),
               _cfg_finding("TRUNK_WITHOUT_NEIGHBOR", category="topology",
                            device="sw1", interface="Gi0/1", severity="medium")]
    result = correlate(signals, config, "c", {})
    dos = [i for i in result.incidents if i.rule_id == DOS_SATURATION]
    assert len(dos) == 1
    assert dos[0].multi_engine is True
    assert set(dos[0].engines) == {ENGINE_A, ENGINE_B, ENGINE_C}


def test_link_degradation_rule(config):
    signals = [_health(),
               _cfg_finding("HIGH_INTERFACE_ERRORS", category="performance",
                            device="sw1", interface="Gi0/2", severity="high")]
    result = correlate(signals, config, "c", {})
    deg = [i for i in result.incidents if i.rule_id == LINK_DEGRADATION]
    assert len(deg) == 1
    assert deg[0].affected_devices == ("sw1",)
    # No cyber signal present -> engines are B + C only.
    assert set(deg[0].engines) == {ENGINE_B, ENGINE_C}


def test_link_degradation_suppressed_when_attack_present(config):
    signals = [_attack(), _health(),
               _cfg_finding("HIGH_INTERFACE_ERRORS", category="performance",
                            device="sw1", severity="high")]
    result = correlate(signals, config, "c", {})
    assert not [i for i in result.incidents if i.rule_id == LINK_DEGRADATION]


def test_config_exposure_rule(config):
    signals = [_cfg_finding("UNUSED_PORT_ADMIN_UP", category="port",
                            device="sw1", interface="Gi0/7", severity="low")]
    result = correlate(signals, config, "c", {})
    exposure = [i for i in result.incidents if i.rule_id == CONFIG_EXPOSURE]
    assert len(exposure) == 1
    assert exposure[0].affected_devices == ("sw1",)


def test_vlan_policy_risk_rule(config):
    signals = [_attack(),
               _cfg_finding("WRONG_ACCESS_VLAN", category="vlan", device="sw1",
                            interface="Gi0/4", vlan="99", severity="medium")]
    result = correlate(signals, config, "c", {})
    vlan = [i for i in result.incidents if i.rule_id == VLAN_POLICY_RISK]
    assert len(vlan) == 1
    assert vlan[0].related_vlans == ("99",)
    assert ENGINE_A in vlan[0].engines and ENGINE_C in vlan[0].engines


def test_single_engine_high_risk_fallback(config):
    # A lone high finding that matches no cross-engine rule bucket.
    signals = [_cfg_finding("WEAK_SNMP", category="security", device="sw1",
                            severity="high")]
    result = correlate(signals, config, "c", {})
    single = [i for i in result.incidents if i.rule_id == SINGLE_ENGINE_HIGH_RISK]
    assert len(single) == 1
    assert single[0].severity == "high"


def test_single_engine_ignores_low_severity(config):
    signals = [_cfg_finding("WEAK_SNMP", category="security", device="sw1",
                            severity="low")]
    result = correlate(signals, config, "c", {})
    assert not [i for i in result.incidents
                if i.rule_id == SINGLE_ENGINE_HIGH_RISK]


def test_deterministic_incident_id(config):
    signals = [_attack(), _health()]
    first = correlate(signals, config, "c", {}).incidents
    second = correlate(signals, config, "c", {}).incidents
    assert [i.incident_id for i in first] == [i.incident_id for i in second]
    dos = next(i for i in first if i.rule_id == DOS_SATURATION)
    assert dos.incident_id == incident_id(DOS_SATURATION, dos.signals)


# --------------------------------------------------------------- scoring


def test_same_interface_bonus_raises_confidence(config):
    # Engine B (non-aggregate here) and Engine C share the exact interface.
    aligned = [_health(aggregate=False, device="sw1", interface="Gi0/2"),
               _cfg_finding("HIGH_INTERFACE_ERRORS", category="performance",
                            device="sw1", interface="Gi0/2", severity="high")]
    misaligned = [_health(aggregate=False, device="sw1", interface="Gi9/9"),
                  _cfg_finding("HIGH_INTERFACE_ERRORS", category="performance",
                               device="sw1", interface="Gi0/2", severity="high")]
    a = next(i for i in correlate(aligned, config, "c", {}).incidents
             if i.rule_id == LINK_DEGRADATION)
    m = next(i for i in correlate(misaligned, config, "c", {}).incidents
             if i.rule_id == LINK_DEGRADATION)
    assert a.confidence > m.confidence
    assert any("same_interface" in f for f in a.scoring_factors)


def test_aggregate_penalty_applied(config):
    signals = [_attack(), _health()]           # both aggregate
    dos = next(i for i in correlate(signals, config, "c", {}).incidents
               if i.rule_id == DOS_SATURATION)
    assert dos.aggregate_only is True
    assert any("aggregate_penalty" in f for f in dos.scoring_factors)


def test_disabled_rule_is_ignored(config):
    cfg = copy.deepcopy(config)
    cfg["rules"]["DOS_SATURATION"]["enabled"] = False
    signals = [_attack(), _health()]
    result = correlate(signals, cfg, "c", {})
    assert not [i for i in result.incidents if i.rule_id == DOS_SATURATION]


# --------------------------------------------------------- persistence / report


def test_artifact_persistence(tmp_path: Path, config):
    signals = [_attack(), _health(),
               _cfg_finding("UNUSED_PORT_ADMIN_UP", category="port",
                            device="sw1", interface="Gi0/7", severity="low")]
    result = correlate(signals, config, "run1",
                       {ENGINE_A: "expA", ENGINE_B: "expB", ENGINE_C: "snapC"})
    out = tmp_path / "run1"
    write_correlation(result, out)
    for name in ("signals.json", "signals.csv", "incidents.json",
                 "incidents.csv", "correlation_summary.json",
                 "correlation_report.md"):
        assert (out / name).is_file(), name
    incidents = json.loads((out / "incidents.json").read_text("utf-8"))
    assert incidents and all("incident_id" in i for i in incidents)


def test_summary_correctness(config):
    signals = [_attack(), _health(),
               _cfg_finding("UNUSED_PORT_ADMIN_UP", category="port",
                            device="sw1", severity="low")]
    result = correlate(signals, config, "run1",
                       {ENGINE_A: "expA", ENGINE_B: "expB", ENGINE_C: "snapC"})
    s = result.summary
    assert s.total_signals == 3
    assert s.signals_by_engine == {ENGINE_A: 1, ENGINE_B: 1, ENGINE_C: 1}
    assert s.aggregate_signal_count == 2
    assert s.total_incidents == len(result.incidents)
    assert s.engine_c_source == "snapC"
    assert "no command" in s.safety_note.lower()


def test_report_generation(config):
    signals = [_attack(), _health()]
    report = build_report(correlate(signals, config, "run1", {}))
    assert "# Correlation Report — run1" in report
    assert "No commands were executed" in report
    for header in ("## Executive Summary", "## Inputs Used", "## Signal Summary",
                   "## Correlated Incidents", "## Multi-Engine Incidents",
                   "## Single-Engine High-Risk Items", "## Root-Cause Hypotheses",
                   "## Recommended Operator Actions", "## Safety Notes",
                   "## Artifact Appendix"):
        assert header in report, header


def test_safety_note_constant_mentions_no_execution():
    assert "no command" in SAFETY_NOTE.lower()


# ------------------------------------------------------------------- CLI


class _FakeCtx:
    def __init__(self, tmp: Path):
        self.config = {}

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = tmp / "network_config"
        self.paths.network_health_dir = tmp / "network_health"
        self.paths.experiments_dir = tmp / "experiments"
        self.paths.registry_dir = tmp / "registry"
        self.paths.error_analysis_dir = tmp / "error_analysis"
        self.paths.correlation_dir = tmp / "correlation"


def _build_repo(tmp: Path) -> None:
    snap = tmp / "network_config" / "snap"
    _write(snap, "findings.json", [
        {"finding_id": "F1", "rule_id": "UNUSED_PORT_ADMIN_UP", "title": "unused",
         "severity": "high", "category": "port", "device": "sw1",
         "interface": "Gi0/7", "confidence": "high", "evidence": "e",
         "recommendation": "r", "tags": ["port"]}])
    _write_nh_experiment(tmp / "network_health", "synthetic",
                         "synthetic_if_20260706T190738", n_pred=30)
    _write_engine_a(tmp / "experiments", tmp / "registry", "unsw_nb15",
                    "unsw_nb15_xgboost_20260704T150017")


def test_cli_happy_path(tmp_path: Path, monkeypatch):
    import scripts.run_correlation as cli

    _build_repo(tmp_path)
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(tmp_path))
    code = cli.main(["--engine-c-snapshot", "snap", "--engine-b-dataset",
                     "synthetic", "--engine-a-dataset", "unsw_nb15",
                     "--correlation-id", "sample_correlation"])
    assert code == 0
    out = tmp_path / "correlation" / "sample_correlation"
    assert (out / "incidents.json").is_file()
    assert (out / "correlation_report.md").is_file()
    report = (out / "correlation_report.md").read_text("utf-8")
    assert "No commands were executed" in report


def test_cli_requires_an_engine_input(tmp_path: Path, monkeypatch):
    import scripts.run_correlation as cli

    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(tmp_path))
    assert cli.main([]) == 1


def test_cli_no_usable_signals_fails(tmp_path: Path, monkeypatch):
    import scripts.run_correlation as cli

    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(tmp_path))
    # Snapshot dir/artefacts absent -> Engine C yields no signals.
    assert cli.main(["--engine-c-snapshot", "missing"]) == 1


def test_cli_missing_optional_artifacts_warns_and_continues(
        tmp_path: Path, monkeypatch):
    import scripts.run_correlation as cli

    _build_repo(tmp_path)
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(tmp_path))
    # Engine C is valid; Engine B dataset is missing -> warn but continue.
    code = cli.main(["--engine-c-snapshot", "snap", "--engine-b-dataset",
                     "does_not_exist", "--correlation-id", "partial"])
    assert code == 0
    assert (tmp_path / "correlation" / "partial" / "incidents.json").is_file()
