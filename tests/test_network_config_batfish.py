"""Tests for src.network_config Phase 8 optional Batfish adapter (no service)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.network_config.batfish_adapter import (
    BatfishUnavailableError,
    load_batfish_config,
    run_batfish_validation,
)
from src.network_config.batfish_artifacts import write_batfish


# ---------------------------------------------------------------- fake session


class _Frameable:
    def __init__(self, rows):
        self._rows = rows

    def frame(self):
        return self._rows


class _Answerable:
    def __init__(self, rows):
        self._rows = rows

    def answer(self):
        return _Frameable(self._rows)


class _Question:
    def __init__(self, rows=None, error=None):
        self._rows = rows or []
        self._error = error

    def __call__(self):
        if self._error:
            raise RuntimeError(self._error)
        return _Answerable(self._rows)


class _Q:
    def __init__(self, mapping):
        for attr, question in mapping.items():
            setattr(self, attr, question)


class _FakeSession:
    def __init__(self, mapping):
        self.q = _Q(mapping)
        self.network = None
        self.snapshot = None

    def set_network(self, name):
        self.network = name

    def init_snapshot(self, path, name=None, overwrite=False):
        self.snapshot = name


def _mapping(parse_rows=None, undefined_rows=None, node_rows=None,
             iface_rows=None, l3_rows=None, failing=None):
    mapping = {
        "fileParseStatus": _Question(parse_rows if parse_rows is not None
                                     else [{"File_Name": "core.cfg",
                                            "Status": "PASSED",
                                            "Nodes": ["core"]}]),
        "nodeProperties": _Question(node_rows or [{"Node": "core"}]),
        "interfaceProperties": _Question(iface_rows or [{"Interface": "Gi0/1"}]),
        "layer3Edges": _Question(l3_rows or []),
        "undefinedReferences": _Question(undefined_rows or []),
    }
    if failing:
        mapping[failing] = _Question(error="boom")
    return mapping


def _factory(mapping):
    return lambda config: _FakeSession(mapping)


def _enabled_cfg(**questions):
    q = {k: True for k in ("parse_status", "node_properties",
                           "interface_properties", "l3_edges",
                           "undefined_references")}
    q.update(questions)
    return {"global": {"enabled": True}, "connection": {}, "questions": q}


# ----------------------------------------------------------------- disabled


def test_config_disabled_skips_cleanly(tmp_path: Path):
    result = run_batfish_validation("s", {"global": {"enabled": False}}, tmp_path)
    assert result.status == "disabled"
    assert result.findings == ()


def test_unavailable_skips_when_factory_raises(tmp_path: Path):
    def factory(_config):
        raise BatfishUnavailableError("pybatfish is not installed")

    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=factory)
    assert result.status == "skipped"
    assert "pybatfish" in result.reason.lower()


# --------------------------------------------------------------- lazy import


def test_pybatfish_imported_lazily(monkeypatch, tmp_path: Path):
    # Force the lazy `from pybatfish...` import to fail; the run must degrade to
    # a skipped result rather than raise.
    monkeypatch.setitem(sys.modules, "pybatfish", None)
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path)
    assert result.status == "skipped"
    assert "pybatfish" in result.reason.lower()


def test_no_top_level_pybatfish_import():
    source = Path("src/network_config/batfish_adapter.py").read_text("utf-8")
    # A top-level import would start at column 0; the real import is indented
    # inside _get_session so Engine C loads without pybatfish present.
    assert "\nimport pybatfish" not in source
    assert "\nfrom pybatfish" not in source
    assert "    from pybatfish.client.session import Session" in source


def test_missing_snapshot_dir_skips(tmp_path: Path):
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path / "nope",
                                    session_factory=_factory(_mapping()))
    assert result.status == "skipped"
    assert "snapshot directory not found" in result.reason.lower()


# --------------------------------------------------------------- questions


def test_mocked_successful_parse_status(tmp_path: Path):
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=_factory(_mapping()))
    assert result.status == "success"
    assert result.parse_status_summary["passed"] == 1
    assert result.node_count == 1
    assert result.interface_count == 1
    parse = next(t for t in result.tables if t.name == "parse_status")
    assert parse.status == "success"


def test_mocked_query_failure_recorded(tmp_path: Path):
    mapping = _mapping(failing="layer3Edges")
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=_factory(mapping))
    # Overall still success (other questions worked); the failed query recorded.
    assert result.status == "success"
    failed = next(t for t in result.tables if t.name == "l3_edges")
    assert failed.status == "failed"
    assert failed.error == "boom"


def test_disabled_question_marked_skipped(tmp_path: Path):
    result = run_batfish_validation("s", _enabled_cfg(l3_edges=False), tmp_path,
                                    session_factory=_factory(_mapping()))
    l3 = next(t for t in result.tables if t.name == "l3_edges")
    assert l3.status == "skipped"


# ----------------------------------------------------------------- findings


def test_finding_from_parse_failure(tmp_path: Path):
    mapping = _mapping(parse_rows=[{"File_Name": "bad.cfg", "Status": "FAILED",
                                    "Nodes": ["bad"]}])
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=_factory(mapping))
    fails = [f for f in result.findings if f.severity == "high"]
    assert fails and "failed to parse" in fails[0].title.lower()
    assert fails[0].source == "batfish"


def test_finding_from_undefined_reference(tmp_path: Path):
    mapping = _mapping(undefined_rows=[{"Structure_Type": "route-map",
                                        "Ref_Name": "RM-IN",
                                        "File_Name": "core.cfg"}])
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=_factory(mapping))
    refs = [f for f in result.findings
            if "undefined reference" in f.title.lower()]
    assert refs and refs[0].severity == "medium"
    assert result.undefined_reference_count == 1


def test_partial_parse_is_medium(tmp_path: Path):
    mapping = _mapping(parse_rows=[{"File_Name": "p.cfg",
                                    "Status": "PARTIALLY_PARSED",
                                    "Nodes": ["p"]}])
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=_factory(mapping))
    assert any(f.severity == "medium" and "partially parsed" in f.title.lower()
               for f in result.findings)
    assert result.parse_status_summary["partially_parsed"] == 1


# ----------------------------------------------------------------- artifacts


def test_artifact_persistence(tmp_path: Path):
    mapping = _mapping(
        undefined_rows=[{"Structure_Type": "acl", "Ref_Name": "X",
                         "File_Name": "core.cfg"}])
    result = run_batfish_validation("s", _enabled_cfg(), tmp_path,
                                    session_factory=_factory(mapping))
    paths = write_batfish(result, tmp_path)
    for key in ("summary", "parse_status", "node_properties",
                "interface_properties", "l3_edges", "undefined_references",
                "findings_json", "findings_csv"):
        assert paths[key].is_file()
    summary = json.loads(paths["summary"].read_text("utf-8"))
    assert summary["status"] == "success"
    assert "no device access" in summary["safety_note"].lower()
    assert summary["undefined_reference_count"] == 1


def test_config_loading():
    cfg = load_batfish_config("configs/batfish.yaml")
    assert cfg["global"]["enabled"] is False
    assert cfg["global"]["fail_if_unavailable"] is False


def test_load_config_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_batfish_config(tmp_path / "nope.yaml")


# ------------------------------------------------------------------- CLI


class _FakeCtx:
    def __init__(self, network_config_dir: Path):
        self.config = {"network_config": {}}

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = network_config_dir


def _write_cfg(path: Path, enabled: bool) -> Path:
    path.write_text(
        f"global:\n  enabled: {str(enabled).lower()}\n"
        "  fail_if_unavailable: false\n"
        "connection: {}\n"
        f"inputs:\n  snapshot_root: {path.parent / 'snap'}\n"
        "questions:\n  parse_status: true\n",
        encoding="utf-8")
    return path


def test_cli_disabled_exits_clean(tmp_path: Path, monkeypatch):
    import scripts.run_batfish_validation as cli

    cfg = _write_cfg(tmp_path / "batfish.yaml", enabled=False)
    root = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    code = cli.main(["--snapshot-id", "s", "--batfish-config", str(cfg)])
    assert code == 0
    assert not (root / "s" / "batfish").exists()


def test_cli_missing_dependency_fails_when_strict(tmp_path: Path, monkeypatch):
    import scripts.run_batfish_validation as cli

    cfg = _write_cfg(tmp_path / "batfish.yaml", enabled=False)
    root = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    # No pybatfish -> lazy import fails -> skipped; --fail-if-unavailable => exit 1.
    monkeypatch.setitem(sys.modules, "pybatfish", None)
    code = cli.main(["--snapshot-id", "s", "--batfish-config", str(cfg),
                     "--fail-if-unavailable"])
    assert code == 1


def test_cli_missing_dependency_ok_when_not_strict(tmp_path: Path, monkeypatch):
    import scripts.run_batfish_validation as cli

    cfg = tmp_path / "batfish.yaml"
    cfg.write_text(
        "global:\n  enabled: true\n  fail_if_unavailable: false\n"
        "connection: {}\n"
        f"inputs:\n  snapshot_root: {tmp_path / 'snap'}\n"
        "questions:\n  parse_status: true\n", encoding="utf-8")
    root = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    monkeypatch.setitem(sys.modules, "pybatfish", None)
    code = cli.main(["--snapshot-id", "s", "--batfish-config", str(cfg)])
    assert code == 0


# --------------------------------------------------- intelligence integration


def test_intelligence_report_includes_batfish(tmp_path: Path):
    from src.network_config.intelligence import (
        build_intelligence,
        load_snapshot_artifacts,
    )
    from src.network_config.intelligence_artifacts import write_intelligence

    snap = tmp_path / "snap"
    (snap / "batfish").mkdir(parents=True)
    (snap / "inventory.json").write_text(
        json.dumps({"snapshot_id": "s", "devices": []}), "utf-8")
    (snap / "batfish" / "batfish_summary.json").write_text(json.dumps({
        "snapshot_id": "s", "status": "success", "node_count": 3,
        "interface_count": 10, "l3_edge_count": 4,
        "undefined_reference_count": 1,
        "parse_status_summary": {"passed": 3, "failed": 0,
                                 "partially_parsed": 0},
        "findings": [{"severity": "medium", "category": "config",
                      "title": "Undefined reference in configuration",
                      "device": "core", "evidence": "acl X undefined"}],
        "safety_note": "External configuration validation only; no device "
                       "access and no commands were executed.",
    }), "utf-8")

    artifacts = load_snapshot_artifacts(snap)
    assert artifacts.batfish is not None
    intel = build_intelligence(artifacts)
    report = write_intelligence(intel, snap)["report"].read_text("utf-8")
    assert "## Batfish Validation (external, optional)" in report
    assert "no device access" in report.lower()
