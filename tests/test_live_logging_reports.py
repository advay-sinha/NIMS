"""End-to-end offline ingestion + report + secret-safety tests (Phase 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live_logging import reports
from src.live_logging.scheduler import LiveLogger, run_and_report

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "datasets" / "samples" / "live_logging"

# Secrets embedded in the sample inputs that must never reach persisted outputs.
SECRETS = [
    "s3cr3t-community",
    "sophos-secret-token-must-be-redacted",
    "eyJhbGciOi",
]


def _sophos_cfg():
    return {
        "sophos": {
            "central_api": {
                "enabled": True, "mode": "offline",
                "offline_sample_path": str(SAMPLES / "sophos" / "central_api_alerts.json"),
            },
            "firewall_syslog": {
                "enabled": True, "mode": "offline",
                "offline_sample_path": str(SAMPLES / "sophos" / "firewall_syslog.log"),
            },
        }
    }


def _hirschmann_cfg():
    return {
        "hirschmann": {
            "snmp_polling": {
                "enabled": True, "mode": "offline",
                "offline_sample_path": str(SAMPLES / "hirschmann" / "snmp_metrics.json"),
            },
            "traps": {
                "enabled": True, "mode": "offline",
                "offline_sample_path": str(SAMPLES / "hirschmann" / "traps.log"),
            },
            "config_snapshots": {
                "enabled": True, "mode": "offline",
                "snapshot_dir": str(SAMPLES / "hirschmann" / "config_snapshots"),
            },
            "thresholds": {"in_errors_high": 100, "utilization_high": 90,
                           "temperature_high_c": 70, "temperature_critical_c": 80},
        }
    }


def _live_cfg(tmp_path):
    return {
        "mode": "offline", "output_dir": str(tmp_path), "redact_secrets": True,
        "routing": {"sophos_api": "cyber", "sophos_syslog": "cyber",
                    "hirschmann_snmp": "network_health", "hirschmann_traps": "network_health",
                    "hirschmann_config": "network_config"},
        "checkpoints": {"path": str(tmp_path / "checkpoints")},
        "retry": {"max_attempts": 2},
        "safety": {"read_only": True},
    }


def test_run_once_produces_events_and_routes(tmp_path):
    status, out = run_and_report(
        _live_cfg(tmp_path), _sophos_cfg(), _hirschmann_cfg(), output_dir=tmp_path
    )
    assert status.total_events > 0
    assert status.healthy is True
    # All three engine targets are exercised by the samples.
    assert set(status.events_by_engine) >= {"cyber", "network_health", "network_config"}
    # Status + report files were written.
    assert (Path(out) / reports.STATUS_FILENAME).is_file()
    assert (Path(out) / reports.REPORT_FILENAME).is_file()


def test_no_secrets_in_persisted_outputs(tmp_path):
    run_and_report(_live_cfg(tmp_path), _sophos_cfg(), _hirschmann_cfg(), output_dir=tmp_path)
    for name in ("events.jsonl", "raw_events.jsonl", reports.REPORT_FILENAME,
                 reports.STATUS_FILENAME):
        text = (tmp_path / name).read_text(encoding="utf-8")
        for secret in SECRETS:
            assert secret not in text, f"{secret!r} leaked into {name}"


def test_source_failure_isolated(tmp_path):
    # Point one source at a missing snapshot dir but keep others valid; the run
    # must still succeed for the healthy sources (no crash, isolated failure).
    hirschmann = _hirschmann_cfg()
    hirschmann["hirschmann"]["config_snapshots"]["snapshot_dir"] = str(tmp_path / "missing")
    logger = LiveLogger(_live_cfg(tmp_path), _sophos_cfg(), hirschmann, output_dir=tmp_path)
    status = logger.run_once()
    # Missing dir yields zero config events but does not fail other sources.
    assert status.total_events > 0
    assert any(s.source == "sophos_api" and s.status == "ok" for s in status.sources)


def test_disabled_source_reported(tmp_path):
    sophos = _sophos_cfg()
    sophos["sophos"]["central_api"]["enabled"] = False
    logger = LiveLogger(_live_cfg(tmp_path), sophos, _hirschmann_cfg(), output_dir=tmp_path)
    status = logger.run_once()
    api = next(s for s in status.sources if s.source == "sophos_api")
    assert api.status == "disabled"
