"""Tests for Phase 12 offline industrial-switch syslog ingestion.

Everything here is offline: synthetic log strings and small fixture files are
parsed into structured events, features and findings. No device is contacted,
no packets captured, nothing executed. No real private AAI logs are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.syslog_ingestion.artifacts import (
    ingest,
    parser_summary,
    read_input_files,
    write_run,
)
from src.syslog_ingestion.features import (
    build_windows,
    chronological_split,
    summarize_weak_labels,
)
from src.syslog_ingestion.findings import generate_findings, summarize_findings
from src.syslog_ingestion.labels import compute_labels
from src.syslog_ingestion.models import SyslogEntities, SyslogEvent, severity_label
from src.syslog_ingestion.parser import parse_line, parse_lines, parse_timestamp
from src.syslog_ingestion.preprocess import (
    PreprocessedLine,
    capture_color_hint,
    preprocess_lines,
    strip_ansi,
)

ESC = "\x1b"

CONFIG = {
    "syslog_ingestion": {
        "default_timezone": "Asia/Kolkata",
        "drop_clock_unreliable_from_features": True,
        "deduplicate_exact_repeats": True,
        "default_window_minutes": 5,
        "additional_window_minutes": [15],
    },
    "preprocessing": {"strip_ansi": True, "capture_ansi_color_hint": True,
                      "drop_pager_noise": True, "drop_prompt_echo": True},
    "thresholds": {
        "port_flaps_per_hour_warning": 3, "port_flaps_per_hour_high": 6,
        "mac_moves_warning": 20, "mac_moves_high": 100,
        "snmp_auth_fail_warning_per_5min": 5, "snmp_auth_fail_high_per_5min": 20,
        "erps_events_warning_per_hour": 3, "poe_fault_high": True,
    },
    "splitting": {"strategy": "chronological", "train_ratio": 0.70,
                  "validation_ratio": 0.15, "test_ratio": 0.15,
                  "host_holdout": None},
}


def _pl(cleaned: str, hint: str | None = None, dup: int = 1) -> PreprocessedLine:
    return PreprocessedLine(original_line=cleaned, cleaned_line=cleaned,
                            ansi_color_hint=hint, duplicate_count=dup)


def _parse(cleaned: str) -> SyslogEvent:
    return parse_line(_pl(cleaned), CONFIG)


# --------------------------------------------------------------- preprocessing
def test_strip_ansi_and_color_hint():
    line = f"{ESC}[0m{ESC}[1;33mSep  9 2025 05:08:41 HOST MPU0 %SNMP-X-3:msg{ESC}[0m"
    assert capture_color_hint(line) == "yellow"
    assert ESC not in strip_ansi(line)


def test_color_hint_cyan_and_white():
    assert capture_color_hint(f"{ESC}[1;36mx") == "cyan"
    assert capture_color_hint(f"{ESC}[1;37mx") == "white"
    assert capture_color_hint("no ansi here") is None


def test_pager_prompt_header_noise_removed():
    lines = [
        "=~=~=~= PuTTY log 2026.07.13 =~=~=~=",
        "sh logg",
        "TERM-LW-AAI-IR9-SW1#sh logging ",
        "Logging source configurations",
        "  console is enabled,level: 7(debugging)",
        "The Context of logging file:",
        "",
        f"{ESC}[1;37mSep  8 2025 16:33:15 HOST MPU0 %SHELL-5:System started{ESC}[0m",
    ]
    result = preprocess_lines(lines, CONFIG, source="t")
    assert len(result.kept) == 1
    assert "System started" in result.kept[0].cleaned_line
    reasons = {d["reason"] for d in result.dropped}
    assert {"putty_header", "empty"}.issubset(reasons)


def test_more_marker_is_a_prefix_not_a_drop():
    line = (f"---MORE---             {ESC}[1;33mSep 14 2025 05:08:51 HOST MPU0 "
            f"%SNMP-USER_AUTH_FAILED-3:User belden authentication failed.{ESC}[0m")
    result = preprocess_lines([line], CONFIG, source="t")
    assert len(result.kept) == 1
    assert "---MORE---" not in result.kept[0].cleaned_line


def test_exact_duplicate_collapse_counts():
    line = ("Jun 13 2026 18:42:00 HOST MPU0 %SNMP-COMMUNITY_AUTHOR_FAILED-3:"
            "Community public authentication failed.")
    result = preprocess_lines([line, line, line], CONFIG, source="t")
    assert len(result.kept) == 1
    assert result.kept[0].duplicate_count == 3
    assert result.duplicate_lines_collapsed == 2


# --------------------------------------------------------------- parsing
def test_timestamp_ist_conversion():
    local, utc, unreliable = parse_timestamp("Jun 13 2026 04:44:05", "Asia/Kolkata")
    assert local == "2026-06-13T04:44:05+05:30"
    assert utc == "2026-06-12T23:14:05+00:00"
    assert unreliable is False


def test_jan_1970_flagged_clock_unreliable():
    event = _parse("Jan  1 1970 00:00:15 switch %DCM-PROCESS_START-5:"
                   "Process fsd is starting...")
    assert event.clock_unreliable is True
    assert event.parse_status == "parsed"


def test_grammar_fields_and_severity():
    event = _parse("Jun 13 2026 04:44:05 TERM-LW-AAI-IR8-SW3 MEMBER-0/MPU0 "
                   "%PORTMGR-LINEPROTO_DOWN-3:Line protocol on interface "
                   "gigabitethernet2/0/6, changed state to down.")
    assert event.hostname == "TERM-LW-AAI-IR8-SW3"
    assert event.module == "MEMBER-0/MPU0"
    assert event.facility == "PORTMGR"
    assert event.mnemonic == "LINEPROTO_DOWN"
    assert event.severity_code == 3
    assert event.severity_label == "error"
    assert event.entities.interface_id == "gigabitethernet2/0/6"


def test_code_without_mnemonic_and_missing_module():
    event = _parse("Sep 21 2025 18:50:31 TERM-LW-AAI-IR9-SW1 MPU0 "
                   "%SYSMGMT-4:%clock timeZone set to IST(UTC+05:30)")
    assert event.facility == "SYSMGMT"
    # no module case:
    boot = _parse("Jan  1 1970 00:00:15 switch %DCM-PROCESS_START-5:"
                  "Process fsd is starting...")
    assert boot.module is None
    assert boot.hostname == "switch"


def test_unknown_mnemonic_is_generic_not_crash():
    event = _parse("Jun 13 2026 04:44:05 HOST MPU0 %WIDGET-FROBNICATE-6:something new")
    assert event.parse_status == "generic"
    assert event.facility == "WIDGET"
    assert event.entities.is_empty()


def test_non_grammar_line_is_failed_not_raised():
    event = _parse("this is not a syslog line at all")
    assert event.parse_status == "failed"


def test_severity_label_map():
    assert severity_label(3) == "error"
    assert severity_label(5) == "notice"
    assert severity_label(None) == "unknown"


# --------------------------------------------------------------- extractors
def test_mac_flap_extractor():
    event = _parse("Oct 28 2025 14:38:31 HOST MPU0 %FDB-MAC_ADDR_FLAPPING_VLAN-5:"
                   "Mac address 0060.2b01.362e in vlan 132 has moved from port  "
                   "to port interface tengigabitethernet0/28 for 100 times.")
    assert event.entities.mac_address == "0060.2b01.362e"
    assert event.entities.vlan_id == "132"
    assert event.entities.interface_id == "tengigabitethernet0/28"
    assert event.entities.flap_count == 100
    assert "mac_flap" in event.tags and "loop_risk" in event.tags
    assert event.engine_hints.engine_c_topology


def test_arp_extractor():
    event = _parse("Jul  3 2026 16:31:15 HOST MPU0 %ARP-MAC_CHANGE_TOO_FAST-5:"
                   "Ether address of host 192.168.100.46 change too fast, may be "
                   "duplicate IP address in the network.")
    assert event.entities.ip_address == "192.168.100.46"
    assert event.engine_hints.engine_a_intrusion
    assert "duplicate_ip_possible" in event.tags


def test_snmp_community_and_user_extractors():
    comm = _parse("Sep  9 2025 05:08:41 HOST MPU0 %SNMP-COMMUNITY_AUTHOR_FAILED-3:"
                  "Community public authentication failed.")
    assert comm.entities.community == "public"
    assert comm.engine_hints.engine_a_intrusion
    user = _parse("Dec 29 2025 14:49:02 HOST MPU0 %SNMP-USER_AUTH_FAILED-3:"
                  "User belden authentication failed.")
    assert user.entities.username == "belden"
    assert "management_auth_failed" in user.tags


def test_poe_abnormal_extractor():
    event = _parse("Apr 29 2026 19:10:01 HOST MPU0 %POE-POWER_ABNORMAL-3:"
                   "Short detected on interface gigabitethernet0/7.")
    assert event.entities.interface_id == "gigabitethernet0/7"
    assert "short_detected" in event.tags
    assert event.engine_hints.engine_c_poe


def test_erps_extractors():
    ring = _parse("Sep 21 2025 09:11:06 HOST MPU0 %ERPS-RING_STAT_CHG-4:"
                  "The state of ERPS ring 7 was changed to Protection from Idle.")
    assert ring.entities.erps_ring == "7"
    assert ring.entities.erps_state == "protection"
    port = _parse("Sep 22 2025 00:20:45 switch MPU0 %ERPS-RING_PORT_BLK-4:"
                  "The state of ERPS port tengigabitethernet0/27 was changed to "
                  "blocking on ring 7.")
    assert port.entities.interface_id == "tengigabitethernet0/27"
    assert port.entities.erps_state == "blocking"


def test_telnet_login_extractor():
    event = _parse("Sep  8 2025 16:33:15 HOST MPU0 %TELNET-LOGOUT_OK-5:"
                   "Telnet(vty0) is closed by client or timer (192.168.100.1) OK.")
    assert event.entities.login_protocol == "telnet"
    assert event.entities.ip_address == "192.168.100.1"
    assert "insecure_telnet" in event.tags


def test_lineproto_up_down_extractor():
    down = _parse("Jun 13 2026 04:44:05 HOST MPU0 %PORTMGR-LINEPROTO_DOWN-3:"
                  "Line protocol on interface gigabitethernet2/0/6, changed "
                  "state to down.")
    up = _parse("Jun 13 2026 04:44:31 HOST MPU0 %PORTMGR-LINEPROTO_UP-5:"
                "Line protocol on interface gigabitethernet2/0/6, changed state to up.")
    assert "port_flap" in down.tags and "port_flap" in up.tags
    assert down.engine_hints.engine_b_health


# --------------------------------------------------------------- features/labels
def _flap_events(host: str, iface: str, n: int) -> list[SyslogEvent]:
    events = []
    for i in range(n):
        state = "down" if i % 2 == 0 else "up"
        mnem = "LINEPROTO_DOWN" if state == "down" else "LINEPROTO_UP"
        sev = 3 if state == "down" else 5
        line = (f"Jun 13 2026 04:{i:02d}:05 {host} MPU0 %PORTMGR-{mnem}-{sev}:"
                f"Line protocol on interface {iface}, changed state to {state}.")
        events.append(_parse(line))
    return events


def test_window_aggregation_host_and_interface_scope():
    events = _flap_events("H1", "gi0/1", 6)
    rows = build_windows(events, 60, CONFIG)
    scopes = {r["scope"] for r in rows}
    assert scopes == {"host", "interface"}
    host_row = next(r for r in rows if r["scope"] == "host")
    assert host_row["port_flap_count"] == 6


def test_weak_label_thresholding():
    features = {"window_minutes": 60, "port_flap_count": 10, "mac_move_total": 0,
                "snmp_auth_fail_count": 0, "reboot_or_clock_event_count": 0,
                "power_fault_count": 0, "fan_fault_count": 0, "poe_fault_count": 0,
                "erps_event_count": 0}
    labels = compute_labels(features, CONFIG["thresholds"])
    assert labels["degradation_label"] is True
    assert labels["degradation_level"] == "high"  # 10/hr >= 6
    quiet = compute_labels({"window_minutes": 60, "port_flap_count": 1},
                           CONFIG["thresholds"])
    assert quiet["degradation_label"] is False


def test_chronological_split_manifest():
    events = _flap_events("H1", "gi0/1", 6)
    rows = build_windows(events, 5, CONFIG)
    split_rows, manifest = chronological_split(rows, CONFIG)
    assert manifest["strategy"] == "chronological"
    assert {r["split"] for r in split_rows}.issubset({"train", "validation", "test"})
    # earliest window must be train, latest must be test
    ordered = sorted(split_rows, key=lambda r: r["window_start"])
    assert ordered[0]["split"] == "train"


def test_host_holdout_split():
    cfg = {**CONFIG, "splitting": {**CONFIG["splitting"], "host_holdout": "H2"}}
    events = _flap_events("H1", "gi0/1", 4) + _flap_events("H2", "gi0/2", 4)
    rows = build_windows(events, 5, cfg)
    split_rows, manifest = chronological_split(rows, cfg)
    assert manifest["strategy"] == "host_holdout"
    h2 = [r for r in split_rows if r["hostname"] == "H2"]
    assert all(r["split"] == "test" for r in h2)


# --------------------------------------------------------------- findings
def test_findings_generation_and_cautious_language():
    events = []
    # frequent flapping (interface finding)
    events += _flap_events("H1", "gi0/1", 8)
    # mac flap loop-risk
    events.append(_parse("Oct 28 2025 14:38:31 H1 MPU0 %FDB-MAC_ADDR_FLAPPING_VLAN-5:"
                         "Mac address 0060.2b01.362e in vlan 132 has moved from port "
                         " to port interface te0/28 for 120 times."))
    # snmp burst with insecure community
    for i in range(6):
        events.append(_parse(f"Jun 13 2026 18:42:{i:02d} H1 MPU0 "
                             "%SNMP-COMMUNITY_AUTHOR_FAILED-3:Community public "
                             "authentication failed."))
    findings = generate_findings(events, CONFIG)
    rules = {f.rule_id for f in findings}
    assert {"SYS-PORT-FLAP", "SYS-MAC-FLAP", "SYS-SNMP-AUTHFAIL"}.issubset(rules)
    mac = next(f for f in findings if f.rule_id == "SYS-MAC-FLAP")
    assert mac.severity in {"high", "critical"}
    blob = " ".join((f.evidence or "") + (f.recommendation or "")
                    for f in findings).lower()
    assert any(w in blob for w in ("possible", "candidate", "evidence suggests"))

    summary = summarize_findings(findings)
    assert summary["total"] == len(findings)
    assert "note" in summary


# --------------------------------------------------------------- artifacts/CLI
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "syslog"


def test_read_input_files_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_input_files(None, str(tmp_path))


def test_full_run_writes_all_artifacts(tmp_path):
    contents = read_input_files(None, str(FIXTURE_DIR))
    run = ingest(contents, "fixture_demo", CONFIG)
    assert run.events
    out = tmp_path / "run"
    paths = write_run(run, out)
    for key in ("parsed_events_json", "parser_summary", "engine_b_windows",
                "engine_c_findings", "report"):
        assert Path(paths[key]).is_file()
    # parsed_events.json is valid and has enriched fields
    data = json.loads(Path(paths["parsed_events_json"]).read_text("utf-8"))
    assert data and "entities" in data[0] and "engine_hints" in data[0]
    summary = parser_summary(run)
    assert summary["safety"].startswith("offline")
    assert summary["clock_unreliable_events"] >= 1


def test_empty_input_handled_cleanly(tmp_path):
    empty = tmp_path / "empty.log"
    empty.write_text("\n\n=~=~ PuTTY log =~=~\n", "utf-8")
    contents = read_input_files([str(empty)], None)
    run = ingest(contents, "empty_run", CONFIG)
    assert run.events == []
    paths = write_run(run, tmp_path / "out")
    assert Path(paths["report"]).is_file()


# --------------------------------------------------------------- dashboard/stream
def test_dashboard_loader_and_streaming_source(tmp_path):
    from src.dashboard import loader
    from src.streaming.syslog_source import events_from_syslog

    contents = read_input_files(None, str(FIXTURE_DIR))
    run = ingest(contents, "lw_run", CONFIG)
    syslog_root = tmp_path / "syslog_ingestion"
    write_run(run, syslog_root / "lw_run")

    assert loader.list_syslog_runs(syslog_root) == ["lw_run"]
    assert loader.latest_syslog_run(syslog_root) == "lw_run"
    loaded = loader.load_syslog_run(syslog_root, "lw_run")
    assert loaded["available"] is True
    assert loaded["summary"]["parsed_events"] == len(run.events)

    missing = loader.load_syslog_run(syslog_root, "nope")
    assert missing["available"] is False and missing["message"]

    events = events_from_syslog(syslog_root, "lw_run")
    types = {e.event_type for e in events}
    assert "system_status" in types
    assert any(e.event_type in {"config_finding", "cyber_alert", "topology_warning"}
               for e in events)


# --------------------------------------------------------------- safety
def test_no_live_device_libraries_imported():
    """Guard: the ingestion package must not import live-device clients."""
    import src.syslog_ingestion as pkg

    pkg_dir = Path(pkg.__file__).parent
    banned = ("netmiko", "napalm", "paramiko", "pysnmp", "ncclient", "scapy",
              "telnetlib")
    for py in pkg_dir.glob("*.py"):
        text = py.read_text("utf-8")
        for lib in banned:
            assert f"import {lib}" not in text, f"{py.name} imports {lib}"
