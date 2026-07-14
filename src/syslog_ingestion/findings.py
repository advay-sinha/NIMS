"""Engine C rule-style findings derived from parsed syslog events.

Purpose
-------
Turn structured :class:`SyslogEvent` streams into cautious, operator-facing
findings: frequent port flapping, MAC flapping / possible L2 loops, SNMP
authorization-failure bursts, ARP duplicate-IP indicators, PoE faults, ERPS
topology churn, insecure telnet management and device (power/fan/reboot)
instability.

These are *detection-only* observations. They never execute a change, and their
language stays deliberately cautious ("possible", "candidate", "observed",
"evidence suggests") — the source logs carry no ground-truth attack labels.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from src.syslog_ingestion.models import SyslogEvent
from src.syslog_ingestion.windowing import floor_to_bin, parse_iso

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
INSECURE_COMMUNITIES = {"public", "private"}


@dataclass(frozen=True)
class SyslogFinding:
    """One cautious, detection-only finding from the syslog stream."""

    finding_id: str
    rule_id: str
    title: str
    severity: str
    category: str                 # interface/loop/security/poe/topology/device
    device: str | None = None
    interface: str | None = None
    vlan: str | None = None
    evidence: str | None = None
    recommendation: str | None = None
    confidence: str = "medium"
    source: str = "syslog"
    event_count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None
    tags: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, len(SEVERITY_ORDER))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["tags"] = ";".join(self.tags)
        row["details"] = ""  # keep CSV flat; full detail lives in the JSON
        return row


def _finding_id(rule_id: str, *parts: Any) -> str:
    key = "|".join([rule_id, *("" if p is None else str(p) for p in parts)])
    return f"SYSF-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}"


def _threshold(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return (config.get("thresholds", {}) or {}).get(key, default)


def _time_bounds(events: list[SyslogEvent]) -> tuple[str | None, str | None]:
    stamps = sorted(e.timestamp for e in events if e.timestamp)
    return (stamps[0], stamps[-1]) if stamps else (None, None)


def _weighted(events: list[SyslogEvent]) -> int:
    return sum(max(1, int(e.duplicate_count)) for e in events)


def _peak_per_bin(events: list[SyslogEvent], window_minutes: int) -> int:
    """Max weighted event count in any ``window_minutes`` bin."""
    bins: dict[Any, int] = defaultdict(int)
    for e in events:
        moment = parse_iso(e.timestamp)
        if moment is None:
            continue
        bins[floor_to_bin(moment, window_minutes)] += max(1, int(e.duplicate_count))
    return max(bins.values()) if bins else 0


# --------------------------------------------------------------- rules
def _rule_port_flap(by_iface: dict[tuple, list[SyslogEvent]],
                    config: Mapping[str, Any]) -> list[SyslogFinding]:
    warn = float(_threshold(config, "port_flaps_per_hour_warning", 3))
    high = float(_threshold(config, "port_flaps_per_hour_high", 6))
    findings: list[SyslogFinding] = []
    for (host, iface), events in by_iface.items():
        flaps = [e for e in events if "port_flap" in e.tags]
        if not flaps:
            continue
        peak = _peak_per_bin(flaps, 60)
        if peak < warn:
            continue
        severity = "high" if peak >= high else "medium"
        first, last = _time_bounds(flaps)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-PORT-FLAP", host, iface),
            rule_id="SYS-PORT-FLAP",
            title=f"Frequent port flapping observed on {iface}",
            severity=severity, category="interface", device=host, interface=iface,
            evidence=(f"{_weighted(flaps)} link state changes observed; peak "
                      f"{peak} in a single hour (warning>={warn:g}/h)."),
            recommendation=("Evidence suggests an unstable link or endpoint; "
                            "inspect cabling/SFP and the connected device."),
            confidence="high" if peak >= high else "medium",
            event_count=_weighted(flaps), first_seen=first, last_seen=last,
            tags=("port_flap", "link_state"),
            details={"peak_flaps_per_hour": peak}))
    return findings


def _rule_mac_flap(events: list[SyslogEvent],
                   config: Mapping[str, Any]) -> list[SyslogFinding]:
    warn = int(_threshold(config, "mac_moves_warning", 20))
    high = int(_threshold(config, "mac_moves_high", 100))
    by_key: dict[tuple, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if "mac_flap" in e.tags:
            by_key[(e.hostname, e.entities.mac_address, e.entities.vlan_id)].append(e)
    findings: list[SyslogFinding] = []
    for (host, mac, vlan), evs in by_key.items():
        moves = max((e.entities.flap_count or 0) for e in evs)
        if moves < warn:
            continue
        repeated = len(evs) > 1
        severity = "critical" if (moves >= high and repeated) else (
            "high" if moves >= high else "medium")
        first, last = _time_bounds(evs)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-MAC-FLAP", host, mac, vlan),
            rule_id="SYS-MAC-FLAP",
            title=f"MAC flapping / possible L2 loop for {mac} in VLAN {vlan}",
            severity=severity, category="loop", device=host, vlan=vlan,
            interface=evs[-1].entities.interface_id,
            evidence=(f"MAC {mac} moved up to {moves} times in VLAN {vlan} "
                      f"across {len(evs)} report(s); evidence suggests a "
                      "possible Layer-2 loop or duplicate host."),
            recommendation=("Investigate for a bridging loop or duplicated MAC; "
                            "verify STP/ERPS state on the involved ports."),
            confidence="high", event_count=_weighted(evs),
            first_seen=first, last_seen=last, tags=("mac_flap", "loop_risk"),
            details={"max_moves": moves, "reports": len(evs)}))
    return findings


def _rule_snmp(events: list[SyslogEvent],
               config: Mapping[str, Any]) -> list[SyslogFinding]:
    warn = float(_threshold(config, "snmp_auth_fail_warning_per_5min", 5))
    high = float(_threshold(config, "snmp_auth_fail_high_per_5min", 20))
    by_host: dict[str, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if "snmp_auth_failed" in e.tags or "snmp_user_auth_failed" in e.tags:
            by_host[e.hostname or "unknown"].append(e)
    findings: list[SyslogFinding] = []
    for host, evs in by_host.items():
        total = _weighted(evs)
        peak = _peak_per_bin(evs, 5)
        communities = sorted({e.entities.community for e in evs
                              if e.entities.community})
        insecure = sorted(set(communities) & INSECURE_COMMUNITIES)
        severity = "high" if peak >= high else "medium" if peak >= warn else "low"
        if insecure and severity == "low":
            severity = "medium"  # default community targeted -> at least medium
        first, last = _time_bounds(evs)
        evidence = (f"{total} SNMP authorization failures observed on {host}; "
                    f"peak {peak} in a 5-minute window.")
        if insecure:
            evidence += (f" Default/insecure community string(s) targeted: "
                         f"{', '.join(insecure)}.")
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-SNMP-AUTHFAIL", host),
            rule_id="SYS-SNMP-AUTHFAIL",
            title=f"SNMP authorization failures on {host} (possible recon)",
            severity=severity, category="security", device=host,
            evidence=evidence,
            recommendation=("Evidence suggests possible SNMP reconnaissance or "
                            "brute-force. Restrict SNMP by ACL, disable unused "
                            "communities and prefer SNMPv3."),
            confidence="medium", event_count=total,
            first_seen=first, last_seen=last,
            tags=("snmp_auth_failed", "recon_possible"),
            details={"peak_per_5min": peak, "communities": communities,
                     "insecure_communities": insecure}))
    return findings


def _rule_arp(events: list[SyslogEvent],
              config: Mapping[str, Any]) -> list[SyslogFinding]:
    by_key: dict[tuple, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if "arp_instability" in e.tags:
            by_key[(e.hostname, e.entities.ip_address)].append(e)
    findings: list[SyslogFinding] = []
    for (host, ip), evs in by_key.items():
        count = _weighted(evs)
        severity = "high" if count >= 5 else "medium"
        first, last = _time_bounds(evs)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-ARP-FAST", host, ip),
            rule_id="SYS-ARP-FAST",
            title=f"Rapid ARP/MAC change for host {ip}",
            severity=severity, category="security", device=host,
            evidence=(f"Host {ip} changed hardware address rapidly {count} "
                      "time(s); evidence suggests a possible duplicate IP or "
                      "ARP spoofing candidate."),
            recommendation=("Check for a duplicate IP assignment or unauthorized "
                            "device; confirm the legitimate owner of the address."),
            confidence="medium", event_count=count,
            first_seen=first, last_seen=last,
            tags=("arp_instability", "duplicate_ip_possible"),
            details={"ip_address": ip}))
    return findings


def _rule_poe(events: list[SyslogEvent],
              config: Mapping[str, Any]) -> list[SyslogFinding]:
    by_iface: dict[tuple, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if "poe_fault" in e.tags:
            by_iface[(e.hostname, e.entities.interface_id)].append(e)
    findings: list[SyslogFinding] = []
    for (host, iface), evs in by_iface.items():
        short = any("short_detected" in e.tags for e in evs)
        first, last = _time_bounds(evs)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-POE-FAULT", host, iface),
            rule_id="SYS-POE-FAULT",
            title=f"PoE abnormal/short observed on {iface}",
            severity="high", category="poe", device=host, interface=iface,
            evidence=(f"{_weighted(evs)} PoE fault event(s) observed on {iface}"
                      + (" (short circuit detected)." if short else ".")),
            recommendation=("Inspect the powered device and cabling for a short "
                            "or fault before re-enabling PoE."),
            confidence="high", event_count=_weighted(evs),
            first_seen=first, last_seen=last,
            tags=("poe_fault",) + (("short_detected",) if short else ()),
            details={"short_detected": short}))
    return findings


def _rule_erps(events: list[SyslogEvent],
               config: Mapping[str, Any]) -> list[SyslogFinding]:
    warn = float(_threshold(config, "erps_events_warning_per_hour", 3))
    by_ring: dict[tuple, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if "erps_churn" in e.tags:
            by_ring[(e.hostname, e.entities.erps_ring)].append(e)
    findings: list[SyslogFinding] = []
    for (host, ring), evs in by_ring.items():
        peak = _peak_per_bin(evs, 60)
        if peak < warn:
            continue
        severity = "high" if peak >= warn * 2 else "medium"
        first, last = _time_bounds(evs)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-ERPS-CHURN", host, ring),
            rule_id="SYS-ERPS-CHURN",
            title=f"ERPS ring {ring} protection churn / topology instability",
            severity=severity, category="topology", device=host,
            evidence=(f"{_weighted(evs)} ERPS state changes on ring {ring}; "
                      f"peak {peak} within one hour."),
            recommendation=("Evidence suggests ring instability; check the ring "
                            "links and the connected switches for flaps."),
            confidence="medium", event_count=_weighted(evs),
            first_seen=first, last_seen=last,
            tags=("erps_churn", "topology_instability"),
            details={"ring": ring, "peak_per_hour": peak}))
    return findings


def _rule_telnet(events: list[SyslogEvent],
                 config: Mapping[str, Any]) -> list[SyslogFinding]:
    by_host: dict[str, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if "insecure_telnet" in e.tags:
            by_host[e.hostname or "unknown"].append(e)
    findings: list[SyslogFinding] = []
    for host, evs in by_host.items():
        first, last = _time_bounds(evs)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-TELNET", host),
            rule_id="SYS-TELNET",
            title=f"Insecure telnet management observed on {host}",
            severity="medium", category="security", device=host,
            evidence=(f"{_weighted(evs)} telnet management session event(s) "
                      "observed; telnet is a cleartext protocol."),
            recommendation=("Disable telnet and use SSH for device management."),
            confidence="high", event_count=_weighted(evs),
            first_seen=first, last_seen=last,
            tags=("management_access", "insecure_telnet"), details={}))
    return findings


def _rule_device(events: list[SyslogEvent],
                 config: Mapping[str, Any]) -> list[SyslogFinding]:
    by_host: dict[str, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if {"reboot", "power_fault", "fan_fault", "clock_change"} & set(e.tags):
            by_host[e.hostname or "unknown"].append(e)
    findings: list[SyslogFinding] = []
    for host, evs in by_host.items():
        faults = [e for e in evs if {"power_fault", "fan_fault"} & set(e.tags)]
        reboots = [e for e in evs if "reboot" in e.tags]
        severity = "high" if faults else "medium"
        first, last = _time_bounds(evs)
        findings.append(SyslogFinding(
            finding_id=_finding_id("SYS-DEVICE", host),
            rule_id="SYS-DEVICE",
            title=f"Device health / stability events on {host}",
            severity=severity, category="device", device=host,
            evidence=(f"Observed {len(reboots)} reboot/process and "
                      f"{len(faults)} power/fan fault event(s)."),
            recommendation=("Review environmental (power/fan) status and the "
                            "reboot cause; correlate with clock changes."),
            confidence="medium", event_count=_weighted(evs),
            first_seen=first, last_seen=last, tags=("device_health",),
            details={"reboot_events": len(reboots), "fault_events": len(faults)}))
    return findings


def generate_findings(
    events: Iterable[SyslogEvent], config: Mapping[str, Any] | None = None
) -> list[SyslogFinding]:
    """Run every syslog rule and return findings sorted by severity."""
    config = config or {}
    events = list(events)

    by_iface: dict[tuple, list[SyslogEvent]] = defaultdict(list)
    for e in events:
        if e.entities.interface_id:
            by_iface[(e.hostname, e.entities.interface_id)].append(e)

    findings: list[SyslogFinding] = []
    findings += _rule_port_flap(by_iface, config)
    findings += _rule_mac_flap(events, config)
    findings += _rule_snmp(events, config)
    findings += _rule_arp(events, config)
    findings += _rule_poe(events, config)
    findings += _rule_erps(events, config)
    findings += _rule_telnet(events, config)
    findings += _rule_device(events, config)

    findings.sort(key=lambda f: (f.severity_rank, f.rule_id, f.device or ""))
    return findings


def summarize_findings(findings: list[SyslogFinding]) -> dict[str, Any]:
    """Summary counts by severity, category and rule for the rule summary file."""
    by_severity: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    by_rule: dict[str, int] = defaultdict(int)
    for f in findings:
        by_severity[f.severity] += 1
        by_category[f.category] += 1
        by_rule[f.rule_id] += 1
    return {
        "total": len(findings),
        "by_severity": dict(by_severity),
        "by_category": dict(by_category),
        "by_rule": dict(by_rule),
        "note": ("Detection-only findings with cautious language; no attack "
                 "ground truth and no remediation executed."),
    }
