"""Engine B time-window feature generation from parsed syslog events.

Purpose
-------
Aggregate structured :class:`SyslogEvent` objects into fixed-width time windows
suitable for Engine B anomaly detection (Isolation Forest / LSTM-AE), attach
weak threshold labels and a chronological (never random) train/val/test split.

This is an *adapter-style* output: it produces Engine-B-shaped feature tables on
disk, it does not replace or call the existing Engine B pipeline.

Scope
-----
Two aggregation scopes are emitted into one table (``scope`` column):
    * ``host``      — one row per (hostname, window): every event for the host.
    * ``interface`` — one row per (hostname, interface, window): interface events.
Boot-clock (``clock_unreliable``) and grammar-``failed`` events are excluded from
features by default (config ``drop_clock_unreliable_from_features``).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.syslog_ingestion.labels import compute_labels
from src.syslog_ingestion.models import FAILED, SyslogEvent
from src.syslog_ingestion.windowing import parse_iso, window_bounds

logger = logging.getLogger(__name__)

HOST_SCOPE = "host"
INTERFACE_SCOPE = "interface"

_SEVERITY_COLUMNS = (
    "error", "warning", "notice", "critical", "info", "debug", "unknown",
)


@dataclass
class _WindowAcc:
    """Mutable accumulator for one (scope, host, interface, window) bucket."""

    window_start: str
    window_end: str
    window_minutes: int
    hostname: str
    interface_id: str
    scope: str
    total_events: int = 0
    weighted_event_count: int = 0
    lineproto_up_count: int = 0
    lineproto_down_count: int = 0
    mac_flap_count: int = 0
    mac_move_total: int = 0
    erps_event_count: int = 0
    poe_fault_count: int = 0
    poe_power_churn_count: int = 0
    snmp_auth_fail_count: int = 0
    arp_mac_change_count: int = 0
    reboot_or_clock_event_count: int = 0
    power_fault_count: int = 0
    fan_fault_count: int = 0
    severity: Counter = field(default_factory=Counter)
    facility: Counter = field(default_factory=Counter)
    mnemonic: Counter = field(default_factory=Counter)
    snmp_source_ips: set = field(default_factory=set)

    def add(self, event: SyslogEvent) -> None:
        """Fold one event (weighted by its duplicate_count) into this window."""
        weight = max(1, int(event.duplicate_count))
        self.total_events += 1
        self.weighted_event_count += weight
        self.severity[event.severity_label] += weight
        if event.facility:
            self.facility[event.facility] += weight
        if event.mnemonic:
            self.mnemonic[event.mnemonic] += weight

        tags = set(event.tags)
        mnem = event.mnemonic or ""
        is_link = ("link_state" in tags or mnem.startswith("LINEPROTO")
                   or mnem.startswith("INTERFACE"))
        if is_link:
            if "down" in mnem.lower() or "link_down" in tags:
                self.lineproto_down_count += weight
            elif "up" in mnem.lower() or "link_up" in tags:
                self.lineproto_up_count += weight
        if "mac_flap" in tags:
            self.mac_flap_count += weight
            if event.entities.flap_count:
                self.mac_move_total += int(event.entities.flap_count)
        if "erps_churn" in tags:
            self.erps_event_count += weight
        if "poe_fault" in tags:
            self.poe_fault_count += weight
        if "poe_power_churn" in tags:
            self.poe_power_churn_count += weight
        if "snmp_auth_failed" in tags or "snmp_user_auth_failed" in tags:
            self.snmp_auth_fail_count += weight
            if event.entities.ip_address:
                self.snmp_source_ips.add(event.entities.ip_address)
        if "arp_instability" in tags:
            self.arp_mac_change_count += weight
        if "reboot" in tags or "clock_change" in tags:
            self.reboot_or_clock_event_count += weight
        if "power_fault" in tags:
            self.power_fault_count += weight
        if "fan_fault" in tags:
            self.fan_fault_count += weight

    @property
    def port_flap_count(self) -> int:
        return self.lineproto_up_count + self.lineproto_down_count

    def to_row(self, thresholds: Mapping[str, Any] | None) -> dict[str, Any]:
        """Materialise the accumulator into a flat feature+label row."""
        row: dict[str, Any] = {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_minutes": self.window_minutes,
            "hostname": self.hostname,
            "interface_id": self.interface_id,
            "scope": self.scope,
            "total_events": self.total_events,
            "weighted_event_count": self.weighted_event_count,
            "lineproto_up_count": self.lineproto_up_count,
            "lineproto_down_count": self.lineproto_down_count,
            "port_flap_count": self.port_flap_count,
            "mac_flap_count": self.mac_flap_count,
            "mac_move_total": self.mac_move_total,
            "erps_event_count": self.erps_event_count,
            "poe_fault_count": self.poe_fault_count,
            "poe_power_churn_count": self.poe_power_churn_count,
            "snmp_auth_fail_count": self.snmp_auth_fail_count,
            "unique_snmp_source_ips": len(self.snmp_source_ips),
            "arp_mac_change_count": self.arp_mac_change_count,
            "reboot_or_clock_event_count": self.reboot_or_clock_event_count,
            "power_fault_count": self.power_fault_count,
            "fan_fault_count": self.fan_fault_count,
        }
        for label in _SEVERITY_COLUMNS:
            row[f"severity_{label}"] = int(self.severity.get(label, 0))
        row["facility_counts"] = json.dumps(dict(self.facility), sort_keys=True)
        row["mnemonic_counts"] = json.dumps(dict(self.mnemonic), sort_keys=True)
        row.update(compute_labels(row, thresholds))
        return row


def _feature_eligible(event: SyslogEvent, drop_clock_unreliable: bool) -> bool:
    """Whether an event should contribute to feature windows."""
    if event.parse_status == FAILED:
        return False
    if event.timestamp is None:
        return False
    if drop_clock_unreliable and event.clock_unreliable:
        return False
    return True


def build_windows(
    events: Iterable[SyslogEvent],
    window_minutes: int,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate events into host- and interface-scoped feature windows.

    Rows are returned sorted chronologically then by host/scope/interface so the
    output is deterministic across runs.
    """
    config = config or {}
    ingest = config.get("syslog_ingestion", {}) or {}
    thresholds = config.get("thresholds", {}) or {}
    drop_clock = bool(ingest.get("drop_clock_unreliable_from_features", True))

    buckets: dict[tuple, _WindowAcc] = {}
    for event in events:
        if not _feature_eligible(event, drop_clock):
            continue
        moment = parse_iso(event.timestamp)
        if moment is None:
            continue
        start, end = window_bounds(moment, window_minutes)
        host = event.hostname or "unknown"

        host_key = (HOST_SCOPE, host, "", start)
        acc = buckets.get(host_key)
        if acc is None:
            acc = _WindowAcc(start, end, window_minutes, host, "", HOST_SCOPE)
            buckets[host_key] = acc
        acc.add(event)

        iface = event.entities.interface_id
        if iface:
            iface_key = (INTERFACE_SCOPE, host, iface, start)
            acc_i = buckets.get(iface_key)
            if acc_i is None:
                acc_i = _WindowAcc(start, end, window_minutes, host, iface,
                                   INTERFACE_SCOPE)
                buckets[iface_key] = acc_i
            acc_i.add(event)

    rows = [acc.to_row(thresholds) for acc in buckets.values()]
    rows.sort(key=lambda r: (r["window_start"], r["hostname"], r["scope"],
                             r["interface_id"]))
    return rows


# --------------------------------------------------------------- summaries
def summarize_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate top-level statistics over the feature windows."""
    if not rows:
        return {"window_count": 0, "hosts": [], "scopes": {}}
    hosts = sorted({r["hostname"] for r in rows})
    scopes = Counter(r["scope"] for r in rows)
    starts = [r["window_start"] for r in rows]
    numeric_totals = {
        key: int(sum(r.get(key, 0) for r in rows))
        for key in (
            "total_events", "weighted_event_count", "port_flap_count",
            "mac_flap_count", "mac_move_total", "erps_event_count",
            "poe_fault_count", "snmp_auth_fail_count", "arp_mac_change_count",
        )
    }
    return {
        "window_count": len(rows),
        "hosts": hosts,
        "scopes": dict(scopes),
        "time_range": {"first_window": min(starts), "last_window": max(starts)},
        "totals": numeric_totals,
    }


def summarize_weak_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count how many windows each weak label fired in."""
    label_keys = (
        "degradation_label", "loop_risk_label", "snmp_attack_weak_label",
        "device_instability_label", "poe_fault_label", "erps_churn_label",
    )
    counts = {key: int(sum(1 for r in rows if r.get(key))) for key in label_keys}
    return {
        "window_count": len(rows),
        "positive_windows": counts,
        "note": ("Weak labels are threshold-derived heuristics, not "
                 "ground-truth incidents."),
    }


# --------------------------------------------------------------- splitting
def chronological_split(
    rows: list[dict[str, Any]], config: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign a chronological (never random) train/val/test ``split`` per row.

    When ``splitting.host_holdout`` names a host, that host's rows all go to
    ``test`` and the remaining rows are split by time.
    """
    config = config or {}
    split_cfg = config.get("splitting", {}) or {}
    train_ratio = float(split_cfg.get("train_ratio", 0.70))
    val_ratio = float(split_cfg.get("validation_ratio", 0.15))
    host_holdout = split_cfg.get("host_holdout")

    ordered = sorted(rows, key=lambda r: (r["window_start"], r["hostname"],
                                          r["scope"], r["interface_id"]))

    if host_holdout:
        held = [r for r in ordered if r["hostname"] == host_holdout]
        rest = [r for r in ordered if r["hostname"] != host_holdout]
        # time-split the remaining rows into train/val only; held-out host = test
        n_train = int(len(rest) * train_ratio)
        for i, row in enumerate(rest):
            row["split"] = "train" if i < n_train else "validation"
        for row in held:
            row["split"] = "test"
        splits = rest + held
    else:
        n = len(ordered)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        for i, row in enumerate(ordered):
            if i < n_train:
                row["split"] = "train"
            elif i < n_train + n_val:
                row["split"] = "validation"
            else:
                row["split"] = "test"
        splits = ordered

    manifest = _split_manifest(splits, split_cfg, host_holdout)
    return splits, manifest


def _split_manifest(rows: list[dict[str, Any]], split_cfg: Mapping[str, Any],
                    host_holdout: Any) -> dict[str, Any]:
    """Build the split manifest describing counts and time boundaries."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    boundaries = {}
    for split, split_rows in by_split.items():
        starts = [r["window_start"] for r in split_rows]
        boundaries[split] = {
            "count": len(split_rows),
            "first_window": min(starts) if starts else None,
            "last_window": max(starts) if starts else None,
        }
    return {
        "strategy": "chronological" if not host_holdout else "host_holdout",
        "train_ratio": float(split_cfg.get("train_ratio", 0.70)),
        "validation_ratio": float(split_cfg.get("validation_ratio", 0.15)),
        "test_ratio": float(split_cfg.get("test_ratio", 0.15)),
        "host_holdout": host_holdout,
        "splits": boundaries,
        "note": "Chronological split only; no random shuffling (no leakage).",
    }
