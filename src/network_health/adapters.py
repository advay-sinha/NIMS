"""Network-health dataset adapters.

Purpose
-------
Convert raw/public network-health datasets (SNMP-MIB counter dumps, LCORE-D
core-network monitoring exports, or already-canonical CSVs) into the single
canonical telemetry schema consumed by the Engine B pipeline
(:mod:`src.network_health.validation`, ``preprocessing`` and ``baseline``).

Design
------
One config-driven mapping engine (:func:`to_canonical`) does the work; the
per-type adapters only supply default column aliases. No vendor column name is
hardcoded in pipeline logic — aliases live in defaults here and may be
extended per dataset from configuration. Adapters never fabricate telemetry:
a missing timestamp column is an error, not an invented column.

Outputs
-------
An :class:`AdapterResult` (the canonical frame plus an :class:`AdapterReport`),
persisted as a canonical CSV and ``adapter_report.{json,md}`` by the
``prepare_network_health_dataset`` script.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# Canonical telemetry columns (CLAUDE.md Engine B: config-driven, never
# hardcoded downstream — this is the *contract*, defined once).
CORE_COLUMNS: tuple[str, ...] = ("timestamp", "device_id", "interface_id")

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "ifInOctets", "ifOutOctets", "ifInErrors", "ifOutErrors",
    "ifInDiscards", "ifOutDiscards", "ifOperStatus", "ifAdminStatus",
    "cpu_usage", "memory_usage", "latency_ms", "packet_loss",
    "jitter_ms", "bandwidth_utilization", "label", "anomaly_type",
    "fault_type",
)

CANONICAL_COLUMNS: tuple[str, ...] = CORE_COLUMNS + OPTIONAL_COLUMNS

# Numeric canonical columns (coerced to numeric during conversion).
_NUMERIC_COLUMNS: frozenset[str] = frozenset({
    "ifInOctets", "ifOutOctets", "ifInErrors", "ifOutErrors",
    "ifInDiscards", "ifOutDiscards", "cpu_usage", "memory_usage",
    "latency_ms", "packet_loss", "jitter_ms", "bandwidth_utilization",
})
_STATUS_COLUMNS: frozenset[str] = frozenset({"ifOperStatus", "ifAdminStatus"})
_STATUS_WORD_MAP: dict[str, int] = {
    "up": 1, "down": 2, "operational": 1, "active": 1, "inactive": 2,
    "enabled": 1, "disabled": 2, "true": 1, "false": 2,
}

# Default raw-name aliases per canonical column. Keys are canonical columns;
# values are *normalised* alias tokens (see :func:`_normalise`). A raw column
# whose normalised name equals the canonical column's own normalised name is
# always matched, so only genuine synonyms are listed here.
_DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("time", "ts", "datetime", "date_time", "record_time",
                  "timestamp_utc", "collection_time", "epoch"),
    "device_id": ("device", "node", "node_id", "host", "hostname", "switch",
                  "router", "agent", "agent_ip", "src_device", "source_ip"),
    "interface_id": ("interface", "ifindex", "if_name", "ifname", "port",
                     "link", "iface", "if_descr", "ifdescr"),
    "ifInOctets": ("in_octets", "inoctets", "bytes_in", "rx_bytes",
                   "ifhcinoctets", "octets_in"),
    "ifOutOctets": ("out_octets", "outoctets", "bytes_out", "tx_bytes",
                    "ifhcoutoctets", "octets_out"),
    "ifInErrors": ("in_errors", "inerrors", "rx_errors", "errors_in"),
    "ifOutErrors": ("out_errors", "outerrors", "tx_errors", "errors_out"),
    "ifInDiscards": ("in_discards", "indiscards", "rx_discards",
                     "discards_in", "in_drops"),
    "ifOutDiscards": ("out_discards", "outdiscards", "tx_discards",
                      "discards_out", "out_drops"),
    "ifOperStatus": ("oper_status", "operstatus", "link_status",
                     "if_oper_status", "operational_status"),
    "ifAdminStatus": ("admin_status", "adminstatus", "if_admin_status"),
    "cpu_usage": ("cpu", "cpu_util", "cpu_percent", "cpu_load", "sysload",
                  "cpu_utilization"),
    "memory_usage": ("mem_usage", "memory", "mem", "mem_percent",
                     "ram_usage", "memory_utilization"),
    "latency_ms": ("latency", "delay", "delay_ms", "rtt", "rtt_ms"),
    "packet_loss": ("pkt_loss", "loss", "packet_loss_pct", "loss_rate",
                    "packetloss"),
    "jitter_ms": ("jitter", "jitter_msec"),
    "bandwidth_utilization": ("bandwidth", "bw_util", "utilization",
                              "link_utilization", "bw_utilization"),
    "label": ("anomaly", "is_anomaly", "target", "class", "y", "attack"),
    "anomaly_type": ("anomalytype", "anomaly_class"),
    "fault_type": ("faulttype", "fault", "fault_label", "attack_type",
                   "attack_cat"),
}

# Per-type supplemental aliases (merged on top of the defaults).
_TYPE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "snmp_mib_2016": {
        # SNMP-MIB 2016 counter dumps expose high-capacity MIB counters and a
        # trailing class column.
        "ifInOctets": ("ifhcinoctets", "ifinucastpkts"),
        "ifOutOctets": ("ifhcoutoctets", "ifoutucastpkts"),
        "label": ("class",),
    },
    "lcore_d": {
        # LCORE-D core-network monitoring uses node/link naming and a fault
        # state column.
        "device_id": ("node_name", "core_node"),
        "interface_id": ("link", "link_id", "edge"),
        "fault_type": ("fault_state", "failure_type"),
    },
}


def _normalise(name: str) -> str:
    """Normalise a raw column name for alias matching (lower snake, no punct)."""
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return token.strip("_")


@dataclass(frozen=True)
class AdapterOptions:
    """Per-dataset conversion options resolved from configuration."""

    dataset_type: str
    aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    device_id: str | None = None
    interface_id: str | None = None
    label_map: Mapping[str, int] = field(default_factory=dict)
    preserve_unknown: bool = False
    timestamp_format: str | None = None


@dataclass
class AdapterReport:
    """Structured, JSON-serialisable summary of one conversion."""

    dataset_type: str
    n_rows: int = 0
    n_devices: int = 0
    n_interfaces: int = 0
    source_columns: list[str] = field(default_factory=list)
    mapped_columns: dict[str, str] = field(default_factory=dict)
    generated_columns: list[str] = field(default_factory=list)
    preserved_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    label_mapping_applied: bool = False
    timestamp_span: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a non-fatal conversion warning."""
        self.warnings.append(message)
        logger.warning("%s", message)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "dataset_type": self.dataset_type,
            "n_rows": self.n_rows,
            "n_devices": self.n_devices,
            "n_interfaces": self.n_interfaces,
            "n_source_columns": len(self.source_columns),
            "source_columns": self.source_columns,
            "mapped_columns": self.mapped_columns,
            "generated_columns": self.generated_columns,
            "preserved_columns": self.preserved_columns,
            "dropped_columns": self.dropped_columns,
            "label_mapping_applied": self.label_mapping_applied,
            "timestamp_span": self.timestamp_span,
            "n_warnings": len(self.warnings),
            "warnings": self.warnings,
        }


@dataclass
class AdapterResult:
    """A converted canonical telemetry frame and its report."""

    frame: Any  # pandas.DataFrame
    report: AdapterReport


def _resolve_alias_index(options: AdapterOptions) -> dict[str, str]:
    """Map every known normalised alias token to its canonical column.

    Precedence (later wins on collision): built-in defaults, per-type
    supplements, then config-provided aliases.
    """
    index: dict[str, str] = {}
    layers: list[Mapping[str, Sequence[str]]] = [
        _DEFAULT_ALIASES,
        _TYPE_ALIASES.get(options.dataset_type, {}),
        options.aliases,
    ]
    for canonical in CANONICAL_COLUMNS:
        index[_normalise(canonical)] = canonical  # the column's own name
    for layer in layers:
        for canonical, alias_list in layer.items():
            for alias in alias_list:
                index[_normalise(alias)] = canonical
    return index


def _match_columns(
    columns: Sequence[str], options: AdapterOptions
) -> tuple[dict[str, str], list[str]]:
    """Return ``{canonical: raw_column}`` and the list of unmatched raw columns.

    The first raw column that matches a canonical target wins; later matches
    for the same canonical are left unmatched (and reported).
    """
    index = _resolve_alias_index(options)
    mapped: dict[str, str] = {}
    unmatched: list[str] = []
    for raw in columns:
        canonical = index.get(_normalise(raw))
        if canonical is not None and canonical not in mapped:
            mapped[canonical] = raw
        else:
            unmatched.append(raw)
    return mapped, unmatched


def _coerce_status(series: Any) -> Any:
    """Coerce an interface status column to numeric SNMP-style codes."""
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    words = series.astype(str).str.strip().str.lower().map(_STATUS_WORD_MAP)
    return numeric.fillna(words)


def to_canonical(frame: Any, options: AdapterOptions) -> AdapterResult:
    """Convert a raw telemetry frame to the canonical schema.

    Parameters
    ----------
    frame:
        Raw telemetry rows (already loaded from CSV/CSVs).
    options:
        Resolved conversion options (aliases, id defaults, label map, ...).

    Returns
    -------
    AdapterResult

    Raises
    ------
    ValueError
        When no timestamp column can be identified — telemetry timestamps are
        never fabricated.
    """
    import pandas as pd

    report = AdapterReport(
        dataset_type=options.dataset_type,
        n_rows=int(len(frame)),
        source_columns=[str(c) for c in frame.columns],
    )
    mapped, unmatched = _match_columns(list(frame.columns), options)

    if "timestamp" not in mapped:
        raise ValueError(
            "No timestamp column found (looked for aliases of 'timestamp'); "
            "refusing to fabricate one. Add a 'timestamp' alias in the dataset "
            "config."
        )

    out = pd.DataFrame(index=frame.index)

    # Identity: timestamp (parsed), device_id and interface_id (generated to a
    # configured constant when the raw dataset lacks them).
    timestamps = pd.to_datetime(
        frame[mapped["timestamp"]], errors="coerce",
        format=options.timestamp_format or "mixed",
    )
    n_bad = int(timestamps.isna().sum())
    if n_bad:
        report.warn(
            f"{n_bad} timestamp value(s) were unparseable and left as NaT "
            "(validation will flag them)."
        )
    out["timestamp"] = timestamps

    for column, default in (
        ("device_id", options.device_id),
        ("interface_id", options.interface_id),
    ):
        if column in mapped:
            out[column] = frame[mapped[column]].astype(str)
        else:
            constant = default or f"{column.split('_')[0]}_0"
            out[column] = constant
            report.generated_columns.append(column)
            report.warn(
                f"No '{column}' column found; generated constant "
                f"'{constant}'."
            )

    # Optional canonical payload columns.
    for column in OPTIONAL_COLUMNS:
        if column not in mapped:
            continue
        raw_series = frame[mapped[column]]
        if column in _NUMERIC_COLUMNS:
            out[column] = pd.to_numeric(raw_series, errors="coerce")
        elif column in _STATUS_COLUMNS:
            out[column] = _coerce_status(raw_series)
        elif column == "label":
            out[column] = _map_label(raw_series, options.label_map, report)
        else:
            out[column] = raw_series

    report.mapped_columns = dict(mapped)

    # Unknown raw columns: preserve verbatim (never clobbering canonical
    # names) only when configured; otherwise drop and report.
    if options.preserve_unknown:
        for raw in unmatched:
            target = raw if raw not in out.columns else f"raw_{raw}"
            out[target] = frame[raw]
            report.preserved_columns.append(target)
    else:
        report.dropped_columns = list(unmatched)

    out = out.sort_values(["device_id", "interface_id", "timestamp"]).reset_index(
        drop=True
    )
    report.n_devices = int(out["device_id"].nunique())
    report.n_interfaces = int(
        out.groupby("device_id")["interface_id"].nunique().sum()
    )
    valid_ts = timestamps.dropna()
    if not valid_ts.empty:
        report.timestamp_span = {
            "start": str(valid_ts.min()),
            "end": str(valid_ts.max()),
        }
    logger.info(
        "Converted %d row(s) to canonical schema (%d device(s), %d "
        "interface(s), %d mapped column(s)).",
        report.n_rows, report.n_devices, report.n_interfaces, len(mapped),
    )
    return AdapterResult(frame=out, report=report)


def _map_label(series: Any, label_map: Mapping[str, int], report: AdapterReport) -> Any:
    """Map a raw label column to a 0/1 anomaly label."""
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and not label_map:
        return numeric.astype(int)
    if label_map:
        mapped = series.astype(str).str.strip().map(
            {str(k): int(v) for k, v in label_map.items()}
        )
        report.label_mapping_applied = True
        n_unmapped = int(mapped.isna().sum())
        if n_unmapped:
            report.warn(
                f"{n_unmapped} label value(s) not in label_map; left as NaN."
            )
        return mapped
    # Non-numeric with no map: treat any non-empty, non-"normal" token as 1.
    normal = {"normal", "benign", "0", "none", "healthy", "ok", ""}
    coerced = (~series.astype(str).str.strip().str.lower().isin(normal)).astype(int)
    report.warn(
        "Label column was non-numeric with no label_map; inferred 1 for any "
        "value outside the benign set."
    )
    report.label_mapping_applied = True
    return coerced


# --------------------------------------------------------------- adapters ----
# Each type-specific adapter differs only in its default alias layer (applied
# via ``options.dataset_type``); the mapping engine is shared.


def canonical_csv_adapter(frame: Any, options: AdapterOptions) -> AdapterResult:
    """Pass a frame that is already in (or aliased to) the canonical schema."""
    missing = [c for c in CORE_COLUMNS if c not in frame.columns]
    if not missing:
        # Already canonical: validate core columns, return unchanged.
        report = AdapterReport(
            dataset_type="canonical_csv",
            n_rows=int(len(frame)),
            source_columns=[str(c) for c in frame.columns],
            mapped_columns={c: c for c in frame.columns if c in CANONICAL_COLUMNS},
        )
        report.n_devices = int(frame["device_id"].nunique())
        report.n_interfaces = int(
            frame.groupby("device_id")["interface_id"].nunique().sum()
        )
        return AdapterResult(frame=frame.reset_index(drop=True), report=report)
    # Core columns present under aliases: run the general engine.
    return to_canonical(frame, options)


def snmp_mib_adapter(frame: Any, options: AdapterOptions) -> AdapterResult:
    """Adapter for SNMP-MIB 2016-style counter telemetry."""
    return to_canonical(frame, options)


def lcore_d_adapter(frame: Any, options: AdapterOptions) -> AdapterResult:
    """Adapter for LCORE-D-style core-network monitoring telemetry."""
    return to_canonical(frame, options)


ADAPTERS: dict[str, Any] = {
    "canonical_csv": canonical_csv_adapter,
    "snmp_mib_2016": snmp_mib_adapter,
    "lcore_d": lcore_d_adapter,
}


def inspect_columns(frame: Any, dataset_type: str) -> dict[str, Any]:
    """Infer timestamp/label/metric candidates from a raw frame (schema probe).

    Used by ``--inspect`` when a dataset's exact column scheme is unknown. Reads
    nothing; classifies the columns of an already-loaded sample frame.
    """
    import pandas as pd

    options = AdapterOptions(dataset_type=dataset_type)
    index = _resolve_alias_index(options)
    columns = [str(c) for c in frame.columns]

    timestamp_candidates: list[str] = []
    label_candidates: list[str] = []
    metric_candidates: list[str] = []
    for raw in columns:
        canonical = index.get(_normalise(raw))
        numeric = pd.to_numeric(frame[raw], errors="coerce")
        is_numeric = bool(numeric.notna().any())
        # Numeric columns are never timestamp candidates (a float column parses
        # as epoch-nanoseconds and would masquerade as a datetime).
        parsed = pd.to_datetime(frame[raw], errors="coerce", format="mixed")
        is_time = canonical == "timestamp" or (
            not is_numeric and len(frame) > 0 and parsed.notna().mean() > 0.9
        )
        is_label = canonical in {"label", "anomaly_type", "fault_type"}
        if is_time:
            timestamp_candidates.append(raw)
        elif is_label:
            label_candidates.append(raw)
        elif is_numeric:
            metric_candidates.append(raw)
    return {
        "dataset_type": dataset_type,
        "n_rows_sampled": int(len(frame)),
        "columns": columns,
        "timestamp_candidates": timestamp_candidates,
        "label_candidates": label_candidates,
        "metric_candidates": metric_candidates,
        "recognised_mapping": {
            index[_normalise(c)]: c for c in columns if _normalise(c) in index
        },
    }
