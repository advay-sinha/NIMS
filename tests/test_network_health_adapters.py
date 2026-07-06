"""Tests for src.network_health adapters and dataset registry (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.network_health.adapters import (
    CANONICAL_COLUMNS,
    CORE_COLUMNS,
    AdapterOptions,
    canonical_csv_adapter,
    inspect_columns,
    to_canonical,
)
from src.network_health.artifacts import write_canonical_dataset
from src.network_health.dataset_registry import (
    DatasetDefinition,
    get_dataset,
    inspect_dataset,
    load_registry,
    resolve_pipeline_source,
    run_adapter,
)
from src.network_health.schema import TelemetrySchema
from src.network_health.validation import validate_telemetry


# --------------------------------------------------------------- raw fixtures


def _canonical_frame(n: int = 12) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": np.tile(times, 2),
            "device_id": np.repeat(["sw1", "sw2"], n),
            "interface_id": "eth0",
            "ifInOctets": np.cumsum(np.arange(2 * n) + 1),
            "ifOutErrors": np.arange(2 * n),
            "cpu_usage": np.linspace(10, 40, 2 * n),
            "label": 0,
        }
    )


def _snmp_raw(n: int = 10) -> pd.DataFrame:
    """SNMP-MIB-style dump: MIB-cased counters, a class column, no device col."""
    times = pd.date_range("2026-02-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "Time": times.astype(str),
            "ifIndex": ["Gi0/1"] * n,
            "ifHCInOctets": np.cumsum(np.arange(n) + 100),
            "ifHCOutOctets": np.cumsum(np.arange(n) + 50),
            "ifInErrors": np.arange(n),
            "ifOutDiscards": np.zeros(n, dtype=int),
            "class": ["normal"] * (n - 2) + ["anomaly", "anomaly"],
            "vendor_note": ["x"] * n,  # unknown column
        }
    )


def _lcore_raw(n: int = 10) -> pd.DataFrame:
    """LCORE-D-style monitoring: node/link naming, gauges, fault_state."""
    times = pd.date_range("2026-03-01", periods=n, freq="30s")
    return pd.DataFrame(
        {
            "datetime": times.astype(str),
            "node_name": ["core-a"] * n,
            "link_id": ["l1"] * n,
            "latency": np.linspace(1.0, 5.0, n),
            "packet_loss": np.zeros(n),
            "cpu_util": np.linspace(20, 60, n),
            "fault_state": ["none"] * (n - 1) + ["congestion"],
        }
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _registry_config(datasets: dict) -> dict:
    return {"network_health": {"datasets": datasets}}


# ------------------------------------------------------------------- adapters


def test_canonical_passthrough_unchanged() -> None:
    frame = _canonical_frame()
    result = canonical_csv_adapter(
        frame, AdapterOptions(dataset_type="canonical_csv")
    )
    assert list(result.frame.columns) == list(frame.columns)
    assert len(result.frame) == len(frame)
    assert result.report.n_devices == 2


def test_snmp_alias_mapping_and_label() -> None:
    options = AdapterOptions(
        dataset_type="snmp_mib_2016",
        device_id="snmp_device_0",
        label_map={"normal": 0, "anomaly": 1},
    )
    result = to_canonical(_snmp_raw(), options)
    frame = result.frame
    # MIB-cased raw columns mapped onto canonical counters.
    assert result.report.mapped_columns["ifInOctets"] == "ifHCInOctets"
    assert result.report.mapped_columns["ifOutOctets"] == "ifHCOutOctets"
    assert result.report.mapped_columns["timestamp"] == "Time"
    assert result.report.mapped_columns["interface_id"] == "ifIndex"
    # Missing device column -> generated constant.
    assert (frame["device_id"] == "snmp_device_0").all()
    assert "device_id" in result.report.generated_columns
    # Label mapping applied (normal->0, anomaly->1).
    assert result.report.label_mapping_applied
    assert frame["label"].tolist()[-2:] == [1, 1]
    assert frame["label"].tolist()[0] == 0
    # Unknown column dropped (preserve_unknown defaults to False).
    assert "vendor_note" in result.report.dropped_columns
    assert "vendor_note" not in frame.columns


def test_preserve_unknown_columns() -> None:
    options = AdapterOptions(
        dataset_type="snmp_mib_2016", device_id="d0", preserve_unknown=True
    )
    result = to_canonical(_snmp_raw(), options)
    assert "vendor_note" in result.frame.columns
    assert "vendor_note" in result.report.preserved_columns


def test_lcore_mapping_and_fault_label() -> None:
    options = AdapterOptions(
        dataset_type="lcore_d", device_id="core_node_0", interface_id="link_0"
    )
    result = to_canonical(_lcore_raw(), options)
    frame = result.frame
    assert result.report.mapped_columns["device_id"] == "node_name"
    assert result.report.mapped_columns["interface_id"] == "link_id"
    assert result.report.mapped_columns["latency_ms"] == "latency"
    assert result.report.mapped_columns["cpu_usage"] == "cpu_util"
    # fault_state is preserved as the canonical fault_type text column.
    assert "fault_type" in frame.columns
    assert frame["fault_type"].tolist()[-1] == "congestion"


def test_missing_timestamp_raises() -> None:
    frame = _snmp_raw().drop(columns=["Time"])
    with pytest.raises(ValueError, match="timestamp"):
        to_canonical(frame, AdapterOptions(dataset_type="snmp_mib_2016"))


def test_status_word_coercion() -> None:
    frame = _lcore_raw()
    frame["oper_status"] = ["up"] * (len(frame) - 1) + ["down"]
    result = to_canonical(
        frame, AdapterOptions(dataset_type="lcore_d", device_id="d")
    )
    assert result.frame["ifOperStatus"].tolist()[-1] == 2
    assert result.frame["ifOperStatus"].tolist()[0] == 1


def test_config_alias_override() -> None:
    frame = _snmp_raw().rename(columns={"ifHCInOctets": "weird_in_bytes"})
    options = AdapterOptions(
        dataset_type="snmp_mib_2016",
        aliases={"ifInOctets": ["weird_in_bytes"]},
        device_id="d0",
    )
    result = to_canonical(frame, options)
    assert result.report.mapped_columns["ifInOctets"] == "weird_in_bytes"


# ------------------------------------------------------------------- registry


def test_registry_load_and_definition(tmp_path: Path) -> None:
    src = _write_csv(_snmp_raw(), tmp_path / "raw" / "snmp.csv")
    config = _registry_config(
        {
            "snmp_mib_2016": {
                "type": "snmp_mib_2016",
                "source_path": str(src),
                "output_path": str(tmp_path / "out" / "snmp.csv"),
                "options": {"device_id": "d0", "label_map": {"normal": 0,
                                                             "anomaly": 1}},
            }
        }
    )
    registry = load_registry(config)
    assert set(registry) == {"snmp_mib_2016"}
    definition = get_dataset(config, "snmp_mib_2016")
    assert definition.dataset_type == "snmp_mib_2016"
    assert definition.options.label_map == {"normal": 0, "anomaly": 1}


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        get_dataset(_registry_config({}), "does_not_exist")


def test_unknown_adapter_type_raises() -> None:
    config = _registry_config(
        {"bad": {"type": "mystery", "source_path": "x.csv"}}
    )
    with pytest.raises(ValueError, match="Unknown adapter type"):
        load_registry(config)


def test_run_adapter_missing_source_raises(tmp_path: Path) -> None:
    config = _registry_config(
        {
            "snmp_mib_2016": {
                "type": "snmp_mib_2016",
                "source_path": str(tmp_path / "nope"),
                "output_path": str(tmp_path / "out.csv"),
            }
        }
    )
    definition = get_dataset(config, "snmp_mib_2016")
    with pytest.raises(FileNotFoundError, match="not found"):
        run_adapter(definition)


def test_run_adapter_on_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_csv(_snmp_raw(), raw_dir / "part1.csv")
    definition = DatasetDefinition.from_config(
        "snmp_mib_2016",
        {
            "type": "snmp_mib_2016",
            "source_path": str(raw_dir),
            "output_path": str(tmp_path / "out.csv"),
            "options": {"device_id": "d0", "label_map": {"normal": 0,
                                                         "anomaly": 1}},
        },
    )
    result = run_adapter(definition)
    assert set(CORE_COLUMNS).issubset(result.frame.columns)


def test_resolve_pipeline_source_prefers_output(tmp_path: Path) -> None:
    config = _registry_config(
        {
            "snmp_mib_2016": {
                "type": "snmp_mib_2016",
                "source_path": str(tmp_path / "raw"),
                "output_path": str(tmp_path / "out" / "snmp.csv"),
            },
            "synthetic": {
                "type": "canonical_csv",
                "source_path": str(tmp_path / "syn.csv"),
            },
        }
    )
    resolved, dataset_id = resolve_pipeline_source(config, "snmp_mib_2016")
    assert resolved == tmp_path / "out" / "snmp.csv"
    assert dataset_id == "snmp_mib_2016"
    # canonical_csv with no output_path falls back to source_path.
    resolved2, _ = resolve_pipeline_source(config, "synthetic")
    assert resolved2 == tmp_path / "syn.csv"


# --------------------------------------------------------------------- report


def test_adapter_report_persistence(tmp_path: Path) -> None:
    options = AdapterOptions(
        dataset_type="snmp_mib_2016", device_id="d0",
        label_map={"normal": 0, "anomaly": 1},
    )
    result = to_canonical(_snmp_raw(), options)
    out_csv = tmp_path / "processed" / "snmp.csv"
    paths = write_canonical_dataset(
        result, out_csv, tmp_path / "outputs", "snmp_mib_2016"
    )
    assert paths["csv"].is_file()
    assert paths["json"].is_file() and paths["markdown"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "snmp_mib_2016"
    assert payload["dataset_type"] == "snmp_mib_2016"
    assert "ifInOctets" in payload["mapped_columns"]
    # The written CSV round-trips and carries canonical columns.
    reloaded = pd.read_csv(paths["csv"])
    assert set(CORE_COLUMNS).issubset(reloaded.columns)


# --------------------------------------------------------------------- inspect


def test_inspect_columns_infers_roles() -> None:
    report = inspect_columns(_lcore_raw(), "lcore_d")
    assert "datetime" in report["timestamp_candidates"]
    assert "fault_state" in report["label_candidates"]
    assert "latency" in report["metric_candidates"]
    assert report["recognised_mapping"]["latency_ms"] == "latency"


def test_inspect_dataset_lists_files(tmp_path: Path) -> None:
    src = _write_csv(_lcore_raw(), tmp_path / "raw" / "lcore.csv")
    definition = DatasetDefinition.from_config(
        "lcore_d", {"type": "lcore_d", "source_path": str(src.parent),
                    "output_path": str(tmp_path / "out.csv")},
    )
    report = inspect_dataset(definition)
    assert report["dataset_id"] == "lcore_d"
    assert any("lcore.csv" in f for f in report["files_found"])


# ---------------------------------------------------- pipeline compatibility


def test_output_feeds_validation(tmp_path: Path) -> None:
    """A converted SNMP dataset must pass canonical-schema validation."""
    options = AdapterOptions(
        dataset_type="snmp_mib_2016", device_id="d0",
        label_map={"normal": 0, "anomaly": 1},
    )
    result = to_canonical(_snmp_raw(n=20), options)
    out_csv = tmp_path / "snmp.csv"
    result.frame.to_csv(out_csv, index=False)

    reloaded = pd.read_csv(out_csv)
    schema = TelemetrySchema.from_config(
        {
            "timestamp_column": "timestamp",
            "device_column": "device_id",
            "interface_column": "interface_id",
            "label_column": "label",
            "required_columns": list(CORE_COLUMNS)
            + ["ifInOctets", "ifOutOctets", "ifInErrors"],
            "counter_columns": ["ifInOctets", "ifOutOctets", "ifInErrors"],
            "gauge_columns": [],
            "status_columns": [],
        }
    )
    report = validate_telemetry(reloaded, schema, "snmp_mib_2016")
    assert report.passed
    assert report.n_devices == 1


def test_canonical_columns_contract() -> None:
    # Core + optional columns are all present in the canonical contract.
    for column in CORE_COLUMNS:
        assert column in CANONICAL_COLUMNS
    assert CANONICAL_COLUMNS[:3] == CORE_COLUMNS


# --------------------------------------------------------------------- CLI


class _FakeCtx:
    def __init__(self, config: dict, network_health_dir: Path) -> None:
        self.config = config

        class _P:
            pass

        self.paths = _P()
        self.paths.network_health_dir = network_health_dir


def _wire_cli(monkeypatch, config: dict, nh_dir: Path):
    import scripts.prepare_network_health_dataset as cli

    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(config, nh_dir))
    return cli


def test_cli_convert(tmp_path: Path, monkeypatch, caplog) -> None:
    src = _write_csv(_snmp_raw(), tmp_path / "raw" / "snmp.csv")
    out = tmp_path / "out" / "snmp.csv"
    config = _registry_config(
        {
            "snmp_mib_2016": {
                "type": "snmp_mib_2016",
                "source_path": str(src),
                "output_path": str(out),
                "options": {"device_id": "d0", "label_map": {"normal": 0,
                                                             "anomaly": 1}},
            }
        }
    )
    cli = _wire_cli(monkeypatch, config, tmp_path / "outputs")
    with caplog.at_level("INFO"):
        assert cli.main(["--dataset", "snmp_mib_2016"]) == 0
    assert out.is_file()
    assert (tmp_path / "outputs" / "adapters" / "snmp_mib_2016"
            / "adapter_report.json").is_file()


def test_cli_synthetic_uses_config_output_path(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Default synthetic conversion works with no --output (uses config path)."""
    src = _write_csv(_canonical_frame(), tmp_path / "samples" / "synthetic.csv")
    # Output path under a directory that does NOT exist yet -> must be created.
    out = tmp_path / "processed" / "network_health" / "synthetic.csv"
    config = _registry_config(
        {
            "synthetic": {
                "type": "canonical_csv",
                "source_path": str(src),
                "output_path": str(out),
            }
        }
    )
    assert not out.parent.exists()
    cli = _wire_cli(monkeypatch, config, tmp_path / "outputs")
    with caplog.at_level("INFO"):
        assert cli.main(["--dataset", "synthetic"]) == 0
    assert out.is_file()  # output directory was created automatically
    reloaded = pd.read_csv(out)
    assert set(CORE_COLUMNS).issubset(reloaded.columns)
    assert len(reloaded) == len(_canonical_frame())


def test_cli_inspect(tmp_path: Path, monkeypatch, caplog) -> None:
    src = _write_csv(_lcore_raw(), tmp_path / "raw" / "lcore.csv")
    config = _registry_config(
        {"lcore_d": {"type": "lcore_d", "source_path": str(src.parent),
                     "output_path": str(tmp_path / "out.csv")}}
    )
    cli = _wire_cli(monkeypatch, config, tmp_path / "outputs")
    with caplog.at_level("INFO"):
        assert cli.main(["--dataset", "lcore_d", "--inspect"]) == 0
    assert "timestamp_candidates" in caplog.text
    # Inspect mode must not write any canonical CSV.
    assert not (tmp_path / "out.csv").exists()


def test_cli_missing_source_returns_1(tmp_path: Path, monkeypatch, caplog) -> None:
    config = _registry_config(
        {"snmp_mib_2016": {"type": "snmp_mib_2016",
                           "source_path": str(tmp_path / "nope"),
                           "output_path": str(tmp_path / "out.csv")}}
    )
    cli = _wire_cli(monkeypatch, config, tmp_path / "outputs")
    with caplog.at_level("ERROR"):
        assert cli.main(["--dataset", "snmp_mib_2016"]) == 1
    assert "not found" in caplog.text


def test_cli_unknown_dataset_returns_1(tmp_path: Path, monkeypatch, caplog) -> None:
    cli = _wire_cli(monkeypatch, _registry_config({}), tmp_path / "outputs")
    with caplog.at_level("ERROR"):
        assert cli.main(["--dataset", "ghost"]) == 1
