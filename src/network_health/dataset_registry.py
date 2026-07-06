"""Network-health dataset registry.

Purpose
-------
Turn the config-driven ``network_health.datasets`` block into typed dataset
definitions and dispatch each to the correct adapter
(:mod:`src.network_health.adapters`). This keeps dataset identity, source and
output locations, and per-dataset conversion options out of Python and in
configuration (CLAUDE.md: never hardcode dataset names or paths).

A registry entry looks like::

    datasets:
      snmp_mib_2016:
        type: snmp_mib_2016
        source_path: datasets/raw/snmp_mib_2016
        output_path: datasets/processed/network_health/snmp_mib_2016.csv
        options:
          device_id: switch_0
          label_map: {normal: 0, attack: 1}
          preserve_unknown: false
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.network_health.adapters import (
    ADAPTERS,
    AdapterOptions,
    AdapterResult,
    inspect_columns,
)
from src.network_health.loader import load_telemetry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetDefinition:
    """One registered network-health dataset."""

    dataset_id: str
    dataset_type: str
    source_path: Path
    output_path: Path | None
    options: AdapterOptions

    @classmethod
    def from_config(
        cls, dataset_id: str, entry: Mapping[str, Any]
    ) -> "DatasetDefinition":
        """Build a definition from one ``datasets.<id>`` config block."""
        dataset_type = str(entry.get("type", "canonical_csv"))
        if dataset_type not in ADAPTERS:
            raise ValueError(
                f"Unknown adapter type '{dataset_type}' for dataset "
                f"'{dataset_id}'. Known types: {sorted(ADAPTERS)}."
            )
        source = entry.get("source_path")
        if not source:
            raise ValueError(
                f"Dataset '{dataset_id}' has no 'source_path'."
            )
        output = entry.get("output_path")
        raw_options = dict(entry.get("options") or {})
        options = AdapterOptions(
            dataset_type=dataset_type,
            aliases={
                str(k): [str(v) for v in vs]
                for k, vs in dict(raw_options.get("aliases") or {}).items()
            },
            device_id=raw_options.get("device_id"),
            interface_id=raw_options.get("interface_id"),
            label_map={
                str(k): int(v)
                for k, v in dict(raw_options.get("label_map") or {}).items()
            },
            preserve_unknown=bool(raw_options.get("preserve_unknown", False)),
            timestamp_format=raw_options.get("timestamp_format"),
        )
        return cls(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            source_path=Path(source),
            output_path=Path(output) if output else None,
            options=options,
        )


def load_registry(config: Mapping[str, Any]) -> dict[str, DatasetDefinition]:
    """Build the dataset registry from the effective configuration.

    Reads ``network_health.datasets``. When that block is absent, a single
    ``synthetic`` entry is synthesised from the top-level
    ``network_health.source_path`` so the default pipeline still resolves.
    """
    nh = dict(config.get("network_health") or {})
    entries = dict(nh.get("datasets") or {})
    if not entries and nh.get("source_path"):
        entries = {
            str(nh.get("dataset_id", "synthetic")): {
                "type": "canonical_csv",
                "source_path": nh["source_path"],
            }
        }
    return {
        dataset_id: DatasetDefinition.from_config(dataset_id, entry)
        for dataset_id, entry in entries.items()
    }


def get_dataset(
    config: Mapping[str, Any], dataset_id: str
) -> DatasetDefinition:
    """Resolve one dataset definition by id (raises ``KeyError`` if unknown)."""
    registry = load_registry(config)
    if dataset_id not in registry:
        raise KeyError(
            f"Dataset '{dataset_id}' is not registered. Known datasets: "
            f"{sorted(registry)}."
        )
    return registry[dataset_id]


def resolve_pipeline_source(
    config: Mapping[str, Any], dataset_id: str
) -> tuple[Path, str]:
    """Resolve the canonical CSV a registered dataset feeds to the pipeline.

    Returns the converted ``output_path`` when the dataset has one (it is the
    canonical CSV the adapter produces), otherwise the ``source_path`` (already
    canonical), together with the dataset id used for artefact namespacing.
    """
    definition = get_dataset(config, dataset_id)
    source = definition.output_path or definition.source_path
    return Path(source), definition.dataset_id


def run_adapter(definition: DatasetDefinition) -> AdapterResult:
    """Load a dataset's raw files and convert them to the canonical schema.

    Raises
    ------
    FileNotFoundError
        When the configured ``source_path`` is missing — the adapter refuses to
        invent data (prompt constraint: "do not create fake data").
    """
    if not definition.source_path.exists():
        raise FileNotFoundError(
            f"Raw source for dataset '{definition.dataset_id}' not found: "
            f"{definition.source_path}. Place the dataset there or fix "
            "'source_path'; no synthetic data will be generated."
        )
    frame = load_telemetry(
        definition.source_path,
        device_column=definition.options.device_id or "device_id",
    )
    adapter = ADAPTERS[definition.dataset_type]
    logger.info(
        "Running '%s' adapter on %s (%d row(s)).",
        definition.dataset_type, definition.source_path, len(frame),
    )
    return adapter(frame, definition.options)


def inspect_dataset(definition: DatasetDefinition) -> dict[str, Any]:
    """Probe a dataset's raw files and infer column roles (``--inspect``)."""
    if not definition.source_path.exists():
        raise FileNotFoundError(
            f"Raw source for dataset '{definition.dataset_id}' not found: "
            f"{definition.source_path}."
        )
    source = definition.source_path
    files = [source] if source.is_file() else sorted(source.glob("*.csv"))
    frame = load_telemetry(source)
    result = inspect_columns(frame, definition.dataset_type)
    result["dataset_id"] = definition.dataset_id
    result["files_found"] = [str(f) for f in files]
    result["output_path"] = (
        str(definition.output_path) if definition.output_path else None
    )
    return result
