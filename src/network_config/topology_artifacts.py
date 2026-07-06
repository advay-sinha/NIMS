"""Engine C Phase 2 — topology artefact persistence.

Writes the derived topology alongside the Phase 1 inventory in
``outputs/network_config/<snapshot_id>/``::

    topology.json
    topology_nodes.csv
    topology_edges.csv
    topology_warnings.csv

Read-only analysis in, files out.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any, Sequence

from src.network_config.topology import NetworkTopology, topology_summary

logger = logging.getLogger(__name__)

_NODE_FIELDS = ["node_id", "hostname", "device_type", "management_ip", "source"]
_EDGE_FIELDS = [
    "local_device", "local_interface", "remote_device", "remote_interface",
    "discovery_protocol", "confidence", "evidence", "bidirectional",
]
_WARNING_FIELDS = [
    "warning_id", "severity", "category", "message", "device", "interface",
    "evidence",
]


def write_topology(topology: NetworkTopology, out_dir: Path) -> dict[str, Path]:
    """Persist a topology (JSON + three CSVs) into an existing snapshot dir."""
    from src.utils.io import write_json

    out = Path(out_dir)
    paths: dict[str, Path] = {}
    paths["topology"] = write_json(
        _topology_payload(topology), out / "topology.json"
    )
    paths["nodes"] = _write_csv(out / "topology_nodes.csv", topology.nodes,
                                _NODE_FIELDS)
    paths["edges"] = _write_csv(out / "topology_edges.csv", topology.edges,
                                _EDGE_FIELDS)
    paths["warnings"] = _write_csv(out / "topology_warnings.csv",
                                   topology.warnings, _WARNING_FIELDS)
    logger.info("Topology artefacts written to %s.", out)
    return paths


def _topology_payload(topology: NetworkTopology) -> dict[str, Any]:
    """Full JSON-serialisable topology plus its summary."""
    return {
        "snapshot_id": topology.snapshot_id,
        "summary": topology_summary(topology),
        "nodes": [dataclasses.asdict(n) for n in topology.nodes],
        "edges": [dataclasses.asdict(e) for e in topology.edges],
        "warnings": [dataclasses.asdict(w) for w in topology.warnings],
    }


def _write_csv(path: Path, rows: Sequence[Any], fields: Sequence[str]) -> Path:
    """Write dataclass rows to a CSV with a fixed header."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            record = dataclasses.asdict(row)
            writer.writerow({f: _cell(record.get(f)) for f in fields})
    return path


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return "" if value is None else value
