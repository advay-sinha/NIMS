"""Entry point: offline analysis of saved network device command outputs.

Reads a directory of saved ``show ...`` outputs (read-only), builds a
structured inventory and writes it under
``outputs/network_config/<snapshot_id>/``. No live device access, no SNMP, no
remediation — Engine C Phase 1 is offline-only.

Usage
-----
    python -m scripts.analyze_network_config \\
        --input-dir datasets/samples/network_config --snapshot-id sample_offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_config.artifacts import write_inventory
from src.network_config.inventory import build_inventory

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Offline network-configuration analysis (read-only)."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory of saved command outputs "
             "(defaults to network_config.input_dir).",
    )
    parser.add_argument(
        "--snapshot-id", default=None,
        help="Snapshot id (output namespace; defaults to "
             "network_config.snapshot_id).",
    )
    parser.add_argument(
        "--skip-topology", action="store_true",
        help="Skip Phase 2 topology construction (inventory only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success, ``1`` on a resolvable failure)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    cfg = dict(ctx.config.get("network_config") or {})

    input_dir = Path(args.input_dir or cfg.get("input_dir", ""))
    snapshot_id = args.snapshot_id or str(cfg.get("snapshot_id", "snapshot"))

    try:
        inventory = build_inventory(input_dir, cfg, snapshot_id)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    topo_cfg = dict(cfg.get("topology") or {})
    topology = None
    if topo_cfg.get("enabled", True) and not args.skip_topology:
        from src.network_config.topology import build_topology

        topology = build_topology(inventory, topo_cfg)

    paths = write_inventory(
        inventory, Path(ctx.paths.network_config_dir), topology
    )
    logger.info(
        "Analyzed %d device(s), %d interface(s); snapshot at %s.",
        len(inventory.devices), len(inventory.all_interfaces),
        paths["report"].parent,
    )
    if topology is not None:
        logger.info(
            "Topology: %d node(s), %d edge(s), %d warning(s).",
            len(topology.nodes), len(topology.edges), len(topology.warnings),
        )
    if inventory.files_missing:
        logger.info("Missing input files: %s",
                    ", ".join(inventory.files_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
