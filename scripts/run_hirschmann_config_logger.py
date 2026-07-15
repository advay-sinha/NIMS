"""Entry point: run Hirschmann configuration ingestion (snapshot diffs).

Offline by default. Live mode is gated (disabled unless config_retrieval has
enabled=true, mode=live, read_only=true) and is strictly read-only: a single
allowlisted show command over host-key-verified SSH, no configuration mode, no
write commands.

Usage
-----
    python -m scripts.run_hirschmann_config_logger --offline --run-once
    python -m scripts.run_hirschmann_config_logger --live --run-once
"""

from __future__ import annotations

import argparse
import logging

from scripts._bootstrap import add_common_arguments, bootstrap
from src.live_logging import runtime
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_LIVE_CONFIG = "configs/live_logging.yaml"
DEFAULT_SOPHOS_CONFIG = "configs/sophos_logging.yaml"
DEFAULT_HIRSCHMANN_CONFIG = "configs/hirschmann_logging.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Run Hirschmann config ingestion.")
    add_common_arguments(parser)
    parser.add_argument("--live-config", default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--sophos-config", default=DEFAULT_SOPHOS_CONFIG)
    parser.add_argument("--hirschmann-config", default=DEFAULT_HIRSCHMANN_CONFIG)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true")
    group.add_argument("--mock", action="store_true")
    group.add_argument("--live", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success)."""
    args = build_parser().parse_args(argv)
    bootstrap(args)
    live = load_yaml(args.live_config).get("live_logging", {})
    mode = "live" if args.live else "mock" if args.mock else "offline" if args.offline else None
    status, output_dir = runtime.run(
        ["hirschmann_config"], live, load_yaml(args.sophos_config), load_yaml(args.hirschmann_config),
        mode=mode, dry_run=args.dry_run,
    )
    logger.info("Hirschmann config ingestion complete: %d events, output=%s",
                status.total_events, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
