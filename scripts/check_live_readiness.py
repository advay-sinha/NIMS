"""Entry point: preflight live-ingestion readiness (non-destructive).

Reports, per source, whether a live run-once could proceed — adapter presence,
dependency, mode, enabled/live state, required env vars present (never their
values), bind-port availability, target inventory, path writability, safety
status and the exact remaining setup steps. Performs no state-changing test.

Usage
-----
    python -m scripts.check_live_readiness --source all
    python -m scripts.check_live_readiness --source sophos_firewall_syslog --json
"""

from __future__ import annotations

import argparse
import json
import logging

from src.live_logging.adapters import preflight
from src.live_logging.adapters.registry import SPEC_SOURCES, resolve_source
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_LIVE_CONFIG = "configs/live_logging.yaml"
DEFAULT_SOPHOS_CONFIG = "configs/sophos_logging.yaml"
DEFAULT_HIRSCHMANN_CONFIG = "configs/hirschmann_logging.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Preflight live-ingestion readiness.")
    parser.add_argument("--source", default="all",
                        help="Source name or 'all' (default: all).")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    parser.add_argument("--live-config", default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--sophos-config", default=DEFAULT_SOPHOS_CONFIG)
    parser.add_argument("--hirschmann-config", default=DEFAULT_HIRSCHMANN_CONFIG)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` unless a source is blocked by a safety issue)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    live = load_yaml(args.live_config).get("live_logging", {})
    sophos = load_yaml(args.sophos_config)
    hirschmann = load_yaml(args.hirschmann_config)
    output_dir = args.output_dir or live.get("output_dir") or "outputs/live_logging"

    if args.source == "all":
        reports = preflight.assess_all(live, sophos, hirschmann, output_dir)
    else:
        reports = [preflight.assess_source(resolve_source(args.source), live, sophos, hirschmann, output_dir)]

    preflight.write_readiness(reports, output_dir)

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2, sort_keys=True))
    else:
        for r in reports:
            logger.info("[%s] %s (%s)", r.status, r.friendly_name, r.source)
            logger.info("    mode=%s enabled=%s dependency=%s(%s) read_only=%s",
                        r.mode, r.enabled, r.dependency, "ok" if r.dependency_ok else "missing",
                        r.read_only)
            if r.env_present:
                present = ", ".join(f"{k}={'set' if v else 'unset'}" for k, v in r.env_present.items())
                logger.info("    env: %s", present)
            for step in r.remaining_steps:
                logger.info("    - %s", step)

    blocked = any(r.status == preflight.BLOCKED_BY_SAFETY for r in reports)
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
