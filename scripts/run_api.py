"""Entry point: run the NetSentinel inference API with uvicorn.

Serves the registry-promoted production models for batch CSV/JSON inference.
Host/port come from ``configs/api.yaml``.

Usage
-----
    python -m scripts.run_api
    python -m scripts.run_api --reload
    # equivalent: uvicorn src.api.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import add_common_arguments, bootstrap

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Run the NetSentinel inference API."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload (development only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (blocks until the server stops)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    api_cfg = dict(ctx.config.get("api") or {})

    import uvicorn

    host = str(api_cfg.get("host", "127.0.0.1"))
    port = int(api_cfg.get("port", 8000))
    logger.info("Starting inference API on http://%s:%d", host, port)
    uvicorn.run("src.api.app:app", host=host, port=port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
