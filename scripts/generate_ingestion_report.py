"""Entry point: (re)generate ingestion_status.json + ingestion_report.md.

Reads the persisted normalized events under the live-logging output directory
and recomputes the status counts, then rewrites the JSON status and Markdown
report. Uses an existing ingestion_status.json (if present) to preserve
per-source rows; otherwise emits a counts-only status.

Usage
-----
    python -m scripts.generate_ingestion_report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.live_logging import reports
from src.live_logging.models import IngestionStatus, SourceStatus, utc_now_iso
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_LIVE_CONFIG = "configs/live_logging.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Regenerate the ingestion status + report.")
    add_common_arguments(parser)
    parser.add_argument("--live-config", default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--output-dir", default=None,
                        help="Override the live-logging output directory.")
    return parser


def _load_prior_status(output_dir: Path) -> IngestionStatus:
    path = output_dir / reports.STATUS_FILENAME
    if not path.is_file():
        now = utc_now_iso()
        return IngestionStatus(started_at=now, finished_at=now, mode="offline",
                               read_only=True, total_events=0)
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = [SourceStatus(**{k: v for k, v in s.items() if k in SourceStatus.__annotations__})
               for s in data.get("sources", [])]
    return IngestionStatus(
        started_at=data.get("started_at", ""),
        finished_at=data.get("finished_at", ""),
        mode=data.get("mode", "offline"),
        read_only=bool(data.get("read_only", True)),
        total_events=int(data.get("total_events", 0)),
        sources=sources,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success)."""
    args = build_parser().parse_args(argv)
    bootstrap(args)
    live = load_yaml(args.live_config).get("live_logging", {})
    output_dir = Path(args.output_dir or live.get("output_dir") or "outputs/live_logging")

    status = _load_prior_status(output_dir)
    reports.enrich_status(status, output_dir)
    reports.write_status(status, output_dir)
    report_path = reports.write_report(status, output_dir)
    logger.info("Regenerated ingestion report (%d events) -> %s",
                status.total_events, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
