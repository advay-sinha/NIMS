"""Entry point: launch the offline NIMS monitoring dashboard (Streamlit).

Streamlit is **optional**. This launcher never imports Streamlit at module load,
so it (and the test-suite) works in environments without Streamlit. If Streamlit
is installed it starts the app; otherwise it prints clear install/run
instructions and exits non-zero.

The dashboard is a read-only viewer over persisted artefacts — it never runs an
engine pipeline, trains, infers, polls SNMP, captures packets, contacts a device
or executes a command.

Usage
-----
    python -m scripts.run_dashboard
    # or, equivalently:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"

_INSTALL_HINT = (
    "Streamlit is not installed. The dashboard is optional.\n"
    "Install it, then re-run:\n"
    "    pip install streamlit\n"
    "    python -m scripts.run_dashboard\n"
    "Or run directly:\n"
    "    streamlit run src/dashboard/app.py")


def streamlit_available() -> bool:
    """True if the ``streamlit`` package is importable (no import performed)."""
    return importlib.util.find_spec("streamlit") is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the offline NIMS monitoring dashboard (read-only; "
                    "no command is executed).")
    parser.add_argument("--port", default=None, help="Streamlit server port.")
    parser.add_argument("--server-address", default=None,
                        help="Streamlit server address.")
    return parser


def _launch(extra: list[str]) -> int:
    """Start Streamlit as a subprocess and return its exit code."""
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *extra]
    logger.info("Launching dashboard: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on clean exit; non-zero if Streamlit is absent)."""
    args = build_parser().parse_args(argv)
    if not streamlit_available():
        logger.error("%s", _INSTALL_HINT)
        print(_INSTALL_HINT)
        return 1

    extra: list[str] = []
    if args.port:
        extra += ["--server.port", str(args.port)]
    if args.server_address:
        extra += ["--server.address", str(args.server_address)]
    return _launch(extra)


if __name__ == "__main__":
    sys.exit(main())
