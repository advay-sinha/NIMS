"""Executable entry points for NetSentinel.

Scripts are thin CLI wrappers: they parse arguments, load configuration, set up
logging and delegate to ``src`` (CLAUDE.md > Directory Philosophy:
"scripts/ contains entry points"). No business logic lives here.

Run from the repository root, e.g.::

    python -m scripts.prepare_data --dataset nsl_kdd
"""

from __future__ import annotations
