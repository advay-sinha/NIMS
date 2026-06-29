"""Test suite for NetSentinel.

Mirrors the ``src`` layout: one test module per source module. Tests cover
normal, boundary and invalid-input cases (CLAUDE.md > Testing Policy).

Stubs whose implementation is pending (Phase 1 in progress) are marked
``xfail(raises=NotImplementedError, strict=True)`` so that filling in a
function automatically flips its test from xfail to pass and surfaces it.
"""

from __future__ import annotations
