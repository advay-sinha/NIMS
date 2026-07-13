"""Small Streamlit rendering helpers for the dashboard.

This module imports Streamlit at module load and is therefore imported **only**
by the running app (``src/dashboard/app.py``) — never by ``loader``/``formatting``
or the test-suite, so the package stays importable without Streamlit installed.
These helpers only render already-prepared data; they never load or compute it.
"""

from __future__ import annotations

from typing import Any, Iterable

import streamlit as st

from src.dashboard.formatting import OFFLINE_BANNER


def offline_banner() -> None:
    """Render the persistent 'offline, no execution' banner."""
    st.info(f"🔒 {OFFLINE_BANNER}", icon="🔒")


def section_title(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def metric_cards(cards: list[tuple[str, Any]], per_row: int = 4) -> None:
    """Render (label, value) pairs as metric cards, ``per_row`` per row."""
    for start in range(0, len(cards), per_row):
        chunk = cards[start:start + per_row]
        columns = st.columns(len(chunk))
        for column, (label, value) in zip(columns, chunk):
            column.metric(label, value)


def table(rows: Iterable[dict[str, Any]], empty_msg: str = "No data.") -> None:
    """Render a list of dict rows as a table (or an empty-state caption)."""
    rows = list(rows)
    if not rows:
        st.caption(empty_msg)
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def missing_notice(message: str | None) -> None:
    """Render a clear, non-fatal 'artefact missing' notice with a run command."""
    if not message:
        return
    st.warning(message)


def counts_table(counts: dict[str, int], key_label: str, value_label: str = "Count"
                 ) -> None:
    """Render a {key: count} mapping as a small two-column table."""
    rows = [{key_label: k, value_label: v}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    table(rows, empty_msg="None.")
