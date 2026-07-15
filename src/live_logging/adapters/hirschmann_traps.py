"""Hirschmann trap adapter — offline / mock / live UDP trap receiver.

Live mode binds a UDP socket and receives trap datagrams (source-IP allowlisted,
size-bounded). Text-form traps are parsed with the existing trap parser; binary
SNMP trap PDUs require a decoder (``pysnmp``) that is lazily imported. run_once
waits for one datagram or a timeout and exits — no response, no device change.
"""

from __future__ import annotations

import logging
from typing import Any

from src.live_logging import hirschmann_traps as trap_parser
from src.live_logging.adapters.base import CollectResult, LiveAdapter
from src.live_logging.models import ENGINE_NETWORK_HEALTH

logger = logging.getLogger(__name__)


class HirschmannTrapAdapter(LiveAdapter):
    """Receives/parses Hirschmann SNMP traps."""

    name = "hirschmann_traps"
    source_key = trap_parser.SOURCE_KEY  # "hirschmann_traps"
    engine_target = ENGINE_NETWORK_HEALTH
    friendly_name = "Hirschmann Traps"
    dependency = None  # text trap parsing needs no dep; binary decode is optional

    def _collect_offline(self) -> CollectResult:
        path = self.cfg.get("offline_sample_path")
        return trap_parser.read_offline(path) if path else ([], [])

    def _collect_mock(self) -> CollectResult:
        payload = self.ctx.mock if self.ctx.mock is not None else self.cfg.get("mock", {})
        datagrams = payload.get("datagrams", []) if isinstance(payload, dict) else list(payload or [])
        pairs = [
            (d.get("src"), d.get("line")) if isinstance(d, dict) else (None, d)
            for d in datagrams
        ]
        return self._parse_datagrams(pairs)

    def _collect_live(self) -> CollectResult:  # pragma: no cover - needs a socket peer
        import socket
        import time

        bind_host = str(self.cfg.get("bind_host", "0.0.0.0"))
        bind_port = int(self.cfg.get("bind_port", 5162))
        timeout = float(self.cfg.get("receive_timeout_seconds", 1))
        max_wait = float(self.cfg.get("run_once_max_wait_seconds", timeout * 5))
        max_bytes = int(self.cfg.get("max_datagram_bytes", 65535))

        pairs: list[tuple[str | None, str]] = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, bind_port))
            sock.settimeout(timeout)
            deadline = time.monotonic() + max_wait
            while time.monotonic() < deadline and not pairs:
                try:
                    data, addr = sock.recvfrom(max_bytes)
                except socket.timeout:
                    continue
                pairs.append((addr[0], self._decode(data, addr[0])))
        finally:
            sock.close()
        return self._parse_datagrams(pairs)

    @staticmethod
    def _decode(data: bytes, src: str) -> str:  # pragma: no cover
        """Best-effort decode of a trap datagram to a parseable text line."""
        text = data.decode("utf-8", errors="replace")
        # If it looks like a text trap line, use as-is; otherwise synthesise a
        # minimal line so the parser can record an (unknown) trap from src.
        return text if any(c.isalpha() for c in text[:16]) else f"- {src} snmpTrapPdu"

    def _parse_datagrams(self, pairs: list[tuple[str | None, str]]) -> CollectResult:
        allowed = {str(s) for s in (self.cfg.get("allowed_sources") or [])}
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        for src, line in pairs:
            if line is None:
                continue
            if allowed and src is not None and src not in allowed:
                errors.append(f"trap from {src} not in allowed_sources")
                continue
            try:
                record = trap_parser.parse_trap_line(line)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if record:
                if src:
                    record.setdefault("device_ip", src)
                records.append(record)
        return records, errors

    def _test_connection_live(self) -> None:  # pragma: no cover
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((str(self.cfg.get("bind_host", "0.0.0.0")), int(self.cfg.get("bind_port", 5162))))
        finally:
            sock.close()

    def _validate_extra(self) -> list[str]:
        port = self.cfg.get("bind_port", 5162)
        if not isinstance(port, int) or not (0 < port < 65536):
            return [f"bind_port must be a valid port, got {port!r}"]
        return []
