"""Sophos Firewall syslog adapter — offline / mock / live UDP receiver.

Live mode binds a UDP socket, receives datagrams (source-IP allowlisted,
size-bounded), and parses them with the existing Sophos syslog parser. run_once
waits for a single datagram or a timeout and exits — no persistent listener, no
response sent back to the firewall. No credentials are involved in UDP
reception.
"""

from __future__ import annotations

import logging
from typing import Any

from src.live_logging import sophos_syslog
from src.live_logging.adapters.base import CollectResult, LiveAdapter
from src.live_logging.models import ENGINE_CYBER

logger = logging.getLogger(__name__)


class SophosFirewallSyslogAdapter(LiveAdapter):
    """Receives/parses Sophos Firewall syslog datagrams."""

    name = "sophos_firewall_syslog"
    source_key = sophos_syslog.SOURCE_KEY  # "sophos_syslog"
    engine_target = ENGINE_CYBER
    friendly_name = "Sophos Firewall Syslog"
    dependency = None  # UDP reception uses stdlib sockets only

    # ---- collectors ---------------------------------------------------------

    def _collect_offline(self) -> CollectResult:
        path = self.cfg.get("offline_sample_path")
        return sophos_syslog.read_offline(path) if path else ([], [])

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

        bind_host = str(self.cfg.get("bind_host", "0.0.0.0"))
        bind_port = int(self.cfg.get("bind_port", 5514))
        max_bytes = int(self.cfg.get("max_datagram_bytes", 65535))
        timeout = float(self.cfg.get("receive_timeout_seconds", 1))
        max_wait = float(self.cfg.get("run_once_max_wait_seconds", timeout * 5))

        pairs: list[tuple[str | None, str]] = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, bind_port))
            sock.settimeout(timeout)
            import time

            deadline = time.monotonic() + max_wait
            while time.monotonic() < deadline and not pairs:
                try:
                    data, addr = sock.recvfrom(max_bytes)
                except socket.timeout:
                    continue
                pairs.append((addr[0], data.decode("utf-8", errors="replace")))
        finally:
            sock.close()
        return self._parse_datagrams(pairs)

    # ---- helpers ------------------------------------------------------------

    def _parse_datagrams(self, pairs: list[tuple[str | None, str]]) -> CollectResult:
        allowed = {str(s) for s in (self.cfg.get("allowed_sources") or [])}
        max_bytes = int(self.cfg.get("max_datagram_bytes", 65535))
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        for src, line in pairs:
            if line is None:
                continue
            if len(line.encode("utf-8", errors="replace")) > max_bytes:
                errors.append(f"datagram from {src} exceeds max_datagram_bytes")
                continue
            if allowed and src is not None and src not in allowed:
                errors.append(f"datagram from {src} not in allowed_sources")
                continue
            try:
                record = sophos_syslog.parse_syslog_line(line)
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
            sock.bind((str(self.cfg.get("bind_host", "0.0.0.0")), int(self.cfg.get("bind_port", 5514))))
        finally:
            sock.close()

    def _validate_extra(self) -> list[str]:
        problems: list[str] = []
        port = self.cfg.get("bind_port", 5514)
        if not isinstance(port, int) or not (0 < port < 65536):
            problems.append(f"bind_port must be a valid port, got {port!r}")
        if str(self.cfg.get("protocol", "udp")).lower() != "udp":
            problems.append("only udp protocol is supported in this phase")
        return problems
