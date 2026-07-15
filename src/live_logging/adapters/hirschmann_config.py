"""Hirschmann read-only configuration adapter — offline / mock / live SSH.

Live mode opens an SSH session (host-key verified), runs a single allowlisted
read-only ``show`` command, and treats the output as a configuration snapshot fed
to the existing Engine C config parser + differ. ``paramiko`` is imported lazily.
There is no configuration mode, no write command and no arbitrary command input;
startup is rejected if any write/config flag is enabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.live_logging import config_diff, hirschmann_config as cfg_parser
from src.live_logging.adapters import safety
from src.live_logging.adapters.base import CollectResult, LiveAdapter
from src.live_logging.adapters.errors import ConfigurationError, ConnectionTestError, SafetyViolation
from src.live_logging.models import ENGINE_NETWORK_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_COMMAND = "show running-config"


class HirschmannConfigAdapter(LiveAdapter):
    """Read-only configuration retrieval + diffing."""

    name = "hirschmann_config"
    source_key = config_diff.SOURCE_KEY  # "hirschmann_config"
    engine_target = ENGINE_NETWORK_CONFIG
    friendly_name = "Hirschmann Configuration Backup"
    dependency = "paramiko"

    def required_env_vars(self) -> list[str]:
        return [
            self.cfg.get("username_env", "HIRSCHMANN_SSH_USER"),
            self.cfg.get("password_env", "HIRSCHMANN_SSH_PASSWORD"),
        ]

    def _allowed_commands(self) -> list[str]:
        return list(self.cfg.get("allowed_commands") or [DEFAULT_COMMAND])

    def _collect_offline(self) -> CollectResult:
        snapshot_dir = self.cfg.get("snapshot_dir")
        if not snapshot_dir:
            return [], []
        grouped = cfg_parser.load_snapshots_dir(snapshot_dir)
        records: list[dict[str, Any]] = []
        for snaps in grouped.values():
            records.extend(config_diff.diff_snapshot_series(snaps))
        return records, []

    def _collect_mock(self) -> CollectResult:
        payload = self.ctx.mock if self.ctx.mock is not None else self.cfg.get("mock", {})
        if not isinstance(payload, dict):
            return [], []
        device = str(payload.get("device_id", "mock-switch"))
        prev_text = payload.get("previous_config")
        curr_text = payload.get("current_config")
        if curr_text is None:
            return [], []
        curr = cfg_parser.parse_config_text(curr_text, device_id=device, label="current")
        if prev_text is None:
            return [], []
        prev = cfg_parser.parse_config_text(prev_text, device_id=device, label="previous")
        return config_diff.diff_configs(prev, curr), []

    def _collect_live(self) -> CollectResult:  # pragma: no cover - needs a device
        device, text = self._retrieve_live()
        curr = cfg_parser.parse_config_text(text, device_id=device, label="live")
        prev = self._load_previous(device)
        if prev is None:
            # First retrieval: nothing to diff against yet; record the hash only.
            return [], []
        return config_diff.diff_configs(prev, curr), []

    def _retrieve_live(self) -> tuple[str, str]:  # pragma: no cover
        import paramiko

        host = self.cfg.get("host")
        if not host:
            raise ConfigurationError("config_retrieval requires a configured host")
        command = self._allowed_commands()[0]
        safety.assert_command_allowed(command, self._allowed_commands())

        user = os.environ.get(self.cfg.get("username_env", "HIRSCHMANN_SSH_USER"))
        password = os.environ.get(self.cfg.get("password_env", "HIRSCHMANN_SSH_PASSWORD"))
        if not (user and password):
            raise ConnectionTestError("SSH credentials are not set")

        client = paramiko.SSHClient()
        known_hosts = self.cfg.get("known_hosts_file")
        if known_hosts and os.path.isfile(known_hosts):
            client.load_host_keys(known_hosts)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())  # host-key verification required
        try:
            client.connect(host, username=user, password=password, timeout=15, look_for_keys=False)
            _stdin, stdout, _stderr = client.exec_command(command)
            text = stdout.read().decode("utf-8", errors="replace")
        finally:
            client.close()
        return str(host), text

    def _load_previous(self, device: str):  # pragma: no cover
        snapshot_dir = self.cfg.get("snapshot_dir")
        if not snapshot_dir:
            return None
        grouped = cfg_parser.load_snapshots_dir(snapshot_dir)
        snaps = grouped.get(device) or next(iter(grouped.values()), [])
        return snaps[-1] if snaps else None

    def _test_connection_live(self) -> None:  # pragma: no cover
        self._validate_or_raise()
        for name in self.required_env_vars():
            if not os.environ.get(name):
                raise ConnectionTestError(f"required environment variable {name} is not set")

    def _validate_or_raise(self) -> None:
        problems = self._validate_extra()
        if problems:
            raise SafetyViolation("; ".join(problems))

    def _validate_extra(self) -> list[str]:
        problems: list[str] = []
        if self.cfg.get("allow_config_mode"):
            problems.append("allow_config_mode must be false")
        if self.cfg.get("allow_write_commands"):
            problems.append("allow_write_commands must be false")
        try:
            safety.assert_commands_allowlist(self._allowed_commands())
        except SafetyViolation as exc:
            problems.append(str(exc))
        if (self.mode == "live"
                and not self.cfg.get("known_hosts_file")
                and not self.cfg.get("lab_only_disable_host_key_check")):
            problems.append("live SSH requires known_hosts_file (host-key verification)")
        return problems
