"""Hirschmann SNMPv3 adapter — offline / mock / live read-only polling.

Live mode performs read-only SNMPv3 GETs (never SET) against a configured target
inventory, restricted to an allowlisted OID profile, and maps the results into
the canonical SNMP reading shape parsed by the existing metric parser. ``pysnmp``
is imported lazily inside the live path. Credentials come from environment
variables only; no arbitrary OIDs or targets are accepted from the CLI/frontend.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.live_logging import hirschmann_snmp as snmp_parser
from src.live_logging.adapters.base import CollectResult, LiveAdapter
from src.live_logging.adapters.errors import ConfigurationError, ConnectionTestError
from src.live_logging.models import ENGINE_NETWORK_HEALTH, utc_now_iso

logger = logging.getLogger(__name__)

# Predefined, allowlisted OID profiles. Read-only scalar/columnar OIDs only;
# there is no mechanism to add arbitrary OIDs at runtime.
OID_PROFILES: dict[str, dict[str, str]] = {
    "hirschmann_hios_basic": {
        "sysName": "1.3.6.1.2.1.1.5.0",
        "sysUpTime": "1.3.6.1.2.1.1.3.0",
        "ifOperStatus": "1.3.6.1.2.1.2.2.1.8",
        "ifInOctets": "1.3.6.1.2.1.2.2.1.10",
        "ifInErrors": "1.3.6.1.2.1.2.2.1.14",
        "ifOutDiscards": "1.3.6.1.2.1.2.2.1.19",
    },
}


class HirschmannSnmpAdapter(LiveAdapter):
    """Read-only SNMPv3 poller for Hirschmann switch health."""

    name = "hirschmann_snmp"
    source_key = snmp_parser.SOURCE_KEY  # "hirschmann_snmp"
    engine_target = ENGINE_NETWORK_HEALTH
    friendly_name = "Hirschmann SNMP Health"
    dependency = "pysnmp"

    def required_env_vars(self) -> list[str]:
        return [
            self.cfg.get("username_env", "HIRSCHMANN_SNMP_USER"),
            self.cfg.get("auth_password_env", "HIRSCHMANN_SNMP_AUTH_PASSWORD"),
            self.cfg.get("privacy_password_env", "HIRSCHMANN_SNMP_PRIV_PASSWORD"),
        ]

    def _thresholds(self) -> dict[str, Any]:
        return dict(self.cfg.get("thresholds") or {})

    def _collect_offline(self) -> CollectResult:
        path = self.cfg.get("offline_sample_path")
        return (snmp_parser.read_offline(path, self._thresholds()) if path else []), []

    def _collect_mock(self) -> CollectResult:
        payload = self.ctx.mock if self.ctx.mock is not None else self.cfg.get("mock", {})
        readings = payload.get("readings", []) if isinstance(payload, dict) else list(payload or [])
        return snmp_parser.parse_snmp_metrics(readings, self._thresholds()), []

    def _collect_live(self) -> CollectResult:  # pragma: no cover - needs a device
        readings = [r for r in (self._poll_target(t) for t in self._load_targets()) if r]
        records = snmp_parser.parse_snmp_metrics(readings, self._thresholds())
        for reading in readings:
            heartbeat = snmp_parser.heartbeat_event(reading)
            if heartbeat is not None:
                records.append(heartbeat)
        return records, []

    # ---- live helpers (read-only GET only) ----------------------------------

    def _load_targets(self) -> list[dict[str, Any]]:  # pragma: no cover
        targets_file = self.cfg.get("targets_file")
        if not targets_file or not os.path.isfile(targets_file):
            raise ConfigurationError("SNMP targets_file is not configured or missing")
        import yaml

        with open(targets_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return list(data.get("targets", []))

    def _poll_target(self, target: dict[str, Any]) -> dict[str, Any] | None:  # pragma: no cover
        """Poll one target with read-only SNMPv3 GETs (async pysnmp v6/7 API)."""
        import asyncio

        return asyncio.run(self._poll_target_async(target))

    async def _poll_target_async(self, target: dict[str, Any]) -> dict[str, Any] | None:  # pragma: no cover
        # pysnmp 6/7 moved the v3 USM API here; ``get_cmd`` is a coroutine and
        # the transport is created asynchronously. Read-only GET only (no writes).
        from pysnmp.hlapi.v3arch import (  # lazy import — read-only get_cmd only
            ContextData,
            ObjectIdentity,
            ObjectType,
            SnmpEngine,
            UdpTransportTarget,
            UsmUserData,
            get_cmd,
            usmHMACSHAAuthProtocol,
            usmAesCfb128Protocol,
        )

        profile = OID_PROFILES.get(str(target.get("profile", "hirschmann_hios_basic")))
        if not profile:
            raise ConfigurationError(f"unknown OID profile for target {target.get('name')}")
        user = os.environ.get(target.get("username_env", "HIRSCHMANN_SNMP_USER"))
        auth = os.environ.get(target.get("auth_password_env", "HIRSCHMANN_SNMP_AUTH_PASSWORD"))
        priv = os.environ.get(target.get("privacy_password_env", "HIRSCHMANN_SNMP_PRIV_PASSWORD"))
        if not (user and auth and priv):
            raise ConnectionTestError(f"SNMP credentials missing for {target.get('name')}")

        reading: dict[str, Any] = {
            "device_id": target.get("name"),
            "device_ip": target.get("host"),
            "timestamp": utc_now_iso(),
        }
        engine = SnmpEngine()
        usm = UsmUserData(user, auth, priv,
                          authProtocol=usmHMACSHAAuthProtocol, privProtocol=usmAesCfb128Protocol)
        udp = await UdpTransportTarget.create(
            (target["host"], int(target.get("port", 161))),
            timeout=float(self.cfg.get("request_timeout_seconds", 5)),
            retries=int(self.cfg.get("request_retries", 1)),
        )
        for field, oid in profile.items():
            errInd, errStat, _, binds = await get_cmd(
                engine, usm, udp, ContextData(), ObjectType(ObjectIdentity(oid))
            )
            if errInd or errStat:
                reading["reachable"] = False
                break
            for _oid, val in binds:
                reading[field] = val.prettyPrint()
        reading.setdefault("reachable", True)
        if reading.get("sysName"):
            reading.setdefault("hostname", reading["sysName"])
        return reading

    def _test_connection_live(self) -> None:  # pragma: no cover
        for name in self.required_env_vars():
            if not os.environ.get(name):
                raise ConnectionTestError(f"required environment variable {name} is not set")
        self._load_targets()  # ensures target inventory exists

    def _validate_extra(self) -> list[str]:
        problems: list[str] = []
        # Reject any attempt to configure arbitrary OIDs or SET.
        if self.cfg.get("oids") or self.cfg.get("allow_arbitrary_oids"):
            problems.append("arbitrary OIDs are not permitted; use a predefined profile")
        profile = self.cfg.get("profile")
        if profile and profile not in OID_PROFILES:
            problems.append(f"unknown OID profile '{profile}'")
        return problems
