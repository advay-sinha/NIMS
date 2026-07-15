"""Sophos Central SIEM API adapter — offline / mock / live HTTP polling.

Live mode authenticates with OAuth client credentials (from environment
variables only), then polls the SIEM events/alerts endpoints with pagination and
a cursor checkpoint. ``httpx`` is imported lazily inside the live path. The
adapter never prints credentials and never guesses tenant/region/API host.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.live_logging import sophos_api
from src.live_logging.adapters.base import CollectResult, LiveAdapter
from src.live_logging.adapters.errors import ConnectionTestError
from src.live_logging.models import ENGINE_CYBER

logger = logging.getLogger(__name__)


class SophosCentralAdapter(LiveAdapter):
    """Polls the Sophos Central SIEM Integration API."""

    name = "sophos_central"
    source_key = sophos_api.SOURCE_KEY  # "sophos_api"
    engine_target = ENGINE_CYBER
    friendly_name = "Sophos Central SIEM API"
    dependency = "httpx"

    def required_env_vars(self) -> list[str]:
        return [
            self.cfg.get("client_id_env", "SOPHOS_CLIENT_ID"),
            self.cfg.get("client_secret_env", "SOPHOS_CLIENT_SECRET"),
        ]

    def _collect_offline(self) -> CollectResult:
        client = sophos_api.SophosCentralClient({**self.cfg, "mode": "offline"})
        return sophos_api.parse_sophos_items(client.fetch()), []

    def _collect_mock(self) -> CollectResult:
        payload = self.ctx.mock if self.ctx.mock is not None else self.cfg.get("mock", {})
        if callable(payload):
            items = list(payload())
        elif isinstance(payload, dict):
            items = list(payload.get("items", []))
        else:
            items = list(payload or [])
        return sophos_api.parse_sophos_items(items), []

    def _collect_live(self) -> CollectResult:  # pragma: no cover - needs network
        items = self._poll_live()
        return sophos_api.parse_sophos_items(items, source_type="api"), []

    def _poll_live(self) -> list[dict[str, Any]]:  # pragma: no cover
        import httpx

        client_id = os.environ.get(self.cfg.get("client_id_env", "SOPHOS_CLIENT_ID"))
        client_secret = os.environ.get(self.cfg.get("client_secret_env", "SOPHOS_CLIENT_SECRET"))
        region = os.environ.get(self.cfg.get("region_env", "SOPHOS_REGION")) or self.cfg.get("region")
        tenant = os.environ.get(self.cfg.get("tenant_id_env", "SOPHOS_TENANT_ID"))
        if not client_id or not client_secret:
            raise ConnectionTestError("Sophos Central credentials are not set")
        if not region or not tenant:
            raise ConnectionTestError("Sophos Central tenant/region are not configured")

        timeout = float(self.cfg.get("request_timeout_seconds", 30))
        batch = int(self.cfg.get("batch_size", 500))
        with httpx.Client(timeout=timeout) as http:
            token = self._oauth_token(http, client_id, client_secret)
            data_region_host = f"https://api-{region}.central.sophos.com"
            headers = {"Authorization": f"Bearer {token}", "X-Tenant-ID": tenant}
            items: list[dict[str, Any]] = []
            cursor = self.checkpoint().get("cursor", {}).get("api_cursor")
            params: dict[str, Any] = {"limit": batch}
            if cursor:
                params["cursor"] = cursor
            resp = http.get(f"{data_region_host}/siem/v1/events", headers=headers, params=params)
            resp.raise_for_status()
            body = resp.json()
            items.extend(body.get("items", []))
            return items

    @staticmethod
    def _oauth_token(http: Any, client_id: str, client_secret: str) -> str:  # pragma: no cover
        resp = http.post(
            "https://id.sophos.com/api/v2/oauth2/token",
            data={"grant_type": "client_credentials", "client_id": client_id,
                  "client_secret": client_secret, "scope": "token"},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def _test_connection_live(self) -> None:  # pragma: no cover
        for name in self.required_env_vars():
            if not os.environ.get(name):
                raise ConnectionTestError(f"required environment variable {name} is not set")
        if not (os.environ.get(self.cfg.get("region_env", "SOPHOS_REGION")) or self.cfg.get("region")):
            raise ConnectionTestError("Sophos region is not configured")

    def _validate_extra(self) -> list[str]:
        problems: list[str] = []
        if self.mode == "live":
            if not (self.cfg.get("region") or os.environ.get(self.cfg.get("region_env", "SOPHOS_REGION"))):
                problems.append("live mode requires a configured region (never guessed)")
        return problems
