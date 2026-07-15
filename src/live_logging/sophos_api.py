"""Sophos Central SIEM Integration API ingestion (offline-first) (Phase 9).

Purpose
-------
Parse Sophos Central alerts/events into source records for the normalizer. The
default path is OFFLINE: a saved sample JSON file is read. A live client is
represented by an injectable ``fetcher`` callable and is *disabled by default*;
this module never contacts Sophos on its own and never stores credentials —
credentials would be supplied to a live fetcher via environment variables named
in configuration (values are never read or logged here).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

SOURCE_KEY = "sophos_api"
VENDOR = "sophos"
PRODUCT = "sophos_central"

# Sophos alert ``type`` prefixes mapped to a coarse category when no explicit
# category field is present.
_TYPE_CATEGORY: dict[str, str] = {
    "Event::Endpoint::Threat": "antivirus",
    "Event::Endpoint": "endpoint",
    "Event::Firewall": "firewall",
    "Event::IPS": "ips",
    "Event::Web": "web_filter",
    "Event::Sandbox": "sandbox",
    "Event::Authentication": "authentication",
}


def _category_for(item: Mapping[str, Any]) -> str:
    explicit = item.get("category") or item.get("group")
    if explicit:
        return str(explicit).lower()
    event_type = str(item.get("type", ""))
    for prefix, category in _TYPE_CATEGORY.items():
        if event_type.startswith(prefix):
            return category
    return "security"


def parse_sophos_items(
    items: list[Mapping[str, Any]],
    source_type: str = "api",
) -> list[dict[str, Any]]:
    """Convert Sophos Central alert/event dicts into normalizer source records."""
    records: list[dict[str, Any]] = []
    for item in items:
        source_info = item.get("source_info") or {}
        src_ip = item.get("src_ip") or source_info.get("ip")
        dst_ip = item.get("dst_ip") or item.get("destination_ip")
        message = (
            item.get("description")
            or item.get("name")
            or item.get("message")
            or str(item.get("type", "sophos event"))
        )
        correlation_keys = {
            k: v
            for k, v in {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": item.get("protocol") or item.get("proto"),
            }.items()
            if v
        }
        records.append(
            {
                "source_vendor": VENDOR,
                "source_product": PRODUCT,
                "source_type": source_type,
                "source_key": SOURCE_KEY,
                "source_name": "sophos_central",
                "timestamp": item.get("created_at") or item.get("timestamp"),
                "category": _category_for(item),
                "subcategory": (str(item.get("type")) if item.get("type") else None),
                "severity": item.get("severity"),
                "message": str(message),
                "device_ip": src_ip,
                "hostname": item.get("location") or item.get("hostname"),
                "device_id": item.get("endpoint_id") or item.get("device_id"),
                "raw_ref": item.get("id") or item.get("alert_id"),
                "correlation_keys": correlation_keys,
                "normalized_fields": {"sophos_type": item.get("type")},
                "raw_payload": dict(item),
            }
        )
    return records


class SophosCentralClient:
    """Offline-first Sophos Central client.

    In ``offline`` mode ``fetch`` reads the configured sample JSON. In ``live``
    mode ``fetch`` delegates to an injected ``fetcher`` — but only when the
    source is explicitly enabled *and* a fetcher was provided. There is no
    built-in network path, so a misconfiguration can never reach Sophos.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        fetcher: Optional[Callable[[], list[Mapping[str, Any]]]] = None,
    ) -> None:
        self.config = dict(config or {})
        self.mode = str(self.config.get("mode", "offline")).lower()
        self.enabled = bool(self.config.get("enabled", False))
        self._fetcher = fetcher

    def fetch(self) -> list[Mapping[str, Any]]:
        """Return raw Sophos items for this run (offline sample or live fetcher)."""
        if self.mode == "live":
            if not self.enabled:
                raise RuntimeError("Sophos Central live mode requested but source is disabled.")
            if self._fetcher is None:
                raise RuntimeError(
                    "Sophos Central live mode requires an explicit fetcher; none provided "
                    "(live ingestion is disabled by default and needs user approval)."
                )
            return list(self._fetcher())
        return self._read_offline()

    def _read_offline(self) -> list[Mapping[str, Any]]:
        sample_path = self.config.get("offline_sample_path")
        if not sample_path:
            return []
        path = Path(sample_path)
        if not path.is_file():
            logger.warning("Sophos offline sample not found: %s", path)
            return []
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("items") or data.get("alerts") or data.get("events") or []
        return list(data)
