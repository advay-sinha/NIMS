"""Adapter registry — build adapters from configuration by source name.

Maps the five canonical source names to their adapter class and the config block
each reads, and assembles the shared :class:`AdapterContext`. The routing keys
(``sophos_syslog`` / ``sophos_api`` / …) used by the normalizer are preserved on
each adapter, while the registry is keyed by the spec source names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.live_logging.adapters.base import AdapterContext, LiveAdapter
from src.live_logging.adapters.hirschmann_config import HirschmannConfigAdapter
from src.live_logging.adapters.hirschmann_snmp import HirschmannSnmpAdapter
from src.live_logging.adapters.hirschmann_traps import HirschmannTrapAdapter
from src.live_logging.adapters.sophos_central import SophosCentralAdapter
from src.live_logging.adapters.sophos_firewall_syslog import SophosFirewallSyslogAdapter
from src.live_logging.routing import DEFAULT_ROUTING

SPEC_SOURCES: tuple[str, ...] = (
    "sophos_firewall_syslog",
    "sophos_central",
    "hirschmann_snmp",
    "hirschmann_traps",
    "hirschmann_config",
)

# Short aliases accepted on the CLI (--source firewall_syslog, snmp, …) and the
# internal routing keys, all resolving to the canonical spec source name.
ALIASES: dict[str, str] = {
    "firewall_syslog": "sophos_firewall_syslog",
    "sophos_syslog": "sophos_firewall_syslog",
    "central_api": "sophos_central",
    "sophos_api": "sophos_central",
    "snmp": "hirschmann_snmp",
    "traps": "hirschmann_traps",
    "config": "hirschmann_config",
}


def resolve_source(name: str) -> str:
    """Resolve an alias/short name to a canonical spec source name."""
    key = str(name).strip().lower()
    if key in SPEC_SOURCES:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise KeyError(f"unknown source '{name}' (known: {', '.join(SPEC_SOURCES)})")


def _sophos_block(sophos_cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    return dict((sophos_cfg.get("sophos") or sophos_cfg).get(key, {}))


def _hirschmann_block(hirschmann_cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    return dict((hirschmann_cfg.get("hirschmann") or hirschmann_cfg).get(key, {}))


def _config_block(hirschmann_cfg: Mapping[str, Any]) -> dict[str, Any]:
    # Merge legacy snapshot config with the new retrieval block so snapshot_dir
    # (offline) and SSH settings (live) both apply.
    merged = _hirschmann_block(hirschmann_cfg, "config_snapshots")
    merged.update(_hirschmann_block(hirschmann_cfg, "config_retrieval"))
    return merged


def _snmp_block(hirschmann_cfg: Mapping[str, Any]) -> dict[str, Any]:
    block = _hirschmann_block(hirschmann_cfg, "snmp_polling")
    block.setdefault("thresholds", (hirschmann_cfg.get("hirschmann") or hirschmann_cfg).get("thresholds", {}))
    return block


def config_for(source: str, sophos_cfg: Mapping[str, Any], hirschmann_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the config block a source's adapter reads."""
    source = resolve_source(source)
    return {
        "sophos_firewall_syslog": lambda: _sophos_block(sophos_cfg, "firewall_syslog"),
        "sophos_central": lambda: _sophos_block(sophos_cfg, "central_api"),
        "hirschmann_snmp": lambda: _snmp_block(hirschmann_cfg),
        "hirschmann_traps": lambda: _hirschmann_block(hirschmann_cfg, "traps"),
        "hirschmann_config": lambda: _config_block(hirschmann_cfg),
    }[source]()


_CLASSES: dict[str, type[LiveAdapter]] = {
    "sophos_firewall_syslog": SophosFirewallSyslogAdapter,
    "sophos_central": SophosCentralAdapter,
    "hirschmann_snmp": HirschmannSnmpAdapter,
    "hirschmann_traps": HirschmannTrapAdapter,
    "hirschmann_config": HirschmannConfigAdapter,
}


def build_adapter(
    source: str,
    sophos_cfg: Mapping[str, Any],
    hirschmann_cfg: Mapping[str, Any],
    context: AdapterContext,
    mode: str | None = None,
) -> LiveAdapter:
    """Construct one adapter for a source with its config block bound."""
    source = resolve_source(source)
    cfg = config_for(source, sophos_cfg, hirschmann_cfg)
    if mode is not None:
        cfg = {**cfg, "mode": mode}
    return _CLASSES[source](cfg, context)


def build_context(
    live_cfg: Mapping[str, Any],
    sophos_cfg: Mapping[str, Any],
    hirschmann_cfg: Mapping[str, Any],
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    mock: Any = None,
) -> AdapterContext:
    """Assemble the shared adapter context from the loaded configs."""
    live = dict(live_cfg or {})
    out = Path(output_dir or live.get("output_dir") or "outputs/live_logging")
    routing = {**DEFAULT_ROUTING, **dict(live.get("routing") or {})}
    checkpoint_dir = live.get("checkpoints", {}).get("path")
    secret_env_vars: list[str] = []
    _collect_env_names(sophos_cfg, secret_env_vars)
    _collect_env_names(hirschmann_cfg, secret_env_vars)
    return AdapterContext(
        output_dir=out,
        routing=routing,
        secret_env_vars=secret_env_vars,
        redact_secrets=bool(live.get("redact_secrets", True)),
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        dry_run=dry_run,
        mock=mock,
    )


def _collect_env_names(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key.endswith("_env") and isinstance(value, str):
                out.append(value)
            else:
                _collect_env_names(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_env_names(item, out)
