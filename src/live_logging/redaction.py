"""Secret redaction for raw and normalized event payloads (Phase 9).

Purpose
-------
Secrets must never appear in persisted events, reports or manifests (spec
Phase 9 > Security and Secret Handling). This module masks common secret
patterns from strings and (recursively) from mappings/sequences *before*
anything is written to disk.

It is deliberately conservative: it masks by key name (password, token,
secret, community, authorization, private key…) and by value pattern (bearer
tokens, PEM blocks, ``community=...`` fragments), and can additionally mask the
literal values of configured secret environment variables if they ever leak
into a payload.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

MASK = "***REDACTED***"

# Key names whose values are always masked (case-insensitive substring match).
_SECRET_KEY_TOKENS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "client_secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "community",
    "snmp_community",
    "private_key",
    "privkey",
    "credential",
    "session_key",
    "bearer",
)

# Value patterns masked wherever they occur inside a string.
_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(authorization\s*[:=]\s*)\S+"),
    # SNMP community strings appear as `community <value>` or `community=<value>`.
    re.compile(r"(?i)\bcommunity\s*[:=]?\s+\S+"),
    re.compile(r"(?i)\b(password\s*[:=]\s*)\S+"),
    re.compile(r"(?i)\b(client_secret\s*[:=]\s*)\S+"),
    re.compile(r"(?i)\b(token\s*[:=]\s*)\S+"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.DOTALL,
    ),
)


def _is_secret_key(key: str) -> bool:
    """True when a mapping key name implies its value is a secret."""
    lowered = key.lower()
    return any(token in lowered for token in _SECRET_KEY_TOKENS)


def redact_text(text: str, extra_values: Iterable[str] = ()) -> str:
    """Mask secret patterns (and any literal ``extra_values``) inside a string."""
    if not isinstance(text, str) or not text:
        return text
    result = text
    for value in extra_values:
        if value:
            result = result.replace(value, MASK)
    for pattern in _VALUE_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda m: m.group(1) + " " + MASK, result)
        else:
            result = pattern.sub(MASK, result)
    return result


def _env_secret_values(env_var_names: Iterable[str]) -> list[str]:
    """Resolve the *values* of the named secret env vars that are actually set."""
    values: list[str] = []
    for name in env_var_names:
        value = os.environ.get(name)
        if value:
            values.append(value)
    return values


def redact(obj: Any, secret_env_vars: Iterable[str] = ()) -> Any:
    """Return a deep copy of ``obj`` with secrets masked.

    Parameters
    ----------
    obj:
        A string, mapping or sequence (e.g. a raw event payload).
    secret_env_vars:
        Names of environment variables whose *values*, if present in ``obj``,
        must also be masked.

    Returns
    -------
    Any
        A structurally identical object with secret keys/values replaced by
        :data:`MASK`.
    """
    extra_values = _env_secret_values(secret_env_vars)
    return _redact_inner(obj, extra_values)


def _redact_inner(obj: Any, extra_values: list[str]) -> Any:
    if isinstance(obj, dict):
        redacted: dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and _is_secret_key(key):
                redacted[key] = MASK
            else:
                redacted[key] = _redact_inner(value, extra_values)
        return redacted
    if isinstance(obj, (list, tuple)):
        return [_redact_inner(item, extra_values) for item in obj]
    if isinstance(obj, str):
        return redact_text(obj, extra_values)
    return obj


def contains_secret(obj: Any, needles: Iterable[str]) -> bool:
    """Audit helper: True if any ``needle`` string survives anywhere in ``obj``.

    Used by tests to assert secrets never reach persisted outputs.
    """
    real = [n for n in needles if n]
    if not real:
        return False
    text = repr(obj)
    return any(n in text for n in real)
