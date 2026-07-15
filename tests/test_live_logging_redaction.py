"""Tests for src.live_logging.redaction (Phase 9, offline)."""

from __future__ import annotations

from src.live_logging.redaction import (
    MASK,
    contains_secret,
    redact,
    redact_text,
)


def test_secret_keys_are_masked():
    out = redact({"username": "netops", "password": "hunter2", "api_key": "abc123"})
    assert out["username"] == "netops"
    assert out["password"] == MASK
    assert out["api_key"] == MASK


def test_nested_structures_are_masked():
    out = redact({"a": [{"client_secret": "x"}, {"ok": "y"}]})
    assert out["a"][0]["client_secret"] == MASK
    assert out["a"][1]["ok"] == "y"


def test_value_patterns_masked_in_text():
    assert "eyJ" not in redact_text('authorization="Bearer eyJabc.def"')
    assert "public" not in redact_text("snmp-server community public ro")
    pem = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
    assert "abc" not in redact_text(pem)


def test_env_var_values_masked(monkeypatch):
    monkeypatch.setenv("SOPHOS_CLIENT_SECRET", "topsecretvalue")
    out = redact({"note": "leaked topsecretvalue here"}, secret_env_vars=["SOPHOS_CLIENT_SECRET"])
    assert "topsecretvalue" not in out["note"]


def test_contains_secret_helper():
    assert contains_secret({"x": "abc"}, ["abc"]) is True
    assert contains_secret({"x": MASK}, ["abc"]) is False
    assert contains_secret({"x": "y"}, []) is False
