"""Shared fixtures and mock-transport helpers for the clef-router tests.

No test talks to the real Cloudflare API: upstreams are simulated with
``httpx.MockTransport`` and canned envelopes shaped exactly like the real
Clef responses (see ``tests/fixtures/``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from clef_router.config import RouterConfig

FIXTURES = Path(__file__).parent / "fixtures"

ACCOUNT_ID = "test-account"
API_TOKEN = "test-token"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a canned API envelope from ``tests/fixtures``."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def make_config(**overrides: Any) -> RouterConfig:
    """A valid config with fast, deterministic retries for tests."""
    values: dict[str, Any] = {
        "account_id": ACCOUNT_ID,
        "api_token": API_TOKEN,
        "retry_backoff": 0.0,
    }
    values.update(overrides)
    return RouterConfig(**values)


def json_response(
    body: dict[str, Any],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """An httpx response carrying a JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=body,
        headers=headers or {},
        request=httpx.Request("POST", "https://example.invalid/"),
    )


def recording_transport(
    responder: Callable[[httpx.Request], httpx.Response],
    requests: list[httpx.Request],
) -> httpx.MockTransport:
    """A MockTransport that records every request before responding."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responder(request)

    return httpx.MockTransport(handler)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A clean environment with valid clef-router credentials."""
    values = {
        "CLEF_ACCOUNT_ID": ACCOUNT_ID,
        "CLEF_API_TOKEN": API_TOKEN,
        "CLEF_RETRY_BACKOFF": "0",
    }
    for name in (
        "CLEF_ACCOUNT_ID",
        "CLEF_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLEF_MODEL",
        "CLEF_TIMEOUT",
        "CLEF_MAX_RETRIES",
        "CLEF_LOG_LEVEL",
        "CLEF_HOST",
        "CLEF_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values
