"""Tests for the retry policy: backoff, Retry-After, budget exhaustion."""

from __future__ import annotations

import logging

import httpx
import pytest

from clef_router.client import ClefRouter
from clef_router.errors import ClefRateLimitError, ClefServerError, ClefTimeoutError
from clef_router.models import TIER_CHEAP
from tests.conftest import json_response, load_fixture, make_config, recording_transport


def router_for(responder, **overrides) -> tuple[ClefRouter, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    config = make_config(**overrides)
    transport = recording_transport(responder, requests)
    return ClefRouter(config=config, transport=transport), requests


def ok() -> httpx.Response:
    return json_response(load_fixture("decide_cheap.json"))


class TestSyncRetries:
    def test_transient_500_then_success(self) -> None:
        attempts: list[int] = []

        def responder(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                return json_response(
                    load_fixture("error_server.json"), status_code=500
                )
            return ok()

        router, requests = router_for(responder)
        routing = router.route("hi")
        assert routing.tier == TIER_CHEAP
        assert len(requests) == 3
        router.close()

    def test_retry_budget_exhausted_raises_server_error(self) -> None:
        router, requests = router_for(
            lambda req: json_response(
                load_fixture("error_server.json"), status_code=500
            )
        )
        with pytest.raises(ClefServerError):
            router.route("hi")
        assert len(requests) == 3  # first attempt + 2 retries (default)
        router.close()

    def test_non_retryable_error_skips_retries(self) -> None:
        router, requests = router_for(
            lambda req: json_response(
                load_fixture("error_auth.json"), status_code=401
            )
        )
        with pytest.raises(ClefServerError.__mro__[1]):  # ClefAPIError family
            router.route("hi")
        assert len(requests) == 1
        router.close()

    def test_rate_limit_honors_retry_after_then_succeeds(self) -> None:
        attempts: list[int] = []

        def responder(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return json_response(
                    load_fixture("error_rate_limited.json"),
                    status_code=429,
                    headers={"Retry-After": "0"},
                )
            return ok()

        router, requests = router_for(responder)
        routing = router.route("hi")
        assert routing.tier == TIER_CHEAP
        assert len(requests) == 2
        router.close()

    def test_exhausted_rate_limit_exposes_retry_after(self) -> None:
        router, _ = router_for(
            lambda req: json_response(
                load_fixture("error_rate_limited.json"),
                status_code=429,
                headers={"Retry-After": "7"},
            ),
            max_retries=0,
        )
        with pytest.raises(ClefRateLimitError) as excinfo:
            router.route("hi")
        assert excinfo.value.retry_after == 7.0
        router.close()

    def test_retries_are_logged_as_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        attempts: list[int] = []

        def responder(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return json_response(
                    load_fixture("error_server.json"), status_code=500
                )
            return ok()

        router, _ = router_for(responder)
        with caplog.at_level(logging.WARNING, logger="clef_router"):
            router.route("hi")
        assert any("retry" in r.message.lower() for r in caplog.records)
        router.close()

    def test_timeout_is_retried_then_succeeds(self) -> None:
        attempts: list[int] = []

        def responder(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ReadTimeout("slow", request=request)
            return ok()

        router, _ = router_for(responder)
        routing = router.route("hi")
        assert routing.tier == TIER_CHEAP
        assert len(attempts) == 2
        router.close()

    def test_transport_error_is_retried(self) -> None:
        attempts: list[int] = []

        def responder(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ConnectError("boom", request=request)
            return ok()

        router, _ = router_for(responder)
        router.route("hi")
        assert len(attempts) == 2
        router.close()

    def test_timeout_error_carries_configured_timeout(self) -> None:
        def responder(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        router, _ = router_for(responder, max_retries=0, timeout=3.5)
        with pytest.raises(ClefTimeoutError) as excinfo:
            router.route("hi")
        assert excinfo.value.timeout == 3.5
        router.close()
