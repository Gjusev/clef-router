"""Tests for the async client: parity with the sync client's behavior."""

from __future__ import annotations

import httpx
import pytest

from clef_router.client import AsyncClefRouter, ClefRouter
from clef_router.errors import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefTimeoutError,
)
from clef_router.models import TIER_CHEAP, TIER_FRONTIER
from tests.conftest import json_response, load_fixture, make_config, recording_transport


def make_async(responder, **overrides):
    requests: list[httpx.Request] = []
    config = make_config(**overrides)
    transport = recording_transport(responder, requests)
    return AsyncClefRouter(config=config, transport=transport), requests


class TestAsyncRouting:
    async def test_cheap_decision(self) -> None:
        router, requests = make_async(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        routing = await router.route("What is the capital of France?")
        assert routing.tier == TIER_CHEAP
        assert requests[0].url.path.endswith("/ai/run/@cf/cloudflare/clef-flash")
        await router.close()

    async def test_frontier_decision(self) -> None:
        router, _ = make_async(
            lambda req: json_response(load_fixture("decide_frontier.json"))
        )
        routing = await router.route("Write a raytracer")
        assert routing.tier == TIER_FRONTIER
        await router.close()

    async def test_request_body_carries_model_selector(self) -> None:
        router, requests = make_async(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        await router.route("hi")
        import json as _json

        body = _json.loads(requests[0].read())
        assert body["model"] == "clef-flash"
        await router.close()

    async def test_sync_and_async_agree(self) -> None:
        async_router, _ = make_async(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        sync_router = ClefRouter(
            config=make_config(),
            transport=httpx.MockTransport(
                lambda req: json_response(load_fixture("decide_cheap.json"))
            ),
        )
        async_routing = await async_router.route("Summarize this article")
        sync_routing = sync_router.route("Summarize this article")
        assert async_routing.tier == sync_routing.tier
        assert async_routing.reason == sync_routing.reason
        await async_router.close()
        sync_router.close()


class TestAsyncErrors:
    async def test_auth_error_raises_immediately(self) -> None:
        router, requests = make_async(
            lambda req: json_response(
                load_fixture("error_auth.json"), status_code=401
            )
        )
        with pytest.raises(ClefAuthError):
            await router.route("hi")
        assert len(requests) == 1
        await router.close()

    async def test_timeout_maps_to_structured_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        router, _ = make_async(handler, max_retries=0)
        with pytest.raises(ClefTimeoutError):
            await router.route("hi")
        await router.close()

    async def test_network_failure_maps_to_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        router, _ = make_async(handler, max_retries=0)
        with pytest.raises(ClefNetworkError):
            await router.route("hi")
        await router.close()


class TestAsyncRetries:
    async def test_transient_500_then_success(self) -> None:
        attempts: list[int] = []

        def responder(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return json_response(
                    load_fixture("error_server.json"), status_code=500
                )
            return json_response(load_fixture("decide_cheap.json"))

        router, requests = make_async(responder)
        routing = await router.route("hi")
        assert routing.tier == TIER_CHEAP
        assert len(requests) == 2
        await router.close()

    async def test_budget_exhaustion_raises(self) -> None:
        router, requests = make_async(
            lambda req: json_response(
                load_fixture("error_server.json"), status_code=500
            ),
            max_retries=1,
        )
        with pytest.raises(ClefAPIError):
            await router.route("hi")
        assert len(requests) == 2
        await router.close()


class TestAsyncLifecycle:
    async def test_context_manager_closes(self) -> None:
        async with make_async(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )[0] as router:
            await router.route("hi")
        assert router._client.is_closed

    async def test_missing_credentials_fail_fast(self, monkeypatch) -> None:
        monkeypatch.delenv("CLEF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        with pytest.raises(Exception, match="CLEF_ACCOUNT_ID"):
            AsyncClefRouter()
