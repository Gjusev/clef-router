"""Tests for the sync ClefRouter: routing, error mapping, lifecycle."""

from __future__ import annotations

import json

import httpx
import pytest

from clef_router.client import ClefRouter
from clef_router.errors import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
    ConfigurationError,
)
from clef_router.models import TIER_CHEAP, TIER_FRONTIER
from tests.conftest import json_response, load_fixture, make_config, recording_transport


def router_for(
    responder, **overrides
) -> tuple[ClefRouter, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    config = make_config(**overrides)
    transport = recording_transport(responder, requests)
    return ClefRouter(config=config, transport=transport), requests


class TestConstruction:
    def test_builds_from_environment(self, env) -> None:
        router = ClefRouter()
        assert router.config.account_id == "test-account"
        router.close()

    def test_missing_credentials_fail_fast(self, monkeypatch) -> None:
        monkeypatch.delenv("CLEF_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        with pytest.raises(ConfigurationError, match="CLEF_ACCOUNT_ID"):
            ClefRouter()

    def test_invalid_selector_fails_fast(self) -> None:
        with pytest.raises(ConfigurationError, match="clef-flash"):
            ClefRouter(account_id="a", api_token="t", model_selector="gpt-4o")

    def test_route_url_targets_selected_model(self, env) -> None:
        router = ClefRouter()
        assert router.route_url == (
            "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run"
            "/@cf/cloudflare/clef-flash"
        )
        router.close()


class TestRouting:
    def test_cheap_decision(self) -> None:
        router, requests = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        routing = router.route("What is the capital of France?")
        assert routing.tier == TIER_CHEAP
        assert routing.decision.latency_ms > 0.0
        router.close()

    def test_frontier_decision(self) -> None:
        router, _ = router_for(
            lambda req: json_response(load_fixture("decide_frontier.json"))
        )
        routing = router.route("Design a rate limiter")
        assert routing.tier == TIER_FRONTIER
        router.close()

    def test_request_contract(self) -> None:
        router, requests = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        router.route("hello", system_prompt="Route billing to frontier")
        request = requests[0]
        assert request.url.path.endswith("/ai/run/@cf/cloudflare/clef-flash")
        assert request.headers["Authorization"] == "Bearer test-token"
        body = json.loads(request.read())
        assert body["model"] == "clef-flash"
        assert set(body["questions"]) == {"urgency", "team"}
        assert body["state"].startswith("Instructions:\nRoute billing to frontier")
        assert body["state"].endswith("Prompt:\nhello")
        router.close()

    def test_custom_questions_replace_defaults(self) -> None:
        router, requests = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        questions = {
            "team": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"cheap": "small", "frontier": "big"},
            }
        }
        router.route("hello", questions=questions)
        body = json.loads(requests[0].read())
        assert set(body["questions"]) == {"team"}
        router.close()

    def test_decide_returns_raw_decision_with_usage(self) -> None:
        router, _ = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        decision = router.decide("Invoice is overdue", questions=None)
        assert decision.usage.input_tokens == 148
        assert decision.usage.output_tokens == 12
        router.close()

    def test_invalid_questions_raise_before_any_http(self) -> None:
        router, requests = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        with pytest.raises(ClefResponseError, match="questions"):
            router.route("hi", questions={})
        assert requests == []
        router.close()


class TestErrorMapping:
    def test_auth_error_not_retryable(self) -> None:
        router, requests = router_for(
            lambda req: json_response(load_fixture("error_auth.json"), status_code=401)
        )
        with pytest.raises(ClefAuthError) as excinfo:
            router.route("hi")
        assert excinfo.value.status_code == 401
        assert excinfo.value.retryable is False
        assert len(requests) == 1
        router.close()

    def test_rate_limit_carries_retry_after(self) -> None:
        router, _ = router_for(
            lambda req: json_response(
                load_fixture("error_rate_limited.json"),
                status_code=429,
                headers={"Retry-After": "0"},
            )
        )
        with pytest.raises(ClefRateLimitError) as excinfo:
            router.route("hi")
        assert excinfo.value.retry_after == 0.0
        assert excinfo.value.retryable is True
        router.close()

    def test_server_error_is_retryable(self) -> None:
        router, _ = router_for(
            lambda req: json_response(
                load_fixture("error_server.json"), status_code=500
            )
        )
        with pytest.raises(ClefServerError):
            router.route("hi")
        router.close()

    def test_error_envelope_detail_is_extracted(self) -> None:
        router, _ = router_for(
            lambda req: json_response(
                load_fixture("error_envelope.json"), status_code=400
            )
        )
        with pytest.raises(ClefAPIError, match="Invalid request headers") as excinfo:
            router.route("hi")
        assert excinfo.value.error_code == 7003
        router.close()

    def test_timeout_maps_to_structured_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        router, _ = router_for(handler, max_retries=0)
        with pytest.raises(ClefTimeoutError):
            router.route("hi")
        router.close()

    def test_network_failure_maps_to_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        router, _ = router_for(handler, max_retries=0)
        with pytest.raises(ClefNetworkError):
            router.route("hi")
        router.close()

    def test_non_json_success_body_raises_response_error(self) -> None:
        router, _ = router_for(
            lambda req: httpx.Response(200, text="<html>oops</html>")
        )
        with pytest.raises(ClefResponseError):
            router.route("hi")
        router.close()

    def test_success_false_envelope_raises_response_error(self) -> None:
        router, _ = router_for(
            lambda req: json_response(load_fixture("error_envelope.json"))
        )
        with pytest.raises(ClefResponseError):
            router.route("hi")
        router.close()


class TestLifecycle:
    def test_context_manager_closes_client(self) -> None:
        router, _ = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        with router as entered:
            assert entered is router
            entered.route("hi")
        assert router._client.is_closed

    def test_close_is_idempotent(self) -> None:
        router, _ = router_for(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        router.close()
        router.close()
