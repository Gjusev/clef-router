"""Tests for shared transport helpers: payload, headers, backoff, parsing."""

from __future__ import annotations

import pytest

from clef_router.errors import ClefError
from clef_router.transport import (
    RETRYABLE_STATUS_CODES,
    auth_headers,
    backoff_delay,
    build_chat_state,
    build_decide_payload,
    extract_error_fields,
    is_retryable_status,
    normalize_images,
    request_id_from_headers,
    retry_after_seconds,
    run_endpoint,
)

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "pick a tier",
        "criteria": {"cheap": "small model", "frontier": "big model"},
    }
}


class TestBuildDecidePayload:
    def test_body_carries_required_model_selector(self) -> None:
        payload = build_decide_payload("state", QUESTIONS, model_selector="clef-flash")
        assert payload["model"] == "clef-flash"
        assert payload["state"] == "state"
        assert payload["questions"] == QUESTIONS

    def test_images_are_included(self) -> None:
        images = [{"content_type": "image/png", "base64": "aGk="}]
        payload = build_decide_payload(
            "s", QUESTIONS, images, model_selector="clef"
        )
        assert payload["images"] == images
        assert payload["model"] == "clef"

    def test_more_than_64_questions_rejected(self) -> None:
        questions = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)}
        with pytest.raises(ClefError, match="1..64"):
            build_decide_payload("s", questions, model_selector="clef")

    def test_more_than_four_images_rejected(self) -> None:
        images = [{"content_type": "image/png", "base64": "aGk="}] * 5
        with pytest.raises(ClefError, match="4 images"):
            build_decide_payload("s", QUESTIONS, images, model_selector="clef")


class TestBuildChatState:
    def test_without_system_prompt(self) -> None:
        assert build_chat_state(None, "hi") == "Prompt:\nhi"

    def test_system_prompt_becomes_routing_instructions(self) -> None:
        state = build_chat_state("Route billing to frontier", "refund me")
        assert state.startswith("Instructions:\nRoute billing to frontier")
        assert "Prompt:\nrefund me" in state


class TestEndpointAndHeaders:
    def test_endpoint_uses_full_model_id(self) -> None:
        url = run_endpoint(
            "https://api.cloudflare.com/client/v4",
            "acct",
            "@cf/cloudflare/clef-flash",
        )
        assert url.endswith("/accounts/acct/ai/run/@cf/cloudflare/clef-flash")

    def test_endpoint_tolerates_trailing_slash(self) -> None:
        url = run_endpoint("https://api.cloudflare.com/client/v4/", "a", "@cf/m")
        assert "//accounts" not in url

    def test_auth_headers_use_bearer_token(self) -> None:
        assert auth_headers("tok") == {
            "Authorization": "Bearer tok",
            "Content-Type": "application/json",
        }


class TestRetryHelpers:
    def test_retryable_statuses(self) -> None:
        assert RETRYABLE_STATUS_CODES == frozenset({429, 500, 502, 503, 504})
        assert is_retryable_status(429)
        assert not is_retryable_status(401)

    def test_backoff_delay_is_bounded_by_exponential_ceiling(self) -> None:
        for attempt in range(4):
            ceiling = 0.5 * (2**attempt)
            for _ in range(50):
                assert 0.0 <= backoff_delay(attempt, 0.5) <= ceiling

    def test_zero_backoff_means_zero_delay(self) -> None:
        assert backoff_delay(3, 0.0) == 0.0

    def test_retry_after_seconds_forms(self) -> None:
        assert retry_after_seconds({"Retry-After": "120"}) == 120.0
        assert retry_after_seconds({"retry-after": "0.5"}) == 0.5
        assert retry_after_seconds({"Retry-After": "Wed, 21 Oct 2026 07:28:00"}) is None
        assert retry_after_seconds({}) is None

    def test_request_id_from_cf_ray(self) -> None:
        assert request_id_from_headers({"cf-ray": "abc123"}) == "abc123"
        assert request_id_from_headers({}) is None

    def test_extract_error_fields(self) -> None:
        data = {"errors": [{"code": 7003, "message": "Invalid request headers"}]}
        assert extract_error_fields(data) == ("Invalid request headers", 7003)
        assert extract_error_fields({}) == (None, None)
        assert extract_error_fields({"errors": ["boom"]}) == (None, None)


class TestNormalizeImages:
    def test_none_and_empty_become_none(self) -> None:
        assert normalize_images(None) is None
        assert normalize_images([]) is None

    def test_images_pass_through_copied(self) -> None:
        images = [{"content_type": "image/png", "base64": "aGk="}]
        result = normalize_images(images)
        assert result == images
        assert result is not images

    def test_more_than_four_rejected(self) -> None:
        with pytest.raises(ClefError, match="4 images"):
            normalize_images([{"content_type": "i", "base64": "x"}] * 5)
