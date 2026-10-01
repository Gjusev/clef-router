"""Real-API integration tests (opt-in, marked ``integration``).

Excluded from the default run via ``addopts`` in ``pyproject.toml``; run
them explicitly with ``pytest -m integration``. They need real credentials:

    export CLEF_ACCOUNT_ID=...
    export CLEF_API_TOKEN=...

Each test is skipped when credentials are absent so a plain ``pytest -m
integration`` never fails on machines without a Cloudflare token.
"""

from __future__ import annotations

import os

import pytest

from clef_router import ClefRouter
from clef_router.config import RouterConfig
from clef_router.models import TIER_CHEAP, TIER_FRONTIER

pytestmark = pytest.mark.integration

requires_credentials = pytest.mark.skipif(
    not (os.environ.get("CLEF_ACCOUNT_ID") and os.environ.get("CLEF_API_TOKEN")),
    reason="CLEF_ACCOUNT_ID and CLEF_API_TOKEN are required for integration tests",
)


@requires_credentials
def test_route_against_real_clef():
    config = RouterConfig.from_env()
    with ClefRouter(config=config) as router:
        routing = router.route("Design a distributed rate limiter with sliding windows")
    assert routing.tier in (TIER_CHEAP, TIER_FRONTIER)
    assert routing.reason
    assert routing.decision.usage.input_tokens > 0


@requires_credentials
def test_decide_against_real_clef_with_custom_questions():
    config = RouterConfig.from_env()
    questions = {
        "security": {
            "type": "noul",
            "instructions": "Does this request involve a security-sensitive task?",
        }
    }
    with ClefRouter(config=config) as router:
        decision = router.decide(
            "Prompt:\nWrite a poem about trees", questions=questions
        )
    assert "security" in decision.answers
    assert 0.0 <= decision.answers["security"].noul <= 1.0
