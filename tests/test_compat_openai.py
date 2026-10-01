"""In-process OpenAI-compatible client (compat module) with mocked upstreams.

The upstream mock dispatches on URL: Clef decide calls get a Clef envelope,
Workers AI chat calls get a Workers AI chat envelope, exactly like the real
shapes.
"""

from __future__ import annotations

import json

import httpx
import pytest

from clef_router.compat import AsyncClefOpenAI, ClefOpenAI
from clef_router.errors import ClefResponseError
from tests.conftest import json_response, load_fixture, make_config

CHEAP_MODEL = "@cf/meta-llama/llama-3.1-8b-instruct"
FRONTIER_MODEL = "@cf/meta-llama/llama-3.3-70b-instruct"


def dispatching_upstream(request: httpx.Request) -> httpx.Response:
    """Clef calls get Clef envelopes; Workers AI calls get chat envelopes."""
    if "llama" in request.url.path:
        return json_response(load_fixture("workers_chat_response.json"))
    if request.url.path.endswith("clef-flash"):
        return json_response(load_fixture("decide_cheap.json"))
    return json_response(load_fixture("decide_frontier.json"))


def messages(content: str = "Say hi in three words") -> list[dict]:
    return [{"role": "user", "content": content}]


class TestSyncCompat:
    def test_cheap_tier_completion(self):
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            lambda req: (requests.append(req), dispatching_upstream(req))[1]
        )
        client = ClefOpenAI(config=make_config(), transport=transport)
        completion = client.chat.completions.create(
            model="auto", messages=messages(), max_tokens=32
        )
        assert completion["object"] == "chat.completion"
        assert completion["model"] == CHEAP_MODEL
        assert completion["choices"][0]["message"]["content"] == "Hello there, friend"
        assert completion["usage"]["prompt_tokens"] == 9
        routing = completion["clef_routing"]
        assert routing["tier"] == "cheap"
        assert routing["router_model"] == "clef-flash"
        assert routing["router_usage"]["input_tokens"] == 148
        # First call is the Clef decision, second the chat completion.
        assert requests[0].url.path.endswith("/ai/run/@cf/cloudflare/clef-flash")
        assert requests[1].url.path.endswith(f"/ai/run/{CHEAP_MODEL}")
        chat_body = json.loads(requests[1].content)
        assert chat_body["max_tokens"] == 32
        client.close()

    def test_frontier_tier_completion(self):
        def frontier_upstream(request: httpx.Request) -> httpx.Response:
            if "llama" in request.url.path:
                return json_response(load_fixture("workers_chat_response.json"))
            return json_response(load_fixture("decide_frontier.json"))

        transport = httpx.MockTransport(frontier_upstream)
        client = ClefOpenAI(config=make_config(), transport=transport)
        completion = client.chat.completions.create(
            model="auto", messages=messages()
        )
        assert completion["model"] == FRONTIER_MODEL
        assert completion["clef_routing"]["tier"] == "frontier"
        client.close()

    def test_rejects_non_auto_model(self):
        transport = httpx.MockTransport(dispatching_upstream)
        client = ClefOpenAI(config=make_config(), transport=transport)
        with pytest.raises(ClefResponseError, match="model must be 'auto'"):
            client.chat.completions.create(model="gpt-4", messages=messages())
        client.close()

    def test_requires_user_message(self):
        transport = httpx.MockTransport(dispatching_upstream)
        client = ClefOpenAI(config=make_config(), transport=transport)
        with pytest.raises(ClefResponseError, match="user message"):
            client.chat.completions.create(
                model="auto",
                messages=[{"role": "system", "content": "be brief"}],
            )
        client.close()

    def test_list_content_user_message(self):
        transport = httpx.MockTransport(dispatching_upstream)
        client = ClefOpenAI(config=make_config(), transport=transport)
        completion = client.chat.completions.create(
            model="auto",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this "},
                        {"type": "image_url", "image_url": {"url": "x.png"}},
                    ],
                }
            ],
        )
        assert completion["choices"][0]["message"]["content"]
        client.close()

    def test_workers_list_response_form(self):
        def list_response(request: httpx.Request) -> httpx.Response:
            if "llama" in request.url.path:
                return json_response(
                    {
                        "result": {
                            "response": [
                                {"response": "Hello "},
                                {"response": "there"},
                            ]
                        },
                        "success": True,
                    }
                )
            return json_response(load_fixture("decide_cheap.json"))

        transport = httpx.MockTransport(list_response)
        client = ClefOpenAI(config=make_config(), transport=transport)
        completion = client.chat.completions.create(
            model="auto", messages=messages()
        )
        assert completion["choices"][0]["message"]["content"] == "Hello there"
        client.close()

    def test_missing_upstream_text_raises(self):
        def empty_response(request: httpx.Request) -> httpx.Response:
            if "llama" in request.url.path:
                return json_response({"result": {}, "success": True})
            return json_response(load_fixture("decide_cheap.json"))

        transport = httpx.MockTransport(empty_response)
        client = ClefOpenAI(config=make_config(), transport=transport)
        with pytest.raises(ClefResponseError, match="result.response"):
            client.chat.completions.create(model="auto", messages=messages())
        client.close()


class TestAsyncCompat:
    async def test_create_success(self):
        transport = httpx.MockTransport(dispatching_upstream)
        client = AsyncClefOpenAI(config=make_config(), transport=transport)
        completion = await client.chat.completions.create(
            model="auto", messages=messages(), temperature=0.2
        )
        assert completion["choices"][0]["message"]["content"] == "Hello there, friend"
        assert completion["clef_routing"]["tier"] == "cheap"
        await client.close()

    async def test_async_context_manager(self):
        transport = httpx.MockTransport(dispatching_upstream)
        async with AsyncClefOpenAI(
            config=make_config(), transport=transport
        ) as client:
            completion = await client.chat.completions.create(
                model="auto", messages=messages()
            )
        assert completion["clef_routing"]["tier"] == "cheap"
        assert client.router._client.is_closed

    async def test_rejects_non_auto_model_async(self):
        transport = httpx.MockTransport(dispatching_upstream)
        client = AsyncClefOpenAI(config=make_config(), transport=transport)
        with pytest.raises(ClefResponseError, match="model must be 'auto'"):
            await client.chat.completions.create(model="claude", messages=messages())
        await client.close()
