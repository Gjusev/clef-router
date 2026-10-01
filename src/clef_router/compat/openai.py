"""OpenAI-compatible in-process chat completions backed by Workers AI.

This module complements the HTTP proxy in :mod:`clef_router.server`: point
your code at :class:`ClefOpenAI` and call ``chat.completions.create`` with
the familiar OpenAI shapes. Clef picks the tier and the completion runs on
a Workers AI model from the same Cloudflare account, so no OpenAI key or
extra service is involved.

The response is a plain dict in the OpenAI ``chat.completion`` wire format
(openai SDK objects ignore unknown fields), plus a ``clef_routing``
extension key that records the decision — the analogue of the proxy's
``clef`` extension and laya's ``x-laya-*`` headers.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

from ..client import AsyncClefRouter, ClefRouter
from ..config import RouterConfig
from ..errors import ClefResponseError
from ..models import TIER_CHEAP, TIER_FRONTIER

logger = logging.getLogger("clef_router")

__all__ = ["ClefOpenAI", "AsyncClefOpenAI"]

DEFAULT_CHEAP_MODEL = "@cf/meta-llama/llama-3.1-8b-instruct"
DEFAULT_FRONTIER_MODEL = "@cf/meta-llama/llama-3.3-70b-instruct"


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    """Extract the most recent user message content for the router."""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping)
                )
    raise ClefResponseError("messages must contain at least one user message")


def _model_path(account_id: str, model_id: str) -> str:
    """Relative URL path for running *model_id* under *account_id*."""
    return f"/accounts/{account_id}/ai/run/{model_id}"


def _build_chat_payload(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Build a Workers AI chat completion payload (streaming excluded)."""
    payload: dict[str, Any] = {"messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _extract_completion_text(data: Mapping[str, Any]) -> str:
    """Pull the assistant text out of a Workers AI chat response."""
    result = data.get("result")
    if isinstance(result, Mapping):
        response = result.get("response")
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            return "".join(
                str(part.get("response", ""))
                for part in response
                if isinstance(part, Mapping)
            )
    raise ClefResponseError("Workers AI response has no 'result.response' text")


def _chat_completion_response(
    *,
    model: str,
    content: str,
    usage: Mapping[str, Any] | None,
    routing: dict[str, Any],
) -> dict[str, Any]:
    """Assemble an OpenAI-shaped ``chat.completion`` response dict."""
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "clef_routing": routing,
    }
    if usage is not None:
        response["usage"] = dict(usage)
    return response


def _resolve_model(tier: str, cheap_model: str, frontier_model: str) -> str:
    """Map a routing tier to its configured Workers AI model."""
    if tier == TIER_CHEAP:
        return cheap_model
    if tier == TIER_FRONTIER:
        return frontier_model
    raise ClefResponseError(f"unexpected routing tier {tier!r}")


def _routing_extension(routing: Any, router_model: str) -> dict[str, Any]:
    """Build the ``clef_routing`` extension recorded on the response."""
    return {
        "tier": routing.tier,
        "reason": routing.reason,
        "latency_ms": routing.decision.latency_ms,
        "router_model": router_model,
        "router_usage": routing.decision.usage.to_dict(),
    }


class ClefOpenAI:
    """OpenAI-compatible sync interface: ``client.chat.completions.create``.

    Example:
        >>> client = ClefOpenAI()  # doctest: +SKIP
        >>> completion = client.chat.completions.create(  # doctest: +SKIP
        ...     model="auto",
        ...     messages=[{"role": "user", "content": "Say hi in three words"}],
        ... )
        >>> completion["choices"][0]["message"]["content"]  # doctest: +SKIP
        'Hello there, friend'
        >>> completion["clef_routing"]["tier"]  # doctest: +SKIP
        'cheap'

    Attributes:
        cheap_model: Workers AI model used for the cheap tier.
        frontier_model: Workers AI model used for the frontier tier.
        router: The underlying :class:`ClefRouter` making decisions.
    """

    def __init__(
        self,
        router: ClefRouter | None = None,
        *,
        cheap_model: str = DEFAULT_CHEAP_MODEL,
        frontier_model: str = DEFAULT_FRONTIER_MODEL,
        config: RouterConfig | None = None,
        transport: Any = None,
    ) -> None:
        """Create the interface; a router is created lazily if not given."""
        self.cheap_model = cheap_model
        self.frontier_model = frontier_model
        self._router = router
        self._config = config
        self._transport = transport

    @property
    def router(self) -> ClefRouter:
        """The underlying sync router, created on first access."""
        if self._router is None:
            self._router = ClefRouter(config=self._config, transport=self._transport)
        return self._router

    @property
    def chat(self) -> _SyncChat:
        """Entry point mirroring the OpenAI SDK: ``.chat.completions``."""
        return _SyncChat(self)

    def _run_model(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one chat completion on Workers AI with the router's retries."""
        return self.router._post(
            _model_path(self.router.config.account_id, model), payload
        )

    def _create(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Shared sync implementation for ``completions.create``."""
        model = kwargs.get("model", "auto")
        if model not in (None, "auto"):
            raise ClefResponseError(
                "model must be 'auto'; clef-router chooses the model itself"
            )
        messages = kwargs.get("messages") or []
        prompt = _last_user_content(messages)
        routing = self.router.route(prompt)
        completion_model = _resolve_model(
            routing.tier, self.cheap_model, self.frontier_model
        )
        logger.debug(
            "routing to %s model %s (reason: %s)",
            routing.tier,
            completion_model,
            routing.reason,
        )
        data = self._run_model(
            completion_model,
            _build_chat_payload(
                messages,
                max_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
            ),
        )
        result = data.get("result")
        return _chat_completion_response(
            model=completion_model,
            content=_extract_completion_text(data),
            usage=(
                result.get("usage") if isinstance(result, Mapping) else None
            ),
            routing=_routing_extension(routing, self.router.config.model_selector),
        )

    def close(self) -> None:
        """Release the underlying router's connections, if it was created."""
        if self._router is not None:
            self._router.close()

    def __enter__(self) -> ClefOpenAI:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _SyncChat:
    """``chat`` namespace of :class:`ClefOpenAI`."""

    def __init__(self, owner: ClefOpenAI) -> None:
        self._owner = owner

    @property
    def completions(self) -> _SyncCompletions:
        """The ``completions`` namespace."""
        return _SyncCompletions(self._owner)


class _SyncCompletions:
    """``chat.completions`` namespace of :class:`ClefOpenAI`."""

    def __init__(self, owner: ClefOpenAI) -> None:
        self._owner = owner

    def create(
        self,
        *,
        model: str = "auto",
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Route *messages* and return an OpenAI-shaped completion dict."""
        return self._owner._create(
            {
                "model": model,
                "messages": messages or [],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )


class AsyncClefOpenAI:
    """Asynchronous counterpart of :class:`ClefOpenAI` (awaitable create)."""

    def __init__(
        self,
        router: AsyncClefRouter | None = None,
        *,
        cheap_model: str = DEFAULT_CHEAP_MODEL,
        frontier_model: str = DEFAULT_FRONTIER_MODEL,
        config: RouterConfig | None = None,
        transport: Any = None,
    ) -> None:
        """Create the interface; a router is created lazily if not given."""
        self.cheap_model = cheap_model
        self.frontier_model = frontier_model
        self._router = router
        self._config = config
        self._transport = transport

    @property
    def router(self) -> AsyncClefRouter:
        """The underlying async router, created on first access."""
        if self._router is None:
            self._router = AsyncClefRouter(
                config=self._config, transport=self._transport
            )
        return self._router

    @property
    def chat(self) -> _AsyncChat:
        """Entry point mirroring the OpenAI SDK: ``.chat.completions``."""
        return _AsyncChat(self)

    async def _run_model(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one chat completion on Workers AI with the router's retries."""
        return await self.router._post(
            _model_path(self.router.config.account_id, model), payload
        )

    async def _create(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Shared async implementation for ``completions.create``."""
        model = kwargs.get("model", "auto")
        if model not in (None, "auto"):
            raise ClefResponseError(
                "model must be 'auto'; clef-router chooses the model itself"
            )
        messages = kwargs.get("messages") or []
        prompt = _last_user_content(messages)
        routing = await self.router.route(prompt)
        completion_model = _resolve_model(
            routing.tier, self.cheap_model, self.frontier_model
        )
        logger.debug(
            "routing to %s model %s (reason: %s)",
            routing.tier,
            completion_model,
            routing.reason,
        )
        data = await self._run_model(
            completion_model,
            _build_chat_payload(
                messages,
                max_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
            ),
        )
        result = data.get("result")
        return _chat_completion_response(
            model=completion_model,
            content=_extract_completion_text(data),
            usage=(
                result.get("usage") if isinstance(result, Mapping) else None
            ),
            routing=_routing_extension(routing, self.router.config.model_selector),
        )

    async def close(self) -> None:
        """Release the underlying router's connections, if it was created."""
        if self._router is not None:
            await self.router.close()

    async def __aenter__(self) -> AsyncClefOpenAI:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


class _AsyncChat:
    """``chat`` namespace of :class:`AsyncClefOpenAI`."""

    def __init__(self, owner: AsyncClefOpenAI) -> None:
        self._owner = owner

    @property
    def completions(self) -> _AsyncCompletions:
        """The ``completions`` namespace."""
        return _AsyncCompletions(self._owner)


class _AsyncCompletions:
    """``chat.completions`` namespace of :class:`AsyncClefOpenAI`."""

    def __init__(self, owner: AsyncClefOpenAI) -> None:
        self._owner = owner

    async def create(
        self,
        *,
        model: str = "auto",
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Route *messages* and return an OpenAI-shaped completion dict."""
        return await self._owner._create(
            {
                "model": model,
                "messages": messages or [],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
