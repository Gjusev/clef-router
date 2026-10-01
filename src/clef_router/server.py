"""FastAPI server exposing clef-router as an OpenAI-compatible proxy.

Endpoints:

* ``POST /v1/chat/completions`` — OpenAI Chat Completions in, an OpenAI
  chat completion out whose assistant message content is the JSON of the
  Clef routing decision. Questions come from the ``clef_questions``
  extension field (what the OpenAI SDKs send via ``extra_body``); without
  them the default routing question set is used and the system prompt is
  passed to Clef as routing instructions.
* ``POST /v1/decide`` — native Clef pass-through with typed models.
* ``GET /healthz`` — config summary plus optional upstream reachability
  probe; secrets are never included or logged.

The server never forwards prompts to a completion model: it is a decision
proxy. Clients that want the routed completion call the chosen model
themselves, exactly as with laya-router's decision headers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from ._version import __version__
from .client import AsyncClefRouter
from .config import RouterConfig
from .decision_log import append_decision_log
from .errors import ClefAPIError, ClefRateLimitError
from .metrics import RouterMetrics
from .schemas import (
    ChatCompletionRequest,
    DecideRequest,
    DecideResponse,
    answers_to_out,
)

logger = logging.getLogger("clef_router")

__all__ = ["create_app", "main"]


def _error_response(exc: ClefAPIError) -> JSONResponse:
    """Map a client exception to a JSON error response.

    Upstream rate limits surface as 429 with the server's ``Retry-After``
    honored; every other upstream failure surfaces as 502 so callers can
    distinguish "the proxy's upstream is unhappy" from "your request was
    invalid" (which FastAPI answers with 422 before the client runs).
    """
    if isinstance(exc, ClefRateLimitError):
        headers = (
            {"Retry-After": str(int(exc.retry_after))}
            if exc.retry_after is not None and exc.retry_after >= 1
            else None
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": str(exc),
                    "type": "rate_limit_error",
                    "code": exc.error_code,
                    "request_id": exc.request_id,
                }
            },
            headers=headers,
        )
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": str(exc),
                "type": "upstream_error",
                "code": exc.error_code,
                "request_id": exc.request_id,
            }
        },
    )


def _last_user_text(messages: list[Any]) -> str:
    """Extract the most recent user message's text for routing.

    List-of-parts content contributes its text parts; a request with no
    user message is a client error.
    """
    for message in reversed(messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
    raise ValueError("messages must contain at least one user message")


def _system_text(messages: list[Any]) -> str | None:
    """Return the first system message's text, if any."""
    for message in messages:
        if message.role == "system":
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                ).strip()
                if text:
                    return text
    return None


def _completion_response(
    *,
    model_selector: str,
    routing: Any,
    latency_ms: float,
) -> dict[str, Any]:
    """Assemble the OpenAI-shaped chat completion carrying the decision."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_selector,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(routing.to_dict()),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": routing.decision.usage.input_tokens,
            "completion_tokens": routing.decision.usage.output_tokens,
            "total_tokens": (
                routing.decision.usage.input_tokens
                + routing.decision.usage.output_tokens
            ),
        },
        "clef": {
            "tier": routing.tier,
            "reason": routing.reason,
            "latency_ms": latency_ms,
            "router_version": __version__,
        },
    }


def _sse(payload: dict[str, Any]) -> str:
    """Format one Server-Sent Events data frame."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _streaming_response(
    *,
    model_selector: str,
    routing: Any,
    latency_ms: float,
) -> StreamingResponse:
    """Render the decision as an OpenAI-shaped SSE completion stream.

    Three chunks (role, content, stop) followed by ``data: [DONE]``, so any
    OpenAI SDK streaming client works unchanged. The clef extension rides in
    the first chunk.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(
        delta: dict[str, Any], finish_reason: str | None = None
    ) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_selector,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    async def generate() -> AsyncIterator[str]:
        first = chunk({"role": "assistant"})
        first["clef"] = {
            "tier": routing.tier,
            "reason": routing.reason,
            "latency_ms": latency_ms,
            "router_version": __version__,
        }
        yield _sse(first)
        yield _sse(chunk({"content": json.dumps(routing.to_dict())}))
        yield _sse(chunk({}, finish_reason="stop"))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(
    config: RouterConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the clef-router FastAPI application.

    Args:
        config: Router configuration; built from the environment when
            omitted and validated before the app accepts traffic.
        transport: Optional httpx async transport (used by tests to mock
            the Cloudflare upstream).

    Returns:
        The configured :class:`fastapi.FastAPI` application.

    Raises:
        ConfigurationError: If the resolved config is invalid; every
            problem is listed in the message so startup fails loudly.
    """
    if config is None:
        config = RouterConfig.from_env()
    config.ensure_valid()

    router = AsyncClefRouter(config=config, transport=transport)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close the upstream client when the app shuts down."""
        yield
        await router.close()

    app = FastAPI(title="clef-router", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.router = router
    app.state.metrics = metrics = RouterMetrics()

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> JSONResponse | StreamingResponse:
        """Route an OpenAI chat request through Clef and return the decision."""
        try:
            prompt = _last_user_text(request.messages)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": {"message": str(exc), "type": "invalid_request"}},
            )
        system_prompt = _system_text(request.messages)
        try:
            started = time.perf_counter()
            routing = await router.route(
                prompt,
                system_prompt=system_prompt,
                questions=request.to_contract_questions(),
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
        except ClefAPIError as exc:
            logger.warning("routing failed: %s", exc)
            metrics.observe(
                "error", (time.perf_counter() - started) * 1000.0, status="error"
            )
            return _error_response(exc)
        metrics.observe(routing.tier, latency_ms)
        if config.decision_log:
            append_decision_log(
                config.decision_log,
                routing=routing,
                model_selector=config.model_selector,
                latency_ms=latency_ms,
                prompt=prompt,
            )
        if request.stream:
            return _streaming_response(
                model_selector=config.model_selector,
                routing=routing,
                latency_ms=latency_ms,
            )
        return JSONResponse(
            _completion_response(
                model_selector=config.model_selector,
                routing=routing,
                latency_ms=latency_ms,
            )
        )

    @app.post("/v1/decide")
    async def decide(request: DecideRequest) -> JSONResponse:
        """Pass a native Clef request through and return typed answers."""
        state, questions, images = request.to_contract()
        try:
            started = time.perf_counter()
            decision = await router.decide(state, questions, images)
            metrics.observe("decide", (time.perf_counter() - started) * 1000.0)
        except ClefAPIError as exc:
            logger.warning("decide failed: %s", exc)
            metrics.observe(
                "decide", (time.perf_counter() - started) * 1000.0, status="error"
            )
            return _error_response(exc)
        return JSONResponse(
            DecideResponse(
                model=decision.model,
                answers=answers_to_out(decision.answers),
                usage=decide_usage(decision),
            ).model_dump()
        )

    def decide_usage(decision: Any) -> dict[str, int]:
        """Plain usage dict from a decision (kept tiny for the handler)."""
        return {
            "input_tokens": decision.usage.input_tokens,
            "output_tokens": decision.usage.output_tokens,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        """Prometheus text exposition of decision counters and latency."""
        return PlainTextResponse(metrics.render())

    @app.get("/healthz")
    async def healthz(
        probe: bool = Query(
            default=False,
            description="Also probe upstream reachability with an unauthenticated "
            "GET to the API root.",
        ),
    ) -> JSONResponse:
        """Report config shape and, optionally, upstream reachability.

        Secrets are never included: only whether credentials are set.
        """
        summary: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "config": config.redacted_summary(),
            "upstream": {
                "url": router.route_url,
                "reachable": None,
            },
        }
        if probe:
            try:
                response = await router._client.get("/")
                summary["upstream"]["reachable"] = True
                summary["upstream"]["probe_status"] = response.status_code
            except httpx.TransportError as exc:
                summary["upstream"]["reachable"] = False
                summary["upstream"]["probe_error"] = type(exc).__name__
        return JSONResponse(summary)

    return app


def main() -> None:
    """Console entrypoint: run the clef-router server with uvicorn.

    Host and port come from ``--host``/``--port`` or the ``CLEF_HOST``/
    ``CLEF_PORT`` environment variables (defaults ``127.0.0.1:8000``); all
    other configuration comes from the standard environment variables.
    Invalid configuration aborts startup with every problem listed.
    """
    parser = argparse.ArgumentParser(
        prog="clef-router",
        description="OpenAI-compatible routing proxy for Cloudflare's Clef",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CLEF_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1, env CLEF_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CLEF_PORT", "8000")),
        help="Bind port (default: 8000, env CLEF_PORT)",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=("clef", "clef-flash"),
        help="Clef selector override (default: env CLEF_MODEL or clef-flash)",
    )
    args = parser.parse_args()

    config = RouterConfig.from_env(model_selector=args.model)
    config.ensure_valid()
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(create_app(config=config), host=args.host, port=args.port)
