"""Sync and async clients for the Cloudflare Clef decision model.

Both clients share the request shaping, error mapping, and retry policy so
their behavior is identical; only the I/O primitives differ.

The retry policy: retryable failures (timeouts, connection errors, HTTP 429
and transient 5xx) are retried up to ``max_retries`` times with exponential
backoff with jitter, honoring a server-provided ``Retry-After`` when
present. Non-retryable failures (authentication, validation, malformed
responses) raise immediately as structured :class:`~clef_router.errors`
types. No request body, header, or token is ever logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from .config import RouterConfig
from .errors import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
)
from .models import (
    DEFAULT_QUESTIONS,
    ClefDecision,
    RoutingDecision,
    derive_tier,
    parse_decision,
    validate_questions,
)
from .transport import (
    auth_headers,
    backoff_delay,
    build_chat_state,
    build_decide_payload,
    extract_error_fields,
    normalize_images,
    request_id_from_headers,
    retry_after_seconds,
    run_endpoint,
)

logger = logging.getLogger("clef_router")

__all__ = ["ClefRouter", "AsyncClefRouter"]


def _raise_for_status(response: httpx.Response) -> None:
    """Map a non-2xx response to the matching structured error."""
    if response.is_success:
        return
    status_code = response.status_code
    request_id = request_id_from_headers(response.headers)
    message, error_code = extract_error_fields(_safe_json(response))
    detail = message or "no error detail in response body"
    if status_code in (401, 403):
        raise ClefAuthError(
            f"Clef API authentication failed ({status_code}): {detail}",
            status_code=status_code,
            request_id=request_id,
            error_code=error_code,
        )
    if status_code == 429:
        raise ClefRateLimitError(
            f"Clef API rate limited (429): {detail}",
            request_id=request_id,
            error_code=error_code,
            retry_after=retry_after_seconds(response.headers),
        )
    if status_code >= 500:
        raise ClefServerError(
            f"Clef API server error ({status_code}): {detail}",
            status_code=status_code,
            request_id=request_id,
            error_code=error_code,
            retryable=True,
        )
    raise ClefAPIError(
        f"Clef API error ({status_code}): {detail}",
        status_code=status_code,
        request_id=request_id,
        error_code=error_code,
        retryable=False,
    )


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    """Return the response body as a dict, or an empty dict on parse failure."""
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _retry_delay(attempt: int, config: RouterConfig, exc: ClefAPIError) -> float:
    """Seconds to wait before retry *attempt* after *exc*."""
    if isinstance(exc, ClefRateLimitError) and exc.retry_after is not None:
        return exc.retry_after
    return backoff_delay(attempt, config.retry_backoff)


def _checked_questions(questions: Mapping[str, Mapping[str, Any]] | None) -> dict[
    str, dict[str, Any]
]:
    """Validate and copy a question mapping, or default to the routing set.

    Raises:
        ClefResponseError: If the mapping violates the Clef contract. (A
            ValueError-shaped problem, but raised through the API error
            tree so server handlers map it uniformly.)
    """
    if questions is None:
        return {qid: dict(q) for qid, q in DEFAULT_QUESTIONS.items()}
    problems = validate_questions(questions)
    if problems:
        raise ClefResponseError(
            "invalid questions: " + "; ".join(problems)
        )
    return {qid: dict(q) for qid, q in questions.items()}


class ClefRouter:
    """Synchronous client for the Cloudflare Clef decision model.

    Example:
        >>> router = ClefRouter()  # doctest: +SKIP
        >>> decision = router.route("Design a rate limiter")  # doctest: +SKIP
        >>> decision.tier  # doctest: +SKIP
        'frontier'

    Configuration comes from ``CLEF_ACCOUNT_ID``/``CLEF_API_TOKEN``
    (with ``CLOUDFLARE_*`` fallbacks) unless overridden explicitly.
    Credentials are validated at construction time.
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        *,
        model_selector: str | None = None,
        base_url: str | None = None,
        min_confidence: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        config: RouterConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a router; see :class:`RouterConfig` for the options.

        Raises:
            ConfigurationError: If the resolved config is invalid. The
                message lists every problem found.
        """
        if config is None:
            config = RouterConfig.from_env(
                account_id=account_id,
                api_token=api_token,
                model_selector=model_selector,
                base_url=base_url,
                min_confidence=min_confidence,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )
        config.ensure_valid()
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
            headers=auth_headers(config.api_token),
            transport=transport,
        )

    @property
    def _route_path(self) -> str:
        """Path of the Clef decision-model endpoint for this account."""
        return f"/accounts/{self.config.account_id}/ai/run/{self.config.model_id}"

    @property
    def route_url(self) -> str:
        """Absolute URL of the Clef decision-model endpoint."""
        return run_endpoint(
            self.config.base_url, self.config.account_id, self.config.model_id
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one payload, mapping transport and HTTP failures to errors."""
        try:
            response = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ClefTimeoutError(
                f"Clef request timed out after {self.config.timeout}s",
                timeout=self.config.timeout,
            ) from exc
        except httpx.TransportError as exc:
            raise ClefNetworkError(f"could not reach Clef API: {exc}") from exc
        _raise_for_status(response)
        return _parse_success_body(response)

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]] | None = None,
        images: list[Any] | None = None,
    ) -> ClefDecision:
        """Evaluate *state* against *questions* and return the raw decision.

        Args:
            state: The content to evaluate (string, object, or array).
            questions: Contract-shaped question mapping; defaults to
                :data:`clef_router.models.DEFAULT_QUESTIONS`.
            images: Optional image inputs (max 4).

        Returns:
            The parsed :class:`ClefDecision`, including token usage.

        Raises:
            ClefError: If the question/image counts violate the contract.
            ClefAPIError: For API failures after the retry budget is spent.
            ClefResponseError: If the response shape is unusable.
        """
        return self._decide_with_retries(
            state, _checked_questions(questions), normalize_images(images)
        )

    def route(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        questions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RoutingDecision:
        """Classify *prompt* and derive the ``cheap``/``frontier`` tier.

        The system prompt, when given, becomes routing instructions inside
        the Clef state. Custom *questions* replace the default set; tier
        derivation still follows :func:`clef_router.models.derive_tier`.
        """
        state = build_chat_state(system_prompt, prompt)
        decision = self.decide(state, questions)
        routing = derive_tier(decision, min_confidence=self.config.min_confidence)
        logger.debug(
            "routed to %s tier (%s) in %.1f ms",
            routing.tier,
            routing.reason,
            decision.latency_ms,
        )
        return routing

    def _decide_with_retries(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        images: list[dict[str, Any]] | None,
    ) -> ClefDecision:
        """Run the retry loop around one decide POST."""
        payload = build_decide_payload(
            state, questions, images, model_selector=self.config.model_selector
        )
        started = time.perf_counter()
        for attempt in range(self.config.max_retries + 1):
            try:
                envelope = self._post(self._route_path, payload)
            except ClefAPIError as exc:
                if not exc.retryable or attempt >= self.config.max_retries:
                    raise
                delay = _retry_delay(attempt, self.config, exc)
                logger.warning(
                    "retryable Clef API error (%s); retry %d/%d in %.2fs",
                    exc,
                    attempt + 1,
                    self.config.max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            return parse_decision(envelope, latency_ms=latency_ms)
        raise AssertionError("unreachable: retry loop must return or raise")

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> ClefRouter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncClefRouter:
    """Asynchronous counterpart of :class:`ClefRouter`.

    Same configuration, request shaping, error mapping, and retry policy as
    the sync client; ``decide``/``route`` are coroutines and the client must
    be closed (or used as an async context manager).
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        *,
        model_selector: str | None = None,
        base_url: str | None = None,
        min_confidence: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        config: RouterConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create an async router; see :class:`RouterConfig` for options."""
        if config is None:
            config = RouterConfig.from_env(
                account_id=account_id,
                api_token=api_token,
                model_selector=model_selector,
                base_url=base_url,
                min_confidence=min_confidence,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )
        config.ensure_valid()
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
            headers=auth_headers(config.api_token),
            transport=transport,
        )

    @property
    def _route_path(self) -> str:
        """Path of the Clef decision-model endpoint for this account."""
        return f"/accounts/{self.config.account_id}/ai/run/{self.config.model_id}"

    @property
    def route_url(self) -> str:
        """Absolute URL of the Clef decision-model endpoint."""
        return run_endpoint(
            self.config.base_url, self.config.account_id, self.config.model_id
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one payload, mapping transport and HTTP failures to errors."""
        try:
            response = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ClefTimeoutError(
                f"Clef request timed out after {self.config.timeout}s",
                timeout=self.config.timeout,
            ) from exc
        except httpx.TransportError as exc:
            raise ClefNetworkError(f"could not reach Clef API: {exc}") from exc
        _raise_for_status(response)
        return _parse_success_body(response)

    async def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]] | None = None,
        images: list[Any] | None = None,
    ) -> ClefDecision:
        """Evaluate *state* against *questions* and return the raw decision."""
        return await self._decide_with_retries(
            state, _checked_questions(questions), normalize_images(images)
        )

    async def route(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        questions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RoutingDecision:
        """Classify *prompt* and derive the tier (awaitable)."""
        state = build_chat_state(system_prompt, prompt)
        decision = await self.decide(state, questions)
        routing = derive_tier(decision, min_confidence=self.config.min_confidence)
        logger.debug(
            "routed to %s tier (%s) in %.1f ms",
            routing.tier,
            routing.reason,
            decision.latency_ms,
        )
        return routing

    async def _decide_with_retries(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        images: list[dict[str, Any]] | None,
    ) -> ClefDecision:
        """Run the retry loop around one decide POST."""
        payload = build_decide_payload(
            state, questions, images, model_selector=self.config.model_selector
        )
        started = time.perf_counter()
        for attempt in range(self.config.max_retries + 1):
            try:
                envelope = await self._post(self._route_path, payload)
            except ClefAPIError as exc:
                if not exc.retryable or attempt >= self.config.max_retries:
                    raise
                delay = _retry_delay(attempt, self.config, exc)
                logger.warning(
                    "retryable Clef API error (%s); retry %d/%d in %.2fs",
                    exc,
                    attempt + 1,
                    self.config.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            return parse_decision(envelope, latency_ms=latency_ms)
        raise AssertionError("unreachable: retry loop must return or raise")

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncClefRouter:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def _parse_success_body(response: httpx.Response) -> dict[str, Any]:
    """Parse a 2xx body, mapping malformed bodies to :class:`ClefResponseError`.

    The Cloudflare error envelope (``{"success": false, "errors": [...]}``)
    can arrive with any HTTP status, including 200, so the flag is checked
    explicitly and surfaces as a structured :class:`ClefAPIError`.
    """
    try:
        data = response.json()
    except ValueError as exc:
        raise ClefResponseError(
            f"Clef API returned a non-JSON body (status {response.status_code})",
            status_code=response.status_code,
            request_id=request_id_from_headers(response.headers),
        ) from exc
    if not isinstance(data, dict):
        raise ClefResponseError(
            "Clef API returned a non-object JSON body",
            status_code=response.status_code,
            request_id=request_id_from_headers(response.headers),
        )
    if data.get("success") is False:
        message, error_code = extract_error_fields(data)
        raise ClefResponseError(
            f"Clef API reported failure: {message or 'no error detail in body'}",
            status_code=response.status_code,
            request_id=request_id_from_headers(response.headers),
            error_code=error_code,
        )
    return data
