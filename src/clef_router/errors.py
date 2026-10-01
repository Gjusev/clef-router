"""Structured exceptions raised by clef-router.

Every exception derives from :class:`ClefError`, so callers can catch a
single type to handle all failure modes this library produces. HTTP-derived
failures raise :class:`ClefAPIError` subclasses that carry the HTTP status
code, a request id (the Cloudflare ``cf-ray`` header when present), the
Cloudflare error code from the body, and a ``retryable`` flag the internal
retry loop uses to decide whether a request may be attempted again.
"""

from __future__ import annotations

__all__ = [
    "ClefError",
    "ConfigurationError",
    "ClefAPIError",
    "ClefAuthError",
    "ClefRateLimitError",
    "ClefServerError",
    "ClefResponseError",
    "ClefTimeoutError",
    "ClefNetworkError",
]


class ClefError(Exception):
    """Base class for every error raised by clef-router."""


class ConfigurationError(ClefError):
    """Raised when configuration is missing or invalid.

    The message lists every problem found, one per line, and names the
    environment variable that can supply each missing value.
    """


class ClefAPIError(ClefError):
    """Raised when the Cloudflare API exchange fails.

    Attributes:
        status_code: HTTP status code of the failed response, or ``None``
            when the failure happened before a response arrived.
        request_id: Value of the ``cf-ray`` response header when present,
            useful when correlating with Cloudflare support.
        error_code: Cloudflare error code from the response body, if any.
        retryable: Whether the caller may retry the same request.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        error_code: int | str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.error_code = error_code
        self.retryable = retryable


class ClefAuthError(ClefAPIError):
    """Raised on HTTP 401/403: the token is missing, invalid, or forbidden.

    Never retried automatically; fix the API token or its permissions.
    """


class ClefRateLimitError(ClefAPIError):
    """Raised on HTTP 429 when the retry budget is exhausted.

    Attributes:
        retry_after: Server-advertised wait in seconds from the
            ``Retry-After`` header, when provided.
    """

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        error_code: int | str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=429,
            request_id=request_id,
            error_code=error_code,
            retryable=True,
        )
        self.retry_after = retry_after


class ClefServerError(ClefAPIError):
    """Raised on transient HTTP 5xx responses from the API."""


class ClefResponseError(ClefAPIError):
    """Raised when the API response does not match the expected shape.

    Covers non-JSON bodies, non-object JSON, a missing ``result`` object,
    and answers with unknown question types. Not retried automatically
    because repeating the same request would produce the same body.
    """


class ClefTimeoutError(ClefAPIError):
    """Raised when a request to the API exceeds the configured timeout.

    Attributes:
        timeout: The configured timeout in seconds.
    """

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id, retryable=True)
        self.timeout = timeout


class ClefNetworkError(ClefAPIError):
    """Raised when the API cannot be reached at all (DNS, socket, TLS).

    Always retryable: connection failures are typically transient, so the
    client's retry loop treats them like timeouts.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)
