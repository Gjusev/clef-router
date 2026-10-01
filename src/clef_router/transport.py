"""HTTP plumbing shared by the sync and async clients.

Everything here is either a pure function or a tiny helper, so the retry
and request-shaping logic lives in exactly one place and is trivially
testable. Payload shaping follows the Clef contract on Workers AI: the
body carries the ``model`` selector, the ``state`` to evaluate, a
``questions`` mapping (noul / choice / score), and optional ``images``.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from typing import Any

from .errors import ClefError
from .models import MAX_IMAGES, MAX_QUESTIONS

__all__ = [
    "RETRYABLE_STATUS_CODES",
    "build_decide_payload",
    "build_chat_state",
    "run_endpoint",
    "auth_headers",
    "is_retryable_status",
    "backoff_delay",
    "retry_after_seconds",
    "request_id_from_headers",
    "extract_error_fields",
    "normalize_images",
]

#: Retryable HTTP status codes: rate limiting plus transient server errors.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_RETRY_AFTER_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")


def build_decide_payload(
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    images: list[dict[str, Any]] | None = None,
    *,
    model_selector: str,
) -> dict[str, Any]:
    """Build the Clef request body for one decision.

    Args:
        state: The content to evaluate (string, object, or array).
        questions: The question mapping, already validated by the caller.
        images: Optional list of image inputs (max 4), each either
            ``{"content_type": ..., "base64": ...}`` or a data URL string.
        model_selector: The required body ``model`` field, ``clef`` or
            ``clef-flash`` (the endpoint URL separately carries the full
            ``@cf/cloudflare/<selector>`` id).

    Returns:
        The JSON-serializable request body.

    Raises:
        ClefError: If the question count is outside the contract range or
            more than four images are supplied.
    """
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ClefError(
            f"a Clef request needs 1..{MAX_QUESTIONS} questions, got {len(questions)}"
        )
    payload: dict[str, Any] = {
        "model": model_selector,
        "state": state,
        "questions": {
            qid: dict(question) for qid, question in questions.items()
        },
    }
    if images:
        if len(images) > MAX_IMAGES:
            raise ClefError(
                f"a Clef request accepts at most {MAX_IMAGES} images, "
                f"got {len(images)}"
            )
        payload["images"] = list(images)
    return payload


def build_chat_state(system_prompt: str | None, prompt: str) -> str:
    """Build the ``state`` string for routing one chat prompt.

    The system prompt, when present, is included as routing instructions so
    Clef can honor caller policy (e.g. "route anything about billing to the
    frontier team") without the caller needing custom questions.
    """
    if system_prompt:
        return f"Instructions:\n{system_prompt}\n\nPrompt:\n{prompt}"
    return f"Prompt:\n{prompt}"


def run_endpoint(base_url: str, account_id: str, model_id: str) -> str:
    """Build the absolute URL of the Workers AI run endpoint.

    The URL always uses the full model id (``@cf/cloudflare/clef`` or
    ``@cf/cloudflare/clef-flash``), never the short selector.
    """
    return f"{base_url.rstrip('/')}/accounts/{account_id}/ai/run/{model_id}"


def auth_headers(api_token: str) -> dict[str, str]:
    """Build default request headers, including the bearer authorization."""
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }


def is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status code warrants retrying the request."""
    return status_code in RETRYABLE_STATUS_CODES


def backoff_delay(attempt: int, base_backoff: float) -> float:
    """Exponential backoff with full jitter for *attempt* (0-based).

    The ceiling grows as ``base_backoff * 2**attempt`` and the actual sleep
    is a uniform random value in ``[0, ceiling)`` so concurrent clients do
    not synchronize their retries.
    """
    ceiling = base_backoff * (2**attempt)
    return random.uniform(0.0, ceiling)


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Parse the ``Retry-After`` header, returning seconds or ``None``.

    Only the seconds form is handled; HTTP-date values return ``None`` and
    the caller falls back to its own backoff schedule.
    """
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    raw = raw.strip()
    if _RETRY_AFTER_PATTERN.match(raw):
        return float(raw)
    return None


def request_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Extract the ``cf-ray`` request id from response headers, if present."""
    ray = headers.get("cf-ray") or headers.get("CF-Ray")
    return ray or None


def extract_error_fields(
    data: Mapping[str, Any],
) -> tuple[str | None, int | str | None]:
    """Pull ``(message, code)`` out of a Cloudflare error response body."""
    errors = data.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            code = first.get("code")
            return (
                message if isinstance(message, str) else None,
                code if isinstance(code, (int, str)) else None,
            )
    return None, None


def normalize_images(images: list[Any] | None) -> list[dict[str, Any]] | None:
    """Pass images through unchanged if they are within the contract limit.

    Accepts both accepted image forms (mapping with ``content_type`` and
    ``base64``, or a data-URL string) and returns them as-is; structural
    validation is left to the API. ``None`` or an empty list means "no
    images".

    Raises:
        ClefError: If more than four images are supplied.
    """
    if not images:
        return None
    if len(images) > MAX_IMAGES:
        raise ClefError(
            f"a Clef request accepts at most {MAX_IMAGES} images, got {len(images)}"
        )
    return list(images)
