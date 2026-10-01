"""clef-router: Route prompts between cheap and frontier LLMs using Cloudflare's Clef.

Quickstart as a library::

    from clef_router import ClefRouter

    router = ClefRouter()  # reads CLEF_ACCOUNT_ID / CLEF_API_TOKEN
    routing = router.route("What is the capital of France?")
    print(routing.tier)  # "cheap" or "frontier"

Quickstart as an OpenAI-compatible proxy::

    clef-router --port 8000   # POST /v1/chat/completions, /v1/decide, /healthz

The async mirror is :class:`AsyncClefRouter`, and
:class:`clef_router.compat.ClefOpenAI` exposes an in-process
``chat.completions`` interface that completes on the tier the decision
picks.
"""

from __future__ import annotations

from ._version import __version__
from .client import AsyncClefRouter, ClefRouter
from .config import RouterConfig
from .errors import (
    ClefAPIError,
    ClefAuthError,
    ClefError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
    ConfigurationError,
)
from .models import (
    DEFAULT_QUESTIONS,
    ChoiceAnswer,
    ClefDecision,
    NoulAnswer,
    RoutingDecision,
    ScoreAnswer,
    Usage,
    derive_tier,
)
from .server import create_app
from .transport import build_decide_payload

__all__ = [
    "__version__",
    "ClefRouter",
    "AsyncClefRouter",
    "RouterConfig",
    "create_app",
    "ClefDecision",
    "RoutingDecision",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "Usage",
    "DEFAULT_QUESTIONS",
    "derive_tier",
    "build_decide_payload",
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

_LAZY_EXPORTS = {"create_app": "clef_router.server"}


def __getattr__(name: str) -> object:
    """Lazily export server symbols so ``import clef_router`` stays light.

    The HTTP server pulls in FastAPI and uvicorn; resolving ``create_app``
    only when accessed keeps the library import free of web-framework
    overhead.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
