"""Smoke tests: the package imports, reports its version, and builds an app."""

from __future__ import annotations

import re

import clef_router
from clef_router import create_app
from clef_router._version import __version__


def test_package_exposes_single_version_source():
    assert clef_router.__version__ == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", clef_router.__version__)


def test_public_surface_is_importable():
    for name in (
        "ClefRouter",
        "AsyncClefRouter",
        "RouterConfig",
        "ClefDecision",
        "RoutingDecision",
        "ClefError",
        "ConfigurationError",
        "ClefAPIError",
        "ClefAuthError",
        "ClefRateLimitError",
        "ClefServerError",
        "ClefResponseError",
        "ClefTimeoutError",
        "ClefNetworkError",
    ):
        assert hasattr(clef_router, name), name


def test_app_builds_with_valid_config():
    from tests.conftest import make_config

    app = create_app(make_config())
    assert app.title == "clef-router"
    assert app.version == clef_router.__version__
