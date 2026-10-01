"""Tests for the CLI: output formats and exit codes."""

from __future__ import annotations

import json

import httpx
import pytest

from clef_router import __version__
from clef_router.cli import build_parser, main
from clef_router.client import ClefRouter
from tests.conftest import json_response, load_fixture


@pytest.fixture
def cli_router(monkeypatch):
    """Replace the CLI's router factory with one using a scripted upstream.

    Returns a setter: ``install("decide_cheap.json")`` makes every CLI run
    replay that fixture (the last one repeats on further attempts).
    """

    def install(*fixture_names: str) -> list[httpx.Request]:
        names = list(fixture_names)
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            index = min(len(requests), len(names) - 1)
            name = names[index]
            requests.append(request)
            fixture = load_fixture(name)
            if fixture.get("success") is False and fixture.get("result") is None:
                # canned API errors: 401 for auth, 500 otherwise
                is_auth = any(
                    e.get("code") == 10000 for e in fixture.get("errors", [])
                )
                return json_response(
                    fixture, status_code=401 if is_auth else 500
                )
            return json_response(fixture)

        monkeypatch.setattr(
            "clef_router.cli.ClefRouter",
            lambda **kwargs: ClefRouter(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )
        return requests

    return install


class TestSuccess:
    def test_human_output(self, env, cli_router, capsys) -> None:
        requests = cli_router("decide_cheap.json")
        code = main(["What is the capital of France?"])
        out = capsys.readouterr().out
        assert code == 0
        assert "tier:       cheap" in out
        assert "reason:" in out
        assert "model:      clef-flash" in out
        assert "input tokens:  148" in out
        assert requests[0].headers["Authorization"] == "Bearer test-token"

    def test_json_output(self, env, cli_router, capsys) -> None:
        cli_router("decide_frontier.json")
        code = main(["Design a rate limiter", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["tier"] == "frontier"
        assert payload["decision"]["model"] == "clef-flash"
        assert "reason" in payload

    def test_system_flag_forwarded(self, env, cli_router, capsys) -> None:
        requests = cli_router("decide_cheap.json")
        code = main(["refund me", "--system", "Billing policy: escalate"])
        body = requests[0].read().decode()
        assert code == 0
        assert "Billing policy: escalate" in body

    def test_model_selector_forwarded(self, env, cli_router, capsys) -> None:
        requests = cli_router("decide_cheap.json")
        code = main(["hi", "--model", "clef"])
        assert code == 0
        assert requests[0].url.path.endswith("/ai/run/@cf/cloudflare/clef")


class TestFailure:
    def test_missing_credentials_exit_1(self, monkeypatch, capsys) -> None:
        for name in (
            "CLEF_ACCOUNT_ID",
            "CLOUDFLARE_ACCOUNT_ID",
            "CLEF_API_TOKEN",
            "CLOUDFLARE_API_TOKEN",
        ):
            monkeypatch.delenv(name, raising=False)
        code = main(["hi"])
        assert code == 1

    def test_api_error_exit_1(self, env, cli_router) -> None:
        cli_router("error_auth.json")
        assert main(["hi"]) == 1

    def test_api_error_retries_by_default(self, env, cli_router) -> None:
        requests = cli_router("error_server.json")
        main(["hi"])
        assert len(requests) == 3  # config default: 2 retries


class TestParser:
    def test_version_flag(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_model_choices_enforced(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["hi", "--model", "gpt-4o"])

    def test_defaults(self) -> None:
        args = build_parser().parse_args(["prompt text"])
        assert args.prompt == "prompt text"
        assert args.system is None
        assert args.model is None
        assert args.json is False
