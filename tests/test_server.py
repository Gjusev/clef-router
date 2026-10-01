"""Server tests: OpenAI-compatible proxy endpoints against mocked upstreams."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from clef_router.config import RouterConfig
from clef_router.errors import ConfigurationError
from clef_router.server import create_app
from tests.conftest import json_response, load_fixture, make_config


def build_client(handler) -> TestClient:
    """A TestClient wired to a mocked Cloudflare upstream."""
    app = create_app(make_config(), transport=httpx.MockTransport(handler))
    return TestClient(app)


CHAT_BODY = {"model": "auto", "messages": [{"role": "user", "content": "hello there"}]}


class TestChatCompletions:
    def test_returns_openai_completion_with_decision_json(self):
        client = build_client(
            lambda req: json_response(load_fixture("decide_cheap.json"))
        )
        with client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "clef-flash"
        assert body["id"].startswith("chatcmpl-")
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["role"] == "assistant"
        decision = json.loads(body["choices"][0]["message"]["content"])
        assert decision["tier"] == "cheap"
        assert decision["decision"]["usage"] == {
            "input_tokens": 148,
            "output_tokens": 12,
        }
        assert body["usage"] == {
            "prompt_tokens": 148,
            "completion_tokens": 12,
            "total_tokens": 160,
        }
        assert body["clef"]["tier"] == "cheap"

    def test_frontier_decision_content(self):
        client = build_client(
            lambda req: json_response(load_fixture("decide_frontier.json"))
        )
        with client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY)
        decision = json.loads(response.json()["choices"][0]["message"]["content"])
        assert decision["tier"] == "frontier"

    def test_default_questions_sent_upstream(self):
        seen: dict[str, httpx.Request] = {}

        def upstream(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return json_response(load_fixture("decide_cheap.json"))

        client = build_client(upstream)
        with client:
            client.post("/v1/chat/completions", json=CHAT_BODY)
        body = json.loads(seen["request"].content)
        assert body["model"] == "clef-flash"
        assert set(body["questions"]) == {"urgency", "team"}
        assert body["questions"]["team"]["type"] == "choice"
        assert set(body["questions"]["team"]["criteria"]) == {"cheap", "frontier"}

    def test_clef_questions_passthrough(self):
        seen: dict[str, httpx.Request] = {}

        def upstream(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return json_response(load_fixture("decide_cheap.json"))

        custom = {
            "security": {
                "type": "noul",
                "instructions": "Is this a security incident?",
                "criteria": {"true": "yes", "false": "no"},
            }
        }
        client = build_client(upstream)
        with client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "clef_questions": custom}
            )
        assert response.status_code == 200
        body = json.loads(seen["request"].content)
        assert body["questions"] == custom

    def test_system_prompt_becomes_state_instructions(self):
        seen: dict[str, httpx.Request] = {}

        def upstream(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return json_response(load_fixture("decide_cheap.json"))

        client = build_client(upstream)
        messages = [
            {"role": "system", "content": "Route anything about billing carefully."},
            {"role": "user", "content": "invoice question"},
        ]
        with client:
            client.post("/v1/chat/completions", json={"messages": messages})
        body = json.loads(seen["request"].content)
        assert body["state"] == (
            "Instructions:\nRoute anything about billing carefully."
            "\n\nPrompt:\ninvoice question"
        )

    def test_list_content_user_message(self):
        seen: dict[str, httpx.Request] = {}

        def upstream(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return json_response(load_fixture("decide_cheap.json"))

        client = build_client(upstream)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at "},
                    {"type": "image_url", "image_url": {"url": "x.png"}},
                    {"type": "text", "text": "this chart"},
                ],
            }
        ]
        with client:
            client.post("/v1/chat/completions", json={"messages": messages})
        body = json.loads(seen["request"].content)
        assert "look at" in body["state"]
        assert "this chart" in body["state"]

    def test_no_user_message_is_422(self):
        client = build_client(lambda req: json_response({}))
        with client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "system", "content": "hi"}]},
            )
        assert response.status_code == 422
        assert response.json()["error"]["type"] == "invalid_request"

    def test_upstream_rate_limit_maps_to_429_with_header(self):
        def limited(request: httpx.Request) -> httpx.Response:
            return json_response(
                load_fixture("error_envelope.json"),
                status_code=429,
                headers={"Retry-After": "9", "cf-ray": "ray429"},
            )

        client = build_client(limited)
        with client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "9"
        error = response.json()["error"]
        assert error["type"] == "rate_limit_error"
        assert error["request_id"] == "ray429"

    def test_upstream_auth_failure_maps_to_502(self):
        def unauthorized(request: httpx.Request) -> httpx.Response:
            return json_response(
                {"errors": [{"code": 10000, "message": "Authentication error"}]},
                status_code=401,
                headers={"cf-ray": "ray401"},
            )

        client = build_client(unauthorized)
        with client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert response.status_code == 502
        error = response.json()["error"]
        assert error["type"] == "upstream_error"
        assert error["request_id"] == "ray401"
        assert "401" in error["message"]

    def test_malformed_request_is_422(self):
        client = build_client(lambda req: json_response({}))
        with client:
            response = client.post("/v1/chat/completions", json={"model": "auto"})
        assert response.status_code == 422


class TestDecide:
    def test_native_pass_through(self):
        seen: dict[str, httpx.Request] = {}

        def upstream(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return json_response(load_fixture("decide_full.json"))

        client = build_client(upstream)
        payload = {
            "state": "Assess this request",
            "questions": {
                "urgency": {"type": "noul", "instructions": "Urgent?"},
                "team": {
                    "type": "choice",
                    "instructions": "Pick a team",
                    "criteria": {"cheap": "x", "frontier": "y"},
                },
                "quality": {
                    "type": "score",
                    "instructions": "Rate quality",
                    "criteria": ["poor", "fair", "good", "excellent"],
                },
            },
            "images": [{"content_type": "image/png", "data": "AAA"}],
        }
        with client:
            response = client.post("/v1/decide", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "clef"
        assert body["answers"]["urgency"] == {"type": "noul", "noul": 0.44}
        assert body["answers"]["team"]["choice"] == "cheap"
        assert body["answers"]["quality"]["score"] == 3.0
        assert body["answers"]["quality"]["legend"]["3"] == "good"
        assert body["usage"] == {"input_tokens": 320, "output_tokens": 22}

        upstream_body = json.loads(seen["request"].content)
        assert upstream_body["images"] == [
            {"content_type": "image/png", "base64": "AAA"}
        ]
        assert upstream_body["questions"]["quality"]["criteria"] == [
            "poor", "fair", "good", "excellent",
        ]

    def test_empty_questions_is_422(self):
        client = build_client(lambda req: json_response({}))
        with client:
            response = client.post("/v1/decide", json={"state": "s", "questions": {}})
        assert response.status_code == 422

    def test_bad_choice_criteria_is_422(self):
        client = build_client(lambda req: json_response({}))
        with client:
            response = client.post(
                "/v1/decide",
                json={
                    "state": "s",
                    "questions": {
                        "team": {
                            "type": "choice",
                            "instructions": "pick",
                            "criteria": {"only": "one"},
                        }
                    },
                },
            )
        assert response.status_code == 422
        assert "2..255" in response.text

    def test_upstream_error_maps_to_502(self):
        def broken(request: httpx.Request) -> httpx.Response:
            return json_response({}, status_code=500)

        client = build_client(broken)
        with client:
            response = client.post(
                "/v1/decide",
                json={
                    "state": "s",
                    "questions": {"q": {"type": "noul", "instructions": "x"}},
                },
            )
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "upstream_error"


class TestHealthz:
    def test_config_summary_without_secrets(self):
        client = build_client(lambda req: json_response({}))
        with client:
            response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["config"]["model_selector"] == "clef-flash"
        assert body["config"]["account_id_set"] is True
        assert body["config"]["api_token_set"] is True
        assert body["upstream"]["reachable"] is None
        assert "test-token" not in response.text
        assert "/ai/run/" in body["upstream"]["url"]

    def test_probe_reachable(self):
        def upstream_root(request: httpx.Request) -> httpx.Response:
            return json_response({"ok": True}, status_code=403)

        client = build_client(upstream_root)
        with client:
            response = client.get("/healthz", params={"probe": "true"})
        body = response.json()
        assert body["upstream"]["reachable"] is True
        assert body["upstream"]["probe_status"] == 403

    def test_probe_unreachable(self):
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = build_client(down)
        with client:
            response = client.get("/healthz", params={"probe": "true"})
        body = response.json()
        assert body["upstream"]["reachable"] is False
        assert body["upstream"]["probe_error"] == "ConnectError"


class TestAppFactory:
    def test_invalid_config_fails_at_startup_listing_all(self):
        config = RouterConfig(account_id="", api_token="", model_selector="gpt-4")
        with pytest.raises(ConfigurationError) as excinfo:
            create_app(config=config)
        message = str(excinfo.value)
        assert "account_id" in message
        assert "api_token" in message
        assert "CLEF_MODEL" in message

    def test_app_factory_reads_env(self, env):
        app = create_app()
        assert app.state.config.account_id == env["CLEF_ACCOUNT_ID"]

    def test_config_env_controls_log_level(self, monkeypatch):
        environ = {
            "CLEF_ACCOUNT_ID": "a",
            "CLEF_API_TOKEN": "t",
            "CLEF_LOG_LEVEL": "ERROR",
        }
        app = create_app(RouterConfig.from_env(environ=environ))
        assert app.state.config.log_level == "ERROR"


class TestEntrypoint:
    def test_main_runs_uvicorn_with_env_host_port(self, env, monkeypatch):
        import clef_router.server as server_module

        seen: dict = {}

        def fake_run(app, *, host, port):
            seen["app"] = app
            seen["host"] = host
            seen["port"] = port

        monkeypatch.setattr(server_module.uvicorn, "run", fake_run)
        monkeypatch.setattr("sys.argv", ["clef-router"])
        monkeypatch.setenv("CLEF_HOST", "0.0.0.0")
        monkeypatch.setenv("CLEF_PORT", "9100")
        server_module.main()
        assert seen["host"] == "0.0.0.0"
        assert seen["port"] == 9100
        assert seen["app"].title == "clef-router"

    def test_main_flag_overrides(self, env, monkeypatch):
        import clef_router.server as server_module

        seen: dict = {}

        def fake_run(app, *, host, port):
            seen.update(host=host, port=port, app=app)

        monkeypatch.setattr(server_module.uvicorn, "run", fake_run)
        monkeypatch.setattr(
            "sys.argv", ["clef-router", "--host", "127.0.0.2", "--port", "8080",
                         "--model", "clef"]
        )
        server_module.main()
        assert seen["host"] == "127.0.0.2"
        assert seen["port"] == 8080
        assert seen["app"].state.config.model_selector == "clef"

    def test_main_fails_loudly_without_credentials(self, monkeypatch):
        import clef_router.server as server_module

        for name in ("CLEF_ACCOUNT_ID", "CLEF_API_TOKEN",
                     "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr("sys.argv", ["clef-router"])
        with pytest.raises(ConfigurationError):
            server_module.main()
