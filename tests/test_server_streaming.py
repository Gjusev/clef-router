"""Tests for streaming SSE, /metrics, and the JSONL decision log."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from clef_router.config import RouterConfig
from clef_router.server import create_app
from tests.conftest import json_response, load_fixture, make_config

CHAT_BODY = {"model": "auto", "messages": [{"role": "user", "content": "hello there"}]}


def build_client(handler, **config_overrides) -> TestClient:
    app = create_app(
        make_config(**config_overrides), transport=httpx.MockTransport(handler)
    )
    return TestClient(app)


def ok_upstream(request: httpx.Request) -> httpx.Response:
    return json_response(load_fixture("decide_cheap.json"))


def sse_frames(text: str) -> list[dict | str]:
    """Parse SSE ``data:`` frames; terminal [DONE] stays a string."""
    frames: list[dict | str] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            frames.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return frames


class TestStreaming:
    def test_stream_true_yields_openai_sse_chunks(self):
        client = build_client(ok_upstream)
        with client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = sse_frames(response.text)
        assert frames[-1] == "[DONE]"
        assert len(frames) == 4
        role_chunk, content_chunk, stop_chunk = frames[0], frames[1], frames[2]
        assert role_chunk["object"] == "chat.completion.chunk"
        assert role_chunk["choices"][0]["delta"] == {"role": "assistant"}
        assert role_chunk["clef"]["tier"] == "cheap"
        decision = json.loads(content_chunk["choices"][0]["delta"]["content"])
        assert decision["tier"] == "cheap"
        assert decision["decision"]["answers"]["team"]["choice"] == "cheap"
        assert stop_chunk["choices"][0]["finish_reason"] == "stop"
        assert stop_chunk["choices"][0]["delta"] == {}

    def test_stream_defaults_to_false(self):
        from clef_router.schemas import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert request.stream is False

    def test_non_stream_response_still_json(self):
        client = build_client(ok_upstream)
        with client:
            response = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["choices"][0]["finish_reason"] == "stop"

    def test_upstream_error_keeps_json_error_shape(self):
        def failing(request: httpx.Request) -> httpx.Response:
            return json_response(load_fixture("error_auth.json"), status_code=401)

        client = build_client(failing)
        with client:
            response = client.post(
                "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
            )
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "upstream_error"


class TestMetrics:
    def test_metrics_render_counters_and_histogram(self):
        client = build_client(ok_upstream)
        with client:
            client.post("/v1/chat/completions", json=CHAT_BODY)
            client.post("/v1/chat/completions", json=CHAT_BODY)
            metrics = client.get("/metrics").text
        assert "clef_router_requests_total{tier=\"cheap\",status=\"ok\"} 2" in metrics
        assert "clef_router_routing_seconds_count 2" in metrics
        assert 'clef_router_routing_seconds_bucket{le="+Inf"} 2' in metrics
        assert "# TYPE clef_router_routing_seconds histogram" in metrics

    def test_decide_calls_are_counted(self):
        decide_body = {
            "state": "hello",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"cheap": "small", "frontier": "big"},
                }
            },
        }
        client = build_client(ok_upstream)
        with client:
            client.post("/v1/decide", json=decide_body)
            metrics = client.get("/metrics").text
        assert 'clef_router_requests_total{tier="decide",status="ok"} 1' in metrics

    def test_errors_counted_with_error_status(self):
        def failing(request: httpx.Request) -> httpx.Response:
            return json_response(load_fixture("error_auth.json"), status_code=401)

        client = build_client(failing)
        with client:
            client.post("/v1/chat/completions", json=CHAT_BODY)
            metrics = client.get("/metrics").text
        assert 'clef_router_requests_total{tier="error",status="error"} 1' in metrics

    def test_empty_metrics_still_valid(self):
        client = build_client(ok_upstream)
        with client:
            metrics = client.get("/metrics").text
        assert "clef_router_routing_seconds_count 0" in metrics


class TestDecisionLog:
    def test_decision_log_gets_one_json_line_per_decision(self, tmp_path):
        log_path = tmp_path / "decisions.jsonl"
        client = build_client(ok_upstream, decision_log=str(log_path))
        with client:
            client.post("/v1/chat/completions", json=CHAT_BODY)
            client.post("/v1/chat/completions", json=CHAT_BODY)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["tier"] == "cheap"
        assert entry["model"] == "clef-flash"
        assert entry["status"] == "ok"
        assert entry["input_tokens"] == 148
        assert entry["prompt_preview"] == "hello there"
        assert "ts" in entry and "latency_ms" in entry

    def test_log_disabled_by_default(self, tmp_path):
        client = build_client(ok_upstream)
        with client:
            client.post("/v1/chat/completions", json=CHAT_BODY)
        assert not (tmp_path / "decisions.jsonl").exists()

    def test_prompt_preview_is_truncated_and_flat(self, tmp_path):
        log_path = tmp_path / "decisions.jsonl"
        client = build_client(ok_upstream, decision_log=str(log_path))
        long_prompt = "word " * 80 + "\nnewlines\ninside"
        with client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": long_prompt}],
                },
            )
        entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert len(entry["prompt_preview"]) == 120
        assert "\n" not in entry["prompt_preview"]


class TestDecisionLogConfig:
    def test_env_variable_sets_decision_log(self, monkeypatch):
        monkeypatch.setenv("CLEF_DECISION_LOG", "/tmp/decisions.jsonl")
        config = RouterConfig.from_env(account_id="a", api_token="t")
        assert config.decision_log == "/tmp/decisions.jsonl"

    def test_empty_env_disables_decision_log(self, monkeypatch):
        monkeypatch.setenv("CLEF_DECISION_LOG", "")
        config = RouterConfig.from_env(account_id="a", api_token="t")
        assert config.decision_log is None
