# Changelog

Notable changes per release. Versions follow semver; the package is
pre-1.0, so minor versions may break.

## 0.3.0

- Streaming: `POST /v1/chat/completions` accepts `"stream": true` and
  answers with OpenAI-shaped SSE chunks (`role`, decision content,
  `finish_reason`, `data: [DONE]`).
- `GET /metrics` exposes Prometheus text: `clef_router_requests_total`,
  `clef_router_routing_seconds` histogram buckets, sums and counts.
- Decision log: set `CLEF_DECISION_LOG` to a path and every routing
  decision appends one JSON line (tier, confidence, reason, latency,
  token usage, truncated prompt preview).
- Docker: `Dockerfile` and `.dockerignore`; `docker run -p 8000:8000`
  serves the proxy with `--host 0.0.0.0`.
- Dependabot for pip and GitHub Actions.

## 0.2.0

Production-quality release.

- Contract-correct Clef client: sync + async, retries with jittered
  exponential backoff, `Retry-After` support, structured errors carrying
  status, Cloudflare error code, and `cf-ray` request ids.
- OpenAI-compatible proxy (`clef-router --port 8000`):
  `POST /v1/chat/completions`, `POST /v1/decide`, `GET /healthz`; plus an
  in-process `ClefOpenAI` client that completes on Workers AI.
- `clef_router.compactor`: one-pass relevance scoring for RAG with
  LangChain and LlamaIndex adapters as extras.
- Eval harness: labeled fixture dataset, offline (mock) and live (api)
  modes, regression-gate composite action, Kaggle kernels (CPU policy
  rerun and GPU real-model rerun).
- Full type hints, `py.typed`, configured logging, env validation at
  startup, CI matrix 3.10/3.11/3.12.
- Measured: policy accuracy 1.00 on the committed fixture bench (n=42);
  real `clef-flash` accuracy 92.9% on a Kaggle T4; library decision
  overhead p50 0.12 ms.

## 0.1.0

Initial release: sync client, CLI, confidence gate escalation.
