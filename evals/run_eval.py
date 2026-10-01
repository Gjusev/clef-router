"""Reproducible evaluation harness for the clef-router routing policy.

Modes
-----

``mock`` (default, offline)
    Replays the per-row ``simulated_answer`` fixtures committed in the
    dataset through the full real pipeline (HTTP mock -> envelope parsing ->
    tier derivation). Measures **policy accuracy** -- does the tier policy
    (confidence gate, urgency fallback, fail-safe escalation) map each
    labeled decision to the expected tier -- plus library overhead latency
    and cost per 1k calls derived from the fixtures' token usage. No
    network, fully reproducible, safe as a CI gate.

``api`` (live)
    Sends every prompt to the real Cloudflare Clef decision model via the
    configured credentials. Measures true end-to-end accuracy, latency
    percentiles, and token cost. Requires ``CLEF_ACCOUNT_ID`` /
    ``CLEF_API_TOKEN`` (or their ``CLOUDFLARE_*`` fallbacks).

What this is NOT: a quality benchmark of the Clef model itself. Cloudflare
publishes the Decision Index for that; this harness measures OUR policy and
client. Unmeasured values are reported as ``null``, never invented.

Usage::

    python evals/run_eval.py                          # mock, committed dataset
    python evals/run_eval.py --mode api               # live, needs credentials
    python evals/run_eval.py --out results/current.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from clef_router import __version__
from clef_router.client import AsyncClefRouter, ClefRouter
from clef_router.config import RouterConfig

DEFAULT_DATASET = Path(__file__).parent / "data" / "routing_bench.jsonl"
DEFAULT_OUT = Path(__file__).parent / "results" / "routing-eval-mock.json"

#: Cloudflare's published Workers AI price for Clef input tokens, USD per million.
PRICE_PER_MTOKEN_INPUT_USD = 0.24

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_CREDENTIALS = 3


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate the JSONL dataset; abort on malformed rows."""
    rows: list[dict[str, Any]] = []
    required = {"id", "category", "prompt", "expected_tier"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise SystemExit(
                    f"{path}:{line_number}: row missing fields {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: dataset is empty")
    return rows


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of *values* (values must be non-empty)."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def route_mock(
    row: dict[str, Any], config: RouterConfig
) -> tuple[str, float, dict[str, int]]:
    """Run one decision against the row's fixture; return (tier, ms, usage)."""
    envelope = {
        "success": True,
        "result": {
            "model": config.model_selector,
            "answers": row["simulated_answer"],
            "usage": row["usage"],
        },
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=envelope))
    with ClefRouter(config=config, transport=transport) as router:
        started = time.perf_counter()
        routing = router.route(row["prompt"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    return routing.tier, elapsed_ms, dict(routing.decision.usage.to_dict())


async def route_api(
    router: AsyncClefRouter, prompt: str
) -> tuple[str, float, dict[str, int]]:
    """Run one live decision; return (tier, ms, usage)."""
    started = time.perf_counter()
    routing = await router.route(prompt)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return routing.tier, elapsed_ms, dict(routing.decision.usage.to_dict())


def evaluate(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate per-row results into the metrics block."""
    correct = sum(1 for r in results if r["correct"])
    by_category: dict[str, dict[str, float]] = {}
    for row, result in zip(rows, results, strict=True):
        bucket = by_category.setdefault(
            row["category"], {"n": 0, "correct": 0}
        )
        bucket["n"] += 1
        bucket["correct"] += int(result["correct"])
    for bucket in by_category.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["n"], 4)

    latencies = [r["latency_ms"] for r in results]
    input_tokens = [r["usage"]["input_tokens"] for r in results]
    mean_input_tokens = sum(input_tokens) / len(input_tokens)
    expected_frontier = [
        r for r, row in zip(results, rows, strict=True)
        if row["expected_tier"] == "frontier"
    ]
    over_escalations = sum(
        1
        for r, row in zip(results, rows, strict=True)
        if r["tier"] == "frontier" and row["expected_tier"] == "cheap"
    )
    under_routes = sum(
        1
        for r, row in zip(results, rows, strict=True)
        if r["tier"] == "cheap" and row["expected_tier"] == "frontier"
    )

    return {
        "n": len(results),
        "accuracy": round(correct / len(results), 4),
        "by_category": by_category,
        "escalations": {
            "over_escalations": over_escalations,
            "under_routes": under_routes,
            "escalation_precision": (
                round(
                    (len(expected_frontier) - under_routes) / max(1, sum(
                        1 for r in results if r["tier"] == "frontier"
                    )),
                    4,
                )
            ),
        },
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "p99": round(percentile(latencies, 99), 3),
            "mean": round(sum(latencies) / len(latencies), 3),
        },
        "cost": {
            "price_per_mtok_input_usd": PRICE_PER_MTOKEN_INPUT_USD,
            "input_tokens_mean": round(mean_input_tokens, 1),
            "cost_per_1k_calls_usd": round(
                mean_input_tokens * 1000 / 1_000_000 * PRICE_PER_MTOKEN_INPUT_USD, 4
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Measure clef-router routing accuracy, latency, and cost.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL dataset path"
    )
    parser.add_argument(
        "--mode", choices=("mock", "api"), default="mock",
        help="mock replays committed fixtures offline; api calls the real Clef API",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output JSON path"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="override the confidence gate for this run",
    )
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset)
    overrides: dict[str, Any] = {}
    if args.min_confidence is not None:
        overrides["min_confidence"] = args.min_confidence

    notes = [
        "mock mode measures the offline routing POLICY over committed fixtures; "
        "latency reflects library overhead only (no network)",
    ]
    if args.mode == "api":
        if not (RouterConfig.from_env().validate() == []):
            print(
                "api mode needs CLEF_ACCOUNT_ID / CLEF_API_TOKEN "
                "(or CLOUDFLARE_* fallbacks); not set.",
                file=sys.stderr,
            )
            return EXIT_NO_CREDENTIALS
        notes = ["api mode measures real end-to-end decisions against Cloudflare Clef."]

    results: list[dict[str, Any]] = []
    if args.mode == "mock":
        config = RouterConfig(
            account_id="eval",
            api_token="eval",
            retry_backoff=0.0,
            **overrides,
        )
        for row in rows:
            tier, elapsed_ms, usage = route_mock(row, config)
            results.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "expected": row["expected_tier"],
                    "tier": tier,
                    "correct": tier == row["expected_tier"],
                    "latency_ms": round(elapsed_ms, 3),
                    "usage": usage,
                }
            )
    else:
        config = RouterConfig.from_env(**overrides)
        config.ensure_valid()

        async def run_all() -> None:
            async with AsyncClefRouter(config=config) as router:
                for row in rows:
                    tier, elapsed_ms, usage = await route_api(router, row["prompt"])
                    results.append(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "expected": row["expected_tier"],
                            "tier": tier,
                            "correct": tier == row["expected_tier"],
                            "latency_ms": round(elapsed_ms, 3),
                            "usage": usage,
                        }
                    )

        import asyncio

        asyncio.run(run_all())

    report: dict[str, Any] = {
        "meta": {
            "mode": args.mode,
            "dataset": str(args.dataset),
            "package_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "min_confidence": config.min_confidence,
            "notes": notes,
        },
        "metrics": evaluate(rows, results),
        "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics = report["metrics"]
    print(
        f"[{args.mode}] n={metrics['n']} accuracy={metrics['accuracy']:.2%} "
        f"latency p50/p95/p99="
        f"{metrics['latency_ms']['p50']}/{metrics['latency_ms']['p95']}/"
        f"{metrics['latency_ms']['p99']} ms "
        f"cost/1k=${metrics['cost']['cost_per_1k_calls_usd']}"
    )
    print(f"wrote {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
