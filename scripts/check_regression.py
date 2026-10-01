"""Compare a fresh measurement JSON against a committed baseline.

Used by the regression-gate composite action and usable standalone:

    python scripts/check_regression.py \
        --current evals/results/current.json \
        --baseline evals/results/baselines/routing-eval-mock.json \
        --metric metrics.accuracy:max \
        --metric metrics.escalations.escalation_precision:max:0.05

Metric specs: ``path:goal[:tolerance]`` where *path* is a dotted path into
the JSON, *goal* is ``max`` (accuracy-like: fail when the current value
drops) or ``min`` (ECE-like: fail when it rises), and *tolerance* defaults
to 0.02.

Exit codes: 0 pass, 1 regression, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCE = 0.02

#: Wall-clock style fields move run to run for boring reasons; gating on
#: them produces noise, so the tool refuses them outright.
DRIFTING_FIELDS = frozenset({"wall_seconds", "decisions_per_second", "latency_ms"})

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2


def resolve_path(data: dict[str, Any], dotted: str) -> Any:
    """Walk a dotted path, trying the longest matching keys first.

    JSON keys may contain literal dots (``"xnli.en"``); greedy-longest
    matching lets ``claims.xnli.en.measured`` resolve correctly.
    """
    segments = dotted.split(".")
    value: Any = data
    index = 0
    while index < len(segments):
        if not isinstance(value, dict):
            raise KeyError(dotted)
        for width in range(len(segments) - index, 0, -1):
            candidate = ".".join(segments[index : index + width])
            if candidate in value:
                value = value[candidate]
                index += width
                break
        else:
            raise KeyError(dotted)
    return value


def check_regression(
    current: dict[str, Any],
    baseline: dict[str, Any],
    metrics: list[tuple[str, str, float]],
) -> dict[str, Any]:
    """Compare *metrics* between the two documents.

    Returns ``{"regressed": bool, "results": [MetricResult...]}`` where each
    result records path, goal, baseline, current, delta, and ok.
    """
    results: list[dict[str, Any]] = []
    regressed = False
    for path, goal, tolerance in metrics:
        if any(field in path for field in DRIFTING_FIELDS):
            raise ValueError(
                f"{path}: gating on wall-clock style metrics is not allowed "
                f"(drifting fields: {sorted(DRIFTING_FIELDS)})"
            )
        try:
            base_value = float(resolve_path(baseline, path))
            current_value = float(resolve_path(current, path))
        except (KeyError, TypeError, ValueError):
            results.append(
                {
                    "path": path,
                    "goal": goal,
                    "baseline": None,
                    "current": None,
                    "delta": None,
                    "ok": None,
                }
            )
            continue
        if goal == "max":
            ok = current_value >= base_value - tolerance
        else:
            ok = current_value <= base_value + tolerance
        results.append(
            {
                "path": path,
                "goal": goal,
                "baseline": base_value,
                "current": current_value,
                "delta": round(current_value - base_value, 6),
                "ok": ok,
            }
        )
        if not ok:
            regressed = True
    return {"regressed": regressed, "results": results}


def parse_metric(spec: str) -> tuple[str, str, float]:
    """Parse ``path:goal[:tolerance]`` and validate the pieces."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"metric spec must be path:goal[:tolerance], got {spec!r}")
    path, goal = parts[0], parts[1]
    if goal not in ("max", "min"):
        raise ValueError(f"goal must be 'max' or 'min', got {goal!r}")
    tolerance = float(parts[2]) if len(parts) == 3 else DEFAULT_TOLERANCE
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    return path, goal, tolerance


def summary(report: dict[str, Any]) -> str:
    """Human-readable lines for a check_regression report."""
    lines: list[str] = []
    for result in report["results"]:
        if result["ok"] is None:
            lines.append(f"MISSING {result['path']}: not found in both documents")
            continue
        verdict = "OK" if result["ok"] else "REGRESS"
        lines.append(
            f"{verdict} {result['path']}: {result['baseline']} -> "
            f"{result['current']} (delta {result['delta']:+}, goal {result['goal']})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", required=True, help="fresh measurement JSON")
    parser.add_argument("--baseline", required=True, help="committed baseline JSON")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        dest="metrics",
        help="path:goal[:tolerance], repeatable",
    )
    args = parser.parse_args(argv)
    if not args.metrics:
        print("no --metric specs given; nothing to check", file=sys.stderr)
        return EXIT_USAGE

    try:
        current = json.loads(Path(args.current).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        metrics = [parse_metric(spec) for spec in args.metrics]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        report = check_regression(current, baseline, metrics)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(summary(report))
    return EXIT_REGRESSION if report["regressed"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
