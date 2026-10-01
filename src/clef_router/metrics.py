"""In-memory Prometheus metrics for the clef-router proxy.

Counters and a fixed-bucket histogram, rendered in the Prometheus text
exposition format with zero dependencies. Metrics are per-process (fine
for a single uvicorn worker; a multi-worker deployment should scrape each
worker or add a shared store).
"""

from __future__ import annotations

import time
from collections import Counter

__all__ = ["RouterMetrics"]

#: Routing latency histogram buckets, in seconds.
HISTOGRAM_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


class RouterMetrics:
    """Counters for decisions plus a latency histogram, in seconds."""

    def __init__(self) -> None:
        self.requests_total: Counter[tuple[str, str]] = Counter()
        self._buckets: Counter[float] = Counter()
        self._count = 0
        self._sum = 0.0
        self.started_at = time.time()

    def observe(self, tier: str, latency_ms: float, status: str = "ok") -> None:
        """Record one routing decision (or error) with its latency."""
        self.requests_total[(tier, status)] += 1
        seconds = max(0.0, latency_ms) / 1000.0
        self._count += 1
        self._sum += seconds
        for bucket in HISTOGRAM_BUCKETS:
            if seconds <= bucket:
                self._buckets[bucket] += 1

    def render(self) -> str:
        """Prometheus text exposition of every registered metric."""
        lines: list[str] = [
            "# HELP clef_router_requests_total Routing decisions by tier and status.",
            "# TYPE clef_router_requests_total counter",
        ]
        for (tier, status), value in sorted(self.requests_total.items()):
            lines.append(
                f'clef_router_requests_total{{tier="{tier}",status="{status}"}} {value}'
            )
        if not self.requests_total:
            lines.append('clef_router_requests_total{tier="none",status="none"} 0')
        lines += [
            "# HELP clef_router_routing_seconds "
            "Time spent producing one routing decision.",
            "# TYPE clef_router_routing_seconds histogram",
        ]
        cumulative = 0
        for bucket in HISTOGRAM_BUCKETS:
            cumulative = sum(
                count for le, count in self._buckets.items() if le <= bucket
            )
            lines.append(
                f'clef_router_routing_seconds_bucket{{le="{_format_le(bucket)}"}} '
                f"{cumulative}",
            )
        lines.append(
            'clef_router_routing_seconds_bucket{le="+Inf"} ' + str(self._count)
        )
        lines += [
            f"clef_router_routing_seconds_count {self._count}",
            f"clef_router_routing_seconds_sum {self._sum:.6f}",
        ]
        return "\n".join(lines) + "\n"


def _format_le(bucket: float) -> str:
    """Format a bucket bound the way Prometheus expects (0.1, 1, 2.5)."""
    if bucket == int(bucket):
        return str(int(bucket))
    return f"{bucket:g}"
