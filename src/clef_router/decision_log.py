"""Append-only JSONL decision log for the clef-router proxy.

One line per routing decision: enough to audit tiers after the fact and to
replay confidence-gate calibration offline. The file is opened per write so
log rotation and multiple worker processes stay safe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import RoutingDecision

__all__ = ["append_decision_log"]

#: Prompts are truncated to keep every line scannable in a terminal.
PROMPT_PREVIEW_CHARS = 120


def append_decision_log(
    path: str | Path,
    *,
    routing: RoutingDecision,
    model_selector: str,
    latency_ms: float,
    prompt: str,
    status: str = "ok",
) -> None:
    """Append one JSON line describing a routing decision to *path*.

    Credentials never enter the log; the prompt is truncated and flattened
    so a single decision is always a single line.
    """
    preview = " ".join(str(prompt).split())[:PROMPT_PREVIEW_CHARS]
    line = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "tier": routing.tier,
        "model": model_selector,
        "confidence": routing.decision.answer("team").confidence
        if routing.decision.answer("team") is not None
        else None,
        "reason": routing.reason,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": routing.decision.usage.input_tokens,
        "status": status,
        "prompt_preview": preview,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
