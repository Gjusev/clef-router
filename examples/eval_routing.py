"""Route a small prompt set through the real Clef API and print a table.

Requires CLEF_ACCOUNT_ID and CLEF_API_TOKEN in the environment. Run with
``make eval``. Cost note: Clef is billed per input token ($0.24 per million
input tokens); the output-token price is not published, so output tokens
are reported but never priced.
"""

from __future__ import annotations

import logging
import sys

from clef_router import ClefRouter, RouterConfig

PROMPTS = [
    "What is the capital of France?",
    "Summarize this paragraph in two sentences: <long text omitted>",
    "Design a distributed rate limiter with sliding windows and per-tenant quotas",
    "Our prod cluster is returning 500s right now, customer impact is growing",
    "Rewrite this sentence to be friendlier: send me the report",
]


def main() -> int:
    """Route each prompt and print tier, reason, and token usage."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = RouterConfig.from_env()
        config.ensure_valid()
    except Exception as exc:  # noqa: BLE001 - CLI-style guard for a human runner
        print(f"eval needs valid credentials: {exc}", file=sys.stderr)
        return 1

    header = f"{'prompt':<58} {'tier':<9} {'in tok':>7} {'out tok':>8}"
    print(header)
    print("-" * len(header))
    input_tokens = 0
    with ClefRouter(config=config) as router:
        for prompt in PROMPTS:
            routing = router.route(prompt)
            usage = routing.decision.usage
            input_tokens += usage.input_tokens
            print(
                f"{prompt:<58.58} {routing.tier:<9} "
                f"{usage.input_tokens:>7} {usage.output_tokens:>8}"
            )
    print("-" * len(header))
    print(
        f"{len(PROMPTS)} decisions, {input_tokens} input tokens "
        f"(~${input_tokens / 1_000_000 * 0.24:.6f} at $0.24/M input; "
        f"output-token price is not published)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
