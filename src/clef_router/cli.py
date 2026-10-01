"""CLI for clef-router: route one prompt from the command line.

The prompt is classified by the Clef decision model and the resulting tier
is printed to stdout. Pass ``--json`` for the full decision as JSON.
Diagnostics go through the ``logging`` module on stderr; credentials are
never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ._version import __version__
from .client import ClefRouter
from .config import RouterConfig
from .errors import ClefError, ConfigurationError

logger = logging.getLogger("clef_router")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``clef-route`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="clef-route",
        description="Classify a prompt with Cloudflare's Clef decision model",
    )
    parser.add_argument("prompt", help="The prompt to classify")
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt passed to Clef as routing instructions",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=("clef", "clef-flash"),
        help="Clef selector override (default: env CLEF_MODEL or clef-flash)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full decision as JSON"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one routing decision and print the outcome.

    Returns a process exit code: 0 on success, 1 on configuration or API
    errors (the message goes to stderr via logging, never stdout).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        config = RouterConfig.from_env(model_selector=args.model)
        config.ensure_valid()
        with ClefRouter(config=config) as router:
            routing = router.route(args.prompt, system_prompt=args.system)
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 1
    except ClefError as exc:
        logger.error("routing failed: %s", exc)
        return 1
    if args.json:
        print(json.dumps(routing.to_dict(), indent=2))
    else:
        print(f"tier:       {routing.tier}")
        print(f"reason:     {routing.reason}")
        print(f"model:      {routing.decision.model}")
        print(f"input tokens:  {routing.decision.usage.input_tokens}")
        print(f"output tokens: {routing.decision.usage.output_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
