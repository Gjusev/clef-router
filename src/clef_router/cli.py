"""CLI for clef-router."""

import argparse
import json

from . import ClefRouter


def main():
    p = argparse.ArgumentParser(prog="clef-route", description="Route a prompt using Clef")
    p.add_argument("prompt", help="The prompt to classify")
    p.add_argument("--context", "-c", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    router = ClefRouter()
    result = router.route(args.prompt, context=args.context)

    if args.json:
        print(json.dumps({
            "tier": result.tier,
            "complexity": result.complexity,
            "confidence": result.confidence,
            "reason": result.reason,
        }, indent=2))
    else:
        print(f"tier:       {result.tier}")
        print(f"complexity: {result.complexity}")
        print(f"confidence: {result.confidence:.2f}")
        print(f"reason:     {result.reason}")


if __name__ == "__main__":
    main()
