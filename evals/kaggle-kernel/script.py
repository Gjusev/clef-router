"""Rerun the clef-router offline evaluation on Kaggle hardware.

Clones this repository, installs it, runs the committed evaluation dataset
through the real policy pipeline (no Cloudflare credentials needed: the
dataset replays canned Clef answers), and writes /kaggle/working/
routing-eval.json. Expected wall time: about two minutes on Kaggle CPU.
"""

import json
import subprocess
import sys


def run(command: str) -> None:
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


run("git clone --depth 1 https://github.com/Gjusev/clef-router.git")
run(f"{sys.executable} -m pip install -q --no-input ./clef-router[dev]")

run(
    f"{sys.executable} clef-router/evals/run_eval.py "
    "--out /kaggle/working/routing-eval.json"
)

with open("/kaggle/working/routing-eval.json", encoding="utf-8") as handle:
    report = json.load(handle)
metrics = report["metrics"]
print("=== Verdict ===", flush=True)
print(f"rows: {metrics['n']}", flush=True)
print(f"policy accuracy: {metrics['accuracy']:.2%}", flush=True)
print(
    "latency p50/p95/p99 ms: "
    f"{metrics['latency_ms']['p50']} / {metrics['latency_ms']['p95']} / "
    f"{metrics['latency_ms']['p99']}",
    flush=True,
)
print(f"cost per 1k decisions: ${metrics['cost']['cost_per_1k_calls_usd']}", flush=True)
print(
    "over-escalations:", metrics["escalations"]["over_escalations"],
    "under-routes:", metrics["escalations"]["under_routes"],
    flush=True,
)
