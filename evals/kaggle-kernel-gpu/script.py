"""Measure clef-flash's REAL routing accuracy on Kaggle GPU.

Runs the actual Cloudflare/clef-flash weights (9.4B) locally on the labeled
routing dataset: no Cloudflare credentials, no API. The routing question set
and the tier policy come from the installed clef-router package, so this
measures the same decisions the library would make, just served by the
weights instead of the hosted endpoint.

Outputs /kaggle/working/routing-gpu-eval.json with accuracy, latency
percentiles (first calls excluded as warmup), escalation counts, and the
per-row decisions. Expected wall time: 10-20 minutes (about 19 GB model
download included).

This kernel needs a real GPU. It fails loudly when nvidia-smi is missing:
Kaggle silently drops the GPU for unverified accounts, and a CPU run of a
9B model would take hours, not minutes.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def run(command: str) -> None:
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def fail(message: str) -> None:
    print(f"FAIL: {message}", flush=True)
    sys.exit(1)


# 1. Loud GPU check (Kaggle drops GPUs silently on unverified accounts).
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
        capture_output=True, text=True, timeout=30,
    )
    print(smi.stdout, flush=True)
    if smi.returncode != 0:
        fail("nvidia-smi failed: this session has NO GPU. Verify the Kaggle "
             "account (phone) and re-push, or run the CPU kernel instead.")
except FileNotFoundError:
    fail("nvidia-smi not found: no GPU in this session.")

# 2. Dependencies: pinned transformers per the model card; keep the image's
# torch (CUDA build) and add bitsandbytes for 4-bit loading on T4.
run(f"{sys.executable} -m pip install -q --no-input "
    '"transformers==5.10.2" accelerate bitsandbytes "huggingface_hub>=0.34"')

import torch  # noqa: E402

print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    fail("torch sees no CUDA device; the GPU is not attached to this session.")

# 3. Clone the repository for the dataset, the question set, and the policy.
run("git clone --depth 1 https://github.com/Gjusev/clef-router.git")
run(f"{sys.executable} -m pip install -q --no-input ./clef-router")

sys.path.insert(0, "clef-router")
sys.path.insert(0, str(Path("clef-router/src").resolve()))

from clef_router.models import DEFAULT_QUESTIONS, derive_tier, parse_decision  # noqa: E402

# 4. Download and load clef-flash in 4-bit on the first GPU. The release
# loader pins device_map to a single device, so 4-bit is how 9.4B fits a
# 16 GB T4 with room for activations. T4 has no native bf16; use fp16.
from huggingface_hub import snapshot_download  # noqa: E402

print("downloading Cloudflare/clef-flash (about 19 GB)...", flush=True)
model_path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, model_path)  # joint_schema_model.py ships with the weights

from joint_schema_model import load_release_model, systemone  # noqa: E402

import torch  # noqa: E402
from transformers import BitsAndBytesConfig  # noqa: E402

t0 = time.perf_counter()
model, processor = load_release_model(
    model_path,
    device="cuda:0",
    dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
)
print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

# 5. Route every labeled prompt through the real model + real policy.
rows = [
    json.loads(line)
    for line in Path("clef-router/evals/data/routing_bench.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]

WARMUP_CALLS = 2
MIN_CONFIDENCE = 0.45
PRICE_PER_MTOK = 0.24

results = []
for index, row in enumerate(rows):
    request = {
        "model": "clef-flash",
        "state": f"User prompt to classify:\n{row['prompt']}",
        "questions": DEFAULT_QUESTIONS,
    }
    started = time.perf_counter()
    response = systemone(model, processor, request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    decision = parse_decision(
        {
            "success": True,
            "result": {
                "model": "clef-flash-local",
                "answers": response["answers"],
                "usage": response["usage"],
            },
        },
        latency_ms=elapsed_ms,
    )
    routing = derive_tier(decision, min_confidence=MIN_CONFIDENCE)
    results.append(
        {
            "id": row["id"],
            "category": row["category"],
            "expected": row["expected_tier"],
            "tier": routing.tier,
            "correct": routing.tier == row["expected_tier"],
            "reason": routing.reason,
            "confidence": decision.answer("team").confidence
            if decision.answer("team") is not None
            else None,
            "latency_ms": round(elapsed_ms, 1),
            "usage": response["usage"],
        }
    )
    print(
        f"[{index + 1}/{len(rows)}] {row['id']}: {routing.tier} "
        f"({routing.reason}; {elapsed_ms:.0f} ms)",
        flush=True,
    )

# 6. Aggregate: accuracy, warm latency percentiles, escalations, cost.
warm = [r["latency_ms"] for r in results[WARMUP_CALLS:]] or [0.0]


def percentile(values, pct):
    ordered = sorted(values)
    rank = max(1, -(-pct * len(ordered) // 100))
    return ordered[min(rank, len(ordered)) - 1]


correct = sum(1 for r in results if r["correct"])
input_tokens = [r["usage"]["input_tokens"] for r in results]
mean_tokens = sum(input_tokens) / len(input_tokens)
frontier_expected = sum(1 for r in results if r["expected"] == "frontier")
under_routes = sum(1 for r in results if r["tier"] == "cheap" and r["expected"] == "frontier")
over_escalations = sum(1 for r in results if r["tier"] == "frontier" and r["expected"] == "cheap")

report = {
    "meta": {
        "mode": "gpu-local",
        "model": "Cloudflare/clef-flash (local weights, 4-bit nf4, fp16 compute)",
        "hardware": subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "package_version": __import__("clef_router").__version__,
        "min_confidence": MIN_CONFIDENCE,
        "n": len(results),
        "warmup_calls_excluded": WARMUP_CALLS,
        "notes": [
            "measures the real model over labeled prompts; policy and question "
            "set are the library's own",
            "latency includes tokenization and the joint-schema forward pass "
            "on one quantized T4; the hosted endpoint (38.8 ms median per "
            "Cloudflare) runs unquantized on datacenter GPUs",
        ],
    },
    "metrics": {
        "n": len(results),
        "accuracy": round(correct / len(results), 4),
        "escalations": {
            "under_routes": under_routes,
            "over_escalations": over_escalations,
            "frontier_expected": frontier_expected,
        },
        "latency_ms": {
            "cold_first_call": results[0]["latency_ms"],
            "p50": round(percentile(warm, 50), 1),
            "p95": round(percentile(warm, 95), 1),
            "p99": round(percentile(warm, 99), 1),
        },
        "cost": {
            "input_tokens_mean": round(mean_tokens, 1),
            "hosted_equivalent_cost_per_1k_calls_usd": round(
                mean_tokens * 1000 / 1_000_000 * PRICE_PER_MTOK, 4
            ),
            "local_gpu_cost_per_1k_calls_usd": 0.0,
        },
    },
    "rows": results,
}
out = Path("/kaggle/working/routing-gpu-eval.json")
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

m = report["metrics"]
print("=== Verdict ===", flush=True)
print(f"rows: {m['n']}  accuracy: {m['accuracy']:.2%}", flush=True)
print(
    f"latency ms  cold: {m['latency_ms']['cold_first_call']}  "
    f"p50: {m['latency_ms']['p50']}  p95: {m['latency_ms']['p95']}  "
    f"p99: {m['latency_ms']['p99']}",
    flush=True,
)
print(f"under-routes: {under_routes}  over-escalations: {over_escalations}", flush=True)
wrong = [r for r in results if not r["correct"]]
for r in wrong:
    print(f"MISS {r['id']}: expected {r['expected']}, got {r['tier']} ({r['reason']})", flush=True)
print(f"wrote {out}", flush=True)
