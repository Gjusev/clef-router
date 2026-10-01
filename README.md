# clef-router

Route prompts between cheap and frontier LLMs using Cloudflare's [Clef](https://blog.cloudflare.com/clef-decision-models/) decision model.

## Quick start

```bash
pip install clef-router
export CLEF_ACCOUNT_ID=your_cloudflare_account_id
export CLEF_API_TOKEN=your_cloudflare_api_token
```

```python
from clef_router import ClefRouter

router = ClefRouter()
result = router.route("What is the capital of France?")
print(result.tier)       # "cheap"
print(result.reason)     # "classified as simple, confidence 0.90"

result = router.route("Design a distributed consensus algorithm")
print(result.tier)       # "frontier"
```

## How it works

```
prompt → Clef decision model (Cloudflare Workers AI) → tier
              ├─ simple   → cheap model
              ├─ standard  → cheap model
              ├─ complex   → frontier model
              └─ uncertain → frontier (confidence gate)
```

## clef vs laya

| | laya | Clef |
|---|---|---|
| Latency | 5.8ms | 38.8ms (flash) / 209.3ms (full) |
| Quality (BFCL) | 38.13 | 98.76 |
| Context | 32k | 64k |
| Hosting | local CPU | Cloudflare API |
| Cost | $0 | Workers AI pricing |

Use [laya-router](https://github.com/Gjusev/laya-router) for sub-10ms local routing. Use clef-router when quality matters more than latency.

## License

Apache 2.0
