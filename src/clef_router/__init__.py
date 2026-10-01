"""clef-router: Route prompts between cheap and frontier LLMs using Cloudflare's Clef."""

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class RouteResult:
    tier: str
    complexity: str
    confidence: float
    reason: str
    raw_response: dict = field(default_factory=dict, repr=False)


@dataclass
class ClefRouter:
    account_id: str = field(default_factory=lambda: os.environ.get("CLEF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.environ.get("CLEF_API_TOKEN", ""))
    model: str = "@cf/cloudflare/clef-flash"
    base_url: str = "https://api.cloudflare.com/client/v4"
    min_confidence: float = 0.45
    timeout: float = 30.0

    def __post_init__(self):
        if not self.account_id:
            raise ValueError("Set CLEF_ACCOUNT_ID env var or pass account_id")
        if not self.api_token:
            raise ValueError("Set CLEF_API_TOKEN env var or pass api_token")

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}

    def route(self, prompt: str, context: Optional[str] = None) -> RouteResult:
        state = f"User prompt to classify:\n{prompt}"
        if context:
            state = f"Context: {context}\n\n{state}"

        payload = {
            "state": state,
            "questions": {
                "complexity": {
                    "type": "choice",
                    "context": "How computationally complex is this prompt? "
                               "simple = greeting, translation, basic lookup. "
                               "standard = summarization, rewording, simple analysis. "
                               "complex = reasoning, coding, multi-step, creative.",
                    "choices": ["simple", "standard", "complex"],
                },
                "confidence": {
                    "type": "score",
                    "context": "How confident are you in this classification?",
                    "levels": ["very_low", "low", "medium", "high", "very_high"],
                },
            },
        }

        resp = httpx.post(
            self._endpoint, json=payload, headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        result = data.get("result", {})
        complexity = result.get("complexity", "complex")
        conf_map = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "very_high": 0.9}
        confidence = conf_map.get(result.get("confidence", "low"), 0.3)

        if confidence < self.min_confidence:
            tier = "frontier"
            reason = f"confidence {confidence:.2f} below threshold {self.min_confidence}, escalated"
        elif complexity == "complex":
            tier = "frontier"
            reason = f"classified as {complexity}"
        else:
            tier = "cheap"
            reason = f"classified as {complexity}, confidence {confidence:.2f}"

        return RouteResult(
            tier=tier, complexity=complexity, confidence=confidence,
            reason=reason, raw_response=data,
        )


__version__ = "0.1.0"
__all__ = ["ClefRouter", "RouteResult"]
