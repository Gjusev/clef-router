"""Data types for Clef questions, answers, and routing decisions.

Everything here is pure data: no HTTP, no clocks, no environment. The
shapes mirror the Clef request/response contract on Workers AI exactly:

* ``noul``  -> ``{"type": "noul", "noul": 0..1}``  (P(yes))
* ``choice``-> ``{"type": "choice", "choice": str, "probabilities": {...},
  "confidence": 0..1}``
* ``score`` -> ``{"type": "score", "score": float, "legend": {...},
  "probabilities": {...}, "confidence": 0..1}``

The router's default question set is documented on
:data:`DEFAULT_QUESTIONS` and the deterministic tier policy on
:func:`derive_tier`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ClefResponseError

__all__ = [
    "QUESTION_TYPES",
    "TYPE_NOUL",
    "TYPE_CHOICE",
    "TYPE_SCORE",
    "TIER_CHEAP",
    "TIER_FRONTIER",
    "MAX_QUESTIONS",
    "MAX_IMAGES",
    "DEFAULT_QUESTIONS",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "Usage",
    "ClefDecision",
    "RoutingDecision",
    "validate_questions",
    "parse_decision",
    "derive_tier",
]

TYPE_NOUL = "noul"
TYPE_CHOICE = "choice"
TYPE_SCORE = "score"
QUESTION_TYPES = (TYPE_NOUL, TYPE_CHOICE, TYPE_SCORE)

TIER_CHEAP = "cheap"
TIER_FRONTIER = "frontier"

#: Contract limits: 1..64 questions per request, at most 4 images.
MAX_QUESTIONS = 64
MAX_IMAGES = 4

#: The default routing question set used by ``route()`` and by the server's
#: ``/v1/chat/completions`` when the caller does not pass ``clef_questions``.
#:
#: * ``urgency`` (noul): P(the request is time-critical). Informational on
#:   its own, and the fallback tier signal when no team answer is present.
#: * ``team`` (choice): which tier should handle the request. The option
#:   labels are the tiers themselves so the mapping stays explicit.
DEFAULT_QUESTIONS: dict[str, dict[str, Any]] = {
    "urgency": {
        "type": TYPE_NOUL,
        "instructions": (
            "Decide whether the user's prompt is time-critical: an outage, "
            "an incident, a security problem, or anything that must be "
            "handled immediately rather than routinely."
        ),
        "criteria": {
            "true": "Time-critical: outages, incidents, security issues, "
            "deadlines in hours.",
            "false": "Routine work: questions, drafts, research, refactors, "
            "low-stakes lookups.",
        },
    },
    "team": {
        "type": TYPE_CHOICE,
        "instructions": (
            "Choose the model team that should handle this prompt. Prefer "
            "the cheaper team unless the prompt needs deep reasoning, "
            "code understanding, or high-stakes accuracy."
        ),
        "criteria": {
            TIER_CHEAP: "Fast inexpensive model: greetings, simple lookups, "
            "short rewrites, straightforward Q&A.",
            TIER_FRONTIER: "Frontier model: reasoning, coding, multi-step "
            "analysis, nuanced or high-stakes work.",
        },
    },
}


@dataclass(frozen=True)
class NoulAnswer:
    """Answer to a ``noul`` question: the probability of "yes" in [0, 1]."""

    qid: str
    noul: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation of this answer."""
        return {"type": TYPE_NOUL, "noul": self.noul}


@dataclass(frozen=True)
class ChoiceAnswer:
    """Answer to a ``choice`` question with per-option probabilities."""

    qid: str
    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation of this answer."""
        return {
            "type": TYPE_CHOICE,
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class ScoreAnswer:
    """Answer to a ``score`` question on a labeled numeric scale."""

    qid: str
    score: float
    confidence: float
    legend: dict[str, str] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation of this answer."""
        return {
            "type": TYPE_SCORE,
            "score": self.score,
            "confidence": self.confidence,
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
        }


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class Usage:
    """Token usage reported by the Clef API.

    Cloudflare publishes the input-token price ($0.24 per million input
    tokens); the output-token price is not published, so this type never
    computes a total cost — report ``input_tokens`` and ``output_tokens``
    separately.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Plain-JSON representation of this usage."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class ClefDecision:
    """The raw outcome of one Clef request.

    Attributes:
        model: Model selector the API reports in ``result.model``.
        answers: Parsed answers keyed by question id.
        usage: Token usage for the request.
        latency_ms: Wall-clock duration of the request in milliseconds.
        raw_response: The full API envelope, for auditing.
    """

    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict, repr=False)

    def answer(self, qid: str) -> Answer | None:
        """Return the answer for *qid*, or ``None`` if absent."""
        return self.answers.get(qid)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation of this decision."""
        return {
            "model": self.model,
            "answers": {qid: a.to_dict() for qid, a in self.answers.items()},
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class RoutingDecision:
    """A routing decision derived from a :class:`ClefDecision`.

    Attributes:
        tier: Chosen tier, ``"cheap"`` or ``"frontier"``.
        reason: Human-readable trace of why the tier was chosen.
        decision: The underlying Clef decision this was derived from.
    """

    tier: str
    reason: str
    decision: ClefDecision

    @property
    def escalated(self) -> bool:
        """Whether the decision escalated to the frontier tier."""
        return self.tier == TIER_FRONTIER

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation, the shape used as chat content."""
        return {
            "tier": self.tier,
            "reason": self.reason,
            "decision": self.decision.to_dict(),
        }


def validate_questions(questions: Mapping[str, Any]) -> list[str]:
    """Validate a question mapping against the Clef contract.

    Returns every problem found as human-readable lines (empty when valid):
    the mapping must be non-empty and within 1..64 questions, and each
    entry needs a known ``type`` plus type-appropriate ``criteria`` —
    ``choice`` needs a mapping of 2..255 option descriptions, ``score`` a
    list of 2..10 level labels ordered lowest-first, ``noul`` an optional
    ``{"true": ..., "false": ...}`` mapping.
    """
    problems: list[str] = []
    if not isinstance(questions, Mapping) or not questions:
        return ["questions must be a non-empty mapping of qid -> question"]
    if len(questions) > MAX_QUESTIONS:
        problems.append(
            f"questions must contain at most {MAX_QUESTIONS} entries, "
            f"got {len(questions)}"
        )
    for qid, question in questions.items():
        prefix = f"questions[{qid!r}]"
        if not isinstance(question, Mapping):
            problems.append(
                f"{prefix} must be a mapping, got {type(question).__name__}"
            )
            continue
        qtype = question.get("type")
        if qtype not in QUESTION_TYPES:
            problems.append(
                f"{prefix}.type must be one of {list(QUESTION_TYPES)}, got {qtype!r}"
            )
            continue
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            problems.append(f"{prefix}.instructions must be a non-empty string")
        criteria = question.get("criteria")
        if qtype == TYPE_CHOICE:
            if not isinstance(criteria, Mapping):
                problems.append(
                    f"{prefix}.criteria must be a mapping of "
                    f"option -> description for choice questions"
                )
            elif not 2 <= len(criteria) <= 255:
                problems.append(
                    f"{prefix}.criteria must have 2..255 options, "
                    f"got {len(criteria)}"
                )
        elif qtype == TYPE_SCORE:
            if not isinstance(criteria, (list, tuple)):
                problems.append(
                    f"{prefix}.criteria must be a list of level labels "
                    f"ordered lowest-first for score questions"
                )
            elif not 2 <= len(criteria) <= 10:
                problems.append(
                    f"{prefix}.criteria must have 2..10 levels, "
                    f"got {len(criteria)}"
                )
        else:  # noul
            if criteria is not None and not (
                isinstance(criteria, Mapping)
                and set(criteria) <= {"true", "false"}
            ):
                problems.append(
                    f"{prefix}.criteria for noul questions may only map "
                    f"'true' and 'false' to descriptions"
                )
    return problems


def parse_decision(
    envelope: Mapping[str, Any],
    *,
    latency_ms: float = 0.0,
) -> ClefDecision:
    """Parse a successful Clef API envelope into a :class:`ClefDecision`.

    The envelope is the Cloudflare ``{"result": ..., "success": true}``
    object; ``result`` must contain ``answers`` keyed by question id.
    Unknown answer types raise :class:`ClefResponseError` because the
    contract defines exactly three; missing or malformed answers raise the
    same error since guessing would silently corrupt routing.

    Raises:
        ClefResponseError: If ``result``/``answers`` is missing or an
            answer has an unknown or inconsistent shape.
    """
    result = envelope.get("result")
    if not isinstance(result, Mapping):
        raise ClefResponseError("API response has no 'result' object")
    answers_body = result.get("answers")
    if not isinstance(answers_body, Mapping):
        raise ClefResponseError("API response 'result' has no 'answers' mapping")

    answers: dict[str, Answer] = {}
    for qid, body in answers_body.items():
        if not isinstance(body, Mapping):
            raise ClefResponseError(f"answer for {qid!r} is not a mapping")
        answer = _parse_answer(qid, body)
        answers[qid] = answer

    usage_body = result.get("usage")
    usage = (
        Usage(
            input_tokens=int(usage_body.get("input_tokens", 0) or 0),
            output_tokens=int(usage_body.get("output_tokens", 0) or 0),
        )
        if isinstance(usage_body, Mapping)
        else Usage()
    )

    return ClefDecision(
        model=str(result.get("model", "")),
        answers=answers,
        usage=usage,
        latency_ms=latency_ms,
        raw_response=dict(envelope),
    )


def _parse_answer(qid: str, body: Mapping[str, Any]) -> Answer:
    """Convert one raw answer body to its typed answer dataclass."""
    qtype = body.get("type")
    if qtype == TYPE_NOUL:
        noul = body.get("noul")
        if not isinstance(noul, (int, float)):
            raise ClefResponseError(f"noul answer for {qid!r} has no numeric 'noul'")
        return NoulAnswer(qid=qid, noul=float(noul))
    if qtype == TYPE_CHOICE:
        choice = body.get("choice")
        if not isinstance(choice, str) or not choice:
            raise ClefResponseError(f"choice answer for {qid!r} has no 'choice' string")
        return ChoiceAnswer(
            qid=qid,
            choice=choice,
            confidence=_as_float(body.get("confidence"), 0.0),
            probabilities=_as_float_map(body.get("probabilities")),
        )
    if qtype == TYPE_SCORE:
        score = body.get("score")
        if not isinstance(score, (int, float)):
            raise ClefResponseError(f"score answer for {qid!r} has no numeric 'score'")
        return ScoreAnswer(
            qid=qid,
            score=float(score),
            confidence=_as_float(body.get("confidence"), 0.0),
            legend=_as_str_map(body.get("legend")),
            probabilities=_as_float_map(body.get("probabilities")),
        )
    raise ClefResponseError(
        f"answer for {qid!r} has unknown type {qtype!r}; "
        f"expected one of {list(QUESTION_TYPES)}"
    )


def _as_float(value: Any, default: float) -> float:
    """Best-effort float coercion with a fallback default."""
    return float(value) if isinstance(value, (int, float)) else default


def _as_float_map(value: Any) -> dict[str, float]:
    """Coerce a raw mapping to ``{str: float}``, ignoring bad entries."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(val)
        for key, val in value.items()
        if isinstance(val, (int, float))
    }


def _as_str_map(value: Any) -> dict[str, str]:
    """Coerce a raw mapping to ``{str: str}``, ignoring bad entries."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def derive_tier(decision: ClefDecision, *, min_confidence: float) -> RoutingDecision:
    """Apply the deterministic routing policy to a parsed decision.

    Policy, in order:

    1. If a ``team`` choice answer names ``cheap`` or ``frontier``, that is
       the tier — unless its confidence is below *min_confidence*, in which
       case the request escalates to ``frontier`` (fail-safe).
    2. Else, if a ``team`` answer names an unknown option, escalate to
       ``frontier`` (fail-safe).
    3. Else, if an ``urgency`` noul answer exists, urgent (``noul >= 0.5``)
       maps to ``frontier`` and routine maps to ``cheap``.
    4. With no usable signal at all, escalate to ``frontier``: the router
       never silently picks the cheap tier without evidence.
    """
    team = decision.answer("team")
    if isinstance(team, ChoiceAnswer):
        if team.choice == TIER_FRONTIER:
            return RoutingDecision(
                TIER_FRONTIER, "team selected frontier", decision
            )
        if team.choice == TIER_CHEAP:
            if team.confidence >= min_confidence:
                return RoutingDecision(
                    TIER_CHEAP,
                    f"team selected cheap with confidence {team.confidence:.2f}",
                    decision,
                )
            return RoutingDecision(
                TIER_FRONTIER,
                f"team selected cheap but confidence {team.confidence:.2f} "
                f"below threshold {min_confidence:.2f}, escalated",
                decision,
            )
        return RoutingDecision(
            TIER_FRONTIER,
            f"team selected unknown option {team.choice!r}, escalated (fail-safe)",
            decision,
        )

    urgency = decision.answer("urgency")
    if isinstance(urgency, NoulAnswer):
        if urgency.noul >= 0.5:
            return RoutingDecision(
                TIER_FRONTIER, f"urgency noul {urgency.noul:.2f} >= 0.5", decision
            )
        return RoutingDecision(
            TIER_CHEAP, f"urgency noul {urgency.noul:.2f} < 0.5", decision
        )

    return RoutingDecision(
        TIER_FRONTIER,
        "no team or urgency answer found, escalated (fail-safe)",
        decision,
    )
