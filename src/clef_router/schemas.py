"""Pydantic request/response schemas for the clef-router HTTP server.

These models are the typed surface of the HTTP API. They validate the
caller-facing contract and convert to plain dicts that the Clef client
understands, so the server handlers stay thin. Unknown fields are allowed
on chat requests (the OpenAI SDK sends ``extra_body`` entries at the top
level, so ``clef_questions`` arrives alongside the standard fields).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import (
    MAX_IMAGES,
    MAX_QUESTIONS,
    TYPE_CHOICE,
    TYPE_SCORE,
)

__all__ = [
    "ClefQuestionSpec",
    "ImageInput",
    "DecideRequest",
    "NoulAnswerOut",
    "ChoiceAnswerOut",
    "ScoreAnswerOut",
    "AnswerOut",
    "UsageOut",
    "DecideResponse",
    "ChatMessage",
    "ChatCompletionRequest",
]


class ClefQuestionSpec(BaseModel):
    """One Clef question, validated against the API contract.

    ``criteria`` is type-dependent: a mapping of option descriptions for
    ``choice`` questions, a list of level labels ordered lowest-first for
    ``score`` questions, and either ``None`` or a mapping with ``true``/
    ``false`` descriptions for ``noul`` questions.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def _check_criteria(self) -> ClefQuestionSpec:
        """Enforce the per-type criteria rules."""
        problems: list[str] = []
        if self.type == TYPE_CHOICE:
            if not isinstance(self.criteria, dict):
                problems.append("choice questions need criteria as a mapping "
                                "of option -> description")
            elif not 2 <= len(self.criteria) <= 255:
                problems.append("choice questions need 2..255 options, "
                                f"got {len(self.criteria)}")
        elif self.type == TYPE_SCORE:
            if not isinstance(self.criteria, list):
                problems.append("score questions need criteria as a list of "
                                "level labels ordered lowest-first")
            elif not 2 <= len(self.criteria) <= 10:
                problems.append(f"score questions need 2..10 levels, "
                                f"got {len(self.criteria)}")
        else:  # noul
            if self.criteria is not None and (
                not isinstance(self.criteria, dict)
                or not set(self.criteria) <= {"true", "false"}
            ):
                problems.append(
                    "noul questions may only set criteria with 'true'/'false' "
                    "descriptions"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def to_contract(self) -> dict[str, Any]:
        """Convert to the plain dict the Clef API expects."""
        body: dict[str, Any] = {
            "type": self.type,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


class ImageInput(BaseModel):
    """One image input, in either accepted form.

    Either a structured mapping (``content_type`` plus base64 ``data``) or
    a ready data URL string; at most four images are allowed per request.
    """

    model_config = ConfigDict(extra="forbid")

    content_type: str | None = None
    data: str | None = None
    data_url: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> ImageInput:
        """Require either the structured form or the data URL form."""
        if self.data_url is None and not (self.content_type and self.data):
            raise ValueError(
                "each image needs either data_url= or both content_type= and data="
            )
        return self

    def to_contract(self) -> dict[str, Any] | str:
        """Convert to the plain form the Clef API expects."""
        if self.data_url is not None:
            return self.data_url
        return {"content_type": self.content_type, "base64": self.data}


class DecideRequest(BaseModel):
    """Request body for ``POST /v1/decide``: a native Clef pass-through."""

    model_config = ConfigDict(extra="forbid")

    state: Any
    questions: dict[str, ClefQuestionSpec] = Field(
        min_length=1, max_length=MAX_QUESTIONS
    )
    images: list[ImageInput] | None = Field(default=None, max_length=MAX_IMAGES)

    def to_contract(self) -> tuple[Any, dict[str, dict[str, Any]], list[Any] | None]:
        """Return ``(state, questions, images)`` in Clef contract form."""
        questions = {qid: q.to_contract() for qid, q in self.questions.items()}
        images = [img.to_contract() for img in self.images] if self.images else None
        return self.state, questions, images


class NoulAnswerOut(BaseModel):
    """Typed response for a ``noul`` answer: P(yes) in [0, 1]."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswerOut(BaseModel):
    """Typed response for a ``choice`` answer."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = 0.0
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswerOut(BaseModel):
    """Typed response for a ``score`` answer."""

    type: Literal["score"] = "score"
    score: float
    confidence: float = 0.0
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)


AnswerOut = NoulAnswerOut | ChoiceAnswerOut | ScoreAnswerOut


class UsageOut(BaseModel):
    """Token usage as reported by Clef; prices are never computed here."""

    input_tokens: int = 0
    output_tokens: int = 0


class DecideResponse(BaseModel):
    """Response body for ``POST /v1/decide``."""

    model: str = ""
    answers: dict[str, AnswerOut] = Field(default_factory=dict)
    usage: UsageOut = Field(default_factory=UsageOut)


class ChatMessage(BaseModel):
    """One OpenAI-style chat message.

    ``content`` accepts the plain string form or the list-of-parts form;
    non-text parts are ignored for routing.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion request, with the clef extension field.

    ``clef_questions`` is the pass-through hook (what ``extra_body`` makes
    easy in the OpenAI SDKs); when absent, the server routes with its
    default question set and passes the system prompt along as instructions.
    Unknown top-level fields are tolerated and ignored.
    """

    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    clef_questions: dict[str, ClefQuestionSpec] | None = Field(
        default=None, max_length=MAX_QUESTIONS
    )

    def to_contract_questions(self) -> dict[str, dict[str, Any]] | None:
        """Convert ``clef_questions`` to Clef contract form, or ``None``."""
        if self.clef_questions is None:
            return None
        return {qid: q.to_contract() for qid, q in self.clef_questions.items()}


def answers_to_out(answers: dict[str, Any]) -> dict[str, AnswerOut]:
    """Convert parsed answer dataclasses to typed response models.

    Raises:
        ValueError: If an answer is not one of the three known types.
    """
    out: dict[str, AnswerOut] = {}
    for qid, answer in answers.items():
        kind = type(answer).__name__
        if kind == "NoulAnswer":
            out[qid] = NoulAnswerOut(noul=answer.noul)
        elif kind == "ChoiceAnswer":
            out[qid] = ChoiceAnswerOut(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
            )
        elif kind == "ScoreAnswer":
            out[qid] = ScoreAnswerOut(
                score=answer.score,
                confidence=answer.confidence,
                legend=answer.legend,
                probabilities=answer.probabilities,
            )
        else:
            raise ValueError(f"unknown answer type {kind} for {qid!r}")
    return out
