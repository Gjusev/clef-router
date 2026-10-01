"""Context compaction for RAG, scored by Clef in one forward pass.

Clef answers every question of a request in a single joint scoring pass, so
the natural way to rank many documents is one question per document. This
module batches those questions (the contract allows up to 64 per request),
maps the returned scores to keep-or-cut decisions against a token budget,
and reports a reason for every cut. Nothing is rewritten: documents are
kept verbatim or dropped, which keeps the compaction auditable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..client import ClefRouter
from ..errors import ClefError
from ..models import MAX_QUESTIONS, ScoreAnswer

__all__ = [
    "RELEVANCE_CRITERIA",
    "CompactResult",
    "CompactStats",
    "CutDoc",
    "ScoredDoc",
    "compact",
    "default_token_counter",
    "relevance_questions",
]

#: The relevance scale, lowest first. Scores arrive as 0..3.
RELEVANCE_CRITERIA = [
    "Irrelevant to the query: answering it does not need this document.",
    "Background only: related topic, but the query can be answered without it.",
    "Relevant: contains information the answer should use.",
    "Essential: the answer depends on this document.",
]

QUESTION_ID_TEMPLATE = "doc_{index}"


def default_token_counter(text: str) -> int:
    """Dependency-free token estimate: whitespace-separated word count."""
    return len(text.split())


def relevance_questions(count: int) -> dict[str, dict[str, Any]]:
    """Build one relevance ``score`` question per document index."""
    return {
        QUESTION_ID_TEMPLATE.format(index=index): {
            "type": "score",
            "instructions": (
                "How relevant is document number "
                f"{index} to the query? Judge the document text against the "
                "query in the state, using the given scale."
            ),
            "criteria": list(RELEVANCE_CRITERIA),
        }
        for index in range(count)
    }


@dataclass(frozen=True)
class ScoredDoc:
    """A document that survived compaction."""

    index: int
    doc: str
    score: float
    tokens: int


@dataclass(frozen=True)
class CutDoc:
    """A document that was dropped, with the reason."""

    index: int
    doc: str
    score: float
    tokens: int
    reason: str


@dataclass(frozen=True)
class CompactStats:
    """Counts for one compaction run."""

    docs_total: int
    docs_kept: int
    docs_cut: int
    budget: int
    original_tokens: int
    kept_tokens: int
    cut_tokens: int

    @property
    def savings_pct(self) -> float:
        """Share of original tokens removed, rounded to one decimal."""
        if self.original_tokens == 0:
            return 0.0
        return round(100.0 * self.cut_tokens / self.original_tokens, 1)


@dataclass(frozen=True)
class CompactResult:
    """Kept and cut documents plus run statistics."""

    kept: list[ScoredDoc] = field(default_factory=list)
    cut: list[CutDoc] = field(default_factory=list)
    stats: CompactStats | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation, used by the CLI."""
        stats = self.stats
        return {
            "kept": [
                {"index": d.index, "score": d.score, "tokens": d.tokens, "doc": d.doc}
                for d in self.kept
            ],
            "cut": [
                {
                    "index": d.index,
                    "score": d.score,
                    "tokens": d.tokens,
                    "reason": d.reason,
                    "doc": d.doc,
                }
                for d in self.cut
            ],
            "stats": (
                {
                    "docs_total": stats.docs_total,
                    "docs_kept": stats.docs_kept,
                    "docs_cut": stats.docs_cut,
                    "budget": stats.budget,
                    "original_tokens": stats.original_tokens,
                    "kept_tokens": stats.kept_tokens,
                    "cut_tokens": stats.cut_tokens,
                    "savings_pct": stats.savings_pct,
                }
                if stats
                else None
            ),
        }


def _batched(documents: list[str], size: int) -> Iterable[tuple[int, int]]:
    """Yield ``(start, end)`` index ranges of at most *size* documents."""
    for start in range(0, len(documents), size):
        yield start, min(start + size, len(documents))


def _score_documents(
    router: ClefRouter, query: str, documents: list[str]
) -> list[float]:
    """Score every document's relevance, batching to the contract limit."""
    scores: list[float] = []
    for start, end in _batched(documents, MAX_QUESTIONS):
        chunk = documents[start:end]
        questions = relevance_questions(len(chunk))
        state = {
            "query": query,
            "documents": [
                {"index": start + offset, "text": doc}
                for offset, doc in enumerate(chunk)
            ],
        }
        decision = router.decide(state, questions)
        for offset in range(len(chunk)):
            qid = QUESTION_ID_TEMPLATE.format(index=offset)
            answer = decision.answer(qid)
            if not isinstance(answer, ScoreAnswer):
                raise ClefError(
                    f"expected a score answer for {qid!r}, "
                    f"got {type(answer).__name__}"
                )
            scores.append(answer.score)
    return scores


def compact(
    query: str,
    documents: list[str],
    budget: int,
    *,
    router: ClefRouter | None = None,
    min_score: float = 1.0,
    token_counter: Callable[[str], int] | None = None,
) -> CompactResult:
    """Rank *documents* by Clef relevance and keep what fits *budget*.

    Args:
        query: The retrieval query the documents should answer.
        documents: Candidate documents, kept verbatim when selected.
        budget: Maximum tokens across kept documents.
        router: A configured :class:`ClefRouter`; created from the
            environment when omitted.
        min_score: Scores below this are cut before budget math.
        token_counter: Token estimate function; defaults to a
            whitespace word count.

    Returns:
        A :class:`CompactResult` with kept docs (highest score first),
        cut docs with reasons, and run statistics.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if not documents:
        return CompactResult()

    count_tokens = token_counter or default_token_counter
    owned_router = router is None
    active = router if router is not None else ClefRouter()
    try:
        scores = _score_documents(active, query, list(documents))
    finally:
        if owned_router:
            active.close()

    tokens = [count_tokens(doc) for doc in documents]
    original_tokens = sum(tokens)
    remaining = budget
    kept: list[ScoredDoc] = []
    cut: list[CutDoc] = []

    # Stable order: score descending, original index ascending on ties.
    order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
    for index in order:
        score = scores[index]
        if score < min_score:
            cut.append(
                CutDoc(index, documents[index], score, tokens[index],
                       f"score {score:g} below min_score {min_score:g}")
            )
            continue
        if tokens[index] <= remaining:
            kept.append(ScoredDoc(index, documents[index], score, tokens[index]))
            remaining -= tokens[index]
        else:
            cut.append(
                CutDoc(index, documents[index], score, tokens[index],
                       f"budget exhausted: needs {tokens[index]} tokens, "
                       f"{max(0, remaining)} left")
            )

    kept.sort(key=lambda d: d.index)
    cut.sort(key=lambda d: d.index)
    stats = CompactStats(
        docs_total=len(documents),
        docs_kept=len(kept),
        docs_cut=len(cut),
        budget=budget,
        original_tokens=original_tokens,
        kept_tokens=sum(d.tokens for d in kept),
        cut_tokens=sum(d.tokens for d in cut),
    )
    return CompactResult(kept=kept, cut=cut, stats=stats)
