"""Tests for the compactor: scoring, budget math, reasons, adapters."""

from __future__ import annotations

import pytest

from clef_router.compactor import (
    compact,
    default_token_counter,
    relevance_questions,
)
from clef_router.compactor.langchain import ClefDocumentCompressor
from clef_router.compactor.llamaindex import (
    ClefNodePostprocessor,
    NodeWithScore,
    QueryBundle,
)
from clef_router.errors import ClefError
from clef_router.models import ClefDecision, ScoreAnswer, Usage


class FakeRouter:
    """A router whose decide() returns canned relevance scores."""

    def __init__(self, scores_by_position, capture=None):
        self.scores = scores_by_position
        self.capture = capture
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        if self.capture is not None:
            self.capture.append((state, questions))
        answers = {}
        for qid in questions:
            position = int(qid.rsplit("_", 1)[1])
            answers[qid] = ScoreAnswer(
                qid=qid, score=self.scores[position], confidence=0.9
            )
        return ClefDecision(
            model="clef-flash", answers=answers, usage=Usage(120, 10)
        )


class TestRelevanceQuestions:
    def test_one_score_question_per_document(self) -> None:
        questions = relevance_questions(3)
        assert list(questions) == ["doc_0", "doc_1", "doc_2"]
        for question in questions.values():
            assert question["type"] == "score"
            assert len(question["criteria"]) == 4

    def test_criteria_order_is_lowest_first(self) -> None:
        first = relevance_questions(1)["doc_0"]["criteria"][0]
        last = relevance_questions(1)["doc_0"]["criteria"][-1]
        assert "Irrelevant" in first
        assert "Essential" in last


class TestCompact:
    def test_scores_map_to_documents(self) -> None:
        router = FakeRouter([3.0, 0.0, 2.0])
        result = compact("q", ["a", "b", "c"], budget=100, router=router)
        assert [d.index for d in result.kept] == [0, 2]
        assert [d.score for d in result.kept] == [3.0, 2.0]

    def test_highest_scores_fill_the_budget_first(self) -> None:
        # doc lengths: 5, 5, 5 tokens; budget fits exactly two
        router = FakeRouter([1.0, 3.0, 2.0])
        result = compact("q", ["x " * 4 + "x", "y " * 4 + "y", "z " * 4 + "z"],
                         budget=10, router=router)
        assert [d.index for d in result.kept] == [1, 2]
        cut = {d.index: d.reason for d in result.cut}
        assert "budget exhausted" in cut[0]

    def test_ties_keep_original_order(self) -> None:
        router = FakeRouter([2.0, 2.0, 2.0])
        result = compact("q", ["a b", "c d", "e f"], budget=100, router=router)
        assert [d.index for d in result.kept] == [0, 1, 2]

    def test_min_score_cut_carry_reasons(self) -> None:
        router = FakeRouter([3.0, 0.5])
        result = compact("q", ["keep me", "drop me"], budget=100,
                         router=router, min_score=1.0)
        assert [d.index for d in result.kept] == [0]
        dropped = result.cut[0]
        assert dropped.index == 1
        assert "below min_score" in dropped.reason

    def test_budget_exhaustion_reason_reports_remaining(self) -> None:
        router = FakeRouter([3.0, 3.0])
        result = compact("q", ["one two three four five", "six seven eight"],
                         budget=6, router=router)
        assert [d.index for d in result.kept] == [0]
        cut = result.cut[0]
        assert "budget exhausted" in cut.reason
        assert "1 left" in cut.reason

    def test_stats_compute_savings(self) -> None:
        router = FakeRouter([3.0, 0.0])
        result = compact("q", ["a b c", "d e f"], budget=100, router=router)
        stats = result.stats
        assert stats.docs_total == 2
        assert stats.docs_kept == 1
        assert stats.docs_cut == 1
        assert stats.original_tokens == 6
        assert stats.kept_tokens == 3
        assert stats.savings_pct == 50.0

    def test_empty_documents_short_circuit(self) -> None:
        result = compact("q", [], budget=10, router=FakeRouter([]))
        assert result.kept == []
        assert result.stats is None

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            compact("q", ["a"], budget=-1, router=FakeRouter([1.0]))

    def test_custom_token_counter(self) -> None:
        router = FakeRouter([3.0])
        result = compact("q", ["abcdefgh"], budget=4, router=router,
                         token_counter=lambda text: len(text))
        assert result.kept == []  # 8 chars do not fit a 4-token budget

    def test_batches_above_the_question_limit(self) -> None:
        scores = [float(i % 4) for i in range(100)]
        captured = []
        router = FakeRouter(scores, capture=captured)
        docs = [f"doc {i}" for i in range(100)]
        result = compact("q", docs, budget=10_000, router=router)
        assert router.calls == 2  # 64 + 36
        assert [d.index for d in result.kept] == [
            i for i in range(100) if scores[i] >= 1.0
        ]

    def test_non_score_answer_is_an_error(self) -> None:
        from clef_router.models import NoulAnswer

        class WeirdRouter(FakeRouter):
            def decide(self, state, questions):
                return ClefDecision(
                    model="clef-flash",
                    answers={qid: NoulAnswer(qid=qid, noul=0.5) for qid in questions},
                    usage=Usage(1, 1),
                )

        with pytest.raises(ClefError, match="score answer"):
            compact("q", ["a"], budget=10, router=WeirdRouter([]))

    def test_to_dict_is_json_shaped(self) -> None:
        router = FakeRouter([3.0, 0.0])
        body = compact("q", ["a b", "c"], budget=100, router=router).to_dict()
        assert body["stats"]["docs_kept"] == 1
        assert body["kept"][0]["doc"] == "a b"
        assert "reason" in body["cut"][0]


class TestTokenCounter:
    def test_word_count_default(self) -> None:
        assert default_token_counter("one two three") == 3
        assert default_token_counter("") == 0


class TestLangChainAdapter:
    def test_compress_documents_keeps_relevant(self) -> None:
        class Doc:
            def __init__(self, text):
                self.page_content = text

        compressor = ClefDocumentCompressor(budget=100, min_score=1.0)
        kept = compressor.compress_documents(
            [Doc("good"), Doc("bad")], "query", router=FakeRouter([3.0, 0.0])
        )
        assert [doc.page_content for doc in kept] == ["good"]

    def test_document_construction(self) -> None:
        from clef_router.compactor.langchain import Document

        doc = Document(page_content="content", metadata={"source": "test"})
        assert doc.page_content == "content"
        assert doc.metadata == {"source": "test"}


class TestLlamaIndexAdapter:
    def test_postprocess_keeps_relevant(self) -> None:
        class Node:
            def __init__(self, text):
                self._text = text

            def get_content(self):
                return self._text

        postprocessor = ClefNodePostprocessor(budget=100, min_score=1.0)
        nodes = [NodeWithScore(Node("good")), NodeWithScore(Node("bad"))]
        kept = postprocessor._postprocess_nodes(
            nodes, QueryBundle("query"), router=FakeRouter([3.0, 0.0])
        )
        assert [n.node.get_content() for n in kept] == ["good"]

    def test_missing_query_bundle_raises(self) -> None:
        postprocessor = ClefNodePostprocessor()
        with pytest.raises(ValueError, match="query_bundle"):
            postprocessor._postprocess_nodes([], None)
