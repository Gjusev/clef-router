"""Tests for question validation, decision parsing, and tier derivation."""

from __future__ import annotations

import pytest

from clef_router.errors import ClefResponseError
from clef_router.models import (
    DEFAULT_QUESTIONS,
    ChoiceAnswer,
    ClefDecision,
    NoulAnswer,
    ScoreAnswer,
    Usage,
    derive_tier,
    parse_decision,
    validate_questions,
)
from tests.conftest import load_fixture

TIER_CHEAP = "cheap"
TIER_FRONTIER = "frontier"


def cheap_confidence(confidence: float) -> ClefDecision:
    """A decision whose team answer picks cheap with *confidence*."""
    return parse_decision(
        {
            "success": True,
            "result": {
                "model": "clef-flash",
                "answers": {
                    "team": {
                        "type": "choice",
                        "choice": TIER_CHEAP,
                        "confidence": confidence,
                        "probabilities": {TIER_CHEAP: confidence},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        }
    )


class TestValidateQuestions:
    def test_default_questions_are_valid(self) -> None:
        assert validate_questions(DEFAULT_QUESTIONS) == []

    def test_empty_mapping_rejected(self) -> None:
        assert validate_questions({}) != []

    def test_more_than_64_questions_rejected(self) -> None:
        questions = {
            f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)
        }
        problems = validate_questions(questions)
        assert any("at most 64" in p for p in problems)

    def test_unknown_type_rejected(self) -> None:
        problems = validate_questions({"q": {"type": "vibe", "instructions": "x"}})
        assert any("type" in p for p in problems)

    def test_blank_instructions_rejected(self) -> None:
        problems = validate_questions({"q": {"type": "noul", "instructions": "  "}})
        assert any("instructions" in p for p in problems)

    def test_choice_needs_mapping_criteria_with_2_to_255_options(self) -> None:
        problems = validate_questions(
            {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}
        )
        assert any("criteria" in p for p in problems)
        problems = validate_questions(
            {"q": {"type": "choice", "instructions": "x", "criteria": {"only": "a"}}}
        )
        assert any("2..255" in p for p in problems)

    def test_score_needs_list_of_2_to_10_levels(self) -> None:
        problems = validate_questions(
            {"q": {"type": "score", "instructions": "x", "criteria": {"a": "b"}}}
        )
        assert any("criteria" in p for p in problems)
        problems = validate_questions(
            {"q": {"type": "score", "instructions": "x", "criteria": ["just-one"]}}
        )
        assert any("2..10" in p for p in problems)

    def test_noul_criteria_may_only_map_true_false(self) -> None:
        problems = validate_questions(
            {
                "q": {
                    "type": "noul",
                    "instructions": "x",
                    "criteria": {"yes": "a", "no": "b"},
                }
            }
        )
        assert any("true" in p and "false" in p for p in problems)


class TestParseDecision:
    def test_cheap_envelope_parses_fully(self) -> None:
        decision = parse_decision(load_fixture("decide_cheap.json"), latency_ms=12.5)
        assert decision.model == "clef-flash"
        team = decision.answer("team")
        assert isinstance(team, ChoiceAnswer)
        assert team.choice == TIER_CHEAP
        assert team.confidence == pytest.approx(0.91)
        assert team.probabilities[TIER_FRONTIER] == pytest.approx(0.09)
        urgency = decision.answer("urgency")
        assert isinstance(urgency, NoulAnswer)
        assert urgency.noul == pytest.approx(0.03)
        assert decision.usage == Usage(input_tokens=148, output_tokens=12)
        assert decision.latency_ms == pytest.approx(12.5)
        assert decision.raw_response["success"] is True

    def test_frontier_envelope_parses(self) -> None:
        decision = parse_decision(load_fixture("decide_frontier.json"))
        team = decision.answer("team")
        assert isinstance(team, ChoiceAnswer)
        assert team.choice == TIER_FRONTIER

    def test_score_answer_parses_with_legend(self) -> None:
        decision = parse_decision(
            {
                "result": {
                    "model": "clef",
                    "answers": {
                        "severity": {
                            "type": "score",
                            "score": 2.4,
                            "confidence": 0.7,
                            "legend": {"0": "low", "1": "mid", "2": "high"},
                            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                        }
                    },
                }
            }
        )
        severity = decision.answer("severity")
        assert isinstance(severity, ScoreAnswer)
        assert severity.score == pytest.approx(2.4)
        assert severity.legend["2"] == "high"

    def test_missing_result_raises(self) -> None:
        with pytest.raises(ClefResponseError, match="result"):
            parse_decision({"success": True})

    def test_missing_answers_raises(self) -> None:
        with pytest.raises(ClefResponseError, match="answers"):
            parse_decision({"result": {"model": "clef"}})

    def test_unknown_answer_type_raises(self) -> None:
        with pytest.raises(ClefResponseError, match="unknown type"):
            parse_decision(
                {"result": {"answers": {"q": {"type": "vibe", "vibe": 1}}}}
            )

    def test_choice_answer_without_choice_raises(self) -> None:
        with pytest.raises(ClefResponseError, match="choice"):
            parse_decision(
                {"result": {"answers": {"q": {"type": "choice", "confidence": 0.5}}}}
            )

    def test_noul_answer_without_number_raises(self) -> None:
        with pytest.raises(ClefResponseError, match="noul"):
            parse_decision(
                {"result": {"answers": {"q": {"type": "noul", "noul": "high"}}}}
            )

    def test_score_answer_without_score_raises(self) -> None:
        with pytest.raises(ClefResponseError, match="score"):
            parse_decision(
                {"result": {"answers": {"q": {"type": "score", "confidence": 0.5}}}}
            )

    def test_to_dict_is_json_shaped(self) -> None:
        decision = parse_decision(load_fixture("decide_cheap.json"))
        body = decision.to_dict()
        assert body["model"] == "clef-flash"
        assert body["answers"]["team"]["choice"] == TIER_CHEAP
        assert body["usage"] == {"input_tokens": 148, "output_tokens": 12}


class TestDeriveTier:
    def test_team_cheap_with_confidence_stays_cheap(self) -> None:
        routing = derive_tier(cheap_confidence(0.91), min_confidence=0.45)
        assert routing.tier == TIER_CHEAP
        assert not routing.escalated
        assert "confidence 0.91" in routing.reason

    def test_team_cheap_below_threshold_escalates(self) -> None:
        routing = derive_tier(cheap_confidence(0.20), min_confidence=0.45)
        assert routing.tier == TIER_FRONTIER
        assert "below threshold 0.45" in routing.reason

    def test_zero_gate_disables_confidence_escalation(self) -> None:
        routing = derive_tier(cheap_confidence(0.20), min_confidence=0.0)
        assert routing.tier == TIER_CHEAP

    def test_team_frontier_goes_frontier_regardless_of_gate(self) -> None:
        decision = parse_decision(load_fixture("decide_frontier.json"))
        routing = derive_tier(decision, min_confidence=0.99)
        assert routing.tier == TIER_FRONTIER
        assert "team selected frontier" in routing.reason

    def test_unknown_team_option_escalates_fail_safe(self) -> None:
        decision = parse_decision(
            {
                "result": {
                    "answers": {
                        "team": {
                            "type": "choice",
                            "choice": "billing",
                            "confidence": 0.9,
                        }
                    }
                }
            }
        )
        routing = derive_tier(decision, min_confidence=0.45)
        assert routing.tier == TIER_FRONTIER
        assert "fail-safe" in routing.reason

    def test_urgent_prompt_escalates_without_team_answer(self) -> None:
        decision = parse_decision(
            {
                "result": {
                    "answers": {
                        "urgency": {"type": "noul", "noul": 0.87},
                    }
                }
            }
        )
        routing = derive_tier(decision, min_confidence=0.45)
        assert routing.tier == TIER_FRONTIER
        assert "urgency" in routing.reason

    def test_routine_prompt_stays_cheap_without_team_answer(self) -> None:
        decision = parse_decision(
            {
                "result": {
                    "answers": {
                        "urgency": {"type": "noul", "noul": 0.03},
                    }
                }
            }
        )
        routing = derive_tier(decision, min_confidence=0.45)
        assert routing.tier == TIER_CHEAP

    def test_no_signal_escalates_fail_safe(self) -> None:
        decision = parse_decision({"result": {"answers": {}}})
        routing = derive_tier(decision, min_confidence=0.45)
        assert routing.tier == TIER_FRONTIER
        assert "fail-safe" in routing.reason

    def test_routing_to_dict_carries_nested_decision(self) -> None:
        routing = derive_tier(cheap_confidence(0.91), min_confidence=0.45)
        body = routing.to_dict()
        assert body["tier"] == TIER_CHEAP
        assert body["decision"]["answers"]["team"]["choice"] == TIER_CHEAP
