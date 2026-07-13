"""Tests for the reflection pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern's own demo modules or through small scripts built inline, with
no network call and no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Message, MockProvider

from patterns.reflection import generator_critic, reflexion, rubric, self_refine, tool_grounded
from patterns.reflection.loop import Critique, parse_critique, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine


# --- parse_critique edge cases ---------------------------------------------


def test_parse_critique_extracts_score() -> None:
    crit = parse_critique("SCORE: 7\nThe intro is weak.")
    assert crit.score == 7.0
    assert crit.approved is False


def test_parse_critique_extracts_decimal_score() -> None:
    crit = parse_critique("SCORE: 8.5/10\nAlmost there.")
    assert crit.score == 8.5


def test_parse_critique_no_score_is_none() -> None:
    crit = parse_critique("This draft needs more detail on the mechanism.")
    assert crit.score is None


def test_parse_critique_approved_field_yes() -> None:
    crit = parse_critique("APPROVED: yes\nSCORE: 9\nLooks solid.")
    assert crit.approved is True


def test_parse_critique_approved_field_no() -> None:
    crit = parse_critique("APPROVED: no\nSCORE: 4\nNeeds work.")
    assert crit.approved is False


def test_parse_critique_bare_sentinel_phrase() -> None:
    crit = parse_critique("No changes needed. This is ready to publish.")
    assert crit.approved is True


def test_parse_critique_empty_text_is_blank() -> None:
    crit = parse_critique("")
    assert crit == Critique(comments="", score=None, approved=False)


def test_parse_critique_whitespace_only_is_blank() -> None:
    crit = parse_critique("   \n  ")
    assert crit.comments == ""


# --- run_reflection_loop mechanics -----------------------------------------


def _scripted_loop(
    drafts: list[str],
    critiques: list[str],
    *,
    max_iterations: int = 3,
    score_threshold: float | None = None,
):
    """Build a run_reflection_loop call from a plain script of drafts/critiques.

    `drafts` supplies the initial draft plus one entry per refine call, in
    order. `critiques` supplies one raw critique string per round.
    """
    draft_iter = iter(drafts)
    critique_iter = iter(critiques)

    def generate() -> str:
        return next(draft_iter)

    def critique(_draft: str) -> Critique:
        return parse_critique(next(critique_iter))

    def refine(_draft: str, _crit: Critique) -> str:
        return next(draft_iter)

    return run_reflection_loop(
        generate, critique, refine, max_iterations=max_iterations, score_threshold=score_threshold
    )


def test_loop_hits_max_iterations_when_never_approved() -> None:
    result = _scripted_loop(
        drafts=["d1", "d2", "d3"],
        critiques=["SCORE: 3\nweak", "SCORE: 4\nstill weak", "SCORE: 5\nstill not there"],
        max_iterations=3,
    )
    assert len(result.iterations) == 3
    assert result.stop_reason == "max_iterations"


def test_loop_stops_early_on_approval_sentinel() -> None:
    result = _scripted_loop(
        drafts=["d1", "d2"],
        critiques=["SCORE: 4\nweak", "APPROVED: yes\nSCORE: 9\ngood"],
        max_iterations=5,
    )
    assert len(result.iterations) == 2
    assert len(result.iterations) < 5
    assert result.stop_reason == "approved"


def test_loop_stops_early_on_score_threshold() -> None:
    result = _scripted_loop(
        drafts=["d1", "d2", "d3"],
        critiques=["SCORE: 4\nweak", "SCORE: 6\nbetter", "SCORE: 9\ngood enough"],
        max_iterations=5,
        score_threshold=8.0,
    )
    assert len(result.iterations) == 3
    assert result.stop_reason == "score_threshold"


def test_loop_ends_on_first_crossing_of_rising_scores() -> None:
    result = _scripted_loop(
        drafts=["d1", "d2", "d3", "d4"],
        critiques=["SCORE: 2\n", "SCORE: 5\n", "SCORE: 7\n", "SCORE: 8\n"],
        max_iterations=10,
        score_threshold=7.0,
    )
    assert len(result.iterations) == 3
    assert result.best_score == 7.0


def test_loop_refined_output_differs_from_initial_draft() -> None:
    result = _scripted_loop(
        drafts=["weak draft", "improved draft"],
        critiques=["SCORE: 4\nneeds work", "APPROVED: yes\nSCORE: 9\ngood"],
    )
    assert result.final_draft != result.initial_draft
    assert result.initial_draft == "weak draft"
    assert result.final_draft == "improved draft"


def test_loop_passes_critique_text_into_refine_prompt() -> None:
    provider = MockProvider(
        [
            "first draft",
            "SCORE: 3\nname the target audience explicitly",
            "second draft naming the audience",
            "APPROVED: yes\nSCORE: 9\ngood",
        ]
    )
    generate = make_generate(provider, "write a tagline", system="you write taglines")
    critique = make_critique(provider, "write a tagline", system="you critique taglines")
    refine = make_refine(provider, "write a tagline", system="you write taglines")
    run_reflection_loop(generate, critique, refine, max_iterations=3)

    refine_call = provider.calls[2]
    refine_prompt = refine_call["messages"][0].content
    assert "name the target audience explicitly" in refine_prompt


def test_loop_best_so_far_survives_a_regression() -> None:
    result = _scripted_loop(
        drafts=["d1", "d2", "d3"],
        critiques=["SCORE: 6\n", "SCORE: 9\n", "SCORE: 5\n"],
        max_iterations=3,
    )
    assert result.best_score == 9.0
    assert result.best_draft == "d2"
    assert result.final_draft == "d3"
    assert result.best_draft != result.final_draft


def test_loop_empty_critique_stops_immediately_output_unchanged() -> None:
    result = _scripted_loop(drafts=["only draft"], critiques=[""], max_iterations=5)
    assert len(result.iterations) == 1
    assert result.stop_reason == "empty_critique"
    assert result.final_draft == "only draft"
    assert result.best_draft == "only draft"


def test_loop_blank_refinement_guard_keeps_prior_draft() -> None:
    provider = MockProvider(["draft one", "SCORE: 4\nneeds work", ""])
    generate = make_generate(provider, "task", system="s")
    critique = make_critique(provider, "task", system="s")
    refine = make_refine(provider, "task", system="s")
    result = run_reflection_loop(generate, critique, refine, max_iterations=5)
    assert result.stop_reason == "blank_refinement"
    assert result.final_draft == "draft one"


def test_loop_no_change_convergence_stops_the_loop() -> None:
    result = _scripted_loop(
        drafts=["same text", "same text"],
        critiques=["SCORE: 5\nkeep the tone but tighten it"],
        max_iterations=5,
    )
    assert result.stop_reason == "no_change"
    assert len(result.iterations) == 1


def test_loop_transcript_length_matches_iteration_count() -> None:
    result = _scripted_loop(
        drafts=["d1", "d2", "d3"],
        critiques=["SCORE: 3\n", "SCORE: 4\n", "SCORE: 5\n"],
        max_iterations=3,
    )
    assert len(result.iterations) == 3
    for i, iteration in enumerate(result.iterations, start=1):
        assert iteration.index == i


# --- variant modules ---------------------------------------------------


def test_self_refine_demo_converges_and_revises() -> None:
    result = self_refine.run_self_refine_demo()
    assert result.stop_reason == "approved"
    assert result.final_draft != result.initial_draft
    assert "hash function" in result.best_draft


def test_self_refine_guard_demo_stops_on_empty_critique() -> None:
    result = self_refine.run_guard_demo()
    assert result.stop_reason == "empty_critique"
    assert result.final_draft == result.initial_draft


def test_generator_critic_demo_uses_two_independent_providers() -> None:
    generator_provider = MockProvider(["vague draft", "specific revised draft"])
    critic_provider = MockProvider(["SCORE: 3\nno specifics", "APPROVED: yes\nSCORE: 9\ngood"])
    result = generator_critic.run_generator_critic_demo(generator_provider, critic_provider)
    assert result.stop_reason == "approved"
    # the critic never receives the generator's system prompt persona
    assert critic_provider.calls[0]["system"] != generator_provider.calls[0]["system"]


def test_generator_critic_demo_frames_draft_as_external_submission() -> None:
    generator_provider = MockProvider(["vague draft", "specific revised draft"])
    critic_provider = MockProvider(["SCORE: 3\nno specifics", "APPROVED: yes\nSCORE: 9\ngood"])
    generator_critic.run_generator_critic_demo(generator_provider, critic_provider)

    critic_prompt = critic_provider.calls[0]["messages"][0].content
    assert "submitted by another author" in critic_prompt
    assert "your previous answer" not in critic_prompt


def test_rubric_demo_returns_best_scoring_not_last_draft() -> None:
    result = rubric.run_rubric_demo()
    assert result.stop_reason == "max_iterations"
    assert result.best_score == 8.5
    assert result.best_draft != result.final_draft


def test_tool_grounded_demo_fails_then_passes_the_checker() -> None:
    result, action = tool_grounded.run_tool_grounded_demo()
    assert result.iterations[0].critique.approved is False
    assert "Level" in result.iterations[0].critique.comments
    assert result.stop_reason == "approved"
    assert action.startswith("AUTHORIZED")


def test_tool_grounded_checker_runs_offline_with_no_llm_in_critique() -> None:
    # A model that keeps returning differently-worded but still buggy,
    # case-sensitive code each time it is asked to refine, so each refine
    # actually changes the text and the loop runs to the iteration cap
    # instead of tripping the no-change convergence guard.
    def buggy(n: int) -> str:
        return f"```python\ndef is_palindrome(s):\n    # attempt {n}\n    return s == s[::-1]\n```"

    provider = MockProvider([buggy(1), buggy(2), buggy(3)])
    result, action = tool_grounded.run_tool_grounded_demo(provider)
    refine_rounds = sum(1 for it in result.iterations if it.note == "refine")
    # provider calls = 1 generate + N refines; the 3 critique rounds that
    # ran never touched the provider, since the checker grounds them
    assert len(provider.calls) == 1 + refine_rounds
    assert result.stop_reason == "max_iterations"
    assert action.startswith("BLOCKED")


def test_reflexion_demo_carries_lesson_into_second_attempt_prompt() -> None:
    result, memory = reflexion.run_reflexion_demo()
    assert result.stop_reason == "approved"
    assert len(memory) == 1
    lesson = memory[0]
    assert "8 boxes" in lesson
    assert result.iterations[0].note == "refine"


def test_reflexion_lesson_text_appears_in_retry_prompt() -> None:
    provider = MockProvider(
        [
            "Rina needs 7 boxes, with 3 cupcakes left over.",
            "Rina needs 8 boxes, with 3 cupcakes left over.",
        ]
    )
    reflexion.run_reflexion_demo(provider)
    retry_prompt = provider.calls[1]["messages"][0].content
    assert "42 cupcakes" in retry_prompt
    assert "Lessons from earlier attempts" in retry_prompt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
