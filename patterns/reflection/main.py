"""Reflection pattern: generate, critique, refine, repeat.

Reflection is the pattern where an agent evaluates its own work against
explicit criteria and uses that evaluation to revise the work: a producer
drafts, a critic names what is wrong, and a refiner rewrites, repeating
until the output clears a bar or a budget runs out. Current framework docs
(LangGraph, OpenAI Agents SDK) call the same loop evaluator-optimizer.

This demo runs nine sub-variants end to end, entirely offline against
`MockProvider` with scripted, coherent conversations, no network call and
no API key:

1. Self-refine: one model plays generator, critic, and refiner.
2. Generator/critic separation: two independent models, plus external
   framing of the draft to the critic to avoid the self-correction blind
   spot.
3. Rubric-based, score-gated stopping, with best-so-far tracking shown by a
   score sequence that peaks then regresses.
4. Tool-grounded critique (verifier-gated action): the critic is a
   deterministic local test runner, not a model, and a pass both stops the
   loop and authorizes a terminal action.
5. Memory-augmented (Reflexion-style) reflection: a failed attempt writes a
   verbal lesson to memory that a later attempt reads.
6. Multi-critic aggregation: three specialist lenses run in parallel and
   are combined by a veto policy, then a weighted policy where one heavily
   weighted lens flips the verdict.
7. Sampled-verdict judging: one critic sampled three times per round,
   aggregated by median score and majority approval, to denoise a single
   noisy judge call.
8. Adaptive stop: a pre-critique revision gate that skips an already-good
   draft, and a diminishing-returns stop that fires on a plateaued score
   before the iteration cap.
9. Reasoning critic: a single extended-thinking pass benchmarked against
   the explicit multi-call loop on the same task.

It also runs the empty-critique guard on its own, showing a loop stop
immediately with the draft returned unrevised.

Run it from the repository root:

    python -m patterns.reflection.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

from patterns.reflection import (
    adaptive_stop,
    generator_critic,
    multi_critic,
    reasoning_critic,
    reflexion,
    rubric,
    sampled_verdict,
    self_refine,
    tool_grounded,
)
from patterns.reflection.loop import ReflectionResult
from patterns.reflection.transcript import format_transcript


def main() -> None:
    """Run every reflection sub-variant demo and print a readable transcript."""
    print("REFLECTION PATTERN: generate, critique, refine\n")

    result = self_refine.run_self_refine_demo()
    print(format_transcript(result, title="1. Self-refine (single model, all roles)"))
    print()

    guard_result = self_refine.run_guard_demo()
    print(format_transcript(guard_result, title="1b. Empty-critique guard"))
    assert guard_result.final_draft == guard_result.initial_draft
    print()

    result = generator_critic.run_generator_critic_demo()
    print(format_transcript(result, title="2. Generator/critic separation, external framing"))
    print()

    result = rubric.run_rubric_demo()
    print(format_transcript(result, title="3. Rubric-based, score-gated stopping"))
    print(f"note: best draft is round {_best_round(result)}, not the final round {len(result.iterations)}")
    print()

    tool_result, action = tool_grounded.run_tool_grounded_demo()
    print(format_transcript(tool_result, title="4. Tool-grounded critique (verifier-gated action)"))
    print(action)
    print()

    reflexion_result, memory = reflexion.run_reflexion_demo()
    print(format_transcript(reflexion_result, title="5. Memory-augmented (Reflexion-style) reflection"))
    print(f"episodic memory carried into later attempts: {memory}")
    print()

    result = multi_critic.run_multi_critic_demo()
    print(format_transcript(result, title="6. Multi-critic aggregation (veto policy)"))
    print()

    weighted_result = multi_critic.run_weighted_flip_demo()
    weighted_scores = [it.critique.score for it in weighted_result.iterations]
    print("6b. Weighted policy: a 3x-weighted safety lens flips the verdict")
    print(f"    round scores {weighted_scores}: round 1 fails threshold 8.0 despite correctness=9, style=8")
    print(f"    stopped: {weighted_result.stop_reason} after {len(weighted_result.iterations)} round(s)")
    print()

    sampled_result, sample_log = sampled_verdict.run_sampled_verdict_demo()
    print("7. Sampled-verdict judging: one critic sampled 3 times per round")
    for round_index, scores in enumerate(sample_log, start=1):
        scored = [score for score in scores if score is not None]
        assert len(scored) == len(scores), "sampled-verdict demo script always scores every sample"
        sorted_scores = sorted(scored)
        print(f"   round {round_index} samples: {scores} -> median {sorted_scores[len(sorted_scores) // 2]:g}")
    print(f"   stopped: {sampled_result.stop_reason} after {len(sampled_result.iterations)} round(s), "
          f"final answer: {sampled_result.final_draft[:60]!r}...")
    print()

    gate_skip_result = adaptive_stop.run_gate_skip_demo()
    print(format_transcript(gate_skip_result, title="8. Adaptive stop: revision gate skips an already-good draft"))
    print()

    diminishing_result = adaptive_stop.run_diminishing_returns_demo()
    round_scores = [it.critique.score for it in diminishing_result.iterations]
    print("8b. Adaptive stop: diminishing-returns stop")
    print(f"    round scores {round_scores}: round 3's gain (0.2) is below epsilon (0.5)")
    print(f"    stopped: {diminishing_result.stop_reason} after {len(diminishing_result.iterations)} round(s), "
          f"never reaching the round-5 cap it was budgeted for")
    print()

    benchmark = reasoning_critic.run_benchmark()
    print("9. Reasoning critic: single reasoning pass vs the explicit loop")
    print(f"   single-pass reasoning trace: {benchmark.single_reasoning}")
    print(f"   single-pass answer: {benchmark.single_answer!r} in {benchmark.single_calls} provider call")
    print(f"   explicit-loop answer: {benchmark.loop_answer!r} in {benchmark.loop_calls} provider calls")
    print(f"   answers agree: {reasoning_critic.answers_agree(benchmark)}")
    print()

    print("All nine sub-variants completed without exhausting their scripts.")


def _best_round(result: ReflectionResult) -> int:
    """Find which iteration's draft ended up as the returned best draft."""
    for it in result.iterations:
        if it.draft == result.best_draft:
            return it.index
    return 0


if __name__ == "__main__":
    main()
