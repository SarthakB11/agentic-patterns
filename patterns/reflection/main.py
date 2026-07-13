"""Reflection pattern: generate, critique, refine, repeat.

Reflection is the pattern where an agent evaluates its own work against
explicit criteria and uses that evaluation to revise the work: a producer
drafts, a critic names what is wrong, and a refiner rewrites, repeating
until the output clears a bar or a budget runs out. Current framework docs
(LangGraph, OpenAI Agents SDK) call the same loop evaluator-optimizer.

This demo runs five sub-variants end to end, entirely offline against
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

from patterns.reflection import generator_critic, reflexion, rubric, self_refine, tool_grounded
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

    print("All five sub-variants completed without exhausting their scripts.")


def _best_round(result: ReflectionResult) -> int:
    """Find which iteration's draft ended up as the returned best draft."""
    for it in result.iterations:
        if it.draft == result.best_draft:
            return it.index
    return 0


if __name__ == "__main__":
    main()
