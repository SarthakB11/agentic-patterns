"""Sub-module: native reasoning self-critique, benchmarked against the loop.

Every other variant here scaffolds critique as separate provider calls. A
reasoning model (DeepSeek-R1-style) performs generate, self-check, and
revise inside one extended-thinking generation instead, carried in the
opaque `reasoning` channel on `Completion`
(`agentic_patterns.core.types.Completion.reasoning`). This reads that
channel as an already-self-corrected answer and benchmarks a single
reasoning pass against the explicit multi-call loop on the same task, so an
engineer decides whether to add a loop at all. `run_single_pass` treats
`reasoning` as opaque, captured for inspection and never parsed or
rewritten; no separate critic system prompt is sent, since the self-check
happens inside the one generation the model already produced.

Ranked last because this is a benchmark plus a note, not a new loop shape,
in the spirit of the ReAct folder's reasoning module. The caveat it exists
to teach: forcing a reasoning model through the external scaffold is not
free. Replication studies report these models re-checking and
self-overturning already-correct answers, concentrated in the first turn
(survey, arXiv:2505.00551); NoWait (Wang et al., arXiv:2506.08343) removes
explicit thinking tokens with no accuracy loss; structure snowballing
(Zhou, arXiv:2604.06066) shows constrained decoding can make a model chase
format and miss the semantic error; the overthinking survey
(arXiv:2508.02120) catalogs the degradation as measured. Decision rule:
benchmark the single pass first, add the loop only when it measurably
underperforms and a grounded verifier exists to gate it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_patterns import Completion, Message, Provider, get_provider

from patterns.reflection.loop import ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = "A rectangle is 14 units long and 9 units wide. What is its area, in square units?"

_SINGLE_PASS_SYSTEM = (
    "You are a careful reasoning model. Solve the problem and state the final "
    "numeric answer as the last line, prefixed with 'Answer:'."
)

_LOOP_GENERATOR_SYSTEM = "You solve short arithmetic word problems. State the final numeric answer as the last line."

_LOOP_CRITIC_SYSTEM = (
    "You check arithmetic answers against the stated problem. Reply with a "
    "SCORE out of 10 and comments naming the error, if any. If the numeric "
    "answer is correct, start your reply with APPROVED."
)

_ANSWER_RE = re.compile(r"answer:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _normalize_answer(text: str) -> str | None:
    """Pull the final numeric answer out of a response for parity comparison."""
    match = _ANSWER_RE.search(text)
    return match.group(1) if match else None


def run_single_pass(provider: Provider, task: str = _TASK, *, system: str = _SINGLE_PASS_SYSTEM) -> Completion:
    """Call a reasoning-capable provider once and return the raw completion.

    No external critique or refine call is made. `completion.reasoning`
    holds whatever self-check trace the model produced; it is returned
    as-is for inspection, never parsed or rewritten.

    Args:
        provider: The reasoning-capable model, called exactly once.
        task: The problem to solve.
        system: System prompt asking for a final-line numeric answer.

    Returns:
        The raw `Completion`, with `content` as the self-corrected final
        answer and `reasoning` as the opaque self-check trace.
    """
    return provider.complete([Message.user(task)], system=system)


@dataclass
class ReasoningBenchmark:
    """Result of comparing the explicit loop against a single reasoning pass.

    Attributes:
        loop_answer: The explicit loop's final draft.
        loop_calls: Number of provider calls the explicit loop made.
        single_answer: The single reasoning pass's content.
        single_calls: Number of provider calls the single pass made (always 1).
        single_reasoning: The single pass's captured reasoning trace.
    """

    loop_answer: str
    loop_calls: int
    single_answer: str
    single_calls: int
    single_reasoning: str


def run_benchmark(
    loop_provider: Provider | None = None, single_pass_provider: Provider | None = None, task: str = _TASK
) -> ReasoningBenchmark:
    """Run the same task through the explicit loop and the single reasoning pass.

    Args:
        loop_provider: Drives the loop's generate, critique, and refine
            calls (self-refine shape). Defaults to a `MockProvider`
            scripted with a wrong first answer, a critique naming the
            error, and a corrected, approved second answer: 4 calls.
        single_pass_provider: Drives `run_single_pass`. Defaults to a
            `MockProvider` scripted with one completion whose `reasoning`
            shows a self-check catching the same arithmetic slip in place.
        task: The problem given to both paths.

    Returns:
        A `ReasoningBenchmark` with both answers and both call counts, so a
        caller can compare call economy directly.
    """
    if loop_provider is None:
        loop_provider = get_provider(
            script=[
                "14 times 9 is 108.\nAnswer: 108",
                "SCORE: 2\n14 * 9 = 126, not 108. Recompute the multiplication.",
                "14 times 9: 14 * 9 = 126.\nAnswer: 126",
                "APPROVED: yes\nSCORE: 10\nCorrect: 126.",
            ]
        )
    if single_pass_provider is None:
        single_pass_provider = get_provider(
            script=[
                Completion(
                    content="14 times 9: 14 * 9 = 126.\nAnswer: 126",
                    reasoning=(
                        "Let me compute 14 * 9. First pass: 14 * 9 = 108, that used 9*12 "
                        "by mistake. Let me redo it: 14 * 9 = 14*10 - 14 = 140 - 14 = 126. "
                        "126 is correct."
                    ),
                )
            ]
        )

    generate = make_generate(loop_provider, task, system=_LOOP_GENERATOR_SYSTEM)
    critique = make_critique(loop_provider, task, system=_LOOP_CRITIC_SYSTEM)
    refine = make_refine(loop_provider, task, system=_LOOP_GENERATOR_SYSTEM)
    loop_result: ReflectionResult = run_reflection_loop(generate, critique, refine, max_iterations=3)

    single_completion = run_single_pass(single_pass_provider, task)

    return ReasoningBenchmark(
        loop_answer=loop_result.final_draft,
        loop_calls=len(loop_provider.calls),
        single_answer=single_completion.content,
        single_calls=len(single_pass_provider.calls),
        single_reasoning=single_completion.reasoning,
    )


def answers_agree(benchmark: ReasoningBenchmark) -> bool:
    """True if the loop's and the single pass's normalized final answers match."""
    return _normalize_answer(benchmark.loop_answer) == _normalize_answer(benchmark.single_answer)
