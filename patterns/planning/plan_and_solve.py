"""Plan-and-Solve prompting: a single-model technique, no tools, no executor.

One completion asks the model to first understand the problem and devise a
plan, then carry that plan out itself in the same generation. There is no
separate planner call, no `Step` objects, and nothing to validate or
execute: the model's own text is both the plan and the work. The PS+
variant adds explicit instructions to extract variables and compute
intermediate results, which the original paper shows cuts calculation and
missing-step errors versus plain zero-shot chain-of-thought.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, get_provider

PLAN_AND_SOLVE_SYSTEM = (
    "Let's first understand the problem and devise a plan to solve it. Then "
    "let's carry out the plan and solve the problem step by step."
)

PLAN_AND_SOLVE_PLUS_SYSTEM = (
    "Let's first understand the problem, extract relevant variables and their "
    "corresponding numerals, and devise a complete plan. Then let's carry out "
    "the plan, calculate intermediate results (pay attention to correct "
    "calculation and commonsense), solve the problem step by step, and show "
    "the answer."
)


@dataclass
class PlanAndSolveRun:
    """The outcome of one Plan-and-Solve completion.

    Attributes:
        question: The word problem that was asked.
        response: The model's full response: plan and solution together.
    """

    question: str
    response: str


def run_plan_and_solve(provider: Provider, question: str, plus: bool = False) -> PlanAndSolveRun:
    """Ask the model to plan and solve a problem in a single generation.

    Args:
        provider: Any `Provider`; this variant never passes tools.
        question: The problem to solve.
        plus: Use the PS+ system prompt (extract variables, compute
            intermediates) instead of the plain Plan-and-Solve prompt.
    """
    system = PLAN_AND_SOLVE_PLUS_SYSTEM if plus else PLAN_AND_SOLVE_SYSTEM
    completion = provider.complete([Message.user(question)], system=system)
    return PlanAndSolveRun(question=question, response=completion.content)


def demo() -> None:
    """Run Plan-and-Solve (PS+) on a small arithmetic word problem and print it."""
    question = (
        "A bakery bakes 144 cookies and packs them into boxes of 12. They have "
        "already sold 5 boxes at the morning market. How many boxes are left "
        "to sell?"
    )
    response = (
        "Plan: (1) find the total number of boxes by dividing cookies by box "
        "size, (2) subtract the boxes already sold.\n"
        "Solve: total boxes = 144 / 12 = 12. boxes left = 12 - 5 = 7.\n"
        "Answer: 7 boxes are left to sell."
    )
    provider = get_provider(script=[response])

    print("=== Plan-and-Solve (PS+) ===")
    print(f"Question: {question}")
    run = run_plan_and_solve(provider, question, plus=True)
    print(f"Model response (plan and solution in one generation):\n{run.response}")
    print("Note: no tool registry, no executor, no validation; a single")
    print("completion is both the plan and the execution.")


if __name__ == "__main__":
    demo()
