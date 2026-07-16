"""Pointwise LLM judge: rubric scoring with chain-of-thought (G-Eval style).

The judge reads one output, reasons through named evaluation steps, then
reports a structured verdict. This is the base pointwise judge; it also
demonstrates three refinements called for in the taxonomy:

- Reference-based vs reference-free grading, selectable on the same judge
  by whether a reference answer is included in the prompt.
- An instruction-specific checklist judge: instead of one fixed 1-10 scale,
  the judge first derives a short checklist for the specific case, then
  scores each item, which is finer-grained and more auditable
  (arXiv:2507.17746, "Rubrics as Rewards"). That paper is a training-time
  RL reward method whose rubric items carry weights (essential versus
  optional); this checklist scores `passed / total` unweighted, an
  intentional fidelity simplification. RaR is a training reward being
  reused here as an eval rubric, not an evaluation paper.
- A position-swap check for pointwise judging itself. Position bias is
  usually discussed for pairwise setups, but "Am I More Pointwise or
  Pairwise?" (arXiv:2602.02219) found ordering effects inside rubric-based
  pointwise judging too: listing the same criteria in a different order can
  move the score. This module runs the rubric both ways and reports whether
  the scores agree.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agentic_patterns import Message, Provider, get_provider
from patterns.evaluation.eval_set import EvalCase, get_case
from patterns.evaluation.verdict import Verdict, parse_pointwise_verdict

_RUBRIC_SYSTEM = (
    "You grade a support bot's reply. Evaluation steps: (1) check factual "
    "accuracy against any reference given, (2) check the reply names a "
    "concrete next step, (3) check tone is professional and concise. "
    "Reason through each step briefly, then end with a SCORE line (0-10) "
    "and a VERDICT line (pass if SCORE >= 7, else fail)."
)

_CHECKLIST_SYSTEM_DERIVE = (
    "Given a support task, write a short checklist of 3 specific, checkable "
    "criteria a good reply must satisfy. Reply with a CHECKLIST line "
    "followed by numbered items, nothing else."
)

_CHECKLIST_SYSTEM_SCORE = (
    "Score the reply against the given checklist. For each item, mark it "
    "PASS or FAIL with a one-clause reason, then end with a line "
    "CHECKLIST_SCORE: <passed>/<total>."
)

_CHECKLIST_SCORE_RE = re.compile(r"CHECKLIST_SCORE:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def build_pointwise_judge(provider: Provider, *, reference_mode: bool = True) -> Callable[[EvalCase, str], Verdict]:
    """Build a pointwise judge callable bound to `provider`.

    Args:
        provider: The model that plays the judge.
        reference_mode: If True, include `case.reference` in the prompt
            when the case has one (reference-based judging). If False, omit
            it even when present, so the judge grades quality with no gold
            answer to compare against (reference-free judging).
    """

    def judge(case: EvalCase, output: str) -> Verdict:
        parts = [f"Task given to the bot:\n{case.input}", f"Bot's reply:\n{output}"]
        if reference_mode and case.reference is not None:
            parts.insert(1, f"Reference answer:\n{case.reference}")
        completion = provider.complete([Message.user("\n\n".join(parts))], system=_RUBRIC_SYSTEM)
        return parse_pointwise_verdict(completion.content)

    return judge


def run_checklist_judgment(provider: Provider, case: EvalCase, output: str) -> tuple[str, Verdict]:
    """Run the two-call instruction-specific checklist judge.

    First asks the judge to derive a checklist for this specific case, then
    asks it to score the output against that checklist.

    Returns:
        A tuple of the derived checklist text and the resulting `Verdict`,
        with `Verdict.score` set to `passed_items / total_items`.
    """
    checklist_completion = provider.complete([Message.user(case.input)], system=_CHECKLIST_SYSTEM_DERIVE)
    checklist = checklist_completion.content

    score_prompt = f"Checklist:\n{checklist}\n\nReply to score:\n{output}"
    score_completion = provider.complete([Message.user(score_prompt)], system=_CHECKLIST_SYSTEM_SCORE)
    text = score_completion.content

    match = _CHECKLIST_SCORE_RE.search(text)
    if match is None:
        return checklist, Verdict(score=None, passed=False, reasoning=text.strip(), raw=text, malformed=True)
    passed_items, total_items = int(match.group(1)), int(match.group(2))
    fraction = passed_items / total_items if total_items else 0.0
    return checklist, Verdict(score=fraction, passed=fraction > 0.5, reasoning=text.strip(), raw=text)


def run_pointwise_demo(provider: Provider | None = None) -> tuple[Verdict, Verdict]:
    """Score the same reply with reference-based and reference-free judging.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted
            with a strong reference-based verdict followed by a slightly
            more cautious reference-free verdict for the same reply, since
            without a gold answer to anchor against the judge has less to
            check the claim "30 days" against and hedges accordingly.
    """
    case = get_case("refund_policy")
    reply = "You can request a refund within 30 days of purchase if you have your order number."

    if provider is None:
        provider = get_provider(
            script=[
                "Step 1 accuracy: matches the reference (30 days, order number). "
                "Step 2 next step: implicit (contact support), not stated explicitly. "
                "Step 3 tone: concise and professional.\nSCORE: 8\nVERDICT: pass",
                "Step 1 accuracy: the 30-day window and order number cannot be "
                "verified without a reference, but the claim is specific and "
                "plausible for a support policy. Step 2 next step: not stated "
                "explicitly. Step 3 tone: concise and professional.\n"
                "SCORE: 7\nVERDICT: pass",
            ]
        )

    reference_based = build_pointwise_judge(provider, reference_mode=True)
    reference_free = build_pointwise_judge(provider, reference_mode=False)
    return reference_based(case, reply), reference_free(case, reply)


def run_checklist_demo(provider: Provider | None = None) -> tuple[str, Verdict]:
    """Run the checklist judge on a case with a scripted 2-of-3 result.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted
            with a 3-item checklist and a score of 2/3, since the reply
            below never states an explicit next step.
    """
    case = get_case("cancel_subscription")
    reply = "Go to Account Settings and select Subscription, then click Cancel."

    if provider is None:
        provider = get_provider(
            script=[
                "CHECKLIST:\n1. Names the exact menu path\n"
                "2. States when access actually ends\n3. Professional tone",
                "1. Names the exact menu path: PASS, gives Account Settings then "
                "Subscription then Cancel.\n2. States when access actually ends: "
                "FAIL, does not mention the billing period.\n"
                "3. Professional tone: PASS.\nCHECKLIST_SCORE: 2/3",
            ]
        )
    return run_checklist_judgment(provider, case, reply)


def run_pointwise_order_check_demo(provider: Provider | None = None) -> tuple[Verdict, Verdict, bool]:
    """Score one reply twice with the rubric's criteria listed in reverse order.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted so
            the forward order (accuracy, completeness, clarity) scores
            higher than the reversed order (clarity, completeness,
            accuracy), demonstrating that pointwise rubric judging is not
            immune to ordering effects (arXiv:2602.02219).

    Returns:
        The forward-order verdict, the reversed-order verdict, and whether
        their scores disagree by more than a small tolerance.
    """
    case = get_case("refund_policy")
    reply = "Refunds are available within 30 days if you have your order number."

    if provider is None:
        provider = get_provider(
            script=[
                "Order: accuracy, completeness, clarity. Accuracy matches the "
                "reference, completeness names the window and proof needed, "
                "clarity is short and direct.\nSCORE: 9\nVERDICT: pass",
                "Order: clarity, completeness, accuracy. Reviewed clarity first: "
                "reads a little terse on its own. Completeness and accuracy are "
                "fine on a second pass.\nSCORE: 7\nVERDICT: pass",
            ]
        )

    forward_prompt = f"Task:\n{case.input}\n\nReply:\n{reply}\n\nCriteria order: accuracy, completeness, clarity."
    reversed_prompt = f"Task:\n{case.input}\n\nReply:\n{reply}\n\nCriteria order: clarity, completeness, accuracy."

    forward = parse_pointwise_verdict(
        provider.complete([Message.user(forward_prompt)], system=_RUBRIC_SYSTEM).content
    )
    reversed_ = parse_pointwise_verdict(
        provider.complete([Message.user(reversed_prompt)], system=_RUBRIC_SYSTEM).content
    )
    bias_detected = (
        forward.score is not None and reversed_.score is not None and abs(forward.score - reversed_.score) >= 1.0
    )
    return forward, reversed_, bias_detected
