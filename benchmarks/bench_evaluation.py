"""Benchmark: how reliable is an LLM-as-judge against human labels.

Reuses the pairwise judge (`patterns/evaluation/pairwise.py`) from the
evaluation pattern, plus `cohens_kappa` from `meta.py`. Twenty question/answer
pairs are hand-labeled "good" or "bad" by a human (this file's author); the
good and bad answers are built to be unambiguous (correct and on-topic versus
factually wrong or non-responsive) so the gold label is defensible.

The pointwise judge here is built locally (not via
`patterns.evaluation.pointwise.build_pointwise_judge`) because that builder
is hardcoded to a support-desk rubric ("names a concrete next step",
"professional tone") shared by `trajectory.py`, `ensemble.py`, and
`selective.py`. The human labels in this file measure something narrower:
is the answer factually correct and does it address what was asked. Grading
those labels with a support-polish rubric is a metric-construction mismatch,
not a judge-reliability finding: it previously produced false negatives on
short, correct answers penalized for lacking a "next step" (e.g. "Paris."
marked failing for a capital-of-France question). `_CORRECTNESS_SYSTEM`
below asks the judge to grade exactly what the labels grade, reusing the
same `parse_pointwise_verdict` engine every judge variant in this pattern
shares, so the SCORE/VERDICT contract stays identical.

Three variants are reported:

- `pointwise_accuracy`: judge each of the 20 answers good/bad, accuracy
  against the human label. This is the headline number.
- `cohens_kappa`: chance-corrected agreement between the judge's pass/fail
  calls and the human labels, over the same 20 items (see `meta.py` for why
  raw agreement overstates a judge that leans toward one verdict).
- `position_consistency`: over 5 pairs (one clearly-better answer vs one
  clearly-worse answer to the same question, 10 items total), the fraction
  of pairs where the pairwise judge picks the same underlying winner in both
  the (A, B) and (B, A) presentation order. 1.0 means no detectable position
  bias on this slice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import Message, Provider
from benchmarks.harness import BenchResult, finalize, live_provider, mock_provider
from patterns.evaluation.eval_set import EvalCase
from patterns.evaluation.meta import cohens_kappa
from patterns.evaluation.pairwise import run_pairwise_judgment
from patterns.evaluation.verdict import Verdict, parse_pointwise_verdict

MODEL = "gemini-3.1-flash-lite"
BUDGET_USD = 0.5

# Grades exactly what the human labels in this file grade: is the answer
# factually correct and does it address what was asked. Deliberately does
# not check for a "next step", tone, or support-desk polish, since a terse
# but fully correct and on-topic answer (e.g. "Paris." for "What is the
# capital of France?") must pass under these labels.
_CORRECTNESS_SYSTEM = (
    "You grade whether a reply correctly and relevantly answers a question. "
    "Evaluation steps: (1) check the reply is factually accurate, (2) check "
    "the reply is directly responsive to what was asked, on-topic and not "
    "evasive. Ignore tone, phrasing style, length, and whether a next step "
    "or extra context is offered; a short, correct, on-topic reply is a "
    "pass. Reason through each step briefly, then end with a SCORE line "
    "(0-10) and a VERDICT line (pass if SCORE >= 7, else fail)."
)


def _build_correctness_judge(provider: Provider) -> Callable[[EvalCase, str], Verdict]:
    """Build a pointwise judge that grades correctness and relevance only.

    Unlike `patterns.evaluation.pointwise.build_pointwise_judge`, which
    grades a fixed support-desk rubric (accuracy, presence of a next step,
    tone), this judge grades only whether the reply is factually correct
    and responsive to the question, matching what the human labels in this
    benchmark encode.

    Args:
        provider: The model that plays the judge.

    Returns:
        A callable taking an `EvalCase` and a candidate answer, returning a
        `Verdict` parsed with the same `parse_pointwise_verdict` engine
        every judge variant in this pattern shares.
    """

    def judge(case: EvalCase, output: str) -> Verdict:
        prompt = f"Question:\n{case.input}\n\nAnswer to grade:\n{output}"
        completion = provider.complete([Message.user(prompt)], system=_CORRECTNESS_SYSTEM)
        return parse_pointwise_verdict(completion.content)

    return judge


@dataclass(frozen=True)
class JudgeItem:
    """One human-labeled question/answer pair for the pointwise judge.

    Attributes:
        id: Stable identifier for the item.
        question: The question posed.
        answer: A candidate answer to that question.
        human_label: True if a human would call this answer good (correct
            and responsive), False if bad (factually wrong or
            non-responsive).
    """

    id: str
    question: str
    answer: str
    human_label: bool


# 20 items, 10 good and 10 bad, authored so the label is unambiguous: a good
# answer is factually correct and directly answers the question; a bad
# answer is either factually wrong or does not address what was asked.
JUDGE_ITEMS: list[JudgeItem] = [
    JudgeItem("q01", "What is your refund policy?",
              "Refunds are available within 30 days of purchase with your order number.", True),
    JudgeItem("q02", "What is your refund policy?",
              "We are open Monday through Friday, 9am to 5pm.", False),
    JudgeItem("q03", "How do I cancel my subscription?",
              "Go to Account Settings, select Subscription, and click Cancel.", True),
    JudgeItem("q04", "How do I cancel my subscription?",
              "Subscriptions cannot be cancelled once purchased.", False),
    JudgeItem("q05", "What is the capital of France?", "Paris.", True),
    JudgeItem("q06", "What is the capital of France?", "Lyon.", False),
    JudgeItem("q07", "How many days are in a leap year?", "366 days.", True),
    JudgeItem("q08", "How many days are in a leap year?", "365 days.", False),
    JudgeItem("q09", "What is the boiling point of water at sea level in Celsius?", "100 degrees Celsius.", True),
    JudgeItem("q10", "What is the boiling point of water at sea level in Celsius?", "212 degrees Celsius.", False),
    JudgeItem("q11", "What is the status of order 48213?", "Order 48213 shipped on July 2.", True),
    JudgeItem("q12", "What is the status of order 48213?", "I don't have any information about that.", False),
    JudgeItem("q13", "How do I reset my password?",
              "Click 'Forgot password' on the login page and follow the emailed link.", True),
    JudgeItem("q14", "How do I reset my password?", "Try turning your computer off and on again.", False),
    JudgeItem("q15", "What payment methods do you accept?", "We accept Visa, Mastercard, and PayPal.", True),
    JudgeItem("q16", "What payment methods do you accept?", "We only accept cash delivered in person.", False),
    JudgeItem("q17", "What is 12 multiplied by 12?", "144.", True),
    JudgeItem("q18", "What is 12 multiplied by 12?", "121.", False),
    JudgeItem("q19", "Who wrote the play Hamlet?", "William Shakespeare.", True),
    JudgeItem("q20", "Who wrote the play Hamlet?", "Charles Dickens.", False),
]

# 5 pairs (10 of the items above), each pair sharing a question with one
# clearly-better and one clearly-worse answer, used for the position-bias
# check. Order is (question_id, better_item_id, worse_item_id).
PAIRWISE_ITEMS: list[tuple[str, str, str]] = [
    ("refund_policy", "q01", "q02"),
    ("cancel_subscription", "q03", "q04"),
    ("capital_of_france", "q05", "q06"),
    ("leap_year_days", "q07", "q08"),
    ("boiling_point", "q09", "q10"),
]

_ITEMS_BY_ID = {item.id: item for item in JUDGE_ITEMS}


def _run(provider: Provider) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    """Core logic shared by `run_mock` and `run_live`.

    Runs the pointwise judge over all 20 items, computes accuracy and
    Cohen's kappa against the human labels, then runs the pairwise judge
    over the 5 better/worse pairs in both presentation orders and computes
    position consistency.

    Returns:
        A tuple of (variants, detail, tasks) ready to drop into a
        `BenchResult`.
    """
    judge = _build_correctness_judge(provider)
    tasks: list[dict[str, object]] = []
    judge_labels: list[bool] = []
    human_labels: list[bool] = []
    correct = 0

    for item in JUDGE_ITEMS:
        case = EvalCase(id=item.id, input=item.question)
        verdict = judge(case, item.answer)
        judge_pass = bool(verdict.passed) if verdict.passed is not None else False
        is_correct = judge_pass == item.human_label
        correct += int(is_correct)
        judge_labels.append(judge_pass)
        human_labels.append(item.human_label)
        tasks.append(
            {
                "id": item.id,
                "variant": "pointwise",
                "question": item.question,
                "human_label": item.human_label,
                "judge_label": judge_pass,
                "correct": is_correct,
            }
        )

    pointwise_accuracy = correct / len(JUDGE_ITEMS)
    kappa = cohens_kappa(judge_labels, human_labels)

    consistent_pairs = 0
    for question_id, better_id, worse_id in PAIRWISE_ITEMS:
        better = _ITEMS_BY_ID[better_id]
        worse = _ITEMS_BY_ID[worse_id]
        case = EvalCase(id=question_id, input=better.question)
        result = run_pairwise_judgment(provider, case, better.answer, worse.answer)
        # candidate_a is the better answer here, so a bias-free judge should
        # land on "candidate_a" in both orders.
        is_consistent = not result.position_bias_detected
        consistent_pairs += int(is_consistent)
        tasks.append(
            {
                "id": question_id,
                "variant": "pairwise",
                "order_ab_winner": result.order_ab.winner,
                "order_ba_winner": result.order_ba.winner,
                "aggregated_winner": result.winner,
                "consistent": is_consistent,
            }
        )

    position_consistency = consistent_pairs / len(PAIRWISE_ITEMS)

    variants = {
        "pointwise_accuracy": pointwise_accuracy,
        "cohens_kappa": kappa,
        "position_consistency": position_consistency,
    }
    detail = {
        "n_pointwise_items": len(JUDGE_ITEMS),
        "n_pairwise_pairs": len(PAIRWISE_ITEMS),
        "judge_pass_rate": sum(judge_labels) / len(judge_labels),
        "human_pass_rate": sum(human_labels) / len(human_labels),
        "correct_count": correct,
        "consistent_pairs": consistent_pairs,
    }
    return variants, detail, tasks


def run_mock() -> BenchResult:
    """Run the judge-reliability benchmark against scripted mock verdicts.

    The pointwise script agrees with the human label on 17 of 20 items (3
    scripted disagreements so accuracy and kappa are both non-trivial and
    below 1.0), and the pairwise script agrees across both orderings for 4
    of 5 pairs with one scripted position flip on the fifth, so
    `position_consistency` lands at 0.8 instead of trivially 1.0. This
    proves the plumbing end to end for free.
    """
    script: list[str] = []
    # 3 scripted misses: q02 (bad, judge wrongly says pass), q08 (bad, judge
    # wrongly says pass), q18 (bad, judge wrongly says pass). Everything
    # else matches the human label.
    wrong_pass_ids = {"q02", "q08", "q18"}
    for item in JUDGE_ITEMS:
        if item.id in wrong_pass_ids:
            script.append("Step 1: looks plausible on a skim.\nSCORE: 7\nVERDICT: pass")
        elif item.human_label:
            script.append("Step 1: accurate and responsive.\nSCORE: 9\nVERDICT: pass")
        else:
            script.append("Step 1: inaccurate or non-responsive.\nSCORE: 2\nVERDICT: fail")

    # Pairwise: 4 pairs agree in both orders (candidate_a, the better
    # answer, wins both times), the 5th (boiling_point) flips depending on
    # slot to model a real position-biased call.
    for question_id, _better_id, _worse_id in PAIRWISE_ITEMS:
        if question_id == "boiling_point":
            script.append("Candidate in slot A is more precise.\nWINNER: a")
            script.append("Candidate in slot A is more precise.\nWINNER: a")
        else:
            script.append("Candidate A is accurate and on-topic; Candidate B is not.\nWINNER: a")
            script.append("Candidate B is accurate and on-topic; Candidate A is not.\nWINNER: b")

    provider = mock_provider(script, model=MODEL)
    variants, detail, tasks = _run(provider)
    result = BenchResult(
        name="bench_evaluation",
        model=MODEL,
        n=len(JUDGE_ITEMS),
        variants=variants,
        headline=(
            f"[mock] pointwise judge accuracy {variants['pointwise_accuracy']:.2f}, "
            f"kappa {variants['cohens_kappa']:.2f}, "
            f"position consistency {variants['position_consistency']:.2f} (plumbing check, not a real finding)"
        ),
        detail=detail,
        tasks=tasks,
    )
    return finalize(result, provider)


def run_live() -> BenchResult:
    """Run the same benchmark against a live, budgeted Gemini call.

    Uses `harness.live_provider` at a hard $0.50 ceiling. Each pointwise
    item costs one short judge call; each pairwise pair costs two, so the
    full run is 20 + 10 = 30 short completions, well inside the cap.
    """
    provider = live_provider(model=MODEL, budget_usd=BUDGET_USD)
    variants, detail, tasks = _run(provider)
    result = BenchResult(
        name="bench_evaluation",
        model=MODEL,
        n=len(JUDGE_ITEMS),
        variants=variants,
        headline=(
            f"LLM judge ({MODEL}) scored {variants['pointwise_accuracy']:.0%} pointwise accuracy "
            f"(kappa {variants['cohens_kappa']:.2f}) against {len(JUDGE_ITEMS)} human labels, "
            f"with {variants['position_consistency']:.0%} position consistency over "
            f"{len(PAIRWISE_ITEMS)} pairwise comparisons."
        ),
        detail=detail,
        tasks=tasks,
    )
    return finalize(result, provider)


if __name__ == "__main__":
    outcome = run_mock()
    print(f"bench_evaluation ({outcome.model}, n={outcome.n})")
    for variant_name, value in outcome.variants.items():
        print(f"  {variant_name}: {value:.3f}")
    print(f"  cost: ${outcome.usage.get('cost_usd', 0.0):.4f}")
    print(outcome.headline)
