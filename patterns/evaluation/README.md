# Evaluation

An evaluation loop turns "did this change make the system better or worse?" into a repeatable, automatable answer. It has three parts: a versioned eval set of input cases, one or more scorers that grade a candidate output per case (from exact programmatic checks to an LLM acting as a judge), and a regression gate that aggregates a run's scores and compares them to a baseline, returning pass or fail so CI can block a change that degrades quality.

## When to use it

Use an evaluation loop whenever you iterate on a prompt, model, retrieval setup, or agent and need to know if a change helped. Reach for an LLM judge specifically when the output is open-ended (summaries, chat replies, extractions) so string equality and regex are too brittle, yet human grading of every candidate is too slow. Skip the judge when a deterministic check suffices: a verifiable answer (valid JSON, correct arithmetic, a passing unit test) is cheaper and perfectly reliable to check directly. Avoid LLM judges for high-stakes safety or legal decisions without human sign-off, and be careful using one as an optimization target, since a system tuned against a flawed judge learns to please the judge rather than the user.

## How this example works

Every case in the eval set is scored by whichever evaluator fits it: an exact evaluator when the task has a verifiable answer, an LLM judge when it does not. Pairwise judging always runs both presentation orders and aggregates before trusting a winner. Per-case scores roll up into run metrics, which a regression gate checks against a stored baseline.

```mermaid
flowchart TD
    A[Eval case: input + optional reference] --> B{Exact check applies?}
    B -->|yes| C[Regex / JSON-schema evaluator]
    B -->|no| D{Pairwise judge?}
    D -->|yes| E[Run order A,B and order B,A]
    E --> F{Orderings agree?}
    F -->|yes| G[Winner = agreed candidate]
    F -->|no| H[Tie: position bias detected]
    D -->|no| I[Pointwise judge: reason, then SCORE / VERDICT]
    I --> J[Parse structured verdict, safe fallback if malformed]
    C --> K[Score]
    G --> K
    H --> K
    J --> K
    K --> L[Aggregate: mean score, pass rate, win rate]
    L --> M{Within tolerance of baseline?}
    M -->|yes| N[Regression gate: pass, exit 0]
    M -->|no| O[Regression gate: fail, exit 1]
```

## Variants implemented

- `eval_set.py`: the eval set as versioned data, five cases spanning exact-checkable and open-ended tasks for one running scenario, a subscription-product support bot.
- `exact.py`: two exact evaluators, regex/string match and JSON-schema validity, fully offline and model-free.
- `semantic.py`: a semantic-similarity evaluator using `HashEmbedder` and cosine distance against a reference, catching paraphrases exact match misses.
- `verdict.py`: shared structured-verdict parsing (`SCORE`/`VERDICT` and `WINNER` lines) with a safe, fail-closed fallback for malformed judge output, reused by every judge module.
- `pointwise.py`: a pointwise rubric judge that reasons through named evaluation steps before scoring (G-Eval style), reference-based and reference-free modes on the same judge, an instruction-specific checklist judge that derives criteria per case, and a position-order check showing pointwise scoring is not immune to ordering effects either.
- `pairwise.py`: a pairwise comparison judge run in both presentation orders and aggregated, so a candidate only wins if preferred in both orders and a disagreement falls back to a tie instead of trusting either single call.
- `ensemble.py`: a jury of three independent judges combined by majority vote.
- `trajectory.py`: agent-as-judge trajectory evaluation, grading an agent's whole action/observation trace against its goal, contrasted directly against final-answer-only judging on the same shortcut trajectory.
- `aggregate.py`: pure metrics functions, mean score, pass rate, pairwise win rate, and an Elo-style ranking rolled up from pairwise verdicts (the same idea behind the LMSYS Chatbot Arena leaderboard, using an Elo update rather than a full Bradley-Terry fit for simplicity).
- `regression.py`: a regression gate comparing a run's aggregate metric to a stored baseline within a tolerance band, with the non-zero CI exit code a real pipeline would check.
- `meta.py`: meta-evaluation, Cohen's kappa for chance-corrected judge-vs-human agreement, and a test-retest same-verdict rate for judge stability across repeated runs.

Not implemented: a full Bradley-Terry maximum-likelihood ranking. `aggregate.py` uses an Elo update instead, which serves the same purpose (turning pairwise verdicts into one global ranking) with far less code, at the cost of being order-sensitive to the sequence of matches, a fine tradeoff for a teaching example.

## Run it

```
python -m patterns.evaluation.main
```

Expected output (truncated):

```
EVALUATION PATTERN: eval set, scorers, regression gate

=== 1. Eval set (version 2026.07.1) ===
  [refund_policy] tags=['billing', 'open_ended'] reference=yes expected_property=-
  ...
=== 5. Pairwise judge (both orderings, position-bias cancellation) ===
fair comparison: winner=candidate_b position_bias_detected=False
biased comparison: winner=tie position_bias_detected=True
...
=== 9. Regression gate ===
candidate run: metric=1.00 baseline=1.00 passed=True exit_code=0
regressed run: metric=0.50 baseline=1.00 passed=False exit_code=1
All ten sections completed without exhausting their scripts.
```

## Real providers

Set `AGENTIC_PATTERNS_PROVIDER=openai` (with `OPENAI_API_KEY` set) or `AGENTIC_PATTERNS_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY` set) to run every judge in this pattern against a real model, with no source change: each demo function builds its provider through `agentic_patterns.get_provider`. Set `AGENTIC_PATTERNS_EMBEDDER=openai` (with `OPENAI_API_KEY` set) to run `semantic.py` against real embeddings instead of the deterministic `HashEmbedder`.

## Sources

- Chip Huyen, _AI Engineering_ (O'Reilly, 2025), Ch. 3 "Evaluation Methodology" and Ch. 4 "Evaluate AI Systems".
- Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment," EMNLP 2023. arXiv:2303.16634.
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023. arXiv:2306.05685.
- "A Survey on LLM-as-a-Judge," arXiv:2411.15594.
- Krumdick et al., "Reliability without Validity," arXiv:2606.19544: the consistency-bias paradox, kappa-deflation versus raw agreement, and temperature-dependent same-verdict rates.
- Zhuge et al., "Agent-as-a-Judge," arXiv:2410.10934: whole-trajectory grading against final-answer-only judging.
