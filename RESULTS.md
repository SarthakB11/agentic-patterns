# Measured results

These numbers come from running the repo's own pattern code against a real
model: Google Gemini 3.1 Flash-Lite, reached through its OpenAI-compatible
endpoint with no change to `agentic_patterns` (the same `Provider` seam that
drives the offline mock). Each benchmark asks one question: does the technique
this repo implements change the outcome on a task with real ground truth.

What this is: a check that the implementations work against a live model and
that the upgrades have the effect the research predicts, on small purpose-built
task sets (N is stated per benchmark). What this is not: a leaderboard result
or a reproduction of any paper's headline number. Ground truth is stated,
every per-task outcome is recorded under `benchmarks/results/*.json`, and the
whole suite is reproducible with `GEMINI_KEY=... python3 -m benchmarks.run_all --live`.

Total cost to produce every number below: about **$0.035** on Gemini 3.1
Flash-Lite, on a cold run. Runs are cached under `benchmarks/.cache/`
(gitignored), so a re-run reproduces the same numbers for about a cent (only
the ReAct tool-loop, whose message sequence is not perfectly deterministic at
temperature 0, re-calls a few times).

## Summary

| Benchmark                          | N   | Result                                                   | What it shows                                                        |
| ---------------------------------- | --- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| [ReAct](patterns/react/)           | 15  | direct 0.07, react **0.67**, react+verify 0.60           | tool use turns a near-total failure into a majority-solved task      |
| [Routing](patterns/routing/)       | 24  | semantic **0.96**, classifier 0.88, always-strong 0.50   | a router matches most oracle choices at about half the cost          |
| [RAG](patterns/rag/)               | 30  | dense 0.93, hybrid 0.90, +rerank **1.00** (hit@1)        | reranking recovers every miss; naive fusion can hurt strong dense    |
| [Evaluation](patterns/evaluation/) | 20  | accuracy **0.95**, kappa 0.90, position-consistency 1.00 | an aligned judge agrees strongly with humans and shows no order bias |
| [Reflection](patterns/reflection/) | 14  | one-shot 1.00, reflection 1.00                           | ceiling: the model needed no correction on these tasks (see note)    |

## ReAct: reasoning and acting beats a single call

Fifteen two-hop questions over an invented knowledge base (fictional rivers,
countries, capitals), so the model cannot answer from memory and must chain a
`search` then a `lookup`. Ground truth is the gold answer string.

- direct (one completion, no tools): **0.067**
- react (the native tool-calling loop): **0.667**
- react + verify (verify-before-finish gate): **0.600**

The tool-using loop takes a task the model solves 1 time in 15 on its own and
solves it 10 times in 15. The verify gate did not help here and slightly hurt:
it spends an extra model call per finish attempt, so under a fixed iteration
budget it occasionally runs out on a task plain ReAct would have finished. That
is an honest cost of verification, not a free win, and it matches the research
caution that self-checking is not always worth its tokens.

## Routing: match the strong model's choices for half the cost

Twenty-four prompts hand-labeled as needing a cheap model or a strong one.
Metric is routing accuracy against those labels, plus the projected answer cost
of each router's tier choices (priced from the model rate card, not by actually
answering every task on the strong model).

- semantic router (embedding nearest-centroid): **0.958**
- LLM-classifier router: 0.875
- always-strong baseline: 0.500

A router that sends only the hard prompts to the strong model kept accuracy
near the oracle while cutting projected answer cost roughly in half against
sending everything to the strong model. This is the routing pattern's whole
argument, measured.

## RAG: reranking recovers the hard retrievals

A 28-passage corpus with keyword-collision distractors and 30 questions, half
of them paraphrased so the answer passage shares little vocabulary with the
question. Dense retrieval uses real Gemini embeddings. Metric is hit@1 (the
gold passage ranked first).

- dense: **0.933**
- hybrid (BM25 + dense, reciprocal rank fusion): 0.900
- hybrid + LLM rerank: **1.000**

Two honest findings. Reranking an over-fetched shortlist recovered every
first-rank miss and reached a perfect hit@1. And adding BM25 fusion to strong
dense embeddings slightly lowered hit@1 on this set: the lexical signal pulls
keyword-collision distractors up the list. Hybrid is not always better, and the
reranker is what earns the top result.

## Evaluation: a judge is only as good as the question you ask it

Twenty question-and-answer items, each hand-labeled good or bad, where a good
answer is correct and on-topic and a bad one is wrong or off-topic. Metrics are
the judge's accuracy against the human labels, chance-corrected agreement
(Cohen's kappa), and position consistency from a pairwise order swap.

- accuracy: **0.950**
- Cohen's kappa: **0.900**
- position consistency: 1.000

This benchmark also caught its own bug. The first version scored 0.60 accuracy
with kappa 0.20, and every error was the judge marking a correct answer bad. The
cause was a rubric mismatch: the pointwise judge was grading customer-support
polish and penalized a correct but terse answer ("Paris.") for offering no next
step, while the human labels graded correctness. Aligning the judge's criterion
with what the labels measure moved accuracy to 0.95 and kappa to 0.90. The
lesson is the evaluation pattern's own: a judge that measures the wrong thing
looks unreliable, and the fix is the question, not the model.

## Reflection: a ceiling result, reported as-is

Fourteen code-generation tasks graded by real unit tests run in an isolated
subprocess (edit distance, a wildcard matcher, a full valid-number validator,
atoi with 32-bit clamping, fraction-to-decimal with recurring digits, and
similar). The reflection variant feeds failing tests back as a critique and
refines, up to two rounds.

- one-shot: **1.000**
- reflection: **1.000**

Gemini 3.1 Flash-Lite solved all fourteen tasks on the first attempt, so the
reflection loop had nothing to correct. This is an honest ceiling, reported
rather than hidden: the loop demonstrably fixes failures when they occur (the
offline harness recovers a scripted 50 percent one-shot rate to 93 percent), but
this model on these tasks produced no first-attempt failures for it to work on.
Reflection's value shows on tasks a model gets wrong, and a capable model on
well-specified algorithmic tasks is not that case. Two rounds of honest task
hardening did not change this, and manufacturing failures by escalating
difficulty further would not have been an honest measurement.

## Honesty notes

- Task sets are small and purpose-built to isolate one variable each. They show
  direction and effect, not a population estimate.
- The invented knowledge base for ReAct and the distractor corpus for RAG are
  deliberate: they stop the model from answering from training memory, so the
  retrieval and tool-use effects are real rather than recalled.
- Every per-task outcome, including the raw judge and answer strings, is in
  `benchmarks/results/*.json` for inspection.
- Numbers were produced with Gemini 3.1 Flash-Lite at temperature 0. The
  provider layer is cached so the reported numbers reproduce exactly.
