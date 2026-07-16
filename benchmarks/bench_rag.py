"""RAG retrieval quality: dense vs. hybrid vs. hybrid+rerank.

Measures the repo's real retrievers from `patterns/rag/` (`dense.py`,
`bm25.py`, `hybrid.py`, `rerank.py`) against a small, self-contained corpus
with hand-labeled gold chunks. Dense and hybrid retrieval only need
embeddings, which `gemini_embedder()` caches on disk, so a live run is
nearly free; only the `hybrid_rerank` variant spends a chat-model call, and
only one per question.

Corpus: 28 short passages documenting "Meridian", a fictional workflow
automation product, split across five topics (triggers, retries,
permissions, webhooks, billing). Eight of the passages are keyword-collision
distractors: they share salient vocabulary with a specific question but do
not answer it, so a purely lexical (BM25) match can outrank the real answer.
About half of the 30 questions are phrased with paraphrase or synonyms
instead of the gold passage's own wording, so lexical overlap with the gold
chunk is low and a bag-of-words match is not enough; this is the case dense
embeddings exist to handle, and where a keyword-colliding distractor is most
likely to fool BM25. Every question still has exactly one passage that
answers it.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.harness import (
    BenchProvider,
    BenchResult,
    CachedEmbedder,
    finalize,
    gemini_embedder,
    live_provider,
    mock_embedder,
    mock_provider,
)
from patterns.rag.bm25 import build_bm25_index
from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.dense import build_dense_index, dense_retrieve
from patterns.rag.hybrid import hybrid_retrieve
from patterns.rag.rerank import rerank_chunks

TOP_K = 3
FETCH_K = 8


@dataclass(frozen=True)
class Question:
    """One benchmark question with its gold chunk id.

    Attributes:
        id: Question identifier.
        text: The question text.
        gold_id: Id of the single `Chunk` that answers it.
    """

    id: str
    text: str
    gold_id: str


_PASSAGES: list[tuple[str, str]] = [
    ("triggers#0", "Meridian workflows start from a trigger. A schedule trigger fires on a cron "
                    "expression evaluated in UTC. A webhook trigger fires when Meridian receives a "
                    "POST to the workflow's unique inbound URL."),
    ("triggers#1", "A manual trigger only fires when a user clicks Run in the Meridian console or "
                    "calls the runWorkflow API. Manual triggers are the default for workflows still "
                    "in draft status."),
    ("triggers#2", "A file-watch trigger fires when a new object appears in a connected storage "
                    "bucket. Meridian polls the bucket every sixty seconds; near-real-time delivery "
                    "requires the webhook trigger instead."),
    ("triggers#3", "Meridian's schedule trigger accepts standard five-field cron syntax and an "
                    "optional seconds field for sub-minute cadences. Every schedule trigger a "
                    "workspace defines is listed on the workspace's Schedules page."),
    ("retries#0", "A failed step retries automatically up to three times with exponential backoff: "
                   "one second, four seconds, then sixteen seconds. After the third failure the step "
                   "is marked Failed and the workflow halts."),
    ("retries#1", "The retry count and backoff schedule can be overridden per step in the step's "
                   "Settings panel. Setting retries to zero disables automatic retry entirely for "
                   "that step."),
    ("retries#2", "A step that times out after 30 seconds counts as a failure for retry purposes. "
                   "The timeout itself is fixed platform-wide and cannot be raised above 30 seconds "
                   "on any plan."),
    ("retries#3", "A workflow-level retry, distinct from a per-step retry, reruns the entire "
                   "workflow from its trigger if the final step fails. Workflow-level retry is off "
                   "by default and must be enabled in the workflow's Advanced settings."),
    ("permissions#0", "Meridian workspaces have three roles: Owner, Editor, and Viewer. Only an "
                       "Owner can delete the workspace or transfer billing to another account."),
    ("permissions#1", "An Editor can create, edit, and run workflows but cannot invite new members "
                       "or change another member's role. Role changes require Owner access."),
    ("permissions#2", "A Viewer can see workflow definitions and run history but cannot trigger a "
                       "run or edit a step. Viewer access is meant for auditors and support staff."),
    ("permissions#3", "Inviting a new member sends an email with a join link valid for seven days. "
                       "An expired invite must be resent from the Members page; it cannot be "
                       "extended."),
    ("webhooks#0", "Every inbound webhook URL includes a signing secret. Meridian signs each request "
                    "with HMAC-SHA256 over the raw body and sends the signature in the "
                    "X-Meridian-Signature header."),
    ("webhooks#1", "An outbound webhook step retries a failed delivery up to five times over one "
                    "hour, then gives up and marks the step Failed. Delivery attempts beyond the "
                    "first are logged in the step's Attempts tab."),
    ("webhooks#2", "Webhook payloads are capped at 5 MB. A larger payload is rejected with a 413 "
                    "response before the workflow trigger evaluates it, so oversized payloads never "
                    "start a run."),
    ("webhooks#3", "The signing secret for an inbound webhook can be rotated from the trigger's "
                    "settings panel. Rotating the secret invalidates the old one immediately, so any "
                    "sender still using it will fail signature verification."),
    ("billing#0", "Meridian bills monthly based on completed workflow runs, not steps. A workflow "
                   "that fails on its first step still counts as one billed run."),
    ("billing#1", "Upgrading from Starter to Team mid-cycle prorates the price difference onto the "
                   "next invoice automatically; no manual credit request is needed."),
    ("billing#2", "The Starter plan caps a workspace at 1,000 runs per month. Runs beyond the cap "
                   "are queued and executed at the start of the next billing cycle rather than "
                   "rejected."),
    ("billing#3", "Invoice disputes must be filed within thirty days of the charge through the "
                   "billing tab. Meridian support responds to disputes within two business days."),
    # --- keyword-collision distractors: share vocabulary with a question,
    # but answer a different question than the one they're meant to tempt.
    ("triggers#4", "Downgrading a workspace to a lower plan is evaluated on a schedule, not "
                    "immediately: it takes effect at the start of the next billing cycle so a team "
                    "does not lose mid-cycle access."),
    ("retries#4", "A workflow can be configured to send a Slack notification after its third "
                   "consecutive failed run, so a team is not left checking the console for status "
                   "manually."),
    ("permissions#4", "Deleting a workspace is irreversible and removes every workflow, run history, "
                       "and stored credential after a seven-day grace period during which an Owner "
                       "can still cancel the deletion."),
    ("webhooks#4", "A workflow's run history page shows every trigger event, including manual runs "
                    "started from the console, with a timestamp and the member who started each "
                    "run."),
    ("billing#4", "A support ticket about a failed run is answered within two business days on the "
                   "Starter plan and within four business hours on the Team plan."),
    ("triggers#5", "The runWorkflow API accepts an optional idempotency key so a client retrying a "
                    "network timeout does not accidentally start the same workflow run twice."),
    ("retries#5", "A step's Attempts tab lists every delivery attempt for outbound requests the step "
                   "makes, including the HTTP status code and response body Meridian received back."),
    ("webhooks#5", "Meridian's inbound webhook endpoint responds with a 200 immediately on receipt "
                    "and evaluates the trigger asynchronously, so a slow downstream step never delays "
                    "the sender's response."),
]

CHUNKS: list[Chunk] = [
    Chunk(id=cid, source_id=cid.split("#")[0], text=text, start=0, end=len(text)) for cid, text in _PASSAGES
]

QUESTIONS: list[Question] = [
    # --- original wording, close lexical overlap with the gold passage ---
    Question("q00", "How often does Meridian check a storage bucket for new files?", "triggers#2"),
    Question("q01", "What HTTP method does a webhook trigger's inbound URL expect?", "triggers#0"),
    Question("q03", "How many times does a failed step retry automatically, and with what delays?", "retries#0"),
    Question("q06", "Which role is required to move billing to a different account?", "permissions#0"),
    Question("q09", "What header carries the signature on an inbound webhook request?", "webhooks#0"),
    Question("q11", "What happens if a webhook payload is larger than 5 MB?", "webhooks#2"),
    Question("q12", "If a workflow fails on its very first step, does that still count as a billed "
                     "run?", "billing#0"),
    Question("q14", "What happens to runs beyond the 1,000-run Starter cap?", "billing#2"),
    Question("q15", "How long do I have to file an invoice dispute?", "billing#3"),
    Question("q18", "What cron timezone does Meridian's schedule trigger use?", "triggers#0"),
    # --- paraphrased / synonym wording, low lexical overlap with the gold
    # passage; several are built to tempt lexical match toward a distractor.
    Question("q02", "If a workflow hasn't been published yet, what kind of trigger kicks it off "
                     "unless someone changes it?", "triggers#1"),
    Question("q04", "How do I turn off automatic retries for a single step?", "retries#1"),
    Question("q05", "Can I get more than half a minute before a slow step counts as unsuccessful, "
                     "even on a premium tier?", "retries#2"),
    Question("q07", "Is a workspace collaborator with edit rights allowed to bring on a new "
                     "teammate?", "permissions#1"),
    Question("q08", "What can a Viewer do in a Meridian workspace?", "permissions#2"),
    Question("q10", "How many times does Meridian retry an outbound webhook delivery, and over what "
                     "window?", "webhooks#1"),
    Question("q13", "What happens to the price when a workspace upgrades from Starter to Team mid "
                     "cycle?", "billing#1"),
    Question("q16", "Where is the HMAC signing secret used, and what algorithm signs the request?", "webhooks#0"),
    Question("q17", "Who can see a workflow's run history without being able to trigger a run?", "permissions#2"),
    Question("q19", "Does exceeding the outbound webhook retry window mark the step Failed?", "webhooks#1"),
    # --- new questions targeting the distractor passages and other
    # paraphrase / hard-negative cases ---
    Question("q20", "When does moving a workspace to a cheaper tier actually kick in?", "triggers#4"),
    Question("q21", "How does a team get pinged automatically after a workflow keeps failing?", "retries#4"),
    Question("q22", "After an Owner removes a workspace entirely, is there any window to change "
                     "their mind?", "permissions#4"),
    Question("q23", "Where can a member check who kicked off a run and when?", "webhooks#4"),
    Question("q25", "What keeps a flaky network retry from launching the same run a second time?", "triggers#5"),
    Question("q26", "Where would I find the response code Meridian got back for each outbound "
                     "delivery attempt?", "retries#5"),
    Question("q27", "Does the sender of an inbound webhook wait for the whole workflow to finish "
                     "before getting a response back?", "webhooks#5"),
    Question("q28", "How long is a new member's invitation link good for before it expires?", "permissions#3"),
    Question("q29", "Can Meridian's cron trigger fire more often than once a minute?", "triggers#3"),
    Question("q30", "Is there a way to invalidate an inbound webhook's secret without disabling the "
                     "trigger?", "webhooks#3"),
]


def _build_rerank_script() -> list[str]:
    """Script one `RANK:` reply per question, gold chunk first, rest in corpus order."""
    script = []
    for q in QUESTIONS:
        rest = [c.id for c in CHUNKS if c.id != q.gold_id][: FETCH_K - 1]
        script.append("RANK: " + ", ".join([q.gold_id, *rest]))
    return script


def _reciprocal_rank(ranked: list[ScoredChunk], gold_id: str) -> float:
    """1/rank of the gold id in a ranked list, 0.0 if absent."""
    for rank, scored in enumerate(ranked, start=1):
        if scored.chunk.id == gold_id:
            return 1.0 / rank
    return 0.0


def _hit_at_k(ranked: list[ScoredChunk], gold_id: str, k: int) -> bool:
    """Whether the gold id appears in the top-k prefix of a ranked list."""
    return any(scored.chunk.id == gold_id for scored in ranked[:k])


def _run(provider: BenchProvider, embedder: CachedEmbedder) -> BenchResult:
    """Core benchmark logic shared by `run_mock` and `run_live`.

    Builds dense and BM25 indexes over the fixed corpus, then for each
    question runs all three variants and scores hit@1, hit@3, and MRR
    against the gold chunk id.

    Args:
        provider: Chat provider, used only by the `hybrid_rerank` variant.
        embedder: Embedder for dense retrieval, real or mock.

    Returns:
        A `BenchResult` with per-variant hit@1 as the headline metric.
    """
    dense_index = build_dense_index(CHUNKS, embedder)
    bm25_index = build_bm25_index(CHUNKS)

    variant_names = ("dense", "hybrid", "hybrid_rerank")
    hits_at_1: dict[str, int] = dict.fromkeys(variant_names, 0)
    hits_at_3: dict[str, int] = dict.fromkeys(variant_names, 0)
    rr_sums: dict[str, float] = dict.fromkeys(variant_names, 0.0)
    tasks: list[dict[str, object]] = []

    for q in QUESTIONS:
        dense_ranked = dense_retrieve(q.text, dense_index, embedder, top_k=TOP_K)
        hybrid_ranked = hybrid_retrieve(q.text, dense_index, bm25_index, embedder, top_k=TOP_K, fetch_k=FETCH_K)
        rerank_candidates = hybrid_retrieve(
            q.text, dense_index, bm25_index, embedder, top_k=FETCH_K, fetch_k=FETCH_K
        )
        reranked = rerank_chunks(q.text, rerank_candidates, provider, top_k=TOP_K)

        outcomes = {
            "dense": [sc.chunk.id for sc in dense_ranked],
            "hybrid": [sc.chunk.id for sc in hybrid_ranked],
            "hybrid_rerank": [sc.chunk.id for sc in reranked],
        }
        for variant, ranked in (("dense", dense_ranked), ("hybrid", hybrid_ranked), ("hybrid_rerank", reranked)):
            if _hit_at_k(ranked, q.gold_id, 1):
                hits_at_1[variant] += 1
            if _hit_at_k(ranked, q.gold_id, 3):
                hits_at_3[variant] += 1
            rr_sums[variant] += _reciprocal_rank(ranked, q.gold_id)

        tasks.append(
            {
                "id": q.id,
                "question": q.text,
                "gold_id": q.gold_id,
                "outcomes": outcomes,
                "correct": {variant: q.gold_id in ids for variant, ids in outcomes.items()},
            }
        )

    n = len(QUESTIONS)
    variants = {name: round(hits_at_1[name] / n, 4) for name in variant_names}
    hit_at_3 = {name: round(hits_at_3[name] / n, 4) for name in variant_names}
    mrr = {name: round(rr_sums[name] / n, 4) for name in variant_names}
    best_variant = max(variants, key=lambda name: (variants[name], hit_at_3[name], mrr[name]))

    headline = (
        f"On {n} questions over a {len(CHUNKS)}-passage corpus (with keyword-collision "
        f"distractors and paraphrased questions), hit@1 was dense={variants['dense']:.2f}, "
        f"hybrid={variants['hybrid']:.2f}, hybrid_rerank={variants['hybrid_rerank']:.2f} "
        f"(best: {best_variant})."
    )

    return BenchResult(
        name="bench_rag",
        model=provider.model,
        n=n,
        variants=variants,
        headline=headline,
        detail={
            "hit@1": variants,
            "hit@3": hit_at_3,
            "mrr": mrr,
            "top_k": TOP_K,
            "fetch_k": FETCH_K,
            "embed_calls": embedder.calls,
        },
        tasks=tasks,
    )


def run_mock() -> BenchResult:
    """Run every variant against `mock_provider`/`mock_embedder` for a free smoke test."""
    embedder = mock_embedder()
    provider = mock_provider(_build_rerank_script())
    result = _run(provider, embedder)
    return finalize(result, provider)


def run_live() -> BenchResult:
    """Run every variant against real Gemini embeddings and the rerank chat model.

    Dense and hybrid retrieval spend nothing beyond the one-time embedding
    cost, cached by `CachedEmbedder`. Only `hybrid_rerank` spends chat-model
    tokens, one call per question.
    """
    embedder = gemini_embedder()
    provider = live_provider(model="gemini-3.1-flash-lite", budget_usd=0.5)
    result = _run(provider, embedder)
    return finalize(result, provider)


if __name__ == "__main__":
    res = run_mock()
    print(f"bench_rag: n={res.n} model={res.model} (mock)")
    for variant, score in res.variants.items():
        hit3 = res.detail["hit@3"][variant]
        mrr = res.detail["mrr"][variant]
        print(f"  {variant:<15} hit@1={score:.2f}  hit@3={hit3:.2f}  mrr={mrr:.2f}")
    print(f"  embed_calls={res.detail['embed_calls']}  cost=${res.usage.get('cost_usd', 0.0):.4f}")
    print(res.headline)
