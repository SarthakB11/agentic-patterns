"""An offline LongMemEval-style recall benchmark, with abstention scoring.

Every other module in this pattern builds a memory mechanism; none of them
scores one. LongMemEval (Wu, Wang, Yu, Zhang, Chang, Yu, ICLR 2025,
arXiv:2410.10813) fixes five abilities a memory system must get right:
information extraction, multi-session reasoning, temporal reasoning,
knowledge updates, and abstention. LoCoMo (Maharana, Lee, Tulyakov, Bansal,
Barbieri, Fang, arXiv:2402.17753) supplies the long multi-session dialogue
shape most later work scores against. This module borrows both: a small
fixed dataset of multi-session cases tagged by ability, a runner that
replays each case's sessions through a pluggable memory backend's write
path then asks its question, and a scorer that treats abstention
("I do not know") as the only correct answer when the fact was never
stored.

The MemDelta critique (arXiv:2606.29914) is the reason this module can mean
anything at all: MemDelta varies one component at a time on LongMemEval-S
and shows headline memory-method gains are routinely a base-model or
embedding-model confound rather than a real difference. Under
`MockProvider`, the reader step is held to the same scripted-answer
authoring discipline across every backend under test, so a score difference
between two backends can only come from what each backend's write path put
in the store, never from the reader model changing. See
`_reader_answer`'s docstring for exactly what "held fixed" means when the
model is a script, not a live weight.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider
from patterns.memory.mem0_update import apply_candidate_fact
from patterns.memory.retrieval import RetrievalConfig, retrieve
from patterns.memory.semantic import SemanticMemory
from patterns.memory.vector_store import VectorStore

Ability = Literal["extraction", "multi_session", "temporal", "knowledge_update", "abstention"]

ABSTAIN = "I do not know."

WriteFn = Callable[[Provider, VectorStore, Embedder, str, str], None]
AnswerFn = Callable[[Provider, VectorStore, Embedder, str, str], str]
JudgeFn = Callable[[Provider, str, str], bool]


@dataclass
class BenchSession:
    """One session's worth of raw turns to write into a backend."""

    turns: list[str]


@dataclass
class BenchCase:
    """One benchmark case: sessions to write, a question, and the gold answer.

    Attributes:
        case_id: Short identifier, used as the case's store namespace.
        sessions: Sessions to replay through the backend's write path, in order.
        question: The question asked after all sessions are written.
        gold_answer: The expected answer, or `ABSTAIN` if the fact was
            never stored and the correct behavior is to say so.
        ability: Which LongMemEval ability this case exercises.
    """

    case_id: str
    sessions: list[BenchSession]
    question: str
    gold_answer: str
    ability: Ability


@dataclass
class CaseResult:
    """One case's outcome: the answer produced and whether it was correct."""

    case_id: str
    ability: Ability
    answer: str
    correct: bool


@dataclass
class BenchReport:
    """Aggregate benchmark outcome.

    Attributes:
        results: Per-case results, in dataset order.
        accuracy: Overall fraction of cases scored correct.
        accuracy_by_ability: Accuracy broken out per ability tag.
    """

    results: list[CaseResult] = field(default_factory=list)
    accuracy: float = 0.0
    accuracy_by_ability: dict[str, float] = field(default_factory=dict)


def write_overwrite(provider: Provider, store: VectorStore, embedder: Embedder, namespace: str, turn: str) -> None:
    """Overwrite-by-key write path: a turn already shaped as "subject:
    value" is written straight into `SemanticMemory`, keyed by its literal
    subject string. Two turns whose subjects differ, even if they state the
    same underlying fact under different phrasing, both persist: this is
    the accumulation gap `semantic.write_fact`'s docstring now names.
    """
    memory = SemanticMemory(store, embedder, namespace)
    subject, _, value = turn.partition(":")
    memory.write_fact(subject.strip(), value.strip())


def write_naive_append(provider: Provider, store: VectorStore, embedder: Embedder, namespace: str, turn: str) -> None:
    """Naive baseline write path: always add a new record under a fresh id,
    never overwrite or merge. Every other write path in this pattern
    improves on this; it exists here so a knowledge-update case has a
    backend that fails it even when the overwrite backend, keyed on the
    same subject, passes.
    """
    new_id = f"raw-{len(store.all(namespace)) + 1}"
    store.upsert(new_id, namespace, turn, embedder.embed([turn])[0])


def write_mem0(provider: Provider, store: VectorStore, embedder: Embedder, namespace: str, turn: str) -> None:
    """Similarity-gated write path: each turn is decided against its
    nearest existing memories via `mem0_update.apply_candidate_fact`, so a
    later turn that restates an earlier fact under a new subject key can
    UPDATE the existing record instead of accumulating a second one.
    """
    apply_candidate_fact(provider, store, embedder, namespace, turn)


def _reader_answer(provider: Provider, context: str, question: str) -> str:
    """Call the reader once. Every backend's `answer_fn` in this module
    routes through this same function with the same system prompt: what
    "the reader model is held fixed" means for a scripted `MockProvider`
    is that every backend gets its answer through this one call shape, so
    the only thing that can differ between backends is `context`, which is
    a deterministic function of what each backend's write path actually
    stored, not of anything the script author chose per backend.
    """
    completion = provider.complete(
        [Message.user(f"Question: {question}\n\nMemory: {context}")],
        system=(
            "Answer the question using only the memory given, in one short "
            "sentence. If the memory does not contain the answer, reply "
            f"exactly: {ABSTAIN}"
        ),
    )
    return completion.content.strip()


def default_answer_fn(provider: Provider, store: VectorStore, embedder: Embedder, namespace: str, question: str) -> str:
    """Retrieve the top matching records for `question` and answer from
    them, regardless of which write path populated the store. Backend
    differences show up here as a different retrieved `context`, not as a
    different code path.
    """
    config = RetrievalConfig(top_k=3, min_similarity=0.1)
    hits = retrieve(store, embedder, namespace, question, config)
    context = "; ".join(h.record.text for h in hits) if hits else "(no relevant memory found)"
    return _reader_answer(provider, context, question)


def default_judge_fn(provider: Provider, answer: str, gold: str) -> bool:
    """Score one answer against the gold answer with one scripted judge call."""
    completion = provider.complete(
        [Message.user(f"Gold answer: {gold}\nGiven answer: {answer}")],
        system=(
            "Reply CORRECT if the given answer matches the gold answer in "
            "meaning, or if both express not knowing when the gold answer "
            "is 'I do not know.'. Otherwise reply WRONG."
        ),
    )
    return completion.content.strip().upper().startswith("CORRECT")


def run_bench(
    provider: Provider,
    embedder: Embedder,
    cases: list[BenchCase],
    write_fn: WriteFn,
    answer_fn: AnswerFn = default_answer_fn,
    judge_fn: JudgeFn = default_judge_fn,
    namespace_prefix: str = "bench",
) -> BenchReport:
    """Replay every case's sessions through `write_fn`, ask its question
    through `answer_fn`, score with `judge_fn`, and aggregate.

    Each case gets its own fresh, isolated namespace and store, so cases
    never leak memory into one another.
    """
    results: list[CaseResult] = []
    for case in cases:
        store = VectorStore()
        namespace = f"{namespace_prefix}:{case.case_id}"
        for session in case.sessions:
            for turn in session.turns:
                write_fn(provider, store, embedder, namespace, turn)
        answer = answer_fn(provider, store, embedder, namespace, case.question)
        correct = judge_fn(provider, answer, case.gold_answer)
        results.append(CaseResult(case.case_id, case.ability, answer, correct))

    accuracy = sum(r.correct for r in results) / len(results) if results else 0.0
    by_ability: dict[str, list[bool]] = {}
    for r in results:
        by_ability.setdefault(r.ability, []).append(r.correct)
    accuracy_by_ability = {ability: sum(flags) / len(flags) for ability, flags in by_ability.items()}
    return BenchReport(results=results, accuracy=accuracy, accuracy_by_ability=accuracy_by_ability)


def run_memory_bench_demo(provider: Provider | None = None) -> dict[str, object]:
    """Run a small mixed-ability dataset against the overwrite backend, then
    replay the one knowledge-update case that trips it up against the
    mem0-style backend to show the delta, with the reader held fixed.
    """
    cases = [
        BenchCase(
            case_id="extraction-1",
            sessions=[BenchSession(["favorite_language: Python"])],
            question="What is the user's favorite programming language?",
            gold_answer="Python",
            ability="extraction",
        ),
        BenchCase(
            case_id="multi-session-1",
            sessions=[
                BenchSession(["deployment_region: us-west-2"]),
                BenchSession(["iac_tool: Terraform"]),
            ],
            question="What region and IaC tool does the user's deployment use?",
            gold_answer="us-west-2 with Terraform",
            ability="multi_session",
        ),
        BenchCase(
            case_id="abstention-1",
            sessions=[BenchSession(["favorite_language: Python"])],
            question="What is the user's home address?",
            gold_answer=ABSTAIN,
            ability="abstention",
        ),
        BenchCase(
            case_id="knowledge-update-1",
            sessions=[
                BenchSession(["plan: pro tier, 1M requests/month"]),
                BenchSession(["subscription: free tier, 10k requests/month"]),
            ],
            question="What plan is the user currently on?",
            gold_answer="free tier, 10k requests/month",
            ability="knowledge_update",
        ),
    ]

    if provider is None:
        provider = get_provider(
            script=[
                "Python",  # extraction reader
                "CORRECT",  # extraction judge
                "us-west-2 with Terraform",  # multi-session reader
                "CORRECT",  # multi-session judge
                ABSTAIN,  # abstention reader: correctly declines
                "CORRECT",  # abstention judge
                # knowledge-update case on the overwrite backend: both the
                # pro-tier and free-tier records persisted under different
                # subject keys, so the reader sees a contradiction and
                # answers with the stale value.
                "pro tier, 1M requests/month",
                "WRONG",
            ]
        )
    embedder = get_embedder()
    overwrite_report = run_bench(provider, embedder, cases, write_fn=write_overwrite)

    # Replay only the knowledge-update case through the mem0-style backend,
    # same reader/judge call shape, to show the delta the overwrite
    # backend's accumulation bug caused.
    mem0_provider = get_provider(
        script=[
            "ADD",  # first session: empty namespace
            "UPDATE mem-1: plan: free tier, 10k requests/month",  # merges under the same id, keyword intact
            "free tier, 10k requests/month",  # reader: the single merged record answers cleanly
            "CORRECT",  # judge
        ]
    )
    mem0_report = run_bench(mem0_provider, embedder, [cases[3]], write_fn=write_mem0)

    return {
        "overwrite_accuracy": overwrite_report.accuracy,
        "overwrite_accuracy_by_ability": overwrite_report.accuracy_by_ability,
        "overwrite_knowledge_update_answer": overwrite_report.results[3].answer,
        "overwrite_knowledge_update_correct": overwrite_report.results[3].correct,
        "mem0_knowledge_update_answer": mem0_report.results[0].answer,
        "mem0_knowledge_update_correct": mem0_report.results[0].correct,
    }
