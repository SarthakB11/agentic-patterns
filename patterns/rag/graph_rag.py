"""GraphRAG-lite: retrieval that follows structure across chunks, instead of
ranking each chunk independently against a query.

GraphRAG (Edge et al., "From Local to Global," arXiv:2404.16130, revised
February 2025) builds an entity graph over a corpus and pregenerates a
summary per community of related entities. Local search answers an
entity-focused question from that entity's neighborhood, reaching a fact
that shares no vocabulary with the query but sits one hop away through a
shared entity. Global search answers a corpus-level "what are the themes"
question by map-reduce over every community summary, with no single chunk
holding the answer. LightRAG (Guo et al., arXiv:2410.05779) and HippoRAG 2
(Gutierrez et al., arXiv:2502.14802) refine the same idea with dual-level
retrieval and a Personalized PageRank scorer; this module keeps to connected
components for communities and one-to-two-hop traversal, and names PageRank
as the production scorer rather than implementing it.

A production graph store and a learned entity extractor are out of scope,
not the graph mechanism itself. Entities are extracted per chunk (a
scripted-LLM call for the demo, a deterministic capitalized-noun-phrase
heuristic with zero model calls for tests), edges connect entities that
co-occur in a chunk and carry that chunk's id as evidence, and communities
are connected components, not a learned clustering: small enough to build
and traverse in pure Python and test deterministically under `MockProvider`.

The skeptical 2025 finding matters as much as the mechanism: "When to use
Graphs in RAG" (Xiang et al., arXiv:2506.05690) finds graph methods
frequently underperform vanilla RAG, and that they earn their cost only on
multi-hop and global-summary questions a flat retriever cannot serve. The
demo below runs a question the graph helps with next to one it does not, so
the trade-off is visible, not asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve
from patterns.rag.generation import ABSTAIN_ANSWER, GroundedAnswer, extract_citations, generate_grounded_answer

_ENTITY_RUN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:[-\s][A-Z][a-zA-Z0-9]*)*\b")
_ENTITY_STOPWORDS = {
    "the", "a", "an", "if", "it", "this", "that", "every", "trading", "customers",
    "application", "database", "deployment", "primary", "refund", "invoice", "faq",
}

_ENTITY_EXTRACTION_SYSTEM = (
    "For each numbered chunk below, list the distinct named entities or concepts it introduces "
    "as a short comma-separated list, skipping generic company-name mentions. Reply with one line "
    "per chunk in the form chunk_id: entity, entity. If a chunk introduces nothing worth tracking, "
    "reply with chunk_id: (none)."
)

_COMMUNITY_SUMMARY_SYSTEM = (
    "Summarize the shared theme of the chunks below in one or two sentences, naming the entities "
    "and what they describe together."
)

_GLOBAL_MAP_SYSTEM = (
    "Given the question and one community's summary, write a short partial answer covering only "
    "what this community's theme contributes. If this community is not relevant to the question, "
    "reply with exactly: NOT RELEVANT."
)

_GLOBAL_REDUCE_SYSTEM = (
    "Combine the partial answers below into one coherent answer to the original question, citing "
    "the community each point came from in square brackets, like [community-2]."
)

_LOCAL_DEMO_QUERY = (
    "If a SEV1 is caused by a bad deploy, what mitigation step is taken, and are deployment "
    "freezes also in effect during the incident?"
)
_GLOBAL_DEMO_QUERY = "What are the main operational themes covered in Aurora's internal documentation?"
_SKEPTIC_DEMO_QUERY = "What is Aurora's default API rate limit per minute?"


def extract_entities_heuristic(text: str) -> list[str]:
    """Extract capitalized noun-phrase entities from text with no model call.

    A run of consecutive capitalized words is a candidate entity. A leading
    sentence-initial capital ("A", "If", "It", and similar) is dropped from
    the phrase rather than excluded outright, so "A SEV1" yields "SEV1"
    instead of nothing. Teaching-scale: it captures proper nouns and
    acronyms well enough to build a graph with zero model calls, not a
    substitute for a trained named-entity recognizer.

    Args:
        text: Text to scan, typically one chunk.

    Returns:
        Distinct entity phrases, in the order they first appear.
    """
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RUN_RE.finditer(text):
        tokens = match.group().split()
        if len(tokens) == 1 and tokens[0].lower() in _ENTITY_STOPWORDS:
            continue
        if tokens and tokens[0].lower() in _ENTITY_STOPWORDS:
            tokens = tokens[1:]
        phrase = " ".join(tokens)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        entities.append(phrase)
    return entities


def build_entities_by_chunk_heuristic(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Run `extract_entities_heuristic` over every chunk, with no model call."""
    return {chunk.id: extract_entities_heuristic(chunk.text) for chunk in chunks}


def build_entity_extraction_prompt(chunks: list[Chunk]) -> str:
    """Build the prompt asking the model to extract entities for every chunk at once."""
    return "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)


def parse_entity_extraction(text: str) -> dict[str, list[str]]:
    """Parse a `chunk_id: entity, entity` reply, one line per chunk, into a mapping."""
    entities_by_chunk: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        chunk_id, rest = stripped.split(":", 1)
        chunk_id = chunk_id.strip()
        rest = rest.strip()
        if not chunk_id:
            continue
        if rest.lower() in {"(none)", "none"}:
            entities_by_chunk[chunk_id] = []
        else:
            entities_by_chunk[chunk_id] = [part.strip() for part in rest.split(",") if part.strip()]
    return entities_by_chunk


def extract_entities_llm(chunks: list[Chunk], provider: Provider) -> dict[str, list[str]]:
    """Ask the model to extract entities for every chunk in one batched call.

    Args:
        chunks: Chunks to extract entities from.
        provider: `Provider` used for the extraction call.

    Returns:
        A mapping from chunk id to the entities the model listed for it.
    """
    completion = provider.complete([Message.user(build_entity_extraction_prompt(chunks))], system=_ENTITY_EXTRACTION_SYSTEM)
    return parse_entity_extraction(completion.content)


@dataclass
class GraphEdge:
    """A co-occurrence edge between two entities (unordered; source/target is
    just storage order), weighted by co-occurrence count and evidenced by
    the chunk ids that established it."""

    source: str
    target: str
    weight: int = 0
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """An entity co-occurrence graph built over a chunked corpus.

    Attributes:
        entities: Every distinct entity, in first-appearance order.
        edges: Every co-occurrence edge, one per distinct entity pair.
        adjacency: Entity name to the edges touching it.
        entities_by_chunk: Chunk id to the entities extracted from it.
        chunks_by_id: Chunk id to the `Chunk` itself.
    """

    entities: list[str]
    edges: list[GraphEdge]
    adjacency: dict[str, list[GraphEdge]]
    entities_by_chunk: dict[str, list[str]]
    chunks_by_id: dict[str, Chunk]


def build_graph(chunks: list[Chunk], entities_by_chunk: dict[str, list[str]]) -> KnowledgeGraph:
    """Build a co-occurrence graph from per-chunk entity lists.

    Two entities in the same chunk get an edge, or an existing edge's weight
    increments and gains that chunk as further evidence. Pure Python over
    fixed inputs, so the same chunks and entity lists always produce the
    same graph.

    Args:
        chunks: Chunks in a fixed order, fixing entity and edge ordering.
        entities_by_chunk: Chunk id to entities, from either
            `build_entities_by_chunk_heuristic` or `extract_entities_llm`.

    Returns:
        The resulting `KnowledgeGraph`.
    """
    entities: list[str] = []
    seen_entities: set[str] = set()
    edge_by_pair: dict[tuple[str, str], GraphEdge] = {}
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    for chunk in chunks:
        chunk_entities = entities_by_chunk.get(chunk.id, [])
        for entity in chunk_entities:
            if entity not in seen_entities:
                seen_entities.add(entity)
                entities.append(entity)
        for i in range(len(chunk_entities)):
            for j in range(i + 1, len(chunk_entities)):
                pair = tuple(sorted((chunk_entities[i], chunk_entities[j])))
                edge = edge_by_pair.get(pair)
                if edge is None:
                    edge = GraphEdge(source=pair[0], target=pair[1])
                    edge_by_pair[pair] = edge
                edge.weight += 1
                edge.chunk_ids.append(chunk.id)

    edges = list(edge_by_pair.values())
    adjacency: dict[str, list[GraphEdge]] = {entity: [] for entity in entities}
    for edge in edges:
        adjacency[edge.source].append(edge)
        adjacency[edge.target].append(edge)

    return KnowledgeGraph(
        entities=entities,
        edges=edges,
        adjacency=adjacency,
        entities_by_chunk={cid: list(ents) for cid, ents in entities_by_chunk.items()},
        chunks_by_id=chunks_by_id,
    )


@dataclass
class Community:
    """A connected component of the entity graph.

    Attributes:
        id: Deterministic id, by first-appearance rank of the community's earliest entity.
        entities: Entities belonging to this community.
        chunk_ids: Every chunk whose entities fall in this community,
            including chunks that contributed no edge (mentioning only one
            of the community's entities).
        summary: The pregenerated summary, empty until `summarize_communities` runs.
    """

    id: int
    entities: set[str]
    chunk_ids: set[str]
    summary: str = ""


def detect_communities(graph: KnowledgeGraph) -> list[Community]:
    """Detect communities as connected components, via union-find over the
    graph's edges: two entities sharing an edge, directly or transitively,
    land in the same community, and an entity with no edges forms its own
    singleton. Ids are assigned by the minimum first-appearance rank among
    each community's entities, so the same graph always yields the same ids.

    Args:
        graph: The `KnowledgeGraph` to partition.

    Returns:
        Communities in id order.
    """
    parent: dict[str, str] = {entity: entity for entity in graph.entities}

    def find(entity: str) -> str:
        while parent[entity] != entity:
            parent[entity] = parent[parent[entity]]
            entity = parent[entity]
        return entity

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in graph.edges:
        union(edge.source, edge.target)

    rank = {entity: i for i, entity in enumerate(graph.entities)}
    groups: dict[str, list[str]] = {}
    for entity in graph.entities:
        groups.setdefault(find(entity), []).append(entity)

    ordered_roots = sorted(groups, key=lambda root: min(rank[e] for e in groups[root]))
    communities: list[Community] = []
    for community_id, root in enumerate(ordered_roots):
        entities = set(groups[root])
        chunk_ids = {cid for cid, ents in graph.entities_by_chunk.items() if set(ents) & entities}
        communities.append(Community(id=community_id, entities=entities, chunk_ids=chunk_ids))
    return communities


def build_community_summary_prompt(community: Community, chunks_by_id: dict[str, Chunk]) -> str:
    """Build the prompt asking the model to summarize one community's chunks."""
    chunk_lines = "\n\n".join(f"[{cid}] {chunks_by_id[cid].text}" for cid in sorted(community.chunk_ids))
    return f"Entities: {', '.join(sorted(community.entities))}\n\nChunks:\n{chunk_lines}"


def summarize_communities(communities: list[Community], chunks_by_id: dict[str, Chunk], provider: Provider) -> None:
    """Pregenerate and cache a one-call summary for every community, in place.

    Args:
        communities: Communities to summarize; each gets its `summary` field set.
        chunks_by_id: All chunks in the corpus, keyed by id.
        provider: `Provider` used for one call per community.
    """
    for community in communities:
        completion = provider.complete(
            [Message.user(build_community_summary_prompt(community, chunks_by_id))], system=_COMMUNITY_SUMMARY_SYSTEM
        )
        community.summary = completion.content.strip()


@dataclass
class GraphSearchResult:
    """The outcome of one local or global graph search.

    Attributes:
        mode: "local" or "global".
        entities_touched: Entities seeded on or traversed to (local only).
        communities_touched: Community ids consulted (global only).
        chunk_ids: Chunks the answer was grounded in (local only; global
            answers from summaries, not chunks).
        answer: The grounded answer.
    """

    mode: str
    entities_touched: list[str] = field(default_factory=list)
    communities_touched: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    answer: GroundedAnswer = field(default_factory=lambda: GroundedAnswer(answer=ABSTAIN_ANSWER, abstained=True))


def local_search(query: str, graph: KnowledgeGraph, provider: Provider, *, hops: int = 1) -> GraphSearchResult:
    """Answer an entity-focused question from a seed entity's graph neighborhood.

    Seeds on every graph entity that appears (case-insensitively) in the
    query, collects chunks that directly mention a seed entity, then
    expands `hops` steps along edges, collecting each traversed edge's
    evidence chunks and each newly reached entity's own direct chunks. This
    reaches a fact sharing no vocabulary with the query, as long as it
    shares an entity with something the query does mention.

    Args:
        query: The user's question.
        graph: The `KnowledgeGraph` to search.
        provider: `Provider` used for the final grounded-generation call.
        hops: Number of edge-traversal steps from the seed entities.

    Returns:
        A `GraphSearchResult` with mode "local".
    """
    seeds = [entity for entity in graph.entities if entity.lower() in query.lower()]
    visited = set(seeds)
    touched_chunks: set[str] = set()
    for entity in seeds:
        touched_chunks.update(cid for cid, ents in graph.entities_by_chunk.items() if entity in ents)

    frontier = set(seeds)
    for _ in range(hops):
        next_frontier: set[str] = set()
        for entity in frontier:
            for edge in graph.adjacency.get(entity, []):
                other = edge.target if edge.source == entity else edge.source
                touched_chunks.update(edge.chunk_ids)
                if other not in visited:
                    next_frontier.add(other)
                    touched_chunks.update(cid for cid, ents in graph.entities_by_chunk.items() if other in ents)
        visited.update(next_frontier)
        frontier = next_frontier

    chunks = [graph.chunks_by_id[cid] for cid in sorted(touched_chunks)]
    answer = generate_grounded_answer(query, chunks, provider)
    return GraphSearchResult(
        mode="local", entities_touched=sorted(visited), chunk_ids=sorted(touched_chunks), answer=answer
    )


def build_global_map_prompt(query: str, community: Community) -> str:
    """Build the prompt asking for one community's partial answer to a global question."""
    return f"Question: {query}\n\nCommunity {community.id} summary: {community.summary}"


def build_global_reduce_prompt(query: str, partials: dict[int, str]) -> str:
    """Build the prompt asking the model to reduce partial answers into one report."""
    lines = [f"Question: {query}", "", "Partial answers:"]
    for community_id, text in partials.items():
        lines.append(f"[community-{community_id}] {text}")
    return "\n".join(lines)


def global_search(query: str, communities: list[Community], provider: Provider) -> GraphSearchResult:
    """Answer a corpus-level question by map-reduce over community summaries.

    No chunk is retrieved: every community is mapped over its pregenerated
    summary for a partial answer (or NOT RELEVANT), and the partials are
    reduced into one report. This is the shape a single passage cannot
    answer, since no one chunk holds "what are the themes in this corpus."

    Args:
        query: The corpus-level question.
        communities: Communities with summaries already set by `summarize_communities`.
        provider: `Provider` used for one map call per community plus one reduce call.

    Returns:
        A `GraphSearchResult` with mode "global" and an empty `chunk_ids`.
    """
    partials: dict[int, str] = {}
    for community in communities:
        completion = provider.complete([Message.user(build_global_map_prompt(query, community))], system=_GLOBAL_MAP_SYSTEM)
        text = completion.content.strip()
        if text.upper() != "NOT RELEVANT":
            partials[community.id] = text

    if not partials:
        return GraphSearchResult(mode="global", communities_touched=[])

    completion = provider.complete([Message.user(build_global_reduce_prompt(query, partials))], system=_GLOBAL_REDUCE_SYSTEM)
    valid_ids = {f"community-{cid}" for cid in partials}
    citations = extract_citations(completion.content, valid_ids)
    answer = GroundedAnswer(answer=completion.content, citations=citations, abstained=False)
    return GraphSearchResult(mode="global", communities_touched=sorted(partials), answer=answer)


def graph_adds_value(local_chunk_ids: list[str], flat_chunk_ids: list[str]) -> bool:
    """Report whether local search reached a chunk flat retrieval, at the same
    fetch size, did not.

    This is the module's honest answer to arXiv:2506.05690: a graph earns
    its cost only when it reaches evidence flat retrieval misses. On a
    single-hop factoid, `local_search` and `dense_retrieve` typically land
    on the identical chunk set, and this returns False.

    Args:
        local_chunk_ids: Chunk ids `local_search` grounded its answer in.
        flat_chunk_ids: Chunk ids a flat retriever returned for the same query.

    Returns:
        True if local search reached at least one chunk flat retrieval missed.
    """
    return bool(set(local_chunk_ids) - set(flat_chunk_ids))


_DEMO_ENTITY_SCRIPT = "\n".join(
    [
        "deploy-policy#0: (none)",
        "deploy-policy#1: SEV1, Deployment Freeze",
        "deploy-policy#2: SEV1",
        "incident-runbook#0: SEV1, Incident Declaration",
        "incident-runbook#1: SEV1, Rollback Command",
        "incident-runbook#2: (none)",
        "oncall-rotation#0: On-Call Escalation, PagerDuty",
        "oncall-rotation#1: PagerDuty",
        "data-retention#0: Data Retention, GDPR Deletion",
        "data-retention#1: GDPR Deletion",
        "api-rate-limits#0: API Rate Limit",
        "api-rate-limits#1: API Rate Limit",
        "billing-faq#0: Billing Plans, Proration Credit",
        "billing-faq#1: Billing Plans, Invoice Disputes",
    ]
)


@dataclass
class GraphRagDemoResult:
    """Everything the graph-RAG demo produced, for `main.py` and tests to inspect.

    Attributes:
        entities_by_chunk: The scripted entity extraction result.
        communities: Communities detected over the demo graph, with summaries.
        local_result: The local-search result for the multi-hop demo query.
        local_flat_baseline: Chunk ids a flat, single-hop dense fetch (`top_k=1`)
            returned for the same query, for comparison against `local_result`.
        global_result: The global-search result for the corpus-themes query.
        skeptic_result: The local-search result for the single-hop skeptic query.
        skeptic_flat_baseline: Chunk ids a flat dense fetch (matched `top_k`)
            returned for the skeptic query.
    """

    entities_by_chunk: dict[str, list[str]]
    communities: list[Community]
    local_result: GraphSearchResult
    local_flat_baseline: list[str]
    global_result: GraphSearchResult
    skeptic_result: GraphSearchResult
    skeptic_flat_baseline: list[str]


def run_graph_rag_demo(
    provider: Provider | None = None,
    *,
    chunks: list[Chunk] | None = None,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> GraphRagDemoResult:
    """Demonstrate local multi-hop search, global map-reduce, and the skeptic case.

    Scripted entity extraction plants five communities: SEV1 incident
    mechanics (SEV1, rollback, deployment freeze, severity declaration),
    on-call escalation, data retention, API rate limits, and billing. Local
    search asks whether deployment freezes apply during a SEV1 caused by a
    bad deploy; a flat `top_k=1` dense fetch only reaches the rollback
    chunk, but one-hop traversal from "SEV1" also reaches the freeze chunk.
    Global search asks a corpus-level "what are the themes" question,
    answered by mapping over every community summary, with no chunk-level
    retrieval call. The skeptic query asks a single-hop API-limit factoid:
    that entity has no edges, so local search's direct-mention lookup
    returns exactly what a flat dense fetch at the same `top_k` already
    returns, the arXiv:2506.05690 finding shown honestly rather than papered over.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with entity extraction, five community
            summaries, the local-search answer, five global map calls, and
            the global reduce.
        chunks: The corpus chunks to build the graph over. Defaults to `default_chunks()`.
        dense_index: A prebuilt `DenseIndex`, for the flat-retrieval
            baselines. Built fresh with `embedder` when omitted.
        embedder: Embedder for the flat-retrieval baselines. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        A `GraphRagDemoResult` covering all three searches.
    """
    all_chunks = chunks if chunks is not None else default_chunks()
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(all_chunks, embedder)

    if provider is None:
        provider = get_provider(
            script=[
                _DEMO_ENTITY_SCRIPT,
                # community summaries, in id order: ops, on-call, data, api-limit, billing
                "This community covers Aurora's SEV1 incident-response mechanics: how a SEV1 is "
                "declared, the rollback-first mitigation, and the deployment freeze that applies "
                "while one is active.",
                "This community covers on-call escalation: PagerDuty pages the primary, and "
                "escalates to the secondary and then the engineering manager if unacknowledged.",
                "This community covers data handling: log and backup retention windows and the "
                "GDPR deletion-request deadline.",
                "This community covers API rate limiting: the default per-key request limit and "
                "the burst allowance.",
                "This community covers billing mechanics: monthly and annual plans, the automatic "
                "proration credit on upgrade, and where invoice disputes go.",
                # local search: grounded answer over the traversed SEV1 neighborhood
                "The first mitigation step for a deploy-caused SEV1 is always a rollback, not a "
                "forward fix [incident-runbook#1]. Deployment freezes also apply during any active "
                "SEV1 incident [deploy-policy#1].",
                # global search: one map call per community, in id order
                "Incident response: SEV1s are declared within five minutes, mitigated first with a "
                "rollback, and trigger deployment freezes.",
                "On-call: PagerDuty escalates from primary to secondary to the engineering manager "
                "if a page goes unacknowledged.",
                "Data handling: logs and backups have fixed retention windows, and GDPR deletion "
                "requests are honored within thirty days.",
                "API usage: requests are capped at one hundred per minute with a short burst allowance.",
                "Billing: plans automatically prorate on upgrade, and disputes route to the billing team.",
                # global reduce
                "Aurora's documentation covers five operational themes: incident response and "
                "deployment freezes [community-0], on-call escalation [community-1], data retention "
                "and GDPR compliance [community-2], API rate limiting [community-3], and billing and "
                "proration [community-4].",
                # skeptic local search: same two chunks a flat fetch already finds
                "The default API rate limit is one hundred requests per minute per key, with a short "
                "burst allowance of two hundred requests [api-rate-limits#0].",
            ]
        )

    entities_by_chunk = extract_entities_llm(all_chunks, provider)
    graph = build_graph(all_chunks, entities_by_chunk)
    communities = detect_communities(graph)
    summarize_communities(communities, graph.chunks_by_id, provider)

    local_result = local_search(_LOCAL_DEMO_QUERY, graph, provider, hops=1)
    local_flat_baseline = [sc.chunk.id for sc in dense_retrieve(_LOCAL_DEMO_QUERY, dense_index, embedder, top_k=1)]

    global_result = global_search(_GLOBAL_DEMO_QUERY, communities, provider)

    skeptic_result = local_search(_SKEPTIC_DEMO_QUERY, graph, provider, hops=1)
    skeptic_flat_baseline = [
        sc.chunk.id for sc in dense_retrieve(_SKEPTIC_DEMO_QUERY, dense_index, embedder, top_k=len(skeptic_result.chunk_ids) or 1)
    ]

    return GraphRagDemoResult(
        entities_by_chunk=entities_by_chunk,
        communities=communities,
        local_result=local_result,
        local_flat_baseline=local_flat_baseline,
        global_result=global_result,
        skeptic_result=skeptic_result,
        skeptic_flat_baseline=skeptic_flat_baseline,
    )
