"""Memory (short-term + vector) pattern: the plumbing deciding what an
agent keeps, evicts, persists, and pulls back into context.

This package implements the pattern's major sub-variants across separate,
composable modules: short-term memory (full buffer, sliding window,
summarization/compaction), a shared in-memory vector store, the three
long-term memory types (semantic, episodic, procedural), retrieval scoring
strategies, a hot-path/background write policy with fact extraction and
conflict resolution, a context assembler that merges every source under a
token budget, a MemGPT-style paged memory variant, a memory-as-tools mode,
a file-directory memory backend, a Mem0-style similarity-gated update
decision (ADD/UPDATE/DELETE/NOOP), decay/TTL/capacity/intent-aware
forgetting, an offline LongMemEval-style recall benchmark, and sleep-time
compute's offline pre-derivation amortized across queries.

See `patterns/memory/README.md` for the full write-up and
`patterns/memory/main.py` for a runnable demo of every variant.
"""
