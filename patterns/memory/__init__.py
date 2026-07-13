"""Memory (short-term + vector) pattern: the plumbing deciding what an
agent keeps, evicts, persists, and pulls back into context.

This package implements the pattern's major sub-variants across separate,
composable modules: short-term memory (full buffer, sliding window,
summarization/compaction), a shared in-memory vector store, the three
long-term memory types (semantic, episodic, procedural), retrieval scoring
strategies, a hot-path/background write policy with fact extraction and
conflict resolution, a context assembler that merges every source under a
token budget, a MemGPT-style paged memory variant, a memory-as-tools mode,
and a file-directory memory backend.

See `patterns/memory/README.md` for the full write-up and
`patterns/memory/main.py` for a runnable demo of every variant.
"""
