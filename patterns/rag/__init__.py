"""RAG (retrieval-augmented generation): naive, hybrid, and reranked variants.

Retrieval grounds a model's answer in text fetched at query time from a
corpus, instead of relying only on what the model memorized during training.
This package builds the pattern up in small, independently testable pieces:
chunking, dense retrieval, term-based retrieval, hybrid fusion, a middle-tier
late-interaction retriever, reranking, query transformation, contextual
chunk embedding, relevance and sufficiency grading, grounded generation with
citations and an abstain path, and an agentic loop that calls retrieval as a
tool. See `patterns/rag/README.md` for the full variant list and `main.py`
for a runnable, end-to-end demo of all of them.
"""
