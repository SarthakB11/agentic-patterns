"""MCP (Model Context Protocol) pattern: a client and server built from scratch.

This package implements a minimal but honest subset of MCP over the stdio
transport: a JSON-RPC 2.0 codec, the `initialize` handshake with capability
negotiation, `tools/list` and `tools/call` (including an `isError` result
and a JSON-RPC protocol error), `resources/list` and `resources/read`
(text and binary), `prompts/list` and `prompts/get`, a sampling round trip,
a multi-server host, and a small HTTP transport variant on loopback.

It also covers four mechanisms that move underneath that handshake-based
core in 2025-2026: the stateless protocol core that removes the handshake
entirely (`stateless.py`), tool-definition pinning against the rug-pull
attack (`integrity.py`), durable async tasks (`tasks.py`), server-initiated
elicitation of structured input (`elicitation.py`), and registry-based
server discovery (`discovery.py`).

See `patterns/mcp/README.md` for the full write-up, the protocol revision
this subset targets, and what is deliberately left out, and
`patterns/mcp/main.py` for a runnable demo of every piece.
"""
