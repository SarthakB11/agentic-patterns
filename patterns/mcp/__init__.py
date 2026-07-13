"""MCP (Model Context Protocol) pattern: a client and server built from scratch.

This package implements a minimal but honest subset of MCP over the stdio
transport: a JSON-RPC 2.0 codec, the `initialize` handshake with capability
negotiation, `tools/list` and `tools/call` (including an `isError` result
and a JSON-RPC protocol error), `resources/list` and `resources/read`
(text and binary), `prompts/list` and `prompts/get`, a sampling round trip,
a multi-server host, and a small HTTP transport variant on loopback.

See `patterns/mcp/README.md` for the full write-up, the protocol revision
this subset targets, and what is deliberately left out, and
`patterns/mcp/main.py` for a runnable demo of every piece.
"""
