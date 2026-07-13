"""MCP pattern: a client and server built from scratch, talking JSON-RPC 2.0.

This demo runs six sections end to end, entirely offline, no network call
and no API key. The only subprocess involved is this repository's own
`patterns/mcp/server.py`, spawned locally over stdio; every model call goes
through `MockProvider` with a scripted, coherent conversation.

1. JSON-RPC codec: a request round-trips through encode and decode, and a
   malformed line is rejected without crashing the caller.
2. Baseline stdio walkthrough: handshake, `tools/list`, a successful
   `tools/call`, an `isError: true` result (divide by zero), a JSON-RPC
   protocol error (unknown tool), `resources/list` and `resources/read`
   (text, binary, and a missing URI), and `prompts/list` and `prompts/get`.
3. Host loop: a real MCP server's tools registered into the core
   `ToolRegistry` and driven by a scripted `MockProvider`, end to end.
4. Sampling: the server borrows the client's model for one tool, once with
   the capability offered and once without.
5. Multi-server host: two server subprocesses, one merged and namespaced
   tool catalog, both calls routed to the right connection.
6. HTTP transport: the same JSON-RPC semantics, framed over loopback HTTP
   instead of stdio pipes.

Run it from the repository root:

    python -m patterns.mcp.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run sections 3-5 against a
real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

from patterns.mcp import bridge, http_transport, jsonrpc, multi_server, sampling
from patterns.mcp.client import MCPClient, MCPProtocolError


def _demo_codec() -> None:
    print("=== 1. JSON-RPC codec: round-trip and rejection ===")
    request = jsonrpc.build_request("demo-1", "tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}})
    line = jsonrpc.encode_line(request)
    decoded = jsonrpc.decode_line(line)
    print(f"encoded:  {line.strip()}")
    print(f"decoded matches original: {decoded == request}")
    try:
        jsonrpc.decode_line("{not json, and this\nhas an embedded newline}")
    except jsonrpc.JSONRPCDecodeError as exc:
        print(f"malformed frame rejected: {exc}")
    print()


def _demo_basics() -> None:
    print("=== 2. Baseline stdio walkthrough ===")
    client = MCPClient()
    client.connect()
    init_result = client.initialize()
    client.notify_initialized()
    print(f"negotiated protocol version: {client.negotiated_version}")
    print(f"server: {init_result['serverInfo']['name']} v{init_result['serverInfo']['version']}")

    tools = client.list_tools()
    print(f"tools/list: {[t['name'] for t in tools]}")

    ok = client.call_tool("add", {"a": 12, "b": 30})
    print(f"tools/call add(12, 30): {ok['content'][0]['text']!r} isError={ok['isError']}")

    err_result = client.call_tool("divide", {"a": 10, "b": 0})
    print(f"tools/call divide(10, 0): {err_result['content'][0]['text']!r} isError={err_result['isError']}")

    try:
        client.call_tool("multiply", {"a": 2, "b": 3})
    except MCPProtocolError as exc:
        print(f"tools/call multiply (unknown tool) raised JSON-RPC error: code={exc.code}")

    resources = client.list_resources()
    print(f"resources/list: {[r['uri'] for r in resources]}")
    text_contents = client.read_resource("note://todo")
    print(f"resources/read note://todo: {text_contents[0]['text']!r}")
    blob_contents = client.read_resource("asset://logo")
    print(f"resources/read asset://logo: {len(blob_contents[0]['blob'])} base64 chars, mimeType={blob_contents[0]['mimeType']}")
    try:
        client.read_resource("note://missing")
    except MCPProtocolError as exc:
        print(f"resources/read missing URI raised: code={exc.code}")

    prompts = client.list_prompts()
    print(f"prompts/list: {[p['name'] for p in prompts]}")
    filled = client.get_prompt("summarize_notes", {"tone": "casual"})
    print(f"prompts/get summarize_notes(tone=casual): {filled['messages'][0]['content']['text']!r}")

    client.shutdown()
    print()


def _demo_host_loop() -> None:
    print("=== 3. Host loop: MockProvider driving a real MCP server ===")
    answers = bridge.run_host_loop_demo()
    for answer in answers:
        print(f"final answer: {answer}")
    print()


def _demo_sampling() -> None:
    print("=== 4. Sampling: server borrows the client's model ===")
    accepted, refused = sampling.run_sampling_demo()
    print(f"sampling-capable client: isError={accepted['isError']} summary={accepted['content'][0]['text']!r}")
    print(f"sampling-unaware client: isError={refused['isError']} message={refused['content'][0]['text']!r}")
    print()


def _demo_multi_server() -> None:
    print("=== 5. Multi-server host: two servers, one merged namespace ===")
    merged_names, final_answer = multi_server.run_multi_server_demo()
    print(f"merged tool names: {merged_names}")
    print(f"final answer: {final_answer}")
    print()


def _demo_http_transport() -> None:
    print("=== 6. HTTP transport: same semantics, loopback framing ===")
    result = http_transport.run_http_transport_demo()
    print(f"server: {result['server_info']['name']}")
    print(f"tools/list over HTTP: {result['tool_names']}")
    print(f"add(2, 3) over HTTP: {result['add_result']['content'][0]['text']!r}")
    print()


def main() -> None:
    """Run every MCP demo section and print a readable transcript."""
    print("MCP PATTERN: a client and server built from scratch over JSON-RPC 2.0\n")
    _demo_codec()
    _demo_basics()
    _demo_host_loop()
    _demo_sampling()
    _demo_multi_server()
    _demo_http_transport()
    print("All six sections completed without exhausting a script or leaking a subprocess.")


if __name__ == "__main__":
    main()
