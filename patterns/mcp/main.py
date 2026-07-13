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
7. Stateless core (2026-07-28 RC): no handshake, protocol version and
   capabilities riding in `_meta` on every request, two independent server
   instances answering the same client correctly.
8. Tool-definition integrity: pinning tool definitions by content hash,
   screening descriptions for hidden instructions, and failing closed on a
   rug pull between two `tools/list` calls.
9. Durable async tasks: a task-augmented `tools/call` returns a receipt,
   polling advances it deterministically, cancellation and error paths.
10. Elicitation: a server asking the human mid-call for structured input,
    form mode and URL mode, across accept, decline, and cancel.
11. Registry-based discovery: filtering a static server registry and
    connecting the selected record's real subprocess.

Run it from the repository root:

    python -m patterns.mcp.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run sections 3-5 against a
real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

from patterns.mcp import (
    bridge,
    discovery,
    elicitation,
    http_transport,
    integrity,
    jsonrpc,
    multi_server,
    sampling,
    stateless,
    tasks,
)
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


def _demo_stateless() -> None:
    print("=== 7. Stateless core (2026-07-28 RC): no handshake, _meta per request ===")
    result = stateless.run_stateless_demo()
    add_1_content, served_by_1 = result["add_1"]
    add_2_content, served_by_2 = result["add_2"]
    print(f"add(12, 30) served by {served_by_1}: {add_1_content['content'][0]['text']!r}")
    print(f"add(100, 1) served by {served_by_2}: {add_2_content['content'][0]['text']!r}")
    print(f"sampling-gated tool with capability offered: isError={result['gated_ok']['isError']}")
    print(f"sampling-gated tool without capability: isError={result['gated_refused']['isError']}")
    print(f"initialize under the stateless core: error code={result['handshake_gone_code']}")
    print(f"missing _meta.protocolVersion: error code={result['missing_meta_code']}")
    print()


def _demo_integrity() -> None:
    print("=== 8. Tool-definition integrity: pinning and the rug-pull defense ===")
    result = integrity.run_integrity_demo()
    print(f"clean server, both lists: approved={result['clean_report_2'].approved}, flagged={result['clean_report_2'].flagged}")
    print(f"poisoned description denied at pin time: flagged={result['denied_report'].flagged}, call refused: isError={result['denied_call']['isError']}")
    print(f"same poisoned description approved on retry: approved={result['accepted_report'].approved}")
    print(f"zero-width smuggling flagged even though it renders clean: {result['zero_width_report'].flagged}")
    print(f"rug pull across two lists: mutated={result['rugpull_report_2'].mutated}, call after mutation: isError={result['rugpull_call']['isError']}")
    print()


def _demo_tasks() -> None:
    print("=== 9. Durable async tasks: create, poll, retrieve, cancel ===")
    result = tasks.run_tasks_demo()
    print(f"task receipt status: {result['receipt_status']}, poll 1: {result['poll_1_status']}, poll 2: {result['poll_2_status']}")
    print(f"tasks/result content: {result['final_content']!r} (matches synchronous call: {result['final_content'] == result['sync_content']})")
    print(f"cancel mid-flight: {result['cancelled_status']}, tasks/result still readable: isError={result['cancelled_result_isError']}")
    print(f"cancel of an already-terminal task rejected: {result['cancel_twice_raised']}")
    print(f"unknown taskId rejected: {result['unknown_task_raised']}")
    print(f"required-support tool called without 'task' rejected: {result['required_gate_raised']}")
    print()


def _demo_elicitation() -> None:
    print("=== 10. Elicitation: server asks the human mid-call ===")
    results = elicitation.run_elicitation_demo()
    for label in ("accept", "decline", "cancel", "schema_violation", "no_capability", "url_mode"):
        content, is_error = results[label]
        print(f"{label}: isError={is_error} -> {content[0]['text']!r}")
    print()


def _demo_discovery() -> None:
    print("=== 11. Registry-based discovery: find a server, then connect it ===")
    result = discovery.run_discovery_demo()
    print(f"servers declaring 'tools': {result['tools_capable_names']}")
    print(f"exact-name lookup: {result['named_lookup_names']}")
    print(f"predicate matching nothing: {result['empty_result']}")
    print(f"connected the discovered server, tools/list: {result['connected_tool_names']}")
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
    _demo_stateless()
    _demo_integrity()
    _demo_tasks()
    _demo_elicitation()
    _demo_discovery()
    print("All eleven sections completed without exhausting a script or leaking a subprocess.")


if __name__ == "__main__":
    main()
