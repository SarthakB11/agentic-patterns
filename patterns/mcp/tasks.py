"""Durable async tasks: create, poll, retrieve, cancel.

Every tool in `server.py` answers a `tools/call` synchronously, in one
round trip. MCP's tasks utility (SEP-1686, experimental in the 2025-11-25
revision this pattern targets; reshaped into a stateless extension by
SEP-2663 in the 2026-07-28 release candidate) is for the opposite case:
work that should not hold the connection open. A task-augmented call
returns a receipt, not a result; the caller polls for completion and
retrieves the result in a second call. The receipt carries a real state
machine (`working`, `input_required`, `completed`, `failed`, `cancelled`,
with the last three terminal), and support is negotiated two levels deep:
a server-wide `tasks` capability, then a per-tool `execution.taskSupport`
of `"optional"` or `"required"`.

`TaskServer` runs in-process rather than over a subprocess and stdio pipe.
Task polling has nothing to do with framing or process lifecycle, the two
things a transport varies; what is new here is a state machine, and a
plain dispatch function shows it with no transport plumbing in the way.
The task advances by a poll counter on the record, not by wall-clock time,
so the same number of `tasks/get` calls always reaches `completed` on the
same poll, nothing here sleeps or races a clock. Task ids and the
`createdAt` / `lastUpdatedAt` fields likewise come from a deterministic
tick counter rather than real timestamps or random ids; a production
server must use unpredictable, collision-resistant task ids, since a
guessable id lets one caller poll or cancel another caller's task.

This is a separate server from `server.py` on purpose: `server.py`'s three
tools are all fast and synchronous, so folding task support into it would
mean faking slowness on tools that have none. Flagging the consequence for
the rest of this pattern: `server.py`'s `SERVER_CAPABILITIES` correctly
omits `tasks` today, but the day it grows a genuinely long-running tool it
must advertise `tasks.requests.tools.call` the way
`TASKS_SERVER_CAPABILITY` does here, and no client should send a `task`
field to a server that has not.

Error mapping followed here is the 2025-11-25 shape this pattern targets:
unknown `taskId` on get/result/cancel is `-32602` (`INVALID_PARAMS`), and a
`taskSupport: "required"` tool called without a `task` field is `-32601`
(`METHOD_NOT_FOUND`, exposed below as `TASK_REQUIRED_ERROR`). The RC is
reported to reshape this toward `-32600` for the task-required case; that
mapping could not be independently confirmed against a shipped spec page,
so this module stays on the 2025-11-25 numbers rather than guess at the RC's.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from patterns.mcp import jsonrpc
from patterns.mcp.client import MCPProtocolError

TASKS_SERVER_CAPABILITY: dict[str, Any] = {"tasks": {"requests": {"tools": {"call": True}}, "list": True, "cancel": True}}

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# See the module docstring's "Error mapping" paragraph for why this is
# METHOD_NOT_FOUND rather than the RC's reported (unverified) -32600.
TASK_REQUIRED_ERROR = jsonrpc.METHOD_NOT_FOUND


def _run_slow_add(arguments: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    try:
        return [{"type": "text", "text": str(arguments["a"] + arguments["b"])}], False
    except (KeyError, TypeError) as exc:
        return [{"type": "text", "text": f"invalid arguments for slow_add: {exc}"}], True


def _run_priority_report(arguments: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    topic = arguments.get("topic")
    if not topic:
        return [{"type": "text", "text": "missing required argument: topic"}], True
    return [{"type": "text", "text": f"Priority report on {topic}: no open incidents, all green."}], False


_TASK_TOOLS: dict[str, dict[str, Any]] = {
    "slow_add": {
        "spec": {
            "name": "slow_add",
            "description": "Add two numbers. Modeled as slow enough that a caller may run it as a task.",
            "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
            "execution": {"taskSupport": "optional"},
        },
        "run": _run_slow_add,
    },
    "priority_report": {
        "spec": {
            "name": "priority_report",
            "description": "Generate a priority report for a topic. Always requires task augmentation.",
            "inputSchema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
            "execution": {"taskSupport": "required"},
        },
        "run": _run_priority_report,
    },
}


@dataclass
class TaskRecord:
    """One task's state.

    Attributes:
        task_id: Deterministic id, `"task-N"` in creation order.
        tool_name: The tool this task will eventually run.
        arguments: The arguments that tool will be called with.
        status: One of `working`, `completed`, `failed`, `cancelled`.
        poll_count: Number of `tasks/get` calls this task has answered.
        complete_after: The poll count at which `status` flips to a terminal state.
        result: The `CallToolResult`-shaped dict, set only once `status` is terminal.
        created_tick: A deterministic sequence number standing in for a
            creation timestamp; see the module docstring.
        updated_tick: The same, refreshed on every state-changing operation.
    """

    task_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str = "working"
    poll_count: int = 0
    complete_after: int = 2
    result: dict[str, Any] | None = None
    created_tick: int = 0
    updated_tick: int = 0


class TaskServer:
    """An in-process server exposing task-augmented `tools/call` plus `tasks/*`."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._next_id = itertools.count(1)
        self._tick = itertools.count(1)

    def _new_task_id(self) -> str:
        return f"task-{next(self._next_id)}"

    def _now(self) -> int:
        return next(self._tick)

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message and return the response, if any."""
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        params = message.get("params", {}) or {}

        if method == "tools/list":
            return jsonrpc.build_response(msg_id, {"tools": [entry["spec"] for entry in _TASK_TOOLS.values()]})
        if method == "tools/call":
            return self._handle_tools_call(msg_id, params)
        if method == "tasks/get":
            return self._handle_tasks_get(msg_id, params)
        if method == "tasks/result":
            return self._handle_tasks_result(msg_id, params)
        if method == "tasks/cancel":
            return self._handle_tasks_cancel(msg_id, params)
        if method == "tasks/list":
            return jsonrpc.build_response(msg_id, {"tasks": [self._task_view(t) for t in self._tasks.values()]})
        if is_notification:
            return None
        return jsonrpc.build_error(msg_id, jsonrpc.METHOD_NOT_FOUND, f"unknown method: {method!r}")

    def _handle_tools_call(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        entry = _TASK_TOOLS.get(name)
        if entry is None:
            return jsonrpc.build_error(msg_id, jsonrpc.METHOD_NOT_FOUND, f"unknown tool: {name!r}")
        wants_task = "task" in params
        support = entry["spec"]["execution"]["taskSupport"]
        if support == "required" and not wants_task:
            return jsonrpc.build_error(
                msg_id, TASK_REQUIRED_ERROR, f"{name!r} requires task augmentation; call it with a 'task' field"
            )
        arguments = params.get("arguments", {})
        if not wants_task:
            content, is_error = entry["run"](arguments)
            return jsonrpc.build_response(msg_id, {"content": content, "isError": is_error})

        task_id = self._new_task_id()
        tick = self._now()
        record = TaskRecord(task_id=task_id, tool_name=name, arguments=arguments, created_tick=tick, updated_tick=tick)
        self._tasks[task_id] = record
        return jsonrpc.build_response(msg_id, {"task": self._task_view(record)})

    def _advance(self, record: TaskRecord) -> None:
        """Move a working task one poll closer to completion.

        Flips to a terminal status only once `poll_count` reaches
        `complete_after`, deterministically, with no dependence on how much
        real time has passed between polls.
        """
        if record.status != "working":
            return
        record.poll_count += 1
        record.updated_tick = self._now()
        if record.poll_count >= record.complete_after:
            entry = _TASK_TOOLS[record.tool_name]
            content, is_error = entry["run"](record.arguments)
            record.status = "failed" if is_error else "completed"
            record.result = {"content": content, "isError": is_error}

    def _handle_tasks_get(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        record = self._tasks.get(params.get("taskId"))
        if record is None:
            return jsonrpc.build_error(msg_id, jsonrpc.INVALID_PARAMS, f"unknown taskId: {params.get('taskId')!r}")
        self._advance(record)
        return jsonrpc.build_response(msg_id, {"task": self._task_view(record)})

    def _handle_tasks_result(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        record = self._tasks.get(params.get("taskId"))
        if record is None:
            return jsonrpc.build_error(msg_id, jsonrpc.INVALID_PARAMS, f"unknown taskId: {params.get('taskId')!r}")
        if record.status not in _TERMINAL_STATUSES:
            return jsonrpc.build_error(msg_id, jsonrpc.INVALID_PARAMS, f"task {record.task_id!r} has not reached a terminal status yet: {record.status!r}")
        return jsonrpc.build_response(msg_id, record.result)

    def _handle_tasks_cancel(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        record = self._tasks.get(params.get("taskId"))
        if record is None:
            return jsonrpc.build_error(msg_id, jsonrpc.INVALID_PARAMS, f"unknown taskId: {params.get('taskId')!r}")
        if record.status in _TERMINAL_STATUSES:
            return jsonrpc.build_error(msg_id, jsonrpc.INVALID_PARAMS, f"task {record.task_id!r} is already terminal ({record.status!r}); cannot cancel")
        record.status = "cancelled"
        record.updated_tick = self._now()
        record.result = {"content": [{"type": "text", "text": f"task {record.task_id!r} was cancelled"}], "isError": True}
        return jsonrpc.build_response(msg_id, {"task": self._task_view(record)})

    def _task_view(self, record: TaskRecord) -> dict[str, Any]:
        return {
            "taskId": record.task_id,
            "status": record.status,
            "createdAt": f"tick-{record.created_tick}",
            "lastUpdatedAt": f"tick-{record.updated_tick}",
            "ttl": 3600,
            "pollInterval": 0,
        }


class TaskClient:
    """Thin request wrapper over a `TaskServer`.

    Mirrors `MCPClient`'s call shapes without spawning a subprocess: task
    polling has no framing or process-lifecycle concern to demonstrate,
    only the state machine, so this talks to `TaskServer.handle` directly.
    """

    def __init__(self, server: TaskServer) -> None:
        self._server = server
        self._ids = itertools.count(1)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        msg_id = f"tsk-{next(self._ids)}"
        response = self._server.handle(jsonrpc.build_request(msg_id, method, params))
        assert response is not None  # requests always get a response from TaskServer
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(err["code"], err["message"])
        return response["result"]

    def call_as_task(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call `name` with task augmentation and return the receipt (`{taskId, status, ...}`)."""
        return self._request("tools/call", {"name": name, "arguments": arguments, "task": {}})["task"]

    def call_sync(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call `name` the plain, synchronous way, for comparison with the task path."""
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def poll(self, task_id: str) -> dict[str, Any]:
        """Advance and return one task's status."""
        return self._request("tasks/get", {"taskId": task_id})["task"]

    def result(self, task_id: str) -> dict[str, Any]:
        """Retrieve a terminal task's `CallToolResult`."""
        return self._request("tasks/result", {"taskId": task_id})

    def cancel(self, task_id: str) -> dict[str, Any]:
        """Cancel a non-terminal task."""
        return self._request("tasks/cancel", {"taskId": task_id})["task"]

    def list_tasks(self) -> list[dict[str, Any]]:
        """List every task the server currently holds."""
        return self._request("tasks/list", {})["tasks"]


def run_tasks_demo() -> dict[str, Any]:
    """Run the task lifecycle end to end: create, two polls, retrieve, cancel, and error paths.

    Returns:
        A dict of outcomes for `main.py` to print and `tests/test_mcp.py`
        to assert against.
    """
    server = TaskServer()
    client = TaskClient(server)

    receipt = client.call_as_task("slow_add", {"a": 12, "b": 30})
    task_id = receipt["taskId"]
    poll_1 = client.poll(task_id)
    poll_2 = client.poll(task_id)
    final_result = client.result(task_id)
    sync_equivalent = client.call_sync("slow_add", {"a": 12, "b": 30})

    receipt_2 = client.call_as_task("slow_add", {"a": 1, "b": 1})
    task_id_2 = receipt_2["taskId"]
    cancelled = client.cancel(task_id_2)
    cancelled_result = client.result(task_id_2)
    try:
        client.cancel(task_id_2)
        cancel_twice_raised = False
    except MCPProtocolError as exc:
        cancel_twice_raised = exc.code == jsonrpc.INVALID_PARAMS

    try:
        client.poll("task-does-not-exist")
        unknown_task_raised = False
    except MCPProtocolError as exc:
        unknown_task_raised = exc.code == jsonrpc.INVALID_PARAMS

    try:
        client.call_sync("priority_report", {"topic": "checkout latency"})
        required_gate_raised = False
    except MCPProtocolError as exc:
        required_gate_raised = exc.code == TASK_REQUIRED_ERROR

    return {
        "receipt_status": receipt["status"],
        "poll_1_status": poll_1["status"],
        "poll_2_status": poll_2["status"],
        "final_content": final_result["content"][0]["text"],
        "sync_content": sync_equivalent["content"][0]["text"],
        "cancelled_status": cancelled["status"],
        "cancelled_result_isError": cancelled_result["isError"],
        "cancel_twice_raised": cancel_twice_raised,
        "unknown_task_raised": unknown_task_raised,
        "required_gate_raised": required_gate_raised,
    }
