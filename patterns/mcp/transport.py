"""Stdio transport: newline-delimited JSON-RPC over a child process's pipes.

This is the default MCP transport for local integrations. The client
launches the server as a subprocess and exchanges one JSON-RPC message per
line over the child's stdin and stdout. Nothing but JSON-RPC may touch
stdout on the server side; a stray `print` there would corrupt the stream,
which is why `StdioServerTransport` routes all logging to stderr and why
the server module in this package never calls the builtin `print`.

Protocol semantics (the JSON-RPC message shapes, the initialize handshake,
the methods) are identical across transports; only framing and process
lifecycle differ. See `http_transport.py` for the same semantics carried
over loopback HTTP instead of pipes.
"""

from __future__ import annotations

import selectors
import subprocess
import sys
from typing import Any

from patterns.mcp.jsonrpc import JSONRPCDecodeError, decode_line, encode_line


class StdioServerTransport:
    """The server side of the stdio transport: read stdin, write stdout, log to stderr."""

    def __init__(self) -> None:
        self._stdin = sys.stdin
        self._stdout = sys.stdout

    def read_message(self) -> dict[str, Any] | None:
        """Read and decode the next line from stdin.

        Returns:
            The decoded message, or `None` at end of stream (the client
            closed its write end, e.g. as the first step of shutdown).
        """
        line = self._stdin.readline()
        if line == "":
            return None
        return decode_line(line)

    def write_message(self, message: dict[str, Any]) -> None:
        """Encode and write one message to stdout, then flush immediately."""
        self._stdout.write(encode_line(message))
        self._stdout.flush()

    @staticmethod
    def log(text: str) -> None:
        """Write a log line to stderr. Stdout is reserved for JSON-RPC only."""
        print(text, file=sys.stderr, flush=True)


class TransportTimeoutError(TimeoutError):
    """Raised when a response does not arrive within the caller's timeout."""


class TransportClosedError(EOFError):
    """Raised when the server process closes its stdout before a response arrives."""


class StdioClientTransport:
    """The client side of the stdio transport: spawn a server subprocess and talk to it."""

    def __init__(self, command: list[str]) -> None:
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered, required for prompt request/response turnaround
        )
        self._selector = selectors.DefaultSelector()
        assert self._proc.stdout is not None
        self._selector.register(self._proc.stdout, selectors.EVENT_READ)

    @property
    def pid(self) -> int:
        """Process id of the spawned server, for logging and tests."""
        return self._proc.pid

    def is_running(self) -> bool:
        """True if the server subprocess has not exited."""
        return self._proc.poll() is None

    def send(self, message: dict[str, Any]) -> None:
        """Encode and write one message to the server's stdin, then flush."""
        assert self._proc.stdin is not None
        self._proc.stdin.write(encode_line(message))
        self._proc.stdin.flush()

    def receive(self, timeout: float) -> dict[str, Any]:
        """Read and decode the next line from the server's stdout.

        Args:
            timeout: Seconds to wait for a line before giving up.

        Raises:
            TransportTimeoutError: No line arrived within `timeout`.
            TransportClosedError: The server closed stdout (process exited).
            JSONRPCDecodeError: The line was not a valid JSON-RPC message.
        """
        ready = self._selector.select(timeout=timeout)
        if not ready:
            raise TransportTimeoutError(f"no response from server within {timeout}s")
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if line == "":
            raise TransportClosedError("server closed its stdout")
        return decode_line(line)

    def stderr_text(self) -> str:
        """Drain and return whatever the server has logged to stderr so far."""
        assert self._proc.stderr is not None
        return self._proc.stderr.read() if self._proc.stderr.readable() else ""

    def close(self, wait_seconds: float = 2.0) -> None:
        """Shut down the server cleanly: close stdin, wait, then escalate.

        Closing stdin signals end-of-stream to a well-behaved server, which
        should exit its read loop and terminate on its own. If it has not
        exited after `wait_seconds`, escalate to SIGTERM, then to SIGKILL as
        a last resort.
        """
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=wait_seconds)
        self._selector.close()


__all__ = [
    "StdioServerTransport",
    "StdioClientTransport",
    "TransportTimeoutError",
    "TransportClosedError",
    "JSONRPCDecodeError",
]
