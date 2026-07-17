# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave2-Fixer-4's bug fixes.

Covers findings #2, #3, #7, #14, and #17 of ``tmp/findings-4.md``:

* #2 — ``MCPManager.call_tool`` snapshots ``conn.session`` once, so the
  manager-loop thread nulling the session between the check and the use
  (TOCTOU) yields the friendly ``"Error: ..."`` string instead of an
  ``AttributeError`` escaping the tool wrapper.
* #3 — a ``connect()`` timeout tears the straggler connection down
  (removed from the manager, stop signalled) instead of leaving a
  poisoned record that a late success turns into contradictory
  session-set-but-error-marked state with a leaked server task.
* #7 — ``_bash_streaming``'s timeout timer racing a naturally finishing
  command no longer discards a completed command's output (a natural
  exit status is trusted over the spurious ``timed_out`` flag); genuine
  timeouts are still reported.
* #14 — ``Write`` writes content verbatim (``newline=""``), matching
  ``Edit``, so a Write-then-read round trip is byte-identical.
* #17 — ``MCPServerConfig.source`` is bookkeeping only (excluded from
  equality), so re-discovering the same server from a different config
  file reuses the healthy connection instead of reconnecting.

No mocks, patches, or fakes: a real FastMCP stdio server subprocess and
real shell subprocesses are used throughout.  (The #2 test drives the
race by performing, from a plain thread, the exact ``conn.session``
transitions ``_maintain_connection``'s ``finally`` performs on the
manager loop, with a tiny GIL switch interval so the two-bytecode
check-to-use window is actually hit.)
"""

from __future__ import annotations

import dataclasses
import os
import pty
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.sorcar.mcp_servers as mcp_servers_module
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.core.useful_tools import UsefulTools

_SERVER_SCRIPT = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("testsrv")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


if __name__ == "__main__":
    mcp.run()
'''

# Sleeps well past the shrunken CONNECT_TIMEOUT before serving, so the
# connection attempt times out first and only succeeds afterwards.
_SLOW_SERVER_SCRIPT = '''
import time

time.sleep(6)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slowsrv")


@mcp.tool()
def ping() -> str:
    """Return pong."""
    return "pong"


if __name__ == "__main__":
    mcp.run()
'''


def _stdio_config(
    tmp_path: Path, name: str = "testsrv", script_body: str = _SERVER_SCRIPT,
) -> MCPServerConfig:
    """Return a stdio server config running a real FastMCP test server."""
    script = tmp_path / f"{name}.py"
    script.write_text(script_body, encoding="utf-8")
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=(str(script),),
    )


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    The MCP stdio transport passes ``sys.stderr`` as the spawned
    server's stderr and calls ``.fileno()`` on it; under pytest the std
    streams are capture objects without a real descriptor, so the
    transport cannot start.  Point stdin at a pty slave and the
    transport's bound ``errlog`` default at a plain file (same
    technique as ``test_sorcar_mcp.py`` and
    ``test_fixer5_mcp_web_bugs.py``).
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)

    from mcp.client.stdio import stdio_client

    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


# ---------------------------------------------------------------------------
# Finding #17: config.source participates in equality → spurious reconnects


def test_source_change_reuses_live_connection(
    tmp_path: Path, real_stdin: None,
) -> None:
    """The same server re-discovered from another file is not reconnected.

    Pre-fix ``MCPServerConfig.__eq__`` included the bookkeeping
    ``source`` label, so an identical server whose config moved (e.g.
    from ``.mcp.json`` to ``.kiss/mcp.json``) compared unequal and the
    healthy connection was torn down and re-established.
    """
    manager = MCPManager()
    try:
        cfg = _stdio_config(tmp_path)
        conn1 = manager.connect(cfg)
        assert conn1.error == ""
        assert conn1.session is not None

        moved = dataclasses.replace(cfg, source="project")
        assert moved == cfg, "source must not participate in equality"
        conn2 = manager.connect(moved)
        assert conn2 is conn1, "healthy connection was torn down on a source change"
        assert manager.call_tool("testsrv", "add", {"a": 2, "b": 3}) == "5"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Finding #2: call_tool session TOCTOU


def test_call_tool_session_nulled_between_check_and_use(
    tmp_path: Path, real_stdin: None,
) -> None:
    """``call_tool`` never leaks an ``AttributeError`` when the session dies.

    A plain thread performs the exact write ``_maintain_connection``'s
    ``finally`` performs on the manager loop (``conn.session = None``,
    here alternated with the live session so the loop can keep making
    real calls) while the caller thread invokes ``call_tool``.  A tiny
    GIL switch interval makes the two-bytecode window between the
    ``conn.session is None`` check and the ``conn.session.call_tool``
    attribute access get hit.  Pre-fix this raised ``AttributeError``
    out of the tool wrapper; post-fix every call returns a string (the
    result or the friendly not-connected error).
    """
    manager = MCPManager()
    try:
        cfg = _stdio_config(tmp_path)
        conn = manager.connect(cfg)
        assert conn.error == ""
        assert conn.session is not None
        real_session = conn.session
        stop_flipping = threading.Event()
        escaped: list[BaseException] = []

        def flipper() -> None:
            while not stop_flipping.is_set():
                conn.session = None
                conn.session = real_session

        def caller() -> None:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and not escaped:
                try:
                    result = manager.call_tool("testsrv", "add", {"a": 1, "b": 2})
                except BaseException as exc:  # noqa: BLE001 - recorded for assertion
                    escaped.append(exc)
                    return
                if not isinstance(result, str):
                    escaped.append(TypeError(f"non-string result: {result!r}"))
                    return

        old_interval = sys.getswitchinterval()
        flip_threads = [
            threading.Thread(target=flipper, daemon=True) for _ in range(3)
        ]
        call_threads = [
            threading.Thread(target=caller, daemon=True) for _ in range(4)
        ]
        sys.setswitchinterval(1e-6)
        for thread in flip_threads + call_threads:
            thread.start()
        try:
            for thread in call_threads:
                thread.join(timeout=60)
                assert not thread.is_alive()
        finally:
            sys.setswitchinterval(old_interval)
            stop_flipping.set()
            for thread in flip_threads:
                thread.join(timeout=10)
            conn.session = real_session
        assert not escaped, f"call_tool leaked an exception: {escaped[0]!r}"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Finding #3: connect() timeout leaves a poisoned record on late success


def test_connect_timeout_straggler_is_torn_down_not_poisoned(
    tmp_path: Path,
    real_stdin: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out connection that succeeds late is unwound, not poisoned.

    Pre-fix the record stayed in the manager: once the slow server came
    up the record was live (``session`` set) yet marked
    ``"connection timed out"`` — contradictory state — and its task
    parked on ``stop.wait()`` forever, leaking the server subprocess.
    Post-fix the timeout removes the record and signals stop, so the
    late success immediately unwinds, and a fresh connect works.
    """
    manager = MCPManager()
    try:
        cfg = _stdio_config(tmp_path, name="slowsrv", script_body=_SLOW_SERVER_SCRIPT)
        monkeypatch.setattr(mcp_servers_module, "CONNECT_TIMEOUT", 2.0)
        start = time.monotonic()
        conn = manager.connect(cfg)
        assert time.monotonic() - start < 5.5
        assert conn.error == "connection timed out"

        # Wait for the slow server's connection attempt to conclude
        # (ready is set on success and on failure alike).
        assert conn.ready.wait(60), "connection attempt never concluded"
        time.sleep(0.5)

        record = manager._connections.get(cfg.name)
        assert record is None or not (
            record.session is not None and record.error
        ), "poisoned record: live session marked failed"

        deadline = time.monotonic() + 30
        while not conn.task.done():
            assert time.monotonic() < deadline, (
                "timed-out connection's task leaked (never unwound)"
            )
            time.sleep(0.1)
        assert conn.session is None

        # Recovery: with an adequate timeout the same server connects
        # cleanly from scratch.
        monkeypatch.setattr(mcp_servers_module, "CONNECT_TIMEOUT", 45.0)
        fresh = manager.connect(cfg)
        assert fresh.error == ""
        assert fresh.session is not None
        assert manager.call_tool("slowsrv", "ping", {}) == "pong"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Finding #7: _bash_streaming timeout-vs-completion misreport


def test_bash_streaming_genuine_timeout_still_reported(tmp_path: Path) -> None:
    """A command that really overruns its timeout reports the timeout."""
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    out = tools.Bash("echo started; sleep 30", "timeout case", timeout_seconds=1)
    assert out == "Error: Command execution timeout"


def test_bash_streaming_completed_command_returns_output(tmp_path: Path) -> None:
    """A completed streaming command returns its full output, not a timeout."""
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    out = tools.Bash("echo done-quickly", "fast case", timeout_seconds=10)
    assert "done-quickly" in out
    assert "timeout" not in out.lower()
    assert "done-quickly\n" in lines


# ---------------------------------------------------------------------------
# Finding #14: Write newline translation


def test_write_round_trip_is_byte_identical(tmp_path: Path) -> None:
    """``Write`` stores *content* verbatim — no newline translation.

    Locks the ``newline=""`` contract shared with ``Edit``: the bytes
    on disk equal the content exactly (on Windows the pre-fix default
    translation would CRLF-ify every LF; on POSIX the translation is an
    identity, so this test guards the contract rather than reproducing
    a POSIX-visible failure).
    """
    tools = UsefulTools(work_dir=str(tmp_path))
    target = tmp_path / "mixed.txt"
    content = "unix\nwindows\r\nold-mac\rend"
    out = tools.Write(str(target), content)
    assert out.startswith("Successfully wrote")
    assert target.read_bytes() == content.encode()
