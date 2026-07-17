# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Fixer-5's MCP / web / skills bug fixes.

Covers findings #1, #4, #5, #6, #9, #11 of ``tmp/findings-4.md``:

* #1 — a stop request racing the connection task's start is never lost
  (``_Connection.stop`` is created eagerly), so ``disconnect_all``
  actually terminates a just-scheduled stdio server task.
* #4 — ``MCPManager.shutdown`` resets the singleton, so a later
  ``MCPManager.instance().connect(...)`` gets a live loop instead of
  blocking ``CONNECT_TIMEOUT`` on a dead one; ``shutdown`` is
  idempotent.
* #5 — the accounts.google.com abort route is installed on
  non-persistent contexts too.
* #6 — pages adopted via ``go_to_url("tab:N")`` get the crash handler,
  so a renderer crash on such a page is recovered from.
* #9 — ``_OAuthCallbackServer.close()`` unblocks a pending ``wait()``,
  so the ``sorcar mcp auth`` event-loop teardown is not stalled by an
  executor thread sitting out ``AUTH_TIMEOUT``.
* #11 — the skill name is XML-escaped in ``load_skill_content`` output
  (consistent with the catalog).

No mocks, patches, or fakes: a real FastMCP stdio server subprocess, a
real headless Chromium via Playwright, real HTTP servers and threads,
and real skill directories are used throughout.
"""

from __future__ import annotations

import asyncio
import os
import pty
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.agents.sorcar.skills import discover_skills, load_skill_content
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.ui.cli.mcp_cli import _OAuthCallbackServer

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


def _stdio_config(tmp_path: Path, name: str = "testsrv") -> MCPServerConfig:
    """Return a stdio server config running a real FastMCP test server."""
    script = tmp_path / "testsrv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
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
    technique as ``test_sorcar_mcp.py``).
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
# Finding #1: stop signal racing the connection task's start


def test_disconnect_racing_connection_start_stops_task(
    tmp_path: Path, real_stdin: None,
) -> None:
    """A stop arriving before the connection coroutine runs is honoured.

    The manager loop is stalled so ``connect()`` schedules the
    connection task but the coroutine cannot start; ``disconnect_all``
    is called inside that window.  The stop must not be dropped: the
    task has to finish (unwinding its transport and terminating the
    server subprocess) instead of parking on ``stop.wait()`` forever.
    """
    manager = MCPManager()
    try:
        cfg = _stdio_config(tmp_path)
        # Stall the loop thread so the just-scheduled coroutine cannot
        # start before disconnect_all runs (deterministic race window).
        manager._loop.call_soon_threadsafe(time.sleep, 1.0)

        connect_thread = threading.Thread(
            target=manager.connect, args=(cfg,), daemon=True,
        )
        connect_thread.start()
        # Wait for connect() to register the connection record.
        deadline = time.monotonic() + 5
        while cfg.name not in manager._connections:
            assert time.monotonic() < deadline, "connect never registered"
            time.sleep(0.01)
        conn = manager._connections[cfg.name]

        start = time.monotonic()
        manager.disconnect_all()
        elapsed = time.monotonic() - start
        connect_thread.join(timeout=30)
        assert not connect_thread.is_alive()

        # Pre-fix: stop was None when disconnect_all checked it, the
        # signal was dropped, the task parked forever, and
        # disconnect_all swallowed a 10 s TimeoutError.
        deadline = time.monotonic() + 15
        while not conn.task.done():
            assert time.monotonic() < deadline, (
                "connection task leaked: stop signal was dropped"
            )
            time.sleep(0.05)
        assert conn.session is None
        assert elapsed < 9.5, "disconnect_all sat out its 10s task timeout"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Finding #4: shutdown poisons the singleton


def test_shutdown_resets_singleton_and_reconnect_is_prompt(
    tmp_path: Path, real_stdin: None,
) -> None:
    """After ``shutdown``, ``instance()`` yields a fresh, working manager.

    Pre-fix the singleton kept pointing at the dead manager, so the
    ``connect`` below would block ``CONNECT_TIMEOUT`` (60 s) and report
    "connection timed out" instead of connecting.
    """
    m1 = MCPManager.instance()
    m1.shutdown()
    # Idempotent: a second shutdown is an immediate no-op.
    start = time.monotonic()
    m1.shutdown()
    assert time.monotonic() - start < 1

    m2 = MCPManager.instance()
    assert m2 is not m1
    try:
        start = time.monotonic()
        conn = m2.connect(_stdio_config(tmp_path))
        elapsed = time.monotonic() - start
        assert conn.error == ""
        assert conn.session is not None
        assert elapsed < 30, f"connect took {elapsed:.1f}s (dead-loop manager?)"
        out = m2.call_tool("testsrv", "add", {"a": 2, "b": 3})
        assert out == "5"
    finally:
        m2.shutdown()


# ---------------------------------------------------------------------------
# Findings #5 and #6: web tool (real headless Chromium)


def test_accounts_google_blocked_in_non_persistent_context() -> None:
    """The accounts.google.com abort route also covers user_data_dir=None.

    Pre-fix the route was installed only on the persistent-context
    branch, so a non-persistent context loaded the page successfully
    and this navigation returned an accessibility tree, not an error.
    """
    tool = WebUseTool(headless=True, user_data_dir=None)
    try:
        result = tool.go_to_url("https://accounts.google.com/")
        assert result.startswith("Error navigating to"), result
    finally:
        tool.close()


def test_renderer_crash_on_tab_switched_page_recovers() -> None:
    """A crash on a page adopted via ``tab:N`` is detected and recovered.

    Pre-fix the ``tab:N`` branch never registered the crash handler, so
    after the crash ``_page`` still pointed at a crashed-but-open page
    (``_is_alive`` is True for it) and every later call failed forever.
    """
    tool = WebUseTool(headless=True, user_data_dir=None)
    try:
        first = tool.go_to_url("data:text/html,<title>one</title><h1>one</h1>")
        assert not first.startswith("Error"), first
        # Open a second real tab in the same context and switch to it
        # through the public tab-switch path under test.
        page2 = tool._context.new_page()
        page2.goto("data:text/html,<title>two</title><h1>two</h1>")
        switched = tool.go_to_url("tab:1")
        assert not switched.startswith("Error"), switched
        assert tool._page is page2

        # Crash the adopted tab's renderer for real.
        crash = tool.go_to_url("chrome://crash")
        assert crash.startswith("Error navigating to"), crash

        # The sync Playwright API only delivers the pending crash event
        # during a later Playwright call, so the first retry may still
        # error while the handler fires; with the fix the tool then
        # tears down the dead page and recovers within a few calls.
        # Pre-fix the handler never fires (`_page` keeps pointing at the
        # crashed-but-open page, which `_is_alive` reports live), so
        # every retry fails with "Page crashed" forever.
        out = ""
        recovered = False
        for _ in range(5):
            out = tool.go_to_url("data:text/html,<title>ok</title><h1>ok</h1>")
            if not out.startswith("Error"):
                recovered = True
                break
            time.sleep(0.2)
        assert recovered, f"tool never recovered from adopted-tab crash: {out}"
    finally:
        tool.close()


# ---------------------------------------------------------------------------
# Finding #9: auth teardown stalled by a blocked callback wait


def test_oauth_callback_close_unblocks_pending_wait() -> None:
    """``close()`` wakes a thread blocked in ``wait()`` promptly."""
    callback = _OAuthCallbackServer()
    errors: list[BaseException] = []

    def waiter() -> None:
        try:
            callback.wait(300)
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            errors.append(exc)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    time.sleep(0.2)
    callback.close()
    thread.join(timeout=10)
    assert not thread.is_alive(), "wait() still blocked after close()"
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "no code returned" in str(errors[0])


def test_auth_style_event_loop_teardown_not_stalled() -> None:
    """A failed connect does not stall ``asyncio.run`` on the executor.

    Reproduces the exact ``_cmd_auth`` structure: the callback wait
    blocks on the default executor while the connect coroutine fails.
    ``asyncio.run``'s teardown joins the executor; pre-fix nothing set
    the callback's event, so the CLI hung for up to ``AUTH_TIMEOUT``
    (300 s).  With the fix (close() inside the coroutine's ``finally``
    setting ``_done``), teardown completes in seconds.
    """
    callback = _OAuthCallbackServer()

    async def failing_flow() -> None:
        loop = asyncio.get_running_loop()
        # The OAuth provider's callback_handler: a blocked executor wait.
        pending = loop.run_in_executor(None, callback.wait, 300.0)
        await asyncio.sleep(0.2)
        try:
            raise RuntimeError("connection failed before the redirect")
        finally:
            callback.close()
            # Keep a reference alive like the real flow does; never
            # awaited because the flow already failed.
            del pending

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="connection failed"):
        asyncio.run(failing_flow())
    elapsed = time.monotonic() - start
    assert elapsed < 30, f"event-loop teardown stalled {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Finding #11: skill name escaping in load_skill_content


def test_skill_name_xml_escaped_in_content(tmp_path: Path) -> None:
    """A skill whose directory name needs escaping renders valid XML."""
    weird = 'a"b<c&d'
    skill_dir = tmp_path / "project" / ".kiss" / "skills" / weird
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Weird-named test skill.\n---\nUse it wisely.\n",
        encoding="utf-8",
    )
    skills = discover_skills(str(tmp_path / "project"))
    assert weird in skills
    content = load_skill_content(skills[weird])
    assert '<skill_content name="a&quot;b&lt;c&amp;d">' in content
    assert "Use it wisely." in content
