# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave3-Fixer-2's bug fixes.

Covers findings A-F2, A-F3, A-F5 of ``tmp/w3-findings-A.md`` and C-2,
C-4 of ``tmp/w3-findings-C.md``:

* A-F2 — ``_stop_event_writer`` mutated ``_event_writer_thread`` /
  ``_event_writer_stop`` without ``_event_writer_lock``, racing
  ``_start_event_writer``: the stop path could clobber a freshly
  published live writer (and clear its stop flag), leaving an orphan
  writer running forever alongside a newly spawned one — two live
  writers draining the same FIFO queue break per-task event ordering.
* A-F3 — ``_on_page_crash`` unconditionally nulled ``self._page``, so a
  BACKGROUND tab crash cleared the healthy active page and forced a
  full browser teardown + relaunch, discarding every open tab.
* A-F5 — ``MCPManager.connect``/``call_tool`` ignored ``_shut_down``:
  a stale manager reference scheduled onto the stopped loop and burnt
  the full ``CONNECT_TIMEOUT`` (60s); the post-timeout ``conn.error``
  stamp also raced outside ``self._lock``.
* C-2 — ``UsefulTools.Read`` validated ``start_line`` but not
  ``max_lines``: ``max_lines=0`` returned only a truncation marker and
  negative values produced wrong slices/counts.
* C-4 — ``sorcar mcp add`` silently ignored extra tokens after the URL
  for http/sse, ``--env`` for http/sse, and ``--header`` for stdio.

No mocks, patches, or fakes: real SQLite databases and threads, a real
headless Chromium via Playwright, a real (shut down) MCP manager event
loop, and real CLI entry-point invocations are used throughout.
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.useful_tools import UsefulTools
from kiss.ui.cli.mcp_cli import run_mcp_cli

# ---------------------------------------------------------------------------
# A-F2: _stop_event_writer racing _start_event_writer
# ---------------------------------------------------------------------------


def _redirect(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _live_writer_count() -> int:
    return sum(
        1
        for t in threading.enumerate()
        if t.name == "kiss-event-writer" and t.is_alive()
    )


class TestStopStartEventWriterRace:
    """Concurrent start/stop cycles must never leave two live writers."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        th._stop_event_writer()

    def teardown_method(self) -> None:
        th._stop_event_writer()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stop_start_stress_single_writer_no_orphans(self) -> None:
        task_id, _ = th._add_task("wave3 stop/start writer race stress")
        errors: list[BaseException] = []
        stop_all = threading.Event()
        max_seen = [0]

        def producer(n: int) -> None:
            try:
                for i in range(40):
                    th._queue_chat_event(
                        {"type": "note", "text": f"p{n}-{i}"}, task_id,
                    )
                    time.sleep(random.random() * 0.005)
                th._flush_chat_events()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def stopper() -> None:
            try:
                while not stop_all.is_set():
                    th._stop_event_writer()
                    time.sleep(random.random() * 0.005)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def starter() -> None:
            try:
                while not stop_all.is_set():
                    th._start_event_writer()
                    time.sleep(random.random() * 0.003)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def sampler() -> None:
            while not stop_all.is_set():
                max_seen[0] = max(max_seen[0], _live_writer_count())
                time.sleep(0.001)

        threads = [
            threading.Thread(target=producer, args=(n,)) for n in range(4)
        ]
        threads += [
            threading.Thread(target=stopper),
            threading.Thread(target=starter),
        ]
        sampler_t = threading.Thread(target=sampler, daemon=True)
        sampler_t.start()
        producers = threads[:4]
        for t in threads:
            t.start()
        for t in producers:
            t.join(timeout=60)
        stop_all.set()
        for t in threads[4:]:
            t.join(timeout=60)
        sampler_t.join(timeout=5)

        assert not errors, errors
        assert max_seen[0] <= 1, (
            f"{max_seen[0]} concurrent kiss-event-writer threads observed"
        )

        # Final stop must leave zero live writers: a clobbered-but-live
        # orphan (its reference nulled and its stop flag cleared by the
        # racing stop path) would survive here forever pre-fix.
        th._flush_chat_events()
        th._stop_event_writer()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _live_writer_count() > 0:
            time.sleep(0.01)
        assert _live_writer_count() == 0, "orphan event writer left running"

        chat = th._load_chat_events_by_task_id(task_id)
        assert chat is not None
        events = cast(list[dict[str, object]], chat["events"])
        texts = [e.get("text") for e in events]
        assert len(texts) == 4 * 40, f"expected 160 events, got {len(texts)}"
        assert len(set(texts)) == 4 * 40, "duplicate events persisted"


# ---------------------------------------------------------------------------
# A-F3: background-tab crash must not clear the healthy active page
# ---------------------------------------------------------------------------


def test_background_tab_crash_keeps_active_page() -> None:
    """A renderer crash on a BACKGROUND tab must not tear down the session.

    Pre-fix ``_on_page_crash`` unconditionally nulled ``self._page``, so
    the crash of the old (background) tab cleared the healthy active
    page; ``_ensure_browser`` then treated ``_page is None`` as a
    renderer crash of the active page and did a full teardown +
    relaunch, replacing the context and closing every open tab.
    """
    pytest.importorskip("playwright.sync_api")
    tool = WebUseTool(headless=True, user_data_dir=None)
    try:
        first = tool.go_to_url("data:text/html,<title>one</title><h1>one</h1>")
        assert not first.startswith("Error"), first
        page_a = tool._page
        assert page_a is not None
        ctx = tool._context
        assert ctx is not None

        page2 = ctx.new_page()
        page2.goto("data:text/html,<title>two</title><h1>two</h1>")
        switched = tool.go_to_url("tab:1")
        assert not switched.startswith("Error"), switched
        assert tool._page is page2

        # Crash the BACKGROUND tab's renderer for real.  The navigation
        # itself raises once the renderer dies.
        try:
            page_a.goto("chrome://crash", timeout=10000)
        except Exception:
            pass

        # The sync Playwright API delivers the pending crash event only
        # during later Playwright calls; pump a few so the (buggy)
        # handler gets its chance to null the active page.
        out = ""
        for _ in range(5):
            out = tool.go_to_url(
                "data:text/html,<title>ok</title><h1>ok</h1>"
            )
            time.sleep(0.2)

        assert tool._context is ctx, (
            "background-tab crash triggered a full browser teardown"
        )
        assert not page2.is_closed(), "healthy active tab was closed"
        assert tool._page is page2, "active page reference was dropped"
        assert not out.startswith("Error"), out
    finally:
        tool.close()


# ---------------------------------------------------------------------------
# A-F5: connect()/call_tool() on a shut-down manager fail fast
# ---------------------------------------------------------------------------


def test_connect_after_shutdown_fails_fast() -> None:
    """``connect`` on a shut-down manager must not burn CONNECT_TIMEOUT.

    Pre-fix a stale manager reference scheduled the connection
    coroutine onto the stopped loop (it never runs) and blocked the
    full 60s ``CONNECT_TIMEOUT`` before returning a poisoned record.
    """
    manager = MCPManager()
    manager.shutdown()
    config = MCPServerConfig(
        name="wave3-shutdown-srv",
        transport="stdio",
        command=sys.executable,
        args=("-c", "pass"),
    )
    start = time.monotonic()
    conn = manager.connect(config)
    elapsed = time.monotonic() - start
    assert elapsed < 10, f"connect blocked {elapsed:.1f}s on a dead loop"
    assert "shut down" in conn.error, conn.error
    assert conn.ready.is_set()


def test_call_tool_after_shutdown_fails_fast() -> None:
    """``call_tool`` on a shut-down manager returns an error immediately."""
    manager = MCPManager()
    manager.shutdown()
    start = time.monotonic()
    out = manager.call_tool("wave3-shutdown-srv", "add", {"a": 1, "b": 2})
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"call_tool blocked {elapsed:.1f}s on a dead loop"
    assert out.startswith("Error"), out
    assert "shut down" in out, out


# ---------------------------------------------------------------------------
# C-2: Read must validate max_lines
# ---------------------------------------------------------------------------


class TestReadMaxLinesValidation:
    """``Read`` rejects ``max_lines < 1`` with a clear error."""

    def _tools(self, tmp_path: Path) -> tuple[UsefulTools, str]:
        path = tmp_path / "five.txt"
        path.write_text("l1\nl2\nl3\nl4\nl5\n")
        return UsefulTools(work_dir=str(tmp_path)), str(path)

    def test_zero_max_lines_rejected(self, tmp_path: Path) -> None:
        tools, path = self._tools(tmp_path)
        out = tools.Read(path, max_lines=0)
        assert out.startswith("Error: max_lines must be >= 1"), out

    def test_negative_max_lines_rejected(self, tmp_path: Path) -> None:
        tools, path = self._tools(tmp_path)
        out = tools.Read(path, max_lines=-3)
        assert out.startswith("Error: max_lines must be >= 1"), out

    def test_positive_max_lines_still_truncates(self, tmp_path: Path) -> None:
        tools, path = self._tools(tmp_path)
        out = tools.Read(path, max_lines=2)
        assert out.startswith("l1\nl2\n"), out
        assert "[truncated: 3 more lines]" in out


# ---------------------------------------------------------------------------
# C-4: `sorcar mcp add` rejects silently-ignored options/arguments
# ---------------------------------------------------------------------------


class TestMcpAddRejectsIgnoredOptions:
    """``mcp add`` errors out instead of silently dropping input."""

    def test_extra_tokens_after_url_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            run_mcp_cli(
                [
                    "add", "--transport", "http", "--scope", "project",
                    "srv", "https://x.example", "extra", "tokens",
                ],
                str(tmp_path),
            )
        assert "extra" in str(exc.value).lower(), exc.value

    def test_env_rejected_for_http(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            run_mcp_cli(
                [
                    "add", "--transport", "http", "--scope", "project",
                    "--env", "K=V", "srv", "https://x.example",
                ],
                str(tmp_path),
            )
        assert "--env" in str(exc.value), exc.value

    def test_header_rejected_for_stdio(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            run_mcp_cli(
                [
                    "add", "--scope", "project", "--header", "X: y",
                    "srv", sys.executable, "-c", "pass",
                ],
                str(tmp_path),
            )
        assert "--header" in str(exc.value), exc.value

    def test_valid_http_add_still_works(self, tmp_path: Path) -> None:
        rc = run_mcp_cli(
            [
                "add", "--transport", "http", "--scope", "project",
                "--header", "X-Key: abc", "srv", "https://x.example",
            ],
            str(tmp_path),
        )
        assert rc == 0

    def test_valid_stdio_add_still_works(self, tmp_path: Path) -> None:
        rc = run_mcp_cli(
            [
                "add", "--scope", "project", "--env", "K=V",
                "srv2", sys.executable, "-c", "pass",
            ],
            str(tmp_path),
        )
        assert rc == 0
