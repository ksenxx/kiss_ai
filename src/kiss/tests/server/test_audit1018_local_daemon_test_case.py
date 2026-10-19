# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the shared ``LocalDaemonTestCase`` base is strict about its teardown.

Audit 2026-10-18, finding 1.1: the ``_ApiLaunchBase`` of
``test_kiss_web_launch.py`` joined each task thread with a timeout and
then dropped the registry, closed the history DB and removed the tmpdir
whether or not the join had succeeded, so a live task thread leaked
into the next test without a word.  Both "daemon on a loop thread"
bases now share :class:`LocalDaemonTestCase`, which raises instead and
(review of that fix) keeps the straggler's home, history.db and globals
in place rather than restoring them under a thread that still uses them.

Each test here builds a real subclass, runs it under ``unittest`` (a
real daemon over a real local endpoint, a real thread parked in the
real registry) and inspects the result.
"""

from __future__ import annotations

import io
import json
import threading
import unittest
from pathlib import Path

from kiss.agents.sorcar import persistence
from kiss.server import agent_state
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.parallel_agent_harness import LocalDaemonTestCase

# Written by the test methods below so the outer test can inspect the
# state the inner one saw while its daemon was up.
_SEEN: dict[str, object] = {}


def _finalize(release: threading.Event) -> None:
    """Thread target: a task finalizer that outlives the test's teardown.

    It records the history.db path it would write to before and after it
    is *released* by the outer test.  Registered with
    ``stop_acknowledged=True`` (a runner persisting its stopped row), it
    ignores the cooperative stop event and the daemon's shutdown sweep
    refuses to inject ``KeyboardInterrupt`` into it, so it is still alive
    when the harness would close its DB and remove its home.
    """
    _SEEN["db_before_release"] = persistence._DB_PATH
    release.wait(60)
    _SEEN["db_after_release"] = persistence._DB_PATH


def _run(case: unittest.TestCase) -> unittest.TestResult:
    """Run *case* under ``unittest`` with a silent stream and return the result."""
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(
        unittest.TestSuite([case])
    )


def test_daemon_base_serves_a_local_client_and_tears_down_cleanly() -> None:
    """A subclass gets a reachable daemon; every cleanup runs and leaves nothing."""

    class Case(LocalDaemonTestCase):
        home_prefix = "kiss-audit1018-base-"

        def setUp(self) -> None:
            super().setUp()
            self.start_daemon()

        def test_ping(self) -> None:
            _SEEN["endpoint_exists"] = Path(self.endpoint_file).exists()
            _SEEN["loop"] = self.loop
            _SEEN["loop_thread"] = self.loop_thread

            async def _ping() -> dict[str, object]:
                reader, writer = await open_local_connection(self.server)
                writer.write(b'{"type": "ping"}\n')
                await writer.drain()
                reply: dict[str, object] = json.loads(await reader.readline())
                writer.close()
                return reply

            _SEEN["reply"] = self.run_on_loop(_ping(), timeout=10)

    result = _run(Case("test_ping"))
    assert result.wasSuccessful(), result.errors + result.failures
    assert _SEEN["endpoint_exists"] is True
    assert isinstance(_SEEN["reply"], dict) and _SEEN["reply"].get("type") == "pong", _SEEN
    loop_thread = _SEEN["loop_thread"]
    assert isinstance(loop_thread, threading.Thread) and not loop_thread.is_alive()
    assert _SEEN["loop"].is_closed()  # type: ignore[attr-defined]
    assert agent_state.snapshot() == []


def test_live_task_thread_at_teardown_is_an_error_not_a_leak() -> None:
    """A task thread still running after the join timeout fails the test and keeps its home.

    The old ``_ApiLaunchBase`` passed this scenario silently and went on
    to close the thread's history DB; the first shared base reported it
    but still deleted the home and restored the persistence path under
    the live thread.  Now the daemon cleanups (stop, loop close) still
    run, the registry is emptied, and the home, its history.db and the
    persistence globals are left exactly as the straggler sees them.
    """
    release = threading.Event()
    worker = threading.Thread(target=_finalize, args=(release,), daemon=True)
    db_path_before = persistence._DB_PATH

    class Case(LocalDaemonTestCase):
        home_prefix = "kiss-audit1018-leak-"
        task_join_timeout = 0.5

        def setUp(self) -> None:
            super().setUp()
            self.start_daemon()

        def test_park_a_live_task(self) -> None:
            state = agent_state.AgentState(
                "stuck-task",
                tab_id="tab-stuck",
                stop_event=threading.Event(),
                task_thread=worker,
            )
            state.stop_acknowledged = True
            agent_state.register(state)
            worker.start()

    class NextCase(LocalDaemonTestCase):
        home_prefix = "kiss-audit1018-next-"
        task_join_timeout = 0.5
        ran = False

        def test_nothing(self) -> None:
            NextCase.ran = True

    case = Case("test_park_a_live_task")
    try:
        result = _run(case)
        # Observed before the worker is released: the real case, a
        # worker still alive once every cleanup has run.
        alive_after_teardown = worker.is_alive()
        home_after_teardown = Path(case.tmpdir).exists()
        db_path_after_teardown = persistence._DB_PATH
        registry_after_teardown = agent_state.snapshot()
        # The persistence path is process-wide: a later case must not
        # redirect it under the straggler, so it refuses to start.
        blocked = _run(NextCase("test_nothing"))
        db_path_while_blocked = persistence._DB_PATH
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(timeout=5)
    assert not worker.is_alive()
    assert NextCase.ran is False
    assert len(blocked.failures) == 1, (blocked.failures, blocked.errors)
    assert "task thread(s) of an earlier test are still alive" in blocked.failures[0][1]
    assert db_path_while_blocked == case.home.kiss_home / "history.db"
    # Once the straggler has ended the next case runs as usual.
    after = _run(NextCase("test_nothing"))
    assert after.wasSuccessful(), after.failures + after.errors
    assert NextCase.ran is True
    assert alive_after_teardown is True
    # unittest files an ``AssertionError`` raised by a cleanup as a failure.
    assert result.errors == [], result.errors
    assert len(result.failures) == 1, result.failures
    traceback_text = result.failures[0][1]
    assert "task thread of tab 'tab-stuck' still alive" in traceback_text, traceback_text
    assert f"its home {case.tmpdir} is left in place" in traceback_text, traceback_text
    # The failed join did not stop the daemon cleanups ...
    assert not case.loop_thread.is_alive()
    assert not Path(case.endpoint_file).exists()
    assert registry_after_teardown == []
    # ... but the destructive ones were skipped: the home survived and
    # the persistence path still named the straggler's history.db, before
    # and after the join failed, until the worker itself ended.
    inner_db = case.home.kiss_home / "history.db"
    assert home_after_teardown is True
    assert db_path_after_teardown == inner_db, (db_path_after_teardown, inner_db)
    assert _SEEN["db_before_release"] == inner_db
    assert _SEEN["db_after_release"] == inner_db
    # Now that the worker is gone the home can be released the normal way.
    persistence._close_db()
    case.home.cleanup()
    assert not Path(case.tmpdir).exists()
    assert persistence._DB_PATH == db_path_before
