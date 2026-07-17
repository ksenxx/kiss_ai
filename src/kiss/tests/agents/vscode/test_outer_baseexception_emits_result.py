# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Outer ``BaseException`` path must broadcast a ``type: result`` event.

Symptom: the user runs a long task that completes meaningful work
(e.g. the agent finishes a ``finish(is_continue=True, success=False,
summary="...")`` paused continuation and persists a deliverable
file), but the VS Code chat panel renders the task outcome as
``"(no result)"`` even though a real result summary was computed and
persisted in ``task_history``.

Reproduction: a ``BaseException`` that is NOT ``Exception`` and NOT
``KeyboardInterrupt`` (e.g. ``SystemExit``, ``GeneratorExit``,
``asyncio.CancelledError`` on 3.11+, or any custom ``BaseException``
subclass propagated from the streaming context manager) escapes
``tab.agent.run``.  The inner per-subtask ``try/except`` handlers in
:meth:`_TaskRunnerMixin._run_task_inner` match only
``KeyboardInterrupt`` and ``Exception``, so the unwind falls through
to the outer ``except BaseException as _outer_exc:`` block.  That
block now correctly recomputes ``result_summary`` and
``task_end_event`` (fixed by ``test_agent_failed_abruptly.py``), but
it never broadcasts a ``type: result`` event — only the cleanup
``finally`` broadcasts ``task_end_event`` (``task_stopped`` /
``task_error`` / ``task_interrupted``), which is NOT a ``result``
event.

Consequence: the webview (``main.js:1939``) reads the result text
from a ``type: result`` event and falls back to ``"(no result)"``
when none was received, masking the real summary persisted in
``task_history.result``.

This test drives ``_run_task`` end-to-end with a stub agent that
raises a non-``Exception`` ``BaseException`` and asserts that the
broadcast captured includes a single ``type: result`` event whose
``text`` matches the recomputed summary (NOT the literal
``"(no result)"`` fallback string and NOT empty).
"""

from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


class _CustomBaseException(BaseException):
    """A ``BaseException`` subclass that is NOT ``Exception``.

    Mirrors ``SystemExit`` / ``asyncio.CancelledError`` (3.11+) /
    ``GeneratorExit`` semantics for the inner per-subtask handlers
    without aborting the test interpreter or the active asyncio loop.
    """


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.server import VSCodeServer

    return VSCodeServer()


def _drive_run_task_with_base_exception(
    exc_to_raise: BaseException,
    *,
    tab_id: str,
    work_dir: str,
) -> list[dict[str, Any]]:
    """Drive ``_run_task`` with a stub agent that raises *exc*; collect events.

    Returns the full ordered list of broadcasts captured by overriding
    ``server.printer.broadcast`` so the test can assert on the
    presence (or absence) of a ``type: result`` event for the outer-
    BaseException unwind.
    """
    server = _make_server()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(dict(event))

    server.printer.broadcast = capture  # type: ignore[assignment]

    tab = server._get_tab(tab_id)
    agent = WorktreeSorcarAgent("Sorcar VS Code")
    tab.agent = agent
    tab.chat_id = ""

    def fake_run(**kwargs: Any) -> None:
        agent.total_tokens_used = 4242
        agent.budget_used = 0.1234
        agent.step_count = 17
        agent._chat_id = agent._chat_id or "test-chat-id"
        task_id, _ = _persistence._add_task(
            kwargs.get("prompt_template", ""),
            chat_id=agent._chat_id,
            extra={
                "model": kwargs.get("model_name", ""),
                "work_dir": kwargs.get("work_dir", ""),
                "version": "test",
                "is_parallel": bool(kwargs.get("is_parallel", False)),
                "is_worktree": bool(kwargs.get("use_worktree", False)),
            },
        )
        agent._last_task_id = task_id
        raise exc_to_raise

    agent.run = fake_run  # type: ignore[assignment]

    tab.stop_event = threading.Event()
    tab.user_answer_queue = queue.Queue()

    task_thread = threading.Thread(
        target=server._run_task,
        args=({
            "type": "run",
            "prompt": "reproduce-no-result-on-outer-baseexception",
            "tabId": tab_id,
            "workDir": work_dir,
            "useParallel": False,
            "useWorktree": False,
            "autoCommit": False,
        },),
        daemon=True,
    )
    tab.task_thread = task_thread
    task_thread.start()
    task_thread.join(timeout=15)
    assert not task_thread.is_alive(), "task thread did not finish"
    # Allow late post-status broadcasts (none expected post-fix) to land.
    time.sleep(0.1)
    with lock:
        return list(events)


class TestOuterBaseExceptionEmitsResult(TestCase):
    """The outer-BaseException path must surface a ``type: result`` event."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="kiss-no-result-")
        self._work_dir = str(Path(self._tmpdir) / "wd")
        Path(self._work_dir).mkdir()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_custom_baseexception_emits_result_event(self) -> None:
        """A non-``Exception`` ``BaseException`` must produce a result event.

        Before the fix the outer ``except BaseException`` block
        recomputes ``result_summary`` but never broadcasts
        ``type: result``, so the VS Code webview displays
        ``"(no result)"`` instead of the recovered summary.
        """
        events = _drive_run_task_with_base_exception(
            _CustomBaseException("simulated async cancel"),
            tab_id="no-result-1",
            work_dir=self._work_dir,
        )
        result_events = [e for e in events if e.get("type") == "result"]
        assert result_events, (
            "outer BaseException path did not broadcast a `type: result` "
            "event; the VS Code webview will fall back to `(no result)` "
            f"in main.js. Broadcast types observed: "
            f"{[e.get('type') for e in events]!r}"
        )
        # The result text must carry the recovered summary, not the
        # webview fallback string and not the empty string.
        texts = [e.get("text", "") for e in result_events]
        assert all(texts), (
            f"`type: result` event has empty `text`: {result_events!r}"
        )
        assert all(t != "(no result)" for t in texts), (
            f"`type: result` event carries the literal fallback: "
            f"{result_events!r}"
        )
        # The recovered text should reference the exception so the user
        # gets diagnostic context (matches the format chosen by the
        # outer ``except BaseException`` for non-``KeyboardInterrupt``
        # cases: "Task failed: <type>: <msg>").
        assert any("_CustomBaseException" in t for t in texts), (
            f"expected exception type in result `text`, got {texts!r}"
        )
        # The event must declare the failure outcome (``success: False``)
        # so the webview renders the failure styling.
        assert all(e.get("success") is False for e in result_events), (
            f"result event missing success=False: {result_events!r}"
        )

    def test_system_exit_emits_result_event(self) -> None:
        """``SystemExit`` from the agent run is also a ``BaseException``.

        The same broadcast guarantee applies; the recovered text must
        be non-empty and not the ``(no result)`` fallback.
        """
        events = _drive_run_task_with_base_exception(
            SystemExit("simulated stray sys.exit"),
            tab_id="no-result-2",
            work_dir=self._work_dir,
        )
        result_events = [e for e in events if e.get("type") == "result"]
        assert result_events, (
            "SystemExit path did not broadcast a `type: result` event; "
            f"observed types: {[e.get('type') for e in events]!r}"
        )
        for ev in result_events:
            text = ev.get("text", "")
            assert text and text != "(no result)", (
                f"SystemExit result event has bad text: {ev!r}"
            )
