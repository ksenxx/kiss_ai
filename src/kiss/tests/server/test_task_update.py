# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the task-info panel's task update
(:mod:`kiss.server.task_update` and the ``getTaskUpdate`` command).

* :class:`TaskUpdateRunner` is driven through ``poll`` with a scripted
  agent runner whose completion the test controls, so the one-minute
  wait after the task's start, the lazy 10-minute schedule, the
  refresh path, the single-run-in-flight rule, the failure path and
  pruning are all observed from the outside.
* :func:`run_task_update_sea` runs a real :class:`ChatSorcarAgent`
  child against the scripted local chat-completions server, proving
  the child is the ``/ask`` agent (its system prompt, playbook and
  ``task_context`` tool), lands in the parent's chat as its sub-agent
  and has its spend charged to the parent.
* The time-left estimate is followed through that real child, from the
  answer it passes to ``finish`` to the panel's payload: the sentence
  cut out of the answer however it is written, the three finish times
  and the sooner refresh they bring.
* ``getTaskUpdate`` is exercised over a live WSS connection and over
  the local channel of a real :class:`RemoteAccessServer`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import pytest
from websockets.asyncio.client import connect

from kiss.agents.seas.ask import ask_sea
from kiss.agents.seas.task_update import task_update_sea
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _get_db,
    _rw_lock,
)
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server import agent_state, task_update
from kiss.server.task_update import (
    TaskUpdateRunner,
    mark_legacy_updates_as_side_channels,
    run_task_update_sea,
)
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.conftest import find_free_port, no_verify_ssl
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    StandInModelServer,
    finish_response,
    tool_call_response,
)


class _ScriptedSea:
    """A stand-in for :func:`run_task_update_sea` the test releases by hand.

    Each call records ``(parent_agent, task_id)``, blocks until
    :meth:`release` (at most 10 s), then returns the queued outcome or
    raises it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []
        self.outcomes: list[tuple[str, float] | BaseException] = []
        self._gate = threading.Event()

    def __call__(self, parent_agent: Any, task_id: str) -> tuple[str, float]:
        self.calls.append((parent_agent, task_id))
        self._gate.wait(10)
        self._gate.clear()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def release(self, outcome: tuple[str, float] | BaseException) -> None:
        """Let the blocked call finish with *outcome*."""
        self.outcomes.append(outcome)
        self._gate.set()


def _started_task(age_s: float, **extra: Any) -> str:
    """Persist a task row that started *age_s* seconds ago; return its id.

    The task runner allocates every run's row with its own ``startTs``;
    the runner's first-update delay counts from it.
    """
    task_id, _chat_id = _add_task(
        f"task started {age_s}s ago", chat_id="",
        extra={"startTs": int((time.time() - age_s) * 1000), **extra},
    )
    return task_id


def _wait_until(predicate: Any, timeout: float = 10.0) -> None:
    """Block until *predicate* is true (polling), or fail after *timeout*."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


class TestTaskUpdateRunner(unittest.TestCase):
    """The lazy schedule behind the panel's poll.

    Not covered: ``poll`` marks an update ``running`` only after its
    worker thread has started, so a ``Thread.start()`` that raises
    (``RuntimeError: can't start new thread`` under thread exhaustion)
    cannot pin the update on "running" for good.  That branch needs the
    interpreter to fail creating a thread, which no end-to-end test can
    provoke without a test double.
    """

    def test_first_run_waits_a_minute_after_the_task_started(self) -> None:
        """A young task is not asked about yet; the panel learns when it will be."""
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_young = _started_task(10.0)

        young = runner.poll(task_young, parent)
        self.assertFalse(young.running)
        self.assertEqual(young.text, "")
        self.assertAlmostEqual(young.started_at, time.time() - 10.0, places=0)
        self.assertAlmostEqual(
            young.due_at, young.started_at + task_update.FIRST_UPDATE_DELAY_S,
        )
        payload = young.payload()
        self.assertIs(payload["exists"], True)
        self.assertEqual(payload["updatedAt"], 0)
        self.assertEqual(payload["dueAt"], int(young.due_at * 1000))
        self.assertEqual(payload["sig"], "0.000:0")
        time.sleep(0.05)
        self.assertFalse(runner.poll(task_young, parent).running)
        self.assertEqual(sea.calls, [])

        # The refresh button does not wait.
        forced = runner.poll(task_young, parent, force=True)
        self.assertTrue(forced.running)
        _wait_until(lambda: len(sea.calls) == 1)
        sea.release(("early", 0.01))
        _wait_until(lambda: not runner.poll(task_young, parent).running)
        done = runner.poll(task_young, parent)
        self.assertEqual(done.text, "early")
        # Once a run has ended the regular interval takes over.
        self.assertAlmostEqual(
            done.due_at, done.finished_at + task_update.UPDATE_INTERVAL_S,
        )

        # A legacy row without ``start_ts`` is dated by its insertion
        # time; a task without a row counts from its first poll.
        legacy = _started_task(0.0, startTs=0)
        before = time.time()
        for tid in (legacy, "task-without-a-row"):
            upd = runner.poll(tid, parent)
            self.assertFalse(upd.running)
            self.assertGreaterEqual(upd.started_at, before - 1.0)
            self.assertLessEqual(upd.started_at, time.time())
        self.assertEqual(len(sea.calls), 1)

    def test_first_poll_runs_once_and_fresh_reports_wait(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_a = _started_task(120.0)

        first = runner.poll(task_a, parent)
        self.assertTrue(first.running)
        self.assertEqual(first.text, "")
        self.assertEqual(first.finished_at, 0.0)
        self.assertEqual(first.sig, "0.000:1")
        self.assertEqual(
            first.payload(),
            {
                "exists": True, "content": "", "error": "", "running": True,
                "cost": 0.0, "updatedAt": 0,
                "dueAt": int(first.due_at * 1000),
                # No answer yet, so no estimate of the time left.
                "etaAt": 0, "etaLowAt": 0, "etaHighAt": 0,
                "sig": "0.000:1",
            },
        )
        _wait_until(lambda: len(sea.calls) == 1)
        self.assertEqual(sea.calls[0], (parent, task_a))

        # A poll while the run is in flight — even a forced one — never
        # doubles the run.
        again = runner.poll(task_a, parent, force=True)
        self.assertTrue(again.running)
        self.assertEqual(len(sea.calls), 1)

        sea.release(("<h4>Report</h4>", 0.25))
        _wait_until(lambda: not runner.poll(task_a, parent).running)
        done = runner.poll(task_a, parent)
        self.assertEqual(done.text, "<h4>Report</h4>")
        self.assertEqual(done.cost, 0.25)
        self.assertEqual(done.error, "")
        self.assertGreater(done.finished_at, 0.0)
        self.assertEqual(done.sig, f"{done.finished_at:.3f}:0")
        self.assertEqual(done.payload()["updatedAt"], int(done.finished_at * 1000))
        # A fresh report is served from the cache: no new run.
        self.assertEqual(len(sea.calls), 1)

    def test_refresh_reruns_and_a_failure_keeps_the_previous_report(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_b = _started_task(120.0)
        runner.poll(task_b, parent)
        sea.release(("first report", 0.1))
        _wait_until(lambda: not runner.poll(task_b, parent).running)

        forced = runner.poll(task_b, parent, force=True)
        self.assertTrue(forced.running)
        self.assertEqual(forced.text, "first report")
        _wait_until(lambda: len(sea.calls) == 2)
        sea.release(RuntimeError("model down"))
        _wait_until(lambda: not runner.poll(task_b, parent).running)
        failed = runner.poll(task_b, parent)
        self.assertEqual(failed.text, "first report")
        self.assertEqual(failed.error, "RuntimeError: model down")
        self.assertEqual(failed.cost, 0.0)
        self.assertFalse(failed.payload()["running"])
        self.assertTrue(failed.payload()["exists"])

        # A ``BaseException`` that is not the stop's interrupt (a
        # cancelled asyncio call) is a failure too, not an empty success.
        runner.poll(task_b, parent, force=True)
        _wait_until(lambda: len(sea.calls) == 3)
        sea.release(asyncio.CancelledError("cancelled"))
        _wait_until(lambda: not runner.poll(task_b, parent).running)
        cancelled = runner.poll(task_b, parent)
        self.assertEqual(cancelled.text, "first report")
        self.assertEqual(cancelled.error, "CancelledError: cancelled")

    def test_interval_elapsed_reruns_and_idle_reports_are_pruned(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_c = _started_task(120.0)
        task_other = _started_task(120.0)
        runner.poll(task_c, parent)
        sea.release(("r1", 0.0))
        _wait_until(lambda: not runner.poll(task_c, parent).running)
        # Shorten the interval only around the one poll that must be due:
        # while the polling waits run, a poll landing 50 ms after a finish
        # would otherwise start an unscripted extra run.
        saved = task_update.UPDATE_INTERVAL_S
        task_update.UPDATE_INTERVAL_S = 0.05
        self.addCleanup(setattr, task_update, "UPDATE_INTERVAL_S", saved)
        try:
            time.sleep(0.06)
            due = runner.poll(task_c, parent)
        finally:
            task_update.UPDATE_INTERVAL_S = saved
        self.assertTrue(due.running)
        self.assertEqual(len(sea.calls), 2)
        sea.release(("r2", 0.0))
        _wait_until(lambda: runner.poll(task_c, parent).text == "r2")

        # A report nobody polled for an hour is dropped on the next poll
        # of any task; a later poll of the pruned task starts afresh.
        with runner._lock:
            runner._updates[task_c].polled_at -= task_update._PRUNE_AFTER_S
        runner.poll(task_other, parent)
        with runner._lock:
            self.assertNotIn(task_c, runner._updates)
            self.assertIn(task_other, runner._updates)
        fresh = runner.poll(task_c, parent)
        self.assertEqual(fresh.text, "")
        self.assertTrue(fresh.running)
        sea.release(("other", 0.0))
        sea.release(("c-again", 0.0))
        _wait_until(lambda: len(sea.calls) == 4 and not any(
            runner.poll(t, parent).running for t in (task_other, task_c)
        ))

        # An update whose OWN poll is the one that prunes it (the panel
        # comes back to the tab after an hour) is recreated dated from
        # the task's row, not from this poll: the overdue run starts now.
        with runner._lock:
            runner._updates[task_c].polled_at -= task_update._PRUNE_AFTER_S
        back = runner.poll(task_c, parent)
        self.assertEqual(back.text, "")
        self.assertTrue(back.running)
        self.assertAlmostEqual(back.started_at, time.time() - 120.0, places=0)
        sea.release(("c-back", 0.0))
        _wait_until(lambda: runner.poll(task_c, parent).text == "c-back")

    def test_concurrent_first_polls_create_one_update_and_one_run(self) -> None:
        """Simultaneous first polls (several surfaces showing one tab) share one update."""
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_id = _started_task(120.0)
        results: list[task_update.TaskUpdate] = []
        go = threading.Barrier(8)

        def poll_once() -> None:
            go.wait(5)
            results.append(runner.poll(task_id, parent))

        threads = [threading.Thread(target=poll_once) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r.running for r in results))
        with runner._lock:
            self.assertEqual(list(runner._updates), [task_id])
        _wait_until(lambda: len(sea.calls) == 1)
        sea.release(("one", 0.0))
        _wait_until(lambda: runner.poll(task_id, parent).text == "one")
        self.assertEqual(len(sea.calls), 1)


def _chat_rows(chat_id: str) -> list[dict[str, Any]]:
    """Return ``(id, task, parent_task_id, cost, is_side_channel)`` of every row in *chat_id*."""
    with _rw_lock.read_lock():
        rows = _get_db().execute(
            "SELECT id, task, parent_task_id, cost, is_side_channel "
            "FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp ASC, rowid ASC",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def test_run_task_update_sea_runs_as_a_subagent_in_the_parents_chat(tmp_path: Path) -> None:
    """The child is the /ask agent asked about the parent; it nests under it and bills it.

    The parent is built the way the task runner leaves it mid-run: a
    persisted task row in a chat, exposed through ``last_task_id``, and
    configured for the scripted local model (which the child inherits).
    """
    parent = ChatSorcarAgent("task-update-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Parent task prompt", chat_id="",
        extra={"model": MODEL, "work_dir": str(tmp_path)},
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    budget_before = _agent_usage(parent)[0]

    report = "<p>Read parser.py; no results yet.</p>"
    script = [
        tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500),
        finish_body(report, prompt_tokens=800),
    ]
    with serve(script) as (url, requests):
        parent.model_name = MODEL
        parent.model_config = {"base_url": url, "api_key": "local"}
        text, cost = run_task_update_sea(parent, task_id)

    assert text == report
    assert cost > 0.0
    assert _agent_usage(parent)[0] == pytest.approx(budget_before + cost)

    rows = _chat_rows(chat_id)
    assert len(rows) == 2 and rows[0]["task"] == "Parent task prompt"
    # The child's row holds the prompt the update run actually used: the
    # ``/ask`` question with the ask SEA's ``prompt(task)`` suffix and
    # ``{task_id}`` filled in.
    prompt = str(rows[1]["task"])
    assert prompt.startswith(task_update.UPDATE_QUESTION)
    assert prompt.endswith(ask_sea.ADD_TO_PROMPT.replace("{task_id}", task_id))
    assert "{task_id}" not in prompt
    assert rows[1]["parent_task_id"] == task_id
    assert rows[1]["cost"] == pytest.approx(cost)
    # A side channel: the daemon replays the finished child as
    # ``subagentDone`` rather than re-opening its tab on every reload.
    assert rows[1]["is_side_channel"] == 1
    assert rows[0]["is_side_channel"] == 0
    # The parent is still running: its row keeps the totals its own
    # final save will write (from the live counters charged above).
    assert rows[0]["cost"] == 0.0

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    names = {t["function"]["name"] for t in agentic[0]["tools"]}
    assert names == {"finish", "task_context"}
    system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
    # The ask SEA's ``system_prompt`` replaces the assembled prompt: the
    # default base prompt the run would otherwise start from is gone.
    assert str(system["content"]).startswith(ask_sea.AskSea().system_prompt("ASSEMBLED"))
    assert SYSTEM_PROMPT not in str(system["content"])
    assert SYSTEM_PROMPT_LITE not in str(system["content"])
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert prompt in str(user["content"])
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    digest = str(tool_results[0]["content"])
    assert f"Task id: {task_id}" in digest
    assert "Task prompt: Parent task prompt" in digest


def _update_parent(tmp_path: Path) -> tuple[ChatSorcarAgent, str]:
    """Return an agent as the task runner leaves it mid-run, and its task's id.

    The task has a persisted row in a chat of its own, which the agent
    exposes through ``last_task_id``; the agent's model is the scripted
    local one, which the child of an update inherits.
    """
    parent = ChatSorcarAgent("task-update-eta-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Parent task prompt", chat_id="",
        extra={"model": MODEL, "work_dir": str(tmp_path)},
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    parent.model_name = MODEL
    return parent, task_id


def _child_replies(task_id: str, answers: list[str]) -> list[bytes]:
    """Return the model's replies to the ``/ask`` children of one update run per answer.

    Each child reads the task (``task_context``), then passes its answer
    to ``finish``: two replies a run.
    """
    script: list[bytes] = []
    for answer in answers:
        script.append(tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500))
        script.append(finish_body(answer, prompt_tokens=800))
    return script


def _run_update(
    runner: TaskUpdateRunner, task_id: str, parent: ChatSorcarAgent,
) -> tuple[float, float, task_update.TaskUpdate]:
    """Force one update run of *task_id* through the real child and wait for its end.

    Returns:
        ``(before, after, update)``: the run asked its question between
        the ``time.time()`` values ``before`` and ``after``; ``update``
        is the task's update once the run has ended.
    """
    before = time.time()
    runner.poll(task_id, parent, force=True)
    _wait_until(lambda: not runner.poll(task_id, parent).running, timeout=60.0)
    return before, time.time(), runner.poll(task_id, parent)


_SAID = "Time left: 5-10 min, most likely 7 min."
_QUOTED = "<p>Updated the fallback to <code>Time left: unknown.</code> in task_update.py.</p>"
_COMMENT = "<!-- Don't show > <a title=\" Time left: unknown. -->"
# ``(answer, content, minutes)``: what the child passes to ``finish``, what
# the panel must show, and the ``(low, best, high)`` minutes the answer's last
# sentence states (``None``: no estimate, three zeros).
_ESTIMATE_CASES: list[tuple[str, str, tuple[float, float, float] | None]] = [
    # The sentence as it is asked for; a dash as an entity or a character
    # (the em dash, the non-breaking hyphen).
    (
        "<p>Read parser.py; no results yet. Time left: 6&ndash;15 min, most likely 9 min.</p>",
        "<p>Read parser.py; no results yet.</p>", (6, 9, 15),
    ),
    ("<p>A. Time left: 5\u201410 minutes, most likely 7 minutes.</p>", "<p>A.</p>", (5, 7, 10)),
    ("<p>A. Time left: 5\u201110 min, most likely 7 min.</p>", "<p>A.</p>", (5, 7, 10)),
    # Decimals; a unit of its own on <low>; "to"; seconds, minutes, hours and
    # days in each of their spellings.
    ("<p>A. Time left: 0.5-3 min, most likely 1 min.</p>", "<p>A.</p>", (0.5, 1, 3)),
    ("<p>A. Time left: 45 min-2 h, most likely 1 h.</p>", "<p>A.</p>", (45, 60, 120)),
    ("<p>A. Time left: 30-90 s, most likely 45 secs.</p>", "<p>A.</p>", (0.5, 0.75, 1.5)),
    (
        "<p>A. Time left: 30 sec-90 seconds, most likely 1 minute.</p>",
        "<p>A.</p>", (0.5, 1, 1.5),
    ),
    ("<p>A. Time left: 6 second to 2 mins, most likely 1 m.</p>", "<p>A.</p>", (0.1, 1, 2)),
    ("<p>A. Time left: 1 hr-2 hrs, most likely 1.5 h.</p>", "<p>A.</p>", (60, 90, 120)),
    (
        "<p>A. Time left: 12 hours to 2 d, most likely 1 day.</p>",
        "<p>A.</p>", (720, 1440, 2880),
    ),
    (
        "<p>A. Time left: 1 hour to 2 days, most likely 1.5 days.</p>",
        "<p>A.</p>", (60, 2160, 2880),
    ),
    # Finishing right now is an estimate too: three equal, positive times.
    ("<p>A. Time left: 0-0 min, most likely 0 min.</p>", "<p>A.</p>", (0, 0, 0)),
    # Plain text (``finish`` renders it as a paragraph), another case, neither
    # comma nor full stop; the long s, which Python reads as an "s" when it
    # ignores case.
    ("A. TIME LEFT: 5-10 MIN MOST LIKELY 7 MIN", "<p>A.</p>", (5, 7, 10)),
    (
        "<p>A. Time left: 30-90 \u017f, most likely 45 \u017fecs.</p>",
        "<p>A.</p>", (0.5, 0.75, 1.5),
    ),
    # The sentence is read as it is displayed: white space, inline tags and
    # character references between its words, with their semicolon or not,
    # a stop written as a reference.
    (
        "<p>Done. Time&nbsp;left: 5-10&#160;min, most&#xa0;likely 7\u00a0min.</p>",
        "<p>Done.</p>", (5, 7, 10),
    ),
    (
        "<p>Done. Time <b>left:</b> 5-10 min, most likely <b>7</b> min.</p>",
        "<p>Done.</p>", (5, 7, 10),
    ),
    (
        "<p>Done. <font size=2>Time left: <time>5-10 min</time>, most likely 7 min.</font></p>",
        "<p>Done.</p>", (5, 7, 10),
    ),
    ("<p>R&amp;D. Time&nbspleft&colon; unknown&excl;</p>", "<p>R&amp;D.</p>", None),
    (
        "<p>Done&period; Time left: 5-10 min, most likely 7 min&#46;</p>",
        "<p>Done&period;</p>", (5, 7, 10),
    ),
    # What the agent adds to its sentence belongs to it, inline tags and a
    # stop inside a number or a name included (a letter, a digit or an
    # underscore right after it).
    (
        "<p>A. Time left: 900-2400 min, most likely 1500 min, counted from when you answer, "
        "since a trial took about 1.6 h.</p>",
        "<p>A.</p>", (900, 1500, 2400),
    ),
    (
        "<p>E. Time left: unknown, <code>repair_chrome.sh</code> and run.<b>sh</b> wait...</p>",
        "<p>E.</p>", None,
    ),
    ("<p>E. Time left: unknown, <code>self._lock</code> is held.</p>", "<p>E.</p>", None),
    ("<p>E. Time left: unknown\u2014the user has to answer first.</p>", "<p>E.</p>", None),
    (
        "<ul><li>F. G.</li><li>E. Time left: unknown</li></ul>",
        "<ul><li>F. G.</li><li>E.</li></ul>", None,
    ),
    # Unknown: nothing in that sentence is an estimate, numbers included.
    ("<p>E. Time left: unknown.</p>", "<p>E.</p>", None),
    (
        "<p>E. Time left: unknown (previously 5-10 min, most likely 7 min).</p>",
        "<p>E.</p>", None,
    ),
    # Numbers that are not low <= best <= high, or beyond a year, however
    # long they are: the sentence goes, no estimate.
    ("<p>A. Time left: 8-10 min, most likely 7 min.</p>", "<p>A.</p>", None),
    ("<p>A. Time left: 5-10 min, most likely 12 min.</p>", "<p>A.</p>", None),
    ("<p>D. Time left: 5-10 min, most likely 123456789 min.</p>", "<p>D.</p>", None),
    ("<p>D. Time left: 1-366 d, most likely 2 d.</p>", "<p>D.</p>", None),
    ("<p>D. Time left: 1-365 d, most likely 2 d.</p>", "<p>D.</p>", (1440, 2880, 525600)),
    (f"<p>D. Time left: 5-{'9' * 400} min, most likely 7 min.</p>", "<p>D.</p>", None),
    # Only the answer's last sentence is read: the sentences before it stay
    # as they are, time-left sentences too (review round 1), a quoted one
    # (round 2) and one in a comment (round 2: a comment is one unit,
    # whatever it holds).
    (
        f"<p>Time left: unknown.</p><p>B.</p><p>Time left: 1-2 min, most likely 1 min. {_SAID}</p>",
        "<p>Time left: unknown.</p><p>B.</p><p>Time left: 1-2 min, most likely 1 min.</p>",
        (5, 7, 10),
    ),
    (f"<p>{_SAID}</p><p>C. Time left: unknown.</p>", f"<p>{_SAID}</p><p>C.</p>", None),
    (f"{_QUOTED}<p>{_SAID}</p>", _QUOTED, (5, 7, 10)),
    (f"<p>Done.</p>{_COMMENT}<p>{_SAID}</p>", f"<p>Done.</p>{_COMMENT}", (5, 7, 10)),
    (
        f'<p>The log printed "Time left: 3 min". B. {_SAID}</p>',
        '<p>The log printed "Time left: 3 min". B.</p>', (5, 7, 10),
    ),
    # The sentence before it may end with closing quotes, brackets and
    # inline tags.
    (
        f'<p>The review ended with "No demonstrated issues found." {_SAID}</p>',
        '<p>The review ended with "No demonstrated issues found."</p>', (5, 7, 10),
    ),
    (
        "<p>It said <i>\u201cstop.</i>\u201d) Time left: unknown.</p>",
        "<p>It said <i>\u201cstop.</i>\u201d)</p>", None,
    ),
    # Markup is read as a browser reads it: an attribute value may hold ">",
    # a quote outside a value is no quote, names are in any case, "<?...>"
    # is a comment, the last tag need not be closed; a reference to no
    # character and one with thousands of digits are read, a control
    # character is text.
    (f"<p>A.</p><p title=\"a > b\" class=it's>{_SAID}</p>", "<p>A.</p>", (5, 7, 10)),
    ("<p>Done. Time left: unknown <a title=it's>x</a>.</p>", "<p>Done.</p>", None),
    (
        "<!DOCTYPE html><P>A.</P><DIV><B>Time left:</B> 5-10 min, most likely 7 min.</div>",
        "<!DOCTYPE html><P>A.</P>", (5, 7, 10),
    ),
    (
        "<p>Done.</p><?xml version='1.0'?>Time left: unknown.",
        "<p>Done.</p><?xml version='1.0'?>", None,
    ),
    (f"<p>A. {_SAID}</p", "<p>A.</p", (5, 7, 10)),
    (f"<p>A. Time left: unknown &#1;&#{'0' * 5000}46;</p>", "<p>A.</p>", None),
    ("<p>A. Time left: unknown \x01.</p>", "<p>A.</p>", None),
    # Only the sentence's text goes: the markup around it stays balanced, and
    # the elements this leaves empty go with the white space before them
    # (what a browser collapses, not a non-breaking space).
    (f"<p>A.</p>\u00a0<p>{_SAID}</p>", "<p>A.</p>\u00a0", (5, 7, 10)),
    (f"<p><em>Done. {_SAID}</em></p>", "<p><em>Done.</em></p>", (5, 7, 10)),
    (f"<p>A.</p>\n<div><p><strong><em>{_SAID}</em></strong></p></div>", "<p>A.</p>", (5, 7, 10)),
    (f"<ul><li>A.</li><li> {_SAID} </li></ul>", "<ul><li>A.</li></ul>", (5, 7, 10)),
    (f"<div>\n<p>A.</p>\n<p>{_SAID}</p>\n</div>", "<div>\n<p>A.</p>\n</div>", (5, 7, 10)),
    (f"<p>A.</p> <b><p>{_SAID}</p></b>", "<p>A.</p>", (5, 7, 10)),
    (f"<p>{_SAID}</p>", "", (5, 7, 10)),
    ("Time <b>left</b>: unknown.", "", None),
    ("<p><b>Done. Time</b> left: <i>unknown</i>.</p>", "<p><b>Done.</b></p>", None),
    ("<p>Done.<b> Time <i>left: unknown.</i></b></p>", "<p>Done.</p>", None),
    # An element that was empty before stays, and so does markup that was
    # not balanced.
    (f"<div><p></p><p>{_SAID}</p></div>", "<div><p></p></div>", (5, 7, 10)),
    ("<p><b>Time left: unknown.</i></i></p>", "<p><b></i></i></p>", None),
    (f"<div><p>{_SAID}</div>", "<div><p></div>", (5, 7, 10)),
    (f"<p>{_SAID}</p></div>", "</div>", (5, 7, 10)),
]
# Answers that do not end with the sentence: shown as they are, no estimate.
_PROSE_CASES = [
    "<p>A. There is little time left before the deadline.</p>",
    # Text, an opening tag or a comment after the sentence (review round 1:
    # nothing after it is deleted, no half of it is left; round 2: a comment
    # after an estimate does not replace it).
    f"<p>Done. {_SAID} Tests are still running.</p>",
    "<p>E. Time left: unknown! The user has to answer first.</p>",
    "<p>Done. Time left:<br>5-10 min, most likely 7 min.</p>",
    f"<p>{_SAID}</p><p>C.</p>",
    "<ul><li>E. Time left: unknown</li><li>F</li></ul>",
    f"<p>Done. {_SAID}<br></p>",
    f"<p>Done. {_SAID}<b></b></p>",
    f"<p>Done. {_SAID}</p>{_COMMENT}",
    "<p>(Done. Time left: unknown.)</p>",
    # Review round 2: a comment does not end at an apostrophe, a tag or ">"
    # inside it (one that is not closed runs to the end), and a stop written
    # as a reference ends a sentence.
    "<p>Done.</p><!-- Don't show: Time left: unknown --><p>Tests are still running.</p>",
    "<p>Done.</p><!-- a > b. Time left: unknown.",
    "<p>Done. Time left: 5-10 min, most likely 7 min&#46; Tests are still running.</p>",
    "<p>Done. Time left: 5-10 min, most likely 7 min&period; Tests are still running.</p>",
    # A sentence the answer quotes last, in the middle of a sentence of its own.
    "<p>Updated the fallback to <code>Time left: unknown.</code></p>",
    "<p>The log printed <tt>Time left: unknown.</tt></p>",
    '<p>The log printed "Time left: unknown."</p>',
    "<p>See run.<b>Time left: unknown.</b></p>",
    # Another form after the label (review round 1: a sign, another unit).
    # A word is the one the answer displays: a tag, a character reference or
    # a character that displays nothing does not end it (owner round 1), and
    # neither joins two words or the digits of a number.
    "<p>D. Time left: unknowns.</p>",
    "<p>D. Time left: unknow&#110;s.</p>",
    "<p>D. Time left: 5-10 min, most likely 7 &#109;onths.</p>",
    "<p>D. Time left: 5-10 min, most likely 7 m<b>onths</b>.</p>",
    "<p>D. Time left: 5-10 min, most likely 7 m\u200bonths.</p>",
    "<p>D. Time left: 1<b>5</b>-20 min, most likely 17 min.</p>",
    "<p>D. Time<b></b>left: unknown.</p>",
    "<p>D. Time left: 5-10 min, most<b></b>likely 7 min.</p>",
    "<p>D. Time left: 5-10 min, most likely -7 min.</p>",
    "<p>D. Time left: 5-10 min, most likely 7 weeks.</p>",
    "<p>D. Time left: 5-10 min, most likely 7 months.</p>",
    "<p>D. Time left: 1-2 days, most likely 1.5.</p>",
    "<p>D. Time left: about 5-10 min, most likely 7 min.</p>",
    "<p>D. Time left: 5 or 10 min, most likely 7 min.</p>",
    # Inside a tag (review round 1), and after an element whose content is
    # not the answer's text, nothing is read.
    '<p title="Time left: unknown.">Done.</p>',
    f'<p title="a < b. {_SAID}">Done. More.</p>',
    f'<p>A.</p><script>var p = "</p>";</script><p>{_SAID}</p>',
    f"<p>A. {_SAID}</p><template>Time left: unknown.</template>",
    # A control character and a reference to none are text, not white space.
    "<p>A.</p><p>B \x05 Time left: unknown.</p>",
    "<p>A.</p><p>&#1;Time left: unknown.</p>",
    "<p>D. Time\x1cleft: unknown.</p>",
]
# What the prefix of a generated answer is made of: whole and broken tags,
# comments, character references with and without their semicolon, stops,
# and time-left sentences in running text, quoted, in an attribute and in a
# comment.
_PREFIX_PIECES = [
    "<p>", "</p>", "<b>", "</b>", "<div>", "</div>", "<ul><li>", "</li></ul>", "<br>", "<hr/>",
    "<a href='x>y' title=it's>", "</a>", '<p title="Time left: unknown.">', "<i ", '<a x="',
    "<p class='", "<", ">", "</", "<!", "<?", "<!--", "-->", "<!-->", '"', "'", "=", "/", "&",
    "&amp;", "&lt;b&gt;", "&#46;", "&period;", "&nbsp;", "&ndash;", "&#x2e", "&bogus;", "&#0;",
    " ", "\n", ". ", "! ", "? ", "A", "R&D", "1.6 h", "Tests pass", "\x01", "\u00a0",
    _QUOTED, _COMMENT, f"<p>{_SAID}</p>", "Time left: 1-2 min, most likely 1 min. ",
    "Time left: unknown", "<code>Time left: unknown.</code>", "<pre>Time left: unknown.\n</pre>",
]
# Ends whatever a prefix left open: a comment, a tag, a quoted attribute value.
_PREFIX_END = "-->\">'>"
_FINAL_SENTENCES: list[tuple[str, tuple[float, float, float] | None]] = [
    (_SAID, (5, 7, 10)),
    ("Time left: unknown.", None),
    ("<b>Time left:</b> 20&ndash;40&nbsp;min, most likely <b>30</b> min&period;", (20, 30, 40)),
]


def _generated_answers(
    count: int, seed: int,
) -> list[tuple[str, str, tuple[float, float, float] | None]]:
    """Return *count* ``(answer, content, minutes)`` with a random prefix each.

    The prefix comes back unchanged in every one of them: with all that
    follows it when the answer does not end with the sentence (no
    sentence, text after it, a comment after it), else without the
    sentence and the elements that held nothing else.
    """
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        pieces = rng.choices(_PREFIX_PIECES, k=rng.randint(0, 12))
        prefix = "<p>Partial results.</p>" + "".join(pieces) + _PREFIX_END
        sentence, minutes = rng.choice(_FINAL_SENTENCES)
        ending = rng.randrange(6)
        if ending == 0:
            answer, content, minutes = f"{prefix}<p>Done.</p>", f"{prefix}<p>Done.</p>", None
        elif ending == 1:
            answer, content = f"{prefix}<p>{sentence}</p>", prefix
        elif ending == 2:
            answer, content = f"{prefix}<p>Done. {sentence}</p>\n", f"{prefix}<p>Done.</p>\n"
        elif ending == 3:
            answer, content = f"{prefix}\n<div><p><em>{sentence}</em></p>\n</div>\n", prefix
        elif ending == 4:
            answer = content = f"{prefix}<p>{sentence} Tests are still running.</p>"
            minutes = None
        else:
            answer = content = f"{prefix}<p>{sentence}</p><!-- end -->"
            minutes = None
        cases.append((answer, content, minutes))
    return cases


def _assert_published(
    tmp_path: Path, subtests: pytest.Subtests,
    cases: list[tuple[str, str, tuple[float, float, float] | None]],
) -> list[dict[str, Any]]:
    """Run one update per case through the real ``/ask`` child and check what it publishes.

    The question asks for the time left, the child answers through
    ``finish``, and the panel's payload carries the case's content and
    three finish times: they count from the moment the question was
    asked, the stated range widened by
    :data:`task_update.ETA_RANGE_WIDENING`.

    Returns:
        The requests the model server received.
    """
    parent, task_id = _update_parent(tmp_path)
    widening = task_update.ETA_RANGE_WIDENING
    answers = [answer for answer, _content, _minutes in cases]
    with serve(_child_replies(task_id, answers)) as (url, requests):
        parent.model_config = {"base_url": url, "api_key": "local"}
        for answer, content, minutes in cases:
            # A runner of its own: no text and no estimate of the case before.
            before, after, upd = _run_update(TaskUpdateRunner(), task_id, parent)
            with subtests.test(answer=answer):
                payload = upd.payload()
                assert payload["error"] == ""
                assert payload["content"] == content
                times = (payload["etaLowAt"], payload["etaAt"], payload["etaHighAt"])
                if minutes is None:
                    assert times == (0, 0, 0)
                    continue
                low, best, high = (60.0 * m for m in minutes)
                asked_at = upd.eta_at - best
                assert before - 0.001 <= asked_at <= after + 0.001
                assert upd.eta_low_at == pytest.approx(asked_at + low / widening, abs=0.001)
                assert upd.eta_high_at == pytest.approx(asked_at + high * widening, abs=0.001)
                # On the wire: whole milliseconds, positive and in order.
                assert times == (
                    int(upd.eta_low_at * 1000), int(upd.eta_at * 1000),
                    int(upd.eta_high_at * 1000),
                )
                assert 0 < times[0] <= times[1] <= times[2]
                if low:
                    # The published range is wider than the one the child stated.
                    assert upd.eta_low_at < asked_at + low
                    assert upd.eta_high_at > asked_at + high
    assert len(requests) == 2 * len(cases)
    return requests


def test_runner_publishes_the_estimate_of_the_real_child(
    tmp_path: Path, subtests: pytest.Subtests,
) -> None:
    """The whole chain, however the child writes its estimate and whatever else it writes."""
    cases = _ESTIMATE_CASES + [(answer, answer, None) for answer in _PROSE_CASES]
    requests = _assert_published(tmp_path, subtests, cases)
    # The model was asked for the sentence, in the two forms the parser reads.
    asked = next(str(m["content"]) for m in requests[0]["messages"] if m["role"] == "user")
    assert '"Time left: <low>-<high> min, most likely <best> min."' in asked
    assert '"Time left: unknown."' in asked


def test_estimate_is_the_last_sentence_of_the_answer_or_nothing(
    tmp_path: Path, subtests: pytest.Subtests,
) -> None:
    """Whatever stands before an answer's last sentence comes back unchanged.

    Generated answers: a prefix of random markup, then no sentence, the
    sentence (in a paragraph of its own, sharing one, wrapped) or the
    sentence with something after it.  The parser returns the answer as
    it is, or without exactly its last sentence.
    """
    cases = _generated_answers(150, seed=20261120)
    # Every ending is in the sample, and half of the answers keep their sentence.
    assert 60 < sum(answer == content for answer, content, _minutes in cases) < 90
    _assert_published(tmp_path, subtests, cases)


def _due_in(upd: task_update.TaskUpdate) -> float:
    """Return the seconds from the end of *upd*'s last run to its next unforced run."""
    return upd.due_at - upd.finished_at


def test_real_update_with_an_estimate_is_due_after_half_of_the_time_left(tmp_path: Path) -> None:
    """Near the task's end the update is refreshed sooner than every 10 minutes.

    One task's updates, each run through the real child; ``due_at`` is
    the time before which no unforced poll starts the next run.
    """
    parent, task_id = _update_parent(tmp_path)
    runner = TaskUpdateRunner()
    interval = pytest.approx(task_update.UPDATE_INTERVAL_S, abs=0.001)
    first_delay = pytest.approx(task_update.FIRST_UPDATE_DELAY_S, abs=0.001)
    script = _child_replies(task_id, [
        "<p>A.</p>",
        "<p>A. Time left: 60-120 min, most likely 90 min.</p>",
        "<p>B. Time left: 4-12 min, most likely 8 min.</p>",
        "<p>C. Time left: 0-0 min, most likely 0 min.</p>",
        "<p>Time left: 20-40 min, most likely 30 min.</p>",
        "",
    ])
    with serve(script) as (url, requests):
        parent.model_config = {"base_url": url, "api_key": "local"}
        # No estimate, or one far away: the regular interval.
        plain = _run_update(runner, task_id, parent)[2]
        assert (plain.text, plain.error, plain.eta_at) == ("<p>A.</p>", "", 0.0)
        assert _due_in(plain) == interval
        far = _run_update(runner, task_id, parent)[2]
        assert far.eta_at > far.finished_at + 89 * 60.0
        assert _due_in(far) == interval

        # Eight minutes left when the question was asked: the next run is
        # due after half of what is left of them when the answer is in.
        near = _run_update(runner, task_id, parent)[2]
        assert _due_in(near) == pytest.approx((near.eta_at - near.finished_at) / 2, abs=0.001)
        assert 180.0 < _due_in(near) < 240.0
        assert near.payload()["dueAt"] == int(near.due_at * 1000)
        # The fingerprint changed with the estimate.
        assert near.eta_at != far.eta_at
        assert near.sig != far.sig

        # A failed run (the task was stopped under it, so its child asked
        # the model nothing) keeps the answer and its estimate, but is not
        # retried before the whole interval.
        stop_event = threading.Event()
        stop_event.set()
        state = agent_state.AgentState(
            task_id, agent=parent, tab_id="tab-update-eta", server_owned=True,
            stop_event=stop_event, is_task_active=True,
        )
        agent_state.register(state)
        try:
            failed = _run_update(runner, task_id, parent)[2]
        finally:
            agent_state.unregister(task_id, state)
        assert failed.error == "The task was stopped before this update finished."
        assert failed.text == "<p>B.</p>"
        assert (failed.eta_low_at, failed.eta_at, failed.eta_high_at) == (
            near.eta_low_at, near.eta_at, near.eta_high_at,
        )
        assert _due_in(failed) == interval
        assert failed.sig != near.sig

        # About to finish: no sooner than the first update's delay.
        ending = _run_update(runner, task_id, parent)[2]
        assert (ending.text, ending.error) == ("<p>C.</p>", "")
        assert 0.0 < ending.eta_low_at == ending.eta_at == ending.eta_high_at
        assert _due_in(ending) == first_delay
        # An answer that is nothing but the sentence keeps the text and
        # brings its estimate ...
        bare = _run_update(runner, task_id, parent)[2]
        assert bare.text == "<p>C.</p>"
        assert bare.eta_at > ending.eta_at + 29 * 60.0
        assert _due_in(bare) == interval
        # ... and an empty answer keeps the text too, but drops the
        # estimate: it is always the last answer's.
        empty = _run_update(runner, task_id, parent)[2]
        assert (empty.text, empty.error) == ("<p>C.</p>", "")
        assert (empty.eta_low_at, empty.eta_at, empty.eta_high_at) == (0.0, 0.0, 0.0)
        assert _due_in(empty) == interval
    assert len(requests) == 12


def test_update_of_a_subagent_hangs_off_the_subagents_client_tab(
    tmp_path: Path,
) -> None:
    """A task update requested on a running SUB-agent announces itself
    under the tab every client shows that sub-agent in.

    The sub-agent's own ``_tab_id`` is the fan-out's synthetic
    ``task-…__sub_N`` id, which no client has: a ``new_tab`` carrying
    it is dropped by every client (``main.js`` ``case 'new_tab'``) and
    the update's tab never opens anywhere.  The clients show the
    sub-agent as ``{parent_tab_id}__sub_{task_id}``, and that is what
    the update's ``new_tab`` must carry — before any client has
    subscribed to the sub-agent's stream.
    """
    printer = CapturePrinter()
    grandparent_id, chat_id = _add_task(
        "Grandparent prompt", chat_id="",
        extra={"model": MODEL, "work_dir": str(tmp_path)},
    )
    subagent = ChatSorcarAgent("Parallel-child")
    subagent.work_dir = str(tmp_path)
    subagent.printer = printer
    subagent._tab_id = f"task-{grandparent_id}__sub_0"
    subagent._subagent_info = {
        "parent_task_id": grandparent_id,
        "parent_tab_id": "root-tab",
        "reviewer": False,
    }
    sub_task_id, _ = _add_task(
        "Sub-agent prompt", chat_id=chat_id,
        extra={
            "model": MODEL, "work_dir": str(tmp_path),
            "subagent": subagent._subagent_info,
        },
    )
    subagent.resume_chat_by_id(chat_id)
    with subagent._task_id_lock:
        subagent._last_task_id = sub_task_id

    # The child streams (its parent has a printer), so it needs the
    # streaming-capable stand-in endpoint rather than the scripted one.
    def respond(request: dict[str, Any]) -> dict[str, Any]:
        if any(m.get("role") == "tool" for m in request.get("messages", [])):
            return finish_response("<h4>Working</h4>")
        return tool_call_response("task_context", {"task_id": sub_task_id})

    standin = StandInModelServer(respond)
    try:
        subagent.model_name = STANDIN_MODEL
        subagent.model_config = standin.model_config
        text, _cost = run_task_update_sea(subagent, sub_task_id)
    finally:
        standin.stop()

    assert text == "<h4>Working</h4>"
    (new_tab,) = printer.events_of_type("new_tab")
    assert new_tab["parent_tab_id"] == f"root-tab__sub_{sub_task_id}"
    update_row = next(
        r for r in _chat_rows(chat_id) if r["parent_task_id"] == sub_task_id
    )
    assert update_row["is_side_channel"] == 1
    # Completion names the tab every surface opened on the ``new_tab``
    # (``{parent_tab_id}__sub_{task_id}``) and the update's synthetic tab.
    derived_tab, synthetic_tab = (
        e["tab_id"] for e in printer.events_of_type("subagentDone")
    )
    assert derived_tab == f"{new_tab['parent_tab_id']}__sub_{new_tab['task_id']}"
    assert synthetic_tab.startswith(f"task-{sub_task_id}__update-")


def test_mark_legacy_updates_as_side_channels_stamps_only_update_children() -> None:
    """Pre-flag task-update rows get stamped; other children and stamped rows are untouched."""
    parent_id, chat_id = _add_task("Parent task prompt", chat_id="")
    child: dict[str, Any] = {"subagent": {"parent_task_id": parent_id}}
    legacy_update, _ = _add_task(
        task_update_sea.build_prompt(parent_id), chat_id=chat_id, extra=child,
    )
    other_child, _ = _add_task("Review the diff", chat_id=chat_id, extra=child)
    already_stamped, _ = _add_task(
        task_update_sea.build_prompt(parent_id), chat_id=chat_id,
        extra={"subagent": {"parent_task_id": parent_id, "side_channel": True}},
    )
    # An update prompt naming ANOTHER task is not this parent's update.
    foreign_update, _ = _add_task(
        task_update_sea.build_prompt("0" * 32), chat_id=chat_id, extra=child,
    )

    assert mark_legacy_updates_as_side_channels() == 1
    flags = {r["id"]: r["is_side_channel"] for r in _chat_rows(chat_id)}
    assert flags == {
        parent_id: 0,
        legacy_update: 1,
        other_child: 0,
        already_stamped: 1,
        foreign_update: 0,
    }
    # Idempotent: a second daemon start stamps nothing.
    assert mark_legacy_updates_as_side_channels() == 0


def test_run_task_update_sea_adds_its_spend_to_a_finished_parents_row(
    tmp_path: Path,
) -> None:
    """A parent that finished while the update ran gets the spend on its row.

    The parent's final save has already written its totals (``endTs``
    set), so banking on the live agent alone would lose the update's
    cost; it is added to the ``task_history`` row instead.
    """
    parent = ChatSorcarAgent("task-update-finished-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Finished parent", chat_id="",
        extra={
            "model": MODEL, "work_dir": str(tmp_path), "cost": 0.5,
            "tokens": 1000, "steps": 3, "endTs": int(time.time() * 1000),
        },
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    budget_before = _agent_usage(parent)[0]
    script = [
        tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500),
        finish_body("<p>finished</p>", prompt_tokens=800),
    ]
    with serve(script) as (url, _requests):
        parent.model_name = MODEL
        parent.model_config = {"base_url": url, "api_key": "local"}
        text, cost = run_task_update_sea(parent, task_id)
    assert text == "<p>finished</p>"
    assert cost > 0.0
    rows = _chat_rows(chat_id)
    assert rows[0]["id"] == task_id
    assert rows[0]["cost"] == pytest.approx(0.5 + cost)
    assert rows[1]["parent_task_id"] == task_id
    # Not ALSO banked on the live counters: the row is the single account.
    assert _agent_usage(parent)[0] == pytest.approx(budget_before)


class TestGetTaskUpdateOverWss(IsolatedAsyncioTestCase):
    """``getTaskUpdate`` over a live WSS connection."""

    async def asyncSetUp(self) -> None:
        self.port = find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": ""})
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        self.sea = _ScriptedSea()
        self._agents: dict[str, Any] = {}
        self.server._task_updates = TaskUpdateRunner(run_sea=self.sea)
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    def _register_task(
        self, tab_id: str, *, active: bool = True, state_key: str = "",
        age_s: float = 120.0,
    ) -> str:
        """Register a running agent state for *tab_id*; return its task id.

        *state_key* is the registry key (the persisted task id once the
        run allocated its row; ``""`` uses the task id).  The task's row
        started *age_s* seconds ago; two minutes by default, past the
        runner's first-update delay.
        """
        agent = ChatSorcarAgent(f"task-update-wss {tab_id}")
        task_id, chat_id = _add_task(
            f"prompt for {tab_id}",
            extra={"startTs": int((time.time() - age_s) * 1000)},
        )
        agent.resume_chat_by_id(chat_id)
        with agent._task_id_lock:
            agent._last_task_id = task_id
        state = agent_state.AgentState(
            state_key or task_id, agent=agent, tab_id=tab_id, server_owned=True,
            is_task_active=active,
        )
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, state_key or task_id, state)
        self._agents[task_id] = agent
        return task_id

    async def _poll(self, ws: Any, fields: dict[str, Any]) -> dict[str, Any]:
        """Send one ``getTaskUpdate`` and return its ``taskUpdate`` reply."""
        await ws.send(json.dumps({"type": "getTaskUpdate", **fields}))
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            ev: dict[str, Any] = json.loads(raw)
            if ev.get("type") == "taskUpdate":
                return ev
        raise AssertionError("no taskUpdate reply received")

    async def _connect(self) -> Any:
        ws = await connect(f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl())
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        await asyncio.wait_for(ws.recv(), timeout=5)
        return ws

    async def test_tab_without_a_running_task_replies_nothing(self) -> None:
        idle_task = self._register_task("t-idle", active=False)
        ws = await self._connect()
        try:
            unknown = await self._poll(ws, {"tabId": "t-unknown", "token": "1"})
            idle = await self._poll(ws, {"tabId": "t-idle", "token": "2"})
            no_tab = await self._poll(ws, {"token": "3"})
        finally:
            await ws.close()
        for reply in (unknown, idle, no_tab):
            self.assertIs(reply["exists"], False)
            self.assertEqual(reply["content"], "")
            self.assertEqual(reply["sig"], "")
            self.assertEqual(reply["taskId"], "")
            self.assertNotIn("unchanged", reply)
            # No estimate for a task that is not running.
            for key in ("etaAt", "etaLowAt", "etaHighAt"):
                self.assertEqual(reply[key], 0)
        self.assertEqual(unknown["tabId"], "t-unknown")
        self.assertEqual(unknown["token"], "1")
        self.assertEqual(idle["token"], "2")
        self.assertEqual(no_tab["tabId"], "")
        self.assertEqual(self.sea.calls, [])
        self.assertIsNotNone(idle_task)

    async def test_previous_task_id_before_row_allocation_replies_nothing(self) -> None:
        """A reused tab's agent still names the PREVIOUS task before its row exists.

        The task runner marks the state active before the run allocates
        its history row (which re-keys the state to the new id), so a
        state keyed otherwise than the agent's ``last_task_id`` is a run
        in that window: nothing is shown and no agent is started.
        """
        self._register_task("t-reused", state_key="pre-allocation-key")
        ws = await self._connect()
        try:
            reply = await self._poll(ws, {"tabId": "t-reused", "token": "4"})
        finally:
            await ws.close()
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["taskId"], "")
        self.assertEqual(reply["token"], "4")
        self.assertEqual(self.sea.calls, [])

    async def test_young_task_replies_when_its_first_update_is_due(self) -> None:
        """A task under a minute old gets no run yet, only the time of the first one."""
        before = time.time()
        task_id = self._register_task("t-young", age_s=5.0)
        after = time.time()
        ws = await self._connect()
        try:
            reply = await self._poll(ws, {"tabId": "t-young", "token": "5"})
        finally:
            await ws.close()
        self.assertIs(reply["exists"], True)
        self.assertIs(reply["running"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["taskId"], task_id)
        self.assertEqual(reply["updatedAt"], 0)
        self.assertEqual(reply["sig"], "0.000:0")
        # dueAt is the row's start (5 s before registration, truncated to
        # the millisecond) plus the first-update delay; recover that start
        # and bound it by the registration window so a slow connect or
        # poll under load cannot skew the check.
        started = reply["dueAt"] / 1000.0 - task_update.FIRST_UPDATE_DELAY_S + 5.0
        self.assertGreaterEqual(started, before - 0.001)
        self.assertLessEqual(started, after)
        self.assertEqual(self.sea.calls, [])

    async def test_running_task_is_reported_polled_and_refreshed(self) -> None:
        task_id = self._register_task("t-run")
        ws = await self._connect()
        try:
            first = await self._poll(ws, {"tabId": "t-run", "token": "7"})
            self.assertIs(first["exists"], True)
            self.assertIs(first["running"], True)
            self.assertEqual(first["taskId"], task_id)
            self.assertEqual(first["content"], "")
            self.assertEqual(first["token"], "7")
            self.assertEqual(first["tabId"], "t-run")
            self.assertEqual(first["sig"], "0.000:1")
            self.assertEqual((first["etaLowAt"], first["etaAt"], first["etaHighAt"]), (0, 0, 0))
            await asyncio.to_thread(_wait_until, lambda: len(self.sea.calls) == 1)
            self.assertEqual(self.sea.calls[0][1], task_id)
            self.assertIs(self.sea.calls[0][0], self._agents[task_id])

            same = await self._poll(
                ws, {"tabId": "t-run", "knownSig": first["sig"], "token": "7"},
            )
            self.assertIs(same["unchanged"], True)
            self.assertNotIn("content", same)
            self.assertEqual(same["sig"], first["sig"])

            self.sea.release((
                "<h4>Progress</h4><p>read 3 files</p>"
                "<p>Time left: 4-12 min, most likely 8 min.</p>", 0.07,
            ))
            await asyncio.to_thread(
                _wait_until,
                lambda: not self.server._task_updates.poll(
                    task_id, self._agents[task_id],
                ).running,
            )
            done = await self._poll(
                ws, {"tabId": "t-run", "knownSig": first["sig"], "token": "7"},
            )
            self.assertNotIn("unchanged", done)
            self.assertIs(done["running"], False)
            self.assertEqual(done["content"], "<h4>Progress</h4><p>read 3 files</p>")
            self.assertEqual(done["error"], "")
            self.assertEqual(done["cost"], 0.07)
            self.assertGreater(done["updatedAt"], 0)
            self.assertNotEqual(done["sig"], first["sig"])
            # The answer's time-left sentence arrives as three finish times,
            # about 8 minutes ahead, and brings the next run forward to half
            # of that.
            self.assertLess(done["etaLowAt"], done["etaAt"])
            self.assertLess(done["etaAt"], done["etaHighAt"])
            left_ms = done["etaAt"] - done["updatedAt"]
            self.assertGreater(left_ms, 7 * 60_000)
            self.assertLessEqual(left_ms, 8 * 60_000)
            self.assertLessEqual(abs(done["dueAt"] - done["updatedAt"] - left_ms / 2), 1)
            # Still fresh: no new run.
            self.assertEqual(len(self.sea.calls), 1)

            refreshed = await self._poll(
                ws, {"tabId": "t-run", "knownSig": done["sig"], "token": "7",
                     "refresh": True},
            )
            self.assertIs(refreshed["running"], True)
            self.assertEqual(refreshed["content"], done["content"])
            self.assertNotEqual(refreshed["sig"], done["sig"])
            await asyncio.to_thread(_wait_until, lambda: len(self.sea.calls) == 2)
            self.sea.release(("<h4>Progress</h4><p>read 5 files</p>", 0.02))
            await asyncio.to_thread(
                _wait_until,
                lambda: self.server._task_updates.poll(
                    task_id, self._agents[task_id],
                ).text.endswith("5 files</p>"),
            )
            # The newer answer has no estimate: the times are zero again
            # and the fingerprint changed with them.
            last = await self._poll(
                ws, {"tabId": "t-run", "knownSig": done["sig"], "token": "7"},
            )
            self.assertNotIn("unchanged", last)
            self.assertEqual((last["etaLowAt"], last["etaAt"], last["etaHighAt"]), (0, 0, 0))
            # ... and the next run waits the whole interval (whole milliseconds).
            wait_ms = last["dueAt"] - last["updatedAt"]
            self.assertLessEqual(abs(wait_ms - task_update.UPDATE_INTERVAL_S * 1000), 1)
        finally:
            await ws.close()


class TestGetTaskUpdateOverLocalChannel(unittest.TestCase):
    """A local-channel ``getTaskUpdate`` gets a direct ``taskUpdate`` reply.

    Chat pages on the local WSS connection poll ``getTaskUpdate`` to
    fill the Task Info view, so the command is served to local and remote clients alike and
    the reply must come back on the requesting connection.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=self.tmp.name,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=15)

    def tearDown(self) -> None:
        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)],
            timeout=10,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def test_local_get_task_update_gets_direct_reply(self) -> None:
        """A chat editor panel's poll under an idle tab id is answered on its connection."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await open_local_connection(self.server)
            try:
                writer.write(
                    json.dumps({
                        "type": "getTaskUpdate", "tabId": "editor-panel-tab",
                        "knownSig": "", "token": "9", "refresh": False,
                    }).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(timeout=15)
        self.assertEqual(event.get("type"), "taskUpdate")
        self.assertIs(event.get("exists"), False)
        self.assertEqual(event.get("content"), "")
        self.assertEqual(event.get("tabId"), "editor-panel-tab")
        self.assertEqual(event.get("token"), "9")


if __name__ == "__main__":
    unittest.main()
