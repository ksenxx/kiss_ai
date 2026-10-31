# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Runs that return before their subtask loop end like every other run.

``_run_task_inner`` answers ``/xxx help`` (and refuses a run with no
model, or during a merge) before its subtask loop starts, so the
loop's ``finally`` — ``_end_agent_loop`` — never ran for them:

* ``<task>`` follow-ups the user typed while the SEA's ``description()``
  ran (untrusted code, unbounded) were queued on the state, echoed as
  accepted, and then silently cleared by ``_release_run_state``, while
  a plain prompt typed in the same window is re-run as the tab's next
  run;
* a Stop clicked while the help row was being persisted was never
  acknowledged, so the Stop watchdog injected ``KeyboardInterrupt``
  into the SQLite write and the run ended as "Task stopped by user",
  although nothing was left to stop;
* the re-run joined the plain prompt and the re-tagged follow-ups
  into one prompt, which ``parse_task_tags`` reduced to the tagged
  blocks (the plain prompt vanished), and pooled every follow-up's
  files, so each block got the union of the files;
* a ``<task>`` message typed once the follow-up queue was closed (the
  help row was still being persisted) took the steering path and sat
  beside the plain prompt in ``pending_user_messages``; the re-run
  joined them all into one prompt with the union of their files, so
  the plain prompt vanished again and every task got every file.

All runs are real: a ``RemoteAccessServer`` on its local endpoint,
a registered user SEA whose ``description()`` blocks until the test
releases it, the real worker thread and the real watchdog; the
re-runs that need a model are served by a local stand-in endpoint
recording what each one received.  The persistence window is held
open by write-locking ``history.db`` from a second connection, which
parks the help row's insert in SQLite's busy wait.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sqlite3
import tempfile
import textwrap
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import TestCase

from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import local_endpoint, persistence, sea_commands
from kiss.core.config import kiss_home
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    StandInModelServer,
    finish_response,
)

_HELP_TEXT = "slowhelp: the help text"

_BLOCKING_DESCRIPTION = textwrap.dedent(
    f"""
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def description(self):
            \"\"\"Block until released, then answer.\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    return {_HELP_TEXT!r}
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the release")
    """
)


class _LocalClient:
    """JSON client of the server's local endpoint, reading on a thread."""

    def __init__(self, endpoint_file: Path) -> None:
        self.ws = local_endpoint.connect(endpoint_file)
        self.events: list[dict[str, Any]] = []
        self._cond = threading.Condition()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while True:
            try:
                message = self.ws.recv()
            except (ConnectionClosed, OSError):
                return
            with self._cond:
                self.events.append(json.loads(message))
                self._cond.notify_all()

    def send(self, cmd: dict[str, Any]) -> None:
        self.ws.send(json.dumps(cmd))

    def matching(self, event_type: str, tab_id: str) -> list[dict[str, Any]]:
        """Return every *event_type* event received for *tab_id* so far."""
        with self._cond:
            return [
                ev for ev in self.events
                if ev.get("type") == event_type and ev.get("tabId") == tab_id
            ]

    def wait_for(
        self, event_type: str, tab_id: str, timeout: float = 30.0, **fields: Any,
    ) -> dict[str, Any]:
        """Return the first *event_type* event for *tab_id* matching *fields*."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for ev in self.events:
                    if (
                        ev.get("type") == event_type
                        and ev.get("tabId") == tab_id
                        and all(ev.get(k) == v for k, v in fields.items())
                    ):
                        return ev
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"no {event_type!r} for {tab_id!r} matching {fields}; got "
                        f"{[e.get('type') for e in self.events]}",
                    )
                self._cond.wait(remaining)

    def close(self) -> None:
        self.ws.close()


def _help_results() -> Counter[str]:
    """How many persisted ``/slowhelp help`` rows end in each result text."""
    return Counter(
        str(e.get("result"))
        for e in persistence._load_history()
        if e.get("task") == "/slowhelp help"
    )


def _last_user_message(request: dict[str, Any]) -> tuple[str, list[bytes]]:
    """Return the text and the decoded image bytes of *request*'s last user message."""
    message = [m for m in request["messages"] if m.get("role") == "user"][-1]
    content = message["content"]
    if isinstance(content, str):
        return content, []
    texts = [part["text"] for part in content if part.get("type") == "text"]
    images = [
        base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        for part in content
        if part.get("type") == "image_url"
    ]
    return "\n".join(texts), images


class _RecordingEndpoint:
    """A stand-in model endpoint that finishes every run and keeps each request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
        return finish_response("<p>done</p>")

    def user_messages(self) -> list[tuple[str, list[bytes]]]:
        """The last user message of every request so far, oldest first."""
        with self._lock:
            return [_last_user_message(r) for r in self.requests]


def _png(name: str) -> dict[str, Any]:
    """A wire attachment entry whose bytes name the message it belongs to."""
    return {
        "name": f"{name}.png",
        "mimeType": "image/png",
        "data": base64.b64encode(f"png-of-{name}".encode()).decode("ascii"),
    }


class TestEarlyReturnRuns(TestCase):
    """``/slowhelp help`` runs end like a run whose subtask loop finished."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-early-return-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        sea = self.tmp / "seas" / "slowhelp" / "slowhelp_sea.py"
        sea.parent.mkdir(parents=True)
        sea.write_text(_BLOCKING_DESCRIPTION, encoding="utf-8")
        self.sea_dir = sea.parent
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        seas_md = home / "SEAS.md"
        seas_md.write_text(str(self.tmp / "seas") + "\n", encoding="utf-8")
        self.addCleanup(sea_commands._reset_for_tests)
        self.addCleanup(seas_md.unlink)
        sea_commands.refresh_registry()
        assert sea_commands.get_command("slowhelp") == sea.resolve()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.remote = RemoteAccessServer(
            local_endpoint_file=self.tmp / "sorcar-local.json",
            work_dir=str(self.work_dir),
        )
        asyncio.run_coroutine_threadsafe(
            self.remote.start_private_async(), self.loop,
        ).result(timeout=30)
        self.client = _LocalClient(self.tmp / "sorcar-local.json")

    def tearDown(self) -> None:
        (self.sea_dir / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        self.client.close()
        asyncio.run_coroutine_threadsafe(
            self.remote.stop_async(), self.loop,
        ).result(timeout=60)
        self.loop.call_soon_threadsafe(self.loop.stop)
        agent_state.agent_states.clear()

    def _run_help(self, tab_id: str, **overrides: Any) -> None:
        """Send a real ``/slowhelp help`` run and park it in ``description()``.

        The run's settings are what the re-run of a leftover inherits;
        *overrides* replace them (a model served by a stand-in endpoint).
        """
        self.client.send({
            "type": "run",
            "prompt": "/slowhelp help",
            "tabId": tab_id,
            "taskId": f"tok-{tab_id}",
            # Unknown to every provider: a run of the follow-up below
            # is refused with "No model available" instead of calling
            # a model.  The help answer itself needs no model.
            "model": "no-such-model-early-return",
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
            **overrides,
        })
        entered = self.sea_dir / "entered"
        deadline = time.monotonic() + 30
        while not entered.exists():
            if time.monotonic() > deadline:
                raise AssertionError("description() never ran")
            time.sleep(0.02)

    def test_task_followup_typed_during_help_runs_next(self) -> None:
        """A ``<task>`` message typed while ``description()`` runs becomes the tab's next run."""
        tab_id = "help-followup-tab"
        self._run_help(tab_id)
        followup = "<task>follow-up typed during help</task>"
        self.client.send({
            "type": "run",
            "prompt": followup,
            "tabId": tab_id,
            "workDir": str(self.work_dir),
        })
        # The follow-up is accepted (echoed as a steering prompt) and
        # queued on the run.
        self.client.wait_for("prompt", tab_id, text=followup)
        (self.sea_dir / "release").write_text("1", encoding="utf-8")
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(result["text"], _HELP_TEXT)
        self.assertTrue(result["success"])
        # The queued follow-up is re-run as the tab's next run (its
        # unknown model makes that run end at once), exactly like a
        # plain prompt typed in the same window.
        rerun = self.client.wait_for("result", tab_id, timeout=30.0, success=False)
        self.assertIn(
            "No model available", rerun["text"],
            "BUG: the <task> follow-up accepted during the help run was dropped",
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (
            sum(not e["running"] for e in self.client.matching("status", tab_id)) < 2
        ):
            time.sleep(0.05)
        self.assertEqual(
            [e["running"] for e in self.client.matching("status", tab_id)],
            [True, False, True, False],
        )
        self.assertEqual(len(self.client.matching("result", tab_id)), 2)

    def test_plain_and_tagged_followups_typed_during_help_each_run_with_own_file(self) -> None:
        """Plain and ``<task>`` messages typed during the help all run, each with its own file."""
        endpoint = _RecordingEndpoint()
        self.addCleanup(endpoint.server.stop)
        tab_id = "help-mixed-followups-tab"
        self._run_help(
            tab_id,
            model=STANDIN_MODEL,
            modelConfig=endpoint.server.model_config,
            classifyTasks=False,
            useWebTools=False,
        )
        plain = "plain prompt typed during help"
        task_a = "tagged follow-up A typed during help"
        task_b = "tagged follow-up B typed during help"
        for prompt, name in (
            (plain, "plain"),
            (f"<task>{task_a}</task>", "a"),
            (f"<task>{task_b}</task>", "b"),
        ):
            self.client.send({
                "type": "run",
                "prompt": prompt,
                "tabId": tab_id,
                "workDir": str(self.work_dir),
                "attachments": [_png(name)],
            })
            # Accepted and echoed before the next one is typed.
            self.client.wait_for("prompt", tab_id, text=prompt)
        (self.sea_dir / "release").write_text("1", encoding="utf-8")
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(result["text"], _HELP_TEXT)
        # The plain prompt is re-run as the tab's next run and the two
        # follow-ups queue on it in order, each with its own file: the
        # endpoint sees three model calls, one per message.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and len(endpoint.user_messages()) < 3:
            time.sleep(0.05)
        # A subtask's prompt recaps the earlier ones, so the message a
        # request is for is the marker that appears last in it.
        received = [
            (max((plain, task_a, task_b), key=text.rfind), images)
            for text, images in endpoint.user_messages()
        ]
        self.assertEqual(
            received,
            [
                (plain, [b"png-of-plain"]),
                (task_a, [b"png-of-a"]),
                (task_b, [b"png-of-b"]),
            ],
            "BUG: a message typed during the help run was dropped or got another's file",
        )
        # One run, three subtasks: the help answer and three results,
        # the tab running twice.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (
            sum(not e["running"] for e in self.client.matching("status", tab_id)) < 2
        ):
            time.sleep(0.05)
        self.assertEqual(
            [e["running"] for e in self.client.matching("status", tab_id)],
            [True, False, True, False],
        )
        self.assertEqual(
            [e["success"] for e in self.client.matching("result", tab_id)],
            [True, True, True, True],
        )

    def test_followups_typed_after_queue_closed_each_run_with_own_file(self) -> None:
        """A plain prompt and ``<task>`` messages typed after closure all run with their own files.

        The plain prompt is typed during ``description()``; the two
        tagged ones while the help row's insert waits on a write-locked
        ``history.db``, when the follow-up queue is already closed, so
        they are queued as steering messages next to the plain one.
        """
        endpoint = _RecordingEndpoint()
        self.addCleanup(endpoint.server.stop)
        tab_id = "help-late-followups-tab"
        self._run_help(
            tab_id,
            model=STANDIN_MODEL,
            modelConfig=endpoint.server.model_config,
            classifyTasks=False,
            useWebTools=False,
        )
        plain = "plain prompt typed during help"
        task_a = "tagged follow-up A typed after closure"
        task_b = "tagged follow-up B typed after closure"
        self.client.send({
            "type": "run",
            "prompt": plain,
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "attachments": [_png("plain")],
        })
        self.client.wait_for("prompt", tab_id, text=plain)
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        blocker = sqlite3.connect(str(persistence._DB_PATH), timeout=30)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            (self.sea_dir / "release").write_text("1", encoding="utf-8")
            # The queue closes before the help row's insert, which
            # then parks on the lock with the run still active.
            deadline = time.monotonic() + 30
            while not state.followup_queue_closed:
                if time.monotonic() > deadline:
                    raise AssertionError("the follow-up queue never closed")
                time.sleep(0.02)
            for prompt, name in ((f"<task>{task_a}</task>", "a"), (f"<task>{task_b}</task>", "b")):
                self.client.send({
                    "type": "run",
                    "prompt": prompt,
                    "tabId": tab_id,
                    "workDir": str(self.work_dir),
                    "attachments": [_png(name)],
                })
                # A late message gets no steering echo (its re-run
                # echoes it); the submit acknowledgment shows it was
                # routed before the next one is typed.
                self.client.wait_for("setTaskText", tab_id, text=prompt)
            self.assertEqual(
                state.pending_user_messages,
                [plain, f"<task>{task_a}</task>", f"<task>{task_b}</task>"],
            )
            self.assertEqual(state.queued_followup_tasks, [])
        finally:
            blocker.rollback()
            blocker.close()
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(result["text"], _HELP_TEXT)
        # The plain prompt is re-run as the tab's next run and the two
        # tagged messages queue on it in order, each with its own file.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and (
            sum(not e["running"] for e in self.client.matching("status", tab_id)) < 2
        ):
            time.sleep(0.05)
        received = [
            (max((plain, task_a, task_b), key=text.rfind), images)
            for text, images in endpoint.user_messages()
        ]
        self.assertEqual(
            received,
            [
                (plain, [b"png-of-plain"]),
                (task_a, [b"png-of-a"]),
                (task_b, [b"png-of-b"]),
            ],
            "BUG: a message typed after the queue closed was dropped or got another's file",
        )
        self.assertEqual(
            [e["running"] for e in self.client.matching("status", tab_id)],
            [True, False, True, False],
        )
        self.assertEqual(
            [e["success"] for e in self.client.matching("result", tab_id)],
            [True, True, True, True],
        )

    def test_stop_during_help_persistence_is_not_injected(self) -> None:
        """A Stop while the help row is written yields the help answer, not a second result."""
        tab_id = "help-persist-stop-tab"
        results_before = _help_results()
        self._run_help(tab_id)
        # A tagged follow-up with a file, typed during ``description()``.
        followup = "<task>follow-up with a file</task>"
        self.client.send({
            "type": "run",
            "prompt": followup,
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "attachments": [{
                "name": "note.txt",
                "mimeType": "text/plain",
                "data": base64.b64encode(b"hello").decode("ascii"),
            }],
        })
        # Queued and echoed (its autocomplete record is a database write
        # of its own, which must not meet the lock below).
        self.client.wait_for("prompt", tab_id, text=followup)
        # Park the help row's insert in SQLite's busy wait: a second
        # connection holds the database's write lock.
        blocker = sqlite3.connect(str(persistence._DB_PATH), timeout=30)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            (self.sea_dir / "release").write_text("1", encoding="utf-8")
            # ``description()`` returns within 20 ms of the release and
            # the runner reaches ``_add_task`` right after; the stop
            # then lands while the insert waits on the lock.
            time.sleep(0.5)
            self.client.send({"type": "stop", "tabId": tab_id, "taskId": f"tok-{tab_id}"})
            ack = self.client.wait_for("stop_ack", tab_id)
            self.assertTrue(ack["accepted"])
            # The watchdog joins 1 s and then injects unless the stop
            # was acknowledged; hold the lock past that point.
            time.sleep(2.5)
            # With the answer complete, the follow-up stays queued with
            # its file for the leftover re-dispatch (the stop then drops
            # it, as it drops a plain leftover prompt).
            state = agent_state.find_by_tab(tab_id)
            assert state is not None
            self.assertTrue(state.stop_acknowledged)
            self.assertTrue(state.followup_queue_closed)
            self.assertEqual(state.queued_followup_tasks, ["follow-up with a file"])
            self.assertEqual(
                [[(a.mime_type, a.data) for a in files] for files in state.queued_followup_files],
                [[("text/plain", b"hello")]],
            )
            self.assertEqual(state.pending_user_messages, [])
        finally:
            blocker.rollback()
            blocker.close()
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(
            result["text"], _HELP_TEXT,
            "BUG: the Stop was injected into the help row's persistence",
        )
        self.client.wait_for("status", tab_id, running=False)
        self.assertEqual(len(self.client.matching("result", tab_id)), 1)
        self.assertEqual(_help_results() - results_before, Counter({_HELP_TEXT: 1}))
