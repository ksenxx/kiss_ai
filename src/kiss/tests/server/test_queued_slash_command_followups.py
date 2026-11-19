# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``<task>/xxx text</task>`` steering message runs the SEA ``xxx`` after the task.

A ``<task>`` block typed while a task runs is queued and, once the task
ends, the subtask loop ran it as a plain subtask of the same run: a
``/xxx text`` block reached the model verbatim and the SEA ``xxx`` was
never invoked.  A queued command now runs as the tab's next run, which
goes through the slash-command pipeline (``TaskRunner._apply_sea``),
and the follow-ups queued after it keep their order.

All real: a ``RemoteAccessServer`` on its local endpoint, a registered
user SEA whose ``prompt()`` marks the task it is given, and a stand-in
model endpoint that holds the first request until the steering messages
are queued and records every request.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.conftest import reset_sea_registry
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    StandInModelServer,
    finish_response,
)
from kiss.tests.server.test_task_runner_early_return_followups import (
    _last_user_message,
    _LocalClient,
)

_MARK_SEA = textwrap.dedent(
    """
    from kiss.agents.seas.base.base_sea import BaseSea


    class Sea(BaseSea):
        def description(self):
            \"\"\"Mark the task.\"\"\"
            return "Marks the task."

        def prompt(self, task):
            \"\"\"Prefix the task with the SEA's marker.\"\"\"
            return "MARKED-BY-SEA " + task
    """
)


class _HoldingEndpoint:
    """Stand-in model endpoint: holds the first request until released, records all."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
            first = len(self.requests) == 1
        if first:
            self.release.wait(60)
        return finish_response("<p>done</p>")

    def user_messages(self) -> list[str]:
        """The text of every request's last user message, oldest first."""
        with self._lock:
            return [_last_user_message(r)[0] for r in self.requests]


class TestQueuedSlashCommandFollowup(TestCase):
    """``<task>/mark ...</task>`` typed mid-run runs ``/mark`` after the run."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-queued-slash-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        sea = self.tmp / "seas" / "mark" / "mark_sea.py"
        sea.parent.mkdir(parents=True)
        sea.write_text(_MARK_SEA, encoding="utf-8")
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        seas_md = home / "SEAS.md"
        seas_md.write_text(str(self.tmp / "seas") + "\n", encoding="utf-8")
        self.addCleanup(reset_sea_registry)
        self.addCleanup(seas_md.unlink)
        sea_commands.refresh_registry()
        assert sea_commands.get_command("mark") == sea.resolve()
        self.endpoint = _HoldingEndpoint()
        self.addCleanup(self.endpoint.server.stop)
        self.addCleanup(self.endpoint.release.set)
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
        self.endpoint.release.set()
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

    def _send(self, tab_id: str, prompt: str, **extra: Any) -> None:
        self.client.send({
            **extra,
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "taskId": f"tok-{tab_id}-{len(self.client.events)}",
            "model": STANDIN_MODEL,
            "modelConfig": self.endpoint.server.model_config,
            "classifyTasks": False,
            "useWebTools": False,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
        })

    def _wait_for_requests(self, count: int) -> list[str]:
        deadline = time.monotonic() + 60
        while len(self.endpoint.user_messages()) < count:
            if time.monotonic() > deadline:
                raise AssertionError(f"fewer than {count} model calls")
            time.sleep(0.02)
        return self.endpoint.user_messages()

    def test_inner_task_block_of_a_command_stays_its_text(self) -> None:
        """``/mark a <task>/mark b</task> c`` runs ``/mark`` on its whole text."""
        self.endpoint.release.set()
        self._send("inner-block-tab", "/mark explain <task>/mark inner</task> please")
        [text] = self._wait_for_requests(1)
        self.assertIn("MARKED-BY-SEA explain <task>/mark inner</task> please", text)
        # Not one whole block either: no command run, so no marker.
        self._send("stray-close-tab", "<task>/mark x</task> trailing </task>")
        self.assertNotIn("MARKED-BY-SEA", self._wait_for_requests(2)[1])

    def test_queued_command_in_a_subagent_stays_a_subtask(self) -> None:
        """A sub-agent's leftovers are never re-run, so its queued command runs as a subtask."""
        tab_id = "queued-slash-sub-tab"
        self._send(tab_id, "sub first task", parentTaskId="no-such-parent-task")
        self._wait_for_requests(1)
        self._send(tab_id, "<task>/mark in sub</task>")
        state = agent_state.find_by_tab(tab_id)
        deadline = time.monotonic() + 30
        while state is None or state.queued_followup_tasks != ["/mark in sub"]:
            if time.monotonic() > deadline:
                raise AssertionError("the follow-up was never queued")
            time.sleep(0.02)
            state = agent_state.find_by_tab(tab_id)
        self.endpoint.release.set()
        texts = self._wait_for_requests(2)
        self.assertIn("/mark in sub", texts[1])
        self.assertNotIn("MARKED-BY-SEA", texts[1])

    def test_queued_command_runs_its_sea_in_order(self) -> None:
        """A plain follow-up, ``/mark``, a later follow-up: each in turn, ``/mark`` as its SEA."""
        tab_id = "queued-slash-tab"
        self._send(tab_id, "first task")
        deadline = time.monotonic() + 30
        while not self.endpoint.user_messages():
            if time.monotonic() > deadline:
                raise AssertionError("the first task never called the model")
            time.sleep(0.02)
        steering = (
            "<task>plain follow-up</task>",
            "<task>/mark hello from steering</task>",
            "<task>after mark</task>",
        )
        self._send(tab_id, f"{steering[0]}\n{steering[1]}")
        self.client.wait_for("prompt", tab_id, text=f"{steering[0]}\n{steering[1]}")
        self._send(tab_id, steering[2])
        self.client.wait_for("prompt", tab_id, text=steering[2])
        self.endpoint.release.set()
        markers = (
            "first task", "plain follow-up", "MARKED-BY-SEA hello from steering", "after mark",
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and (
            len(self.endpoint.user_messages()) < 4
            or sum(not e["running"] for e in self.client.matching("status", tab_id)) < 2
        ):
            time.sleep(0.05)
        # A subtask's prompt recaps the earlier ones, so the message a
        # request is for is the marker that appears last in it.
        received = [
            max(markers, key=text.rfind) for text in self.endpoint.user_messages()
        ]
        self.assertEqual(
            received, list(markers),
            "BUG: the queued /mark follow-up did not run its SEA, or ran out of order",
        )
        # The first run (with the plain follow-up), then the /mark run
        # (with the follow-up queued after it).
        self.assertEqual(
            [e["running"] for e in self.client.matching("status", tab_id)],
            [True, False, True, False],
        )
