# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The server reads the agent's task id through one door.

``ChatSorcarAgent`` publishes the ``task_history`` row id of its
current run into ``_last_task_id`` under ``_task_id_lock`` and exposes
it as the ``last_task_id`` property, which reads under the *same*
lock.  Three server-side readers still reached past that property into
the private attribute:

* ``commands._owner_task_id`` — stamps a follow-up prompt queued while
  a task runs, so the echo lands in the running task's transcript.
* ``merge_flow._state_task_key`` — files the ``autocommit_done`` event
  into the task's persisted chat events.
* ``json_printer._persist_event`` — decides which task an event
  belongs to before persisting it.

The consolidation makes all three go through the property.  These
tests pin the behaviour that must survive that change — including the
one way it could silently go wrong: the property answers ``""`` where
the private attribute answered ``None``, so a reader that kept an
``is not None`` guard would start filing events under an EMPTY task
id, corrupting the event log of every task at once.

Everything is real: a real :class:`~kiss.server.server.VSCodeServer`,
real agents, a real temporary git repository, a real SQLite history
database and a real local OpenAI-compatible SSE endpoint.  No mock,
patch, fake or test double, and no paid model call.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    history_rows,
    request_text,
    run_git,
    tool_call_response,
    wait_for,
)
from kiss.tests.server.test_autocommit_wire_toggle import (
    _FakeCredentials,
)

#: Marker routing the stand-in to the "write a file, then finish" script.
_DIRTY_MARKER = "K2-DIRTY-THE-MAIN-TREE"

#: Marker routing the stand-in to the "block until released" script.
_BLOCK_MARKER = "K2-BLOCK-UNTIL-RELEASED"

_DIRTY_FILE = "k2-autocommitted.txt"


class _ReaderHarness(unittest.TestCase):
    """Real server + agents + repo + local model endpoint."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-k2-readers-")
        self.repo = self.home.repo
        self.home.write_config(
            auto_commit_mode=True,
            is_worktree=False,
            max_budget=5.0,
            use_web_browser=False,
        )
        self.credentials = _FakeCredentials()
        self.release = threading.Event()
        self.blocked = threading.Event()
        self.calls: list[str] = []
        self._calls_lock = threading.Lock()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.repo)

    def tearDown(self) -> None:
        self.release.set()
        with agent_state.STATE_LOCK:
            threads = [
                state.task_thread
                for state in agent_state.agent_states.values()
                if state.task_thread is not None
            ]
        for thread in threads:
            thread.join(timeout=30)
        self.standin.stop()
        self.credentials.restore()
        self.home.cleanup()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Script the agent from the marker carried in the prompt.

        The "dirty the tree" branch is gated on the CALL COUNT, not on
        the conversation text: a ``Bash`` redirection produces no
        stdout, so the tool result carries nothing the responder could
        recognise, and matching on text would hand out the same tool
        call forever.
        """
        text = request_text(request)
        with self._calls_lock:
            self.calls.append(text)
            seen_dirty = sum(1 for seen in self.calls if _DIRTY_MARKER in seen)
        if _BLOCK_MARKER in text:
            self.blocked.set()
            self.release.wait(timeout=60)
            return finish_response("released")
        if _DIRTY_MARKER in text and seen_dirty == 1:
            return tool_call_response(
                "Bash",
                {
                    "command": f"printf 'dirty\\n' > {_DIRTY_FILE}",
                    "description": "dirty the main working tree",
                },
            )
        return finish_response("k2 readers done")

    def _run_cmd(
        self,
        prompt: str,
        *,
        tab_id: str = "k2-readers-tab",
        auto_commit: bool = True,
    ) -> dict[str, Any]:
        """Build a ``run`` command for the real server."""
        return {
            "type": "run",
            "tabId": tab_id,
            "prompt": prompt,
            "model": STANDIN_MODEL,
            "workDir": str(self.repo),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": auto_commit,
            "webTools": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        }

    def _tab_agent(self, tab_id: str = "k2-readers-tab") -> Any:
        """Return the live agent the runner created for *tab_id*."""
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
        assert state is not None, "the runner never registered a state"
        return state.agent


class TestQueuedPromptIsStampedWithTheLiveTaskId(_ReaderHarness):
    """``commands._owner_task_id`` resolves the running agent's id."""

    def test_followup_echo_carries_the_running_tasks_id(self) -> None:
        """A prompt typed mid-run is echoed into that run's transcript."""
        self.server._cmd_run(self._run_cmd(f"{_BLOCK_MARKER}: hold still"))
        self.assertTrue(
            self.blocked.wait(timeout=60), "the first run never reached the model",
        )
        agent = self._tab_agent()
        self.assertTrue(
            wait_for(lambda: bool(agent.last_task_id), timeout=30),
            "the run never published its task id",
        )
        running_task_id = agent.last_task_id

        self.server._cmd_run(self._run_cmd("a follow-up typed mid-run"))

        echoes = [
            event
            for event in self.printer.events_of_type("prompt")
            if event.get("text") == "a follow-up typed mid-run"
        ]
        self.assertTrue(echoes, "the queued follow-up was never echoed")
        self.assertEqual(
            [event.get("taskId") for event in echoes],
            [running_task_id] * len(echoes),
            "the queued follow-up was not attributed to the running task",
        )
        self.release.set()


class TestAutocommitEventIsFiledUnderTheTask(_ReaderHarness):
    """``merge_flow._state_task_key`` resolves the running agent's id."""

    def test_autocommit_done_is_persisted_into_the_tasks_events(self) -> None:
        """The auto-commit outcome joins the task's own event log."""
        self.server._run_task(self._run_cmd(f"{_DIRTY_MARKER}: make a file"))

        # The run has fully completed, so its agent is already
        # released; the task id comes from the history row it wrote.
        rows = history_rows()
        self.assertEqual(len(rows), 1, f"expected one history row, got {rows}")
        task_id = str(rows[0]["id"])
        self.assertEqual(
            run_git(self.repo, "cat-file", "-e", f"HEAD:{_DIRTY_FILE}").returncode,
            0,
            "the run's work was never auto-committed",
        )

        persistence._flush_chat_events()
        session = persistence._load_chat_events_by_task_id(task_id)
        self.assertIsNotNone(session, "the task's history row vanished")
        assert session is not None
        events = session["events"]
        assert isinstance(events, list)
        self.assertTrue(
            any(event.get("type") == "autocommit_done" for event in events),
            "the auto-commit outcome was filed under no task at all",
        )


class TestEventsAreNeverFiledUnderAnEmptyTaskId(_ReaderHarness):
    """``json_printer._persist_event`` must reject an unpublished id."""

    def test_agent_that_never_ran_persists_nothing(self) -> None:
        """A state whose agent has not run yet files no events.

        ``last_task_id`` answers ``""`` (not ``None``) before the first
        run, so a reader that kept the old ``is not None`` guard would
        write every event of this state under an empty task id.
        """
        state = agent_state.AgentState(
            "k2-unpublished",
            chat_id="k2-unpublished-chat",
            tab_id="k2-unpublished-tab",
            server_owned=True,
            stop_event=threading.Event(),
        )
        agent = WorktreeSorcarAgent("K2 never ran")
        state.agent = agent
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, state.task_id, state)
        self.assertEqual(agent.last_task_id, "")

        self.printer.broadcast(
            {
                "type": "text_end",
                "text": "an event nobody owns",
                "taskId": state.task_id,
            },
        )

        persistence._flush_chat_events()
        self.assertFalse(
            persistence._load_chat_events_by_task_id(""),
            "an event was persisted under an empty task id",
        )


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
