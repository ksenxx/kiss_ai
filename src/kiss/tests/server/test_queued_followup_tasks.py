# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``<task>``-tagged steering messages queue follow-up tasks.

When the user sends a message while a task is running, a PLAIN message
is injected into the live agent's conversation as a steering
instruction (``AgentState.pending_user_messages``).  A message wrapped
in ``<task>...</task>`` tags must instead be split into its task blocks
and queued on ``AgentState.queued_followup_tasks``; the task runner's
per-subtask loop drains that list after the current task finishes and
runs each block one-by-one as further sequential subtasks.

Covers:

* :func:`kiss.server.task_runner.contains_task_tags` — the routing
  predicate.
* :meth:`_CommandsMixin._cmd_append_user_message` — routes tagged
  prompts to ``queued_followup_tasks`` (server-owned states only) and
  plain prompts to ``pending_user_messages``, echoing either way.
* ``_cmd_run``'s S3-05 queue path — same routing for a second ``run``
  submitted while a task thread is installed.
* ``TaskRunner._run_task`` end-to-end — queued tasks extend the
  subtask loop and run one-by-one in order (with the
  ``appendToPrompt`` suffix applied), a failed subtask aborts them,
  and leftovers never leak into a later run on the same tab.

Drives the real ``VSCodeServer._run_task`` lifecycle with a real
``WorktreeSorcarAgent`` subclass whose ``run`` returns a scripted
``finish()`` result (the same harness as
``test_suggested_next_task_e2e.py``).  No mocks or patches.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.core.utils import finish
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import contains_task_tags


def _clear_states(prefix: str) -> None:
    """Remove every agent state whose tab id starts with *prefix*."""
    with agent_state.STATE_LOCK:
        stale = [
            st.task_id
            for st in agent_state.snapshot()
            if st.tab_id.startswith(prefix)
        ]
    for task_id in stale:
        agent_state.unregister(task_id)


class TestContainsTaskTags:
    """The routing predicate mirrors ``parse_task_tags``'s fallback."""

    def test_true_for_single_and_multiple_blocks(self) -> None:
        assert contains_task_tags("<task>do A</task>")
        assert contains_task_tags(
            "please:\n<task>do A</task>\n<task>do B</task>",
        )

    def test_false_for_plain_text(self) -> None:
        assert not contains_task_tags("just steer the current task")

    def test_false_for_empty_or_whitespace_blocks(self) -> None:
        # ``parse_task_tags`` falls back to the whole text for these,
        # so they must be treated as plain steering messages.
        assert not contains_task_tags("<task></task>")
        assert not contains_task_tags("<task>   \n\t </task>")

    def test_false_for_unclosed_tag(self) -> None:
        assert not contains_task_tags("<task>never closed")


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """A real :class:`VSCodeServer` whose broadcasts land in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []

    def capture(event: dict[str, Any]) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestSteeringMessageRouting:
    """``_cmd_append_user_message`` routes by ``<task>`` tags."""

    PREFIX = "tab-qft-route"

    def setup_method(self) -> None:
        _clear_states(self.PREFIX)

    def teardown_method(self) -> None:
        _clear_states(self.PREFIX)

    def _active_state(self, task_id: str, *, server_owned: bool = True) -> AgentState:
        st = AgentState(
            task_id,
            tab_id=f"{self.PREFIX}-{task_id}",
            server_owned=server_owned,
        )
        st.is_task_active = True
        agent_state.register(st)
        return st

    def test_task_tagged_prompt_queues_tasks_not_steering(self) -> None:
        server, events = _make_server()
        st = self._active_state("route-1")

        server._cmd_append_user_message({
            "tabId": st.tab_id,
            "prompt": "<task>build it</task>\n<task>test it</task>",
        })

        assert st.queued_followup_tasks == ["build it", "test it"]
        assert st.pending_user_messages == []
        echoes = [e for e in events if e.get("type") == "prompt"]
        assert len(echoes) == 1
        assert "<task>build it</task>" in echoes[0]["text"]

    def test_plain_prompt_still_steers(self) -> None:
        server, events = _make_server()
        st = self._active_state("route-2")

        server._cmd_append_user_message({
            "tabId": st.tab_id, "prompt": "focus on the tests",
        })

        assert st.pending_user_messages == ["focus on the tests"]
        assert st.queued_followup_tasks == []
        assert [e for e in events if e.get("type") == "prompt"]

    def test_non_server_owned_state_falls_back_to_steering(self) -> None:
        """A sub-agent state has no runner loop to drain queued tasks."""
        server, _events = _make_server()
        st = self._active_state("route-3", server_owned=False)

        server._cmd_append_user_message({
            "tabId": st.tab_id, "prompt": "<task>later task</task>",
        })

        assert st.queued_followup_tasks == []
        assert st.pending_user_messages == ["<task>later task</task>"]

    def test_run_command_while_thread_installed_queues_tasks(self) -> None:
        """The S3-05 ``run``-while-running path routes the same way."""
        server, events = _make_server()
        st = self._active_state("route-4")
        # A created-but-unstarted thread counts as alive (S3-05).
        st.task_thread = threading.Thread(target=lambda: None, daemon=True)

        server._cmd_run({
            "tabId": st.tab_id,
            "prompt": "<task>queued via run</task>",
        })

        assert st.queued_followup_tasks == ["queued via run"]
        assert st.pending_user_messages == []
        assert [e for e in events if e.get("type") == "prompt"]

    def test_run_command_plain_prompt_still_steers(self) -> None:
        server, _events = _make_server()
        st = self._active_state("route-5")
        st.task_thread = threading.Thread(target=lambda: None, daemon=True)

        server._cmd_run({
            "tabId": st.tab_id, "prompt": "plain follow-up",
        })

        assert st.pending_user_messages == ["plain follow-up"]
        assert st.queued_followup_tasks == []

    def test_closed_queue_falls_back_to_steering(self) -> None:
        """After the run's last drain, a ``<task>`` message must not be
        queued (it would be echoed and then silently discarded by the
        end-of-run cleanup) — it takes the plain steering path."""
        server, _events = _make_server()
        st = self._active_state("route-6")
        st.followup_queue_closed = True

        server._cmd_append_user_message({
            "tabId": st.tab_id, "prompt": "<task>too late</task>",
        })

        assert st.queued_followup_tasks == []
        assert st.pending_user_messages == ["<task>too late</task>"]


class _QueueScriptedAgent(WorktreeSorcarAgent):
    """Real agent subclass whose ``run`` returns a scripted ``finish()``.

    Records every prompt it is run with.  During its FIRST run it can
    send a steering command through the real server dispatch (the same
    path the frontend takes while a task is running), so the test
    exercises the queue-then-drain flow exactly as production would.
    """

    def __init__(self) -> None:
        super().__init__("Sorcar VS Code")
        self.prompts: list[str] = []
        self.server: VSCodeServer | None = None
        self.tab: str = ""
        self.steer_prompt: str = ""
        self.raise_on_first: bool = False

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate a task row, optionally steer, return scripted result."""
        del args
        prompt_template = str(kwargs.get("prompt_template", ""))
        self.prompts.append(prompt_template)
        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        if len(self.prompts) == 1:
            if self.steer_prompt and self.server is not None:
                self.server._cmd_append_user_message(
                    {"tabId": self.tab, "prompt": self.steer_prompt},
                )
            if self.raise_on_first:
                raise RuntimeError("scripted failure")
        return finish(True, summary_in_html=f"<p>done {prompt_template}</p>")


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        self.events.append(dict(event))
        super().broadcast(event)


def _run_with_steering(
    tmp_path: Path,
    tab_id: str,
    prompt: str,
    steer_prompt: str,
    *,
    raise_on_first: bool = False,
    append_to_prompt: str = "",
) -> tuple[_QueueScriptedAgent, AgentState, VSCodeServer]:
    """Run *prompt* through the real ``_run_task`` with mid-run steering.

    Returns the scripted agent (whose ``prompts`` list records every
    subtask it executed, in order), the run's state object, and the
    server, so a test can submit a follow-up run on the same tab.
    """
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    agent = _QueueScriptedAgent()
    agent.server = server
    agent.tab = tab_id
    agent.steer_prompt = steer_prompt
    agent.raise_on_first = raise_on_first
    state = AgentState(
        f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True,
    )
    agent_state.register(state)
    cmd: dict[str, Any] = {
        "tabId": tab_id,
        "prompt": prompt,
        "workDir": str(tmp_path),
        "model": models[0],
        "useWorktree": False,
        "autoCommit": False,
        "_state_key": state.task_id,
    }
    if append_to_prompt:
        cmd["appendToPrompt"] = append_to_prompt
    server._run_task(cmd)
    return agent, state, server


class TestRunnerDrainsQueuedTasks:
    """The subtask loop runs queued ``<task>`` follow-ups one-by-one."""

    PREFIX = "tab-qft-run"

    def setup_method(self) -> None:
        _clear_states(self.PREFIX)

    def teardown_method(self) -> None:
        _clear_states(self.PREFIX)

    def test_queued_tasks_run_sequentially_after_current(self, tmp_path: Path) -> None:
        agent, state, _server = _run_with_steering(
            tmp_path,
            f"{self.PREFIX}-seq",
            "original task",
            "<task>second task</task>\n<task>third task</task>",
        )

        assert agent.prompts == [
            "original task", "second task", "third task",
        ]
        # Nothing was injected into the live conversation, and nothing
        # leaks into the next run on this tab.
        assert state.pending_user_messages == []
        assert state.queued_followup_tasks == []
        assert state.is_task_active is False
        # The final drain closed the queue: a later <task> message
        # takes the steering path instead of being silently dropped.
        assert state.followup_queue_closed is True

    def test_append_to_prompt_suffix_applied_to_queued_tasks(
        self, tmp_path: Path,
    ) -> None:
        agent, _state, _server = _run_with_steering(
            tmp_path,
            f"{self.PREFIX}-sfx",
            "original task",
            "<task>second task</task>",
            append_to_prompt=" [suffix]",
        )

        assert agent.prompts == [
            "original task [suffix]", "second task [suffix]",
        ]

    def test_failed_task_aborts_queued_tasks_and_clears_them(
        self, tmp_path: Path,
    ) -> None:
        agent, state, _server = _run_with_steering(
            tmp_path,
            f"{self.PREFIX}-fail",
            "doomed task",
            "<task>never runs</task>",
            raise_on_first=True,
        )

        assert agent.prompts == ["doomed task"]
        assert state.queued_followup_tasks == []
        assert state.pending_user_messages == []
        # The failure closed the queue as well.
        assert state.followup_queue_closed is True

    def test_reused_idle_state_reopens_queue(self, tmp_path: Path) -> None:
        """A direct-caller run reusing the tab's idle state must not
        inherit the previous run's closed follow-up queue — otherwise
        every later ``<task>`` steering message on the tab would be
        steered instead of queued, forever."""
        agent, state, server = _run_with_steering(
            tmp_path,
            f"{self.PREFIX}-reuse",
            "first run",
            "<task>first follow-up</task>",
        )
        assert agent.prompts == ["first run", "first follow-up"]
        assert state.followup_queue_closed is True

        # Second run on the same tab WITHOUT ``_state_key``: the
        # direct-caller path of ``_resolve_run_state`` reuses the idle
        # state and must re-open its follow-up queue.
        agent2 = _QueueScriptedAgent()
        agent2.server = server
        agent2.tab = state.tab_id
        agent2.steer_prompt = "<task>second follow-up</task>"
        with agent_state.STATE_LOCK:
            state.agent = agent2
        server._run_task({
            "tabId": state.tab_id,
            "prompt": "second run",
            "workDir": str(tmp_path),
            "model": get_available_models()[0],
            "useWorktree": False,
            "autoCommit": False,
        })

        assert agent2.prompts == ["second run", "second follow-up"]
        assert state.queued_followup_tasks == []
        assert state.followup_queue_closed is True

    def test_plain_steering_message_does_not_extend_subtasks(
        self, tmp_path: Path,
    ) -> None:
        agent, state, _server = _run_with_steering(
            tmp_path,
            f"{self.PREFIX}-plain",
            "original task",
            "just a steering note",
        )

        assert agent.prompts == ["original task"]
        # The plain message took the steering path; the run's cleanup
        # then cleared the undrained entry (scripted agent never calls
        # the pre-step drain).
        assert state.queued_followup_tasks == []
        assert state.pending_user_messages == []
