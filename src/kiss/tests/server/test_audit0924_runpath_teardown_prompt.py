# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a prompt typed while a task tears down is not lost.

``_cmd_run`` / ``_cmd_append_user_message`` queue a typed prompt on
``AgentState.pending_user_messages`` whenever the tab's ``task_thread``
is installed.  Only the agent's pre-step hook drains that list, so a
prompt that lands AFTER the agent's last step — during result
broadcast, persistence and worktree merge, while ``task_thread`` is
still installed — used to be echoed to the user and then cleared by
``_run_task``'s end-of-run cleanup without any agent ever seeing it.

The fix (``TaskRunner._run_task``'s finally) captures the leftover
prompts and re-submits them through ``_cmd_run`` as the tab's next
run — what would have happened had the user typed one second later.
A stopped run does not re-dispatch, and a prompt that arrives once the
agent loop is over (``followup_queue_closed``) is queued WITHOUT the
steering echo because the re-dispatched run echoes it as its own prompt.

Harness (same as ``test_queued_followup_tasks.py``): a real
``VSCodeServer`` + ``JsonPrinter`` subclass, a real
``WorktreeSorcarAgent`` subclass whose ``run`` returns a scripted
``finish()`` result, and the first run started through the REAL
``_cmd_run`` so ``task_thread`` is installed exactly as in production.
The re-dispatched second run must not call a model: the scripted first
run rewrites the shared run command's ``model`` to a name that is not
in ``get_available_models()``, so the second ``_run_task`` takes the
runner's own "No model available" early exit after it has registered
the run (state, ``clear``/``setTaskText`` broadcasts) — no mocks or
patches anywhere.
"""

from __future__ import annotations

import threading
import time
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

_UNKNOWN_MODEL = "audit0924-runpath-no-such-model"
_WAIT_S = 60.0


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


class _TeardownAgent(WorktreeSorcarAgent):
    """Real agent whose first ``run`` submits a prompt right before returning.

    The submission lands after the agent's last (and only) step and
    before ``finish`` is processed by the runner — inside the teardown
    window in which ``task_thread`` is still installed.  ``stop_first``
    additionally sets the run's ``stop_event`` so the runner sees a
    stopped run.
    """

    def __init__(self) -> None:
        super().__init__("Sorcar VS Code")
        self.prompts: list[str] = []
        self.server: VSCodeServer | None = None
        self.tab: str = ""
        self.cmd: dict[str, Any] = {}
        self.late_prompt: str = ""
        self.via_run_command: bool = False
        self.stop_first: bool = False

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate a task row, submit the late prompt once, return."""
        del args
        prompt_template = str(kwargs.get("prompt_template", ""))
        self.prompts.append(prompt_template)
        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        if len(self.prompts) == 1 and self.server is not None:
            # The re-dispatched run must not reach a model.
            self.cmd["model"] = _UNKNOWN_MODEL
            if self.stop_first:
                st = agent_state.find_by_tab(self.tab)
                assert st is not None and st.stop_event is not None
                st.stop_event.set()
            if self.late_prompt:
                if self.via_run_command:
                    self.server._cmd_run(
                        {"tabId": self.tab, "prompt": self.late_prompt},
                    )
                else:
                    self.server._cmd_append_user_message(
                        {"tabId": self.tab, "prompt": self.late_prompt},
                    )
        return finish(True, summary_in_html=f"<p>done {prompt_template}</p>")


class _CapturePrinter(JsonPrinter):
    """Real printer recording every broadcast, optionally injecting on ``task_done``.

    ``inject_on_done`` is submitted through ``_cmd_run`` the first time
    a ``task_done`` event is broadcast — the runner emits it after
    ``followup_queue_closed`` is raised and outside ``STATE_LOCK``, so
    the message hits the post-loop branch of ``_cmd_run``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.server: VSCodeServer | None = None
        self.tab: str = ""
        self.inject_on_done: str = ""
        self._injected = False

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, inject the late prompt on ``task_done``, then broadcast."""
        self.events.append(dict(event))
        if (
            event.get("type") == "task_done"
            and self.inject_on_done
            and not self._injected
            and self.server is not None
        ):
            self._injected = True
            self.server._cmd_run(
                {"tabId": self.tab, "prompt": self.inject_on_done},
            )
        super().broadcast(event)


def _wait_idle(tab_id: str) -> None:
    """Join every worker thread installed on *tab_id*'s state, in turn.

    A re-dispatched run installs its thread from the FIRST run's
    worker (inside ``_run_task``'s finally, after that worker cleared
    its own ``task_thread``), so after each join the tab is looked up
    again; a short settle covers the instant in which the first worker
    has cleared its thread but not yet submitted the follow-up.
    """
    deadline = time.time() + _WAIT_S
    while time.time() < deadline:
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(tab_id)
            thread = st.task_thread if st is not None else None
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.time()))
            assert not thread.is_alive(), "worker thread did not finish"
            continue
        time.sleep(0.3)
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(tab_id)
            if st is None or st.task_thread is None:
                return
    raise AssertionError(f"tab {tab_id} never went idle")


def _start_run(
    tmp_path: Path,
    tab_id: str,
    prompt: str,
    *,
    late_prompt: str = "",
    via_run_command: bool = False,
    stop_first: bool = False,
    inject_on_done: str = "",
) -> tuple[_TeardownAgent, _CapturePrinter, VSCodeServer]:
    """Start *prompt* on *tab_id* through the real ``_cmd_run`` and wait for idle."""
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    printer.server = server
    printer.tab = tab_id
    printer.inject_on_done = inject_on_done
    agent = _TeardownAgent()
    agent.server = server
    agent.tab = tab_id
    agent.late_prompt = late_prompt
    agent.via_run_command = via_run_command
    agent.stop_first = stop_first
    # ``_cmd_run`` carries a previous state's agent over to the new
    # run, which is how the scripted agent reaches ``_run_task_inner``.
    pre = AgentState(
        f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True,
    )
    agent_state.register(pre)
    cmd: dict[str, Any] = {
        "type": "run",
        "tabId": tab_id,
        "prompt": prompt,
        "workDir": str(tmp_path),
        "model": models[0],
        "useWorktree": False,
        "autoCommit": False,
        "classifyTasks": False,
        "taskId": "client-token-first-run",
    }
    agent.cmd = cmd
    server._cmd_run(cmd)
    _wait_idle(tab_id)
    return agent, printer, server


def _events_of(printer: _CapturePrinter, kind: str) -> list[dict[str, Any]]:
    return [e for e in printer.events if e.get("type") == kind]


class TestTeardownPromptIsReDispatched:
    """A prompt queued during teardown becomes the tab's next run."""

    PREFIX = "tab-a0924-teardown"

    def setup_method(self) -> None:
        _clear_states(self.PREFIX)

    def teardown_method(self) -> None:
        _clear_states(self.PREFIX)

    def _assert_redispatched(
        self, printer: _CapturePrinter, tab_id: str, late: str,
    ) -> None:
        # A SECOND run was started on the same tab (two ``clear``
        # broadcasts) ...
        clears = _events_of(printer, "clear")
        assert len(clears) == 2, (
            f"expected a re-dispatched second run, got {len(clears)} clear"
            f" event(s): the late prompt was dropped"
        )
        assert all(c.get("tabId") == tab_id for c in clears)
        # ... whose prompt is the late text: ``_cmd_run`` stamps it on
        # the new state and mirrors it into the task panel.
        st = agent_state.find_by_tab(tab_id)
        assert st is not None
        assert st.last_user_prompt == late
        assert st.task_thread is None
        assert st.pending_user_messages == []
        task_texts = [
            e.get("text") for e in _events_of(printer, "setTaskText")
        ]
        assert task_texts[-1] == late
        # The second ``_run_task`` really ran with the rewritten model
        # and took the runner's "No model available" exit.
        results = _events_of(printer, "result")
        assert any(
            "No model available" in str(r.get("text", "")) for r in results
        )
        # The first run's client token must not leak into the second
        # run's status envelope.
        assert st.client_run_token == ""

    def test_prompt_appended_before_finish_is_run_next(self, tmp_path: Path) -> None:
        tab_id = f"{self.PREFIX}-append"
        late = "follow-up typed during teardown"
        agent, printer, _server = _start_run(
            tmp_path, tab_id, "first task", late_prompt=late,
        )
        assert agent.prompts == ["first task"]
        self._assert_redispatched(printer, tab_id, late)
        # Queued while the loop was still open: echoed once as a
        # steering message, never a second time.
        echoes = [
            e for e in _events_of(printer, "prompt") if e.get("text") == late
        ]
        assert len(echoes) == 1

    def test_run_command_during_teardown_is_run_next(self, tmp_path: Path) -> None:
        tab_id = f"{self.PREFIX}-run"
        late = "second submit while the first tears down"
        _agent, printer, _server = _start_run(
            tmp_path, tab_id, "first task", late_prompt=late, via_run_command=True,
        )
        self._assert_redispatched(printer, tab_id, late)

    def test_prompt_after_loop_close_is_not_echoed_as_steering(
        self, tmp_path: Path,
    ) -> None:
        tab_id = f"{self.PREFIX}-late"
        late = "typed after the agent loop ended"
        _agent, printer, _server = _start_run(
            tmp_path, tab_id, "first task", inject_on_done=late,
        )
        self._assert_redispatched(printer, tab_id, late)
        # The agent loop was already over when the prompt arrived: no
        # steering echo at all — the re-dispatched run owns the text.
        echoes = [
            e for e in _events_of(printer, "prompt") if e.get("text") == late
        ]
        assert echoes == []

    def test_stopped_run_does_not_redispatch(self, tmp_path: Path) -> None:
        tab_id = f"{self.PREFIX}-stopped"
        late = "typed into a run that was stopped"
        agent, printer, _server = _start_run(
            tmp_path, tab_id, "first task", late_prompt=late, stop_first=True,
        )
        assert agent.prompts == ["first task"]
        assert len(_events_of(printer, "clear")) == 1
        st = agent_state.find_by_tab(tab_id)
        assert st is not None
        assert st.last_user_prompt == "first task"
        assert st.pending_user_messages == []
        assert not any(
            "No model available" in str(r.get("text", ""))
            for r in _events_of(printer, "result")
        )

    def test_thread_is_never_started_under_state_lock(self, tmp_path: Path) -> None:
        # The re-dispatch happens from the first run's worker: it must
        # not hold ``STATE_LOCK`` while ``_cmd_run`` runs (lock order),
        # which a blocked lookup from another thread would reveal.
        tab_id = f"{self.PREFIX}-lock"
        late = "lock-order probe"
        seen_locked: list[bool] = []
        stop = threading.Event()

        def probe() -> None:
            while not stop.is_set():
                got = agent_state.STATE_LOCK.acquire(timeout=5.0)
                if got:
                    agent_state.STATE_LOCK.release()
                seen_locked.append(not got)
                time.sleep(0.005)

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        try:
            _agent, printer, _server = _start_run(
                tmp_path, tab_id, "first task", late_prompt=late,
            )
        finally:
            stop.set()
            prober.join(timeout=10)
        self._assert_redispatched(printer, tab_id, late)
        assert not any(seen_locked)
