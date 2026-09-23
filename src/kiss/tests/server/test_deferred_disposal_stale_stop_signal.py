# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stopped task's deferred tab disposal must not wedge the main tree.

User-reported bug: after stopping a long worktree task and closing its
tab, every later non-worktree prompt on the same repository (for
example ``cron list``, which the classifier routes to the main tree)
was refused with "A worktree merge is in progress. Wait for it to
finish before starting a task." — with no merge running anywhere.

Root cause (``~/.kiss/kiss-web-stderr.log``): the runner's end-of-run
cleanup (``_TaskRunnerMixin._run_task``) ran the deferred disposal
(``_dispose_if_closed`` → ``_teardown_tab_resources`` →
``retire_for_disposal`` → auto-commit → commit-message LLM call) on
the task thread while the thread still carried the stopped task's
stop signal.  The model layer turns ANY request failure into
``KeyboardInterrupt("Agent stop requested")`` when the thread's stop
signal is set, and both ``commit_message._run_oneshot_llm`` and
``_teardown_tab_resources`` catch only ``Exception`` — so the
interrupt escaped before ``state.is_merging`` was cleared and the
disposal claim stayed set forever on a registered ``use_worktree``
state whose repo root is the user's repository.
:func:`kiss.server.task_runner._wt_merge_on_repo` then refused every
main-tree run on that repo.

The tests use a real git repository, the real server, the real task
runner driving a real ``run`` command, the real worktree agent and
the real commit-message model adapter — whose request is made to fail
deterministically by pointing the Anthropic client at a closed local
port.  Two things are substituted, as in
:mod:`test_worktree_stop_preserves_main`: the agent's parent-class
``run`` is a stub that does what the user did (write a file, close the
tab, set the task's stop event and raise the stop interrupt — the
effect of ``_stop_task`` on the task thread), and
``printer.broadcast`` is captured into a list.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core import config as config_module
from kiss.core import stop_signal
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _wt_merge_on_repo
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_MERGE_IN_PROGRESS = "A worktree merge is in progress"


def _closed_local_port() -> int:
    """Return a TCP port on 127.0.0.1 that nothing is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Env:
    """Isolated KISS_HOME + repo + a real server whose commit-message LLM
    call fails deterministically (Anthropic client aimed at a closed port)."""

    def __init__(self) -> None:
        self.isolated = IsolatedKissHome("kiss-stale-stop-disposal-")
        self.repo: Path = self.isolated.repo
        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self.server.printer.broadcast = self._capture  # type: ignore[assignment]
        # ``get_fast_model()`` picks Anthropic first when its key is
        # configured; the SDK reads ``ANTHROPIC_BASE_URL`` when the
        # client is built, so the one-shot commit-message request fails
        # with a connection error instead of reaching the network.
        self._saved_key = config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY
        self._saved_base_url = os.environ.get("ANTHROPIC_BASE_URL")
        config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY = "sk-ant-test-not-a-real-key"
        os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{_closed_local_port()}"
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def _capture(self, event: dict[str, Any]) -> None:
        with self._events_lock:
            self.events.append(event)

    def event_types(self) -> list[str]:
        with self._events_lock:
            return [str(e.get("type", "")) for e in self.events]

    def error_texts(self) -> list[str]:
        with self._events_lock:
            return [str(e.get("text", "")) for e in self.events if e.get("type") == "error"]

    def cleanup(self) -> None:
        self._parent_class.run = self._original_run
        stop_signal.set_thread_stop_event(None)
        config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY = self._saved_key
        if self._saved_base_url is None:
            os.environ.pop("ANTHROPIC_BASE_URL", None)
        else:
            os.environ["ANTHROPIC_BASE_URL"] = self._saved_base_url
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        self.isolated.cleanup()

    def run_command(
        self, tab_id: str, prompt: str, *, use_worktree: bool,
    ) -> agent_state.AgentState:
        """Drive a real ``run`` command and wait for its task thread."""
        self.server._handle_command({
            "type": "run",
            "prompt": prompt,
            "workDir": str(self.repo),
            "tabId": tab_id,
            "useWorktree": use_worktree,
            "autoCommit": True,
            "model": "",
        })
        state = agent_state.find_by_tab(tab_id)
        assert state is not None, f"run was refused: {self.error_texts()}"
        thread = state.task_thread
        if thread is not None:
            thread.join(timeout=120)
            assert not thread.is_alive(), "the task thread did not finish"
        return state

    def park_pending_work(
        self, tab_id: str, *, closed: bool, deferred_merge: bool,
    ) -> WorktreeSorcarAgent:
        """Register an idle tab whose agent holds a finished task's
        uncommitted worktree work with auto-commit ON.

        ``closed`` marks the tab closed with the work left for review
        (the deferred disposal will commit and keep it);
        ``deferred_merge`` marks the work as waiting for the main tree
        to free up (``_merge_deferred_worktrees`` will commit and merge
        it).  Either way the commit message is an LLM call.
        """
        agent = WorktreeSorcarAgent(f"stale-stop-{tab_id}")
        agent.auto_commit_enabled = True
        assert agent._try_setup_worktree(self.repo, str(self.repo)) is not None
        wt = agent._wt
        assert wt is not None
        (wt.wt_dir / "agent.txt").write_text("stopped work\n", encoding="utf-8")
        agent._pending_review = closed
        agent.printer = self.server.printer  # type: ignore[attr-defined]
        with self.server._state_lock:
            state = agent_state.AgentState(
                f"task-for-{tab_id}", tab_id=tab_id, server_owned=True, agent=agent,
            )
            state.use_worktree = True
            state.auto_commit_mode = True
            state.frontend_closed = closed
            if deferred_merge:
                state.wt_merge_deferred_branch = wt.branch
            agent_state.register(state)
        if not closed:
            self.server.tab_registry.open_tab(tab_id, "stale-stop")
        return agent


@pytest.fixture
def env() -> Iterator[_Env]:
    e = _Env()
    try:
        yield e
    finally:
        e.cleanup()


def _wt_branches(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "kiss/wt-*"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line.strip().lstrip("+* ").strip() for line in out.splitlines() if line.strip()]


def _file_on_branch(repo: Path, branch: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{branch}:{path}"],
        capture_output=True, text=True, check=True,
    ).stdout


def _merging_states(repo: Path) -> list[agent_state.AgentState]:
    return [s for s in agent_state.snapshot() if _wt_merge_on_repo(s, repo)]


def test_stopped_task_with_closed_tab_does_not_wedge_main_tree_runs(env: _Env) -> None:
    """The user's sequence, end to end: a worktree task writes a file, the
    user closes the tab and presses Stop, the runner disposes the tab at
    task end, and the next main-tree prompt must run."""
    tab_id = "stale-stop-e2e"

    def stub_run(self_agent: object, **kwargs: object) -> str:
        work_dir = kwargs.get("work_dir")
        assert isinstance(work_dir, str) and work_dir
        (Path(work_dir) / "notes.txt").write_text("partial\n", encoding="utf-8")
        # The user closes the tab while the task runs (busy → deferred)…
        env.server._close_tab(tab_id)
        # …and presses Stop: the runner's stop path sets the task's stop
        # event and interrupts the run.
        event = stop_signal.get_thread_stop_event()
        assert event is not None
        event.set()
        raise KeyboardInterrupt("Agent stop requested")

    env._parent_class.run = stub_run
    state = env.run_command(tab_id, "write some notes", use_worktree=True)

    assert "task_stopped" in env.event_types(), env.event_types()
    assert state.is_merging is False, "the deferred disposal left its claim set"
    assert state.merge_thread is None
    assert _merging_states(env.repo) == [], "a phantom merge is still reported"
    assert agent_state.find_by_tab(tab_id) is None, "the closed tab was not disposed"
    # The stopped work survived on its branch and never reached main.
    branches = _wt_branches(env.repo)
    assert len(branches) == 1, branches
    assert _file_on_branch(env.repo, branches[0], "notes.txt") == "partial\n"
    assert not (env.repo / "notes.txt").exists()

    # The user-visible symptom: a main-tree prompt on the same repo.
    def stub_ok(self_agent: object, **kwargs: object) -> str:
        return "ok"

    env._parent_class.run = stub_ok
    env.run_command("stale-stop-next", "cron list", use_worktree=False)
    refused = [t for t in env.error_texts() if _MERGE_IN_PROGRESS in t]
    assert refused == [], refused
    assert "task_done" in env.event_types(), env.event_types()


def test_teardown_releases_claim_when_retire_raises_stop_interrupt(env: _Env) -> None:
    """A retire that unwinds with the stop interrupt (a stale stop signal
    on the disposing thread) must release the disposal claim, keep the
    state registered while the worktree is still pending, and let a later
    disposal attempt finish the job."""
    tab_id = "stale-stop-direct"
    agent = env.park_pending_work(tab_id, closed=True, deferred_merge=False)
    wt = agent._wt
    assert wt is not None
    state = agent_state.find_by_tab(tab_id)
    assert state is not None

    stale = threading.Event()
    stale.set()
    stop_signal.set_thread_stop_event(stale)
    try:
        with pytest.raises(KeyboardInterrupt):
            env.server._dispose_if_closed(tab_id)
    finally:
        stop_signal.set_thread_stop_event(None)

    assert state.is_merging is False, "the disposal claim survived the interrupt"
    assert state.merge_thread is None
    assert _merging_states(env.repo) == []
    # The retire did not finish: the worktree is still pending, so the
    # state stays registered to protect it (exactly like a failed marker).
    assert agent._wt is wt
    assert agent_state.find_by_tab(tab_id) is state
    assert state.busy() is False
    assert wt.wt_dir.is_dir()

    # Without the stale signal the same disposal completes: the work is
    # committed (fallback commit message), kept for review, and dropped.
    env.server._dispose_if_closed(tab_id)
    assert agent._wt is None
    assert agent_state.find_by_tab(tab_id) is None
    assert wt.branch in _wt_branches(env.repo)
    assert _file_on_branch(env.repo, wt.branch, "agent.txt") == "stopped work\n"
    assert not (env.repo / "agent.txt").exists()


def test_stopped_main_tree_task_still_merges_deferred_worktrees(env: _Env) -> None:
    """A worktree whose merge waited for the main tree (another tab was
    running there) is merged when that main-tree task ends — also when
    the user stopped it: its stop signal must not leak into the
    deferred merge's commit-message call and lose the merge."""
    waiting_tab = "stale-stop-deferred-wt"
    agent = env.park_pending_work(waiting_tab, closed=False, deferred_merge=True)
    wt = agent._wt
    assert wt is not None

    def stub_stop(self_agent: object, **kwargs: object) -> str:
        event = stop_signal.get_thread_stop_event()
        assert event is not None
        event.set()
        raise KeyboardInterrupt("Agent stop requested")

    env._parent_class.run = stub_stop
    env.run_command("stale-stop-main", "look around", use_worktree=False)
    assert "task_stopped" in env.event_types(), env.event_types()

    results = [
        e for e in env.events
        if e.get("type") == "worktree_result" and e.get("tabId") == waiting_tab
    ]
    assert len(results) == 1, env.event_types()
    assert results[0].get("success") is True, results[0]
    assert agent._wt is None, "the deferred merge was lost"
    assert (env.repo / "agent.txt").read_text(encoding="utf-8") == "stopped work\n"
    assert _wt_branches(env.repo) == []
    assert _merging_states(env.repo) == []
