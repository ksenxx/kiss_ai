# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the pre-run retire of a carried-over worktree is shielded from the Stop watchdog.

A tab whose previous worktree task is still pending retires that
worktree (squash-merge into the main tree, or ``git worktree remove``)
at the start of its next run, on the task thread, before ``agent.run``.
That rewrite of the user's checkout ran without the
``is_merging``/``merge_thread`` claim the post-task merge holds, so a
Stop pressed while it ran (the tab shows a spinner and nothing else)
armed the watchdog, which one second later injected
``KeyboardInterrupt`` into the git sequence and left the checkout
half-merged.

The fix holds the merge claim around the retire — ``_state_owns_thread``
then refuses the injection — and honours a Stop that arrived meanwhile
cooperatively right after it.

The test drives the REAL ``_run_task`` worker, the REAL stop watchdog,
a REAL git repository and the REAL main-tree claim.  The agent's retire
step presses Stop and dwells past the watchdog's grace period,
recording whether it ran to completion; its LLM loop records whether it
was ever entered.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        self.events.append(dict(event))
        super().broadcast(event)


class _StopDuringRetireAgent(WorktreeSorcarAgent):
    """Agent whose pending-worktree retire presses Stop and outlasts the watchdog's grace."""

    server: VSCodeServer
    tab_id: str
    retire_completed = False
    run_entered = False
    claimed_during_retire = False

    def _retire_previous_worktree(self) -> str | None:
        """Press Stop, dwell 1.6 s (an injection would land here), drop the pending worktree."""
        state = agent_state.find_by_tab(self.tab_id)
        self.claimed_during_retire = bool(
            state is not None
            and state.is_merging
            and state.merge_thread is threading.current_thread()
        )
        self.server._stop_task(self.tab_id)
        time.sleep(1.6)
        self._wt = None
        self.retire_completed = True
        return None

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Record that the loop was reached (it must not be after a Stop)."""
        self.run_entered = True
        return "unreachable"


def _init_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="retire-shield-"))
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, env=env)
    (repo / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    return repo


def test_stop_during_pre_run_retire_is_not_injected() -> None:
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    repo = _init_repo()
    tab_id = "retire-shield-tab"
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    agent = _StopDuringRetireAgent("Sorcar VS Code")
    agent.server = server
    agent.tab_id = tab_id
    # The previous run on this tab left its worktree pending.
    agent._wt = GitWorktree(
        repo_root=repo,
        branch="kiss/pending-task",
        original_branch="main",
        wt_dir=repo / ".kiss-worktrees" / "pending-task",
    )
    # ``_cmd_run`` installs the stop event and the thread before start.
    state = AgentState(
        f"pre-{tab_id}",
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
    )
    agent_state.register(state)
    worker = threading.Thread(
        target=server._run_task,
        args=(
            {
                "type": "run",
                "tabId": tab_id,
                "prompt": "next task on the same tab",
                "workDir": str(repo),
                "model": models[0],
                "useWorktree": True,
                "classifyTasks": False,
                "autoCommit": False,
                "_state_key": state.task_id,
            },
        ),
        daemon=True,
    )
    state.task_thread = worker
    try:
        worker.start()
        worker.join(timeout=30)
        assert not worker.is_alive(), "worker never finished"
    finally:
        with agent_state.STATE_LOCK:
            stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == tab_id]
        for key in stale:
            agent_state.unregister(key)

    assert agent.retire_completed, "the Stop watchdog interrupted the retire"
    assert agent.claimed_during_retire, "retire ran without the merge claim"
    assert not agent.run_entered, "a pending Stop must end the run before the loop"
    assert not state.is_merging and state.merge_thread is None
    results = [e for e in printer.events if e.get("type") == "result"]
    assert [e.get("text") for e in results] == ["Task stopped by user"], results
    assert {"type": "status", "running": False, "tabId": tab_id} in printer.events
    head = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == "main"
