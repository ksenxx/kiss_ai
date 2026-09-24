# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
""""Auto-commit and merge" of a huge worktree finishes and frees the daemon.

Field incident (``~/.kiss/kiss-web-stderr.log``, 2026-09-23 15:11 UTC):
a task had copied 38,081 benchmark result files (38.5 million lines)
into its worktree and then failed on budget.  Clicking "Auto-commit and
merge" ran the commit-message step, which read the entire
``git diff --cached`` into memory (the daemon reached 45 GB) and never
returned.  The merge thread held the tab's ``is_merging`` claim, so
every main-tree prompt was refused with "A worktree merge is in
progress. Wait for it to finish before starting a task." and the claim
could not self-heal because its thread was alive.

This drives the real ``worktreeAction`` command against a real
repository whose worktree stages several megabytes of generated
content, with the commit-message model unreachable
(:class:`OfflineFastModel`), and checks that the merge lands, the claim
is released, the main-tree gate opens, and the merge is logged.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.git_worktree import (
    COMMIT_MESSAGE_DIFF_LIMIT_BYTES,
    GitWorktree,
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _wt_merge_on_repo
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome, OfflineFastModel

_TAB = "huge-merge-tab"
_RESULTS_DIR = "results"
_FILES = 40
_LINES_PER_FILE = 5_000


class _Env:
    """Isolated KISS_HOME, a repo with a pending huge worktree, a server."""

    def __init__(self) -> None:
        self.isolated = IsolatedKissHome("kiss-huge-merge-")
        self.repo: Path = self.isolated.repo
        self.offline = OfflineFastModel()
        self.offline.__enter__()
        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]
        self.state = agent_state.AgentState(
            "task-huge-merge", tab_id=_TAB, server_owned=True,
        )
        self.state.agent = self._pending_worktree_agent()
        self.state.use_worktree = True
        agent_state.register(self.state)
        self.server.tab_registry.open_tab(_TAB, "huge")

    def _pending_worktree_agent(self) -> WorktreeSorcarAgent:
        branch = "kiss/wt-huge-merge"
        original = GitWorktreeOps.current_branch(self.repo)
        assert original is not None
        wt_dir = self.repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        assert GitWorktreeOps.save_original_branch(self.repo, branch, original)
        out = wt_dir / _RESULTS_DIR
        out.mkdir()
        for i in range(_FILES):
            (out / f"run-{i:03d}.log").write_text(
                "".join(f"file {i} line {j} " + "x" * 40 + "\n" for j in range(_LINES_PER_FILE)),
                encoding="utf-8",
            )
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = GitWorktree(
            repo_root=self.repo, branch=branch, original_branch=original, wt_dir=wt_dir,
        )
        return agent

    def cleanup(self) -> None:
        for state in agent_state.snapshot():
            agent_state.unregister(state.task_id, state)
        self.offline.__exit__(None, None, None)
        self.isolated.cleanup()


@pytest.fixture
def env() -> Iterator[_Env]:
    e = _Env()
    try:
        yield e
    finally:
        e.cleanup()


def test_merge_of_a_huge_worktree_lands_and_releases_the_claim(
    env: _Env, caplog: pytest.LogCaptureFixture,
) -> None:
    wt_dir = env.state.agent._wt.wt_dir  # type: ignore[union-attr]
    GitWorktreeOps.stage_all(wt_dir)
    full_patch = _git("diff", "--cached", "--shortstat", cwd=wt_dir).stdout
    assert f"{_FILES} files changed, {_FILES * _LINES_PER_FILE} insertions" in full_patch
    # ~60 bytes per generated line: well past the commit-message cap.
    assert _FILES * _LINES_PER_FILE * 60 > 10 * COMMIT_MESSAGE_DIFF_LIMIT_BYTES

    finished = threading.Event()
    result: dict[str, Any] = {}

    def _merge() -> None:
        try:
            env.server._handle_command({
                "type": "worktreeAction",
                "action": "merge",
                "tabId": _TAB,
                "workDir": str(env.repo),
            })
        finally:
            finished.set()

    with caplog.at_level(logging.INFO, logger="kiss.server.merge_flow"):
        thread = threading.Thread(target=_merge, name="interactive-merge")
        thread.start()
        assert finished.wait(timeout=180), "the merge did not finish"
        thread.join(timeout=10)

    result_events = [e for e in env.events if e.get("type") == "worktree_result"]
    assert result_events, [e.get("type") for e in env.events]
    result = result_events[-1]
    assert result.get("success") is True, result
    assert "Successfully merged" in str(result.get("message")), result

    with env.server._state_lock:
        assert env.state.is_merging is False
        assert env.state.merge_thread is None
    assert _wt_merge_on_repo(env.state, env.repo) is False

    head_files = _git("ls-tree", "-r", "--name-only", "HEAD", cwd=env.repo)
    assert head_files.stdout.count(f"{_RESULTS_DIR}/run-") == _FILES
    # The commit-message step ran (and, with no model reachable, used
    # its documented fallback) rather than being skipped.
    subject = _git("log", "-1", "--format=%s", cwd=env.repo).stdout.strip()
    assert subject == "kiss: auto-commit agent work", subject

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith(f"Worktree merge started: tab={_TAB} worktree=") for m in messages
    ), messages
    assert any(
        m.startswith(f"Worktree merge finished: tab={_TAB} success=True elapsed=")
        for m in messages
    ), messages
