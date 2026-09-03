# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix #2: automatic preserve paths must fail closed on the marker write.

``WorktreeSorcarAgent._release_worktree`` (new task / new chat with a
worktree that cannot be auto-committed) and
``_preserve_pending_worktree_for_review`` (stopped/failed task, tab
close) deliberately leave the worktree on disk and write the durable
``branch.<b>.kiss-preserve`` marker so a later
``GitWorktreeOps.reclaim_orphaned_worktrees`` — in a fresh process,
after this one died — does not squash-merge and delete work the user
never accepted.  Before the fix both paths ignored the marker write's
result and dropped the only in-memory claim (``self._wt``) anyway, so
a transient ``git config`` failure (``.git/config.lock`` held) turned
"kept for review" into "silently published on the next restart".

The owner really exits here: each scenario runs the agent in a real
child process that holds a real ``.git/config.lock`` during the first
preserve attempt, retries the way a later user action would, and then
exits; the parent process then runs the real reclaim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0902-fix-marker-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


_CHILD = """
    import json, os
    from pathlib import Path
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

    repo = Path({repo!r})
    agent = WorktreeSorcarAgent("audit0902-fix-marker")
    # Auto-commit OFF + an uncommitted file: every automatic path must
    # preserve the worktree instead of committing/merging it.
    agent.auto_commit_enabled = False
    assert agent._try_setup_worktree(repo, str(repo)) is not None
    wt = agent._wt
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\\n", encoding="utf-8")
    agent._pending_review = {pending_review}

    lock = repo / ".git" / "config.lock"
    lock.write_text("")
    try:
        first = agent.{first_call}
    finally:
        lock.unlink()
    out = {{
        "branch": wt.branch,
        "wt_dir": str(wt.wt_dir),
        "first_result": repr(first),
        "claim_kept": agent._wt is wt,
        "pending_review_after_failure": agent._pending_review,
        "marker_after_failure": (
            GitWorktreeOps._load_branch_config(repo, wt.branch, "kiss-preserve") == "1"
        ),
        # Review-2 #1(c): the failed write leaves a process-local claim
        # that load_preserve_marker (hence reclaim) already honours.
        "preserved_in_process_after_failure": (
            GitWorktreeOps.load_preserve_marker(repo, wt.branch)
        ),
        "warning": agent._merge_conflict_warning or "",
    }}
    # The retry a later user action performs once the claim is kept.
    if agent._wt is not None:
        out["retry_result"] = repr(agent.{retry_call})
    out["claim_kept_after_retry"] = agent._wt is not None
    out["marker_after_retry"] = GitWorktreeOps.load_preserve_marker(repo, wt.branch)
    print("RESULT " + json.dumps(out))
"""


def _run_owner(
    env: IsolatedKissHome, *, pending_review: bool, first_call: str, retry_call: str,
) -> dict:
    """Run the owner scenario in a child process and return its report."""
    code = textwrap.dedent(_CHILD).format(
        repo=str(env.repo),
        pending_review=pending_review,
        first_call=first_call,
        retry_call=retry_call,
    )
    child_env = dict(os.environ, KISS_HOME=str(env.kiss_home))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(env.repo), timeout=180, env=child_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    report: dict = json.loads(line[len("RESULT "):])
    return report


def _assert_kept_and_not_reclaimed(env: IsolatedKissHome, report: dict) -> None:
    """Common assertions: claim kept on failure, marker durable, reclaim skips."""
    assert report["marker_after_failure"] is False
    assert report["preserved_in_process_after_failure"] is True
    assert report["claim_kept"] is True, (
        "the in-memory claim was dropped although the marker write failed: "
        f"{report}"
    )
    assert report["pending_review_after_failure"] is True
    assert "still pending" in report["warning"], report["warning"]
    assert report["marker_after_retry"] is True
    assert report["claim_kept_after_retry"] is False

    wt_dir = Path(report["wt_dir"])
    branch = report["branch"]
    assert wt_dir.is_dir()
    # The owner has exited: a fresh process's reclaim must respect the
    # (now durable) keep decision.
    assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo) == 0
    assert wt_dir.is_dir(), "the parked worktree was deleted by the reclaim"
    assert GitWorktreeOps.branch_exists(env.repo, branch)
    assert not (env.repo / "agent.txt").exists(), "unaccepted work was published"


def test_release_worktree_keeps_claim_until_marker_is_durable(
    env: IsolatedKissHome,
) -> None:
    """``_release_worktree`` (new task / new chat) must not drop ``_wt``
    when the preserve marker cannot be written."""
    report = _run_owner(
        env,
        pending_review=False,
        first_call="_release_worktree()",
        retry_call="_retire_previous_worktree()",
    )
    assert report["first_result"] == "None"
    _assert_kept_and_not_reclaimed(env, report)


def test_preserve_for_review_keeps_claim_until_marker_is_durable(
    env: IsolatedKissHome,
) -> None:
    """``_preserve_pending_worktree_for_review`` (stop / tab close) must not
    drop ``_wt`` when the preserve marker cannot be written."""
    report = _run_owner(
        env,
        pending_review=True,
        first_call="_preserve_pending_worktree_for_review()",
        retry_call="_preserve_pending_worktree_for_review()",
    )
    # The worktree WAS preserved (it is on disk); only the claim is kept.
    assert report["first_result"] == "True"
    assert report["retry_result"] == "True"
    _assert_kept_and_not_reclaimed(env, report)


def test_new_task_setup_runs_directly_while_claim_is_unretired(
    env: IsolatedKissHome,
) -> None:
    """A new task's worktree setup must not overwrite a claim the retire
    could not release; it falls back to direct execution instead."""
    agent = WorktreeSorcarAgent("audit0902-fix-marker-setup")
    agent.auto_commit_enabled = False
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")

    lock = env.repo / ".git" / "config.lock"
    lock.write_text("")
    try:
        second = agent._try_setup_worktree(env.repo, str(env.repo))
    finally:
        lock.unlink()

    assert second is None, "a new worktree replaced the unretired claim"
    assert agent._wt is wt
    assert agent._pending_review is True
    assert GitWorktreeOps._load_branch_config(env.repo, wt.branch, "kiss-preserve") is None
    assert "still pending" in (agent._merge_conflict_warning or "")
    # Only the one parked worktree is registered besides the main tree.
    registered = [b for _, b in GitWorktreeOps.registered_worktrees(env.repo)]
    assert registered.count(wt.branch) == 1
    assert len([b for b in registered if b.startswith("kiss/wt-")]) == 1

    # With git config writable again the next setup retires it for real
    # and hands out a fresh worktree.
    third = agent._try_setup_worktree(env.repo, str(env.repo))
    assert third is not None
    assert agent._wt is not None and agent._wt is not wt
    assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)
    assert wt.wt_dir.is_dir()
    assert agent._pending_review is False


def test_same_process_reclaim_honours_the_volatile_claim(env: IsolatedKissHome) -> None:
    """Review-2 #1(c): while the marker write has failed, the keep decision
    is held process-locally and ``reclaim_orphaned_worktrees`` respects it
    even with an empty exclusion set (the own-pid exemption is no bypass);
    the successful retry moves the decision to disk and drops the claim."""
    agent = WorktreeSorcarAgent("audit0902-fix2-volatile")
    agent.auto_commit_enabled = False
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")

    lock = env.repo / ".git" / "config.lock"
    lock.write_text("")
    try:
        assert agent._release_worktree() is None
        assert agent._wt is wt
        assert GitWorktreeOps._load_branch_config(env.repo, wt.branch, "kiss-preserve") is None
        assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch), (
            "the process-local keep decision is invisible to load_preserve_marker"
        )
        assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo, exclude_branches=set()) == 0
        assert wt.wt_dir.is_dir()
        assert GitWorktreeOps.branch_exists(env.repo, wt.branch)
        assert not (env.repo / "agent.txt").exists(), "unaccepted work was published"
    finally:
        lock.unlink()

    assert agent._retire_previous_worktree() is None
    assert agent._wt is None
    assert GitWorktreeOps._load_branch_config(env.repo, wt.branch, "kiss-preserve") == "1"
    # The durable marker replaced the volatile claim: without the config
    # entry nothing in this process still reports the branch preserved.
    subprocess.run(
        ["git", "-C", str(env.repo), "config", "--unset", f"branch.{wt.branch}.kiss-preserve"],
        check=True, capture_output=True,
    )
    assert not GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)


def test_marker_success_path_still_drops_claim(env: IsolatedKissHome) -> None:
    """Regression guard: with a writable config the paths behave as before."""
    agent = WorktreeSorcarAgent("audit0902-fix-marker-ok")
    agent.auto_commit_enabled = False
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")
    assert agent._release_worktree() is None
    assert agent._wt is None
    assert agent._pending_review is False
    assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)
    assert "still pending" not in (agent._merge_conflict_warning or "")
