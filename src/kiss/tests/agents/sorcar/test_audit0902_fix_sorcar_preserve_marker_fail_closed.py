# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix #2: automatic preserve paths must fail closed on the marker write.

``ChatSorcarAgent._retire_previous_worktree`` (new task / new chat
with a worktree the agent never merged) and
``_preserve_pending_worktree_for_review`` (stopped/failed task, tab
close) deliberately leave the worktree on disk and write the durable
preserve marker (a file under the repo's git dir) so a later
``GitWorktreeOps.reclaim_orphaned_worktrees`` — in a fresh process,
after this one died — does not squash-merge and delete work the user
never accepted.  The preserve commits the work to its ``kiss/wt-*``
branch and removes the worktree directory; the branch is the only
copy of the work.  Before the fix both paths ignored the marker write's
result and dropped the only in-memory claim (``self._wt``) anyway, so
a transient marker-write failure turned "kept for review" into
"silently published on the next restart".

The owner really exits here: each scenario runs the agent in a real
child process whose first preserve attempt cannot write the marker (a
regular file sits where the marker directory ``.git/kiss-preserve``
goes), retries the way a later user action would, and then exits; the
parent process then runs the real reclaim.
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

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
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
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    repo = Path({repo!r})
    agent = ChatSorcarAgent("audit0902-fix-marker")
    # An uncommitted file the agent never merged: every automatic path
    # must preserve the worktree instead of merging it.
    assert agent._try_setup_worktree(repo, str(repo)) is not None
    wt = agent._wt
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\\n", encoding="utf-8")

    lock = repo / ".git" / "kiss-preserve"
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
        "pending_after_failure": agent._wt_pending,
        "marker_after_failure": (
            GitWorktreeOps._preserve_marker_on_disk(repo, wt.branch)
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
    env: IsolatedKissHome, *, first_call: str, retry_call: str,
) -> dict:
    """Run the owner scenario in a child process and return its report."""
    code = textwrap.dedent(_CHILD).format(
        repo=str(env.repo),
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


def _unlink_marker(repo: Path, branch: str) -> None:
    """Remove *branch*'s durable preserve marker file."""
    marker = GitWorktreeOps._preserve_marker_path(repo, branch)
    assert marker is not None
    marker.unlink()


def _branch_has_agent_file(repo: Path, branch: str) -> bool:
    """Whether ``agent.txt`` (the unaccepted work) is committed on *branch*."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{branch}:agent.txt"],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and proc.stdout == "unaccepted work\n"


def _assert_kept_and_not_reclaimed(env: IsolatedKissHome, report: dict) -> None:
    """Common assertions: claim kept on failure, marker durable, reclaim skips."""
    assert report["marker_after_failure"] is False
    assert report["preserved_in_process_after_failure"] is True
    assert report["claim_kept"] is True, (
        "the in-memory claim was dropped although the marker write failed: "
        f"{report}"
    )
    assert report["pending_after_failure"] is True
    assert "still pending" in report["warning"], report["warning"]
    assert report["marker_after_retry"] is True
    assert report["claim_kept_after_retry"] is False

    branch = report["branch"]
    assert _branch_has_agent_file(env.repo, branch)
    # The owner has exited: a fresh process's reclaim must respect the
    # (now durable) keep decision.
    assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo) == 0
    assert GitWorktreeOps.branch_exists(env.repo, branch), (
        "the parked branch was deleted by the reclaim"
    )
    assert _branch_has_agent_file(env.repo, branch)
    assert not (env.repo / "agent.txt").exists(), "unaccepted work was published"


def test_retire_previous_worktree_keeps_claim_until_marker_is_durable(
    env: IsolatedKissHome,
) -> None:
    """``_retire_previous_worktree`` (new task / new chat) must not drop
    ``_wt`` when the preserve marker cannot be written."""
    report = _run_owner(
        env,
        first_call="_retire_previous_worktree()",
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
        first_call="_preserve_pending_worktree_for_review()",
        retry_call="_preserve_pending_worktree_for_review()",
    )
    # The work WAS preserved (committed on its branch); only the claim is kept.
    assert report["first_result"] == "True"
    assert report["retry_result"] == "True"
    _assert_kept_and_not_reclaimed(env, report)


def test_create_worktree_keeps_the_unretired_claim(
    env: IsolatedKissHome,
) -> None:
    """A new task's ``create_worktree`` must not overwrite a claim the
    retire could not release; it reports the pending worktree as the
    active one instead."""
    agent = ChatSorcarAgent("audit0902-fix-marker-setup")
    agent.work_dir = str(env.repo)
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")

    lock = env.repo / ".git" / "kiss-preserve"
    lock.write_text("")
    try:
        agent._retire_previous_worktree()
        second = agent.create_worktree()
    finally:
        lock.unlink()

    assert "already active" in second, second
    assert agent._wt is wt
    assert agent._wt_pending
    assert not GitWorktreeOps._preserve_marker_on_disk(env.repo, wt.branch)
    assert "still pending" in (agent._merge_conflict_warning or "")
    # The work is parked on its branch; no second worktree was made.
    assert _branch_has_agent_file(env.repo, wt.branch)
    registered = [b for _, b in GitWorktreeOps.registered_worktrees(env.repo)]
    assert [b for b in registered if b.startswith("kiss/wt-")] == []

    # With git config writable again the next retire parks it for real
    # and create_worktree hands out a fresh worktree.
    agent._retire_previous_worktree()
    assert agent._wt is None
    third = agent.create_worktree()
    assert third.startswith("Created worktree"), third
    assert agent._wt is not None and agent._wt is not wt
    assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)
    assert _branch_has_agent_file(env.repo, wt.branch)
    assert agent.discard().startswith("Discarded")


def test_same_process_reclaim_honours_the_volatile_claim(env: IsolatedKissHome) -> None:
    """Review-2 #1(c): while the marker write has failed, the keep decision
    is held process-locally and ``reclaim_orphaned_worktrees`` respects it
    even with an empty exclusion set (the own-pid exemption is no bypass);
    the successful retry moves the decision to disk and drops the claim."""
    agent = ChatSorcarAgent("audit0902-fix2-volatile")
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")

    lock = env.repo / ".git" / "kiss-preserve"
    lock.write_text("")
    try:
        agent._retire_previous_worktree()
        assert agent._wt is wt
        assert not GitWorktreeOps._preserve_marker_on_disk(env.repo, wt.branch)
        assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch), (
            "the process-local keep decision is invisible to load_preserve_marker"
        )
        assert GitWorktreeOps.reclaim_orphaned_worktrees(env.repo, exclude_branches=set()) == 0
        assert GitWorktreeOps.branch_exists(env.repo, wt.branch)
        assert _branch_has_agent_file(env.repo, wt.branch)
        assert not (env.repo / "agent.txt").exists(), "unaccepted work was published"
    finally:
        lock.unlink()

    agent._retire_previous_worktree()
    assert agent._wt is None
    assert GitWorktreeOps._preserve_marker_on_disk(env.repo, wt.branch)
    # The durable marker replaced the volatile claim: without the marker
    # file nothing in this process still reports the branch preserved.
    _unlink_marker(env.repo, wt.branch)
    assert not GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)


def test_marker_success_path_still_drops_claim(env: IsolatedKissHome) -> None:
    """Regression guard: with a writable config the paths behave as before."""
    agent = ChatSorcarAgent("audit0902-fix-marker-ok")
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("unaccepted work\n", encoding="utf-8")
    agent._retire_previous_worktree()
    assert agent._wt is None
    assert not agent._wt_pending
    assert GitWorktreeOps.load_preserve_marker(env.repo, wt.branch)
    assert "still pending" not in (agent._merge_conflict_warning or "")
