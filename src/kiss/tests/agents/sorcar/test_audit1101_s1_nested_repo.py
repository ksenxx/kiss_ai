# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A repository nested inside a task's worktree is not isolated by that worktree.

A worktree under ``<repo>/.kiss-worktrees/kiss_wt-*`` isolates the files
of its own repository.  A clone with its own ``.git`` directory created
inside it (a dependency checked out under the task's tree, a repository
a probe sets up under ``tmp/``) is a distinct repository whose ordinary
checkout that is.  The worktree gate and ``create_worktree`` decided by
the path name alone: a tracked file of the nested repository passed the
gate unguarded, and the mandatory ``create_worktree`` of that repository
did nothing, saying the calling task already isolated it (audit
2026-11-01, S1 F5).  Both now ask which repository owns the path: the
worktree's own files stay isolated by it, a nested repository gets its
own worktree, into which the work dir moves when it lies in that
repository.  A task working in the nested repository's own worktree is
isolated by that worktree, the nearest one of that repository, not by
the outermost worktree of the path, which is another repository's
(round 2, F6).

Real agents over real git repositories; nothing is mocked.  The
fixture's repository may itself lie inside a KISS worktree (pytest's
base directory under a task's tree), so the assertions ask the agents
which repository owns a path instead of reading the path's name.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import init_repo

pytestmark = posix_only("worktree paths are compared as POSIX paths here")

EDIT_TOOLS = frozenset({"Bash", "Read", "Write", "Edit", "create_worktree", "rollback"})


def _agent(name: str, work_dir: Path) -> SorcarAgent:
    """A host agent with the editing tools, working in *work_dir*."""
    agent = SorcarAgent(name)
    agent.work_dir = str(work_dir)
    agent._run_tool_names = EDIT_TOOLS
    return agent


@pytest.fixture
def task(tmp_path: Path) -> Iterator[tuple[SorcarAgent, Path, Path]]:
    """A task in its worktree of ``repo``, with an independent repository nested inside.

    Yields:
        ``(agent, worktree, nested)``: the agent whose work dir is the
        worktree, the worktree directory, and the nested repository
        (its own ``.git`` directory, one committed ``seed.txt``).
    """
    repo = tmp_path / "repo"
    init_repo(repo)
    agent = _agent("outer", repo)
    assert "Created worktree" in agent.create_worktree()
    try:
        worktree = Path(agent.work_dir)
        (wt,) = agent.worktrees
        assert wt.wt_dir.resolve() == worktree.resolve()
        assert wt.repo_root.resolve() == repo.resolve() == worktree.parent.parent.resolve()
        nested = worktree / "vendor" / "independent"
        init_repo(nested)
        assert (nested / ".git").is_dir()
        yield agent, worktree, nested
    finally:
        if agent._wt_pending:
            agent.discard()


def test_the_gate_guards_the_nested_repository_until_it_has_a_worktree(
    task: tuple[SorcarAgent, Path, Path]
) -> None:
    agent, worktree, nested = task
    # The worktree's own tracked file is isolated by it: no refusal.
    assert agent._guard_worktree("Write", {"file_path": str(worktree / "seed.txt")}) is None
    refusal = agent._guard_worktree("Write", {"file_path": str(nested / "seed.txt")})
    assert refusal is not None, "a tracked file of the nested repository's own checkout"
    assert f"inside the git repository {nested}" in refusal
    assert f"Call create_worktree({str(nested)!r}) first" in refusal
    # The nested repository gets a worktree of its own; the work dir stays.
    created = agent.create_worktree(str(nested))
    assert created.startswith("Created worktree "), created
    assert "The work dir is unchanged" in created
    assert agent.work_dir == str(worktree)
    (_, wt_nested) = agent.worktrees
    assert wt_nested.repo_root.resolve() == nested.resolve()
    assert wt_nested.wt_dir.parent.parent.resolve() == nested.resolve()
    assert wt_nested.wt_dir.parent.name == ".kiss-worktrees"
    # Its checkout paths are mapped into that worktree; the worktree's pass.
    refusal = agent._guard_worktree("Edit", {"file_path": str(nested / "seed.txt")})
    assert refusal is not None and f"Edit {wt_nested.wt_dir / 'seed.txt'} instead" in refusal
    assert agent._guard_worktree("Write", {"file_path": str(wt_nested.wt_dir / "x.txt")}) is None
    again = agent.create_worktree(str(nested / "deep" / "y.txt"))
    assert f"already active for {wt_nested.repo_root}: {wt_nested.wt_dir}" in again
    # A second call for the outer worktree itself still names it.
    assert "A worktree is already active: work dir" in agent.create_worktree()


def test_a_task_started_in_the_nested_repository_moves_into_its_own_worktree(
    task: tuple[SorcarAgent, Path, Path]
) -> None:
    _, worktree, nested = task
    child = _agent("in-nested", nested)
    try:
        # Its rollback would restore the nested repository's own checkout.
        rollback = child._guard_worktree("rollback", {})
        assert rollback is not None and f"the git repository {nested}" in rollback
        refusal = child._guard_worktree("Write", {"file_path": "seed.txt"})
        assert refusal is not None and "Call create_worktree() first" in refusal
        created = child.create_worktree()
        assert created.startswith("Created worktree "), created
        assert "The work dir is now" in created
        (wt,) = child.worktrees
        assert wt.repo_root.resolve() == nested.resolve()
        assert wt.wt_dir.parent.parent.resolve() == nested.resolve()
        assert Path(child.work_dir).resolve() == wt.wt_dir.resolve()
        assert child._guard_worktree("Write", {"file_path": "seed.txt"}) is None
        assert child._guard_worktree("rollback", {}) is None
        assert not (worktree / ".kiss-worktrees").exists(), "nothing was made for the outer repo"
    finally:
        if child._wt_pending:
            child.discard()


def test_a_sub_agent_in_the_nested_repository_s_worktree_is_isolated_by_it(
    task: tuple[SorcarAgent, Path, Path]
) -> None:
    """Inside the outer worktree, the nested repository's worktree isolates a task working in it."""
    _, worktree, nested = task
    parent = _agent("in-nested", nested)
    try:
        assert parent.create_worktree().startswith("Created worktree ")
        (wt,) = parent.worktrees
        # A sub-agent started in that worktree without the daemon's inheritance map.
        sub = _agent("sub-in-nested", wt.wt_dir)
        # A checkout path of the nested repository is remapped into the worktree, not
        # refused for a worktree of its own...
        assert sub._guard_worktree("Write", {"file_path": str(nested / "seed.txt")}) is None
        assert sub._guard_worktree("Write", {"file_path": "seed.txt"}) is None
        # ...and asking for one names the worktree the parent already has.
        note = sub.create_worktree(str(nested))
        assert "No new worktree needed" in note and str(wt.wt_dir) in note, note
        assert "already inside the worktree" in sub.create_worktree()
        assert not sub.worktrees and parent.worktrees == [wt]
        # The outer repository's checkout path is isolated by the outer worktree as before.
        assert sub._guard_worktree(
            "Write", {"file_path": str(worktree.parent.parent / "seed.txt")}
        ) is None
    finally:
        if parent._wt_pending:
            parent.discard()


def test_a_sub_agent_of_the_worktree_task_still_edits_the_worktree_in_place(
    task: tuple[SorcarAgent, Path, Path]
) -> None:
    _, worktree, _ = task
    sub = _agent("sub", worktree / "src")
    (worktree / "src").mkdir()
    assert sub._guard_worktree("Write", {"file_path": str(worktree / "seed.txt")}) is None
    assert sub._guard_worktree("Write", {"file_path": "new.py"}) is None
    assert sub._guard_worktree("rollback", {}) is None
    note = sub.create_worktree()
    assert "No new worktree needed" in note and "already inside the worktree" in note
    assert sub.work_dir == str(worktree / "src") and not sub.worktrees
    # A checkout path of the outer repository passes too: the tools remap
    # it into the worktree the work dir is in.
    assert sub._guard_worktree(
        "Write", {"file_path": str(worktree.parent.parent / "seed.txt")}
    ) is None
