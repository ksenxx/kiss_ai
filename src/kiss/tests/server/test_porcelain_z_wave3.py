# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Both porcelain readers take ``git status --porcelain -z`` names verbatim.

``GitWorktreeOps.copy_dirty_state`` and the changed-files fallback of the
merge flow (``_changed_files_of`` when the diff against the original
branch fails) used to read the quoted ``git status --porcelain`` and
undo git's C quoting and its `` -> `` rename join by hand.  Both now
read the NUL-separated ``-z`` form through the one parser
``git_worktree._porcelain_entries``.  These tests drive both readers on
real repositories with names git would quote or that hold a `` -> ``,
a newline, a tab, a backslash, a double quote, padding spaces or
non-ASCII characters.  No mocks.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import init_repo, run_git_checked

from ._memory_printer import MemoryPrinter

pytestmark = posix_only("newlines, tabs, backslashes and quotes in file names")

OLD = " old -> name .txt"
NEW = 'new "q" \\ name\n.txt'
PADDED = "  padded\ttab .txt"
UNTRACKED = "caf\u00e9\u2028x.txt"


def _repo_with_tracked_names(path: Path) -> Path:
    """A repository on ``main`` whose last commit holds *OLD* and *PADDED*."""
    init_repo(path)
    (path / OLD).write_text("v1\n")
    (path / PADDED).write_text("v1\n")
    run_git_checked(path, "add", "--", OLD, PADDED)
    run_git_checked(path, "commit", "-q", "-m", "special names")
    return path


def _dirty(tree: Path) -> None:
    """Stage the rename *OLD* -> *NEW*, edit *PADDED*, add *UNTRACKED*."""
    run_git_checked(tree, "mv", "--", OLD, NEW)
    (tree / PADDED).write_text("edited\n")
    (tree / UNTRACKED).write_text("new\n")


def test_copy_dirty_state_mirrors_names_git_would_quote(tmp_path: Path) -> None:
    repo = _repo_with_tracked_names(tmp_path / "repo")
    wt_dir = tmp_path / "wt"
    run_git_checked(repo, "worktree", "add", "-q", "--detach", str(wt_dir))
    _dirty(repo)

    assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True

    assert not (wt_dir / OLD).exists()
    assert (wt_dir / NEW).read_text() == "v1\n"
    assert (wt_dir / PADDED).read_text() == "edited\n"
    assert (wt_dir / UNTRACKED).read_text() == "new\n"


def test_changed_files_fallback_lists_exact_names_and_both_rename_sides(
    tmp_path: Path,
) -> None:
    repo = _repo_with_tracked_names(tmp_path / "repo")
    server = AgentServer(printer=MemoryPrinter())
    server.work_dir = str(repo)
    wt_agent = ChatSorcarAgent("wt")
    wt_agent._chat_id = "wt-z"
    wt_work = wt_agent._try_setup_worktree(repo, str(repo))
    assert wt_work is not None
    agent_state.register(
        agent_state.AgentState("wt-z-key", agent=wt_agent, tab_id="wt-z", server_owned=True),
    )
    _dirty(Path(wt_work))
    # The original branch no longer resolves, so the diff against it
    # fails and the listing falls back to ``git status``.
    run_git_checked(repo, "branch", "-m", "main", "renamed")
    try:
        changed = server._get_worktree_changed_files("wt-z")
    finally:
        run_git_checked(repo, "branch", "-m", "renamed", "main")
        wt_agent.discard()
        agent_state.agent_states.clear()

    assert {OLD, NEW, PADDED, UNTRACKED} <= set(changed), changed
    assert not [name for name in changed if " -> " in name and name != OLD], changed
