# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 2, Q2 finding 4: NUL-separated git output keeps a carriage return in a path.

``copy_dirty_state``, the merge flow's changed-files fallback and the
other ``-z`` readers (``diff --name-only``, ``ls-files``, ``grep -l``)
took git's output through ``git_worktree._git``, which ran git in text
mode: universal newlines turned a ``\\r`` or ``\\r\\n`` inside a name into
``\\n``.  A dirty file named ``tracked\\rcr.txt`` was looked up as
``tracked\\ncr.txt``: its edit never reached the new worktree, and the
unchanged LF-named neighbour was copied instead.  Real repositories and
git; nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git, _porcelain_entries
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.conftest import posix_only
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.conftest import init_repo, run_git, run_git_checked

pytestmark = posix_only("carriage returns and newlines in file names")

CR = "tracked\rcr.txt"
LF = "tracked\ncr.txt"  # the name text mode read CR as
OLD = "old\r\nname.txt"
NEW = "new\rname\r\n.txt"
UNTRACKED_CR = "a\rb.txt"
UNTRACKED_CRLF = "c\r\nd.txt"


def _repo(path: Path) -> Path:
    """A repository on ``main`` whose last commit holds *CR*, *LF* and *OLD*."""
    init_repo(path)
    for name in (CR, LF, OLD):
        (path / name).write_text(f"committed {name!r}\n")
    run_git_checked(path, "add", "--", CR, LF, OLD)
    run_git_checked(path, "commit", "-q", "-m", "names with carriage returns")
    return path


def _dirty(tree: Path) -> None:
    """Edit *CR* (not *LF*), stage the rename *OLD* -> *NEW*, add two untracked files."""
    (tree / CR).write_text("edited CR\n")
    run_git_checked(tree, "mv", "--", OLD, NEW)
    (tree / UNTRACKED_CR).write_text("untracked CR\n")
    (tree / UNTRACKED_CRLF).write_text("untracked CRLF\n")


def test_porcelain_z_entries_keep_carriage_returns(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    _dirty(repo)

    status = _git("status", "--porcelain", "-z", "-uall", cwd=repo)

    assert sorted(_porcelain_entries(status.stdout)) == sorted([
        (" M", None, CR), ("R ", OLD, NEW), ("??", None, UNTRACKED_CR),
        ("??", None, UNTRACKED_CRLF),
    ])
    # Text output keeps its line ends as before: one entry per line.
    assert _git("status", "--porcelain", cwd=repo).stdout.count("\n") == 4


def test_copy_dirty_state_copies_the_files_named_with_carriage_returns(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    wt_dir = tmp_path / "wt"
    run_git_checked(repo, "worktree", "add", "-q", "--detach", str(wt_dir))
    _dirty(repo)

    assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True

    assert (wt_dir / CR).read_text() == "edited CR\n"
    assert (wt_dir / LF).read_text() == f"committed {LF!r}\n"
    assert not (wt_dir / OLD).exists()
    assert (wt_dir / NEW).read_text() == f"committed {OLD!r}\n"
    assert (wt_dir / UNTRACKED_CR).read_text() == "untracked CR\n"
    assert (wt_dir / UNTRACKED_CRLF).read_text() == "untracked CRLF\n"
    assert not (wt_dir / "a\nb.txt").exists() and not (wt_dir / "c\nd.txt").exists()


def test_name_lists_keep_carriage_returns(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.log\n")
    (repo / "x\ry.log").write_text("ignored\n")
    assert GitWorktreeOps.list_ignored_files(repo) == ["x\ry.log"]

    run_git_checked(repo, "checkout", "-q", "-b", "other")
    (repo / CR).write_text("other side\n")
    run_git_checked(repo, "commit", "-q", "-am", "other side")
    run_git_checked(repo, "checkout", "-q", "main")
    (repo / CR).write_text("main side\n")
    run_git_checked(repo, "commit", "-q", "-am", "main side")
    assert run_git(repo, "merge", "-q", "other").returncode != 0
    assert GitWorktreeOps.unresolved_paths(repo) == [CR]

    run_git_checked(repo, "add", "--", CR)
    assert GitWorktreeOps.paths_with_conflict_markers(repo, [CR, LF]) == [CR]


def test_changed_files_fallback_lists_names_with_carriage_returns(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    server = AgentServer(printer=MemoryPrinter())
    server.work_dir = str(repo)
    wt_agent = ChatSorcarAgent("wt")
    wt_agent._chat_id = "wt-cr"
    wt_work = wt_agent._try_setup_worktree(repo, str(repo))
    assert wt_work is not None
    agent_state.register(
        agent_state.AgentState("wt-cr-key", agent=wt_agent, tab_id="wt-cr", server_owned=True),
    )
    _dirty(Path(wt_work))
    # The original branch no longer resolves, so the diff against it fails
    # and the listing falls back to ``git status --porcelain -z``.
    run_git_checked(repo, "branch", "-m", "main", "renamed")
    try:
        changed = server._get_worktree_changed_files("wt-cr")
    finally:
        run_git_checked(repo, "branch", "-m", "renamed", "main")
        wt_agent.discard()
        agent_state.agent_states.clear()

    assert {CR, OLD, NEW, UNTRACKED_CR, UNTRACKED_CRLF} <= set(changed), changed
    assert not {LF, "a\nb.txt", "c\nd.txt", "new\nname\n.txt"} & set(changed), changed
