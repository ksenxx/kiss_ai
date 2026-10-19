# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end checks for the D2 simplifications.

* ``resolve_tool_path`` is the one implementation of the worktree
  remap-then-stale-fallback sequence that ``Read``, ``Write``, ``Edit``
  and ``WebUseTool.screenshot`` used to carry as four copies: every
  branch of it is exercised through the tools, and ``Write`` and
  ``screenshot`` land at the same parent-repo path when the worktree
  they were given has vanished.
* ``_bash_streaming``'s stdout pipe is closed by the reader thread
  alone: after a timed-out ``Bash`` the process's open-fd count returns
  to its pre-call value.

Real files, a real headless Chromium and a real shell; no mocks.
"""

import os
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools, resolve_tool_path
from kiss.agents.sorcar.web_use_tool import WebUseTool


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A parent repo whose worktree has already been torn down."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    return root


def _vanished_wt(repo: Path) -> Path:
    return repo / ".kiss-worktrees" / "kiss_wt-gone"


def test_resolve_tool_path_live_worktree_remaps_parent_repo_path(repo: Path) -> None:
    """A parent-repo path while work_dir is a live worktree lands inside it."""
    wt = repo / ".kiss-worktrees" / "kiss_wt-live"
    wt.mkdir(parents=True)
    got = resolve_tool_path(str(repo / "docs" / "guide.md"), str(wt))
    assert got == wt / "docs" / "guide.md"
    got = resolve_tool_path(str(repo / "docs" / "guide.md"), str(wt), Path.exists)
    assert got == wt / "docs" / "guide.md"


def test_resolve_tool_path_vanished_worktree_falls_back(repo: Path) -> None:
    """A vanished-worktree path falls back to the parent repo, gated by ``exists``."""
    stale = _vanished_wt(repo) / "docs" / "guide.md"
    # Unconditional (Write / screenshot).
    assert resolve_tool_path(str(stale), None) == repo / "docs" / "guide.md"
    # Gated: the fallback exists -> taken.
    assert resolve_tool_path(str(stale), None, Path.is_file) == repo / "docs" / "guide.md"
    # Gated: the fallback does not exist -> the stale path is returned as is.
    missing = _vanished_wt(repo) / "docs" / "nope.md"
    assert resolve_tool_path(str(missing), None, Path.is_file) == missing
    # Gated: the resolved path itself exists -> no fallback is consulted.
    assert resolve_tool_path(str(repo / "docs" / "guide.md"), None, Path.exists) == (
        repo / "docs" / "guide.md"
    )
    # Outside any worktree, nothing changes.
    plain = repo / "docs" / "other.md"
    assert resolve_tool_path(str(plain), None) == plain
    # Relative paths anchor under work_dir first.
    assert resolve_tool_path("docs/guide.md", str(repo), Path.exists) == (
        repo / "docs" / "guide.md"
    )


def test_write_and_screenshot_under_vanished_worktree_land_in_parent_repo(
    repo: Path,
) -> None:
    """Write and screenshot given a vanished-worktree path both end up at the
    same relative path under the parent repo, and the worktree is not
    resurrected."""
    stale_dir = _vanished_wt(repo) / "out"
    tools = UsefulTools(work_dir=str(repo))
    result = tools.Write(str(stale_dir / "note.txt"), "hello")
    assert result.startswith("Successfully wrote"), result
    assert (repo / "out" / "note.txt").read_text(encoding="utf-8") == "hello"

    web = WebUseTool(headless=True, user_data_dir=None, work_dir=str(repo))
    try:
        web.go_to_url("data:text/html,<h1>vanished</h1>")
        shot = web.screenshot(str(stale_dir / "shot.png"))
    finally:
        web.close()
    assert shot == f"Screenshot saved to {repo / 'out' / 'shot.png'}", shot
    assert (repo / "out" / "shot.png").is_file()
    assert not _vanished_wt(repo).exists(), "zombie worktree resurrected"


def test_read_and_edit_under_vanished_worktree_use_parent_repo_copy(repo: Path) -> None:
    """Read then Edit through a vanished-worktree path share one resolved
    path, so the read-before-edit bookkeeping matches."""
    stale = _vanished_wt(repo) / "docs" / "guide.md"
    tools = UsefulTools(work_dir=str(repo))
    assert "guide" in tools.Read(str(stale))
    result = tools.Edit(str(stale), "guide", "manual")
    assert result.startswith("Successfully"), result
    assert (repo / "docs" / "guide.md").read_text(encoding="utf-8") == "manual\n"
    assert "File not found" in tools.Edit(str(stale.with_name("nope.md")), "a", "b")


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_timed_out_bash_releases_its_stdout_pipe() -> None:
    """After a timed-out Bash the stdout pipe fd is closed by the reader thread."""
    tools = UsefulTools(work_dir=None)
    before = _open_fds()
    result = tools.Bash("sleep 30", "timeout", timeout_seconds=1)
    assert "timeout" in result, result
    deadline = time.monotonic() + 2
    while _open_fds() > before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _open_fds() <= before, f"fd leak: {before} -> {_open_fds()}"


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_completed_bash_releases_its_stdout_pipe() -> None:
    """The EOF path (shell exits normally) also leaves no pipe fd behind."""
    tools = UsefulTools(work_dir=None)
    before = _open_fds()
    assert "done" in tools.Bash("echo done", "echo", timeout_seconds=10)
    deadline = time.monotonic() + 2
    while _open_fds() > before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _open_fds() <= before, f"fd leak: {before} -> {_open_fds()}"
