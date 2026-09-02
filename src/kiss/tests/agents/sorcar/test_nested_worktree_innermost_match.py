# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Nested kiss worktrees: every path guard must anchor on the *innermost* one.

A repo can itself live inside another repo's ``.kiss-worktrees/kiss_wt-*``
directory (a project checked out under a kiss worktree, or the test suite
running from inside one).  ``_worktree_index`` used to return the
*outermost* marker, so with a layout like::

    outer/.kiss-worktrees/kiss_wt-outer/proj/.kiss-worktrees/kiss_wt-inner/

an agent working in ``kiss_wt-inner`` had ``outer/`` treated as its parent
repo.  Paths under ``proj/`` then looked like they were already "inside a
worktree" and the parent-repo -> worktree remap, the stale-worktree
fallback and the Bash parent-repo guard were all silently skipped -- the
agent mutated ``proj/`` directly.

These tests build the nested layout explicitly (so they exercise the bug
no matter where pytest itself is launched from) and drive the public
``UsefulTools`` / ``WebUseTool`` surface end to end.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools


@pytest.fixture()
def nested(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(proj, inner_wt)`` where ``proj`` lives inside an outer live worktree."""
    outer_wt = Path(os.path.realpath(tmp_path)) / "outer" / ".kiss-worktrees" / "kiss_wt-outer"
    proj = outer_wt / "proj"
    inner_wt = proj / ".kiss-worktrees" / "kiss_wt-inner"
    (proj / "src").mkdir(parents=True)
    (inner_wt / "src").mkdir(parents=True)
    (proj / "src" / "app.py").write_text("main\n")
    (inner_wt / "src" / "app.py").write_text("worktree\n")
    return proj, inner_wt


def test_read_remaps_into_innermost_worktree(nested: tuple[Path, Path]) -> None:
    proj, inner_wt = nested
    tools = UsefulTools(work_dir=str(inner_wt))
    assert tools.Read(str(proj / "src" / "app.py")) == "worktree\n"


def test_write_and_edit_land_in_innermost_worktree(nested: tuple[Path, Path]) -> None:
    proj, inner_wt = nested
    tools = UsefulTools(work_dir=str(inner_wt))
    assert "Successfully wrote" in tools.Write(str(proj / "notes.md"), "n\n")
    assert (inner_wt / "notes.md").read_text() == "n\n"
    assert not (proj / "notes.md").exists()
    assert "Successfully replaced" in tools.Edit(
        str(proj / "src" / "app.py"), "worktree", "patched"
    )
    assert (inner_wt / "src" / "app.py").read_text() == "patched\n"
    assert (proj / "src" / "app.py").read_text() == "main\n"


def test_bash_guard_refuses_innermost_parent_repo_path(nested: tuple[Path, Path]) -> None:
    proj, inner_wt = nested
    tools = UsefulTools(work_dir=str(inner_wt))
    out = tools.Bash(f"echo PWNED > {proj}/src/app.py", "evil")
    assert "parent-repo path" in out, out
    assert (proj / "src" / "app.py").read_text() == "main\n"
    # Paths inside the inner worktree itself are still allowed.
    out = tools.Bash(f"cat {inner_wt}/src/app.py", "read")
    assert out.strip() == "worktree", out


def test_stale_innermost_worktree_falls_back_to_its_own_parent(
    nested: tuple[Path, Path],
) -> None:
    proj, inner_wt = nested
    tools = UsefulTools(work_dir=str(inner_wt))
    shutil.rmtree(inner_wt)
    # Read/Edit/Write on the dangling inner-worktree path must fall back to
    # ``proj`` (the inner worktree's parent), not dead-end because the
    # *outer* worktree is still live.
    stale = inner_wt / "src" / "app.py"
    assert tools.Read(str(stale)) == "main\n"
    assert "Successfully replaced" in tools.Edit(str(stale), "main", "edited")
    assert (proj / "src" / "app.py").read_text() == "edited\n"
    assert "Successfully wrote" in tools.Write(str(inner_wt / "new.txt"), "x\n")
    assert (proj / "new.txt").read_text() == "x\n"
    assert not inner_wt.exists()
    # Bash falls back to running from ``proj``.
    out = tools.Bash("pwd", "probe")
    assert Path(out.strip()).resolve() == proj.resolve(), out


def test_screenshot_remaps_into_innermost_worktree(nested: tuple[Path, Path]) -> None:
    from kiss.agents.sorcar.web_use_tool import WebUseTool

    proj, inner_wt = nested
    tool = WebUseTool(headless=True, user_data_dir=None, work_dir=str(inner_wt))
    try:
        tool.go_to_url("data:text/html,<p>nested</p>")
        result = tool.screenshot(str(proj / "shots" / "page.png"))
    finally:
        tool.close()
    assert result.startswith("Screenshot saved to "), result
    assert (inner_wt / "shots" / "page.png").is_file()
    assert not (proj / "shots" / "page.png").exists()
