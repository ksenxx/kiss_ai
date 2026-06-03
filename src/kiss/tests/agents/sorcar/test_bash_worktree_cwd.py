# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: Bash tool runs subprocesses in the agent's work_dir.

When the agent runs inside a git worktree, ``self.work_dir`` is the
worktree path. The Bash tool must launch child processes with that
directory as ``cwd`` and set ``KISS_WORKDIR`` accordingly, otherwise
project scripts (e.g. ``src/kiss/scripts/update_models.py``) that derive
a "project root" from the inherited environment write to the original
checkout and the worktree silently captures no changes — leading to the
worktree being discarded as "empty" while edits leak onto the main
branch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools, _clean_env


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t"], check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "t"], check=True
    )
    (path / "seed.txt").write_text("seed")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "seed"], check=True
    )


def test_clean_env_overrides_kiss_workdir(monkeypatch):
    """``_clean_env(work_dir)`` must replace the inherited KISS_WORKDIR."""
    monkeypatch.setenv("KISS_WORKDIR", "/inherited/main/repo")
    monkeypatch.setenv("VIRTUAL_ENV", "/inherited/venv")
    env = _clean_env("/some/worktree/dir")
    assert env["KISS_WORKDIR"] == "/some/worktree/dir"
    assert "VIRTUAL_ENV" not in env


def test_clean_env_without_work_dir_leaves_kiss_workdir(monkeypatch):
    monkeypatch.setenv("KISS_WORKDIR", "/inherited/main/repo")
    env = _clean_env(None)
    assert env["KISS_WORKDIR"] == "/inherited/main/repo"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix shell semantics")
def test_bash_runs_in_work_dir_sync(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    tools = UsefulTools(work_dir=str(work))
    output = tools.Bash("pwd && echo KISS_WORKDIR=$KISS_WORKDIR", "probe")
    # On macOS /private/tmp -> /tmp symlink; resolve both for comparison.
    pwd_line, env_line = output.strip().splitlines()
    assert Path(pwd_line).resolve() == work.resolve()
    assert env_line == f"KISS_WORKDIR={work}"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix shell semantics")
def test_bash_runs_in_work_dir_streaming(tmp_path: Path):
    work = tmp_path / "work_streaming"
    work.mkdir()
    streamed: list[str] = []
    tools = UsefulTools(stream_callback=streamed.append, work_dir=str(work))
    output = tools.Bash("pwd && echo KISS_WORKDIR=$KISS_WORKDIR", "probe")
    pwd_line, env_line = output.strip().splitlines()
    assert Path(pwd_line).resolve() == work.resolve()
    assert env_line == f"KISS_WORKDIR={work}"
    # streaming sink must have received the same content
    streamed_joined = "".join(streamed)
    assert str(work) in streamed_joined or str(work.resolve()) in streamed_joined


@pytest.mark.skipif(sys.platform == "win32", reason="Unix shell + git workflow")
def test_update_models_style_script_writes_into_worktree(tmp_path: Path):
    """Reproduces the original bug.

    A project script that uses ``KISS_WORKDIR`` to find its "project
    root" and then writes to ``<root>/src/kiss/core/models/model_info.py``
    must land its write inside the worktree directory, not the main
    checkout, when the Bash tool is configured with the worktree as
    its ``work_dir``.
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target_rel = Path("src/kiss/core/models/model_info.py")
    main_target = repo / target_rel
    main_target.parent.mkdir(parents=True, exist_ok=True)
    main_target.write_text("# original main contents\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "add target"], check=True
    )

    # Create a worktree at a sibling path.
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", str(wt), "-b", "feature"],
        check=True,
    )
    wt_target = wt / target_rel

    # Simulate the agent process: it was launched with KISS_WORKDIR
    # pointing at the *main* repo (this is what the VS Code extension
    # injects). The worktree agent sets ``self.work_dir`` to the
    # worktree, and UsefulTools must propagate that to subprocesses.
    old_kiss_workdir = os.environ.get("KISS_WORKDIR")
    os.environ["KISS_WORKDIR"] = str(repo)
    try:
        tools = UsefulTools(work_dir=str(wt))
        # Mini script that mirrors update_models.py's _find_project_root
        # (KISS_WORKDIR first, then cwd/.git, then __file__ fallback).
        script = (
            "python3 -c \""
            "import os, pathlib;"
            "root = pathlib.Path(os.environ.get('KISS_WORKDIR') or '.');"
            "p = root / 'src/kiss/core/models/model_info.py';"
            "p.write_text('# UPDATED by script\\n');"
            "print('wrote', p)\""
        )
        result = tools.Bash(script, "run update")
        assert "wrote" in result, result
    finally:
        if old_kiss_workdir is None:
            os.environ.pop("KISS_WORKDIR", None)
        else:
            os.environ["KISS_WORKDIR"] = old_kiss_workdir

    # The bug: write landed in main repo. The fix: write lands in worktree.
    assert wt_target.read_text() == "# UPDATED by script\n", (
        "worktree file was NOT updated — write leaked outside the worktree"
    )
    assert main_target.read_text() == "# original main contents\n", (
        "main repo file was modified — write should have stayed in the worktree"
    )
