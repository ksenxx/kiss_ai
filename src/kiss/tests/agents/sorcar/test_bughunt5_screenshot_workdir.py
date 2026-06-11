# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 5: screenshot() must anchor paths to the agent work_dir.

Every file tool (Read/Write/Edit via ``_expand_pwd_prefix``, Bash via
``cwd=work_dir``) anchors relative paths and the literal ``PWD/`` prefix
(which the system prompt mandates for artifacts) to the agent's working
directory.  Pre-fix, ``WebUseTool.screenshot`` resolved paths against the
DAEMON PROCESS's cwd instead:

- in worktree mode, ``screenshot("shot.png")`` escaped the worktree (the
  file was never merged back / visible to the task);
- ``screenshot("PWD/tmp/shot.png")`` created a junk literal ``PWD/``
  directory tree in the process cwd.

Uses a real headless Chromium (same pattern as test_web_use_tool.py).
"""

import os
from pathlib import Path

from kiss.agents.sorcar.web_use_tool import WebUseTool


def test_screenshot_relative_path_lands_in_work_dir(tmp_path: Path) -> None:
    """A relative screenshot path must resolve under work_dir, not cwd."""
    cwd = tmp_path / "process_cwd"
    work = tmp_path / "agent_workdir"
    cwd.mkdir()
    work.mkdir()
    old_cwd = os.getcwd()
    tool = WebUseTool(user_data_dir=None, headless=True, work_dir=str(work))
    os.chdir(cwd)
    try:
        nav = tool.go_to_url("data:text/html,<h1>hello</h1>")
        assert not nav.startswith("Error"), nav
        msg = tool.screenshot("shots/out.png")
        assert msg.startswith("Screenshot saved to"), msg
        assert (work / "shots" / "out.png").is_file(), (
            f"screenshot escaped work_dir: {msg!r}; "
            f"cwd contents: {list(cwd.rglob('*'))}"
        )
        assert not (cwd / "shots").exists()
    finally:
        os.chdir(old_cwd)
        tool.close()


def test_screenshot_pwd_prefix_expands_to_work_dir(tmp_path: Path) -> None:
    """The literal PWD/ prefix must expand to work_dir (as in Read/Write)."""
    cwd = tmp_path / "process_cwd"
    work = tmp_path / "agent_workdir"
    cwd.mkdir()
    work.mkdir()
    old_cwd = os.getcwd()
    tool = WebUseTool(user_data_dir=None, headless=True, work_dir=str(work))
    os.chdir(cwd)
    try:
        nav = tool.go_to_url("data:text/html,<h1>hello</h1>")
        assert not nav.startswith("Error"), nav
        msg = tool.screenshot("PWD/tmp/shot.png")
        assert msg.startswith("Screenshot saved to"), msg
        assert (work / "tmp" / "shot.png").is_file(), msg
        assert not (cwd / "PWD").exists(), (
            "literal PWD/ directory created in process cwd"
        )
    finally:
        os.chdir(old_cwd)
        tool.close()


def test_screenshot_absolute_path_unchanged(tmp_path: Path) -> None:
    """Regression guard: absolute paths keep working as before."""
    work = tmp_path / "agent_workdir"
    work.mkdir()
    target = tmp_path / "abs" / "shot.png"
    tool = WebUseTool(user_data_dir=None, headless=True, work_dir=str(work))
    try:
        nav = tool.go_to_url("data:text/html,<h1>hello</h1>")
        assert not nav.startswith("Error"), nav
        msg = tool.screenshot(str(target))
        assert msg.startswith("Screenshot saved to"), msg
        assert target.is_file()
    finally:
        tool.close()
