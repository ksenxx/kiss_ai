# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A per-tool Stop of ``commit_and_merge`` never strands the task's worktree.

``commit_and_merge`` removes the worktree directory before it squash-merges
the branch into the user's checkout.  A merge that fails after that point
checks the branch out again so the task can fix the problem and retry.  The
chat panel's per-tool Stop (``interrupt_tool_call``, which injects
``ToolCallInterrupted`` into the task thread) can land in the merge itself,
here while the main checkout's pre-commit hook runs; the task must then
also get its working copy back instead of keeping a pending worktree with
no directory.

The test drives a real ``SorcarAgent`` run against the scripted local model
server and stops the tool call from another thread, as the server does.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE, interrupt_tool_call
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.conftest import posix_only

# Sleeps (then rejects the commit) only in the main checkout: the squash
# merge commit, not a commit inside the task's linked worktree.
_HOOK = """#!/bin/sh
case "$(git rev-parse --git-dir)" in
  *worktrees*) exit 0 ;;
esac
touch '{marker}'
sleep 3
exit 1
"""


def _git(repo: Path, *args: str) -> str:
    """Run ``git -C repo args`` and return its stdout; fail the test on error."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout


def _repo_with_hook(path: Path, marker: Path) -> Path:
    """A repository with one commit on ``main`` and the slow pre-commit hook."""
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# test\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    hook = path / ".git" / "hooks" / "pre-commit"
    hook.write_text(_HOOK.format(marker=marker))
    hook.chmod(0o755)
    return path


def _call(tool: str, **arguments: Any) -> bytes:
    """One scripted model turn calling *tool*."""
    return tool_call_body(tool, arguments, prompt_tokens=500)


def _run(repo: Path, script: list[bytes], url: str) -> None:
    """Thread body: one top-level agent run in *repo*."""
    SorcarAgent("merge-stop").run(
        prompt_template="Make the change, then merge it.",
        model_name=MODEL,
        max_steps=len(script) + 2,
        max_budget=1.0,
        model_config={"base_url": url, "api_key": "local"},
        work_dir=str(repo),
        use_memory=False,
        verbose=False,
        tool_profile="shell+edit",
    )


@posix_only("the pre-commit hook is a POSIX shell script")
def test_stopped_merge_gives_the_task_its_worktree_back(tmp_path: Path) -> None:
    marker = tmp_path / "merge-commit-started"
    repo = _repo_with_hook(tmp_path / "repo", marker)
    # The task commits its own change, so commit_and_merge makes no
    # model call for a commit message.
    script = [
        _call("create_worktree"),
        _call(
            "Bash",
            command="echo task > task.txt && git add -A && git commit -qm 'task work' --no-verify",
            description="commit the task's change",
        ),
        _call("commit_and_merge"),
        finish_body("<p>done</p>"),
        finish_body("<p>done</p>"),
    ]
    with serve(script) as (url, requests):
        runner = threading.Thread(target=_run, args=(repo, script, url), daemon=True)
        runner.start()
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert time.monotonic() < deadline, "the merge commit never started"
            time.sleep(0.01)
        assert runner.ident is not None
        assert interrupt_tool_call(runner.ident, "commit_and_merge")
        runner.join(timeout=60)
        assert not runner.is_alive()
    tool_results = [
        str(m.get("content")) for m in requests[-1]["messages"] if m.get("role") == "tool"
    ]
    assert USER_INTERRUPTED_MESSAGE in tool_results[2], tool_results[2]
    worktrees = list((repo / ".kiss-worktrees").glob("*/task.txt"))
    assert len(worktrees) == 1, "the pending worktree lost its directory"
    assert worktrees[0].parent.resolve().as_posix() in _git(repo, "worktree", "list")
    assert _git(repo, "log", "--format=%s", "main").split() == ["initial"]
