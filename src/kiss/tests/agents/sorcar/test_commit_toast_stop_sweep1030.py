# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop during the git commit still closes the sticky commit toast.

``auto_commit_changes`` shows a sticky "Generating commit message" toast
(``notify_fn("generating", "")``) and must always follow it with a
terminal ``committed`` or ``failed`` update.  The runner's Stop is a
``KeyboardInterrupt`` raised asynchronously in the task thread
(``PyThreadState_SetAsyncExc``); here it is delivered while the
repository's pre-commit hook runs, so it surfaces from ``git commit``
after the commit message was generated.
"""

from __future__ import annotations

import functools
import subprocess
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.sorcar_agent import auto_commit_changes
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import posix_only

_HOOK = """#!/bin/sh
touch '{marker}'
sleep 2
"""


def _git(repo: Path, *args: str) -> None:
    """Run ``git -C repo args``; fail the test when it fails."""
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo_with_slow_hook(tmp_path: Path, marker: Path) -> Path:
    """A repo with one commit, an uncommitted file and a pre-commit hook that sleeps."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "T")
    (repo / "README.md").write_text("# test\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(_HOOK.format(marker=marker))
    hook.chmod(0o755)
    (repo / "work.txt").write_text("agent work\n")
    return repo


def _fixed_message(commit_dir: Path, user_prompt: str | None, task_result: str | None) -> str:
    """The commit message (the model call has already returned)."""
    return "test: agent work"


def _record(stages: list[str], stage: str, subject: str) -> None:
    """``notify_fn`` that records the toast stages."""
    stages.append(stage)


def _commit(repo: Path, stages: list[str], raised: list[BaseException]) -> None:
    """Thread body: run the auto-commit, keeping the interrupt it raises."""
    try:
        auto_commit_changes(
            repo, None, _fixed_message, notify_fn=functools.partial(_record, stages),
        )
    except BaseException as exc:  # the injected Stop
        raised.append(exc)


@posix_only("the pre-commit hook is a POSIX shell script")
def test_stop_during_commit_hook_closes_the_generating_toast(tmp_path: Path) -> None:
    marker = tmp_path / "hook-started"
    repo = _repo_with_slow_hook(tmp_path, marker)
    stages: list[str] = []
    raised: list[BaseException] = []
    worker = threading.Thread(target=_commit, args=(repo, stages, raised), daemon=True)
    worker.start()
    deadline = time.monotonic() + 30
    while not marker.exists():
        assert time.monotonic() < deadline, "the pre-commit hook never ran"
        time.sleep(0.01)
    assert worker.ident is not None
    assert inject_async_exception(worker.ident, KeyboardInterrupt) == 1
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert [type(exc) for exc in raised] == [KeyboardInterrupt]
    assert stages == ["generating", "failed"]
