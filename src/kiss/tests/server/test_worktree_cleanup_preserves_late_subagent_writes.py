# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sub-agent that finishes during the cleanup wait keeps its work.

``_commit_and_clean_worktree`` waits several seconds for a sub-agent
thread the parent abandoned before it deletes the worktree.  A child
that produces its last file *inside* that wait must not have it
deleted: the wait exists precisely because the child is still writing.

Everything here is real: a real git repository and a real ``git
worktree``, a real ``ThreadPoolExecutor`` fan-out, a real child agent
running against a real (local, free) OpenAI-compatible endpoint, real
threads, and the real cleanup path.  Nothing is mocked or patched.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.worktree_sorcar_agent import (
    _ABANDONED_SUBAGENT_WAIT_SECONDS,
    WorktreeSorcarAgent,
    _WorktreeCleanupOutcome,
)
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
    request_text,
)

# How long the abandoned child keeps working after the cleanup starts.
# It has to be comfortably past the parent's staging/commit pass (a
# handful of git invocations, tens of milliseconds) and comfortably
# inside the parent's abandoned-child wait, so the child's last write
# lands in exactly the window this test is about.
_LATE_WRITE_DELAY_S = 0.5

_LATE_FILE = "late-subagent-output.txt"
_LATE_CONTENT = "written by the sub-agent during the cleanup wait\n"


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo.

    The auto-commit message generator is kept offline for the whole
    test so the cleanup path costs nothing and its timing is that of
    plain git rather than of a network round trip.
    """
    isolated = IsolatedKissHome("kiss-late-subagent-")
    try:
        with OfflineFastModel():
            yield isolated
    finally:
        isolated.cleanup()


class _LateWritingChildModel:
    """Stand-in model that parks the child, then makes it write late.

    The child's model call blocks until the test releases it, which is
    what makes ``_await_subagents`` abandon the child.  Once released
    the child keeps working for :data:`_LATE_WRITE_DELAY_S` and writes
    one last file into the shared worktree before finishing — the
    late-arriving output the cleanup wait is supposed to protect.
    """

    def __init__(self, wt_dir: Path) -> None:
        """Record where the child writes and create its events.

        Args:
            wt_dir: The worktree directory shared with the parent.
        """
        self.wt_dir = wt_dir
        self.wedged = threading.Event()
        self.release = threading.Event()
        self.wrote = threading.Event()

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Park the child's request; answer the parent's normally."""
        if "LATE-WRITER" not in request_text(request):
            return finish_response("parent done")
        self.wedged.set()
        self.release.wait(120)
        time.sleep(_LATE_WRITE_DELAY_S)
        (self.wt_dir / _LATE_FILE).write_text(_LATE_CONTENT, encoding="utf-8")
        self.wrote.set()
        return finish_response("late child done")


def _interrupt_thread(thread: threading.Thread) -> None:
    """Raise ``KeyboardInterrupt`` inside *thread*, as the server does."""
    import ctypes

    assert thread.ident is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident),
        ctypes.py_object(KeyboardInterrupt),
    )


def _branch_file(repo: Path, branch: str, name: str) -> str:
    """Return *name*'s content on *branch*, or ``""`` when absent."""
    done = subprocess.run(
        ["git", "show", f"{branch}:{name}"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout if done.returncode == 0 else ""


def test_child_finishing_during_the_cleanup_wait_keeps_its_output(
    env: IsolatedKissHome,
) -> None:
    """The last file an abandoned child writes survives cleanup."""
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("late-writer-parent")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL

    wt_work_dir = parent._try_setup_worktree(env.repo, str(env.repo))
    assert wt_work_dir is not None
    wt = parent._wt
    assert wt is not None
    parent.work_dir = str(wt_work_dir)

    model = _LateWritingChildModel(wt_work_dir)
    server = StandInModelServer(model)
    parent.model_config = server.model_config

    fanout: dict[str, Any] = {}

    def run_fanout() -> None:
        printer._thread_local.task_id = "latewritertask"
        try:
            fanout["results"] = parent._run_tasks_parallel(
                ["LATE-WRITER child task"], max_workers=1,
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            fanout["error"] = exc

    thread = threading.Thread(target=run_fanout, daemon=True)
    thread.start()
    try:
        assert model.wedged.wait(60), "the sub-agent never reached the model"

        # Interrupt the parent the way the server does.  The child's
        # own stop event is never set, so it keeps running: abandoned,
        # not stopped.
        _interrupt_thread(thread)
        thread.join(timeout=60)
        assert not thread.is_alive(), "the parent thread never unwound"
        assert isinstance(fanout.get("error"), KeyboardInterrupt), fanout
        assert parent._abandoned_subagents, "no child was abandoned"

        # Work the parent itself produced, so the cleanup has a real
        # commit to make before it waits for the child.
        (wt_work_dir / "parent-output.txt").write_text(
            "parent work\n", encoding="utf-8",
        )

        # From here the child runs for _LATE_WRITE_DELAY_S and then
        # writes its last file — inside the parent's abandoned-child
        # wait, and after the parent's own staging pass.
        model.release.set()
        started = time.monotonic()
        outcome, leftover = parent._commit_and_clean_worktree(wt)
        elapsed = time.monotonic() - started
    finally:
        model.release.set()
        parent.reclaim_abandoned_subagents(timeout=120)
        thread.join(timeout=30)
        server.stop()

    assert model.wrote.is_set(), "the child never produced its late file"
    assert elapsed < _ABANDONED_SUBAGENT_WAIT_SECONDS + 30, (
        f"cleanup took {elapsed:.1f}s; it must not wait indefinitely"
    )

    on_disk = (wt_work_dir / _LATE_FILE).exists()
    committed = _branch_file(env.repo, wt.branch, _LATE_FILE)
    assert on_disk or committed == _LATE_CONTENT, (
        "the file the sub-agent wrote while the cleanup was waiting for "
        f"it was deleted with the worktree (outcome={outcome}, "
        f"leftover={leftover!r})"
    )
    # The parent's own work must still be committed on the branch.
    assert _branch_file(env.repo, wt.branch, "parent-output.txt")


def test_cleanup_without_abandoned_children_removes_the_worktree(
    env: IsolatedKissHome,
) -> None:
    """The ordinary path still commits and removes, with no wait."""
    server = StandInModelServer(lambda request: finish_response("quick"))
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("clean-cleanup")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config

    wt_work_dir = parent._try_setup_worktree(env.repo, str(env.repo))
    assert wt_work_dir is not None
    wt = parent._wt
    assert wt is not None
    parent.work_dir = str(wt_work_dir)
    (wt_work_dir / "only-output.txt").write_text("work\n", encoding="utf-8")

    try:
        started = time.monotonic()
        outcome, leftover = parent._commit_and_clean_worktree(wt)
        elapsed = time.monotonic() - started
    finally:
        server.stop()

    assert outcome is _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, leftover
    assert not wt.wt_dir.exists()
    assert elapsed < _ABANDONED_SUBAGENT_WAIT_SECONDS, (
        "a cleanup with nothing to reclaim must not pay the wait"
    )
    assert _branch_file(env.repo, wt.branch, "only-output.txt") == "work\n"
