# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: after an agent task finishes, the ``@``-mention
file index must reflect files the agent created or deleted on disk
inside ``work_dir``.

The index is built lazily on the first ``@``-mention for a given
root and used to be refreshed only on cold start, daemon
``setWorkDir``, or an explicit refresh request — so any files the
agent created or deleted during its turn never reached it.  The next
``@``-mention served stale suggestions: brand-new files (e.g. the
test file the agent just authored) were invisible and deleted files
lingered.

This test reproduces the bug by:

1. warming the index with a ``getFiles`` command,
2. simulating an agent run that creates ``new_file.py`` and deletes
   the previously-existing ``old_file.py`` inside the work_dir,
3. invoking the task-completion hook the production task runner
   fires at the end of every ``_run_task_inner`` cleanup,
4. asserting the next ``getFiles`` returns the post-task file set
   (``new_file.py`` present, ``old_file.py`` absent).

The hook must NOT broadcast an unsolicited ``files`` event: a reply
stamped ``conn_id="", prefix=""`` would be accepted by every client
whose picker shows a bare ``@`` — including windows rooted at a
DIFFERENT work_dir — overwriting their picker with files from this
task's workspace (fixer-5 F5-03/R5-03).  The refreshed index is
served by the next ``getFiles`` instead.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.file_index import FileIndexRegistry, FileView
from kiss.server.server import VSCodeServer

_Workspace = tuple[str, VSCodeServer, list[dict[str, Any]]]


def _wait_for_files_event(
    events: list[dict[str, Any]],
    timeout: float = 5.0,
    *,
    must_contain: str | None = None,
    must_not_contain: str | None = None,
) -> dict[str, Any]:
    """Return the first non-loading ``files`` event matching constraints.

    Polls until either such an event appears or *timeout* elapses.
    The handler emits a ``loading=True`` placeholder when the root is
    not indexed yet; only populated events are considered.  When
    *must_contain* / *must_not_contain* are set, the event must
    also include / exclude that filename.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for e in list(events):
            if e.get("type") != "files" or e.get("loading"):
                continue
            names = _names(e["files"])
            if must_contain is not None and must_contain not in names:
                continue
            if must_not_contain is not None and must_not_contain in names:
                continue
            return e
        time.sleep(0.01)
    raise AssertionError(
        f"no matching files event arrived (must_contain={must_contain!r}, "
        f"must_not_contain={must_not_contain!r}); got {events}"
    )


def _wait_for_view(
    registry: FileIndexRegistry, work_dir: str, containing: str, timeout: float = 5.0,
) -> FileView:
    """Poll until the view of *work_dir* lists *containing* and return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = registry.view_for(work_dir)
        if view is not None and containing in view:
            return view
        time.sleep(0.01)
    view = registry.view_for(work_dir)
    raise AssertionError(
        f"{containing!r} never appeared in the index of {work_dir}; "
        f"got {None if view is None else view.paths}"
    )


def _wait_for_worker_idle(registry: FileIndexRegistry, work_dir: str) -> None:
    """Block until every build queued so far on *registry* has run.

    The registry serves jobs in FIFO order on one worker thread, so a
    callback queued now fires only after the jobs queued before it.
    """
    done = threading.Event()
    registry.ensure(work_dir, done.set)
    assert done.wait(10.0), "registry worker did not drain its queue"


def _names(entries: list[Any]) -> list[str]:
    """Project the ``text`` field out of ranked file suggestions."""
    return [e["text"] if isinstance(e, dict) else str(e) for e in entries]


@pytest.fixture()
def workspace(tmp_path: Path) -> Iterator[_Workspace]:
    """Yield ``(work_dir, server, events)`` for a daemon pinned to a temp dir.

    The work_dir holds a single ``old_file.py``.  The agent simulation
    creates ``new_file.py`` and deletes ``old_file.py``, so both names
    uniquely identify which side of the refresh a file came from.  The
    server's file-index registry is a private one whose home is an
    empty directory, so the work_dir is a root of its own and nothing
    outside ``tmp_path`` is scanned.  ``events`` collects every
    ``printer.broadcast`` call (appended under a lock: replies may
    arrive from the registry's worker thread).
    """
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "old_file.py").write_text("# old file content\n")
    home = tmp_path / "home"
    home.mkdir()

    server = VSCodeServer()
    server.work_dir = str(wd)
    server._file_index.stop()
    server._file_index = FileIndexRegistry(home=str(home), cache_dir=tmp_path / "cache")
    captured: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            captured.append(dict(event))

    server.printer.broadcast = capture  # type: ignore[method-assign]
    try:
        yield str(wd), server, captured
    finally:
        server._file_index.stop()


def test_file_index_refreshes_after_task_completion(
    workspace: _Workspace,
) -> None:
    """End-to-end: the ``@``-mention index must update after an agent
    task that created and deleted files.

    1. Warm the index via ``getFiles`` — asserts the pre-task baseline.
    2. Simulate the agent: create ``new_file.py``, delete ``old_file.py``.
    3. Fire the task-completion hook the production task runner calls
       at the tail of ``_run_task_inner``.
    4. Wait for the index to refresh — WITHOUT any unsolicited
       ``files`` broadcast (which other-workspace pickers would
       wrongly accept).
    5. Issue another ``getFiles`` — it must serve the new list.
    """
    wd, server, events = workspace

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": wd},
    )
    warm = _wait_for_files_event(events, must_contain="./old_file.py")
    warm_names = _names(warm["files"])
    assert "./old_file.py" in warm_names
    assert "./new_file.py" not in warm_names

    # Coarse filesystem mtime clocks: make sure the directory's mtime
    # after the edit differs from the one the warm scan recorded.
    time.sleep(0.02)
    (Path(wd) / "./new_file.py").write_text("# new\n")
    (Path(wd) / "./old_file.py").unlink()

    events.clear()
    server._refresh_files_after_task(wd)

    view = _wait_for_view(server._file_index, wd, "new_file.py")
    assert "old_file.py" not in view, view.paths
    assert [e for e in events if e.get("type") == "files"] == [], (
        "the post-task refresh must not broadcast an unsolicited "
        f"files event (cross-workspace stale picker); got {events}"
    )

    events.clear()
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": wd},
    )
    second = _wait_for_files_event(events, must_contain="./new_file.py")
    second_names = _names(second["files"])
    assert "./new_file.py" in second_names
    assert "./old_file.py" not in second_names


def test_refresh_is_silent_when_no_files_added_or_removed(
    workspace: _Workspace,
) -> None:
    """When the agent only *modified* existing files (no creation or
    deletion), the hook must not broadcast a ``files`` event and the
    index must keep listing the same entries.

    Modifications never change the picker's list, so a broadcast
    would only push a no-op event to every connected client.
    """
    wd, server, events = workspace

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": wd},
    )
    _wait_for_files_event(events, must_contain="./old_file.py")

    time.sleep(0.02)
    (Path(wd) / "old_file.py").write_text("# modified\n")

    events.clear()
    server._refresh_files_after_task(wd)
    _wait_for_worker_idle(server._file_index, wd)

    view = server._file_index.view_for(wd)
    assert view is not None and view.paths == ["old_file.py"]
    files_events = [e for e in events if e.get("type") == "files"]
    assert files_events == [], (
        "Hook must not broadcast a files event when the file set is "
        f"unchanged; got {files_events}"
    )


def test_refresh_no_op_when_index_never_built(
    workspace: _Workspace,
) -> None:
    """When no ``@``-mention picker has ever opened on *work_dir*,
    the hook must be a no-op: there is nothing to keep fresh, and
    the next ``getFiles`` will scan from scratch anyway.
    """
    wd, server, events = workspace
    registry = server._file_index

    assert registry.view_for(wd) is None

    (Path(wd) / "new_file.py").write_text("# new\n")
    server._refresh_files_after_task(wd)

    # A refresh of an unknown root never queues a build, so there is
    # nothing to wait for: the registry must stay empty.
    time.sleep(0.3)
    assert wd not in registry._indexes
    assert registry.view_for(wd) is None
    assert [e for e in events if e.get("type") == "files"] == []
