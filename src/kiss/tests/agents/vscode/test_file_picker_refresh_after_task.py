# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: after an agent task finishes, the ``@``-mention
file picker cache must reflect files the agent created or deleted on
disk inside ``work_dir``.

The file cache is populated lazily on the first ``@``-mention for a
given ``work_dir`` and was only refreshed on cold start, daemon
``setWorkDir``, or an explicit refresh request — so any files the
agent created or deleted during its turn never reached the cache.
The next ``@``-mention served stale suggestions: brand-new files
(e.g. the test file the agent just authored) were invisible and
deleted files lingered.

This test reproduces the bug by:

1. warming the per-``work_dir`` cache with a ``getFiles`` command,
2. simulating an agent run that creates ``new_file.py`` and deletes
   the previously-existing ``old_file.py`` inside the work_dir,
3. invoking the task-completion hook the production task runner
   fires at the end of every ``_run_task_inner`` cleanup,
4. asserting the next ``getFiles`` returns the post-task file set
   (``new_file.py`` present, ``old_file.py`` absent) and that an
   updated ``files`` event was broadcast so any open picker UI can
   refresh without further user action.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.server import VSCodeServer


def _wait_for_files_event(
    events: list[dict[str, Any]],
    timeout: float = 5.0,
    *,
    must_contain: str | None = None,
    must_not_contain: str | None = None,
) -> dict[str, Any]:
    """Return the first non-loading ``files`` event matching constraints.

    Polls until either such an event appears or *timeout* elapses.
    The handler emits a ``loading=True`` placeholder when the cache
    is empty; only populated events are considered.  When
    *must_contain* / *must_not_contain* are set, the event must
    also include / exclude that filename — used to skip the first
    post-warm event when both events arrive while we poll.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for e in events:
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


def _names(entries: list[Any]) -> list[str]:
    """Project the ``text`` field out of ranked file suggestions."""
    return [e["text"] if isinstance(e, dict) else str(e) for e in entries]


@pytest.fixture()
def workspace() -> Iterator[str]:
    """Create a temp work_dir containing a single ``old_file.py``.

    The agent simulation creates ``new_file.py`` and deletes
    ``old_file.py``, so both names uniquely identify which side of
    the cache refresh a file came from.
    """
    wd = tempfile.mkdtemp(prefix="kiss_refresh_after_task_")
    (Path(wd) / "old_file.py").write_text("# old file content\n")
    try:
        yield wd
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _make_server(work_dir: str) -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Build a ``VSCodeServer`` pinned to *work_dir* with a capturing printer."""
    server = VSCodeServer()
    server.work_dir = work_dir
    captured: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            captured.append(dict(event))

    server.printer.broadcast = capture  # type: ignore[method-assign]
    return server, captured


def test_file_cache_refreshes_after_task_completion(
    workspace: str,
) -> None:
    """End-to-end: the ``@``-mention cache must update after an agent
    task that created and deleted files.

    1. Warm the cache via ``getFiles`` — asserts the pre-task baseline.
    2. Simulate the agent: create ``new_file.py``, delete ``old_file.py``.
    3. Fire the task-completion hook the production task runner calls
       at the tail of ``_run_task_inner``.
    4. Wait for the broadcast ``files`` event and assert the new list.
    5. Issue another ``getFiles`` — the cache itself must now match.
    """
    server, events = _make_server(workspace)

    # 1) Warm cache.
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": workspace},
    )
    warm = _wait_for_files_event(events, must_contain="old_file.py")
    warm_names = _names(warm["files"])
    assert "old_file.py" in warm_names
    assert "new_file.py" not in warm_names

    # 2) Simulate agent: create + delete files in work_dir.
    (Path(workspace) / "new_file.py").write_text("# new\n")
    (Path(workspace) / "old_file.py").unlink()

    # 3) Fire the task-completion hook directly.  In production this
    #    is called from the ``_run_task_inner`` cleanup finally; the
    #    hook is what the bug fix introduces, so a missing or no-op
    #    implementation here is exactly the failure mode under test.
    events.clear()
    server._refresh_files_after_task(workspace)

    # 4) Wait for the post-task broadcast that reflects the new state.
    refreshed = _wait_for_files_event(
        events,
        must_contain="new_file.py",
        must_not_contain="old_file.py",
    )
    refreshed_names = _names(refreshed["files"])
    assert "new_file.py" in refreshed_names
    assert "old_file.py" not in refreshed_names

    # 5) The cache itself must be updated so the *next* ``getFiles``
    #    sees the post-task file list synchronously (no rescan).
    events.clear()
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": workspace},
    )
    second = _wait_for_files_event(events, must_contain="new_file.py")
    second_names = _names(second["files"])
    assert "new_file.py" in second_names
    assert "old_file.py" not in second_names


def test_refresh_skipped_when_no_files_added_or_removed(
    workspace: str,
) -> None:
    """When the agent only *modified* existing files (no creation or
    deletion), the hook must not broadcast a redundant ``files`` event.

    Modifications never change the picker's list, so a broadcast
    would only push a no-op event to every connected client.
    """
    server, events = _make_server(workspace)

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": workspace},
    )
    _wait_for_files_event(events, must_contain="old_file.py")

    # Only modify content (no add/delete).
    (Path(workspace) / "old_file.py").write_text("# modified\n")

    events.clear()
    server._refresh_files_after_task(workspace)

    # Give the background thread time to scan and decide not to emit.
    time.sleep(0.5)
    files_events = [e for e in events if e.get("type") == "files"]
    assert files_events == [], (
        "Hook must not broadcast a files event when the file set is "
        f"unchanged; got {files_events}"
    )


def test_refresh_no_op_when_cache_never_warmed(
    workspace: str,
) -> None:
    """When no ``@``-mention picker has ever opened on *work_dir*,
    the hook must be a no-op: there is nothing to keep fresh, and
    the next ``getFiles`` will scan from scratch anyway.
    """
    server, events = _make_server(workspace)

    # Cache is empty for workspace — no prior getFiles.
    assert workspace not in server._file_cache

    (Path(workspace) / "new_file.py").write_text("# new\n")
    server._refresh_files_after_task(workspace)

    time.sleep(0.3)
    # No files event should be broadcast and no cache entry created.
    assert workspace not in server._file_cache
    assert [e for e in events if e.get("type") == "files"] == []
