# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: the ``@``-mention file picker must list files
relative to the *active chat tab's* ``work_dir``, not the daemon-wide
``VSCodeServer.work_dir``.

Each chat tab in the extension/webapp UI can be pinned to its own
working directory (set from background-task events the agent emits
while it runs).  When the user types ``@`` in the chat-input textbox
the frontend posts a ``getFiles`` command that must scope the file
scan to that tab's directory so a user editing in tab A (rooted at
``/proj-a``) doesn't see suggestions from the unrelated tab B (rooted
at ``/proj-b``).

These tests drive ``VSCodeServer._handle_command`` directly with
``{"type": "getFiles", "workDir": ...}`` payloads and verify the
emitted ``files`` event reflects the requested directory regardless
of the daemon-wide ``work_dir``.  They also verify the per-work_dir
file cache is populated independently for each directory so two
tabs scanning different folders never share results.
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
    events: list[dict[str, Any]], timeout: float = 5.0,
) -> dict[str, Any]:
    """Return the first non-loading ``files`` event in *events*.

    Polls until either such an event appears or *timeout* elapses.
    The handler emits a ``loading=True`` placeholder when the cache
    is empty and then a populated event after the background scan
    finishes; only the latter is meaningful for these assertions.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for e in events:
            if e.get("type") == "files" and not e.get("loading"):
                return e
        time.sleep(0.01)
    raise AssertionError(f"no populated files event arrived; got {events}")


def _names(entries: list[Any]) -> list[str]:
    """Project the ``text`` field out of ranked file suggestions."""
    return [e["text"] if isinstance(e, dict) else str(e) for e in entries]


@pytest.fixture()
def two_workspaces() -> Iterator[tuple[str, str]]:
    """Create two temp dirs containing uniquely-named files.

    Folder ``a`` contains only ``alpha.txt``; folder ``b`` contains
    only ``beta.txt``.  Each filename uniquely identifies which
    folder a scan picked up so assertions can match without ambiguity.
    """
    a = tempfile.mkdtemp(prefix="kiss_picker_a_")
    b = tempfile.mkdtemp(prefix="kiss_picker_b_")
    (Path(a) / "alpha.txt").write_text("alpha")
    (Path(b) / "beta.txt").write_text("beta")
    try:
        yield a, b
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def _make_server(work_dir: str) -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Build a ``VSCodeServer`` pinned to *work_dir* with a capturing printer.

    The returned event list is thread-safely appended to from every
    ``printer.broadcast`` call so the tests can poll it from the
    main thread while background scan threads emit results.
    """
    server = VSCodeServer()
    server.work_dir = work_dir
    captured: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            captured.append(dict(event))

    server.printer.broadcast = capture  # type: ignore[method-assign]
    return server, captured


def test_get_files_uses_explicit_work_dir_overriding_daemon_default(
    two_workspaces: tuple[str, str],
) -> None:
    """``getFiles`` with ``workDir=b`` must scan folder B even when
    the daemon-wide ``work_dir`` is folder A.

    Reproduces the bug: previously the handler ignored ``workDir``
    on the command and always scanned ``self.work_dir``, so every
    tab's file picker leaked the daemon-wide files regardless of
    which folder the tab was tied to.
    """
    a, b = two_workspaces
    server, events = _make_server(a)

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": b},
    )
    populated = _wait_for_files_event(events)
    files = _names(populated["files"])
    assert "beta.txt" in files, (
        f"workDir=b must scan folder B; got {files}"
    )
    assert "alpha.txt" not in files, (
        f"folder A files must not leak when workDir=b; got {files}"
    )


def test_get_files_per_tab_caches_are_independent(
    two_workspaces: tuple[str, str],
) -> None:
    """Two tabs pointed at different folders must each get their own
    file list — no cross-contamination from a shared cache.

    Sends two ``getFiles`` commands back-to-back with different
    ``workDir`` values and asserts each event reflects only the
    files in its respective folder.  Then re-queries each directory
    (cache hit) and verifies the cached lists were keyed separately.
    """
    a, b = two_workspaces
    server, events = _make_server(a)

    # Tab A → workDir=a
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": a},
    )
    a_evt = _wait_for_files_event(events)
    a_files = _names(a_evt["files"])
    assert "alpha.txt" in a_files
    assert "beta.txt" not in a_files

    # Tab B → workDir=b (must NOT reuse tab A's cache)
    events.clear()
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": b},
    )
    b_evt = _wait_for_files_event(events)
    b_files = _names(b_evt["files"])
    assert "beta.txt" in b_files
    assert "alpha.txt" not in b_files

    # Both caches are populated independently.
    with server._state_lock:
        assert a in server._file_cache
        assert b in server._file_cache
        assert "alpha.txt" in server._file_cache[a]
        assert "beta.txt" in server._file_cache[b]
        assert "alpha.txt" not in server._file_cache[b]
        assert "beta.txt" not in server._file_cache[a]

    # Cache-hit path: a second query against each folder must return
    # the same files synchronously (no second loading placeholder is
    # required for correctness, but the populated event must match).
    events.clear()
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": a},
    )
    a2 = _wait_for_files_event(events)
    assert "alpha.txt" in _names(a2["files"])
    assert "beta.txt" not in _names(a2["files"])


def test_get_files_falls_back_to_daemon_work_dir_when_workdir_missing(
    two_workspaces: tuple[str, str],
) -> None:
    """A ``getFiles`` command without ``workDir`` must fall back to
    the daemon-wide ``work_dir`` (legacy behaviour).

    This guards against breaking any code path that hasn't been
    updated to stamp ``workDir`` (e.g. older clients) — the handler
    must still serve files, just from the daemon's default folder.
    """
    a, _b = two_workspaces
    server, events = _make_server(a)

    server._handle_command({"type": "getFiles", "prefix": ""})
    populated = _wait_for_files_event(events)
    files = _names(populated["files"])
    assert "alpha.txt" in files, (
        f"missing workDir must fall back to daemon work_dir; got {files}"
    )


def test_get_files_empty_string_workdir_falls_back_to_daemon_work_dir(
    two_workspaces: tuple[str, str],
) -> None:
    """An explicit empty-string ``workDir`` must also fall back.

    The frontend's ``workDirForTab`` helper returns ``''`` when a
    tab has not yet received a background-task event setting its
    ``workDir``, so the handler must treat an empty string the
    same as a missing field.
    """
    a, b = two_workspaces
    server, events = _make_server(a)

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": ""},
    )
    populated = _wait_for_files_event(events)
    files = _names(populated["files"])
    assert "alpha.txt" in files
    assert "beta.txt" not in files, (
        f"empty workDir must NOT scan folder B; got {files}"
    )
    # And folder B's cache must remain untouched (no spurious key).
    with server._state_lock:
        assert b not in server._file_cache
