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
index to that tab's directory so a user editing in tab A (rooted at
``/proj-a``) doesn't see suggestions from the unrelated tab B (rooted
at ``/proj-b``).

These tests drive ``VSCodeServer._handle_command`` directly with
``{"type": "getFiles", "workDir": ...}`` payloads and verify the
emitted ``files`` event reflects the requested directory regardless
of the daemon-wide ``work_dir``.  They also verify the
``FileIndexRegistry`` builds an index per root so two tabs rooted at
different folders never share results.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.file_index import FileIndexRegistry
from kiss.server.server import VSCodeServer

_Picker = tuple[str, str, VSCodeServer, list[dict[str, Any]]]


def _wait_for_files_event(
    events: list[dict[str, Any]], timeout: float = 5.0,
) -> dict[str, Any]:
    """Return the first non-loading ``files`` event in *events*.

    Polls until either such an event appears or *timeout* elapses.
    The handler emits a ``loading=True`` placeholder when the root is
    not indexed yet and then a populated event once the registry
    worker has built the index; only the latter is meaningful for
    these assertions.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for e in list(events):
            if e.get("type") == "files" and not e.get("loading"):
                return e
        time.sleep(0.01)
    raise AssertionError(f"no populated files event arrived; got {events}")


def _names(entries: list[Any]) -> list[str]:
    """Project the ``text`` field out of ranked file suggestions."""
    return [e["text"] if isinstance(e, dict) else str(e) for e in entries]


@pytest.fixture()
def picker(tmp_path: Path) -> Iterator[_Picker]:
    """Yield ``(a, b, server, events)``: two workspaces and a daemon on ``a``.

    Folder ``a`` contains only ``alpha.txt``; folder ``b`` contains
    only ``beta.txt``.  Each filename uniquely identifies which
    folder an index picked up so assertions can match without
    ambiguity.  The server's file-index registry is a private one
    whose home is an empty directory, so both workspaces are roots
    of their own and nothing outside ``tmp_path`` is scanned.
    ``events`` collects every ``printer.broadcast`` call (appended
    under a lock: the populated reply arrives from the registry's
    worker thread while the test polls from the main thread).
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "alpha.txt").write_text("alpha")
    (b / "beta.txt").write_text("beta")
    home = tmp_path / "home"
    home.mkdir()

    server = VSCodeServer()
    server.work_dir = str(a)
    server._file_index.stop()
    server._file_index = FileIndexRegistry(home=str(home), cache_dir=tmp_path / "cache")
    captured: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            captured.append(dict(event))

    server.printer.broadcast = capture  # type: ignore[method-assign]
    try:
        yield str(a), str(b), server, captured
    finally:
        server._file_index.stop()


def test_get_files_uses_explicit_work_dir_overriding_daemon_default(
    picker: _Picker,
) -> None:
    """``getFiles`` with ``workDir=b`` must index folder B even when
    the daemon-wide ``work_dir`` is folder A.

    Reproduces the bug: previously the handler ignored ``workDir``
    on the command and always scanned ``self.work_dir``, so every
    tab's file picker leaked the daemon-wide files regardless of
    which folder the tab was tied to.
    """
    _a, b, server, events = picker

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": b},
    )
    populated = _wait_for_files_event(events)
    files = _names(populated["files"])
    assert "beta.txt" in files, (
        f"workDir=b must index folder B; got {files}"
    )
    assert "alpha.txt" not in files, (
        f"folder A files must not leak when workDir=b; got {files}"
    )


def test_get_files_per_tab_indexes_are_independent(
    picker: _Picker,
) -> None:
    """Two tabs pointed at different folders must each get their own
    file list — no cross-contamination from a shared index.

    Sends two ``getFiles`` commands back-to-back with different
    ``workDir`` values and asserts each event reflects only the
    files in its respective folder.  Then checks the registry holds
    one index per root, each listing only its own files, and
    re-queries directory A: with its index warm the reply must be
    synchronous (no ``loading`` placeholder) and still folder-A only.
    """
    a, b, server, events = picker

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": a},
    )
    a_evt = _wait_for_files_event(events)
    a_files = _names(a_evt["files"])
    assert "alpha.txt" in a_files
    assert "beta.txt" not in a_files

    events.clear()
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": b},
    )
    b_evt = _wait_for_files_event(events)
    b_files = _names(b_evt["files"])
    assert "beta.txt" in b_files
    assert "alpha.txt" not in b_files

    registry = server._file_index
    assert set(registry._indexes) == {a, b}
    view_a = registry.view_for(a)
    view_b = registry.view_for(b)
    assert view_a is not None and view_b is not None
    assert "alpha.txt" in view_a and "beta.txt" not in view_a
    assert "beta.txt" in view_b and "alpha.txt" not in view_b

    events.clear()
    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": a},
    )
    a2 = _wait_for_files_event(events)
    assert "alpha.txt" in _names(a2["files"])
    assert "beta.txt" not in _names(a2["files"])
    assert not any(e.get("loading") for e in events), (
        f"a warm index must answer synchronously; got {events}"
    )


def test_get_files_falls_back_to_daemon_work_dir_when_workdir_missing(
    picker: _Picker,
) -> None:
    """A ``getFiles`` command without ``workDir`` must fall back to
    the daemon-wide ``work_dir`` (legacy behaviour).

    This guards against breaking any code path that hasn't been
    updated to stamp ``workDir`` (e.g. older clients) — the handler
    must still serve files, just from the daemon's default folder.
    """
    _a, _b, server, events = picker

    server._handle_command({"type": "getFiles", "prefix": ""})
    populated = _wait_for_files_event(events)
    files = _names(populated["files"])
    assert "alpha.txt" in files, (
        f"missing workDir must fall back to daemon work_dir; got {files}"
    )


def test_get_files_empty_string_workdir_falls_back_to_daemon_work_dir(
    picker: _Picker,
) -> None:
    """An explicit empty-string ``workDir`` must also fall back.

    The frontend's ``workDirForTab`` helper returns ``''`` when a
    tab has not yet received a background-task event setting its
    ``workDir``, so the handler must treat an empty string the
    same as a missing field — and never index folder B.
    """
    _a, b, server, events = picker

    server._handle_command(
        {"type": "getFiles", "prefix": "", "workDir": ""},
    )
    populated = _wait_for_files_event(events)
    files = _names(populated["files"])
    assert "alpha.txt" in files
    assert "beta.txt" not in files, (
        f"empty workDir must NOT index folder B; got {files}"
    )
    assert b not in server._file_index._indexes
    assert server._file_index.view_for(b) is None
