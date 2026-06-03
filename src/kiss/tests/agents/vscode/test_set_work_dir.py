"""Integration test: changing the VS Code workspace folder must
update the agent's working directory.

The VS Code extension calls ``vscode.workspace.workspaceFolders``
inside its ``_getWorkDir()`` helper and passes the resulting path on
every ``run`` command, but commands that don't carry an explicit
``workDir`` — notably autocomplete (``getFiles``), commit-message
generation, and worktree actions — read ``VSCodeServer.work_dir``,
which was captured once from ``KISS_WORKDIR``/``os.getcwd()`` at
process start.  Without a ``setWorkDir`` command the daemon never
notices that the user switched folders, so file autocomplete and
related commands keep using the stale init value.

These tests reproduce that mismatch and verify the ``setWorkDir``
handler keeps ``server.work_dir`` and the autocomplete file cache
synchronised with the active VS Code folder.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.vscode.server import VSCodeServer


def _wait_for(predicate, timeout: float = 5.0) -> None:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("predicate never became true")


@pytest.fixture()
def two_workspaces() -> Any:
    """Create two temp dirs simulating two VS Code workspace folders.

    Folder A contains ``alpha.txt`` only; folder B contains
    ``beta.txt`` only.  Each folder's basename uniquely identifies
    its file set so autocomplete results can be matched without
    ambiguity.
    """
    a = tempfile.mkdtemp(prefix="kiss_ws_a_")
    b = tempfile.mkdtemp(prefix="kiss_ws_b_")
    (Path(a) / "alpha.txt").write_text("alpha")
    (Path(b) / "beta.txt").write_text("beta")
    try:
        yield a, b
    finally:
        import shutil
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def _make_server(work_dir: str) -> VSCodeServer:
    """Build a ``VSCodeServer`` pinned to ``work_dir``."""
    server = VSCodeServer()
    server.work_dir = work_dir
    return server


def _capture_files_events(server: VSCodeServer) -> list[dict[str, Any]]:
    """Install a printer wrapper that captures every ``files`` event."""
    captured: list[dict[str, Any]] = []
    lock = threading.Lock()
    orig = server.printer.broadcast

    def wrapped(event: dict[str, Any]) -> None:
        with lock:
            captured.append(dict(event))
        orig(event)

    server.printer.broadcast = wrapped  # type: ignore[method-assign]
    return captured


def test_set_work_dir_updates_field_and_invalidates_caches(
    two_workspaces: tuple[str, str],
) -> None:
    """``setWorkDir`` must update ``work_dir`` and clear stale caches.

    The bug: ``VSCodeServer.work_dir`` is captured once at __init__
    from ``KISS_WORKDIR``/``getcwd()`` and never refreshed.  Once
    ``_file_cache``, ``_last_active_file``, or
    ``_last_active_content`` are populated against folder A, those
    values must be discarded the moment the user switches to
    folder B — otherwise stale entries leak across workspaces.
    """
    a, b = two_workspaces
    server = _make_server(a)
    # Seed the caches with values that came from folder A.
    with server._state_lock:
        server._file_cache = ["alpha.txt"]
        server._last_active_file = os.path.join(a, "alpha.txt")
        server._last_active_content = "alpha"

    server._handle_command({"type": "setWorkDir", "workDir": b})

    assert server.work_dir == b, (
        "setWorkDir must update work_dir to the new workspace folder"
    )
    assert server._file_cache is None, (
        "file_cache holds folder-A files; must be cleared on folder change"
    )
    assert server._last_active_file == ""
    assert server._last_active_content == ""


def test_set_work_dir_ignored_when_empty(two_workspaces: tuple[str, str]) -> None:
    """An empty ``workDir`` must be a no-op (no destructive reset)."""
    a, _b = two_workspaces
    server = _make_server(a)
    with server._state_lock:
        server._file_cache = ["alpha.txt"]
    server._handle_command({"type": "setWorkDir", "workDir": ""})
    assert server.work_dir == a
    assert server._file_cache == ["alpha.txt"], (
        "empty workDir must not invalidate caches"
    )


def test_set_work_dir_idempotent_when_unchanged(
    two_workspaces: tuple[str, str],
) -> None:
    """Repeating the same ``workDir`` must not wipe an active cache.

    Folder-change events can fire spuriously (e.g. on every
    workspace mutation); reapplying the same value must be a no-op
    so a freshly-populated file cache survives.
    """
    a, _b = two_workspaces
    server = _make_server(a)
    with server._state_lock:
        server._file_cache = ["alpha.txt"]
    server._handle_command({"type": "setWorkDir", "workDir": a})
    assert server._file_cache == ["alpha.txt"]


def test_get_files_returns_new_workspace_after_set_work_dir(
    two_workspaces: tuple[str, str],
) -> None:
    """End-to-end reproduction: autocomplete must reflect folder B
    after ``setWorkDir`` even though the server started in folder A.

    Without the fix, ``_get_files`` reads ``self.work_dir`` (frozen
    at init time) and emits folder A's files forever; with the fix,
    ``setWorkDir`` invalidates ``_file_cache`` so the next
    ``getFiles`` scan picks up folder B's files.
    """
    a, b = two_workspaces
    server = _make_server(a)
    captured = _capture_files_events(server)

    # Prime the cache with folder A by calling _get_files once and
    # waiting for the background scan to publish results.
    server._handle_command({"type": "getFiles", "prefix": ""})
    _wait_for(
        lambda: any(
            e.get("type") == "files" and not e.get("loading")
            for e in captured
        ),
    )
    a_files = next(
        e["files"] for e in captured
        if e.get("type") == "files" and not e.get("loading")
    )
    def _names(entries: list[Any]) -> list[str]:
        return [e["text"] if isinstance(e, dict) else str(e) for e in entries]

    assert "alpha.txt" in _names(a_files), (
        f"folder A scan must include alpha.txt; got {a_files}"
    )
    assert "beta.txt" not in _names(a_files)

    # User opens folder B in VS Code; extension sends setWorkDir.
    captured.clear()
    server._handle_command({"type": "setWorkDir", "workDir": b})

    # Next autocomplete request must trigger a fresh scan rooted at B.
    server._handle_command({"type": "getFiles", "prefix": ""})
    _wait_for(
        lambda: any(
            e.get("type") == "files" and not e.get("loading")
            for e in captured
        ),
    )
    b_files = next(
        e["files"] for e in captured
        if e.get("type") == "files" and not e.get("loading")
    )
    assert "beta.txt" in _names(b_files), (
        f"after setWorkDir, scan must include folder B files; got {b_files}"
    )
    assert "alpha.txt" not in _names(b_files), (
        f"folder A files must not leak after switching to B; got {b_files}"
    )


def test_set_work_dir_syncs_web_printer_work_dir(
    two_workspaces: tuple[str, str],
) -> None:
    """``setWorkDir`` must propagate to the ``WebPrinter.work_dir``.

    The remote server's ``WebPrinter`` fills ``cfg["work_dir"]`` in
    global ``configData`` events from its own ``work_dir`` attribute.
    The handler only updated ``VSCodeServer.work_dir`` before, so a
    browser client kept seeing the folder the daemon was launched with
    after the user opened a new folder in VS Code.  After the fix the
    printer's ``work_dir`` tracks the active folder and the next
    ``configData`` reports folder B.
    """
    from kiss.agents.vscode.web_server import WebPrinter

    a, b = two_workspaces
    printer = WebPrinter()
    printer.work_dir = a
    server = VSCodeServer(printer=printer)
    server.work_dir = a

    server._handle_command({"type": "setWorkDir", "workDir": b})

    assert printer.work_dir == b, (
        "setWorkDir must sync WebPrinter.work_dir to the new folder"
    )

    # A configData event with an empty work_dir is filled in place by
    # the printer from its own ``work_dir`` attribute, so it must now
    # report folder B (no WebSocket/UDS clients are attached, so the
    # broadcast is otherwise a no-op).
    cfg: dict[str, Any] = {"work_dir": ""}
    event: dict[str, Any] = {"type": "configData", "config": cfg}
    printer.broadcast(event)
    assert cfg["work_dir"] == b, (
        f"configData must report folder B after setWorkDir; got {event}"
    )


def test_set_work_dir_registered_in_handlers() -> None:
    """The dispatch table must include ``setWorkDir``.

    Without this entry the unknown-command branch would broadcast a
    user-visible ``error`` event every time the extension pushes the
    current workspace folder.
    """
    from kiss.agents.vscode.commands import _CommandsMixin

    assert "setWorkDir" in _CommandsMixin._HANDLERS
