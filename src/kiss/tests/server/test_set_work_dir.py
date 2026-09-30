# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
handler keeps ``server.work_dir`` synchronised with the active VS Code
folder, pre-warms the ``@``-mention file index of a newly adopted
folder (``_file_index.ensure``), and queues nothing for an empty or
unchanged folder.  Every server gets a private
:class:`~kiss.server.file_index.FileIndexRegistry` rooted in the test's
temp dir so the real home directory is never scanned.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.file_index import FileIndex, FileIndexRegistry
from kiss.server.server import VSCodeServer


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("predicate never became true")


@pytest.fixture()
def two_workspaces() -> Iterator[tuple[str, str]]:
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
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


class _Servers:
    """Build ``VSCodeServer`` instances with private file-index registries.

    Every registry is rooted in the test's temp dir (``home`` and cache
    dir) and stopped at teardown so no worker outlives the test.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.registries: list[FileIndexRegistry] = []

    def make(self, work_dir: str, printer: Any = None) -> VSCodeServer:
        """Return a server pinned to *work_dir* with a private registry."""
        server = VSCodeServer(printer=printer) if printer is not None else VSCodeServer()
        server.work_dir = work_dir
        server._file_index.stop()
        registry = FileIndexRegistry(
            home=str(self.tmp_path / "home"), cache_dir=self.tmp_path / "cache",
        )
        server._file_index = registry
        self.registries.append(registry)
        return server

    def drain(self, server: VSCodeServer) -> None:
        """Block until every build queued so far on the server's registry ran.

        The registry's single worker serves jobs in FIFO order, so a
        sentinel job on an unrelated directory completing proves the
        earlier jobs (or their absence) have been decided.
        """
        done = threading.Event()
        server._file_index.ensure(str(self.tmp_path / "sentinel"), done.set)
        assert done.wait(10.0), "the file-index worker never ran the sentinel"

    def stop_all(self) -> None:
        for registry in self.registries:
            registry.stop()


@pytest.fixture()
def servers(tmp_path: Path) -> Iterator[_Servers]:
    """Yield a server factory whose registries are stopped at teardown."""
    factory = _Servers(tmp_path)
    try:
        yield factory
    finally:
        factory.stop_all()


def _index_of(server: VSCodeServer, work_dir: str) -> FileIndex | None:
    """Return the index object currently serving *work_dir*, if any."""
    registry = server._file_index
    root, _ = registry.root_for(work_dir)
    with registry._lock:
        return registry._indexes.get(root)


def _build(server: VSCodeServer, work_dir: str) -> FileIndex:
    """Index *work_dir* on the server's registry and return the index."""
    done = threading.Event()
    server._file_index.ensure(work_dir, done.set)
    assert done.wait(10.0), "the index build never finished"
    index = _index_of(server, work_dir)
    assert index is not None
    return index


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


def _names(entries: list[Any]) -> list[str]:
    return [e["text"] if isinstance(e, dict) else str(e) for e in entries]


def test_set_work_dir_updates_field_and_prewarms_the_index(
    two_workspaces: tuple[str, str], servers: _Servers,
) -> None:
    """``setWorkDir`` must update ``work_dir``, drop the connection's
    active-file snapshot, and start indexing the new folder.

    The bug: ``VSCodeServer.work_dir`` is captured once at __init__
    from ``KISS_WORKDIR``/``getcwd()`` and never refreshed.  Once
    ``_last_active_file`` / ``_last_active_content`` are populated
    against folder A, those values must be discarded the moment the
    user switches to folder B — otherwise stale entries leak across
    workspaces — and folder B's ``@``-mention index must be built so
    the first ``getFiles`` there is answered from a warm index.
    """
    a, b = two_workspaces
    server = servers.make(a)
    with server._state_lock:
        server._last_active_file[""] = os.path.join(a, "alpha.txt")
        server._last_active_content[""] = "alpha"
    assert server._file_index.view_for(b) is None

    server._handle_command({"type": "setWorkDir", "workDir": b})

    assert server.work_dir == b, (
        "setWorkDir must update work_dir to the new workspace folder"
    )
    assert server._last_active_file.get("", "") == ""
    assert server._last_active_content.get("", "") == ""
    _wait_for(lambda: server._file_index.view_for(b) is not None)
    view = server._file_index.view_for(b)
    assert view is not None and "beta.txt" in view.paths, (
        "setWorkDir must pre-warm the new folder's file index"
    )


def test_set_work_dir_ignored_when_empty(
    two_workspaces: tuple[str, str], servers: _Servers,
) -> None:
    """An empty ``workDir`` must be a no-op (no adoption, no index build)."""
    a, _b = two_workspaces
    server = servers.make(a)
    server._handle_command({"type": "setWorkDir", "workDir": ""})
    assert server.work_dir == a
    servers.drain(server)
    assert _index_of(server, a) is None, (
        "empty workDir must not queue an index build"
    )
    # Nor may it disturb an index that already exists.
    done = threading.Event()
    server._file_index.ensure(a, done.set)
    assert done.wait(10.0)
    warm = _index_of(server, a)
    assert warm is not None
    server._handle_command({"type": "setWorkDir", "workDir": ""})
    servers.drain(server)
    assert _index_of(server, a) is warm, "empty workDir must leave the warm index alone"


def test_set_work_dir_idempotent_when_unchanged(
    two_workspaces: tuple[str, str], servers: _Servers,
) -> None:
    """Repeating the same ``workDir`` must not rebuild a warm index.

    Folder-change events can fire spuriously (e.g. on every
    workspace mutation); reapplying the same value must be a no-op
    so a freshly-built file index survives untouched.
    """
    a, _b = two_workspaces
    server = servers.make(a)
    index = _build(server, a)
    server._handle_command({"type": "setWorkDir", "workDir": a})
    servers.drain(server)
    assert _index_of(server, a) is index, (
        "re-announcing the same work dir must not rebuild its index"
    )


def test_get_files_returns_new_workspace_after_set_work_dir(
    two_workspaces: tuple[str, str], servers: _Servers,
) -> None:
    """End-to-end reproduction: autocomplete must reflect folder B
    after ``setWorkDir`` even though the server started in folder A.

    Without the fix, ``_get_files`` reads ``self.work_dir`` (frozen
    at init time) and emits folder A's files forever; with the fix,
    ``setWorkDir`` adopts folder B so the next unstamped ``getFiles``
    is served from folder B's index.
    """
    a, b = two_workspaces
    server = servers.make(a)
    captured = _capture_files_events(server)

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
    assert "./alpha.txt" in _names(a_files), (
        f"folder A scan must include alpha.txt; got {a_files}"
    )
    assert "./beta.txt" not in _names(a_files)

    captured.clear()
    server._handle_command({"type": "setWorkDir", "workDir": b})

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
    assert "./beta.txt" in _names(b_files), (
        f"after setWorkDir, scan must include folder B files; got {b_files}"
    )
    assert "./alpha.txt" not in _names(b_files), (
        f"folder A files must not leak after switching to B; got {b_files}"
    )


def test_set_work_dir_syncs_web_printer_work_dir(
    two_workspaces: tuple[str, str], servers: _Servers,
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
    from kiss.server.web_server import WebPrinter

    a, b = two_workspaces
    printer = WebPrinter()
    printer.work_dir = a
    server = servers.make(a, printer=printer)

    server._handle_command({"type": "setWorkDir", "workDir": b})

    assert printer.work_dir == b, (
        "setWorkDir must sync WebPrinter.work_dir to the new folder"
    )

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
    from kiss.server.commands import _CommandsMixin

    assert "setWorkDir" in _CommandsMixin._HANDLERS
