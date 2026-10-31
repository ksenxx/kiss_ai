# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30 (S11): races and failure paths of ``chrome_bridge``.

Real bridges on loopback, driven by the scripted extension peer of
``test_chrome_bridge`` over real sockets.  The races are landed with
:class:`LinePause`, a ``sys.settrace`` hook that holds a thread at one
source line of the real code (nothing is replaced) until the competing
thread has done its part:

* ``_drop_connection`` checked "is this still the live connection?" and
  cleared it in two steps; a new extension's handshake landing between
  them was cleared with the old one, so the bridge reported "not
  connected" while the new extension stayed connected and pinging.  The
  check and the swap are now one step under ``_conn_lock``.
* ``stop()`` read ``self._conn`` twice; a disconnect between the reads
  made it call ``_drop_connection(None)`` and raise, so the config file
  was never withdrawn.  It reads once now.
* ``stop()`` checked the config file's token and unlinked it in two
  steps; a successor publishing in between had its file deleted, so the
  extension could not find the live bridge.  Publication and withdrawal
  now hold the same file lock.
* An unwritable ``$KISS_HOME/chrome_extension`` made ``start()`` raise
  ``OSError`` (not ``ChromeBridgeError``): it escaped ``start_if_enabled``
  (daemon boot and settings saves) and ``use_my_chrome``, and leaked the
  bound port.  It is a ``ChromeBridgeError`` now and the port is closed.
"""

from __future__ import annotations

import inspect
import json
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

import pytest

from kiss.agents.sorcar import chrome_bridge
from kiss.agents.sorcar.chrome_bridge import (
    CONFIG_FILE_NAME,
    ChromeBridge,
    ChromeBridgeError,
    extension_dir,
    start_if_enabled,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.user_config import save_config
from kiss.tests.agents.sorcar.test_chrome_bridge import ExtensionPeer


class LinePause:
    """Hold the first thread about to run *line* of *owner* until the block is left.

    *owner* is a function or a module; *line* is compared with each of its
    source lines stripped of indentation and trailing comment.  On leaving
    the block the held thread is let go, and the hook stays installed
    until the held frame has returned, so :attr:`returned` tells the test
    the window is closed.
    """

    def __init__(self, owner: Callable[..., object] | ModuleType, line: str) -> None:
        source, start = inspect.getsourcelines(owner)
        first = start or 1  # a module's source starts at line 1
        self._file = inspect.getsourcefile(owner)
        self._linenos = {
            first + index
            for index, text in enumerate(source)
            if text.split("#")[0].strip() == line
        }
        assert self._linenos, f"{line!r} not found"
        self.paused = threading.Event()
        self.returned = threading.Event()
        self._go = threading.Event()
        self._held: FrameType | None = None

    def __enter__(self) -> LinePause:
        threading.settrace_all_threads(self._trace_calls)
        return self

    def __exit__(self, *exc: object) -> None:
        self._go.set()
        if self.paused.is_set():
            self.returned.wait(10)
        threading.settrace_all_threads(None)  # type: ignore[arg-type]

    def _trace_calls(self, frame: FrameType, event: str, arg: Any) -> Any:
        return self._trace_lines if frame.f_code.co_filename == self._file else None

    def _trace_lines(self, frame: FrameType, event: str, arg: Any) -> Any:
        if event == "line" and frame.f_lineno in self._linenos and not self.paused.is_set():
            self._held = frame
            self.paused.set()
            self._go.wait(30)
        elif event == "return" and frame is self._held:
            self.returned.set()
        return self._trace_lines


def _eventually(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def _config() -> dict[str, Any]:
    return dict(json.loads((extension_dir() / CONFIG_FILE_NAME).read_text()))


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private ``KISS_HOME`` (extension dir, config file, config.json); no singleton."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    ChromeBridge.shutdown()
    yield tmp_path
    ChromeBridge.shutdown()


@pytest.fixture
def bridge() -> Iterator[ChromeBridge]:
    b = ChromeBridge(port=0, write_config=False)
    b.start()
    yield b
    b.stop()


def test_a_drop_racing_a_new_handshake_keeps_the_new_connection(bridge: ChromeBridge) -> None:
    """The old connection's drop is held past its check while a new extension connects."""
    first = ExtensionPeer(bridge.port, bridge.token)
    assert _eventually(lambda: bridge.extension_connected)
    with LinePause(ChromeBridge._drop_connection, "self._conn = None") as pause:
        first.close()
        assert pause.paused.wait(5), "the old connection was never dropped"
        second = ExtensionPeer(bridge.port, bridge.token, responder=lambda cmd: "answered")
        # The second handler reads frames only after its swap, so a pong
        # proves the swap happened inside the window; with the swap
        # serialised behind the held drop none comes until it is let go.
        second.ws.send(json.dumps({"type": "ping"}))
        try:
            second.incoming.get(timeout=1)
        except queue.Empty:
            pass
    try:
        assert pause.returned.is_set()
        assert _eventually(lambda: bridge.extension_connected)
        assert bridge.call("list_tabs", timeout=5) == "answered"
    finally:
        second.close()


def test_stop_racing_a_disconnect_withdraws_the_config(kiss_home: Path) -> None:
    """The extension disconnects while ``stop()`` is about to take the connection."""
    b = ChromeBridge(write_config=True)
    b.start()
    peer = ExtensionPeer(b.port, b.token)
    assert _eventually(lambda: b.extension_connected)
    errors: list[BaseException] = []

    def stop() -> None:
        try:
            b.stop()
        except BaseException as e:  # noqa: BLE001 - asserted below
            errors.append(e)

    with LinePause(ChromeBridge.stop, "conn = self._conn") as pause:
        stopper = threading.Thread(target=stop)
        stopper.start()
        assert pause.paused.wait(5)
        peer.close()
        assert _eventually(lambda: not b.extension_connected)
    stopper.join(10)
    assert errors == []
    assert not (extension_dir() / CONFIG_FILE_NAME).exists()


def test_a_predecessors_stop_never_unlinks_the_successors_config(kiss_home: Path) -> None:
    """A successor publishes while the predecessor's stop is between its check and unlink."""
    predecessor = ChromeBridge(write_config=True)
    predecessor.start(takeover=True)
    successor = ChromeBridge(write_config=True)
    with LinePause(chrome_bridge, "config.unlink(missing_ok=True)") as pause:
        stopper = threading.Thread(target=predecessor.stop)
        stopper.start()
        assert pause.paused.wait(5), "the predecessor never reached its unlink"
        starter = threading.Thread(target=successor.start, kwargs={"takeover": True})
        starter.start()
        # Unserialised, the successor publishes now; serialised, it waits
        # for the predecessor's withdrawal to finish.
        starter.join(1.0)
    stopper.join(10)
    starter.join(10)
    try:
        assert not starter.is_alive()
        assert _config() == {"port": successor.port, "token": successor.token}
    finally:
        successor.stop()
    assert not (extension_dir() / CONFIG_FILE_NAME).exists()


def _block_extension_dir(kiss_home: Path) -> None:
    (kiss_home / "chrome_extension").write_text("a file where the extension directory belongs")


def test_an_unwritable_extension_dir_is_a_bridge_error_and_frees_the_port(
    kiss_home: Path,
) -> None:
    _block_extension_dir(kiss_home)
    b = ChromeBridge(write_config=True)
    with pytest.raises(ChromeBridgeError, match="cannot publish the bridge"):
        b.start()
    assert b.port != 0
    # Nothing answers on the port any more.  Windows (its firewall's
    # stealth mode) drops the SYN to a closed port instead of refusing
    # it, so there the connect times out; a listener left behind would
    # accept at once either way.
    with pytest.raises((ConnectionRefusedError, TimeoutError)):
        socket.create_connection(("127.0.0.1", b.port), timeout=2).close()


def test_start_if_enabled_survives_an_unwritable_extension_dir(kiss_home: Path) -> None:
    """Daemon boot (and every settings save) logs the failure instead of raising."""
    save_config({"chrome_bridge": True})
    _block_extension_dir(kiss_home)
    start_if_enabled()
    assert ChromeBridge._instance is None


def test_use_my_chrome_reports_an_unwritable_extension_dir(kiss_home: Path) -> None:
    _block_extension_dir(kiss_home)
    out = WebUseTool().use_my_chrome("list_tabs")
    assert out.startswith("Error starting the Chrome bridge: cannot publish the bridge"), out


def test_stop_logs_a_config_it_cannot_withdraw(
    kiss_home: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """``stop()`` runs at exit: a config lock it cannot take is logged, never raised."""
    b = ChromeBridge(write_config=True)
    b.start()
    (kiss_home / "chrome_extension").rename(kiss_home / "moved_away")
    _block_extension_dir(kiss_home)
    with caplog.at_level(logging.WARNING, logger=chrome_bridge.__name__):
        b.stop()
    assert "could not withdraw" in caplog.text
