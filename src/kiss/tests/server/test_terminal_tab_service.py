# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the terminal-tab service behind the remote webapp's "Terminal".

:class:`kiss.server.terminal_tab.TerminalService` runs a real shell on a
real pty here: output streams as ``terminalData`` events to the owning
connection only, the pty follows ``resize``, the shell's exit code
arrives as ``terminalExit``, a tab that was closed hangs its shell up
(and kills one that ignores the hang-up), a dropped connection keeps the
shell for a re-attach until the grace period runs out, and ``shutdown``
reaps everything.  The last class drives the daemon's command catalog
over a real local connection to show that a VS Code window (which has
its own terminal) cannot start a shell.

The Windows branch of ``open`` (no :mod:`pty`) and a shell spawn that
raises cannot be reached on the Linux hosts these tests run on without
replacing module globals, so they are not exercised here.  The browser
half (xterm.js, the "..." menu item, the tab) is covered by
``tests/agents/vscode/test_remote_terminal_tab.py``.

The service runs the developer's ``$SHELL`` as an interactive login
shell, dotfiles included (a ``cd`` in ``~/.zshrc`` moves it out of the
work dir, zsh refuses the first ``exit`` while a job runs and prompts
with ``%``, not ``$``).  The tests therefore point ``SHELL`` at
:func:`hermetic_shell`, a wrapper for ``bash --noprofile --norc``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import pytest

from kiss.core.processes import find_bash
from kiss.server import terminal_tab
from kiss.server.terminal_tab import TerminalService, default_shell
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import open_local_connection

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="terminal tabs need a pty",
)


def hermetic_shell(directory: Path) -> Path:
    """Write and return a ``$SHELL`` for the tests: bash without any dotfile.

    The wrapper execs ``bash --noprofile --norc`` with the arguments the
    service passes (``-l`` on macOS), so the shell starts in the work
    dir whatever the developer's own shell and rc files do, exits on
    the first ``exit``, prompts with ``$`` and prints nothing first
    (macOS's ``/bin/bash`` greets a login shell with its "now zsh"
    notice unless ``BASH_SILENCE_DEPRECATION_WARNING`` is set).

    Args:
        directory: Where to write the wrapper (a per-test temp dir).

    Returns:
        The absolute path of the executable wrapper.
    """
    bash = find_bash()
    assert bash, "the terminal tests need bash on PATH"
    wrapper = directory / "test-shell"
    wrapper.write_text(
        "#!/bin/sh\n"
        "export BASH_SILENCE_DEPRECATION_WARNING=1\n"
        f'exec "{bash}" --noprofile --norc "$@"\n'
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture(autouse=True)
def _bash_without_dotfiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", str(hermetic_shell(tmp_path)))


class ConnPrinter:
    """Collects the per-connection events the service broadcasts."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(dict(event))

    def of_type(self, kind: str, conn_id: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            return [
                e for e in self.events
                if e["type"] == kind and (conn_id is None or e["connId"] == conn_id)
            ]

    def output(self, tab_id: str) -> str:
        return "".join(
            e["data"] for e in self.of_type("terminalData") if e["tab_id"] == tab_id
        )

    def wait_for(self, predicate: Any, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError("timed out waiting for terminal events")


def _wait_until(predicate: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


@pytest.fixture
def service() -> Any:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    yield svc, printer
    svc.shutdown()


def test_default_shell_falls_back_when_shell_env_is_not_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHELL", "/nonexistent/shell-xyz")
    argv = default_shell()
    assert os.access(argv[0], os.X_OK)
    assert argv[0].endswith(("bash", "sh"))
    monkeypatch.setenv("SHELL", argv[0])
    assert default_shell()[0] == argv[0]


def test_shell_runs_in_the_work_dir_and_streams_to_its_connection_only(
    service: Any, tmp_path: Path,
) -> None:
    svc, printer = service
    svc.open("tab-a", "conn-1", str(tmp_path), 100, 30)
    opened = printer.of_type("terminalOpened", "conn-1")
    assert len(opened) == 1
    assert opened[0]["tab_id"] == "tab-a"
    assert opened[0]["attached"] is False
    assert opened[0]["cwd"] == str(tmp_path)
    assert opened[0]["shell"] == os.path.basename(default_shell()[0])

    svc.input("tab-a", "conn-1", "echo marker-$((40+2)); pwd; stty size\n")
    printer.wait_for(
        lambda: "42" in printer.output("tab-a") and "30 100" in printer.output("tab-a"),
    )
    out = printer.output("tab-a")
    assert "marker-42" in out
    assert str(tmp_path) in out
    # Every event names the owning connection and nothing else.
    assert {e["connId"] for e in printer.events} == {"conn-1"}

    # Another connection can neither type into nor resize the shell.
    svc.input("tab-a", "conn-2", "echo intruder\n")
    svc.resize("tab-a", "conn-2", 10, 10)
    # Non-text input is ignored.
    svc.input("tab-a", "conn-1", 123)
    svc.input("tab-a", "conn-1", "")
    svc.resize("tab-a", "conn-1", 120, 40)
    svc.input("tab-a", "conn-1", "stty size\n")
    printer.wait_for(lambda: "40 120" in printer.output("tab-a"))
    assert "intruder" not in printer.output("tab-a")

    svc.input("tab-a", "conn-1", "exit 3\n")
    printer.wait_for(lambda: printer.of_type("terminalExit"))
    assert printer.of_type("terminalExit")[0] == {
        "type": "terminalExit", "tab_id": "tab-a", "code": 3, "connId": "conn-1",
    }
    assert svc.session_count() == 0
    # Commands for a gone tab are no-ops.
    svc.input("tab-a", "conn-1", "echo late\n")
    svc.resize("tab-a", "conn-1", 80, 24)
    svc.close("tab-a", "conn-1")


def test_dimensions_are_clamped_and_defaulted(service: Any, tmp_path: Path) -> None:
    svc, printer = service
    svc.open("tab-b", "conn-1", str(tmp_path), "wide", 5000)
    svc.input("tab-b", "conn-1", "stty size\n")
    printer.wait_for(lambda: "1000 80" in printer.output("tab-b"))
    svc.resize("tab-b", "conn-1", 0, True)
    svc.input("tab-b", "conn-1", "stty size\n")
    printer.wait_for(lambda: "24 1" in printer.output("tab-b"))


def test_missing_work_dir_falls_back_to_home(service: Any, tmp_path: Path) -> None:
    svc, printer = service
    svc.open("tab-c", "conn-1", str(tmp_path / "gone"), 80, 24)
    assert printer.of_type("terminalOpened")[0]["cwd"] == os.path.expanduser("~")


def test_close_hangs_up_and_kills_a_shell_that_ignores_sighup(
    service: Any, tmp_path: Path,
) -> None:
    svc, printer = service
    svc.open("tab-d", "conn-1", str(tmp_path), 80, 24)
    svc.input("tab-d", "conn-1", "trap '' HUP; echo trapped; sleep 300\n")
    printer.wait_for(lambda: "trapped" in printer.output("tab-d"))
    # A stranger cannot close it; the owner can.
    svc.close("tab-d", "conn-2")
    assert svc.session_count() == 1
    started = time.monotonic()
    svc.close("tab-d", "conn-1")
    svc.close("tab-d", "conn-1")  # idempotent
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
    assert time.monotonic() - started < 10
    assert printer.of_type("terminalExit")[0]["code"] != 0
    assert svc.session_count() == 0


def test_close_lets_a_cooperative_shell_exit_at_once(service: Any, tmp_path: Path) -> None:
    svc, printer = service
    svc.open("tab-e", "conn-1", str(tmp_path), 80, 24)
    printer.wait_for(lambda: printer.output("tab-e"))
    started = time.monotonic()
    svc.close("tab-e", "conn-1")
    printer.wait_for(lambda: printer.of_type("terminalExit"))
    assert time.monotonic() - started < terminal_tab._HANGUP_TIMEOUT


def test_shell_exit_is_reported_even_when_a_background_child_keeps_the_pty(
    service: Any, tmp_path: Path,
) -> None:
    """``sleep 300 &`` keeps the pty's slave side open after the shell
    exits, so no EOF arrives on the master: the reader must notice the
    exit through ``waitpid`` instead."""
    svc, printer = service
    svc.open("tab-f", "conn-1", str(tmp_path), 80, 24)
    svc.input("tab-f", "conn-1", "(sleep 300 &) ; echo bg-started; exit 5\n")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
    assert printer.of_type("terminalExit")[0]["code"] == 5
    assert "bg-started" in printer.output("tab-f")


def test_dropped_connection_keeps_the_shell_for_a_reattach(
    service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, printer = service
    svc.open("tab-g", "conn-1", str(tmp_path), 80, 24)
    svc.input("tab-g", "conn-1", "MARK=kept-$$\n")
    printer.wait_for(lambda: "MARK" in printer.output("tab-g"))
    svc.viewer_gone("conn-1")
    svc.viewer_gone("conn-never")  # no shells: nothing happens
    assert svc.session_count() == 1
    # The reconnecting page re-opens with the same tab id from a new
    # connection and gets the SAME shell back (its variables intact).
    svc.open("tab-g", "conn-2", str(tmp_path), 90, 25)
    opened = printer.of_type("terminalOpened", "conn-2")
    assert opened and opened[-1]["attached"] is True
    svc.input("tab-g", "conn-2", "echo $MARK; stty size\n")
    printer.wait_for(
        lambda: "kept-" in printer.output("tab-g") and "25 90" in printer.output("tab-g"),
    )
    assert all(e["connId"] == "conn-2" for e in printer.events[-3:])

    # Without a re-attach the shell is hung up once the grace runs out.
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.3)
    svc.viewer_gone("conn-2")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
    assert svc.session_count() == 0


def test_reattach_before_grace_expiry_cancels_the_hangup(
    service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, printer = service
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.3)
    svc.open("tab-h", "conn-1", str(tmp_path), 80, 24)
    svc.viewer_gone("conn-1")
    svc.open("tab-h", "conn-2", str(tmp_path), 80, 24)
    time.sleep(0.8)
    assert svc.session_count() == 1
    assert not printer.of_type("terminalExit")


def test_a_shell_that_exits_at_once_still_reports_opened_before_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``terminalOpened`` is announced before the reader starts, so even
    ``true`` as the shell yields opened -> exit, never the reverse."""
    true = shutil.which("true")  # /bin/true on Linux, /usr/bin/true on macOS
    assert true
    monkeypatch.setenv("SHELL", true)
    printer = ConnPrinter()
    svc = TerminalService(printer)
    for i in range(10):
        svc.open(f"tab-{i}", "conn-1", str(tmp_path), 80, 24)
    printer.wait_for(lambda: len(printer.of_type("terminalExit")) == 10)
    for i in range(10):
        kinds = [e["type"] for e in printer.events if e["tab_id"] == f"tab-{i}"]
        assert kinds[0] == "terminalOpened"
        assert kinds[-1] == "terminalExit"
    assert svc.session_count() == 0


def test_streaming_background_child_does_not_hide_the_shell_exit(
    service: Any, tmp_path: Path,
) -> None:
    """A child that keeps printing after the shell exited must neither
    keep the tab alive nor starve the exit report."""
    svc, printer = service
    svc.open("tab-k", "conn-1", str(tmp_path), 80, 24)
    svc.input(
        "tab-k", "conn-1",
        "(trap '' HUP; while :; do echo tick; sleep 0.05; done) & exit 5\n",
    )
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
    assert printer.of_type("terminalExit")[0]["code"] == 5
    assert svc.session_count() == 0
    # Stop the stray printer loop this test left behind.
    os.system("pkill -f 'sleep 0.05' >/dev/null 2>&1")


def test_an_earlier_disconnect_timer_cannot_expire_a_later_grace_period(
    service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, printer = service
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 1.0)
    svc.open("tab-l", "conn-1", str(tmp_path), 80, 24)
    svc.viewer_gone("conn-1")          # t = 0: first grace period
    time.sleep(0.6)
    svc.open("tab-l", "conn-2", str(tmp_path), 80, 24)  # re-attached
    time.sleep(0.1)
    svc.viewer_gone("conn-2")          # t = 0.7: second grace period
    time.sleep(0.6)                    # t = 1.3: the FIRST timer has fired
    assert svc.session_count() == 1
    assert not printer.of_type("terminalExit")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="inspects /proc")
def test_a_new_shell_does_not_inherit_older_pty_masters(service: Any, tmp_path: Path) -> None:
    svc, printer = service
    svc.open("tab-m", "conn-1", str(tmp_path), 80, 24)
    svc.open("tab-n", "conn-1", str(tmp_path), 80, 24)
    svc.input("tab-n", "conn-1", "echo masters=$(ls -l /proc/$$/fd | grep -c ptmx)\n")
    printer.wait_for(lambda: "masters=" in printer.output("tab-n").split("echo", 1)[-1])
    assert "masters=0" in printer.output("tab-n")


def test_shutdown_reaps_every_shell(tmp_path: Path) -> None:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    svc.open("tab-i", "conn-1", str(tmp_path), 80, 24)
    svc.open("tab-j", "conn-1", str(tmp_path), 80, 24)
    svc.input("tab-j", "conn-1", "trap '' HUP; sleep 300\n")
    printer.wait_for(lambda: len(printer.of_type("terminalOpened")) == 2)
    time.sleep(0.3)
    svc.shutdown()
    assert svc.session_count() == 0
    assert len(printer.of_type("terminalExit")) == 2
    svc.shutdown()  # nothing left: returns at once


class TestTerminalCommandsFromAVsCodeWindow(unittest.TestCase):
    """The daemon drops terminal commands that arrive over the local
    (VS Code) connection: that window has its own integrated terminal."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
            work_dir=self.tmp.name,
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=30)

    def tearDown(self) -> None:
        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)],
            timeout=30,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def test_local_terminal_commands_are_dropped(self) -> None:
        async def _talk() -> list[str]:
            reader, writer = await open_local_connection(self.server)
            seen: list[str] = []
            try:
                for cmd in (
                    {"type": "terminalOpen", "tab_id": "t1", "cols": 80, "rows": 24},
                    {"type": "terminalInput", "tab_id": "t1", "data": "echo hi\n"},
                    {"type": "terminalResize", "tab_id": "t1", "cols": 10, "rows": 10},
                    {"type": "terminalClose", "tab_id": "t1"},
                    # A reply-bearing command proves the four above were
                    # processed (in order) before we look.
                    {"type": "bogusCommand", "tabId": "t1"},
                ):
                    writer.write(json.dumps(cmd).encode() + b"\n")
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=10)
                    if not line:
                        break
                    event = json.loads(line)
                    seen.append(event.get("type", ""))
                    if event.get("type") == "error":
                        break
            finally:
                writer.close()
                await writer.wait_closed()
            return seen

        seen = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(timeout=20)
        self.assertIn("error", seen)
        self.assertEqual(self.server._vscode_server.terminals.session_count(), 0)
        self.assertFalse({"terminalOpened", "terminalError", "terminalData"} & set(seen))
