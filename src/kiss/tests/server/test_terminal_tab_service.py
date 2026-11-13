# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the terminal-tab service behind the remote webapp's "Terminal".

:class:`kiss.server.terminal_tab.TerminalService` runs a real shell on a
real pty here: output streams as ``terminalData`` events to the owning
connection only, the pty follows ``resize``, the shell's exit code
arrives as ``terminalExit``, a tab that was closed hangs its shell up
(and kills one that ignores the hang-up, its foreground job with it), a
dropped connection keeps the shell for a re-attach until the grace period
runs out, and ``shutdown`` reaps everything.

The Windows branch of ``open`` (no :mod:`pty`) and a shell spawn that
raises cannot be reached on the Linux hosts these tests run on without
replacing module globals, so they are not exercised here.  The browser
half (xterm.js, the "..." menu item, the tab) is covered by
``tests/webapp/test_remote_terminal_tab.py``. The writer's defensive
``BlockingIOError``/``OSError`` handlers cover a kernel readiness change
or an invalid descriptor; the public API has one writer and never closes
its descriptor before that writer, so those failures cannot be forced
here without replacing an OS call or violating descriptor ownership.

The service runs the developer's ``$SHELL`` as an interactive login
shell, dotfiles included (a ``cd`` in ``~/.zshrc`` moves it out of the
work dir, zsh refuses the first ``exit`` while a job runs and prompts
with ``%``, not ``$``).  The tests therefore point ``SHELL`` at
:func:`hermetic_shell`, a wrapper for ``bash --noprofile --norc``.
"""

from __future__ import annotations

import os
import selectors
import shlex
import shutil
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from kiss.core.processes import find_bash
from kiss.server import terminal_tab
from kiss.server.terminal_tab import TerminalService, default_shell
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.server.pause import Pause

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="terminal tabs need a pty",
)


def session_count(svc: TerminalService) -> int:
    """Return how many shells *svc* is running."""
    with svc._lock:
        return len(svc._sessions)


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

    def wait_for(self, predicate: Any, timeout: float = 30.0) -> None:
        """Poll *predicate* until it holds; fail with every event seen so far.

        The default bound covers a shell that is only starting: under a
        full-suite load the hermetic bash has taken more than 10 s from
        spawn to its first prompt (the pty echo of typed input arrives
        long before).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        with self.lock:
            seen = [(e["type"], e.get("tab_id"), e.get("data", "")) for e in self.events]
        raise AssertionError(f"timed out waiting for terminal events; got {seen!r}")


def _wait_until(predicate: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _alive(pid: int) -> bool:
    """Return whether *pid* still exists (an orphan killed is reaped by init at once)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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
    assert session_count(svc) == 0
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
    svc.input("tab-c", "conn-1", 'echo "in=[$(pwd -P)]"\n')
    home = f"in=[{os.path.realpath(os.path.expanduser('~'))}]"
    printer.wait_for(lambda: home in printer.output("tab-c"))


def test_close_hangs_up_and_kills_a_shell_that_ignores_sighup(
    service: Any, tmp_path: Path,
) -> None:
    """The kill reaches the shell's foreground job too: it inherited the
    ignored SIGHUP and runs in a process group of its own (job control)."""
    svc, printer = service
    pid_file = tmp_path / "job.pid"
    job = shlex.quote(f"echo $$ > {shlex.quote(str(pid_file))}; exec sleep 300")
    svc.open("tab-d", "conn-1", str(tmp_path), 80, 24)
    svc.input("tab-d", "conn-1", f"trap '' HUP; echo trapped; /bin/sh -c {job}\n")
    printer.wait_for(lambda: "trapped" in printer.output("tab-d"))
    printer.wait_for(lambda: pid_file.exists() and pid_file.read_text().strip().isdigit())
    job_pid = int(pid_file.read_text())
    try:
        # A stranger cannot close it; the owner can.
        svc.close("tab-d", "conn-2")
        assert session_count(svc) == 1
        started = time.monotonic()
        svc.close("tab-d", "conn-1")
        svc.close("tab-d", "conn-1")  # idempotent
        printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
        assert time.monotonic() - started < 10
        assert printer.of_type("terminalExit")[0]["code"] != 0
        assert session_count(svc) == 0
        _wait_until(lambda: not _alive(job_pid))
    finally:
        if _alive(job_pid):
            os.kill(job_pid, signal.SIGKILL)


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="requires enforced RLIMIT_NPROC")
@pytest.mark.parametrize("drop", ["close", "viewer_gone"])
def test_a_shell_whose_timer_thread_cannot_start_is_not_stranded(
    service: Any, tmp_path: Path, drop: str
) -> None:
    """``close`` (its SIGKILL timer) and a dropped connection (its grace
    timer) each start a thread; when none can start, the shell is torn
    down at once rather than left registered with nothing to end it."""
    svc, printer = service
    ready = tmp_path / "hup-ignored"
    svc.open("no-thread", "c1", str(tmp_path), 80, 24)
    svc.input("no-thread", "c1", f"trap '' HUP; echo > {shlex.quote(str(ready))}\n")
    printer.wait_for(ready.exists)
    with nproc_limit_lowered_to_one():
        if drop == "close":
            svc.close("no-thread", "c1")
        else:
            svc.viewer_gone("c1")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
    assert session_count(svc) == 0
    svc.close("no-thread", "c1")  # nothing left: harmless


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
    assert session_count(svc) == 1
    # The reconnecting page re-opens with the same tab id from a new
    # connection and gets the SAME shell back (its variables intact).
    svc.open("tab-g", "conn-2", str(tmp_path), 90, 25)
    opened = printer.of_type("terminalOpened", "conn-2")
    assert opened and opened[-1]["attached"] is True
    svc.input("tab-g", "conn-2", "echo $MARK; stty size\n")
    # The MARK wait above is met by the pty's echo of the typed line,
    # which precedes the shell's own start: this wait also covers a
    # login bash starting under a loaded machine.
    printer.wait_for(
        lambda: "kept-" in printer.output("tab-g") and "25 90" in printer.output("tab-g"),
        timeout=20,
    )
    assert all(e["connId"] == "conn-2" for e in printer.events[-3:])

    # Without a re-attach the shell is hung up once the grace runs out.
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.3)
    svc.viewer_gone("conn-2")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
    assert session_count(svc) == 0


def test_resize_right_after_open_wins_over_the_opening_size(
    service: Any, tmp_path: Path,
) -> None:
    """A re-attach (or resize) that follows ``open`` at once must leave the
    pty at ITS size once the shell is up.  The opening size is applied on
    the master by the parent, in program order with the later ioctls; a
    forked child applying it before its exec could run after them on a
    loaded machine and the shell would start at the opening size."""
    svc, printer = service
    svc.open("tab-i", "conn-1", str(tmp_path), 80, 24)
    svc.viewer_gone("conn-1")
    svc.open("tab-i", "conn-2", str(tmp_path), 90, 25)
    svc.resize("tab-i", "conn-2", 100, 30)
    svc.input("tab-i", "conn-2", "stty size\n")
    printer.wait_for(lambda: "30 100" in printer.output("tab-i"), timeout=20)
    assert "24 80" not in printer.output("tab-i")


def test_reattach_before_grace_expiry_cancels_the_hangup(
    service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, printer = service
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.3)
    svc.open("tab-h", "conn-1", str(tmp_path), 80, 24)
    svc.viewer_gone("conn-1")
    svc.open("tab-h", "conn-2", str(tmp_path), 80, 24)
    time.sleep(0.8)
    assert session_count(svc) == 1
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
    assert session_count(svc) == 0


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
    assert session_count(svc) == 0
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
    assert session_count(svc) == 1
    assert not printer.of_type("terminalExit")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="inspects /proc")
def test_a_new_shell_does_not_inherit_older_pty_masters(service: Any, tmp_path: Path) -> None:
    svc, printer = service
    svc.open("tab-m", "conn-1", str(tmp_path), 80, 24)
    svc.open("tab-n", "conn-1", str(tmp_path), 80, 24)
    # The ``masters=`` marker is built from a shell variable so the echoed
    # command line never contains it literally; only the command's output
    # does, so the wait cannot be satisfied by the echo alone.
    svc.input("tab-n", "conn-1", "k=masters; echo $k=$(ls -l /proc/$$/fd | grep -c ptmx)\n")
    printer.wait_for(lambda: "masters=" in printer.output("tab-n"))
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
    assert session_count(svc) == 0
    assert len(printer.of_type("terminalExit")) == 2
    svc.shutdown()  # nothing left: returns at once


def test_shutdown_starts_no_shell_for_an_open_behind_it(tmp_path: Path) -> None:
    """A ``terminalOpen`` that arrives while the daemon stops gets no shell.

    While ``shutdown`` waited for a shell that ignores the hang-up, a
    tab whose shell had already left could open a new one (a page
    reloading): nothing hung that shell up, and ``shutdown`` returned
    at its deadline with it still running.
    """
    printer = ConnPrinter()
    svc = TerminalService(printer)
    svc.open("quick", "conn-1", str(tmp_path), 80, 24)
    svc.open("stubborn", "conn-1", str(tmp_path), 80, 24)
    svc.input("stubborn", "conn-1", "trap '' HUP; sleep 300\n")
    printer.wait_for(lambda: len(printer.of_type("terminalOpened")) == 2)
    time.sleep(0.3)
    stopping = threading.Thread(target=svc.shutdown)
    stopping.start()
    try:
        # The first shell is gone; the other one keeps ``shutdown`` waiting.
        printer.wait_for(lambda: printer.of_type("terminalExit"))
        svc.open("quick", "conn-2", str(tmp_path), 80, 24)
        stopping.join(10)
        assert not stopping.is_alive()
        assert session_count(svc) == 0, "a shell opened during the shutdown outlived it"
        assert not printer.of_type("terminalOpened", "conn-2")
        assert not printer.of_type("terminalError", "conn-2")
    finally:
        svc.close("quick")  # the shell an unfixed service started
        _wait_until(lambda: session_count(svc) == 0)


def test_close_interrupts_a_full_pty_write(service: Any, tmp_path: Path) -> None:
    """A foreground child retaining the slave cannot strand the input writer."""
    svc, printer = service
    pid_file = tmp_path / "foreground.pid"
    script = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    svc.open("blocked-write", "c1", str(tmp_path), 80, 24)
    with svc._lock:
        session = svc._sessions["blocked-write"]
    writer = next(t for t in threading.enumerate() if t.name == "terminal-write-blocked-")
    try:
        python = shlex.quote(sys.executable)
        svc.input(
            "blocked-write", "c1",
            f"stty raw -echo; trap '' HUP; {python} -c {shlex.quote(script)}\n",
        )
        printer.wait_for(lambda: pid_file.exists() and pid_file.read_text().isdigit())
        svc.input("blocked-write", "c1", "x" * (1024 * 1024))
        _wait_until(session.writes.empty)
        svc.close("blocked-write", "c1")
        printer.wait_for(lambda: printer.of_type("terminalExit"))
        writer.join(5)
        assert not writer.is_alive(), "PTY input writer survived terminalExit"
        with pytest.raises(OSError):
            os.fstat(session.fd)
    finally:
        if pid_file.exists() and pid_file.read_text().isdigit():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        writer.join(5)


def _after_reap(values: dict[str, Any], session: Any) -> bool:
    return session.reaped and threading.current_thread().name == "terminal-read-reaped"


def test_reconnect_never_attaches_to_a_reaped_shell(service: Any, tmp_path: Path) -> None:
    """Hold the real output drain after waitpid; reopening must refuse the dead shell."""
    svc, printer = service
    svc.open("reaped", "c1", str(tmp_path), 80, 24)
    with svc._lock:
        session = svc._sessions["reaped"]
    after_reap = partial(_after_reap, session=session)
    with Pause(selectors.DefaultSelector.select.__code__, after_reap) as gate:
        svc.input("reaped", "c1", "exit\n")
        assert gate.paused.wait(15), "reader did not reach its post-reap drain"
        svc.open("reaped", "c2", str(tmp_path), 80, 24)
        assert not printer.of_type("terminalOpened", "c2")
        assert printer.of_type("terminalError", "c2")
    printer.wait_for(lambda: printer.of_type("terminalExit", "c1"))
    svc.open("reaped", "c2", str(tmp_path), 80, 24)
    assert printer.of_type("terminalOpened", "c2")[0]["attached"] is False


def _starve_worker(values: dict[str, Any], name: str, limits: ExitStack) -> bool:
    if values["self"].name == name:
        limits.enter_context(nproc_limit_lowered_to_one())
    return False


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="requires enforced RLIMIT_NPROC")
@pytest.mark.parametrize("worker", ["read", "write"])
def test_worker_start_failure_leaves_no_terminal(
    tmp_path: Path, worker: str,
) -> None:
    """Starve each actual worker start after fork, without replacing any product code."""
    printer = ConnPrinter()
    svc = TerminalService(printer)
    try:
        with ExitStack() as limits, Pause(
            threading.Thread.start.__code__,
            partial(_starve_worker, name=f"terminal-{worker}-starved", limits=limits),
        ):
            svc.open("starved", "c1", str(tmp_path), 80, 24)
        assert session_count(svc) == 0
        assert not printer.of_type("terminalOpened")
        assert printer.of_type("terminalError")
        _wait_until(lambda: not any(t.name.endswith("-starved") for t in threading.enumerate()))
        svc.open("starved", "c2", str(tmp_path), 80, 24)
        assert printer.of_type("terminalOpened", "c2")[0]["attached"] is False
    finally:
        svc.shutdown()
        # Also reap the orphaned shell on the unfixed implementation.
        with svc._lock:
            abandoned = svc._sessions.pop("starved", None)
        if abandoned is not None:
            os.kill(abandoned.pid, signal.SIGKILL)
            os.waitpid(abandoned.pid, 0)
            writers = [t for t in threading.enumerate() if t.name == "terminal-write-starved"]
            if writers:
                abandoned.writes.put(None)
                for thread in writers:
                    thread.join(5)
            else:
                os.close(abandoned.fd)


@contextmanager
def nofile_limit_lowered_to_zero() -> Iterator[None]:
    """Lower the soft ``RLIMIT_NOFILE`` to 0 for the block: every new descriptor fails (EMFILE)."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (0, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def _starve_reader_selector(
    values: dict[str, Any], limits: ExitStack, starved: threading.Event,
) -> bool:
    if threading.current_thread().name.startswith("terminal-read-"):
        limits.enter_context(nofile_limit_lowered_to_zero())
        starved.set()
    return False


def test_reader_selector_failure_releases_the_session(tmp_path: Path) -> None:
    """A reader denied its selector (EMFILE after the fork) reaps the shell and frees the tab.

    The descriptor limit is lowered on the reader thread itself, at the
    entry of the real selector constructor, after the shell was forked
    and the session registered: the reader must then kill and reap the
    shell, report the failure and the exit, release the writer, and
    leave the tab free for a fresh shell.
    """
    printer = ConnPrinter()
    svc = TerminalService(printer)
    starved = threading.Event()
    try:
        with ExitStack() as limits, Pause(
            selectors.DefaultSelector.__init__.__code__,
            partial(_starve_reader_selector, limits=limits, starved=starved),
        ):
            svc.open("emfile", "c1", str(tmp_path), 80, 24)
            assert starved.wait(10), "the reader never reached its selector"
            printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=15)
        assert session_count(svc) == 0
        assert "Too many open files" in printer.of_type("terminalError", "c1")[0]["text"]
        _wait_until(lambda: not any(t.name.endswith("-emfile") for t in threading.enumerate()))
        svc.open("emfile", "c2", str(tmp_path), 80, 24)
        assert printer.of_type("terminalOpened", "c2")[0]["attached"] is False
    finally:
        svc.shutdown()
        # Also reap the stranded shell and free its writer on the unfixed
        # implementation.
        with svc._lock:
            stranded = svc._sessions.pop("emfile", None)
        if stranded is not None:
            os.kill(stranded.pid, signal.SIGKILL)
            os.waitpid(stranded.pid, 0)
            stranded.ended.set()
            stranded.writes.put(None)
            for thread in threading.enumerate():
                if thread.name == "terminal-write-emfile":
                    thread.join(5)


def test_terminal_exit_flushes_incomplete_utf8(service: Any, tmp_path: Path) -> None:
    """A truncated final character is replaced before terminalExit, never silently lost."""
    svc, printer = service
    svc.open("utf8", "c1", str(tmp_path), 80, 24)
    svc.input("utf8", "c1", "exec printf '\\342'\n")
    printer.wait_for(lambda: printer.of_type("terminalExit"))
    assert printer.output("utf8").endswith("\ufffd")


def test_malformed_surrogate_paste_still_reaches_shell(service: Any, tmp_path: Path) -> None:
    """One invalid JS code unit does not discard the rest of a paste."""
    svc, printer = service
    svc.open("surrogate", "c1", str(tmp_path), 80, 24)
    svc.input("surrogate", "c1", "v='x\ud800y'; printf '\\n%s\\n' \"$v\"; exit\n")
    printer.wait_for(lambda: printer.of_type("terminalExit"))
    assert "\r\nx?y\r\n" in printer.output("surrogate")
