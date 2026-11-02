"""A timed-out ssh forward takes its ``ProxyCommand`` child down with it (S12 F4).

``SshTunnel`` ran ``ssh`` as a plain child and killed that process
alone when the forward did not come up in time.  The user's ssh
configuration applies to the forward, and a ``ProxyCommand`` is a
real child of ssh: a stalled one survived the kill, kept ssh's
stderr (the tunnel's output pipe) open and so kept the tunnel's
output-reader thread blocked, with no handle left to reach either.
ssh now runs in a process group of its own and the whole group is
killed.
"""

from __future__ import annotations

import inspect
import os
import shutil
import signal
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.core.processes import pid_alive
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import posix_only

LOGIN = "synthetic@unused.invalid"
READER = f"ssh-tunnel-output {LOGIN}"
# A ProxyCommand that writes its pid and then never speaks the ssh
# protocol: the forward can never come up and ssh never exits on its own
# before the tunnel's deadline.
PROXY = (
    "import os, pathlib, sys, time\n"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "time.sleep(120)\n"
)
# A ProxyCommand that leaves a helper behind and exits at once: the
# helper writes its pid, then closes the connection ends (so ssh sees
# EOF and exits on its own) but keeps the inherited stderr, the
# tunnel's output pipe.
EXITING_PROXY = (
    "import os, pathlib, sys, time\n"
    "if os.fork() == 0:\n"
    "    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "    os.close(0)\n"
    "    os.close(1)\n"
    "    time.sleep(120)\n"
    "os._exit(0)\n"
)


def _gone(pid: int) -> bool:
    """Whether *pid* is dead: no such process, or a zombie its new parent has yet to reap."""
    if not pid_alive(pid):
        return True
    try:
        return "State:\tZ" in Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return False


def _reader_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == READER]


@posix_only("a ProxyCommand child process and a signal to its process group")
@pytest.mark.skipif(shutil.which("ssh") is None, reason="no ssh on this host")
def test_tunnel_timeout_kills_the_proxy_command_and_frees_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    pid_file = tmp_path / "proxy.pid"
    proxy = tmp_path / "proxy.py"
    proxy.write_text(PROXY, encoding="utf-8")
    config = tmp_path / "ssh_config"
    config.write_text(
        f"Host *\n    ProxyCommand {sys.executable} {proxy} {pid_file}\n", encoding="utf-8",
    )
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(config))
    tunnel = mach.SshTunnel(LOGIN, 12345)
    child = 0
    try:
        with pytest.raises(mach.MachineError, match="did not open the port forward in time"):
            tunnel.open(timeout=2.0)
        assert pid_file.exists(), "the proxy never started"
        child = int(pid_file.read_text(encoding="utf-8"))
        tunnel.close()
        assert tunnel.process is None
        deadline = time.monotonic() + 10.0
        while not _gone(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(child), "the ProxyCommand outlived the killed ssh"
        for thread in _reader_threads():
            thread.join(10.0)
        assert _reader_threads() == []
    finally:
        tunnel.close()
        if child and not _gone(child):
            os.kill(child, signal.SIGKILL)
        for thread in _reader_threads():
            thread.join(10.0)


@posix_only("a ProxyCommand child process and a signal to its process group")
@pytest.mark.skipif(shutil.which("ssh") is None, reason="no ssh on this host")
def test_a_proxy_helper_outliving_ssh_is_killed_and_frees_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ssh exits by itself; the helper still holding its output pipe goes down with the group."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    pid_file = tmp_path / "helper.pid"
    proxy = tmp_path / "proxy.py"
    proxy.write_text(EXITING_PROXY, encoding="utf-8")
    config = tmp_path / "ssh_config"
    config.write_text(
        f"Host *\n    ProxyCommand {sys.executable} {proxy} {pid_file}\n", encoding="utf-8",
    )
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(config))
    tunnel = mach.SshTunnel(LOGIN, 12345)
    child = 0
    try:
        with pytest.raises(mach.MachineError, match="cannot forward its daemon port"):
            tunnel.open(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "the helper never started"
        child = int(pid_file.read_text(encoding="utf-8"))
        assert tunnel.process is None
        deadline = time.monotonic() + 10.0
        while not _gone(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(child), "the helper outlived the exited ssh"
        for thread in _reader_threads():
            thread.join(10.0)
        assert _reader_threads() == []
    finally:
        tunnel.close()
        if child and not _gone(child):
            os.kill(child, signal.SIGKILL)
        for thread in _reader_threads():
            thread.join(10.0)


def _wait_for_pid(pid_file: Path) -> int:
    """Return the pid a ProxyCommand writes to *pid_file*, once it has (within 10 s)."""
    deadline = time.monotonic() + 10.0
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "the proxy never started"
    return int(pid_file.read_text(encoding="utf-8"))


@posix_only("a ProxyCommand child process and a signal to its process group")
@pytest.mark.skipif(shutil.which("ssh") is None, reason="no ssh on this host")
def test_a_stop_before_the_output_reader_starts_leaves_the_tunnel_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop between the reader thread's creation and its start: close and reopen still work.

    The tunnel's ``open`` runs on the task thread (the ``/machine``
    SEA's tools), where a Stop injects ``KeyboardInterrupt`` at any
    check point.  Landing after ``_forward`` published its output
    reader and before it started it, the Stop left a thread that can
    never be joined, and ``_kill`` joined it anyway: every later
    ``close`` and ``open`` of the tunnel failed with ``RuntimeError``,
    ssh's pipe staying open and the tunnel never reopening.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    pid_file = tmp_path / "proxy.pid"
    proxy = tmp_path / "proxy.py"
    proxy.write_text(PROXY, encoding="utf-8")
    config = tmp_path / "ssh_config"
    config.write_text(
        f"Host *\n    ProxyCommand {sys.executable} {proxy} {pid_file}\n", encoding="utf-8",
    )
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(config))
    tunnel = mach.SshTunnel(LOGIN, 12345)
    source, first_line = inspect.getsourcelines(mach.SshTunnel._forward)
    start_line = first_line + next(
        i for i, text in enumerate(source) if text.strip() == "self._reader.start()"
    )
    paused, release = threading.Event(), threading.Event()
    outcome: list[str] = []

    def trace(frame, event, arg):  # type: ignore[no-untyped-def]
        if (
            frame.f_code is mach.SshTunnel._forward.__code__
            and event == "line" and frame.f_lineno == start_line
        ):
            paused.set()
            assert release.wait(10.0)
        return trace

    def opening() -> None:
        sys.settrace(trace)
        try:
            tunnel.open(timeout=10.0)
        except BaseException as exc:  # the injected Stop
            outcome.append(type(exc).__name__)
        finally:
            sys.settrace(None)

    opener = threading.Thread(target=opening, name="tunnel-opener")
    opener.start()
    children: list[int] = []
    try:
        assert paused.wait(10.0), "the forward never reached its reader's start"
        assert opener.ident is not None
        assert inject_async_exception(opener.ident, KeyboardInterrupt) == 1
        release.set()
        opener.join(10.0)
        assert outcome == ["KeyboardInterrupt"]
        children.append(_wait_for_pid(pid_file))
        assert tunnel.process is not None and tunnel._reader is not None
        assert tunnel._reader.ident is None, "the Stop landed after the reader started"
        tunnel.close()
        assert tunnel.process is None
        assert _reader_threads() == []
        pid_file.unlink()
        # A fresh ssh starts (its ProxyCommand runs anew) and the forward
        # ends the usual way: the port never comes up, so ``open`` times
        # out and takes the group down.
        with pytest.raises(mach.MachineError, match="did not open the port forward in time"):
            tunnel.open(timeout=2.0)
        children.append(_wait_for_pid(pid_file))
        assert tunnel.process is None
        tunnel.close()
        for child in children:
            deadline = time.monotonic() + 10.0
            while not _gone(child) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert _gone(child), "a ProxyCommand outlived its killed ssh"
        for thread in _reader_threads():
            thread.join(10.0)
        assert _reader_threads() == []
    finally:
        release.set()
        opener.join(10.0)
        with suppress(RuntimeError):  # the failure under test must not leak the children
            tunnel.close()
        for child in children:
            if not _gone(child):
                os.kill(child, signal.SIGKILL)
        for thread in _reader_threads():
            thread.join(10.0)
