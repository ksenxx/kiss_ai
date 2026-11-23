# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""One registered machine reached over a single multiplexed ssh connection.

A sync talks to a machine six to eight times (fingerprints, history
manifest, extract, merge, pack, merge).  Each plain ``ssh`` pays a full
handshake -- a second or so to a cloud machine -- so :class:`SshPeer`
opens one master connection (``ControlMaster``) at the start and every
later command rides on it (``ControlPath``) in a few milliseconds; the
master is closed at the end.  Remote work needs nothing but ``python3``:
scripts travel as text inside the command line, as ``kiss.scripts.sync_db``
does.

Every ssh is started by :meth:`SshPeer.call`, which ends it when the run
is stopped or when it stalls: ssh's own deadlines end a connection that
cannot be made or has died, not a login or a command that hangs on a live
one.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from kiss.agents.sorcar.machines import ssh_command
from kiss.core.processes import kill_process_tree, start_process_group
from kiss.core.tool_interrupt import call_without_injection
from kiss.scripts.sync_db import Location, Runner, SyncError

CONTROL_PERSIST = "600"
"""Seconds an idle master lingers before closing itself, should :meth:`SshPeer.close` never run."""

COMMON_OPTIONS = ["StrictHostKeyChecking=accept-new", "ConnectTimeout=10", "LogLevel=ERROR"]
"""ssh options every command of a peer uses (the same ones :func:`ssh_command` sets)."""

POLL_SECONDS = 0.1
"""Seconds between two looks at a run under way: whether it was stopped, how far an ssh has got."""

CONNECT_SECONDS = 60.0
"""Seconds :meth:`SshPeer.open` gives the login.

ssh's ``ConnectTimeout`` bounds the connection and the key exchange, not
what follows them (the authentication, a ``ProxyCommand`` that hangs).
"""

COMMAND_SECONDS = 3600.0
"""Seconds a command on a peer may go without reading or writing a byte.

Not a bound on the command: a first sync ships a whole history, gigabytes
over whatever link there is, and may run for hours while the bytes move.
A peer merging what it received works in silence, for minutes on such a
history, so the hour is for a command that is stuck, not for a slow one.
"""

CLOSE_SECONDS = 10.0
"""Seconds the master is given to answer the ``-O exit`` of :meth:`SshPeer.close`."""

REAP_SECONDS = 2.0
"""Seconds a killed ssh is given to exit before its caller stops waiting for it."""


class PeerError(Exception):
    """A machine could not be reached, or a command on it failed; the message names it."""


def bytes_moved(*streams: Any) -> int:
    """Return how far ssh has read and written the files among *streams*.

    ssh inherits the files themselves, so their offsets are its own: how
    much of its input it has sent and of its output it has received.
    ``subprocess.DEVNULL``, a number like every special value of
    ``subprocess``, has none.
    """
    return sum(
        os.lseek(stream.fileno(), 0, os.SEEK_CUR)
        for stream in streams
        if not isinstance(stream, int)
    )


def ended(proc: subprocess.Popen[bytes]) -> bool:
    """Wait up to :data:`POLL_SECONDS` for *proc* to exit; whether it has.

    Held against an exception injected into this thread: raised inside
    the wait, it could leave the lock ``Popen`` reaps under held, and
    the child unreaped.
    """
    with contextlib.suppress(subprocess.TimeoutExpired):
        call_without_injection(proc.wait, POLL_SECONDS)
    return proc.returncode is not None


def control_directory() -> Path:
    """Create and return a private directory for control sockets, with a short path.

    A Unix socket path is limited to about a hundred bytes, which the
    per-user temporary directory of macOS alone nearly uses up, so the
    directory goes under ``/tmp`` where there is one; ``mkdtemp`` makes
    it readable by this user only.
    """
    parent = "/tmp" if os.path.isdir("/tmp") else None  # noqa: S108 - a private mkdtemp under it
    return Path(tempfile.mkdtemp(prefix="kiss-sync-", dir=parent))


class SshPeer:
    """A machine registered with ``/machine add`` as ``user@host[:port]``.

    Attributes:
        name: The machine's registered name.
        ssh: Its login, ``user@host[:port]``.
        python: The remote interpreter.
        cancel: Set to stop the run: the ssh commands in flight are killed
            and none is started after it.
    """

    def __init__(
        self,
        name: str,
        ssh: str,
        control_dir: Path,
        python: str = "python3",
        cancel: threading.Event | None = None,
    ) -> None:
        """Describe a peer; nothing is connected until :meth:`open`.

        Args:
            name: The machine's registered name.
            ssh: Its login, ``user@host[:port]``.
            control_dir: A private directory for the control socket.
            python: The remote interpreter running shipped scripts.
            cancel: The event that stops the run the peer is part of; one
                of its own, never set, when None.
        """
        self.name = name
        self.ssh = ssh
        self.python = python
        self.cancel = cancel or threading.Event()
        parts = urlsplit("ssh://" + ssh)
        self.port = str(parts.port) if parts.port else None
        self.login = f"{parts.username}@{parts.hostname}" if parts.username else str(parts.hostname)
        self.socket = control_dir / hashlib.sha1(ssh.encode("utf-8")).hexdigest()[:10]  # noqa: S324

    def open(self) -> None:
        """Start the master connection (one handshake); later commands reuse it.

        Raises:
            PeerError: When the machine cannot be logged into.
        """
        options = [
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPath={self.socket}",
            "-o",
            f"ControlPersist={CONTROL_PERSIST}",
            "-N",
            "-f",
        ]
        code, errors = self.call(ssh_command(self.ssh, *options), self.cancel, CONNECT_SECONDS)
        if code != 0:
            detail = errors.decode("utf-8", "replace").strip() or f"exit {code}"
            raise PeerError(f"{self.name} ({self.ssh}): cannot connect: {detail}")

    def close(self) -> None:
        """Close the master connection; a peer that never opened is left alone.

        The close also follows a Stop, so :attr:`cancel` does not end it;
        :data:`CLOSE_SECONDS` do.  A master that does not answer by then
        is left to close itself (:data:`CONTROL_PERSIST`).
        """
        if not self.socket.exists():
            return
        with contextlib.suppress(PeerError):
            self.call(self.command("", "-O", "exit"), threading.Event(), CLOSE_SECONDS)

    def call(
        self,
        command: list[str],
        cancel: threading.Event,
        timeout: float,
        stdin: Any = subprocess.DEVNULL,
        stdout: Any = subprocess.DEVNULL,
    ) -> tuple[int, bytes]:
        """Run an ssh *command* to its end, unless it is stopped or stalls first.

        ``subprocess.run`` waited for ssh with no bound, in a wait no Stop
        reaches: on 2026-11-22 an ssh that never returned kept its task
        from being stopped and the daemon from restarting.  Here ssh runs
        in a process group of its own and is waited for in slices of
        :data:`POLL_SECONDS`; once *cancel* is set, or once ssh has moved
        no byte of *stdin* and *stdout* for *timeout* seconds, the group
        is killed.  stderr goes to a file, not a pipe, so the wait is for
        ssh alone: a pipe stays open for as long as anything ssh started
        holds it (the master ``-f`` leaves behind, a ``ProxyCommand``),
        and a read to its end waited for that as well.

        Args:
            command: The ssh command line.
            cancel: Set to end the command; :attr:`cancel` for all but the close.
            timeout: Seconds ssh may go without moving a byte.
            stdin: What ssh reads: an open file, or ``subprocess.DEVNULL``.
            stdout: What ssh writes to: an open file, or ``subprocess.DEVNULL``.

        Returns:
            ssh's exit status and what it wrote to stderr.

        Raises:
            PeerError: When the command was stopped, or was killed for stalling.
            OSError: When ssh could not be started.
        """
        if cancel.is_set():
            raise PeerError(f"{self.name} ({self.ssh}): stopped")
        # Made before the spawn, so that the handler below holds the child
        # whatever is raised from there on (see ``start_process_group``).
        proc: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
        with tempfile.TemporaryFile() as errors:
            try:
                start_process_group(proc, command, stdin=stdin, stdout=stdout, stderr=errors)
                moved, moved_at = 0, time.monotonic()
                while not ended(proc):
                    if cancel.is_set():
                        raise PeerError(f"{self.name} ({self.ssh}): stopped")
                    progress = bytes_moved(stdin, stdout)
                    if progress != moved:
                        moved, moved_at = progress, time.monotonic()
                    elif time.monotonic() - moved_at >= timeout:
                        raise PeerError(
                            f"{self.name} ({self.ssh}): ssh moved no data for {timeout:g} s "
                            "and was killed"
                        )
            except BaseException:
                if getattr(proc, "pid", None) is not None and proc.returncode is None:
                    kill_process_tree(proc)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=REAP_SECONDS)
                raise
            errors.seek(0)
            return proc.returncode, errors.read()

    @property
    def ssh_options(self) -> list[str]:
        """The ``-o`` options that make a command ride on the master connection."""
        return [f"ControlPath={self.socket}", *COMMON_OPTIONS]

    def command(self, remote: str, *extra: str) -> list[str]:
        """Return the ``ssh`` command line running *remote* on the peer over the master connection.

        Args:
            remote: The remote shell command (empty for none).
            extra: Further ``ssh`` arguments, placed before the destination.
        """
        options = ["-o", f"ControlPath={self.socket}", *extra]
        return ssh_command(self.ssh, *options, command=remote)

    def run(self, remote: str, stdin: bytes = b"") -> bytes:
        """Run *remote* on the peer and return what it wrote to stdout.

        Args:
            remote: The remote shell command.
            stdin: Bytes fed to the command's standard input.

        Raises:
            PeerError: When the command exits non-zero (its stderr is quoted),
                was stopped, or was killed for stalling (:meth:`call`).
        """
        with tempfile.TemporaryFile() as fed, tempfile.TemporaryFile() as out:
            fed.write(stdin)
            fed.seek(0)
            code, errors = self.call(self.command(remote), self.cancel, COMMAND_SECONDS, fed, out)
            out.seek(0)
            output = out.read()
        if code != 0:
            detail = errors.decode("utf-8", "replace").strip() or f"exit {code}"
            raise PeerError(f"{self.name} ({self.ssh}): {detail}")
        return output

    def python_command(self, script: bytes, shipped: dict[str, bytes], args: list[str]) -> str:
        """Return a remote command running *script* with the system ``python3`` and no files.

        The script and its helper modules travel base64-encoded inside the
        command line; the script reads the helpers from the
        ``SHIPPED_SOURCES`` global, as ``kiss.scripts.sync_db`` does.

        Args:
            script: Source text of the script to run as ``__main__``.
            shipped: ``{module name: source text}`` the script may load.
            args: Its command-line arguments.
        """
        helpers = ",".join(
            f"{name!r}:base64.b64decode('{base64.b64encode(src).decode('ascii')}')"
            for name, src in shipped.items()
        )
        bootstrap = (
            f"import base64;SHIPPED_SOURCES={{{helpers}}};"
            f"exec(base64.b64decode('{base64.b64encode(script).decode('ascii')}'))"
        )
        return " ".join(shlex.quote(part) for part in [self.python, "-c", bootstrap, *args])

    def runner(self) -> PeerRunner:
        """Return the ``sync_db`` phase runner that reaches this peer over the master connection."""
        return PeerRunner(self)


class PeerRunner(Runner):
    """The ``sync_db`` runner of one peer: its remote phases ride the peer's master connection.

    Attributes:
        peer: The machine the remote phases run on.
    """

    def __init__(self, peer: SshPeer) -> None:
        """Run remote phases with *peer*'s interpreter, over its master connection."""
        super().__init__(peer.python, peer.port, peer.ssh_options)
        self.peer = peer

    def run(
        self, location: Location, phase: str, args: list[str], inp: str | None, out: str
    ) -> None:
        """Run one phase as ``Runner.run`` does, unless the peer's run was stopped.

        Raises:
            SyncError: When the run was stopped: no phase starts after that,
                not even one on this machine, which no kill of an ssh ends.
                Otherwise as ``Runner.run`` raises it.
        """
        if self.peer.cancel.is_set():
            raise SyncError(f"{self.peer.name} ({self.peer.ssh}): stopped")
        super().run(location, phase, args, inp, out)

    def run_ssh(self, command: list[str], in_file: Any, out_file: Any) -> tuple[int, bytes]:
        """Run a remote phase's ssh *command* with :meth:`SshPeer.call`.

        Raises:
            SyncError: When the command was stopped, or was killed for stalling.
            OSError: When ssh could not be started.
        """
        try:
            return self.peer.call(command, self.peer.cancel, COMMAND_SECONDS, in_file, out_file)
        except PeerError as exc:
            raise SyncError(str(exc)) from exc
