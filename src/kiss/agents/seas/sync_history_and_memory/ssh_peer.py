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
"""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from kiss.agents.sorcar.machines import ssh_command
from kiss.scripts.sync_db import Runner

CONTROL_PERSIST = "600"
"""Seconds an idle master lingers before closing itself, should :meth:`SshPeer.close` never run."""

COMMON_OPTIONS = ["StrictHostKeyChecking=accept-new", "ConnectTimeout=10", "LogLevel=ERROR"]
"""ssh options every command of a peer uses (the same ones :func:`ssh_command` sets)."""


class PeerError(Exception):
    """A machine could not be reached, or a command on it failed; the message names it."""


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
    """

    def __init__(self, name: str, ssh: str, control_dir: Path, python: str = "python3") -> None:
        """Describe a peer; nothing is connected until :meth:`open`.

        Args:
            name: The machine's registered name.
            ssh: Its login, ``user@host[:port]``.
            control_dir: A private directory for the control socket.
            python: The remote interpreter running shipped scripts.
        """
        self.name = name
        self.ssh = ssh
        self.python = python
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
        done = subprocess.run(
            ssh_command(self.ssh, *options),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if done.returncode != 0:
            detail = done.stderr.decode("utf-8", "replace").strip() or f"exit {done.returncode}"
            raise PeerError(f"{self.name} ({self.ssh}): cannot connect: {detail}")

    def close(self) -> None:
        """Close the master connection; a peer that never opened is left alone."""
        if not self.socket.exists():
            return
        subprocess.run(
            self.command("", "-O", "exit"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )

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
            PeerError: When the command exits non-zero (its stderr is quoted).
        """
        done = subprocess.run(self.command(remote), input=stdin, capture_output=True, check=False)
        if done.returncode != 0:
            detail = done.stderr.decode("utf-8", "replace").strip() or f"exit {done.returncode}"
            raise PeerError(f"{self.name} ({self.ssh}): {detail}")
        return done.stdout

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

    def runner(self) -> Runner:
        """Return the ``sync_db`` phase runner that reaches this peer over the master connection."""
        return Runner(self.python, self.port, self.ssh_options)
