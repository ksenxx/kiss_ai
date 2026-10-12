# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A private ``sshd`` for tests of SSH machines (:mod:`kiss.agents.sorcar.machines`).

:func:`start_sshd` runs the system's ``sshd`` unprivileged on a free
loopback port with its own host key, a fresh client key as the only
authorised key, and ``KISS_HOME`` set (``SetEnv``) to the home of the
daemon that plays the remote machine, so ``ssh <user>@127.0.0.1:<port>
cat "$KISS_HOME/sorcar-local.json"`` reads that daemon's endpoint file.
The client side is a config file (key, known-hosts file) meant for
``$KISS_SSH_CONFIG``; nothing under ``~/.ssh`` is read or written.
"""

from __future__ import annotations

import getpass
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from kiss.tests.conftest import find_free_port

SSHD = shutil.which("sshd") or ("/usr/sbin/sshd" if Path("/usr/sbin/sshd").exists() else "")
"""The ``sshd`` binary, empty when the host has none (the SSH tests skip).

Empty on Windows too: its OpenSSH logs in to ``cmd`` or PowerShell, and
an SSH machine's endpoint is read with a POSIX shell command (see
:data:`kiss.agents.sorcar.machines.REMOTE_ENDPOINT_COMMAND`).
"""
if sys.platform == "win32":
    SSHD = ""


class SshServer:
    """Handle of an ``sshd`` started by :func:`start_sshd`."""

    def __init__(self, process: subprocess.Popen[bytes], port: int, config: Path) -> None:
        self.process = process
        self.port = port
        self.config = config
        """Client config file (``-F``) with the key and known-hosts file to use."""

    @property
    def login(self) -> str:
        """``<user>@127.0.0.1:<port>``, the address to register the machine with."""
        return f"{getpass.getuser()}@127.0.0.1:{self.port}"

    def stop(self) -> None:
        """Kill the server and wait for it."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=20)


def start_sshd(workdir: Path, remote_home: Path, timeout: float = 30.0) -> SshServer:
    """Start an ``sshd`` under *workdir* whose sessions see ``KISS_HOME=remote_home``.

    A second server started under the same *workdir* reuses its keys
    and client config (it differs by port and ``KISS_HOME`` only).

    Raises:
        RuntimeError: When ``sshd`` exits or the port stays closed.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    host_key, client_key = workdir / "host_key", workdir / "client_key"
    for key in (host_key, client_key):
        if not key.exists():
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True,
            )
    authorized = workdir / "authorized_keys"
    authorized.write_bytes(client_key.with_suffix(".pub").read_bytes())
    authorized.chmod(0o600)
    config = workdir / "ssh_config"
    config.write_text(
        f"Host *\n  IdentityFile {client_key}\n  IdentitiesOnly yes\n"
        f"  UserKnownHostsFile {workdir / 'known_hosts'}\n",
        encoding="utf-8",
    )
    port = find_free_port()
    process = subprocess.Popen([
        SSHD, "-D", "-e", "-f", "/dev/null", "-p", str(port), "-h", str(host_key),
        "-o", "ListenAddress=127.0.0.1", "-o", "PidFile=none",
        "-o", f"AuthorizedKeysFile={authorized}", "-o", "StrictModes=no",
        "-o", "PasswordAuthentication=no", "-o", "UsePAM=no",
        "-o", f"SetEnv=KISS_HOME={remote_home}", "-o", "LogLevel=ERROR",
    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sshd exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return SshServer(process, port, config)
        except OSError:
            time.sleep(0.2)
    process.kill()
    raise RuntimeError(f"sshd did not open port {port} within {timeout}s")
