# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (E1, E3): :class:`MachineLink` budgets and tunnel ports.

E1: the stale-token retry of :meth:`MachineLink._connect` re-opened
the SSH tunnel (two ``ssh`` logins, on the tunnel's own
``SSH_TIMEOUT`` budget) and then re-ran the handshake *before the same
deadline* the request started with.  On a host where the two logins
take longer than the request timeout, the retried handshake got
``_remaining(deadline)``'s 1 ms floor, failed with "unreachable: timed
out", backed the machine off for ``DOWN_BACKOFF`` seconds and closed
the tunnel it had just opened: the retry could never win.  Now the
handshake has the request's full *timeout* after a tunnel (re)open,
and the reply wait is timed from the moment the socket is up.

The test slows every ``ssh`` down with a wrapper on ``PATH`` (one
second of sleep before the real binary), restarts the daemon on the
same port under the living tunnel and asks for a ping with a request
timeout shorter than the two logins the retry needs.

E3: ``SshTunnel.open`` picked a free port and let ``ssh`` bind it
later; two tunnels opened at once (every SSH machine on the first
CHATS refresh) could be handed the same port, and the loser's
``ExitOnForwardFailure`` exit was reported as the machine being down
for ``DOWN_BACKOFF`` seconds.  The open now retries once with another
port when ``ssh`` exits on "Address already in use".  A probe of the
port cannot tell the winner's listener from ``ssh``'s own (the round-2
review caught the open accepting a competitor's listener while ``ssh``
was still logging in, then returning a dead tunnel), so ``ssh`` now
runs :data:`machines.TUNNEL_COMMAND` and the open waits for its
``TUNNEL_READY`` line, which ``ExitOnForwardFailure`` only lets run
once the local port is bound by this very ``ssh``.  The interleaving
cannot be forced from outside without replacing ``_free_port`` (the
kernel chooses the port), so a wrapper ``ssh`` plays the competitor
on the port the first attempt asks for, see
:func:`test_tunnel_open_retries_once_when_the_port_is_taken`.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.tests.agents.sorcar.test_machines import _endpoint_token, _pong, _titles
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.ssh_server import SSHD, start_sshd

pytestmark = pytest.mark.skipif(not SSHD, reason="no sshd on this host")


def _slow_ssh_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delay: float) -> None:
    """Put an ``ssh`` wrapper that sleeps *delay* seconds before the real one first on PATH."""
    real = shutil.which("ssh")
    assert real, "no ssh client on this host"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "ssh"
    wrapper.write_text(f'#!/bin/sh\nsleep {delay}\nexec "{real}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_stale_token_retry_gets_a_fresh_handshake_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon restarted under a living tunnel is found even when ssh logins are slow."""
    home = tmp_path / "remote-home"
    daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "one"}])
    server = start_sshd(tmp_path / "sshd", home)
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    try:
        machine = mach.add_machine("box", server.login)
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["one"]
        link = mach.link_for(machine)
        assert link.tunnel is not None
        first_token = link.tunnel.token

        port = daemon.port
        daemon.stop()
        daemon = start_remote_daemon(
            home, password="pw", seed=[{"chat_id": "c", "task": "two"}], port=port,
        )
        assert _endpoint_token(daemon) != first_token
        assert link.tunnel.is_open()  # the forward survived the restart

        # Reopening the tunnel now takes two slow logins (> 2 s), longer
        # than the request's whole timeout: the retried handshake must
        # not be charged for them.
        _slow_ssh_on_path(tmp_path, monkeypatch, delay=1.0)
        started = time.monotonic()
        stream = link.open_connection(timeout=1.5)
        assert time.monotonic() - started < mach.DOWN_BACKOFF
        assert link._retry_at == 0.0  # no backoff was started
        assert link.tunnel.token == _endpoint_token(daemon)
        assert _pong(stream)
        stream.close()
        # The request socket cached before the restart died with the old
        # tunnel; the next request connects again on the new one.
        assert link.request({"type": "ping"}, "pong", timeout=1.5)["type"] == "pong"
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
    finally:
        mach.close_links()
        mach.save_machines([])
        daemon.stop()
        server.stop()


def test_tunnel_open_retries_once_when_the_port_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ssh`` losing its port to another listener is retried on a new port, once.

    The wrapper ``ssh`` on PATH takes the very port the first ``-L``
    forward asks for, keeps listening on it while the real ``ssh``
    logs in (so that a probe of the port would find a listener) and
    exits with it; that attempt fails with "Address already in use"
    exactly as a lost race between two concurrent opens does.  The
    marker file lets the second attempt pass through unchanged.
    """
    home = tmp_path / "remote-home"
    daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "one"}])
    server = start_sshd(tmp_path / "sshd", home)
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    real = shutil.which("ssh")
    assert real
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "first-attempt-port"
    wrapper = bin_dir / "ssh"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, socket, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        f"if '-L' in args and not os.path.exists({str(marker)!r}):\n"
        "    port = int(args[args.index('-L') + 1].split(':')[1])\n"
        f"    open({str(marker)!r}, 'w').write(str(port))\n"
        "    with socket.socket() as competitor:\n"
        "        competitor.bind(('127.0.0.1', port))\n"
        "        competitor.listen(8)\n"
        f"        sys.exit(subprocess.call([{real!r}] + args))\n"
        f"os.execv({real!r}, [{real!r}] + args)\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    try:
        # The handshake's tunnel (closed again by add_machine) lost its
        # first port and came up on the second; its log keeps both tries.
        machine = mach.add_machine("box", server.login)
        assert marker.exists(), "the wrapper never saw a forward"
        assert b"Address already in use" in mach.SshTunnel(machine).log_file().read_bytes()
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["one"]
        link = mach.link_for(machine)
        assert link.tunnel is not None and link.tunnel.is_open()
        assert link.tunnel.url != f"https://127.0.0.1:{marker.read_text()}"
        assert link._retry_at == 0.0
    finally:
        mach.close_links()
        mach.save_machines([])
        daemon.stop()
        server.stop()
