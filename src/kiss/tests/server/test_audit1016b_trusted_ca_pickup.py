# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A CA dropped into ``tls/trusted-clients/`` is admitted by the running daemon.

Audit 2026-10-16 (server-web-2 F1): the daemon's trusted-clients watch
reads the enrolled CA files in a worker thread and only loads them into
the live SSL context on the event loop.  This drives a real daemon (a
child process bound on every address of the host) and checks, through
real TLS handshakes from the host's LAN address, that a client
certificate is refused before its CA is enrolled, admitted soon after
the CA file appears (next to a junk ``.pem`` and a directory named
``.pem``, which the loader skips), and that a second CA enrolled later
is picked up as well.
"""

from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.core import tls_certs
from kiss.server.web_server import _get_local_ips
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon

pytestmark = pytest.mark.timeout(180)


def _lan_ip() -> str:
    ips = sorted(_get_local_ips())
    return ips[0] if ips else ""


@pytest.fixture(scope="module")
def daemon(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A daemon with no password (localhost-only) bound on every address of the host."""
    if not _lan_ip():
        pytest.skip("this host has no LAN address")
    child = start_remote_daemon(tmp_path_factory.mktemp("lan-home"), host="0.0.0.0")
    yield child
    child.stop()


HANDSHAKE_FAILED = 0
"""The status of a request the daemon dropped during the TLS handshake."""


def _http_status(url: str, client_cert: tuple[Path, Path]) -> int:
    """GET *url* (its certificate unverified) presenting *client_cert*.

    Once any CA is enrolled the daemon requests a client certificate
    and fails the handshake of one that chains to no enrolled CA:
    that is :data:`HANDSHAKE_FAILED`.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    ctx.load_cert_chain(*map(str, client_cert))
    try:
        with urllib.request.urlopen(url, timeout=10, context=ctx) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:  # ssl.SSLError, RemoteDisconnected, URLError
        return HANDSHAKE_FAILED


def _wait_for_status(url: str, client_cert: tuple[Path, Path], status: int) -> int:
    """GET *url* until it answers *status* or 15 s pass (the daemon polls every 2 s)."""
    deadline = time.monotonic() + 15.0
    while (code := _http_status(url, client_cert)) != status and time.monotonic() < deadline:
        time.sleep(0.2)
    return code


def _enroll(trusted: Path, name: str, ca_pem: Path) -> None:
    """Write *ca_pem* as ``<name>.pem`` the way ``/machine add`` does (temp file + rename)."""
    tmp = trusted / f".{name}.pem.tmp"
    tmp.write_bytes(ca_pem.read_bytes())
    tmp.replace(trusted / f"{name}.pem")


def test_ca_dropped_into_trusted_clients_is_admitted_by_the_running_daemon(
    daemon: RemoteDaemon, tmp_path: Path,
) -> None:
    url = f"https://{_lan_ip()}:{daemon.port}/ca.crt"
    trusted = daemon.home / "tls" / tls_certs.TRUSTED_CLIENTS_DIR
    assert trusted.is_dir()  # created at bind: the enrolling side looks for it
    alice = tls_certs.ensure_local_client_pair(tmp_path / "alice")
    bob = tls_certs.ensure_local_client_pair(tmp_path / "bob")

    # No CA enrolled: a LAN peer with a certificate is just a LAN peer.
    assert _http_status(url, alice) == 403

    (trusted / "junk.pem").write_text("not a certificate", encoding="utf-8")
    (trusted / "subdir.pem").mkdir()
    _enroll(trusted, "alice", tmp_path / "alice" / "ca.pem")
    assert _wait_for_status(url, alice, 200) == 200
    assert _http_status(url, bob) == HANDSHAKE_FAILED  # chains to no enrolled CA

    # A CA enrolled later (the watch, not the startup load) is picked up too.
    time.sleep(1.0)  # past the first rescan: this write is a fresh mtime
    _enroll(trusted, "bob", tmp_path / "bob" / "ca.pem")
    assert _wait_for_status(url, bob, 200) == 200
    assert _http_status(url, alice) == 200

    # Revocation is the file's absence, checked at every handshake (the
    # CA stays in the live context, so the handshake itself still succeeds).
    (trusted / "alice.pem").unlink()
    assert _http_status(url, alice) == 403
    assert _http_status(url, bob) == 200
