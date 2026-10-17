# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of :mod:`kiss.agents.sorcar.machines`.

The remote machines are real ``kiss-web`` daemons started in their own
processes (:mod:`kiss.tests.remote_daemon`), each with its own
``KISS_HOME`` and ``history.db``; the registry under test lives in the
test session's ``KISS_HOME``.
"""

from __future__ import annotations

import json
import os
import random
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect

from kiss.agents.sorcar import machines as mach
from kiss.core import tls_certs
from kiss.core.file_lock import exclusive_file_lock
from kiss.server import sidebar_panels
from kiss.server.web_server import _get_local_ips
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import make_test_tls
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.ssh_server import SSHD, SshServer, start_sshd

needs_sshd = pytest.mark.skipif(not SSHD, reason="no sshd on this host")

ID_SHARED = "a" * 32


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """Every test starts and ends with no registered machine."""
    mach.save_machines([])
    yield
    mach.save_machines([])


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A password-protected daemon with its own local CA and three tasks."""
    daemon = start_remote_daemon(
        tmp_path_factory.mktemp("remote-home"), password="secret-1",
        seed=[
            {"chat_id": "chat-r1", "task": "remote alpha", "result": "ok"},
            {"chat_id": "chat-r1", "task": "remote beta"},
            {"chat_id": "chat-shared", "task": "shared task", "task_id": ID_SHARED},
        ],
    )
    yield daemon
    daemon.stop()


@pytest.fixture(scope="module")
def remote_leaf(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A daemon with an explicit certificate (no ``/ca.crt``) and no password."""
    tls_dir = tmp_path_factory.mktemp("leaf-tls")
    certfile, keyfile, _ca = make_test_tls(tls_dir)
    daemon = start_remote_daemon(
        tmp_path_factory.mktemp("leaf-home"),
        seed=[{"chat_id": "chat-l", "task": "leaf task"}],
        certfile=certfile, keyfile=keyfile,
    )
    yield daemon
    daemon.stop()


def _closed_port() -> int:
    """Return a loopback port nothing listens on.

    Drawn below every OS ephemeral range (Linux 32768+, macOS and
    Windows 49152+), so a listener another test process binds on port
    0 meanwhile can never land on it; the daemon default port 8787 is
    not used either, as sibling processes start daemons there.
    """
    while True:
        port = random.randrange(21000, 32000)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port


def _titles(page: mach.HistoryPage) -> list[str]:
    return [str(r["title"]) for r in page.rows]


def test_normalize_url_accepts_the_usual_spellings() -> None:
    """A bare host, host:port, https and wss forms all become https://host:port."""
    assert mach.normalize_url("10.0.0.5") == "https://10.0.0.5:8787"
    assert mach.normalize_url("box.example.org:9787") == "https://box.example.org:9787"
    assert mach.normalize_url("https://box.example.org:9787/") == "https://box.example.org:9787"
    assert mach.normalize_url("wss://box.example.org/ws") == "https://box.example.org:8787"
    assert mach.normalize_url("https://[::1]:9000") == "https://[::1]:9000"
    assert mach.ws_url("https://[::1]:9000") == "wss://[::1]:9000/ws"
    with pytest.raises(mach.MachineError, match="not a machine address"):
        mach.normalize_url("ftp://box")
    with pytest.raises(mach.MachineError, match="not a machine address"):
        mach.normalize_url("")
    # A port that is no number (or out of range) is the user's mistake,
    # reported as such rather than as urlsplit's ValueError.
    with pytest.raises(mach.MachineError, match="not a machine address"):
        mach.normalize_url("https://127.0.0.1:abc")
    with pytest.raises(mach.MachineError, match="not a machine address"):
        mach.normalize_url("127.0.0.1:99999")


def test_ssh_addresses_are_recognised_normalised_and_named() -> None:
    """``user@host[:port]`` and ``ssh://`` forms are SSH machines; the host names them."""
    assert mach.is_ssh_address("ksen@10.0.0.5") and mach.is_ssh_address("ssh://box")
    assert not mach.is_ssh_address("https://box") and not mach.is_ssh_address("box:8787")
    assert mach.normalize_ssh("ksen@10.0.0.5") == "ksen@10.0.0.5"
    assert mach.normalize_ssh(" ssh://ksen@box.example.org:2222/") == "ksen@box.example.org:2222"
    assert mach.normalize_ssh("ssh://box") == "box"
    assert mach.normalize_ssh("ksen@[::1]:22") == "ksen@[::1]:22"
    for bad in ("ksen@", "ssh://", "ssh://box/path", "ksen@box:notaport"):
        with pytest.raises(mach.MachineError, match="not an SSH address"):
            mach.normalize_ssh(bad)
    assert mach.default_name("ksen@10.0.0.5") == "10.0.0.5"
    assert mach.default_name("ssh://ksen@box.example.org:2222") == "box.example.org"
    assert mach.default_name("https://10.0.0.5:8787") == "10.0.0.5"
    assert mach.default_name("box:9787") == "box"
    assert mach.default_name("ssh://ksen@my_box") == "my_box"
    assert mach.default_name("@@") == "machine"
    cmd = mach.ssh_command("ksen@box:2222", "-N", command="cat x")
    assert cmd[0] == "ssh" and "BatchMode=yes" in cmd and "StrictHostKeyChecking=accept-new" in cmd
    assert "ControlMaster=no" in cmd and "ControlPath=none" in cmd
    assert cmd[-4:] == ["-p", "2222", "ksen@box", "cat x"]
    assert cmd.index("-N") < cmd.index("ksen@box")
    assert mach.ssh_command("box")[-1] == "box" and "-p" not in mach.ssh_command("box")
    assert mach.ssh_command("ksen@[::1]:22")[-1] == "ksen@::1"
    assert mach.Machine("b", ssh="ksen@box").address() == "ssh ksen@box"
    assert mach.Machine("b", "https://h:1").address() == "https://h:1"
    assert mach.ENROLL_COMMAND == (
        "sh -c 'for d in \"$KISS_HOME\" \"$HOME/" + mach.HOME_DIR + "\" \"$HOME\"/.*; do "
        'case "$d" in ""|*/.|*/..) continue;; esac; '
        'if [ -f "$d/sorcar-local.json" ]; then '
        '[ -d "$d/tls/trusted-clients" ] || { '
        'echo "its daemon predates certificate enrollment: upgrade it" >&2; exit 3; }; '
        'cat > "$d/tls/trusted-clients/.NAME.$$.tmp" && '
        'mv -f "$d/tls/trusted-clients/.NAME.$$.tmp" "$d/tls/trusted-clients/NAME.pem" '
        '|| exit 2; cat "$d/sorcar-local.json"; echo; '
        'c=$(sed -n "s/^ *\\"ca\\": *\\"\\(.*\\)\\",*$/\\1/p" "$d/sorcar-local.json"); '
        'cat "${c:-$d/tls/ca.pem}"; exit 0; fi; done; '
        "echo \"no $HOME/.*/sorcar-local.json\" >&2; exit 1'"
    )
    assert "!" not in mach.ENROLL_COMMAND  # csh expands history inside quotes
    assert mach.ssh_host("ksen@127.0.0.1:2222") == "127.0.0.1"
    assert mach.ssh_host("ksen@[::1]") == "::1"


def test_registry_round_trip_and_corrupt_file() -> None:
    """save/load keep order and the mode is 0600; junk files load as empty."""
    mach.save_machines([
        mach.Machine("b", "https://h:1", "pw"), mach.Machine("a", "https://h:2"),
        mach.Machine("s", ssh="ksen@h:22"),
    ])
    assert mach.load_machines() == [
        mach.Machine("b", "https://h:1", "pw"), mach.Machine("a", "https://h:2"),
        mach.Machine("s", ssh="ksen@h:22"),
    ]
    if os.name != "nt":  # Windows keeps only the read-only bit of a mode
        assert oct(mach.machines_file().stat().st_mode & 0o777) == "0o600"
    assert not list(mach.machines_file().parent.glob(".machines.json.*.tmp"))
    mach.machines_file().write_text("{not json", encoding="utf-8")
    assert mach.load_machines() == []
    mach.machines_file().write_text(
        json.dumps({"machines": [{"name": "", "url": "x"}, {"url": "y"}, "junk",
                                 {"name": "ok", "url": "https://h:3"}, {"name": "neither"},
                                 {"name": "old", "url": "https://h:4", "password": "p"}]}),
        encoding="utf-8",
    )
    assert mach.load_machines() == [
        mach.Machine("ok", "https://h:3"), mach.Machine("old", "https://h:4", "p"),
    ]
    mach.machines_file().write_text("[]", encoding="utf-8")
    assert mach.load_machines() == []
    assert mach.remove_machine("nobody") is False


def test_local_machine_name_is_the_short_hostname() -> None:
    """The local label is the host's short name (never empty)."""
    name = mach.local_machine_name()
    assert name and "." not in name


def test_add_machine_pins_the_ca_checks_the_password_and_saves(remote: RemoteDaemon) -> None:
    """A good address and password register the machine; its CA is pinned."""
    machine = mach.add_machine("box", f"127.0.0.1:{remote.port}", "secret-1")
    assert machine == mach.Machine("box", remote.url, "secret-1")
    assert mach.load_machines() == [machine]
    pem = machine.cert_file().read_text(encoding="utf-8")
    assert pem == (remote.home / "tls" / "ca.pem").read_text(encoding="utf-8")
    assert not list(machine.cert_file().parent.glob(".box.pem.*"))
    # The pinned CA really is what the link verifies against.
    ctx = ssl.create_default_context(cafile=str(machine.cert_file()))
    assert ctx.cert_store_stats()["x509_ca"] == 1
    assert mach.machine_status(machine) == "reachable"

    # Re-adding the same name replaces the entry (no duplicates).
    again = mach.add_machine("box", remote.url, "secret-1")
    assert mach.load_machines() == [again]

    assert mach.remove_machine("box") is True
    assert mach.load_machines() == []
    assert not machine.cert_file().exists()


def test_add_machine_pins_the_leaf_when_there_is_no_ca(remote_leaf: RemoteDaemon) -> None:
    """A daemon with its own certificate files is pinned by that certificate."""
    machine = mach.add_machine("leaf", remote_leaf.url)
    assert "BEGIN CERTIFICATE" in machine.cert_file().read_text(encoding="utf-8")
    assert mach.machine_status(machine) == "reachable"
    page = mach.fetch_history(machine, "", "", 10)
    assert [(r["machine"], r["title"], r["machine_url"]) for r in page.rows] == [
        ("leaf", "leaf task", remote_leaf.url),
    ]
    assert page.date_min is not None and page.date_max is not None
    assert page.date_min <= page.rows[0]["timestamp"] <= page.date_max


def test_add_machine_rejects_bad_input_and_keeps_the_old_entry(
    remote: RemoteDaemon, remote_leaf: RemoteDaemon,
) -> None:
    """Nothing changes when the name, the password or the address is wrong."""
    with pytest.raises(mach.MachineError, match="not a valid machine name"):
        mach.add_machine("bad name!", remote.url, "secret-1")
    with pytest.raises(mach.MachineError, match="rejected the password"):
        mach.add_machine("box", remote.url, "wrong")
    assert mach.load_machines() == []
    with pytest.raises(mach.MachineError, match="Cannot reach"):
        mach.add_machine("box", f"https://127.0.0.1:{_closed_port()}", "")
    assert mach.load_machines() == []
    assert not (mach.kiss_home() / mach.MACHINE_CERTS_DIR / "box.pem").exists()

    # A failed replacement leaves the registered machine and its pin intact.
    box = mach.add_machine("box", remote.url, "secret-1")
    pin = box.cert_file().read_text(encoding="utf-8")
    with pytest.raises(mach.MachineError, match="rejected the password"):
        mach.add_machine("box", remote_leaf.url, "wrong")
    assert mach.load_machines() == [box]
    assert box.cert_file().read_text(encoding="utf-8") == pin
    assert mach.machine_status(box) == "reachable"


def test_concurrent_adds_keep_both_machines(
    remote: RemoteDaemon, remote_leaf: RemoteDaemon,
) -> None:
    """Two registrations at once both land (the registry edit is locked)."""
    errors: list[BaseException] = []

    def add(name: str, url: str, password: str) -> None:
        try:
            mach.add_machine(name, url, password)
        except BaseException as exc:  # noqa: BLE001 — reported by the assertion below
            errors.append(exc)

    threads = [
        threading.Thread(target=add, args=("box", remote.url, "secret-1")),
        threading.Thread(target=add, args=("leaf", remote_leaf.url, "")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert {m.name for m in mach.load_machines()} == {"box", "leaf"}

    # Two registrations of the SAME name at once: whichever lands last
    # publishes its own pin with its own entry, so the survivor works.
    threads = [
        threading.Thread(target=add, args=("same", remote.url, "secret-1")),
        threading.Thread(target=add, args=("same", remote_leaf.url, "")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    same = [m for m in mach.load_machines() if m.name == "same"]
    assert len(same) == 1
    assert mach.machine_status(same[0]) == "reachable"
    assert not list(same[0].cert_file().parent.glob(".same.pem.*"))


def test_fetch_history_pages_filters_and_stamps(remote: RemoteDaemon) -> None:
    """Rows come newest first with the machine stamp; query and limit apply."""
    machine = mach.add_machine("box", remote.url, "secret-1")
    page = mach.fetch_history(machine, "", "", 50)
    assert _titles(page) == ["shared task", "remote beta", "remote alpha"]
    assert {r["machine"] for r in page.rows} == {"box"}
    assert {r["machine_url"] for r in page.rows} == {remote.url}
    assert page.rows[0]["task_id"] == ID_SHARED and page.rows[0]["id"] == "chat-shared"
    assert _titles(mach.fetch_history(machine, "alpha", "", 50)) == ["remote alpha"]
    assert _titles(mach.fetch_history(machine, "", "", 2)) == ["shared task", "remote beta"]
    assert mach.fetch_history(machine, "", "coding", 50).rows == []


def test_link_survives_a_dropped_socket_and_reports_a_dead_machine(
    remote: RemoteDaemon, tmp_path: Path,
) -> None:
    """The cached link reconnects after a close; an unreachable machine raises."""
    machine = mach.add_machine("box", remote.url, "secret-1")
    link = mach.link_for(machine)
    assert link is mach.link_for(machine)
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    link.close()
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    assert len(mach.fetch_history(machine, "", "", 50).rows) == 3
    # The whole wait is bounded, however many unrelated frames arrive.
    with pytest.raises(mach.MachineError, match="did not answer"):
        link.request({"type": "ping"}, "never-sent", timeout=0.5)
    assert link._retry_at > time.monotonic()
    link._retry_at = 0.0
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    # An already expired deadline fails even when the reply is buffered.
    assert link._ws is not None
    # (A raw frame on the link's socket must carry the machine's password
    # like the link's own frames do, or the daemon closes the socket.)
    link._ws.send(json.dumps({"type": "ping", "password": "secret-1"}))
    time.sleep(0.3)
    with pytest.raises(mach.MachineError, match="no pong within 0s"):
        link.request({"type": "ping"}, "pong", timeout=0.0)
    assert link._retry_at > time.monotonic()
    link._retry_at = 0.0
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"

    dead = mach.Machine("dead", f"https://127.0.0.1:{_closed_port()}")
    dead.cert_file().write_text(machine.cert_file().read_text(), encoding="utf-8")
    with pytest.raises(mach.MachineError, match="unreachable"):
        mach.fetch_history(dead, "", "", 50)
    assert "unreachable" in mach.machine_status(dead)
    # The cached link backs off: the next attempt fails at once, with the
    # same message, without touching the network (a closed port would
    # answer instantly anyway, so the backoff deadline is what proves it).
    dead_link = mach.link_for(dead)
    assert dead_link._retry_at > time.monotonic()
    dead_link._retry_at = time.monotonic() + 3600
    started = time.monotonic()
    with pytest.raises(mach.MachineError, match="unreachable"):
        mach.fetch_history(dead, "", "", 50)
    assert time.monotonic() - started < 0.5
    dead_link._retry_at = 0.0  # expired: the next attempt connects again
    with pytest.raises(mach.MachineError, match="unreachable"):
        dead_link.request({"type": "ping"}, "pong")
    # A pin that does not match the daemon is a refused TLS handshake.
    wrong = mach.Machine("wrong", remote.url, "secret-1")
    unrelated_cert, _key, _ca = make_test_tls(tmp_path)
    wrong.cert_file().write_text(unrelated_cert.read_text(), encoding="utf-8")
    assert "refused the TLS handshake" in mach.machine_status(wrong)
    # A missing or unreadable pin is a machine error too, never an exception.
    os.unlink(wrong.cert_file())
    assert "no usable pinned certificate" in mach.machine_status(wrong)
    wrong.cert_file().write_text("not a certificate", encoding="utf-8")
    assert "no usable pinned certificate" in mach.machine_status(wrong)
    os.unlink(wrong.cert_file())
    os.unlink(dead.cert_file())


def _local_row(task_id: str, title: str, timestamp: float, running: bool) -> dict[str, Any]:
    return {
        "id": "chat-" + title, "task_id": task_id, "title": title, "timestamp": timestamp,
        "is_running": running,
    }


def test_merged_history_dedups_orders_and_paginates(
    remote: RemoteDaemon, remote_leaf: RemoteDaemon,
) -> None:
    """Of two copies of a task the running / newest one is kept; the union pages."""
    box = mach.add_machine("box", remote.url, "secret-1")
    leaf = mach.add_machine("leaf", remote_leaf.url)
    remote_shared = mach.fetch_history(box, "shared", "", 1).rows[0]

    # An older local copy loses to the remote one (shown as the remote's).
    local = mach.HistoryPage([
        _local_row(ID_SHARED, "shared (local, old)", 1.0, False),
        _local_row("b" * 32, "local running", 2.0, True),
    ], 1.0, 2.0)
    merged = mach.merged_history(local, [box, leaf], "", "", 0, 50)
    titles = _titles(merged)
    assert titles[0] == "local running"
    assert titles.count("shared task") == 1 and "shared (local, old)" not in titles
    assert set(titles[1:]) == {"shared task", "remote beta", "remote alpha", "leaf task"}
    by_title = {r["title"]: r for r in merged.rows}
    assert "machine" not in by_title["local running"]
    assert by_title["shared task"]["machine"] == "box"
    assert by_title["leaf task"]["machine"] == "leaf"
    # The date range spans every machine's history.
    assert merged.date_min == 1.0
    assert merged.date_max is not None and merged.date_max >= remote_shared["timestamp"]

    # A local copy with the same timestamp wins the tie; a running one wins outright.
    tied = mach.HistoryPage([
        _local_row(ID_SHARED, "shared (local, tied)", remote_shared["timestamp"], False),
    ])
    assert "shared (local, tied)" in _titles(mach.merged_history(tied, [box], "", "", 0, 50))
    running = mach.HistoryPage([_local_row(ID_SHARED, "shared (local, running)", 1.0, True)])
    assert _titles(mach.merged_history(running, [box], "", "", 0, 50))[0] == (
        "shared (local, running)"
    )

    # Pagination cuts the union, not each source.
    page = mach.merged_history(local, [box, leaf], "", "", 2, 2)
    assert _titles(page) == titles[2:4]
    assert mach.merged_history(local, [box, leaf], "", "", 10, 50).rows == []
    # A query applies on every machine.
    empty = mach.HistoryPage()
    assert _titles(mach.merged_history(empty, [box, leaf], "leaf", "", 0, 50)) == ["leaf task"]
    # An unreachable machine contributes nothing and breaks nothing.
    dead = mach.Machine("dead", f"https://127.0.0.1:{_closed_port()}")
    dead.cert_file().write_text(box.cert_file().read_text(), encoding="utf-8")
    assert _titles(mach.merged_history(empty, [dead, leaf], "", "", 0, 50)) == ["leaf task"]
    os.unlink(dead.cert_file())
    assert mach.merged_history(empty, [], "", "", 0, 50) == mach.HistoryPage()


def test_merged_pages_are_consistent_with_tied_timestamps(
    tmp_path: Path,
) -> None:
    """Rows that tie keep their source's order, so pages never overlap or skip."""
    tied = start_remote_daemon(
        tmp_path / "tied-home",
        seed=[
            {"chat_id": "c", "task": f"remote {i}", "timestamp": 1_700_000_000.0}
            for i in range(3)
        ] + [{"chat_id": "s", "task": "shared", "task_id": ID_SHARED,
              "timestamp": 1_700_000_000.0}],
    )
    try:
        machine = mach.add_machine("tied", tied.url)
        local = mach.HistoryPage([
            _local_row("1" * 32, "local 1", 1_700_000_000.0, False),
            _local_row(ID_SHARED, "shared (local)", 1_700_000_000.0, False),
            _local_row("2" * 32, "local 2", 1_700_000_000.0, False),
        ])
        whole = _titles(mach.merged_history(local, [machine], "", "", 0, 50))
        assert whole[:3] == ["local 1", "shared (local)", "local 2"]
        assert sorted(whole[3:]) == ["remote 0", "remote 1", "remote 2"]
        assert "shared" not in whole
        paged: list[str] = []
        for offset in range(0, 8, 2):
            page = mach.merged_history(
                mach.HistoryPage(local.rows[: offset + 2]), [machine], "", "", offset, 2,
            )
            paged.extend(_titles(page))
        assert paged == whole
    finally:
        tied.stop()


@pytest.fixture(scope="module")
def sshd(remote: RemoteDaemon, tmp_path_factory: pytest.TempPathFactory) -> Iterator[SshServer]:
    """An sshd whose logins see the password-protected daemon's home as KISS_HOME."""
    if not SSHD:
        pytest.skip("no sshd on this host")
    server = start_sshd(tmp_path_factory.mktemp("sshd"), remote.home)
    env = pytest.MonkeyPatch()
    env.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    yield server
    env.undo()
    server.stop()


def _lan_ip() -> str:
    """Return an address of this host that is not loopback ('' when it has none)."""
    ips = sorted(_get_local_ips())
    return ips[0] if ips else ""


@pytest.fixture(scope="module")
def lan_remote(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    """A daemon with no password (localhost-only) bound on every address of the host."""
    if not _lan_ip():
        pytest.skip("this host has no LAN address")
    daemon = start_remote_daemon(
        tmp_path_factory.mktemp("lan-home"), host="0.0.0.0",
        seed=[{"chat_id": "lan", "task": "lan alpha"}, {"chat_id": "lan", "task": "lan beta"}],
    )
    yield daemon
    daemon.stop()


@pytest.fixture(scope="module")
def lan_sshd(
    lan_remote: RemoteDaemon, tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SshServer]:
    """An sshd on the LAN address whose logins see the LAN daemon's home as KISS_HOME."""
    if not SSHD:
        pytest.skip("no sshd on this host")
    server = start_sshd(tmp_path_factory.mktemp("lan-sshd"), lan_remote.home, host=_lan_ip())
    env = pytest.MonkeyPatch()
    env.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    yield server
    env.undo()
    server.stop()


def _pong(stream: ClientConnection) -> bool:
    """Ping the daemon over the stream socket *stream*; true when it answers within 10 s."""
    stream.send(json.dumps({"type": "ping"}))
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        frame = json.loads(stream.recv(timeout=deadline - time.monotonic()))
        if isinstance(frame, dict) and frame.get("type") == "pong":
            return True
    return False


def _trusted_files(daemon: RemoteDaemon) -> list[Path]:
    return sorted((daemon.home / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).glob("*.pem"))


def _local_ca_pem() -> str:
    return (mach.kiss_home() / "tls" / tls_certs.CA_CERT_FILE).read_text(encoding="utf-8")


def _http_status(url: str, client_cert: tuple[Path, Path] | None) -> int:
    """GET *url* (its certificate unverified), with the client certificate if given."""
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    if client_cert is not None:
        ctx.load_cert_chain(*map(str, client_cert))
    try:
        with urllib.request.urlopen(url, timeout=10, context=ctx) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return exc.code


def _tunnel_peer_port(link: mach.MachineLink) -> int:
    """Return the port the cached socket of *link* is connected to."""
    assert link._ws is not None
    return int(link._ws.socket.getpeername()[1])


def _wait_for_status(url: str, client_cert: tuple[Path, Path] | None, status: int) -> int:
    """GET *url* until it answers *status* (the daemon polls its enrolled CAs) or 15 s pass."""
    deadline = time.monotonic() + 15.0
    while (code := _http_status(url, client_cert)) != status and time.monotonic() < deadline:
        time.sleep(0.2)
    return code


@needs_sshd
def test_ssh_enrollment_tunnels_the_daemon_port_and_the_client_certificate_is_admitted(
    lan_remote: RemoteDaemon, lan_sshd: SshServer, remote_leaf: RemoteDaemon,
) -> None:
    """``add user@host`` enrolls this daemon's CA over ssh; the link then dials an ssh forward.

    The link's socket is connected to the tunnel's loopback port, not
    to the LAN address, and the tunnel is shared by every link to the
    login.  The daemon has no remote password, so a peer on the LAN
    address is refused unless it presents a certificate of an enrolled
    CA: a direct request with the client certificate is admitted, the
    same request without it (a URL machine, or a plain HTTP request)
    is not, and deleting the CA's file on the machine revokes the
    certificate (the tunnelled link, a loopback peer there, stays).
    """
    lan = _lan_ip()
    machine = mach.add_machine("", lan_sshd.login)
    assert machine == mach.Machine(lan, f"https://{lan}:{lan_remote.port}", ssh=lan_sshd.login)
    assert mach.load_machines() == [machine]
    assert machine.cert_file().read_text(encoding="utf-8") == (
        lan_remote.home / "tls" / "ca.pem"
    ).read_text(encoding="utf-8")
    assert not list(machine.cert_file().parent.glob(f".{lan}.pem.*"))
    cert_path, key_path = mach.client_cert_pair()
    assert cert_path.name == "client.pem" and key_path.name == "client-key.pem"
    assert tls_certs.enrolled_client_ca(
        x509.load_pem_x509_certificate(cert_path.read_bytes()).public_bytes(Encoding.DER),
        lan_remote.home / "tls" / tls_certs.TRUSTED_CLIENTS_DIR,
    ) == _trusted_files(lan_remote)[0].stem
    (trusted,) = _trusted_files(lan_remote)
    assert trusted.read_text(encoding="utf-8") == _local_ca_pem()
    assert trusted.stem == tls_certs.ca_label(mach.kiss_home() / "tls" / "ca.pem")
    assert not list(trusted.parent.glob(".*.tmp"))

    page = mach.fetch_history(machine, "", "", 50)
    assert _titles(page) == ["lan beta", "lan alpha"]
    assert {r["machine"] for r in page.rows} == {lan}
    assert {r["machine_url"] for r in page.rows} == {machine.url}
    assert mach.machine_status(machine) == "reachable"
    _row, row = sidebar_panels._machine_row(machine)
    assert row["port"] == lan_remote.port and row["ip"] == lan and row["status"] == "reachable"
    link = mach.link_for(machine)
    tunnel = mach._tunnels[lan_sshd.login]
    assert tunnel.alive() and tunnel.remote_port == lan_remote.port
    assert _tunnel_peer_port(link) == tunnel.local_port != lan_remote.port
    stream = link.open_connection()
    assert _pong(stream)
    assert int(stream.socket.getpeername()[1]) == tunnel.local_port
    stream.close()
    assert mach._tunnels == {lan_sshd.login: tunnel}  # one forward per login
    assert _wait_for_status(machine.url + "/ca.crt", (cert_path, key_path), 200) == 200
    assert _http_status(machine.url + "/ca.crt", None) == 403
    # An enrolled peer must still open with an auth frame (any content).
    ctx = ssl.create_default_context(cafile=str(machine.cert_file()))
    ctx.check_hostname = False
    ctx.load_cert_chain(str(cert_path), str(key_path))
    for first in (json.dumps({"type": "ping"}), "not json"):
        with ws_connect(mach.ws_url(machine.url), ssl=ctx, open_timeout=10) as raw:
            raw.send(first)
            assert json.loads(raw.recv(timeout=10)) == {
                "type": "error", "code": "auth_failed",
                "text": "The first frame must be an auth frame.",
            }
            with pytest.raises(ConnectionClosed):
                raw.recv(timeout=10)
    with ws_connect(mach.ws_url(machine.url), ssl=ctx, open_timeout=10) as raw:
        raw.send(json.dumps({"type": "auth"}))  # no password, no token
        assert json.loads(raw.recv(timeout=10)) == {"type": "auth_ok", "local": True}

    # The certificate is the credential: the same daemon registered by
    # URL without a password is refused (no password is set there).
    plain = mach.Machine("plain", url=machine.url)
    plain_link = mach.MachineLink(plain, cert_file=machine.cert_file())
    with pytest.raises(mach.MachineError, match=r"refused the connection: .*403$"):
        plain_link.request({"type": "ping"}, "pong")
    plain_link.close()

    # Enrolling again (an explicit name, the ssh:// spelling) rewrites
    # the same file; the merge works like for any other machine.
    box = mach.add_machine("box", "ssh://" + lan_sshd.login)
    assert box == mach.Machine("box", machine.url, ssh=lan_sshd.login)
    assert _trusted_files(lan_remote) == [trusted]
    leaf = mach.add_machine("leaf", remote_leaf.url)
    merged = mach.merged_history(mach.HistoryPage(), [box, leaf], "", "", 0, 50)
    assert set(_titles(merged)) == {"lan alpha", "lan beta", "leaf task"}

    # Enrolling again restarted the tunnel (the registry changed); the
    # links to "box" and to the machine's default name share it.
    assert lan_sshd.login in mach._tunnels and mach._tunnels[lan_sshd.login] is not tunnel
    assert _titles(mach.fetch_history(box, "alpha", "", 50)) == ["lan alpha"]
    assert _tunnel_peer_port(mach.link_for(box)) == mach._tunnels[lan_sshd.login].local_port

    # Revoked on the machine: a direct connection with the certificate
    # is refused at once; the tunnelled link (a loopback peer on the
    # machine, admitted by the ssh login) is unaffected.
    trusted.unlink()
    assert _http_status(machine.url + "/ca.crt", (cert_path, key_path)) == 403
    assert _titles(mach.fetch_history(box, "beta", "", 50)) == ["lan beta"]
    mach.close_links()
    assert mach._tunnels == {}
    assert _titles(mach.fetch_history(box, "", "", 50)) == ["lan beta", "lan alpha"]
    assert mach.add_machine("box", lan_sshd.login) == box  # enrolled again
    assert _trusted_files(lan_remote) == [trusted]
    assert mach.machine_status(box) == "reachable"
    assert mach.remove_machine("box") is True and not box.cert_file().exists()
    assert mach._tunnels == {}
    mach.close_links()


@needs_sshd
def test_ssh_enrollment_failures_are_reported_and_store_nothing(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused login, a closed port, a missing or old daemon, no ssh: a clear error, no entry."""
    user = sshd.login.split("@")[0]
    with pytest.raises(mach.MachineError, match=r"ssh nobody-x@.* failed: .*Permission denied"):
        mach.add_machine("", f"nobody-x@127.0.0.1:{sshd.port}")
    with pytest.raises(mach.MachineError, match=r"ssh .* failed: .*refused"):
        mach.add_machine("", f"{user}@127.0.0.1:{_closed_port()}")
    assert mach.load_machines() == []
    assert not list((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob("127.0.0.1.*"))
    assert not list((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob(".127.0.0.1.*"))

    # A machine with no running daemon, one whose daemon predates
    # enrollment, one whose endpoint file is junk, one whose daemon port
    # is closed.
    nothing = tmp_path / "nothing"
    nothing.mkdir()
    idle = start_sshd(sshd.config.parent, nothing)
    monkeypatch.setattr(mach, "ENROLL_TIMEOUT", 2.0)
    try:
        with pytest.raises(mach.MachineError, match=r"no Sorcar daemon is running on .*: no .*"):
            mach.add_machine("", idle.login)
        endpoint = nothing / "sorcar-local.json"
        endpoint.write_text("junk", encoding="utf-8")
        with pytest.raises(mach.MachineError, match="cannot enroll on .*: its daemon predates"):
            mach.add_machine("", idle.login)
        trusted = nothing / "tls" / tls_certs.TRUSTED_CLIENTS_DIR
        trusted.mkdir(parents=True)
        (nothing / "tls" / "ca.pem").write_text(_local_ca_pem(), encoding="utf-8")
        with pytest.raises(mach.MachineError, match="did not answer the enrollment as a Sorcar"):
            mach.add_machine("", idle.login)
        assert [p.name for p in trusted.iterdir()] == [
            tls_certs.ca_label(mach.kiss_home() / "tls" / "ca.pem") + ".pem",
        ]
        endpoint.write_text(json.dumps({"url": "wss://127.0.0.1/ws", "token": "t"}), "utf-8")
        with pytest.raises(mach.MachineError, match="did not answer the enrollment as a Sorcar"):
            mach.add_machine("", idle.login)
        (nothing / "tls" / "ca.pem").unlink()
        endpoint.write_text(json.dumps({"url": "wss://127.0.0.1:1/ws", "token": "t"}), "utf-8")
        with pytest.raises(mach.MachineError, match="did not answer the enrollment as a Sorcar"):
            mach.add_machine("", idle.login)
        (nothing / "tls" / "ca.pem").write_text(_local_ca_pem(), encoding="utf-8")
        endpoint.write_text(
            json.dumps({"url": f"wss://127.0.0.1:{_closed_port()}/ws", "token": "t"}), "utf-8",
        )
        started = time.monotonic()
        with pytest.raises(mach.MachineError, match=r"is unreachable: .*refused"):
            mach.add_machine("", idle.login)
        assert 2.0 <= time.monotonic() - started < 20.0  # retried until ENROLL_TIMEOUT
        # A directory the login cannot write to.
        trusted.chmod(0o500)
        if os.access(trusted, os.W_OK):
            pytest.skip("running as root: the directory stays writable")
        try:
            with pytest.raises(mach.MachineError, match="cannot store this daemon's CA on"):
                mach.add_machine("", idle.login)
        finally:
            trusted.chmod(0o700)
    finally:
        idle.stop()
    assert mach.load_machines() == []
    assert not list((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob("*127.0.0.1*"))

    # A machine registered before enrollment existed (no URL) says so;
    # the cached link backs off after a failure and reports the same.
    old = mach.Machine("old", ssh=sshd.login)
    mach.save_machines([old])
    assert mach.machine_status(old) == (
        f"old (ssh {sshd.login}) was registered before certificate enrollment; "
        f"run /machine add old {sshd.login} again"
    )
    _row, row = sidebar_panels._machine_row(old)
    assert row["port"] == 0 and row["ip"] == "127.0.0.1" and "registered before" in row["error"]
    link = mach.link_for(old)
    assert link._retry_at > time.monotonic()
    started = time.monotonic()
    with pytest.raises(mach.MachineError, match="registered before certificate enrollment"):
        mach.fetch_history(old, "", "", 50)
    assert time.monotonic() - started < 0.5

    # The TLS directory locked by another writer: no client certificate.
    box = mach.add_machine("box", sshd.login)
    mach.close_links()
    monkeypatch.setattr(mach, "TLS_LOCK_TIMEOUT", 0.2)
    with exclusive_file_lock(mach.kiss_home() / "tls" / tls_certs.TLS_LOCK_FILE):
        with pytest.raises(mach.MachineError, match="cannot issue the client certificate"):
            mach.client_cert_pair()
        assert "no usable client certificate: cannot issue" in mach.machine_status(box)
    assert mach.machine_status(box) == "reachable"
    mach.close_links()

    # ssh slower than the budget, an ssh that cannot be asked, no ssh at all.
    with pytest.raises(mach.MachineError, match="did not answer in time"):
        mach.enroll_over_ssh(sshd.login, timeout=0.0)
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(tmp_path / "no-such-config"))
    assert mach.ssh_host("ksen@box.example.org") == "box.example.org"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    with pytest.raises(mach.MachineError, match="ssh is not installed"):
        mach.add_machine("", sshd.login)
    assert mach.ssh_host("ksen@[::1]:22") == "::1"


@needs_sshd
def test_ssh_enrollment_pins_an_explicit_certificate_and_resolves_aliases(
    remote_leaf: RemoteDaemon, sshd: SshServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon started with ``--certfile`` is pinned by what its endpoint names; aliases resolve.

    The endpoint file's ``ca`` is the CA beside the explicit certificate
    (or the certificate itself), not ``tls/ca.pem``, which such a daemon
    need not have.  An ssh config alias is logged into as typed and
    registered at the host it resolves to; the machine is then found by
    its name, by that host and by the alias.
    """
    server = start_sshd(sshd.config.parent, remote_leaf.home)
    config = tmp_path / "ssh_config"
    config.write_text(
        sshd.config.read_text(encoding="utf-8")
        + f"Host leaf-alias\n  HostName 127.0.0.1\n  Port {server.port}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(config))
    user = sshd.login.split("@")[0]
    try:
        assert not (remote_leaf.home / "tls" / "ca.pem").exists()
        assert mach.ssh_host(f"{user}@leaf-alias") == "127.0.0.1"
        machine = mach.add_machine("", f"{user}@leaf-alias")
        assert machine == mach.Machine("leaf-alias", remote_leaf.url, ssh=f"{user}@leaf-alias")
        endpoint = json.loads((remote_leaf.home / "sorcar-local.json").read_text("utf-8"))
        assert machine.cert_file().read_bytes() == Path(endpoint["ca"]).read_bytes()
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["leaf task"]
        assert mach.find_machine("leaf-alias") == machine
        assert mach.find_machine("127.0.0.1") == machine
        custom = mach.add_machine("custom", f"{user}@leaf-alias")
        mach.remove_machine("leaf-alias")
        assert mach.find_machine("leaf-alias") == custom  # by the login's host
        assert mach.find_machine("127.0.0.1") == custom  # by the URL's host
        assert mach.machine_hosts(custom) == {"127.0.0.1", "leaf-alias"}
        assert mach.find_machine("elsewhere") is None
    finally:
        mach.close_links()
        server.stop()


@needs_sshd
def test_ssh_enrollment_finds_the_daemon_whatever_its_brand_home(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path,
) -> None:
    """The daemon is found under ``$KISS_HOME``, else ``~/<local brand>``, else any ``~/.*``.

    A remote re-installed under another brand moves its state directory
    (``~/.kiss`` to ``~/.s10s`` or back); enrollment must find the
    running daemon through such a change without any remote
    configuration.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".s10s").symlink_to(remote.home)  # another brand's daemon, its endpoint file live
    missing = tmp_path / "missing"
    # ``~/.`` and ``~/..`` are not brand homes: neither file is ever read.
    for decoy in (home / "sorcar-local.json", tmp_path / "sorcar-local.json"):
        decoy.write_text(json.dumps({"url": "wss://127.0.0.1:1/ws", "token": "decoy"}), "utf-8")
    # A stale brand home: an endpoint file of a daemon that is gone, a
    # trusted-clients directory and a CA of its own.
    stale = home / mach.HOME_DIR
    (stale / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).mkdir(parents=True)
    stale_ca = "-----BEGIN CERTIFICATE-----\nstale\n-----END CERTIFICATE-----\n"
    (stale / "tls" / "ca.pem").write_text(stale_ca, encoding="utf-8")
    stale_port = _closed_port()

    def enroll_via(remote_home: Path | None) -> tuple[str, str]:
        server = start_sshd(sshd.config.parent, remote_home, home=home)
        try:
            return mach.enroll_over_ssh(server.login)
        finally:
            server.stop()

    live = (remote.url, (remote.home / "tls" / "ca.pem").read_text(encoding="utf-8"))
    # No KISS_HOME and no ~/<local brand> endpoint: the other brand's
    # directory is found; a KISS_HOME naming no endpoint file is skipped.
    assert enroll_via(None) == live
    assert enroll_via(missing) == live
    # The local brand's home wins over other brands' directories.
    (stale / "sorcar-local.json").write_text(
        json.dumps({"url": f"wss://127.0.0.1:{stale_port}/ws", "token": "stale"}), "utf-8",
    )
    assert enroll_via(None) == (f"https://127.0.0.1:{stale_port}", stale_ca)
    assert [p.name for p in (stale / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).iterdir()] == [
        tls_certs.ca_label(mach.kiss_home() / "tls" / "ca.pem") + ".pem",
    ]
    # KISS_HOME wins over both.
    assert enroll_via(remote.home) == live
    # A one-letter brand home (brand.py allows any name but "." and "..") is found too.
    (stale / "sorcar-local.json").unlink()
    (home / ".s10s").rename(home / ".x")
    assert enroll_via(None) == live
    # A home with no endpoint file anywhere: the usual error, naming what was searched.
    (home / ".x").unlink()
    with pytest.raises(mach.MachineError, match=r"no Sorcar daemon is running on .*: no .*/\.\*/"):
        enroll_via(None)
    # End to end: ssh discovery is a one-time step of the registration;
    # the link does not depend on it afterwards.
    (home / ".s10s").symlink_to(remote.home)
    server = start_sshd(sshd.config.parent, None, home=home)
    try:
        machine = mach.add_machine("brand", server.login)
        assert machine.url == remote.url and machine.ssh == server.login
        assert _titles(mach.fetch_history(machine, "alpha", "", 50)) == ["remote alpha"]
        (home / ".s10s").unlink()
        mach.close_links()
        assert mach.machine_status(machine) == "reachable"
    finally:
        mach.close_links()
        server.stop()


@needs_sshd
def test_ssh_machine_survives_a_daemon_restart_on_the_same_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enrolled CA lives on the machine's disk: a restarted daemon admits the link again.

    A daemon restarted on another port is at another address: the
    machine is unreachable until registered again (which enrolls the
    same CA file once more and records the new port).
    """
    home = tmp_path / "remote-home"
    daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "one"}])
    server = start_sshd(tmp_path / "sshd", home)
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    try:
        machine = mach.add_machine("box", server.login)
        assert machine == mach.Machine("box", daemon.url, ssh=server.login)
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["one"]
        link = mach.link_for(machine)
        port = daemon.port
        daemon.stop()
        with pytest.raises(mach.MachineError, match="did not answer|unreachable"):
            mach.fetch_history(machine, "", "", 50)
        daemon = start_remote_daemon(
            home, password="pw", seed=[{"chat_id": "c", "task": "two"}], port=port,
        )
        link._retry_at = 0.0  # the backoff has expired
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
        assert mach.fetch_history(machine, "", "", 50).rows[0]["machine_url"] == machine.url

        daemon.stop()
        daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "three"}])
        if daemon.port == port:
            pytest.skip("the OS handed the restarted daemon the same port")
        link._retry_at = 0.0
        with pytest.raises(mach.MachineError, match="unreachable"):
            mach.fetch_history(machine, "", "", 50)
        again = mach.add_machine("box", server.login)
        assert again == mach.Machine("box", daemon.url, ssh=server.login)
        assert _titles(mach.fetch_history(again, "", "", 50)) == ["three", "two", "one"]
        assert len(_trusted_files(daemon)) == 1
    finally:
        mach.close_links()
        daemon.stop()
        server.stop()


@needs_sshd
def test_ssh_tunnel_reaches_a_daemon_whose_address_is_unreachable(
    remote: RemoteDaemon, sshd: SshServer,
) -> None:
    """The daemon port is only reached through ssh: a blackhole URL host still answers.

    The machine's URL names ``192.0.2.1`` (TEST-NET, nothing routes
    there): a direct dial would hang, the forward to ``127.0.0.1:port``
    on the ssh host answers.  One tunnel serves every request and every
    link to the login; it is opened again after its ssh process died,
    and closed by ``close_links``.
    """
    enrolled = mach.add_machine("near", sshd.login)
    far = mach.Machine("far", f"https://192.0.2.1:{remote.port}", ssh=sshd.login)
    far.cert_file().write_bytes(enrolled.cert_file().read_bytes())
    mach.save_machines([enrolled, far])
    assert mach._tunnels == {}  # the registry changed: every tunnel was closed
    try:
        assert mach.machine_status(far) == "reachable"
        tunnel = mach._tunnels[sshd.login]
        assert tunnel.process is not None and tunnel.alive()
        pid = tunnel.process.pid
        link = mach.link_for(far)
        assert _titles(mach.fetch_history(link.machine, "alpha", "", 50)) == ["remote alpha"]
        assert _tunnel_peer_port(link) == tunnel.local_port
        assert _titles(mach.fetch_history(enrolled, "beta", "", 50)) == ["remote beta"]
        assert _tunnel_peer_port(mach.link_for(enrolled)) == tunnel.local_port
        assert mach._tunnels == {sshd.login: tunnel} and tunnel.process.pid == pid  # reused

        tunnel.process.kill()
        tunnel.process.wait()
        assert not tunnel.alive()
        assert _titles(mach.fetch_history(far, "shared", "", 50)) == ["shared task"]
        assert tunnel.process is not None and tunnel.process.pid != pid and tunnel.alive()
        assert _tunnel_peer_port(link) == tunnel.local_port
    finally:
        mach.close_links()
    assert mach._tunnels == {} and tunnel.process is None and not tunnel.alive()


_FAKE_SSH = '''#!__PYTHON__
"""Stand-in for ssh: ``-G`` names a blackhole host, ``-N -L`` relays to the daemon,
a command runs locally with KISS_HOME set, FAKE_SSH_FAIL refuses the forward,
FAKE_SSH_SLOW (seconds) is a slow login before the forward is up, FAKE_SSH_PIDFILE
records the forward's pid."""
import os
import socket
import subprocess
import sys
import threading
import time

args = sys.argv[1:]
if "-G" in args:
    print("hostname 192.0.2.1")
    sys.exit(0)
if "-N" not in args:
    env = dict(os.environ, KISS_HOME=os.environ["FAKE_SSH_HOME"])
    sys.exit(subprocess.run(args[-1], shell=True, env=env).returncode)
_host, local, _rhost, remote = args[args.index("-L") + 1].split(":")
if os.environ.get("FAKE_SSH_PIDFILE"):
    with open(os.environ["FAKE_SSH_PIDFILE"], "w") as handle:
        handle.write(str(os.getpid()))
if os.environ.get("FAKE_SSH_FAIL"):
    sys.stderr.write(f"bind [127.0.0.1]:{local}: Address already in use\\n"
                     "Could not request local forwarding.\\n")
    sys.exit(255)
if os.environ.get("FAKE_SSH_HANG"):
    time.sleep(60)
time.sleep(float(os.environ.get("FAKE_SSH_SLOW", "0")))


def pump(src, dst):
    try:
        while chunk := src.recv(65536):
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


server = socket.socket()
server.bind(("127.0.0.1", int(local)))
server.listen()
while True:
    client, _ = server.accept()
    try:
        upstream = socket.create_connection(("127.0.0.1", int(remote)))
    except OSError:
        client.close()
        continue
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pump, args=(upstream, client), daemon=True).start()
'''


fake_ssh_only = posix_only(
    "the fake ssh is a shebang script on PATH: Windows resolves ``ssh`` to "
    "System32's ssh.exe whatever PATH says, and runs its command through cmd.exe"
)


def _install_fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_home: Path) -> None:
    """Put :data:`_FAKE_SSH` first on PATH, its logins landing in *remote_home*."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ssh"
    script.write_text(_FAKE_SSH.replace("__PYTHON__", sys.executable), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SSH_HOME", str(remote_home))
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(tmp_path / "no-such-config"))


@fake_ssh_only
def test_ssh_tunnel_through_a_fake_ssh_relay_and_its_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``ssh`` on PATH that relays ``-L`` to a local daemon carries the whole link.

    The fake ssh reports a blackhole host, so the registered URL is
    unreachable directly; enrollment and every request go through it.
    A forward that fails is reported with ssh's last stderr line, one
    that never opens the port times out, and a missing ``ssh`` is
    named.
    """
    daemon = start_remote_daemon(
        tmp_path / "home", password="pw", seed=[{"chat_id": "c", "task": "one"}],
    )
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@far.example.org"
    try:
        machine = mach.add_machine("far", login)
        assert machine == mach.Machine("far", f"https://192.0.2.1:{daemon.port}", ssh=login)
        assert len(_trusted_files(daemon)) == 1
        link = mach.link_for(machine)
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        tunnel = mach._tunnels[login]
        assert tunnel.process is not None and tunnel.remote_port == daemon.port
        pid = tunnel.process.pid
        assert _tunnel_peer_port(link) == tunnel.local_port
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["one"]
        assert tunnel.process.pid == pid  # reused across requests
        assert mach.machine_status(machine) == "reachable" and tunnel.process.pid == pid

        tunnel.process.kill()
        tunnel.process.wait()
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert tunnel.process is not None and tunnel.process.pid != pid and tunnel.alive()

        monkeypatch.setenv("FAKE_SSH_FAIL", "1")
        tunnel.close()
        assert mach.machine_status(machine) == (
            f"far (ssh {login}) is unreachable: ssh {login} cannot forward its daemon port "
            f"{daemon.port}: Could not request local forwarding."
        )
        monkeypatch.delenv("FAKE_SSH_FAIL")
        monkeypatch.setenv("FAKE_SSH_HANG", "1")
        with pytest.raises(mach.MachineError, match="did not open the port forward in time"):
            mach.SshTunnel(login, daemon.port).open(timeout=1.0)
        monkeypatch.delenv("FAKE_SSH_HANG")
        assert mach.machine_status(machine) == "reachable"
        assert mach.remove_machine("far") is True
        assert mach._tunnels == {}
        monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
        with pytest.raises(mach.MachineError, match="ssh is not installed on this machine"):
            mach.SshTunnel(login, daemon.port).open(timeout=1.0)
    finally:
        mach.close_links()
        daemon.stop()


@fake_ssh_only
def test_slow_ssh_login_leaves_the_handshake_its_whole_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request's *timeout* is the handshake's, not the ssh login's (audit 2026-10-16 F1).

    The forward is opened inside ``_connect`` and ssh's login can take
    longer than the request budget (a bastion, a far host).  The
    budget used to start before the login, so a slow login left the
    WebSocket handshake a millisecond, it failed with "timed out
    during opening handshake" and the link backed off for
    ``DOWN_BACKOFF`` seconds although the forward was up.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@slow.example.org"
    try:
        machine = mach.add_machine("slow", login)
        mach.close_tunnels()  # the next request opens the forward again, slowly
        monkeypatch.setenv("FAKE_SSH_SLOW", "1.5")
        link = mach.link_for(machine)
        started = time.monotonic()
        assert link.request({"type": "ping"}, "pong", timeout=1.0)["type"] == "pong"
        assert time.monotonic() - started >= 1.5
        assert link._retry_at == 0.0
        assert mach._tunnels[login].alive()
    finally:
        mach.close_links()
        daemon.stop()


@fake_ssh_only
def test_failed_ssh_add_machine_closes_the_forward_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An SSH machine whose daemon does not answer leaves no ``ssh -N`` behind (audit 2026-10-16).

    The login succeeds (the endpoint file and ``trusted-clients`` are
    still there) but the daemon port is closed: every handshake fails
    until ``ENROLL_TIMEOUT``.  The forward those handshakes opened is
    closed with the failure instead of living until the next registry
    write.
    """
    mach.client_cert_pair()  # issues the local CA the enrollment sends
    remote_home = tmp_path / "remote"
    (remote_home / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).mkdir(parents=True)
    (remote_home / "tls" / tls_certs.CA_CERT_FILE).write_text(_local_ca_pem(), encoding="utf-8")
    (remote_home / mach.LOCAL_ENDPOINT_FILE).write_text(
        json.dumps({"url": f"wss://127.0.0.1:{_closed_port()}/ws", "token": "t"}), "utf-8",
    )
    _install_fake_ssh(tmp_path, monkeypatch, remote_home)
    pidfile = tmp_path / "ssh.pid"
    monkeypatch.setenv("FAKE_SSH_PIDFILE", str(pidfile))
    monkeypatch.setattr(mach, "ENROLL_TIMEOUT", 2.0)
    login = "ksen@down.example.org"
    with pytest.raises(mach.MachineError, match="refused the forwarded connection"):
        mach.add_machine("down", login)
    assert mach._tunnels == {}
    pid = int(pidfile.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the forward is reaped, not left running
    assert not mach.load_machines() and not list((mach.kiss_home() / "machines").glob("*down*"))


@fake_ssh_only
def test_failed_ssh_re_registration_keeps_the_shared_forward_and_its_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed re-registration leaves the login's forward to the links sharing it.

    ``box`` is registered and a stream is open on its link, so the
    login's ``ssh -N`` carries live traffic.  Registering ``box`` again
    then fails: the enrollment now answers with another CA (the
    machine's endpoint file names the local daemon's ``ca.pem``), so
    every handshake of the new pin fails TLS verification until
    ``ENROLL_TIMEOUT``.  The enrollment handshakes use a forward of
    their own, so the failure closes nothing the registered machine
    uses: its forward keeps its ssh process, the stream still answers,
    its link and registry entry are untouched.  The cleanup used to
    pop and kill the login's cached forward, disconnecting the stream.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@box.example.org"
    try:
        # The real ENROLL_TIMEOUT: the daemon admits the new CA at its
        # next TRUSTED_CLIENTS_POLL (2 s), later than that on a loaded host.
        machine = mach.add_machine("box", login)
        link = mach.link_for(machine)
        stream = link.open_connection()
        assert _pong(stream)
        tunnel = mach._tunnels[login]
        assert tunnel.process is not None
        pid = tunnel.process.pid

        # The machine's daemon now answers the enrollment with a CA that
        # did not issue its certificate: same port, wrong pin.
        monkeypatch.setattr(mach, "ENROLL_TIMEOUT", 2.0)
        wrong = tmp_path / "wrong"
        (wrong / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).mkdir(parents=True)
        (wrong / mach.LOCAL_ENDPOINT_FILE).write_text(json.dumps({
            "url": f"wss://127.0.0.1:{daemon.port}/ws", "token": "t",
            "ca": str(mach.kiss_home() / "tls" / tls_certs.CA_CERT_FILE),
        }, indent=2), encoding="utf-8")
        monkeypatch.setenv("FAKE_SSH_HOME", str(wrong))
        with pytest.raises(mach.MachineError, match="refused the TLS handshake"):
            mach.add_machine("box", login)

        assert mach._tunnels == {login: tunnel}
        assert tunnel.process is not None and tunnel.process.pid == pid and tunnel.alive()
        assert _pong(stream)  # the live stream was not disconnected
        assert mach.link_for(machine) is link
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert mach.load_machines() == [machine]
        assert not list((mach.kiss_home() / "machines").glob(".box*"))  # no staged pin left
    finally:
        mach.close_links()
        daemon.stop()
