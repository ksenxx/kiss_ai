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
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import machines as mach
from kiss.tests.conftest import find_free_port
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
    """Return a loopback port nothing listens on (8787 when it is free)."""
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", mach.DEFAULT_PORT)) != 0:
            return mach.DEFAULT_PORT
    return find_free_port()


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
    assert mach.REMOTE_ENDPOINT_COMMAND == (
        "sh -c 'for d in \"$KISS_HOME\" \"$HOME/" + mach.HOME_DIR + "\" \"$HOME\"/.*; do "
        'case "$d" in ""|*/.|*/..) continue;; esac; '
        'if [ -f "$d/sorcar-local.json" ]; then cat "$d/sorcar-local.json"; '
        'cat "$d/remote-url.json" 2>/dev/null; exit 0; fi; done; '
        "echo \"no $HOME/.*/sorcar-local.json\" >&2; exit 1'"
    )
    assert "!" not in mach.REMOTE_ENDPOINT_COMMAND  # csh expands history inside quotes
    assert mach._decode_two('{"a": 1}\n{"b": 2}\n') == ({"a": 1}, {"b": 2})
    assert mach._decode_two('{"a": 1}') == ({"a": 1}, {})
    assert mach._decode_two('{"a": 1} [1, 2] {"c": 3}') == ({"a": 1}, {})
    assert mach._decode_two("junk") == ({}, {}) and mach._decode_two("") == ({}, {})


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
    link._ws.send(json.dumps({"type": "ping"}))
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
    assert "unreachable" in mach.machine_status(wrong)
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


def _endpoint_token(daemon: RemoteDaemon) -> str:
    return str(json.loads((daemon.home / "sorcar-local.json").read_text(encoding="utf-8"))["token"])


def _pong(stream: ClientConnection) -> bool:
    """Ping the daemon over the stream socket *stream*; true when it answers within 10 s."""
    stream.send(json.dumps({"type": "ping"}))
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        frame = json.loads(stream.recv(timeout=deadline - time.monotonic()))
        if isinstance(frame, dict) and frame.get("type") == "pong":
            return True
    return False


@needs_sshd
def test_ssh_machine_tunnels_the_daemon_port_and_logs_in_with_its_token(
    remote: RemoteDaemon, remote_leaf: RemoteDaemon, sshd: SshServer,
) -> None:
    """``add user@host`` needs no password: the tunnel carries the remote's local token."""
    machine = mach.add_machine("", sshd.login)
    assert machine == mach.Machine("127.0.0.1", ssh=sshd.login)
    assert mach.load_machines() == [machine]
    assert machine.cert_file().read_text(encoding="utf-8") == (
        remote.home / "tls" / "ca.pem"
    ).read_text(encoding="utf-8")
    assert not list(machine.cert_file().parent.glob(".127.0.0.1.pem.*"))

    page = mach.fetch_history(machine, "", "", 50)
    assert _titles(page) == ["shared task", "remote beta", "remote alpha"]
    link = mach.link_for(machine)
    assert link.tunnel is not None and link.tunnel.is_open()
    assert {r["machine"] for r in page.rows} == {"127.0.0.1"}
    assert {r["machine_url"] for r in page.rows} == {link.tunnel.web_url}
    assert link.tunnel.url.startswith("https://127.0.0.1:") and link.tunnel.url != remote.url
    published = json.loads((remote.home / "remote-url.json").read_text(encoding="utf-8"))
    assert link.tunnel.web_url == (
        published.get("tunnel") or (published.get("lan") or [link.tunnel.url])[0]
    )
    assert link.tunnel.token == _endpoint_token(remote)
    assert "-L" in link.tunnel.command and "-N" in link.tunnel.command
    assert link.tunnel.log_file().exists()
    assert link.base_url() == link.tunnel.url  # an open tunnel is reused
    assert mach.machine_status(machine) == "reachable"
    # The merge works like for any other machine.
    leaf = mach.add_machine("leaf", remote_leaf.url)
    merged = mach.merged_history(mach.HistoryPage(), [machine, leaf], "", "", 0, 50)
    assert set(_titles(merged)) == {"shared task", "remote beta", "remote alpha", "leaf task"}
    assert mach.MachineLink(leaf).base_url() == remote_leaf.url

    # Evicted from the cache, the link is retired: a caller that fetched
    # it just before cannot re-open a tunnel nothing would close.
    tunnel_url = link.tunnel.url
    mach.close_links()
    assert not link.tunnel.is_open()
    assert link.tunnel.url == tunnel_url and link.tunnel.token  # still describe the last tunnel
    with pytest.raises(mach.MachineError, match="registry changed"):
        link.request({"type": "ping"}, "pong")
    with pytest.raises(mach.MachineError, match="registry changed"):
        link.base_url()
    assert not link.tunnel.is_open() and mach.link_for(machine) is not link
    assert mach.remove_machine("127.0.0.1") is True
    assert not machine.cert_file().exists() and not link.tunnel.log_file().exists()

    # An explicit name and the ssh:// spelling.
    box = mach.add_machine("box", "ssh://" + sshd.login)
    assert box == mach.Machine("box", ssh=sshd.login)
    assert _titles(mach.fetch_history(box, "alpha", "", 50)) == ["remote alpha"]
    mach.close_links()


@needs_sshd
def test_ssh_machine_failures_are_reported_and_store_nothing(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused login, a closed port, a missing daemon or ssh: a clear error, no entry."""
    with pytest.raises(mach.MachineError, match=r"ssh nobody-x@.* failed: .*Permission denied"):
        mach.add_machine("", f"nobody-x@127.0.0.1:{sshd.port}")
    with pytest.raises(mach.MachineError, match=r"ssh .* failed: .*refused"):
        mach.add_machine("", f"{sshd.login.split('@')[0]}@127.0.0.1:{_closed_port()}")
    assert mach.load_machines() == []
    assert not list((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob("127.0.0.1.*"))
    assert not list((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob(".127.0.0.1.*"))

    # A machine with no running daemon: its endpoint file is missing or junk.
    nothing = tmp_path / "nothing"
    nothing.mkdir()
    idle = start_sshd(sshd.config.parent, nothing)
    try:
        with pytest.raises(mach.MachineError, match="no Sorcar daemon is running on"):
            mach.add_machine("", idle.login)
        (nothing / "sorcar-local.json").write_text("junk", encoding="utf-8")
        with pytest.raises(mach.MachineError, match="no Sorcar daemon is running on"):
            mach.add_machine("", idle.login)
        (nothing / "sorcar-local.json").write_text(
            json.dumps({"url": "wss://127.0.0.1:1/ws", "token": ""}), encoding="utf-8",
        )
        with pytest.raises(mach.MachineError, match="no Sorcar daemon is running on"):
            mach.add_machine("", idle.login)
        # A daemon whose file is fine but whose port is closed: the tunnel
        # comes up, the daemon behind it is unreachable.
        (nothing / "sorcar-local.json").write_text(
            json.dumps({"url": f"wss://127.0.0.1:{_closed_port()}/ws", "token": "t"}),
            encoding="utf-8",
        )
        with pytest.raises(mach.MachineError, match="Cannot reach https://127.0.0.1:"):
            mach.add_machine("", idle.login)
        # The web UI link of an SSH machine: the daemon's published public
        # URL, else its first LAN URL, else the tunnel itself.
        (nothing / "sorcar-local.json").write_text(
            json.dumps({"url": f"wss://127.0.0.1:{remote.port}/ws",
                        "token": _endpoint_token(remote)}),
            encoding="utf-8",
        )
        tunnel = mach.SshTunnel(mach.Machine("idle", ssh=idle.login))
        for published, expected in (
            ({"tunnel": "https://x.trycloudflare.com", "lan": ["https://10.0.0.5:8787"]},
             "https://x.trycloudflare.com"),
            ({"local": "https://localhost:8787", "lan": ["https://10.0.0.5:8787"]},
             "https://10.0.0.5:8787"),
            ({"local": "https://localhost:8787", "lan": []}, ""),
            (None, ""),
        ):
            url_file = nothing / "remote-url.json"
            if published is None:
                url_file.unlink()
            else:
                url_file.write_text(json.dumps(published), encoding="utf-8")
            tunnel.close()
            tunnel.open()
            assert tunnel.web_url == (expected or tunnel.url)
        assert tunnel.token == _endpoint_token(remote)
        assert f"127.0.0.1:{tunnel.url.rsplit(':', 1)[1]}:127.0.0.1:{remote.port}" in tunnel.command
        tunnel.close()
        # The forward targets the host the daemon listens on, IPv6 included.
        (nothing / "sorcar-local.json").write_text(
            json.dumps({"url": f"wss://[::1]:{remote.port}/ws", "token": "t"}),
            encoding="utf-8",
        )
        tunnel.open()
        assert f"127.0.0.1:{tunnel.url.rsplit(':', 1)[1]}:[::1]:{remote.port}" in tunnel.command
        with pytest.raises(mach.MachineError, match="Cannot reach"):  # nothing on ::1
            mach.pin_certificate(tunnel.url)
        tunnel.close()
    finally:
        idle.stop()
    assert mach.load_machines() == []

    # The cached link backs off after a failure: no ssh login is retried
    # before DOWN_BACKOFF, and the same message is reported.
    bad = mach.Machine("bad", ssh=f"nobody-x@127.0.0.1:{sshd.port}")
    assert "Permission denied" in mach.machine_status(bad)
    link = mach.link_for(bad)
    with pytest.raises(mach.MachineError, match="Permission denied"):
        mach.fetch_history(bad, "", "", 50)
    assert link._retry_at > time.monotonic()
    started = time.monotonic()
    with pytest.raises(mach.MachineError, match="Permission denied"):
        link.base_url()
    with pytest.raises(mach.MachineError, match="Permission denied"):
        mach.fetch_history(bad, "", "", 50)
    assert time.monotonic() - started < 0.5

    # ssh slower than the budget, and no ssh at all.
    tunnel = mach.SshTunnel(bad)
    with pytest.raises(mach.MachineError, match="did not answer in time"):
        tunnel.open(timeout=0.0)
    assert not tunnel.is_open()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    with pytest.raises(mach.MachineError, match="ssh is not installed"):
        tunnel.open()
    # Not covered: ``ssh -N -L`` logging in but never binding the local
    # port ("did not forward port ... within"); OpenSSH binds it before
    # the session is up, so only a fake ssh could produce that.


@needs_sshd
def test_ssh_machine_finds_the_daemon_whatever_its_brand_home(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path,
) -> None:
    """The endpoint is read from ``$KISS_HOME``, else ``~/<local brand>``, else any ``~/.*``.

    A remote re-installed under another brand moves its state directory
    (``~/.kiss`` to ``~/.s10s`` or back); a registered machine must stay
    reachable through such a change without any remote configuration.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".s10s").symlink_to(remote.home)  # another brand's daemon, its endpoint file live
    stale = {"url": f"wss://127.0.0.1:{_closed_port()}/ws", "token": "stale"}
    missing = tmp_path / "missing"
    # ``~/.`` and ``~/..`` are not brand homes: neither file is ever read.
    for decoy in (home / "sorcar-local.json", tmp_path / "sorcar-local.json"):
        decoy.write_text(json.dumps({"url": "wss://127.0.0.1:1/ws", "token": "decoy"}), "utf-8")

    def endpoint_via(remote_home: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
        server = start_sshd(sshd.config.parent, remote_home, home=home)
        try:
            tunnel = mach.SshTunnel(mach.Machine("brand", ssh=server.login))
            return tunnel._read_endpoint(time.monotonic() + mach.SSH_TIMEOUT)
        finally:
            server.stop()

    published = json.loads((remote.home / "remote-url.json").read_text(encoding="utf-8"))
    live = (json.loads((remote.home / "sorcar-local.json").read_text("utf-8")), published)
    # No KISS_HOME and no ~/<local brand>: the other brand's directory is found,
    # its published URLs with it; a KISS_HOME naming no endpoint file is skipped.
    assert endpoint_via(None) == live
    assert endpoint_via(missing) == live
    # The local brand's home wins over other brands' directories.
    (home / mach.HOME_DIR).mkdir()
    (home / mach.HOME_DIR / "sorcar-local.json").write_text(json.dumps(stale), encoding="utf-8")
    assert endpoint_via(None) == (stale, {})
    # KISS_HOME wins over both.
    assert endpoint_via(remote.home) == live
    # A one-letter brand home (brand.py allows any name but "." and "..") is found too.
    (home / mach.HOME_DIR / "sorcar-local.json").unlink()
    (home / ".s10s").rename(home / ".x")
    assert endpoint_via(None) == live
    # A home with no endpoint file anywhere: the usual error, naming what was searched.
    (home / ".x").unlink()
    with pytest.raises(mach.MachineError, match=r"no Sorcar daemon is running on .*: no .*/\.\*/"):
        endpoint_via(None)
    # End to end: the machine registered while the remote was one brand
    # keeps working after it moved to the other.
    (home / ".s10s").symlink_to(remote.home)
    server = start_sshd(sshd.config.parent, None, home=home)
    try:
        machine = mach.add_machine("brand", server.login)
        assert _titles(mach.fetch_history(machine, "alpha", "", 50)) == ["remote alpha"]
        (home / ".s10s").unlink()
        (home / mach.HOME_DIR).rmdir()
        (home / mach.HOME_DIR).symlink_to(remote.home)
        mach.close_links()
        assert mach.machine_status(machine) == "reachable"
    finally:
        mach.close_links()
        server.stop()


@needs_sshd
def test_ssh_machine_recovers_from_a_daemon_restart_and_a_dead_tunnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted daemon (new port, new token) and a killed ssh are found again."""
    home = tmp_path / "remote-home"
    daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "one"}])
    server = start_sshd(tmp_path / "sshd", home)
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    try:
        machine = mach.add_machine("box", server.login)
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["one"]
        link = mach.link_for(machine)
        assert link.tunnel is not None
        first_url, first_token = link.tunnel.url, link.tunnel.token

        daemon.stop()
        with pytest.raises(mach.MachineError, match="did not answer|unreachable"):
            mach.fetch_history(machine, "", "", 50)
        assert not link.tunnel.is_open()  # closed with the failure: re-read on retry
        daemon = start_remote_daemon(
            home, password="pw", seed=[{"chat_id": "c", "task": "two"}],
        )
        assert _endpoint_token(daemon) != first_token
        link._retry_at = 0.0  # the backoff has expired
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
        assert link.tunnel.token == _endpoint_token(daemon)
        assert link.tunnel.url != first_url
        assert mach.fetch_history(machine, "", "", 50).rows[0]["machine_url"] == link.tunnel.url

        # The ssh process dies (network blip): the next request re-opens
        # the tunnel at once, with a new socket.
        assert link.tunnel.process is not None
        link.tunnel.process.kill()
        link.tunnel.process.wait()
        assert not link.tunnel.is_open()
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
        assert link.tunnel.is_open()
    finally:
        mach.close_links()
        daemon.stop()
        server.stop()


@needs_sshd
def test_ssh_machine_retries_at_once_when_the_tunnel_token_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon restarted on the same port under a living tunnel is found without a backoff.

    The ``ssh -N`` forward connects on demand, so it outlives the
    daemon it points at; the token the link read when it opened the
    tunnel is then stale and the restarted daemon answers
    ``auth_required``.  The link re-reads the endpoint file and
    authenticates again right away, whether a stream
    (:meth:`MachineLink.open_connection`) or a request runs into the
    stale token first, and a request socket cached from before the
    restart is dropped with the old tunnel.  A token the daemon
    rejects twice (the file itself is wrong) still backs off.
    """
    home = tmp_path / "remote-home"
    daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "one"}])
    server = start_sshd(tmp_path / "sshd", home)
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    endpoint_file = home / "sorcar-local.json"
    try:
        machine = mach.add_machine("box", server.login)
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["one"]
        link = mach.link_for(machine)
        assert link.tunnel is not None and link.tunnel.process is not None
        first_ssh, first_token = link.tunnel.process.pid, link.tunnel.token
        assert first_token == _endpoint_token(daemon)

        port = daemon.port
        daemon.stop()
        daemon = start_remote_daemon(
            home, password="pw", seed=[{"chat_id": "c", "task": "two"}], port=port,
        )
        assert daemon.port == port and _endpoint_token(daemon) != first_token
        assert link.tunnel.is_open()  # the forward survived the restart

        # A stream (a tab on the machine) runs into the stale token
        # first: it gets a socket on a new tunnel, without a backoff.
        started = time.monotonic()
        stream = link.open_connection()
        assert time.monotonic() - started < mach.DOWN_BACKOFF
        assert link._retry_at == 0.0  # no backoff was started
        assert link.tunnel.token == _endpoint_token(daemon)
        assert link.tunnel.process is not None and link.tunnel.process.pid != first_ssh
        assert _pong(stream)
        # The request socket cached before the restart went with the
        # old tunnel: the next request connects anew instead of failing
        # on it (and closing the new tunnel under the stream).
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
        assert link._retry_at == 0.0
        assert _pong(stream)
        stream.close()
        assert mach.fetch_history(machine, "", "", 50).rows[0]["machine_url"] == link.tunnel.url

        # A token the file itself gets wrong is rejected twice: the
        # second rejection backs off like any failure.
        endpoint = json.loads(endpoint_file.read_text(encoding="utf-8"))
        endpoint_file.write_text(json.dumps(endpoint | {"token": "0" * 64}), encoding="utf-8")
        link.close()
        with pytest.raises(mach.MachineError, match="rejected the password"):
            mach.fetch_history(machine, "", "", 50)
        assert link._retry_at > time.monotonic()
        assert not link.tunnel.is_open()
        with pytest.raises(mach.MachineError, match="rejected the password"):
            mach.fetch_history(machine, "", "", 50)  # within the backoff: not retried

        # The file is right again: the backoff over, the link recovers.
        endpoint_file.write_text(json.dumps(endpoint), encoding="utf-8")
        link._retry_at = 0.0
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
    finally:
        mach.close_links()
        daemon.stop()
        server.stop()


@needs_sshd
def test_ssh_tunnel_ignores_connection_multiplexing(
    sshd: SshServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ControlMaster in the user's config the tunnel is still its own process."""
    control = tmp_path / "control"
    control.mkdir()
    config = tmp_path / "ssh_config"
    config.write_text(
        sshd.config.read_text(encoding="utf-8")
        + f"  ControlMaster auto\n  ControlPath {control}/%C\n  ControlPersist 60\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(config))
    machine = mach.Machine("mux", ssh=sshd.login)
    pinning = mach.MachineLink(machine)
    machine.cert_file().write_text(mach.pin_certificate(pinning.base_url()), encoding="utf-8")
    pinning.close()
    assert pinning.tunnel is not None and not pinning.tunnel.is_open()
    link = mach.link_for(machine)
    try:
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert link.tunnel is not None and link.tunnel.is_open()
        assert list(control.iterdir()) == []  # no master socket was created
    finally:
        mach.close_links()
        machine.cert_file().unlink(missing_ok=True)
