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

import asyncio
import dis
import json
import logging
import os
import random
import shlex
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import urlsplit

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect

from kiss.agents.sorcar import machines as mach
from kiss.core import tls_certs
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import locked_call
from kiss.server import sidebar_panels
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.server.web_server import _get_local_ips
from kiss.tests.agents.sorcar.test_machine_link_no_resend_wave2 import _Daemon
from kiss.tests.conftest import nproc_limit_lowered_to_one, posix_only, thread_start_can_be_starved
from kiss.tests.local_ws import make_test_tls
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.pause import Pause
from kiss.tests.ssh_server import SSHD, SshServer, start_sshd

needs_sshd = pytest.mark.skipif(not SSHD, reason="no sshd on this host")

ID_SHARED = "a" * 32

#: How much earlier than its timeout a timed wait may end: on Windows the
#: kernel expires a wait on its 15.6 ms timer tick while ``time.monotonic()``
#: reads the high-resolution counter, so a 2 s wait can measure 1.995 s.
WAIT_SLACK = 0.02


def _clear_registry_and_pins() -> None:
    """Empty the registry and drop every pinned certificate.

    Other test modules of this process register a ``box`` too and only
    reset the registry on teardown; their stale ``box.pem`` would fail
    the assertions here that a rejected enrollment leaves no pin.
    """
    mach.save_machines([])
    mach._last_pages.clear()
    certs_dir = mach.kiss_home() / mach.MACHINE_CERTS_DIR
    if certs_dir.is_dir():
        for pin in certs_dir.iterdir():
            pin.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """Every test starts and ends with no registered machine and no pin."""
    _clear_registry_and_pins()
    yield
    _clear_registry_and_pins()


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


def _endpoint_json(url: str, **extra: str) -> str:
    """An endpoint file as a running daemon writes it: its pid (this process's) alive."""
    return json.dumps({"url": url, "token": "t", "pid": os.getpid(), **extra}, indent=2)


def _hold_send_thread(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook for new threads: hold the link's write helper for 50 ms, as a loaded box would.

    Installed with ``threading.setprofile``; every thread started while it
    is installed runs it once, on its first event, and drops it again.
    """
    sys.setprofile(None)
    if threading.current_thread().name == "sorcar-local-send":
        time.sleep(0.05)


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
    # urlsplit itself refuses a bracket that holds no IPv6 address, before any port is read.
    for bad in ("[abc", "https://[abc]"):
        with pytest.raises(mach.MachineError, match="not a machine address"):
            mach.normalize_url(bad)


def test_ssh_addresses_are_recognised_normalised_and_named() -> None:
    """``user@host[:port]`` and ``ssh://`` forms are SSH machines; the host names them."""
    assert mach.is_ssh_address("ksen@10.0.0.5") and mach.is_ssh_address("ssh://box")
    assert not mach.is_ssh_address("https://box") and not mach.is_ssh_address("box:8787")
    assert mach.normalize_ssh("ksen@10.0.0.5") == "ksen@10.0.0.5"
    assert mach.normalize_ssh(" ssh://ksen@box.example.org:2222/") == "ksen@box.example.org:2222"
    assert mach.normalize_ssh("ssh://box") == "box"
    assert mach.normalize_ssh("ksen@[::1]:22") == "ksen@[::1]:22"
    # The last one parses, but not once its host has lost its brackets.
    for bad in ("ksen@", "ssh://", "ssh://box/path", "ksen@box:notaport", "ksen@[abc", "[@[v1.x]"):
        with pytest.raises(mach.MachineError, match="not an SSH address"):
            mach.normalize_ssh(bad)
    assert mach.default_name("ksen@10.0.0.5") == "10.0.0.5"
    assert mach.default_name("ssh://ksen@box.example.org:2222") == "box.example.org"
    assert mach.default_name("https://10.0.0.5:8787") == "10.0.0.5"
    assert mach.default_name("box:9787") == "box"
    assert mach.default_name("ssh://ksen@my_box") == "my_box"
    assert mach.default_name("@@") == mach.default_name("[abc") == "machine"
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
        "sh -c 'for s in 1 0; do "
        'for d in "$KISS_HOME" "$HOME/' + mach.HOME_DIR + '" "$HOME"/.*; do '
        'case "$d" in ""|*/.|*/..) continue;; esac; '
        'if [ -f "$d/sorcar-local.json" ]; then '
        'p=$(sed -n "s/^ *\\"pid\\": *\\([0-9]*\\),*$/\\1/p" "$d/sorcar-local.json"); '
        '[ -n "$p" ] && kill -0 "$p" 2>/dev/null || continue; '
        '[ $s = 1 ] && m=$(ps -p "$p" -o etime= 2>/dev/null | awk -F"[-:]" "{ '
        'm = \\$(NF-1) + 1; if (NF > 2) m += 60 * \\$(NF-2); if (NF > 3) m += 1440 * \\$(NF-3); '
        'print m }" 2>/dev/null) && [ -n "$m" ] && '
        '[ -n "$(find "$d/sorcar-local.json" -mmin "+$m" 2>/dev/null)" ] && continue; '
        '[ -d "$d/tls/trusted-clients" ] || { '
        'echo "its daemon predates certificate enrollment: upgrade it" >&2; exit 3; }; '
        't=$(mktemp -u "$d/tls/trusted-clients/.NAME.XXXXXXXXXX") && (set -C; cat > "$t") && '
        'mv -f "$t" "$d/tls/trusted-clients/NAME.pem" '
        '|| exit 2; cat "$d/sorcar-local.json"; echo; '
        'c=$(sed -n "s/^ *\\"ca\\": *\\"\\(.*\\)\\",*$/\\1/p" "$d/sorcar-local.json"); '
        'cat "${c:-$d/tls/ca.pem}"; exit 0; fi; done; done; '
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
    # Valid JSON of the wrong shape, down to a ``machines`` that is no list.
    for junk in ("[]", '{"machines": null}', '{"machines": 5}'):
        mach.machines_file().write_text(junk, encoding="utf-8")
        assert mach.load_machines() == []
    assert mach.remove_machine("nobody") is False


def test_local_machine_name_is_the_short_hostname() -> None:
    """The local label is the host's short name (never empty)."""
    name = mach.local_machine_name()
    assert name and "." not in name


def test_machine_on_host_finds_the_machine_a_synced_row_ran_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host name maps to the machine whose daemon reported it, else by name or address."""
    by_ip = mach.Machine(name="10.0.0.7", url="https://10.0.0.7:8787")
    by_dns = mach.Machine(name="lab", url="https://gpu.example.com:8787")
    machines = [by_ip, by_dns]
    mach._hosts.pop(by_ip.name, None)
    mach._hosts.pop(by_dns.name, None)
    assert mach.machine_on_host("gpu-box", machines) is None
    mach.note_host(by_ip, "gpu-box")
    try:
        # Saved, so a restarted daemon still knows it while the machine is down.
        assert mach._saved_hosts()["10.0.0.7"] == "gpu-box"
        assert mach.machine_on_host("GPU-box", machines) is by_ip
        assert mach.machine_on_host("lab", machines) is by_dns
        assert mach.machine_on_host("gpu", machines) is by_dns
        assert mach.machine_on_host(" ", machines) is None
        rows: list[dict[str, Any]] = [{"origin": "gpu-box"}, {"origin": "elsewhere"}, {}]
        mach.label_synced_rows(rows, machines)
        assert rows == [
            {"machine": "10.0.0.7", "machine_url": "https://10.0.0.7:8787"}, {}, {},
        ]
        # A home whose hosts file cannot be locked keeps the name in memory only.
        home = tmp_path / "home"
        (home / mach.HOSTS_FILE).with_suffix(".lock").mkdir(parents=True)
        monkeypatch.setenv("KISS_HOME", str(home))
        mach.note_host(by_dns, "lab-host")
        assert mach.machine_on_host("lab-host", machines) is by_dns
        assert not (home / mach.HOSTS_FILE).exists()
        # The same observation once the file can be locked saves the name after all.
        (home / mach.HOSTS_FILE).with_suffix(".lock").rmdir()
        mach.note_host(by_dns, "lab-host")
        assert mach._saved_hosts() == {"lab": "lab-host"}
        assert by_dns.name not in mach._unsaved_hosts
    finally:
        mach._hosts.pop(by_ip.name, None)
        mach._hosts.pop(by_dns.name, None)
        mach._unsaved_hosts.discard(by_dns.name)


_NOTE_HOST_IN_A_PROCESS = """
import sys, time
from pathlib import Path
from kiss.agents.sorcar import machines as mach
name, host, go = sys.argv[1], sys.argv[2], Path(sys.argv[3])
print("ready", flush=True)
while not go.exists():
    time.sleep(0.01)
mach.note_host(mach.Machine(name, "https://" + name + ".example.org:8787"), host)
"""


def test_host_names_noted_at_once_all_survive(tmp_path: Path) -> None:
    """Host names noted by two threads, or two processes, at once are all saved.

    ``note_host`` wrote this process's whole mapping: a first writer
    delayed between serialising and writing put its older mapping over
    a second writer's newer file, and the second machine's name was
    lost from disk for good, as a later identical observation finds it
    in memory and writes nothing (audit S5/F2).  A ``/sync_history_and_memory``
    process noting a machine beside the daemon lost names the same way.
    The thread schedule is forced with a profile hook that holds the
    first writer at its write while the second one notes (now waiting
    for the file lock, so the hold times out); the processes load the
    module first and note on a signal, so each starts from the same
    saved mapping.
    """
    machines = [
        mach.Machine("first", "https://first.example.org:8787"),
        mach.Machine("second", "https://second.example.org:8787"),
        mach.Machine("p1", "https://p1.example.org:8787"),
        mach.Machine("p2", "https://p2.example.org:8787"),
    ]
    for machine in machines:
        mach._hosts.pop(machine.name, None)
    write_code = mach.atomic_write_text.__code__
    writing, noted = threading.Event(), threading.Event()

    def hold_the_first_write(frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is write_code and not writing.is_set():
            writing.set()
            noted.wait(2.0)

    second = threading.Thread(target=mach.note_host, args=(machines[1], "host-second"))
    try:
        threading.setprofile(hold_the_first_write)
        try:
            first = threading.Thread(target=mach.note_host, args=(machines[0], "host-first"))
            first.start()
        finally:
            threading.setprofile(None)
        assert writing.wait(30), "the first writer never reached its write"
        second.start()
        second.join(30)
        noted.set()
        first.join(30)
        assert not first.is_alive() and not second.is_alive()
        saved = mach._saved_hosts()
        assert {"first": "host-first", "second": "host-second"}.items() <= saved.items(), saved

        go = tmp_path / "go"
        script = [sys.executable, "-c", _NOTE_HOST_IN_A_PROCESS]
        processes = [
            subprocess.Popen(
                [*script, name, "host-" + name, str(go)], stdout=subprocess.PIPE, text=True,
            )
            for name in ("p1", "p2")
        ]
        try:
            for process in processes:
                assert process.stdout is not None and process.stdout.readline() == "ready\n"
            go.touch()
            for process in processes:
                assert process.wait(60) == 0
        finally:
            for process in processes:
                process.kill()
        saved = mach._saved_hosts()
        assert {
            "first": "host-first", "second": "host-second", "p1": "host-p1", "p2": "host-p2",
        }.items() <= saved.items(), saved
        # Reloaded, every machine's synced rows name it although none answers.
        rows: list[dict[str, Any]] = [{"origin": "host-" + m.name} for m in machines]
        mach.label_synced_rows(rows, machines)
        assert [row["machine"] for row in rows] == ["first", "second", "p1", "p2"]
    finally:
        threading.setprofile(None)
        noted.set()
        for machine in machines:
            mach._hosts.pop(machine.name, None)


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

    # Re-adding the same name replaces the entry (no duplicates), and a pin
    # that is no longer text: registering again is how a broken pin is mended.
    machine.cert_file().write_bytes(b"\xff\xfe no certificate")
    again = mach.add_machine("box", remote.url, "secret-1")
    assert mach.load_machines() == [again]
    assert mach.machine_status(again) == "reachable"

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
    for name in ("bad name!", "box\n"):  # the tool passes a model's name on unstripped
        with pytest.raises(mach.MachineError, match="not a valid machine name"):
            mach.add_machine(name, remote.url, "secret-1")
    # A malformed address is refused the same way, named or not, by the panel as by the tool.
    for name, address in (("box", "[abc"), ("", "[abc"), ("", "ksen@[abc"), ("", "[@[v1.x]")):
        with pytest.raises(mach.MachineError, match="is not a(n SSH| machine) address"):
            mach.add_machine(name, address)
    assert "not a machine address" in sidebar_panels.machine_action("add", "", "[abc")
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


def _send_in_pieces(
    listener: socket.socket, ctx: ssl.SSLContext, reply: bytes, piece: int, cut_short: list[bool],
) -> None:
    """Send *reply* to *listener*'s next TLS connection, *piece* bytes every 10 ms.

    *cut_short* gets whether the peer was gone before the whole reply was sent.
    """
    try:
        conn, _ = listener.accept()
        with ctx.wrap_socket(conn, server_side=True) as tls:
            tls.recv(4096)
            for start in range(0, len(reply), piece):
                tls.sendall(reply[start:start + piece])
                time.sleep(0.01)
        cut_short.append(False)
    except OSError:
        cut_short.append(True)


def test_pin_certificate_gives_up_on_a_peer_that_is_no_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever answers ``/ca.crt`` in a daemon's place is a ``MachineError``, in bounded time.

    ``http.client``'s own errors (a TLS peer that speaks no HTTP, a host
    it refuses) went past the callers' ``MachineError`` handlers.  A peer
    sending a few bytes at a time held the fetch while it kept sending
    (every read had the whole timeout), as did one a redirect led to,
    and a body was read to its end whatever its size.  The trickled
    replies last a minute: the fetch must hang up on them long before.
    """
    certfile, keyfile, _ca = make_test_tls(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    monkeypatch.setattr(mach, "REQUEST_TIMEOUT", 2.0)
    ok, moved = b"HTTP/1.1 200 OK\r\n", b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\n"
    peers = [  # the reply, its piece size, the error, whether the fetch hangs up on it
        (b"220 mail.example ESMTP ready\r\n", 64, "Cannot reach .*ESMTP", False),
        (ok + b"X-Pad: y\r\n" * 600, 1, "Cannot reach .*timed out", True),
        (ok + b"Content-Length: 9000\r\n\r\n" + b"x" * 6000, 1, "Cannot reach .*timed out", True),
        (ok + b"\r\n" + b"x" * (mach.CERT_MAX_BYTES + 1), 1 << 21, "larger than", False),
        # Not followed: the leaf is asked for next, of a peer that takes one connection.
        (moved + b"Location: http://127.0.0.1:%d/\r\n\r\n" % _closed_port(), 64,
         "Cannot read the TLS certificate", False),
    ]
    with pytest.raises(mach.MachineError, match="Cannot reach"):
        mach.pin_certificate("https://not a host:8787")
    for reply, piece, error, hung_up_on in peers:
        cut_short: list[bool] = []
        with socket.create_server(("127.0.0.1", 0)) as listener:
            listener.settimeout(30)
            server = threading.Thread(
                target=_send_in_pieces, args=(listener, ctx, reply, piece, cut_short), daemon=True,
            )
            server.start()
            try:
                with pytest.raises(mach.MachineError, match=error):
                    mach.pin_certificate(f"https://127.0.0.1:{listener.getsockname()[1]}")
            finally:
                server.join(timeout=90)
        assert cut_short == [hung_up_on], reply[:40]


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
    assert link._health.retry_at > time.monotonic()
    link._health.retry_at = 0.0
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    # An already expired deadline fails even when the reply is buffered,
    # and with the reply's timeout when the thread that writes the
    # command waits for a core first (a loaded box: full-suite run of
    # 2026-11-05), not with a write failure that blames the daemon.
    assert link._ws is not None
    # (A raw frame on the link's socket must carry the machine's password
    # like the link's own frames do, or the daemon closes the socket.)
    link._ws.send(json.dumps({"type": "ping", "password": "secret-1"}))
    time.sleep(0.3)
    threading.setprofile(_hold_send_thread)
    try:
        with pytest.raises(mach.MachineError, match="no pong within 0s"):
            link.request({"type": "ping"}, "pong", timeout=0.0)
    finally:
        threading.setprofile(None)
    assert link._health.retry_at > time.monotonic()
    link._health.retry_at = 0.0
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"

    dead = mach.Machine("dead", f"https://127.0.0.1:{_closed_port()}")
    dead.cert_file().write_text(machine.cert_file().read_text(), encoding="utf-8")
    mach.save_machines([machine, dead])
    with pytest.raises(mach.MachineError, match="unreachable"):
        mach.fetch_history(dead, "", "", 50)
    assert "unreachable" in mach.machine_status(dead)
    # The cached link backs off: the next attempt fails at once, with the
    # same message, without touching the network (a closed port would
    # answer instantly anyway, so the backoff deadline is what proves it).
    dead_link = mach.link_for(dead)
    assert dead_link._health.retry_at > time.monotonic()
    dead_link._health.retry_at = time.monotonic() + 3600
    started = time.monotonic()
    with pytest.raises(mach.MachineError, match="unreachable"):
        mach.fetch_history(dead, "", "", 50)
    assert time.monotonic() - started < 0.5
    dead_link._health.retry_at = 0.0  # expired: the next attempt connects again
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


def _request_a_reply_that_never_comes(link: mach.MachineLink, timeout: float) -> None:
    """Thread body: a request that holds the link's lock for its whole *timeout*."""
    with pytest.raises(mach.MachineError, match="did not answer"):
        link.request({"type": "ping"}, "never-sent", timeout=timeout)


def test_a_stream_handshake_does_not_wait_behind_a_request(remote: RemoteDaemon) -> None:
    """A stream connects beside a request in flight on the shared socket.

    ``open_connection`` counted its handshake in and out under the
    link's lock, which a request holds for its whole exchange: a chat
    tab's stream could not even start its handshake until an unrelated
    request on the machine had returned (its reply budget, up to the
    request's timeout), and its own timeout did not bound that wait.
    """
    machine = mach.add_machine("box", remote.url, "secret-1")
    link = mach.link_for(machine)
    assert link.request({"type": "ping"}, "pong")["type"] == "pong"
    waiting = threading.Thread(target=_request_a_reply_that_never_comes, args=(link, 6.0))
    waiting.start()
    try:
        time.sleep(0.3)  # the request is in its exchange, holding the lock
        assert waiting.is_alive()
        stream = link.open_connection(timeout=5.0)
        stream.close()
        assert waiting.is_alive(), "the stream was opened only after the request had returned"
    finally:
        waiting.join(30)
        link._health.retry_at = 0.0


def _opens_a_stream(
    link: mach.MachineLink, streams: list[ClientConnection], errors: list[Exception],
) -> None:
    """Thread body: ``open_connection`` on *link*, its result or error collected."""
    try:
        streams.append(link.open_connection(timeout=5.0))
    except mach.MachineError as exc:
        errors.append(exc)


def test_a_retired_link_never_starts_a_stream_handshake(remote: RemoteDaemon) -> None:
    """A retirement that returned leaves no stream handshake to start afterwards.

    ``open_connection`` checks the link's availability outside the
    dial lock, then counts its handshake in under it: a ``retire``
    landing between the two found no handshake to wait for and
    returned, and the handshake went on to open its socket (for an
    SSH machine, its tunnel) after ``close_links`` had swept them.
    The retirement check is repeated under the dial lock, where the
    count is: a handshake counted in is one the retirement waits for,
    and one the retirement no longer waits for never starts.
    """
    machine = mach.add_machine("box", remote.url, "secret-1")
    link = mach.link_for(machine)
    streams: list[ClientConnection] = []
    errors: list[Exception] = []
    opening = threading.Thread(target=_opens_a_stream, args=(link, streams, errors))
    # Held at the entry of ``locked_call(self._idle, self._start_dial)``:
    # the availability check has passed, the handshake is not counted.
    with Pause(
        locked_call.__code__, lambda local: getattr(local["fn"], "__name__", "") == "_start_dial",
    ) as held:
        opening.start()
        assert held.paused.wait(30), "open_connection about to count its handshake in"
        started = time.monotonic()
        link.retire()
        assert time.monotonic() - started < 5.0, "retire waited for a handshake not started"
    opening.join(30)
    assert not opening.is_alive()
    for stream in streams:
        stream.close()
    assert streams == [], "a stream handshake started on the retired link"
    assert len(errors) == 1 and "registry changed" in str(errors[0])
    assert not link._dialing


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="Thread.start cannot be starved")
def test_a_dial_out_of_threads_is_a_machine_error(remote: RemoteDaemon) -> None:
    """Out of threads at the WebSocket client's receive thread, a dial fails as any other does.

    The thread's ``RuntimeError`` escaped ``_dial``, whose callers answer
    ``MachineError`` only.  ``RLIMIT_NPROC`` is lowered for exactly that
    start by a :class:`Pause` hook (instrumentation, nothing replaced) when
    the thread is built; the client closes the socket it had opened.
    """
    link = mach.link_for(mach.add_machine("box", remote.url, "secret-1"))
    sockets: list[socket.socket] = []

    def starve_the_reader(frame_locals: dict[str, Any]) -> bool:
        target = frame_locals["target"]
        if getattr(target, "__name__", "") == "recv_events":
            sockets.append(target.__self__.socket)
            starved.enter_context(nproc_limit_lowered_to_one())
        return False

    with ExitStack() as starved, Pause(threading.Thread.__init__.__code__, starve_the_reader):
        with pytest.raises(mach.MachineError, match="the connection could not start"):
            link.open_connection(timeout=5.0)
    assert [sock.fileno() for sock in sockets] == [-1], "the socket was left open"
    assert not link._dialing and link._health.retry_at > time.monotonic()
    link._health.retry_at = 0.0
    link.open_connection(timeout=5.0).close()


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="Thread.start cannot be starved")
def test_a_lookup_out_of_threads_is_an_unanswered_one() -> None:
    """Out of threads, a lookup waited for on a thread of its own ends as one nobody answered.

    ``is_local_machine`` and ``pin_certificate`` run what a resolver may
    never answer on a helper thread (audit 2026-11-12, B4); the
    ``RuntimeError`` of a thread that cannot start does not escape them.
    """
    refused = "Cannot reach https://127.0.0.1:1: can't start new thread"
    with nproc_limit_lowered_to_one():
        assert mach.is_local_machine("stranger.invalid") is False  # its full name?
        assert mach.is_local_machine("203.0.113.9") is False  # an address of its name?
        with pytest.raises(mach.MachineError, match=refused):
            mach.pin_certificate("https://127.0.0.1:1")


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="Thread.start cannot be starved")
def test_a_frame_writer_out_of_threads_is_a_machine_error(remote: RemoteDaemon) -> None:
    """Out of threads at the thread a frame is written on, a request fails as any other does.

    ``send`` writes each frame on a helper thread; that thread's
    ``RuntimeError`` escaped the handshake (its ``auth`` frame) and a
    request on a kept socket (its command) alike.
    """
    link = mach.link_for(mach.add_machine("box", remote.url, "secret-1"))

    def starve_the_writer(frame_locals: dict[str, Any]) -> bool:
        if frame_locals["name"] == "sorcar-local-send":
            starved.enter_context(nproc_limit_lowered_to_one())
        return False

    for kept in (False, True):  # the handshake's auth frame, then a kept socket's command
        assert (link._ws is not None) is kept
        with ExitStack() as starved, Pause(threading.Thread.__init__.__code__, starve_the_writer):
            with pytest.raises(mach.MachineError, match="can't start new thread"):
                link.request({"type": "ping"}, "pong")
        assert link._ws is None and link._health.retry_at > time.monotonic()
        link._health.retry_at = 0.0
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"


def test_an_unusable_proxy_setting_is_a_machine_error(
    remote: RemoteDaemon, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``$https_proxy`` the WebSocket client cannot use fails the dial as any other does.

    The client's ``InvalidProxy`` is neither an ``InvalidHandshake`` nor an
    ``OSError``: it escaped ``_dial``, so the status check raised it and one
    such machine cost the CHATS panel its whole merged reply, local rows too.
    """
    machine = mach.add_machine("box", remote.url, "secret-1")
    monkeypatch.setenv("https_proxy", "bogus://127.0.0.1:1")
    for name in ("no_proxy", "NO_PROXY", "wss_proxy", "WSS_PROXY", "socks_proxy", "SOCKS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    assert "is unreachable" in mach.machine_status(machine)
    local = mach.HistoryPage([_local_row("b" * 32, "local task", 2.0, False)])
    assert _titles(mach.merged_history(local, [machine], "", "", 0, 50)) == ["local task"]


def test_registry_change_keeps_the_links_of_unchanged_machines(
    remote: RemoteDaemon, remote_leaf: RemoteDaemon,
) -> None:
    """Adding, re-adding or removing a machine leaves the other machines' sockets alone.

    Only a machine whose registry entry changed or vanished loses its
    cached link (and the ssh tunnel of a login no longer registered).
    """
    box = mach.add_machine("box", remote.url, "secret-1")
    leaf = mach.add_machine("leaf", remote_leaf.url)
    link = mach.link_for(box)
    assert _titles(mach.fetch_history(box, "alpha", "", 50)) == ["remote alpha"]
    ws = link._ws
    assert ws is not None
    leaf_link = mach.link_for(leaf)
    assert _titles(mach.fetch_history(leaf, "", "", 50)) == ["leaf task"]

    # A third machine, the same machine again, a removed one: the links
    # to "box" and "leaf" and their open sockets are untouched.
    extra = mach.add_machine("extra", remote_leaf.url)
    extra_link = mach.link_for(extra)
    assert extra_link.request({"type": "ping"}, "pong")["type"] == "pong"
    assert mach.add_machine("leaf", remote_leaf.url) == leaf
    assert mach.remove_machine("extra") is True
    assert mach.link_for(box) is link and link._ws is ws
    assert mach.link_for(leaf) is leaf_link and leaf_link._ws is not None
    assert _titles(mach.fetch_history(box, "beta", "", 50)) == ["remote beta"]
    assert link._ws is ws
    # The removed machine's link is retired: it does not reconnect.
    assert extra_link._retired and extra_link._ws is None
    with pytest.raises(mach.MachineError):
        extra_link.request({"type": "ping"}, "pong")
    assert mach.link_for(extra) is not extra_link

    # A changed entry (another password) replaces the link; the stale
    # one is retired, every other link is still untouched.
    with pytest.raises(mach.MachineError):
        mach.add_machine("box", remote.url, "wrong")
    assert mach.link_for(box) is link and link._ws is ws
    mach.save_machines([mach.Machine("box", remote.url, "secret-2"), leaf])
    assert link._retired and link._ws is None
    assert mach.link_for(leaf) is leaf_link and leaf_link._ws is not None
    mach.close_links()
    assert leaf_link._retired and mach._links == {}


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


def test_a_synced_copy_knows_its_chat_runs_on_its_machine(tmp_path: Path) -> None:
    """The copy of a task kept in the merge says what any copy knows of its chat.

    A history sync leaves a copy of a machine's finished task here; this
    daemon says of it that its chat is not running (not here), the copy
    wins the tie against the machine's own row, and the page, told the
    chat is idle, would open the chat at that task while it runs there.
    """
    done_id = "d" * 32
    busy = start_remote_daemon(tmp_path / "busy-home", seed=[
        {"chat_id": "chat-busy", "task": "done there", "task_id": done_id, "result": "ok"},
        {"chat_id": "chat-busy", "task": "running there", "running": True},
    ])
    try:
        box = mach.add_machine("box", busy.url)
        theirs = {r["task_id"]: r for r in mach.fetch_history(box, "", "", 50).rows}[done_id]
        assert theirs["chat_running"] is True, "the machine must say its chat is running"
        synced = mach.HistoryPage([{
            **_local_row(done_id, "done there (synced copy)", theirs["timestamp"], False),
            "endTs": theirs["endTs"], "chat_running": False,
        }])
        kept = {r["task_id"]: r for r in mach.merged_history(synced, [box], "", "", 0, 50).rows}
        assert kept[done_id]["title"] == "done there (synced copy)"
        assert kept[done_id]["chat_running"] is True
    finally:
        busy.stop()


def test_merged_history_keeps_a_machines_last_page_while_it_does_not_answer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A machine that stops answering keeps its chats in the CHATS panel.

    Its last unfiltered page is served again (trimmed to the limit)
    until it answers afresh; a filtered view it has not answered
    contributes nothing then.
    """
    seed = [
        {"chat_id": "c", "task": f"flaky {i}", "timestamp": 1_700_000_000.0 + i}
        for i in range(3)
    ]
    empty = mach.HistoryPage()
    flaky = start_remote_daemon(tmp_path / "flaky-home", seed=seed)
    try:
        machine = mach.add_machine("flaky", flaky.url)
        # The unfiltered view was never fetched: nothing to fall back on yet.
        assert _titles(mach.merged_history(empty, [machine], "flaky 2", "", 0, 50)) == ["flaky 2"]
    finally:
        flaky.stop()
    assert mach.merged_history(empty, [machine], "", "", 0, 50).rows == []
    # The machine is back (at a new address): its unfiltered page is remembered.
    flaky = start_remote_daemon(tmp_path / "flaky-home-2", seed=seed)
    try:
        machine = mach.add_machine("flaky", flaky.url)
        whole = mach.merged_history(empty, [machine], "", "", 0, 50)
        assert _titles(whole) == ["flaky 2", "flaky 1", "flaky 0"]
        # The registration this one replaced left no page behind for the daemon's life.
        assert {key[1] for key in mach._last_pages} == {flaky.url}
    finally:
        flaky.stop()
    # The daemon is gone: the remembered page stands in, with its date
    # range, trimmed to what the caller asked for, and the failure is
    # still logged; a filtered view it never answered has nothing to show.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="kiss.agents.sorcar.machines"):
        again = mach.merged_history(empty, [machine], "", "", 0, 50)
        assert _titles(again) == ["flaky 2", "flaky 1", "flaky 0"]
        assert (again.date_min, again.date_max) == (whole.date_min, whole.date_max)
        two = mach.merged_history(empty, [machine], "", "", 0, 2)
        assert _titles(two) == ["flaky 2", "flaky 1"]
        assert mach.merged_history(empty, [machine], "flaky", "", 0, 50).rows == []
    assert sum("CHATS: flaky" in rec.getMessage() for rec in caplog.records) == 3
    # Local rows still merge in around the remembered ones.
    local = mach.HistoryPage([_local_row("b" * 32, "local newest", 1_800_000_000.0, False)])
    assert _titles(mach.merged_history(local, [machine], "", "", 0, 50))[:2] == [
        "local newest", "flaky 2",
    ]
    # The name registered again for another daemon (another address)
    # is not shown the remembered page: those are not its chats.
    other = mach.Machine("flaky", f"https://127.0.0.1:{_closed_port()}")
    other.cert_file().write_text(machine.cert_file().read_text(), encoding="utf-8")
    assert mach.merged_history(empty, [other], "", "", 0, 50).rows == []


class _HeldRelay:
    """A TCP relay to *upstream* that holds each new connection *hold* seconds first.

    A machine behind a slow link, or one whose SSH port-forward takes
    a while to open: the connection goes through, late.
    """

    def __init__(self, upstream: tuple[str, int], hold: float) -> None:
        self.upstream, self.hold = upstream, hold
        self.accepted = 0
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(0.2)
        self.port = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _peer = self._server.accept()
            except TimeoutError:
                continue
            self.accepted += 1
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _relay(self, client: socket.socket) -> None:
        if self._stop.wait(self.hold):
            client.close()
            return
        upstream = socket.create_connection(self.upstream, timeout=5)
        for src, dst in ((client, upstream), (upstream, client)):
            threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()

    @staticmethod
    def _pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (src, dst):
                try:
                    sock.close()
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2)
        self._server.close()


def test_merged_history_does_not_wait_long_for_a_slow_machine(tmp_path: Path) -> None:
    """The reply goes out after ``HISTORY_FANOUT_WAIT`` without a machine still being fetched.

    The rows this daemon has are shown at once; the machine's page is
    fetched on (one fetch, shared by the replies meanwhile) and, once
    it lands, the ``late`` callback asks the panels to refresh, so the
    next reply has its rows.  A machine that never answers (a port
    that takes the connection and says nothing) costs the reply the
    same bounded wait, and no refresh is asked for when its fetch
    fails with nothing new to show.
    """
    seed = [
        {"chat_id": "c", "task": f"slow {i}", "timestamp": 1_700_000_000.0 + i}
        for i in range(2)
    ]
    local = mach.HistoryPage([_local_row("b" * 32, "local newest", 1_800_000_000.0, False)])
    daemon = start_remote_daemon(tmp_path / "slow-home", seed=seed)
    relay = _HeldRelay(("127.0.0.1", daemon.port), hold=mach.HISTORY_FANOUT_WAIT + 1.0)
    hole = socket.create_server(("127.0.0.1", 0))  # accepts in the kernel, never answers
    nudges: list[float] = []
    try:
        enrolled = mach.add_machine("slow", daemon.url)  # pins the daemon's certificate
        # Registered again at the relay's address (the pin is by name).
        slow = mach.Machine("slow", f"https://127.0.0.1:{relay.port}")
        assert slow.cert_file() == enrolled.cert_file()
        silent = mach.Machine("silent", f"https://127.0.0.1:{hole.getsockname()[1]}")
        silent.cert_file().write_text(enrolled.cert_file().read_text(), encoding="utf-8")
        mach.save_machines([slow, silent])
        started = time.monotonic()
        first = mach.merged_history(
            local, [slow], "", "", 0, 50, late=lambda: nudges.append(time.monotonic()),
        )
        waited = time.monotonic() - started
        assert mach.HISTORY_FANOUT_WAIT - WAIT_SLACK <= waited < relay.hold, waited
        assert _titles(first) == ["local newest"], "the local rows go out without the machine's"
        assert nudges == []
        # A reply asked for meanwhile waits for the same fetch (one
        # connection), which lands within its own wait.
        second = mach.merged_history(local, [slow], "", "", 0, 50)
        assert _titles(second) == ["local newest", "slow 1", "slow 0"]
        assert relay.accepted == 1
        # The first reply's page landed late: one nudge, once.
        deadline = time.monotonic() + 5
        while len(nudges) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(nudges) == 1
        # With the link up, a reply has the rows well within the wait.
        started = time.monotonic()
        third = mach.merged_history(local, [slow], "", "", 0, 50, late=lambda: nudges.append(0.0))
        assert time.monotonic() - started < mach.HISTORY_FANOUT_WAIT
        assert _titles(third) == ["local newest", "slow 1", "slow 0"]
        assert len(nudges) == 1
        # A machine that never answers: the same bounded wait, no nudge.
        started = time.monotonic()
        page = mach.merged_history(local, [silent], "", "", 0, 50, late=lambda: nudges.append(0.0))
        waited = time.monotonic() - started
        assert mach.HISTORY_FANOUT_WAIT - WAIT_SLACK <= waited < mach.REQUEST_TIMEOUT, waited
        assert _titles(page) == ["local newest"]
        deadline = time.monotonic() + mach.REQUEST_TIMEOUT + 5
        while mach._fetches and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not mach._fetches, "the silent machine's fetch gave up"
        assert len(nudges) == 1, "a fetch that brings nothing new asks for no refresh"
    finally:
        hole.close()
        relay.stop()
        daemon.stop()


def _history_daemon(tmp: Path, title: str, hold: Callable[[], Awaitable[object]]) -> _Daemon:
    """A stand-in daemon answering every ``getHistory`` with one row *title*, after *hold*."""

    async def handle(ws: ServerConnection) -> None:
        try:
            async for message in ws:
                command: dict[str, Any] = json.loads(message)
                if command["type"] != "getHistory":
                    continue
                await hold()
                await ws.send(json.dumps({
                    "type": "history", "generation": command["generation"],
                    "sessions": [{"task_id": "c" * 32, "title": title, "timestamp": 1.7e9}],
                    "dateRange": {"min": 1.7e9, "max": 1.7e9},
                }))
        except ConnectionClosed:
            return

    return _Daemon(tmp, handle)


def _register_stand_in(daemon: _Daemon, name: str, password: str = "pw") -> mach.Machine:
    """Register *daemon* as *name*, its CA pinned, and return the machine."""
    port = daemon.url.split(":")[2].split("/")[0]
    machine = mach.Machine(name, f"https://127.0.0.1:{port}", password=password)
    machine.cert_file().parent.mkdir(parents=True, exist_ok=True)
    machine.cert_file().write_text(daemon.ca_file.read_text(encoding="utf-8"), encoding="utf-8")
    mach.save_machines([machine])
    return machine


def _wait_for_fetches_to_end() -> None:
    """Wait until no history fetch is under way (the fan-out pool is idle)."""
    deadline = time.monotonic() + mach.REQUEST_TIMEOUT + 5
    while mach._fetches and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not mach._fetches, "a history fetch never ended"


def test_a_slow_filtered_page_reaches_the_refresh_its_late_callback_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filtered view (a query) whose fetch takes longer than
    ``HISTORY_FANOUT_WAIT`` every time must show its rows on the refresh
    the ``late`` callback asks for, and that refresh must not ask for
    another: the filtered page was not remembered, so every refresh
    served the empty fallback and fetched afresh, without end (audit A09)."""
    monkeypatch.setattr(mach, "HISTORY_FANOUT_WAIT", 0.2)
    daemon = _history_daemon(tmp_path, "remote hit", lambda: asyncio.sleep(1.0))
    empty = mach.HistoryPage()
    nudges: list[float] = []
    try:
        slow = _register_stand_in(daemon, "slow")
        first = mach.merged_history(
            empty, [slow], "hit", "", 0, 50, late=lambda: nudges.append(time.monotonic()),
        )
        assert first.rows == [], "the slow page is not waited for"
        deadline = time.monotonic() + 10
        while not nudges and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(nudges) == 1, "the page landed late: one refresh asked for"
        # The refresh the late callback asked for: the page it brought.
        second = mach.merged_history(
            empty, [slow], "hit", "", 0, 50, late=lambda: nudges.append(time.monotonic()),
        )
        assert _titles(second) == ["remote hit"], "the late page never reached a reply"
        _wait_for_fetches_to_end()
        assert len(nudges) == 1, "the refresh fetched the same page again and asked for another"
        # Another query is fetched afresh, not served the remembered one.
        other = mach.merged_history(empty, [slow], "miss", "", 0, 50)
        assert other.rows == []
        _wait_for_fetches_to_end()
    finally:
        mach.close_links()
        daemon.close()


def test_a_stale_registrations_late_page_does_not_hide_the_replacements(
    tmp_path: Path,
) -> None:
    """A name registered again for another daemon keeps the page the new
    registration answered with when the previous registration's fetch,
    answered before the change, stores its page afterwards: the pages
    are remembered by registration, not by name (audit A09).  Nor is
    that page stored at all: its registration left the registry, whose
    sweep of the remembered pages had already run, so the page and its
    key (the previous password in it) would stay for the daemon's life."""
    released = threading.Event()
    old_daemon = _history_daemon(
        tmp_path / "old", "old chat",
        lambda: asyncio.get_running_loop().run_in_executor(None, released.wait, 30),
    )
    new_daemon = _history_daemon(tmp_path / "new", "new chat", lambda: asyncio.sleep(0))
    fetched = threading.Event()
    stored = threading.Event()
    fetch_code = mach.fetch_history.__code__

    def hold_before_the_store(frame: Any, event: str, arg: Any) -> None:
        if event == "return" and frame.f_code is fetch_code and not fetched.is_set():
            fetched.set()
            stored.wait(30)

    empty = mach.HistoryPage()
    try:
        old = _register_stand_in(old_daemon, "box")
        threading.setprofile(hold_before_the_store)
        try:
            stale = threading.Thread(target=mach._fetch_quietly, args=(old, "", "", 50))
            stale.start()
        finally:
            threading.setprofile(None)
        released.set()
        assert fetched.wait(30), "the old registration's fetch never got its page"
        new = _register_stand_in(new_daemon, "box")
        assert _titles(mach.merged_history(empty, [new], "", "", 0, 50)) == ["new chat"]
        stored.set()
        stale.join(30)
        assert not stale.is_alive()
        assert {key[:2] for key in mach._last_pages} == {("box", new.url)}, (
            "a registration that left the registry stored its page after the sweep"
        )
        new_daemon.close()  # the new daemon stops answering: its page stands in
        assert _titles(mach.merged_history(empty, [new], "", "", 0, 50)) == ["new chat"], (
            "the stale registration's page replaced the new registration's"
        )
    finally:
        stored.set()
        mach.close_links()
        old_daemon.close()
        new_daemon.close()


def test_a_name_registered_again_does_not_join_the_previous_daemons_fetch(tmp_path: Path) -> None:
    """A fetch is shared by registration, not by name alone.

    While ``box`` at one address is still being fetched, ``box``
    registered again at another address (another daemon's chats) gets
    a fetch of its own, as :func:`_last_page` already keeps the pages
    apart; the same registration asked again shares the fetch under
    way.  Both addresses here take the connection and never answer, so
    the fetches stay under way until the holes close.  Both
    registrations are in the registry at once: only a registered
    machine gets a live link (:func:`link_for`), and the retired link of
    an unregistered one fails the fetch in the pool thread at once, so
    that whether the second call joined it would depend on which thread
    ran first.

    Not covered end to end: ``_shared_fetch`` registering its
    forget-callback outside ``_fetches_lock``.  A fetch that is done by
    then runs the callback in the caller's thread, where the
    non-reentrant lock would have deadlocked every later history reply;
    a real pool thread never finishes before the next statement of the
    caller (300 tries, also with a microsecond switch interval), so that
    path is reachable only with a stand-in fetch.
    """
    certfile, _key, _ca = make_test_tls(tmp_path)
    holes = [socket.create_server(("127.0.0.1", 0)) for _ in range(2)]
    try:
        old = mach.Machine("box", f"https://127.0.0.1:{holes[0].getsockname()[1]}")
        new = mach.Machine("box", f"https://127.0.0.1:{holes[1].getsockname()[1]}")
        old.cert_file().parent.mkdir(parents=True, exist_ok=True)
        old.cert_file().write_text(certfile.read_text(encoding="utf-8"), encoding="utf-8")
        mach.save_machines([old, new])
        first = mach._shared_fetch(old, "", "", 50)
        assert mach._shared_fetch(old, "", "", 50) is first, "the same registration shares"
        assert not first.done(), "the hole holds the fetch"
        assert mach._shared_fetch(new, "", "", 50) is not first, "another daemon, another fetch"
        # Registered again with other credentials (a corrected password):
        # the fetch under way would answer with the old ones' refusal.
        corrected = mach.Machine("box", old.url, password="corrected")
        assert mach._shared_fetch(corrected, "", "", 50) is not first, (
            "other credentials joined the previous registration's fetch"
        )
    finally:
        for hole in holes:
            hole.close()
        deadline = time.monotonic() + mach.REQUEST_TIMEOUT + 5
        while mach._fetches and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not mach._fetches, "the fetches ended once the holes closed"


def test_a_page_stored_while_another_fetch_prunes_breaks_neither(tmp_path: Path) -> None:
    """Two fan-out workers storing their pages at once break neither fetch.

    A filtered fetch prunes its registration's stale filtered views
    with a comprehension over ``_last_pages`` while another worker
    stores another machine's page: without one lock around the cache
    the store raised ``RuntimeError: dictionary changed size during
    iteration`` out of the pruning fetch, which ``merged_history``
    re-raised, losing the CHATS reply (audit S5/F1).  The schedule is
    forced with a trace function: the pruning fetch, run in this
    thread, pauses at the second step of its comprehension while a
    second thread fetches and stores the other machine's page (locked,
    that store waits for the pruning to end, so the pause times out).
    """
    a_daemon = _history_daemon(tmp_path / "a", "a chat", lambda: asyncio.sleep(0))
    b_daemon = _history_daemon(tmp_path / "b", "b chat", lambda: asyncio.sleep(0))
    paused, stored = threading.Event(), threading.Event()
    code = mach._fetch_quietly.__code__
    # The first FOR_ITER of the function: the comprehension's walk over the cache.
    comprehension = next(i.offset for i in dis.get_instructions(code) if i.opname == "FOR_ITER")
    steps = [0]

    def trace(frame: Any, event: str, arg: object) -> Any:
        if frame.f_code is not code:
            return None
        frame.f_trace_opcodes = True
        if event == "opcode" and frame.f_lasti == comprehension:
            steps[0] += 1
            if steps[0] == 2:
                paused.set()
                stored.wait(2.0)
        return trace

    def store_b() -> mach.HistoryPage:
        assert paused.wait(30), "the pruning fetch never walked the cache"
        try:
            return mach._fetch_quietly(b, "", "", 50)
        finally:
            stored.set()

    previous = sys.gettrace()
    try:
        a = _register_stand_in(a_daemon, "a")
        b = _register_stand_in(b_daemon, "b")
        mach.save_machines([a, b])
        # Two remembered pages give the comprehension a second step; both links are open.
        assert _titles(mach._fetch_quietly(a, "old", "", 50)) == ["a chat"]
        assert _titles(mach._fetch_quietly(b, "warm", "", 50)) == ["b chat"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            storing = pool.submit(store_b)
            sys.settrace(trace)
            try:
                pruning = mach._fetch_quietly(a, "new", "", 50)
            finally:
                sys.settrace(previous)
                paused.set()
            assert steps[0] >= 2, "the pruning fetch never reached its comprehension"
            assert _titles(pruning) == ["a chat"]
            assert _titles(storing.result(timeout=30)) == ["b chat"]
        assert {k[:1] + k[4:] for k in mach._last_pages} == {
            ("a", "new", ""), ("b", "warm", ""), ("b", "", ""),
        }, "the stale filtered view went, every other page stayed"
        # Both daemons gone: the pages stored under the race stand in for them.
        a_daemon.close()
        b_daemon.close()
        local = mach.HistoryPage([_local_row("d" * 32, "local", 3.0, False)])
        assert sorted(_titles(mach.merged_history(local, [a, b], "", "", 0, 50))) == [
            "b chat", "local",
        ]
        assert sorted(_titles(mach.merged_history(local, [a, b], "new", "", 0, 50))) == [
            "a chat", "local",
        ]
        _wait_for_fetches_to_end()
    finally:
        sys.settrace(previous)
        paused.set()
        mach.close_links()
        a_daemon.close()
        b_daemon.close()


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
    daemon.  The daemon has no remote password, so a peer on the LAN
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
    tunnel = mach._tunnels[lan_sshd.login, lan_remote.port]
    assert tunnel.alive() and tunnel.remote_port == lan_remote.port
    assert _tunnel_peer_port(link) == tunnel.local_port != lan_remote.port
    stream = link.open_connection()
    assert _pong(stream)
    assert int(stream.socket.getpeername()[1]) == tunnel.local_port
    stream.close()
    assert mach._tunnels == {(lan_sshd.login, lan_remote.port): tunnel}  # one per daemon
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
        assert json.loads(raw.recv(timeout=10)) == {
            "type": "auth_ok", "local": True, "echoesGeneration": True,
        }

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

    # Enrolling again kept the tunnel (its daemon is still registered);
    # the links to "box" and to the machine's default name share it.
    assert mach._tunnels == {(lan_sshd.login, lan_remote.port): tunnel}
    assert _titles(mach.fetch_history(box, "alpha", "", 50)) == ["lan alpha"]
    assert _tunnel_peer_port(mach.link_for(box)) == tunnel.local_port

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
    # The machine's default name still names that daemon: its forward stays
    # up until the last machine registered there is gone.
    assert (lan_sshd.login, lan_remote.port) in mach._tunnels
    assert mach.remove_machine(machine.name) is True
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
        # An endpoint file no daemon wrote (no pid alive) is passed over.
        endpoint.write_text("junk", encoding="utf-8")
        with pytest.raises(mach.MachineError, match=r"no Sorcar daemon is running on .*: no .*"):
            mach.add_machine("", idle.login)
        endpoint.write_text(_endpoint_json("wss://127.0.0.1/ws"), encoding="utf-8")
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
        endpoint.write_text(_endpoint_json("wss://127.0.0.1/ws"), "utf-8")
        with pytest.raises(mach.MachineError, match="did not answer the enrollment as a Sorcar"):
            mach.add_machine("", idle.login)
        (nothing / "tls" / "ca.pem").unlink()
        endpoint.write_text(_endpoint_json("wss://127.0.0.1:1/ws"), "utf-8")
        with pytest.raises(mach.MachineError, match="did not answer the enrollment as a Sorcar"):
            mach.add_machine("", idle.login)
        (nothing / "tls" / "ca.pem").write_text(_local_ca_pem(), encoding="utf-8")
        endpoint.write_text(_endpoint_json(f"wss://127.0.0.1:{_closed_port()}/ws"), "utf-8")
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
    assert link._health.retry_at > time.monotonic()
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
    # Nor is an ssh that cannot be run (its PermissionError escaped ``add_machine``).
    (tmp_path / "empty-path").mkdir()
    (tmp_path / "empty-path" / "ssh").write_text("#!/bin/sh\n", encoding="utf-8")
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
    configuration.  The endpoint file a crashed daemon or an earlier
    install left in another home (its pid gone) is passed over, never
    enrolled into (audit A09).
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
    # The local brand's home holds the endpoint file of a daemon that
    # is gone (its pid exited), then one no daemon wrote (no pid): the
    # running daemon's home is found instead, and nothing is written
    # into the stale one.
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    stale_endpoint = {"url": f"wss://127.0.0.1:{stale_port}/ws", "token": "stale"}
    for endpoint in ({**stale_endpoint, "pid": exited.pid}, stale_endpoint):
        (stale / "sorcar-local.json").write_text(json.dumps(endpoint, indent=2), "utf-8")
        assert enroll_via(None) == live
        assert list((stale / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).iterdir()) == []
    # The local brand's home wins over other brands' directories while
    # its daemon runs.
    (stale / "sorcar-local.json").write_text(
        json.dumps({**stale_endpoint, "pid": os.getpid()}, indent=2), "utf-8",
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


@posix_only("the enrollment command is a POSIX sh script")
def test_enrollment_stages_the_ca_under_a_name_nobody_planted(tmp_path: Path) -> None:
    """The staged CA was ``.NAME.<pid>.tmp``, a name anyone can compute (S11 re-fix round 2).

    A symlink waiting there was written through (``cat >`` follows it)
    and then renamed into ``NAME.pem``: the file it named held the CA,
    and the daemon read its trust through a link.  The staging name is
    random now and created by the shell's noclobber (``O_EXCL``).
    """
    home = tmp_path / "state"
    trusted = home / "tls" / tls_certs.TRUSTED_CLIENTS_DIR
    trusted.mkdir(parents=True)
    (home / "tls" / "ca.pem").write_text("daemon CA\n", encoding="utf-8")
    endpoint = json.dumps({"url": "wss://127.0.0.1:1/ws", "pid": os.getpid()}, indent=2)
    (home / "sorcar-local.json").write_text(endpoint, encoding="utf-8")
    victim = tmp_path / "victim"
    victim.write_text("precious\n", encoding="utf-8")
    # The shell that plants the link execs the command: its pid names the old staging file.
    plant = f"ln -s {shlex.quote(str(victim))} {shlex.quote(str(trusted))}/.label.$$.tmp"
    done = subprocess.run(
        ["sh", "-c", f"{plant} && exec {mach.ENROLL_COMMAND.replace('NAME', 'label')}"],
        input="client CA\n", capture_output=True, text=True, timeout=30,
        env={**os.environ, "KISS_HOME": str(home), "HOME": str(tmp_path)},
    )
    assert (done.returncode, done.stdout) == (0, f"{endpoint}\ndaemon CA\n"), done.stderr
    assert victim.read_text(encoding="utf-8") == "precious\n"
    assert [p.name for p in trusted.iterdir() if not p.is_symlink()] == ["label.pem"]
    assert (trusted / "label.pem").read_text(encoding="utf-8") == "client CA\n"


@posix_only("the enrollment command is a POSIX sh script")
def test_enrollment_passes_over_a_home_whose_pid_a_later_process_took(tmp_path: Path) -> None:
    """A dead daemon's endpoint file whose pid is alive again is not a daemon's (audit 2026-11-12).

    ``kill -0`` was the whole proof of a running daemon.  A daemon
    killed with its machine leaves its endpoint file, and after the
    reboot its pid is another process's: the stale home, first in the
    search, was enrolled into and its dead port reported, so the
    machine could not be added.  A process born after the file was
    written did not write it.
    """
    stale, live = tmp_path / ".kiss", tmp_path / ".s10s"
    endpoint = {}
    for home in (stale, live):
        (home / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).mkdir(parents=True)
        (home / "tls" / "ca.pem").write_text(f"CA of {home.name}\n", encoding="utf-8")
        record = {"url": f"wss://127.0.0.1:1/{home.name}", "pid": os.getpid()}
        endpoint[home] = json.dumps(record, indent=2)
        (home / "sorcar-local.json").write_text(endpoint[home], encoding="utf-8")
    command = ["sh", "-c", mach.ENROLL_COMMAND.replace("NAME", "label")]
    env = {**os.environ, "KISS_HOME": str(stale), "HOME": str(tmp_path)}
    long_ago = time.time() - 30 * 86400  # before this process, which has the pid now, was born
    os.utime(stale / "sorcar-local.json", (long_ago, long_ago))
    done = subprocess.run(
        command, input="client CA\n", capture_output=True, text=True, timeout=30, env=env,
    )
    assert (done.returncode, done.stdout) == (0, f"{endpoint[live]}\nCA of .s10s\n"), done.stderr
    assert not list((stale / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).iterdir())
    enrolled = live / "tls" / tls_certs.TRUSTED_CLIENTS_DIR / "label.pem"
    assert enrolled.read_text(encoding="utf-8") == "client CA\n"
    # No file as new as its process (a wall clock stepped forward since
    # the daemon started): the running pid alone decides, as it did.
    os.utime(live / "sorcar-local.json", (long_ago, long_ago))
    done = subprocess.run(
        command, input="client CA\n", capture_output=True, text=True, timeout=30, env=env,
    )
    assert (done.returncode, done.stdout) == (0, f"{endpoint[stale]}\nCA of .kiss\n"), done.stderr


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
        link._health.retry_at = 0.0  # the backoff has expired
        assert _titles(mach.fetch_history(machine, "", "", 50)) == ["two", "one"]
        assert mach.fetch_history(machine, "", "", 50).rows[0]["machine_url"] == machine.url

        daemon.stop()
        daemon = start_remote_daemon(home, password="pw", seed=[{"chat_id": "c", "task": "three"}])
        if daemon.port == port:
            pytest.skip("the OS handed the restarted daemon the same port")
        link._health.retry_at = 0.0
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
    link to the daemon; it is opened again after its ssh process died,
    and closed by ``close_links``.
    """
    enrolled = mach.add_machine("near", sshd.login)
    far = mach.Machine("far", f"https://192.0.2.1:{remote.port}", ssh=sshd.login)
    far.cert_file().write_bytes(enrolled.cert_file().read_bytes())
    mach.save_machines([enrolled, far])
    assert mach._tunnels == {}  # the registry changed: every tunnel was closed
    try:
        assert mach.machine_status(far) == "reachable"
        tunnel = mach._tunnels[sshd.login, remote.port]
        assert tunnel.process is not None and tunnel.alive()
        pid = tunnel.process.pid
        link = mach.link_for(far)
        assert _titles(mach.fetch_history(link.machine, "alpha", "", 50)) == ["remote alpha"]
        assert _tunnel_peer_port(link) == tunnel.local_port
        assert _titles(mach.fetch_history(enrolled, "beta", "", 50)) == ["remote beta"]
        assert _tunnel_peer_port(mach.link_for(enrolled)) == tunnel.local_port
        assert mach._tunnels == {(sshd.login, remote.port): tunnel}
        assert tunnel.process.pid == pid  # reused

        tunnel.process.kill()
        tunnel.process.wait()
        assert not tunnel.alive()
        assert _titles(mach.fetch_history(far, "shared", "", 50)) == ["shared task"]
        assert tunnel.process is not None and tunnel.process.pid != pid and tunnel.alive()
        assert _tunnel_peer_port(link) == tunnel.local_port
    finally:
        mach.close_links()
    assert mach._tunnels == {} and tunnel.process is None and not tunnel.alive()


def _serve_tls(listener: socket.socket, ctx: ssl.SSLContext, stop: threading.Event) -> None:
    """Answer every connection to *listener* with a TLS handshake until *stop* is set."""
    listener.settimeout(0.1)
    while not stop.is_set():
        try:
            conn, _ = listener.accept()
        except TimeoutError:
            continue
        conn.settimeout(5.0)
        try:
            with ctx.wrap_socket(conn, server_side=True):
                pass
        except OSError:
            conn.close()


def test_ssh_tunnel_never_hands_over_a_port_another_program_took(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path,
) -> None:
    """A program that binds the tunnel's port before ssh does never gets the daemon's links.

    The tunnel picks a free port and ssh binds it only after its login.
    A TLS server of another program that took the port in between
    used to pass for the forward (it accepts connections), so the
    link's handshake met a stranger's certificate: ``refused the TLS
    handshake: [SSL: CERTIFICATE_VERIFY_FAILED]``, and advice to add
    the machine again.  Now the forward counts as up only once ssh says
    its port is bound; ssh cannot bind the taken one, and the tunnel
    forwards another port instead.
    """
    machine = mach.add_machine("near", sshd.login)
    certfile, keyfile, _ca = make_test_tls(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    stranger, stop = socket.socket(), threading.Event()
    server = threading.Thread(target=_serve_tls, args=(stranger, ctx, stop), daemon=True)
    picking = Pause(mach.ssh_command.__code__, lambda local: "-N" in local["options"])
    try:
        with picking, ThreadPoolExecutor(1) as pool:
            status = pool.submit(mach.machine_status, machine)
            assert picking.paused.wait(30)
            taken = mach._tunnels[sshd.login, remote.port].local_port
            stranger.bind(("127.0.0.1", taken))
            stranger.listen()
            server.start()
            picking.release()
            assert status.result(timeout=60) == "reachable"
        tunnel = mach._tunnels[sshd.login, remote.port]
        assert tunnel.alive() and tunnel.local_port not in (0, taken)
    finally:
        stop.set()
        # Joined before the close: an accept() on the closed socket raises in the thread.
        if server.is_alive():
            server.join(timeout=5)
        stranger.close()
        mach.close_links()


def test_ssh_tunnel_forwards_even_if_the_ssh_config_clears_forwardings(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ClearAllForwardings yes`` in the user's ssh config does not drop the tunnel's forward.

    ssh would still run the ``LocalCommand`` that reports the forward up,
    with nothing listening on the port.
    """
    machine = mach.add_machine("near", sshd.login)
    config = tmp_path / "ssh_config"
    config.write_text(
        "ClearAllForwardings yes\n" + sshd.config.read_text(encoding="utf-8"), encoding="utf-8",
    )
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(config))
    try:
        assert mach.machine_status(machine) == "reachable"
    finally:
        mach.close_links()


_FAKE_SSH = '''#!__PYTHON__
"""Stand-in for ssh: ``-G`` names a blackhole host, ``-N -L`` relays to the daemon
and runs the ``LocalCommand`` once it listens, a command runs locally with KISS_HOME
set, FAKE_SSH_FAIL refuses the forward (``login``: the login fails; ``stay``: the bind
fails and ssh stays up, as Windows' does), FAKE_SSH_SLOW (seconds) is a slow login
before the forward is up, FAKE_SSH_PIDFILE records the forward's pid."""
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
local_command = next((a.split("=", 1)[1] for a in args if a.startswith("LocalCommand=")), "")
if os.environ.get("FAKE_SSH_PIDFILE"):
    with open(os.environ["FAKE_SSH_PIDFILE"], "w") as handle:
        handle.write(str(os.getpid()))
fail = os.environ.get("FAKE_SSH_FAIL")
if fail == "login":
    sys.stderr.write("ksen@far.example.org: Permission denied (publickey).\\n")
    sys.exit(255)
if fail:
    sys.stderr.write(f"bind [127.0.0.1]:{local}: Address already in use\\n")
    sys.stderr.flush()
    if fail == "stay":
        time.sleep(60)
    sys.stderr.write("Could not request local forwarding.\\n")
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
try:
    server.bind(("127.0.0.1", int(local)))
except OSError as exc:
    sys.stderr.write(f"bind [127.0.0.1]:{local}: {exc.strerror}\\n"
                     "Could not request local forwarding.\\n")
    sys.exit(255)
server.listen()
if local_command:
    subprocess.run(local_command, shell=True)
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
        tunnel = mach._tunnels[login, daemon.port]
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
        monkeypatch.setenv("FAKE_SSH_FAIL", "login")
        assert mach.machine_status(machine) == (
            f"far (ssh {login}) is unreachable: ssh {login} cannot forward its daemon port "
            f"{daemon.port}: ksen@far.example.org: Permission denied (publickey)."
        )
        # An ssh that stays up after a failed bind (Windows') is not waited for to the
        # timeout, and no attempt on a new port starts once the timeout has passed.
        monkeypatch.setenv("FAKE_SSH_FAIL", "stay")
        bind_failed = (
            rf"daemon port {daemon.port}: bind \[127\.0\.0\.1\]:\d+: Address already in use$"
        )
        started = time.monotonic()
        with pytest.raises(mach.MachineError, match=bind_failed):
            mach.SshTunnel(login, daemon.port).open(timeout=30.0)
        assert time.monotonic() - started < 20.0
        started = time.monotonic()
        with pytest.raises(
            mach.MachineError, match=f"{bind_failed}|did not open the port forward in time",
        ):
            mach.SshTunnel(login, daemon.port).open(timeout=1.0)
        assert time.monotonic() - started < 2.9
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
def test_a_kept_socket_follows_the_login_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request dials again when the kept socket's forward is not the login's live one.

    The forward's ssh process ending closes the port the socket is
    connected to, but the socket learns of it on its reader thread
    only: a request that won the race with that thread sent on the
    dead socket, and a close after the send is never retried, so a
    full-load run reported the machine down instead of opening the
    forward again.  The link now asks the forward before the send
    (``MachineLink._forward_gone``): its process exited, it was
    closed, or another lane opened it again on a fresh local port.
    The reader thread's timing is not controllable from a test, so
    the decision is asserted on the link directly as well as through
    the request.  Two branches stay uncovered, both races: a socket
    closed under the request by ``retire`` (``getpeername`` fails),
    and ``_exited`` asked about a process another thread's ``wait``
    reaps at that instant (``ChildProcessError``).
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@far.example.org"
    try:
        machine = mach.add_machine("far", login)
        link = mach.link_for(machine)
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        tunnel = mach._tunnels[login, daemon.port]
        assert tunnel.process is not None and not link._forward_gone()
        assert not mach.MachineLink(mach.Machine("direct", daemon.url))._forward_gone()

        # The forward's ssh process died.
        pid = tunnel.process.pid
        tunnel.process.kill()
        tunnel.process.wait()
        assert link._forward_gone()
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert tunnel.process is not None and tunnel.process.pid != pid and tunnel.alive()
        assert _tunnel_peer_port(link) == tunnel.local_port and not link._forward_gone()

        # Another lane opened the forward again on a fresh port.
        port, pid = tunnel.local_port, tunnel.process.pid
        tunnel.close()
        assert tunnel.process is None and link._forward_gone()
        if tunnel.open() == port:
            pytest.skip("the OS handed the forward opened again the same port")
        assert tunnel.process is not None and tunnel.process.pid != pid
        assert link._forward_gone()
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert _tunnel_peer_port(link) == tunnel.local_port and not link._forward_gone()
    finally:
        mach.close_links()
        daemon.stop()


def _pause_after_opcode(code: Any, offset: int, pause: Callable[[], None]) -> Any:
    """A trace function that calls *pause* once *code* has run its instruction at *offset*."""

    def trace(frame: Any, event: str, arg: object) -> Any:
        if frame.f_code is not code:
            return None
        frame.f_trace_opcodes = True
        if event == "opcode" and frame.f_lasti > offset and not getattr(trace, "done", False):
            trace.done = True  # type: ignore[attr-defined]
            pause()
        return trace

    return trace


@fake_ssh_only
def test_a_forward_closed_under_the_check_is_dialled_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_forward_gone`` reads the forward's process once: a close meanwhile breaks nothing.

    The check runs under the link's lock, not the tunnel's: another
    thread's ``tunnel.close()`` can set the process to ``None`` at any
    point in it.  Read twice (the ``None`` test, then the argument of
    ``_exited``), the second read raised ``AttributeError`` out of
    ``request`` (review 14).  The interleaving is forced with a trace
    function: the request pauses right after its one read while the
    close runs, then finds the process dead and dials again.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@far.example.org"
    previous = sys.gettrace()
    try:
        machine = mach.add_machine("far", login)
        link = mach.link_for(machine)
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        tunnel = mach._tunnels[login, daemon.port]
        code = mach.MachineLink._forward_gone.__code__
        reads = [
            i.offset
            for i in dis.get_instructions(code)
            if i.opname == "LOAD_ATTR" and i.argval == "process"
        ]
        assert len(reads) == 1, reads
        closer = threading.Thread(target=tunnel.close)

        def close_meanwhile() -> None:
            closer.start()
            closer.join(timeout=30)
            assert not closer.is_alive() and tunnel.process is None

        trace = _pause_after_opcode(code, reads[0], close_meanwhile)
        sys.settrace(trace)
        try:
            assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        finally:
            sys.settrace(previous)
        assert trace.done, "the request never reached the check"  # type: ignore[attr-defined]
        assert tunnel.process is not None and tunnel.alive()
        assert _tunnel_peer_port(link) == tunnel.local_port and not link._forward_gone()
    finally:
        sys.settrace(previous)
        mach.close_links()
        daemon.stop()


def _open_noting_the_thread(tunnel: mach.SshTunnel, idents: list[int]) -> int:
    """Note this thread's ident in *idents*, then open *tunnel* on it."""
    idents.append(threading.get_ident())
    return tunnel.open()


@fake_ssh_only
def test_an_interrupted_tunnel_opening_does_not_vouch_for_its_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop during the ssh login leaves ssh running unconfirmed; its port is not trusted.

    The runner's Stop raises ``KeyboardInterrupt`` on the task thread
    wherever it is, here while the opener waits for ssh's word.  Another
    program then binds the port ssh had yet to bind: the next open must
    not take that answering port for the forward.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    monkeypatch.setenv("FAKE_SSH_SLOW", "30")
    tunnel = mach.SshTunnel("ksen@far.example.org", daemon.port)
    stranger = socket.socket()
    idents: list[int] = []
    try:
        with ThreadPoolExecutor(1) as pool:
            opening = pool.submit(_open_noting_the_thread, tunnel, idents)
            deadline = time.monotonic() + 30.0
            while tunnel.process is None and time.monotonic() < deadline:
                time.sleep(0.01)
            taken = tunnel.local_port
            stranger.bind(("127.0.0.1", taken))
            stranger.listen()
            assert inject_keyboard_interrupt(idents[0]) == 1
            with pytest.raises(KeyboardInterrupt):
                opening.result(timeout=30)
        assert tunnel.process is not None and tunnel.process.poll() is None
        assert not tunnel.alive()
        monkeypatch.setenv("FAKE_SSH_SLOW", "0")
        port = tunnel.open(timeout=20.0)
        assert port != taken and tunnel.alive()
    finally:
        tunnel.close()
        stranger.close()
        daemon.stop()


@fake_ssh_only
@pytest.mark.skipif(not thread_start_can_be_starved(), reason="Thread.start cannot be starved")
def test_a_forward_out_of_threads_is_a_machine_error_and_leaves_no_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out of threads at the forward's output reader, the opening fails as any other does.

    The reader's ``RuntimeError`` left ``SshTunnel.open`` with ssh still
    running in the cached tunnel, and went past the ``MachineError``
    handlers of the dial and of the CHATS fan-out, whose history reply
    then failed whole.  ``RLIMIT_NPROC`` is lowered for exactly that
    start, as for the dial's own thread above.
    """
    _install_fake_ssh(tmp_path, monkeypatch, tmp_path)
    mach.client_cert_pair()  # issues the local CA, pinned here in the daemon's place
    machine = mach.Machine("far", "https://192.0.2.1:9", ssh="ksen@far.example.org")
    machine.cert_file().parent.mkdir(parents=True, exist_ok=True)
    machine.cert_file().write_text(_local_ca_pem(), encoding="utf-8")
    mach.save_machines([machine])
    logins: list[subprocess.Popen[bytes]] = []

    def starve_the_reader(frame_locals: dict[str, Any]) -> bool:
        if getattr(frame_locals["target"], "__name__", "") == "_read_output":
            logins.append(frame_locals["args"][0])
            starved.enter_context(nproc_limit_lowered_to_one())
        return False

    try:
        with ExitStack() as starved, Pause(threading.Thread.__init__.__code__, starve_the_reader):
            assert mach._fetch_quietly(machine, "", "", 10) == mach.HistoryPage()
        assert "the port forward could not start" in mach.link_for(machine)._health.last_error
        assert len(logins) == 1 and logins[0].poll() is not None, "ssh was left running"
        assert mach._tunnels[machine.ssh, 9].process is None
    finally:
        mach.close_links()


@fake_ssh_only
def test_removing_a_machine_waits_for_a_stream_handshake_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A removal during a stream's handshake leaves no tunnel of the removed machine.

    A stream's handshake (``open_connection``) runs outside the
    link's lock, so ``remove_machine`` could retire the link and sweep
    the tunnels while the handshake had not opened its tunnel yet; the
    handshake then opened a fresh one that nothing swept.  ``retire``
    waits for the handshake instead, so the sweep comes after it.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@going.example.org"
    streams: list[ClientConnection] = []
    errors: list[Exception] = []

    def connect(link: mach.MachineLink) -> None:
        try:
            streams.append(link.open_connection())
        except mach.MachineError as exc:
            errors.append(exc)

    try:
        machine = mach.add_machine("going", login)
        link = mach.link_for(machine)
        with Pause(mach.tunnel_port.__code__, lambda local: local["ssh"] == login) as held:
            dialling = threading.Thread(target=connect, args=(link,))
            dialling.start()
            assert held.paused.wait(30), "the handshake about to open its tunnel"
            removing = threading.Thread(target=mach.remove_machine, args=("going",))
            removing.start()
            # The removal retires the link and waits for the handshake.
            time.sleep(0.5)
            assert link._retired and removing.is_alive()
        dialling.join(30)
        removing.join(30)
        assert not dialling.is_alive() and not removing.is_alive()
        assert mach.load_machines() == []
        assert mach._tunnels == {}, "the removed machine's tunnel was swept"
        for stream in streams:
            stream.close()
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
        assert link._health.retry_at == 0.0
        assert mach._tunnels[login, daemon.port].alive()
    finally:
        mach.close_links()
        daemon.stop()


@fake_ssh_only
def test_an_ssh_machine_is_dialled_past_the_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A link to an SSH machine goes to the local end of its forward, whatever ``$https_proxy``.

    The WebSocket client takes its proxy from the environment by
    default and, with no ``no_proxy`` entry for 127.0.0.1, asked it for
    the forward's local port: the machine was unreachable though its
    forward was up, from the first handshake of ``/machine add`` on.
    The proxy here accepts into its backlog and never answers.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    proxy = socket.create_server(("127.0.0.1", 0))
    proxy.settimeout(0)
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{proxy.getsockname()[1]}")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    try:
        machine = mach.add_machine("far", "ksen@far.example.org")
        assert mach.machine_status(machine) == "reachable"
        with pytest.raises(BlockingIOError):
            proxy.accept()  # nothing was dialled through it
    finally:
        proxy.close()
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
        _endpoint_json(f"wss://127.0.0.1:{_closed_port()}/ws"), "utf-8",
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
def test_a_stale_dial_never_replaces_the_forward_of_a_new_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handshake dialling the OLD registration's daemon port must not close
    the login's forward the NEW registration opened (audit A09).

    ``box`` is registered again with its daemon on another port while a
    stream's handshake on the old link is about to open the login's
    forward.  The registration retires the old link and waits for that
    handshake; the new registration's link meanwhile opens the login's
    forward to the new port.  The old handshake, resumed, asked
    ``tunnel_port`` for the old port: the cache, keyed by login alone,
    replaced and closed the live forward before any registry check, and
    the check knew the login but not the port.
    """
    daemon = start_remote_daemon(tmp_path / "home", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, daemon.home)
    login = "ksen@box.example.org"
    streams: list[ClientConnection] = []
    errors: list[Exception] = []
    try:
        machine = mach.add_machine("box", login)
        link = mach.link_for(machine)
        old_port = urlsplit(machine.url).port
        moved = mach.Machine(
            machine.name, url=f"https://127.0.0.1:{_closed_port()}", ssh=login,
            password=machine.password,
        )
        with Pause(mach.tunnel_port.__code__, lambda local: local["remote_port"] == old_port):
            dialling = threading.Thread(target=_opens_a_stream, args=(link, streams, errors))
            dialling.start()
            registering = threading.Thread(target=mach.save_machines, args=([moved],))
            # The old link's handshake is held; the registration retires
            # the link and waits for it; the new registration's link
            # opens the login's forward to the moved port meanwhile.
            registering.start()
            deadline = time.monotonic() + 30
            while not link._retired:
                assert time.monotonic() < deadline, "the registration never retired the link"
                time.sleep(0.01)
            with pytest.raises(mach.MachineError, match="refused the forwarded connection"):
                mach.link_for(moved).request({"type": "ping"}, "pong", timeout=5.0)
            current = mach._tunnels[mach._endpoint(moved)]
            assert current.remote_port == urlsplit(moved.url).port and current.alive()
        dialling.join(60)
        registering.join(60)
        assert not dialling.is_alive() and not registering.is_alive()
        assert mach._tunnels == {(login, current.remote_port): current} and current.alive(), (
            "the stale dial replaced and closed the new registration's forward"
        )
        assert streams == [] and len(errors) == 1
        assert "registry changed" in str(errors[0])
    finally:
        for stream in streams:
            stream.close()
        mach.close_links()
        daemon.stop()


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
        tunnel = mach._tunnels[login, daemon.port]
        assert tunnel.process is not None
        pid = tunnel.process.pid

        # The machine's daemon now answers the enrollment with a CA that
        # did not issue its certificate: same port, wrong pin.
        monkeypatch.setattr(mach, "ENROLL_TIMEOUT", 2.0)
        wrong = tmp_path / "wrong"
        (wrong / "tls" / tls_certs.TRUSTED_CLIENTS_DIR).mkdir(parents=True)
        (wrong / mach.LOCAL_ENDPOINT_FILE).write_text(_endpoint_json(
            f"wss://127.0.0.1:{daemon.port}/ws",
            ca=str(mach.kiss_home() / "tls" / tls_certs.CA_CERT_FILE),
        ), encoding="utf-8")
        monkeypatch.setenv("FAKE_SSH_HOME", str(wrong))
        with pytest.raises(mach.MachineError, match="refused the TLS handshake"):
            mach.add_machine("box", login)

        assert mach._tunnels == {(login, daemon.port): tunnel}
        assert tunnel.process is not None and tunnel.process.pid == pid and tunnel.alive()
        assert _pong(stream)  # the live stream was not disconnected
        assert mach.link_for(machine) is link
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        assert mach.load_machines() == [machine]
        assert not list((mach.kiss_home() / "machines").glob(".box*"))  # no staged pin left
    finally:
        mach.close_links()
        daemon.stop()


@fake_ssh_only
def test_two_daemons_behind_one_login_each_keep_their_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two registered daemons of one ssh login (two ports) have a forward each.

    The forwards were cached by login alone: the dial of the second
    daemon found the login's forward on another remote port, replaced
    it and closed it under the first daemon's open stream.  Removing
    one of the two machines closes that one's forward only.
    """
    one = start_remote_daemon(tmp_path / "one", password="pw")
    two = start_remote_daemon(tmp_path / "two", password="pw")
    _install_fake_ssh(tmp_path, monkeypatch, one.home)
    login = "ksen@pair.example.org"
    try:
        first = mach.add_machine("one", login)
        monkeypatch.setenv("FAKE_SSH_HOME", str(two.home))
        second = mach.add_machine("two", login)
        stream = mach.link_for(first).open_connection()
        assert _pong(stream)
        assert mach.machine_status(second) == "reachable"
        assert _pong(stream), "the second daemon's dial closed the first's forward"
        assert set(mach._tunnels) == {(login, one.port), (login, two.port)}
        assert all(tunnel.alive() for tunnel in mach._tunnels.values())
        gone = mach._tunnels[login, two.port]
        assert mach.remove_machine("two") is True
        assert set(mach._tunnels) == {(login, one.port)} and gone.process is None
        assert _pong(stream), "removing the second machine closed the first's forward"
        stream.close()
    finally:
        mach.close_links()
        one.stop()
        two.stop()
