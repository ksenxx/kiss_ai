# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the server-reset flag is published through the shared atomic writer.

``RemoteAccessServer._write_server_reset_flag`` hand-rolled its own
"write a temp file, then ``os.replace``" sequence next to the module's
``_atomic_write_text`` helper, and did it worse: the temp name was the
FIXED ``server-reset-pending.json.tmp`` (so sibling daemons sharing a
KISS home can interleave writes into one temp file — the exact hazard
``_atomic_publish``'s docstring documents and its pid/thread/uuid
suffix prevents) and a failed publish left the temp file behind (the
helper unlinks it).  These tests pin the shared helper's contract on
the flag path: no residue on failure, and a flag that is either absent
or a complete JSON document under concurrent writers from two server
instances (real threads, one shared directory).
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kiss.server.web_server import RemoteAccessServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _server(url_dir: Path) -> RemoteAccessServer:
    return RemoteAccessServer(
        host="127.0.0.1",
        port=_free_port(),
        work_dir=tempfile.mkdtemp(prefix="kiss_reset_flag_wd_"),
        url_file=url_dir / "remote-url.json",
    )


def test_flag_is_written_as_json_with_conn_id() -> None:
    url_dir = Path(tempfile.mkdtemp(prefix="kiss_reset_flag_"))
    server = _server(url_dir)
    server._write_server_reset_flag("conn-42")
    flag = server._server_reset_flag_path()
    data = json.loads(flag.read_text(encoding="utf-8"))
    assert data["conn_id"] == "conn-42"
    assert isinstance(data["requested_at"], float)
    assert sorted(p.name for p in url_dir.iterdir()) == [flag.name]


def test_failed_publish_leaves_no_temp_file() -> None:
    """A publish that cannot land must not leave a temp file behind."""
    url_dir = Path(tempfile.mkdtemp(prefix="kiss_reset_flag_"))
    server = _server(url_dir)
    flag = server._server_reset_flag_path()
    # A stray DIRECTORY at the flag path makes the final rename fail
    # (a file cannot replace a non-empty directory on any platform).
    flag.mkdir()
    (flag / "occupant").write_text("x", encoding="utf-8")
    server._write_server_reset_flag("conn")  # logs, does not raise
    assert flag.is_dir()
    leftovers = sorted(p.name for p in url_dir.iterdir() if p != flag)
    assert leftovers == [], (
        f"BUG: failed flag publish left temp residue behind: {leftovers}"
    )


@pytest.mark.timeout(60)
def test_concurrent_writers_never_expose_a_torn_flag() -> None:
    """Two server instances + many threads: every read is a whole document."""
    url_dir = Path(tempfile.mkdtemp(prefix="kiss_reset_flag_"))
    servers = [_server(url_dir), _server(url_dir)]
    flag = servers[0]._server_reset_flag_path()
    stop = threading.Event()
    torn: list[str] = []
    seen_any = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            try:
                raw = flag.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError:
                continue
            seen_any.set()
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                torn.append(raw)
                return
            if set(doc) != {"requested_at", "conn_id"}:
                torn.append(raw)
                return

    def _writer(server: RemoteAccessServer, tag: str) -> None:
        # A long conn id makes a torn (partially written) file more
        # likely to be caught by the reader if writes are not atomic.
        conn_id = tag * 4000
        for _ in range(150):
            server._write_server_reset_flag(conn_id)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    writers = [
        threading.Thread(target=_writer, args=(servers[i % 2], str(i)))
        for i in range(6)
    ]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    time.sleep(0.05)
    stop.set()
    reader.join(timeout=10)
    assert seen_any.is_set(), "reader never observed the flag"
    assert torn == [], f"torn flag observed: {torn[0][:80]!r}..."
    data = json.loads(flag.read_text(encoding="utf-8"))
    assert set(data) == {"requested_at", "conn_id"}
    assert sorted(p.name for p in url_dir.iterdir()) == [flag.name], (
        "temp files left behind after concurrent publishes"
    )
