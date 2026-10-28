# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for the 2026-10-26 ``sorcar-agent`` sweep.

* A bridge that answers a malformed HTTP reply (a bare status line, a
  body shorter than its ``Content-Length``) is reported as a
  ``KISSError`` by :func:`bridge_exec`, as ``False`` by
  :func:`bridge_alive` and as a failed command by
  ``ExecBridge.run_commands_parallel`` instead of an ``IndexError`` or a
  raw ``http.client`` exception.
* The run's ``UsefulTools`` instance is dropped with the other per-run
  hooks, so a later run (a Docker one never rebinds it) does not list
  the earlier task's written files in its continuation message.
* A Stop (``KeyboardInterrupt``) during the commit-message model call
  still sends the terminal ``failed`` notification, so the sticky
  "Generating commit message" toast is closed.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from kiss.agents.sorcar.exec_bridge import ExecBridge, bridge_alive, bridge_exec
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, auto_commit_changes
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_error import KISSError


def _serve_raw(payload: bytes, conn: socket.socket) -> None:
    """Read one request head, answer *payload* verbatim, close."""
    conn.settimeout(5)
    try:
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
        conn.sendall(payload)
    finally:
        conn.close()


@contextmanager
def _raw_http_server(payload: bytes) -> Iterator[str]:
    """Serve *payload* as the reply to every connection on a local port; yield the URL."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    server.settimeout(0.2)
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            _serve_raw(payload, conn)

    thread = threading.Thread(target=loop, name="raw-http", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}"
    finally:
        stop.set()
        thread.join(5)
        server.close()


BAD_STATUS_LINE = b"garbage\r\n\r\n"
# A valid head whose body stops short of its Content-Length: ``read()``
# raises ``http.client.IncompleteRead``.
TRUNCATED_BODY = (
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n"
    b'{"exit_code": 0, "output": "hi'
)


@pytest.mark.parametrize("payload", [BAD_STATUS_LINE, TRUNCATED_BODY])
def test_bridge_exec_reports_a_malformed_reply_as_kiss_error(payload: bytes) -> None:
    with _raw_http_server(payload) as url:
        with pytest.raises(KISSError, match="Exec bridge"):
            bridge_exec(url, "echo hi", 5)


@pytest.mark.parametrize("payload", [BAD_STATUS_LINE, TRUNCATED_BODY])
def test_bridge_alive_is_false_on_a_malformed_reply(payload: bytes) -> None:
    with _raw_http_server(payload) as url:
        assert bridge_alive(url) is False
        with pytest.raises(KISSError, match="not answering"):
            ExecBridge("exec:" + url).__enter__()


def test_run_commands_parallel_turns_a_malformed_reply_into_a_failed_command() -> None:
    with _raw_http_server(BAD_STATUS_LINE) as url:
        report = ExecBridge("exec:" + url).run_commands_parallel(
            '["echo one", "echo two"]', timeout_seconds=5,
        )
    assert report.startswith("2 commands: 0 succeeded, 2 failed, 0 timed out.")
    assert report.count("Exec bridge") == 2


class _Hooks:
    """Record ``notify_fn`` stages of :func:`auto_commit_changes`."""

    def __init__(self) -> None:
        self.stages: list[str] = []

    def __call__(self, stage: str, subject: str) -> None:
        self.stages.append(stage)


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"], check=True)
    (path / "README.md").write_text("# t\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "initial"], check=True)
    return path


def _stop_during_message(*_args: object) -> str:
    raise KeyboardInterrupt


def test_stop_during_commit_message_sends_the_terminal_failed_notification() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = _git_repo(Path(tmp) / "repo")
        (repo / "f.txt").write_text("hi\n")
        hooks = _Hooks()
        with pytest.raises(KeyboardInterrupt):
            auto_commit_changes(
                repo, user_prompt="add f", message_fn=_stop_during_message, notify_fn=hooks,
            )
        assert hooks.stages == ["generating", "failed"]
        log = subprocess.run(
            ["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True,
            check=True,
        ).stdout
        assert log.count("\n") == 1  # only "initial"; the Stop made no commit


def test_release_run_hooks_drops_the_previous_runs_written_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tools = UsefulTools(work_dir=tmp)
        tools.Write(str(Path(tmp) / "old_task_file.py"), "x = 1\n")
        agent = SorcarAgent("t")
        agent._useful_tools = tools
        agent._begin_task(None)
        assert "old_task_file.py" in agent._continuation_state()
        agent._finish_run(unwinding=False)
        assert agent._useful_tools is None
        # The next run (a Docker one never rebinds ``_useful_tools``)
        # starts with no inherited "Files written so far".
        agent._begin_task(None)
        assert agent._continuation_state() == ""
