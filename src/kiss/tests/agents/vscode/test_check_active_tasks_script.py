# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ``scripts/check-kiss-web-active-tasks.py``.

The helper is invoked from ``scripts/build-extension.sh`` and
``install.sh`` BEFORE either script SIGTERMs the kiss-web daemon.  Its
contract — documented in the script's module docstring — is:

* exit ``0`` when the daemon's UDS reports zero active tasks OR the
  socket is missing / refusing connections (daemon already dead);
* exit ``1`` when the daemon reports one or more active tasks OR the
  probe could not be completed (timeout, malformed response, etc.).

These tests reproduce the regression described in task_history rows
3233/3234 ("Task interrupted by server restart/shutdown"): a real
``RemoteAccessServer`` is started against a temp UDS, the registry is
populated with a fake active tab, and the helper is executed as a
subprocess with ``KISS_SORCAR_SOCK`` overridden to the temp socket.
The pre-fix scripts (no helper, unconditional ``lsof -ti :8787 | kill``)
would have killed the daemon here; the post-fix scripts gate on the
helper's exit code and refuse.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import cast
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[5]
    / "scripts" / "check-kiss-web-active-tasks.py"
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class _FakeActiveTab:
    """Duck-typed stand-in for a real ``_RunningAgentState``.

    ``_handle_active_tasks_query`` only reads three attributes from a
    registry entry: ``is_task_active``, ``task_history_id``, and
    ``last_task_id``.  Defining a tiny class avoids spinning up an
    entire agent worker just to satisfy the type contract.
    """

    def __init__(self, task_id: str) -> None:
        self.is_task_active = True
        self.task_history_id = task_id
        self.last_task_id = task_id


def _run_helper(sock_path: Path, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    """Run the helper script with ``KISS_SORCAR_SOCK`` overridden."""
    env = os.environ.copy()
    env["KISS_SORCAR_SOCK"] = str(sock_path)
    env["KISS_ACTIVE_TASKS_TIMEOUT"] = "2.0"
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class TestCheckActiveTasksScript(IsolatedAsyncioTestCase):
    """End-to-end coverage for the bash-callable active-tasks probe."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        # Ensure the registry is clean across tests — a leftover fake
        # tab from a different test in the same process would flip our
        # idle assertion.
        with _RunningAgentState._registry_lock:
            self._registry_snapshot = dict(
                _RunningAgentState.running_agent_states,
            )
            _RunningAgentState.running_agent_states.clear()

    async def asyncTearDown(self) -> None:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
            _RunningAgentState.running_agent_states.update(
                self._registry_snapshot,
            )
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_helper_script_file_exists_and_is_executable(self) -> None:
        """The helper exists at the path the bash scripts hard-code."""
        self.assertTrue(
            _SCRIPT_PATH.exists(),
            f"missing helper script at {_SCRIPT_PATH}",
        )
        # Mode 0o111 bits — readable and executable by owner at minimum.
        mode = _SCRIPT_PATH.stat().st_mode
        self.assertTrue(mode & 0o100, f"script not executable: {oct(mode)}")

    async def test_idle_daemon_exits_zero(self) -> None:
        """Helper exits 0 when the daemon's UDS reports count=0."""
        # Run subprocess in a worker so the asyncio loop is not blocked
        # while the helper does its synchronous UDS probe (the helper
        # itself talks to the asyncio UDS server we just started, so
        # the loop must remain free to serve it).
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_helper, self.uds_path,
        )
        self.assertEqual(
            result.returncode, 0,
            f"unexpected exit={result.returncode}\n"
            f"stderr={result.stderr}\nstdout={result.stdout}",
        )
        self.assertIn("idle (count=0)", result.stderr)

    async def test_active_task_exits_one(self) -> None:
        """Helper exits 1 when an active task is present.

        Reproduces the SIGTERM regression: pre-fix the bash scripts
        would kill the daemon here; post-fix they abort because the
        helper returns 1.
        """
        fake_tab_id = "ad4ecb65-2878-4c2c-9736-3bb9be18814a"
        fake = _FakeActiveTab("3233")
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states[fake_tab_id] = cast(
                _RunningAgentState, fake,
            )
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _run_helper, self.uds_path,
            )
        finally:
            with _RunningAgentState._registry_lock:
                _RunningAgentState.running_agent_states.pop(fake_tab_id, None)
        self.assertEqual(
            result.returncode, 1,
            f"helper failed to refuse on active tasks: exit={result.returncode}\n"
            f"stderr={result.stderr}\nstdout={result.stdout}",
        )
        self.assertIn("1 in-flight task", result.stderr)
        self.assertIn(fake_tab_id, result.stderr)
        self.assertIn("task=3233", result.stderr)
        self.assertIn("KISS_FORCE_RESTART", result.stderr)

    def test_missing_socket_exits_zero(self) -> None:
        """Helper exits 0 when the socket file is absent (daemon dead)."""
        missing_sock = Path(tempfile.mkdtemp()) / "nope" / "sorcar.sock"
        try:
            result = _run_helper(missing_sock)
            self.assertEqual(
                result.returncode, 0,
                f"unexpected exit={result.returncode}\n"
                f"stderr={result.stderr}\nstdout={result.stdout}",
            )
            self.assertIn("not present", result.stderr)
        finally:
            shutil.rmtree(missing_sock.parent.parent, ignore_errors=True)

    def _run_fake_uds_server(
        self,
        sock_path: Path,
        lines: list[bytes],
    ) -> tuple[socket.socket, threading.Thread]:
        """Spin up a one-shot AF_UNIX server that returns ``lines``.

        Each ``bytes`` element is written verbatim after the helper
        sends its ``activeTasksQuery`` line (so callers can include or
        omit the trailing newline as needed for the test case).  This
        bypasses ``RemoteAccessServer`` entirely so we can simulate an
        OLD daemon that doesn't recognise ``activeTasksQuery`` — the
        exact wire behaviour responsible for the install.sh abort in
        the user-reported bug.
        """
        if sock_path.exists():
            sock_path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(1)
        srv.settimeout(5.0)

        def _serve() -> None:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                conn.settimeout(5.0)
                # Drain the incoming activeTasksQuery before responding
                # so the helper's ``sendall`` does not race with the
                # close.
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                for line in lines:
                    conn.sendall(line)
            finally:
                conn.close()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        return srv, thread

    def test_old_daemon_unknown_command_exits_zero(self) -> None:
        """An OLD daemon's "Unknown command" error → exit 0.

        Reproduces the user report::

            kiss-web UDS probe at ~/.kiss/sorcar.sock returned
            unexpected message {'type': 'error', 'text':
            'Unknown command: activeTasksQuery'}; refusing to kill.

        Before the fix the helper read the first broadcast line, saw
        an unknown ``type``, and exited 1 — blocking install.sh from
        replacing the very daemon that lacked the handler.  After the
        fix the helper recognises the specific error string and exits
        0 so install.sh can proceed.
        """
        scratch = Path(tempfile.mkdtemp())
        sock_path = scratch / "old.sock"
        srv, thread = self._run_fake_uds_server(
            sock_path,
            [b'{"type":"error","text":"Unknown command: activeTasksQuery"}\n'],
        )
        try:
            result = _run_helper(sock_path)
            self.assertEqual(
                result.returncode, 0,
                f"expected exit 0 for OLD-daemon error; got "
                f"exit={result.returncode}\nstderr={result.stderr}\n"
                f"stdout={result.stdout}",
            )
            self.assertIn("predates", result.stderr)
            self.assertIn("activeTasksQuery", result.stderr)
        finally:
            srv.close()
            thread.join(timeout=2.0)
            shutil.rmtree(scratch, ignore_errors=True)

    def test_stray_broadcast_before_response_is_tolerated(self) -> None:
        """Helper drains stray broadcast lines and parses the real reply.

        ``RemoteAccessServer._uds_handler`` registers every connected
        client as a broadcast destination.  Unrelated event lines can
        therefore land on the wire BEFORE our ``activeTasksResponse``.
        The pre-fix helper read only the first line and rejected
        anything that wasn't an ``activeTasksResponse`` — this test
        locks in the fix by interleaving a stray broadcast with the
        real reply and asserting the helper still exits 0.
        """
        scratch = Path(tempfile.mkdtemp())
        sock_path = scratch / "noisy.sock"
        srv, thread = self._run_fake_uds_server(
            sock_path,
            [
                b'{"type":"event","name":"noise"}\n',
                b'{"type":"activeTasksResponse","count":0,"tabs":[]}\n',
            ],
        )
        try:
            result = _run_helper(sock_path)
            self.assertEqual(
                result.returncode, 0,
                f"expected exit 0 after skipping stray broadcast; got "
                f"exit={result.returncode}\nstderr={result.stderr}\n"
                f"stdout={result.stdout}",
            )
            self.assertIn("idle (count=0)", result.stderr)
        finally:
            srv.close()
            thread.join(timeout=2.0)
            shutil.rmtree(scratch, ignore_errors=True)

    def test_stale_socket_with_no_listener_exits_zero(self) -> None:
        """A socket file with no listener is "safe to kill" (daemon dead).

        After a crash the UDS file can persist on disk while no process
        listens on it.  ``connect`` then raises ``ConnectionRefusedError``
        (Linux) or ``FileNotFoundError`` depending on platform — both
        must map to ``count==0`` so the bash script does not block a
        legitimate cleanup when the daemon is already gone.
        """
        scratch = Path(tempfile.mkdtemp())
        stale_sock = scratch / "stale.sock"
        # Bind+close to create the file then unlink the listener by
        # closing the socket — leaves the inode but no acceptor.
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind(str(stale_sock))
            s.close()
            # ``bind`` leaves the inode behind; without ``listen`` the
            # connect attempt is refused.
            result = _run_helper(stale_sock)
            self.assertEqual(
                result.returncode, 0,
                f"unexpected exit={result.returncode}\n"
                f"stderr={result.stderr}\nstdout={result.stdout}",
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
