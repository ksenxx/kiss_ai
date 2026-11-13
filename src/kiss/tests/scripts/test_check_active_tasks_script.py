# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ``scripts/check-kiss-web-active-tasks.py``.

The helper is invoked from ``scripts/build-extension.sh`` and
``install.sh`` BEFORE either script SIGTERMs the kiss-web daemon.  Its
contract — documented in the script's module docstring — is:

* exit ``0`` when the daemon's local endpoint reports zero active tasks
  OR the endpoint file is missing / the daemon refuses connections
  (daemon already dead);
* exit ``1`` when the daemon reports one or more active tasks OR the
  probe could not be completed (timeout, malformed response, etc.).

These tests reproduce the regression described in task_history rows
3233/3234 ("Task interrupted by server restart/shutdown"): a real
``RemoteAccessServer`` is started with a temp endpoint file, the
registry is populated with a fake active tab, and the helper is
executed as a subprocess with ``KISS_SORCAR_LOCAL`` overridden to the
temp endpoint file.  The pre-fix scripts (no helper, unconditional
``lsof -ti :8787 | kill``) would have killed the daemon here; the
post-fix scripts gate on the helper's exit code and refuse.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.server import ServerConnection

import kiss.agents.sorcar.persistence as th
import kiss.core.config
from kiss.agents.sorcar import local_endpoint
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import _generate_self_signed_cert, fake_daemon

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts" / "check-kiss-web-active-tasks.py"
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, Path]:
    saved = (th._DB_PATH, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    return saved


def _restore_persistence(saved: tuple[Path, Path]) -> None:
    th._DB_PATH, th._KISS_DIR = saved


def _run_helper(
    endpoint_file: Path, timeout: float = 5.0, strict: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the helper script with ``KISS_SORCAR_LOCAL`` overridden."""
    env = os.environ.copy()
    env["KISS_SORCAR_LOCAL"] = str(endpoint_file)
    if strict:
        env["KISS_ACTIVE_TASKS_STRICT"] = "1"
    # Windows reports a refused loopback connect only after ~2 s.
    env["KISS_ACTIVE_TASKS_TIMEOUT"] = "5.0" if sys.platform == "win32" else "2.0"
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run_probe(env: dict[str, str], *options: str) -> subprocess.CompletedProcess[str]:
    """Run the helper under exactly *env*, the interpreter started with *options*."""
    return subprocess.run(
        [sys.executable, *options, str(_SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _branded_app(root: Path, home_dir: str) -> Path:
    """Build in *root* what ``kiss.core.config`` imports, under a brand whose home is *home_dir*.

    Returns the directory to put on ``PYTHONPATH``: the app of a
    white-label install, whose daemon keeps its state in ``~/<home_dir>``.
    """
    core = Path(kiss.core.config.__file__).resolve().parent
    (root / "kiss" / "core").mkdir(parents=True)
    shutil.copy(core.parent / "__init__.py", root / "kiss")
    for module in core.glob("*.py"):
        shutil.copy(module, root / "kiss" / "core")
    media = root / "kiss" / "webapp" / "media"
    media.mkdir(parents=True)
    (media / "brand.json").write_text(json.dumps({"home_dir": home_dir}), encoding="utf-8")
    return root


def _one_shot_handler(lines: list[bytes]):
    """Build a fake-daemon handler that answers the probe with *lines*.

    Each ``bytes`` element is sent verbatim as one frame after the
    helper's ``activeTasksQuery`` arrives.  This bypasses
    ``RemoteAccessServer`` entirely so a test can simulate an OLD
    daemon that doesn't recognise ``activeTasksQuery`` — the exact wire
    behaviour responsible for the install.sh abort in the user-reported
    bug — or a daemon that pushes stray broadcasts first.
    """

    async def _handler(ws: ServerConnection) -> None:
        await ws.recv()  # the activeTasksQuery frame
        for line in lines:
            await ws.send(line.decode("utf-8").rstrip("\n"))

    return _handler


class TestCheckActiveTasksScript(IsolatedAsyncioTestCase):
    """End-to-end coverage for the bash-callable active-tasks probe.

    The probe talks to the daemon over its local WSS endpoint on behalf
    of the POSIX shell scripts.
    """

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.endpoint_file = Path(self.tmpdir) / "sorcar-local.json"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=self.endpoint_file,
        )
        await self.server.start_async()
        with agent_state.STATE_LOCK:
            self._registry_snapshot = dict(agent_state.agent_states)
            agent_state.agent_states.clear()

    async def asyncTearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
            agent_state.agent_states.update(self._registry_snapshot)
        await self.server.stop_async()
        th._close_thread_db()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @unittest.skipIf(sys.platform == "win32", "NTFS has no executable bit")
    def test_helper_script_file_exists_and_is_executable(self) -> None:
        """The helper exists at the path the bash scripts hard-code."""
        self.assertTrue(
            _SCRIPT_PATH.exists(),
            f"missing helper script at {_SCRIPT_PATH}",
        )
        mode = _SCRIPT_PATH.stat().st_mode
        self.assertTrue(mode & 0o100, f"script not executable: {oct(mode)}")

    async def test_idle_daemon_exits_zero(self) -> None:
        """Helper exits 0 when the daemon's local endpoint reports count=0."""
        result = await asyncio.to_thread(_run_helper, self.endpoint_file)
        self.assertEqual(
            result.returncode, 0,
            f"unexpected exit={result.returncode}\n"
            f"stderr={result.stderr}\nstdout={result.stdout}",
        )
        self.assertIn("idle (count=0)", result.stderr)
        self.assertEqual(result.stdout, "")

    async def test_active_task_exits_one(self) -> None:
        """Helper exits 1 when an active task is present.

        Reproduces the SIGTERM regression: pre-fix the bash scripts
        would kill the daemon here; post-fix they abort because the
        helper returns 1.
        """
        fake_tab_id = "ad4ecb65-2878-4c2c-9736-3bb9be18814a"
        active = agent_state.AgentState(
            "3233",
            tab_id=fake_tab_id,
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(active)
        try:
            result = await asyncio.to_thread(_run_helper, self.endpoint_file)
        finally:
            agent_state.unregister("3233", active)
        self.assertEqual(
            result.returncode, 1,
            f"helper failed to refuse on active tasks: exit={result.returncode}\n"
            f"stderr={result.stderr}\nstdout={result.stdout}",
        )
        self.assertIn("1 in-flight task", result.stderr)
        self.assertIn(fake_tab_id, result.stderr)
        self.assertIn("task=3233", result.stderr)
        # Nothing reads KISS_FORCE_RESTART any more: the message must not
        # offer it as an override.
        self.assertIn("Refusing to kill.", result.stderr)
        self.assertNotIn("KISS_FORCE_RESTART", result.stderr)
        # The busy daemon names the file install.sh records its deferred
        # restart in: next to its remote-url.json.
        self.assertEqual(
            result.stdout, str(Path(self.tmpdir) / ".kiss-web.restart-pending") + "\n",
        )

    async def test_busy_daemon_that_names_no_record_file_prints_nothing(self) -> None:
        """A daemon from before deferred restarts: exit 1, empty stdout."""
        scratch = Path(tempfile.mkdtemp())
        handler = _one_shot_handler(
            [b'{"type":"activeTasksResponse","count":1,"tabs":["t"]}\n'],
        )
        try:
            async with fake_daemon(scratch, handler) as endpoint_file:
                result = await asyncio.to_thread(_run_helper, endpoint_file, 5.0, True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("1 in-flight task", result.stderr)
            self.assertEqual(result.stdout, "")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_missing_endpoint_file_exits_zero(self) -> None:
        """Helper exits 0 when the endpoint file is absent (daemon dead)."""
        missing = Path(tempfile.mkdtemp()) / "nope" / "sorcar-local.json"
        try:
            result = _run_helper(missing)
            self.assertEqual(
                result.returncode, 0,
                f"unexpected exit={result.returncode}\n"
                f"stderr={result.stderr}\nstdout={result.stdout}",
            )
            self.assertIn("not present", result.stderr)
        finally:
            shutil.rmtree(missing.parent.parent, ignore_errors=True)

    async def test_old_daemon_unknown_command_exits_zero(self) -> None:
        """An OLD daemon's "Unknown command" error → exit 0.

        Reproduces the user report::

            kiss-web probe at wss://127.0.0.1:8787/ws returned
            unexpected message {'type': 'error', 'text':
            'Unknown command: activeTasksQuery'}; refusing to kill.

        Before the fix the helper read the first broadcast frame, saw
        an unknown ``type``, and exited 1 — blocking install.sh from
        replacing the very daemon that lacked the handler.  After the
        fix the helper recognises the specific error string and exits
        0 so install.sh can proceed.
        """
        scratch = Path(tempfile.mkdtemp())
        handler = _one_shot_handler(
            [b'{"type":"error","text":"Unknown command: activeTasksQuery"}\n'],
        )
        try:
            async with fake_daemon(scratch, handler) as endpoint_file:
                result = await asyncio.to_thread(_run_helper, endpoint_file)
            self.assertEqual(
                result.returncode, 0,
                f"expected exit 0 for OLD-daemon error; got "
                f"exit={result.returncode}\nstderr={result.stderr}\n"
                f"stdout={result.stdout}",
            )
            self.assertIn("predates", result.stderr)
            self.assertIn("activeTasksQuery", result.stderr)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    async def test_old_daemon_exits_one_in_strict_mode(self) -> None:
        """``KISS_ACTIVE_TASKS_STRICT=1``: an OLD daemon's answer is "unknown".

        install.sh's runtime setup restarts a daemon it did not build and
        follows the extension's ``unsupported-query`` policy: a daemon
        that cannot report its tasks is not proven idle.
        """
        scratch = Path(tempfile.mkdtemp())
        handler = _one_shot_handler(
            [b'{"type":"error","text":"Unknown command: activeTasksQuery"}\n'],
        )
        try:
            async with fake_daemon(scratch, handler) as endpoint_file:
                result = await asyncio.to_thread(_run_helper, endpoint_file, 5.0, True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("predates", result.stderr)
            self.assertIn("Refusing to kill", result.stderr)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    async def test_stray_broadcast_before_response_is_tolerated(self) -> None:
        """Helper drains stray broadcast frames and parses the real reply.

        ``RemoteAccessServer._ws_handler`` registers every connected
        local client as a broadcast destination.  Unrelated event frames
        can therefore land on the wire BEFORE our ``activeTasksResponse``.
        The pre-fix helper read only the first frame and rejected
        anything that wasn't an ``activeTasksResponse`` — this test
        locks in the fix by interleaving a stray broadcast with the
        real reply and asserting the helper still exits 0.
        """
        scratch = Path(tempfile.mkdtemp())
        handler = _one_shot_handler(
            [
                b'{"type":"event","name":"noise"}\n',
                b'{"type":"activeTasksResponse","count":0,"tabs":[]}\n',
            ],
        )
        try:
            async with fake_daemon(scratch, handler) as endpoint_file:
                result = await asyncio.to_thread(_run_helper, endpoint_file)
            self.assertEqual(
                result.returncode, 0,
                f"expected exit 0 after skipping stray broadcast; got "
                f"exit={result.returncode}\nstderr={result.stderr}\n"
                f"stdout={result.stdout}",
            )
            self.assertIn("idle (count=0)", result.stderr)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_stale_endpoint_file_with_no_listener_exits_zero(self) -> None:
        """An endpoint file with no listener is "safe to kill" (daemon dead).

        After a crash the endpoint file can persist on disk while no
        process listens on its URL.  ``connect`` then raises
        ``ConnectionRefusedError``, which must map to ``count==0`` so the
        bash script does not block a legitimate cleanup when the daemon
        is already gone.
        """
        scratch = Path(tempfile.mkdtemp())
        stale_file = scratch / "sorcar-local.json"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = int(s.getsockname()[1])
        try:
            local_endpoint.write_endpoint(
                stale_file,
                local_endpoint.LocalEndpoint(
                    url=f"ws://127.0.0.1:{free_port}/ws", token="stale",
                    ca=None, pid=os.getpid(),
                ),
            )
            self.assertIsNotNone(json.loads(stale_file.read_text())["url"])
            result = _run_helper(stale_file)
            self.assertEqual(
                result.returncode, 0,
                f"unexpected exit={result.returncode}\n"
                f"stderr={result.stderr}\nstdout={result.stdout}",
            )
            self.assertIn("refused connection", result.stderr)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


class TestDefaultEndpointFile(IsolatedAsyncioTestCase):
    """Without ``KISS_SORCAR_LOCAL`` the probe asks the daemon of its interpreter's app.

    The probe defaulted to ``~/.kiss/sorcar-local.json`` whatever the
    install: on a white-label one, whose daemon publishes its endpoint
    in the brand's home (``~/.s10s``), a busy daemon read as "no daemon,
    safe to kill".
    """

    def setUp(self) -> None:
        self.scratch = Path(tempfile.mkdtemp())
        self.home = self.scratch / "home"
        self.home.mkdir()
        # Nothing of the test runner's environment: no KISS_HOME, no
        # KISS_SORCAR_LOCAL, no PYTHONPATH.  ``USERPROFILE`` is
        # ``Path.home()`` on Windows, where python.exe needs ``SYSTEMROOT``.
        self.env = {
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "PATH": os.defpath,
            "KISS_ACTIVE_TASKS_TIMEOUT": "5.0",
            **{name: os.environ[name] for name in ("SYSTEMROOT",) if name in os.environ},
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    async def test_branded_app_asks_the_daemon_in_the_brand_home(self) -> None:
        """A busy daemon in ``~/.s10s`` is busy; ``$KISS_HOME`` names another home."""
        app = _branded_app(self.scratch / "app", ".s10s")
        env = {**self.env, "PYTHONPATH": str(app)}
        endpoint_file = self.home / ".s10s" / "sorcar-local.json"
        endpoint_file.parent.mkdir()
        handler = _one_shot_handler(
            [b'{"type":"activeTasksResponse","count":1,"tabs":["t"]}\n'],
        )
        elsewhere = self.scratch / "custom"
        async with fake_daemon(self.scratch, handler, endpoint_file=endpoint_file):
            busy = await asyncio.to_thread(_run_probe, env)
            moved = await asyncio.to_thread(_run_probe, {**env, "KISS_HOME": str(elsewhere)})
        self.assertEqual(busy.returncode, 1, busy.stderr)
        self.assertIn("1 in-flight task", busy.stderr)
        self.assertEqual(moved.returncode, 0, moved.stderr)
        self.assertIn(f"not present at {elsewhere / 'sorcar-local.json'}", moved.stderr)

    def test_python_without_the_package_asks_the_stock_home(self) -> None:
        """A bare interpreter cannot know the brand: ``$KISS_HOME``, else ``~/.kiss``."""
        stock = _run_probe(self.env, "-S")
        self.assertEqual(stock.returncode, 0, stock.stderr)
        self.assertIn(
            f"not present at {self.home / '.kiss' / 'sorcar-local.json'}", stock.stderr,
        )
        elsewhere = self.scratch / "custom"
        moved = _run_probe({**self.env, "KISS_HOME": str(elsewhere)}, "-S")
        self.assertEqual(moved.returncode, 0, moved.stderr)
        self.assertIn(f"not present at {elsewhere / 'sorcar-local.json'}", moved.stderr)
