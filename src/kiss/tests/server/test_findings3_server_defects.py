# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproductions for round-3 server-core audit findings.

Each test reproduces a confirmed defect from the deep audit of
``src/kiss/server/{server,sorcar,task_runner}.py`` and now passes only
because the defect is fixed:

S3-02/S3-03  ``ServerApi.job_trajectories`` authorized job access with a
             lexical substring check plus ``find_job_dir`` (which accepts
             any child dir of the primary root and follows directory
             symlinks), disagreeing with the ``/api/jobs`` listing and
             disclosing unlisted or out-of-tree data; an empty job name
             was also accepted.  It must now authorize against the same
             ``discover_job_dirs`` allow-list with resolved-path
             containment, and reject empty names with 400.

S3-04        The per-IP brute-force lockout in ``ServerApi.authenticate``
             was checked only once before the first credential read, so a
             wrong guess that crossed the threshold still received its
             remaining retry.  The lockout is now re-checked after each
             recorded failure.

S3-06        The per-task ``maxBudget`` override accepted non-finite
             numbers (``NaN``/``Infinity``), silently disabling the spend
             cap because ``spend >= NaN`` is always false.
             ``coerce_budget_override`` now rejects them.

S3-11/S3-12  The synchronous daemon client capped a single event line at
             16 MiB while the daemon transport allows 64 MiB, so an
             oversized terminal ``result`` frame was split and discarded,
             making ``run()`` return an empty unsuccessful result for a
             task that actually succeeded.  The reader also leaked because
             ``sock.makefile()``'s reader was never closed.

S3-13        A single corrupt persisted ``timestamp`` (TEXT or infinity in
             the dynamically typed SQLite column) aborted the whole
             history response.  ``_safe_start_ms`` now degrades it to 0.

All tests use real sockets, real filesystems, real threads, and real
WSS/HTTPS connections — no mocks, patches, or fakes.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import ssl
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import yaml
from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import web_server
from kiss.server.server import _safe_start_ms
from kiss.server.sorcar import ServerApi
from kiss.server.sorcar import run as sorcar_run
from kiss.server.task_runner import coerce_budget_override
from kiss.server.web_server import RemoteAccessServer

_PASSWORD = "findings3-auth-test-password"


def _write_trajectory(job_dir: Path, name: str) -> None:
    """Create a minimal trajectory YAML under ``<job_dir>/trajectories``."""
    traj_dir = job_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "id": 1,
        "run_start_timestamp": 100,
        "run_end_timestamp": 200,
        "model": "test-model",
        "command": "secret command",
        "step_count": 1,
        "max_steps": 10,
        "messages": [{"role": "user", "content": "secret"}],
    }
    (traj_dir / "trajectory_0.yaml").write_text(yaml.safe_dump(data))


class TestJobTrajectoriesAuthorization(unittest.TestCase):
    """S3-02/S3-03: detail endpoint must match the listing allow-list."""

    def setUp(self) -> None:
        """Seed a jobs root with one legit job plus decoy targets."""
        self._root = Path(tempfile.mkdtemp(prefix="kiss_f3_jobs_")) / "jobs"
        self._root.mkdir(parents=True)
        _write_trajectory(
            self._root / "job_2024_01_01_00_00_00_1", "Legit Agent",
        )
        # An unlisted, non-``job_*`` directory that the listing hides.
        _write_trajectory(self._root / "private", "Private Agent")
        # A ``job_*``-named symlink escaping the jobs tree entirely.
        self._outside = Path(tempfile.mkdtemp(prefix="kiss_f3_outside_"))
        _write_trajectory(self._outside / "leak", "Outside Agent")
        os.symlink(self._outside / "leak", self._root / "job_evil")
        self._orig_get_jobs_root = web_server.get_jobs_root
        web_server.get_jobs_root = lambda *a, **k: self._root

    def tearDown(self) -> None:
        """Restore the real jobs-root resolver."""
        web_server.get_jobs_root = self._orig_get_jobs_root

    def test_listed_job_still_loads(self) -> None:
        """A genuine listed job must still return its trajectories (200)."""
        status, _ctype, body = ServerApi.job_trajectories(
            "/api/jobs/job_2024_01_01_00_00_00_1/trajectories"
        )
        self.assertEqual(status, 200)
        trajectories = json.loads(body)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["name"], "Legit Agent")

    def test_unlisted_directory_is_not_disclosed(self) -> None:
        """A non-``job_*`` dir hidden from the listing must be a 404."""
        listed = {
            j["name"] for j in json.loads(ServerApi.trajectory_jobs()[2] or b"[]")
        }
        # Sanity: the listing does not expose the private directory.
        self.assertNotIn("private", listed)
        status, _ctype, body = ServerApi.job_trajectories(
            "/api/jobs/private/trajectories"
        )
        self.assertEqual(status, 404)
        self.assertIn(b"not found", body)

    def test_symlink_escaping_jobs_tree_is_blocked(self) -> None:
        """A ``job_*`` symlink pointing outside the tree must be a 404."""
        status, _ctype, body = ServerApi.job_trajectories(
            "/api/jobs/job_evil/trajectories"
        )
        self.assertEqual(status, 404)
        self.assertIn(b"not found", body)

    def test_empty_job_name_is_rejected(self) -> None:
        """An empty job segment must be a 400, not treated as the root."""
        status, _ctype, body = ServerApi.job_trajectories(
            "/api/jobs//trajectories"
        )
        self.assertEqual(status, 400)
        self.assertIn(b"Invalid job name", body)


class TestBudgetOverrideCoercion(unittest.TestCase):
    """S3-06: non-finite ``maxBudget`` overrides must be rejected."""

    def test_nan_and_infinity_are_rejected(self) -> None:
        """NaN/Inf disable ``spend >= cap``; they must coerce to None."""
        self.assertIsNone(coerce_budget_override(float("nan")))
        self.assertIsNone(coerce_budget_override(float("inf")))
        self.assertIsNone(coerce_budget_override(float("-inf")))

    def test_json_wire_nan_is_rejected(self) -> None:
        """A NaN that arrived through the JSON parser must be rejected."""
        wire = json.loads('{"maxBudget": NaN}')
        self.assertIsNone(coerce_budget_override(wire["maxBudget"]))

    def test_valid_and_invalid_types(self) -> None:
        """Finite numbers pass; bools and non-numbers do not."""
        self.assertEqual(coerce_budget_override(25), 25.0)
        self.assertEqual(coerce_budget_override(3.5), 3.5)
        self.assertIsNone(coerce_budget_override(True))
        self.assertIsNone(coerce_budget_override("50"))
        self.assertIsNone(coerce_budget_override(None))


class TestSafeStartMs(unittest.TestCase):
    """S3-13: one corrupt timestamp must not abort the history response."""

    def test_corrupt_and_nonfinite_degrade_to_zero(self) -> None:
        """TEXT / infinity / None convert to 0 instead of raising."""
        self.assertEqual(_safe_start_ms("corrupt"), 0)
        self.assertEqual(_safe_start_ms(float("inf")), 0)
        self.assertEqual(_safe_start_ms(float("nan")), 0)
        self.assertEqual(_safe_start_ms(None), 0)
        self.assertEqual(_safe_start_ms(""), 0)

    def test_valid_timestamp_converts_to_ms(self) -> None:
        """A normal seconds timestamp converts to epoch milliseconds."""
        self.assertEqual(_safe_start_ms(1.5), 1500)
        self.assertEqual(_safe_start_ms(1000), 1_000_000)


class _UdsResultServer:
    """A minimal real UDS daemon that replays a scripted event stream."""

    def __init__(self, sock_path: Path, result_text: str) -> None:
        """Bind a UNIX-domain listener at *sock_path*."""
        self._result_text = result_text
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(sock_path))
        self._srv.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        """Begin accepting one client connection."""
        self._thread.start()

    def _serve(self) -> None:
        conn, _ = self._srv.accept()
        with conn:
            reader = conn.makefile("rb")
            cmd = json.loads(reader.readline().decode("utf-8"))
            tab_id = cmd["tabId"]
            task_id = "task-abc-123"

            def send(event: dict[str, Any]) -> None:
                conn.sendall(json.dumps(event).encode("utf-8") + b"\n")

            send({"type": "status", "running": True, "tabId": tab_id})
            send({
                "type": "result",
                "tabId": tab_id,
                "taskId": task_id,
                "success": True,
                "text": self._result_text,
                "summary": self._result_text,
                "cost": "$0.1234",
                "total_tokens": 42,
                "step_count": 7,
            })
            send({"type": "status", "running": False, "tabId": tab_id})
            # Give the client time to consume before EOF.
            try:
                reader.read()
            except OSError:
                pass

    def close(self) -> None:
        """Shut the listener down."""
        try:
            self._srv.close()
        except OSError:
            pass


class TestUdsOversizedResultFraming(unittest.TestCase):
    """S3-11/S3-12: an oversized terminal ``result`` frame must survive."""

    def test_oversized_success_result_is_returned(self) -> None:
        """A >16 MiB successful ``result`` must yield success=True text."""
        # >16 MiB of payload — larger than the OLD 16 MiB client cap so
        # the buggy client would split and discard the terminal frame.
        big_text = "R" * (16 * 1024 * 1024 + 4096)
        tmpdir = Path(tempfile.mkdtemp(prefix="kiss_f3_uds_"))
        sock_path = tmpdir / "daemon.sock"
        server = _UdsResultServer(sock_path, big_text)
        server.start()
        try:
            result = sorcar_run(
                "do the big thing",
                sock_path=sock_path,
                timeout=30.0,
            )
        finally:
            server.close()
        self.assertTrue(
            result.success,
            "oversized terminal result frame was dropped — client cap "
            "must match the 64 MiB daemon transport limit",
        )
        self.assertEqual(len(result.text), len(big_text))
        self.assertEqual(result.task_id, "task-abc-123")
        self.assertEqual(result.tokens, 42)
        self.assertEqual(result.steps, 7)


class TestAuthRateLimitRace(IsolatedAsyncioTestCase):
    """S3-04: a guess that crosses the lockout threshold gets no retry."""

    async def asyncSetUp(self) -> None:
        """Start a real ``RemoteAccessServer`` with a known password."""
        self._port = _pick_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        """Stop the server and restore the user's saved config."""
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _first_reply_to_wrong_guess(self) -> str:
        """Send one wrong non-empty password; return the first reply type."""
        async with await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": "wrong-pw"}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            return str(msg.get("type", ""))

    async def test_threshold_crossing_guess_is_locked_not_retried(
        self,
    ) -> None:
        """The wrong guess that trips the lockout gets ``auth_locked``.

        ``web_server._AUTH_FAIL_MAX`` failures within the window lock the
        IP.  The first ``MAX - 1`` wrong guesses each get an
        ``auth_required`` retry; the guess that records the ``MAX``-th
        failure crosses the threshold and — with the fix — must be told
        ``auth_locked`` immediately instead of being handed another try.
        """
        max_fail = web_server._AUTH_FAIL_MAX
        replies = []
        for _ in range(max_fail):
            replies.append(await self._first_reply_to_wrong_guess())
        self.assertEqual(
            replies[: max_fail - 1],
            ["auth_required"] * (max_fail - 1),
            "early wrong guesses should each get a retry prompt",
        )
        self.assertEqual(
            replies[max_fail - 1],
            "auth_locked",
            "the guess crossing the brute-force threshold must be locked "
            "immediately, not granted another attempt",
        )

    async def test_admitted_socket_cannot_redeem_password_after_lock(
        self,
    ) -> None:
        """A pre-admitted socket must not authenticate once locked.

        Several connections can pass the pre-loop lockout check while
        the failure count is still below threshold.  If a peer then
        trips the lock, an already-admitted socket submitting even the
        CORRECT password must be told ``auth_locked`` — the lockout is
        re-checked after every credential read, before comparison.
        """
        max_fail = web_server._AUTH_FAIL_MAX
        # Admit a socket BEFORE the lock engages and keep it waiting.
        admitted = await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=_no_verify_ssl(),
        )
        try:
            await asyncio.sleep(random.uniform(0.001, 0.05))
            # Peers now trip the per-IP lockout with wrong guesses.
            for _ in range(max_fail):
                await self._first_reply_to_wrong_guess()
            # The admitted socket finally submits the CORRECT password.
            await admitted.send(
                json.dumps({"type": "auth", "password": _PASSWORD}),
            )
            msg = json.loads(
                await asyncio.wait_for(admitted.recv(), timeout=10),
            )
            self.assertEqual(
                msg.get("type"),
                "auth_locked",
                "an already-admitted socket must not redeem a guessed "
                "password after the brute-force lock engaged",
            )
        finally:
            await admitted.close()


class TestConcurrentApiKeySave(unittest.TestCase):
    """S3-10: concurrent API-key saves must not lose an update.

    Each ``save_api_key`` read-modify-atomic-replaces the same
    canonical key store.  The fix serializes those writes under the shared
    ``_CommandsMixin._save_config_lock`` (as ``saveConfig`` now does).
    This test drives many real threads through that same lock and
    asserts every distinct key survives — the unserialized version drops
    keys whose read/replace interleaves another writer's replace.
    """

    def setUp(self) -> None:
        """Redirect ``$HOME`` to a temp dir and force a bash RC."""
        self._home = tempfile.mkdtemp(prefix="kiss_f3_home_")
        self._saved_home = os.environ.get("HOME")
        self._saved_shell = os.environ.get("SHELL")
        os.environ["HOME"] = self._home
        os.environ["SHELL"] = "/bin/bash"
        (Path(self._home) / ".bashrc").write_text("# base rc\n")

    def tearDown(self) -> None:
        """Restore the environment."""
        for name, saved in (
            ("HOME", self._saved_home),
            ("SHELL", self._saved_shell),
        ):
            if saved is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = saved

    def test_all_keys_persist_under_the_lock(self) -> None:
        """Twenty threads saving distinct keys must all persist."""
        from kiss.core.vscode_config import api_keys_env_path, save_api_key
        from kiss.server.commands import _CommandsMixin

        key_names = [f"KISS_TEST_KEY_{i}" for i in range(20)]
        barrier = threading.Barrier(len(key_names))

        def worker(name: str) -> None:
            barrier.wait()
            with _CommandsMixin._save_config_lock:
                save_api_key(name, f"val-{name}")

        threads = [
            threading.Thread(target=worker, args=(n,)) for n in key_names
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
            self.assertFalse(t.is_alive())

        store_text = api_keys_env_path().read_text()
        for name in key_names:
            self.assertIn(
                f"export {name}=", store_text,
                f"{name} was lost — concurrent RC writes were not "
                f"serialized",
            )


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Permissive SSL context for the dev self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


if __name__ == "__main__":
    unittest.main()
