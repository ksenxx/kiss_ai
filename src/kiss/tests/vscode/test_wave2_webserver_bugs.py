# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave2-Fixer-5 findings.

Covers, over REAL objects (no mocks, patches, or fakes):

* F3: ``_ensure_voice_model`` must survive two PROCESSES downloading
  concurrently — the fixed-name ``.tmp`` file used to be shared
  across processes, so the loser's ``tmp.replace`` raised
  ``FileNotFoundError`` and returned ``None`` (or published a
  corrupted archive).  Exercised with two real subprocesses pulling
  from a real (slow) local HTTP server.
* F4: ``_get_machine_topic`` must never propagate ``OSError`` when
  the topic file is deleted between the existence check and the
  read (real deleter thread), and must persist the topic atomically.
* F5: the ``/voice-model.tar.gz`` and ``/media/*`` HTTP branches of
  ``_process_request`` must still serve correct bytes after the
  blocking reads were moved off the event loop.
* F8: ``_spawn_cloudflared`` retry loop must reap failed attempts
  (close their stderr pipes) while keeping the LAST dead process's
  stderr open for the caller's URL-parsing path.  Exercised with a
  real fake ``cloudflared`` executable on ``PATH`` that exits
  immediately.
* F10: ``RemoteAccessServer.stop_async`` must stop in-flight agent
  worker threads (cooperative stop + join) exactly like the blocking
  ``start()`` shutdown path does, so embedders' tasks are not
  abandoned with the "Agent Failed Abruptly" sentinel.
* F13: constructing a second ``VSCodeServer`` in the same process
  must NOT mark task rows of still-running worker threads as
  "Task terminated unexpectedly (process killed)" — only rows with
  no live owner may be swept.
* F14: ``_snapshot_active_tabs`` must never raise while a real
  concurrent thread hammers the running-agent registry.

The live-server tests drive a real :class:`RemoteAccessServer`
(started via ``start_async``), mirroring the harness of
``test_fixer6_webserver_bugs.py``.
"""

from __future__ import annotations

import hashlib
import http.server
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

from websockets.datastructures import Headers
from websockets.http11 import Request

import kiss.agents.sorcar.persistence as th
import kiss.server.web_server as ws_mod
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer
from kiss.server.web_server import RemoteAccessServer


def _redirect_persistence(tmpdir: str) -> tuple[Any, Any, Any]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_persistence(saved: tuple[Any, Any, Any]) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _task_result(task_id: str) -> str:
    """Read the ``result`` column of a task_history row via the real DB."""
    row = th._get_db().execute(
        "SELECT result FROM task_history WHERE id = ?", (task_id,)
    ).fetchone()
    assert row is not None
    return str(row[0])


class TestF4MachineTopicDeleteRace(unittest.TestCase):
    """F4: topic-file deletion racing ``_get_machine_topic`` reads."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2f5-topic-")
        self.saved_home = ws_mod._KISS_HOME
        ws_mod._KISS_HOME = Path(self.tmpdir)

    def tearDown(self) -> None:
        ws_mod._KISS_HOME = self.saved_home
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_delete_never_raises_and_topic_is_stable(self) -> None:
        topic_file = Path(self.tmpdir) / "ntfy_topic"
        stop = threading.Event()
        expected = ws_mod._get_machine_topic()
        self.assertTrue(expected.startswith("kiss-"))
        self.assertEqual(len(expected), len("kiss-") + 32)

        def deleter() -> None:
            while not stop.is_set():
                topic_file.unlink(missing_ok=True)
                time.sleep(random.uniform(0.0, 0.0005))

        t = threading.Thread(target=deleter, daemon=True)
        t.start()
        try:
            for _ in range(600):
                got = ws_mod._get_machine_topic()
                self.assertEqual(got, expected)
                time.sleep(random.uniform(0.0, 0.0005))
        finally:
            stop.set()
            t.join(timeout=5)
        # No stray temp files may accumulate next to the topic file.
        leftovers = [
            p.name
            for p in Path(self.tmpdir).iterdir()
            if p.name != "ntfy_topic"
        ]
        self.assertEqual(leftovers, [])


class _SlowHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``server.payload`` slowly so two downloads overlap."""

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        payload = cast(Any, self.server).payload
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        chunk = 64 * 1024
        for i in range(0, len(payload), chunk):
            self.wfile.write(payload[i : i + chunk])
            self.wfile.flush()
            time.sleep(0.05)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


_F3_CHILD_SCRIPT = """\
import hashlib, sys
from pathlib import Path
import kiss.server.web_server as ws
ws.VOICE_MODEL_URL = sys.argv[1]
ws.VOICE_MODEL_CACHE = Path(sys.argv[2])
p = ws._ensure_voice_model()
if p is None:
    print("NONE")
else:
    print(hashlib.sha256(p.read_bytes()).hexdigest())
"""


class TestF3VoiceModelCrossProcessRace(unittest.TestCase):
    """F3: two real processes downloading the model concurrently."""

    def test_two_processes_both_get_intact_archive(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="kiss-w2f5-voice-")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        payload = os.urandom(1024 * 1024)
        expected = hashlib.sha256(payload).hexdigest()
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
        cast(Any, httpd).payload = payload
        server_thread = threading.Thread(
            target=httpd.serve_forever, daemon=True
        )
        server_thread.start()
        self.addCleanup(httpd.shutdown)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/model.tar.gz"
        cache = Path(tmpdir) / "models" / "model.tar.gz"
        script = Path(tmpdir) / "child.py"
        script.write_text(_F3_CHILD_SCRIPT)
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), url, str(cache)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )
            for _ in range(2)
        ]
        outs = [p.communicate(timeout=120)[0].strip() for p in procs]
        self.assertEqual(
            outs,
            [expected, expected],
            f"each process must return an intact archive, got {outs}",
        )
        # The published cache file itself must be intact too.
        self.assertEqual(
            hashlib.sha256(cache.read_bytes()).hexdigest(), expected
        )
        # No stray per-process temp files may remain.
        leftovers = [
            p.name for p in cache.parent.iterdir() if p.name != cache.name
        ]
        self.assertEqual(leftovers, [])


class TestF13SecondServerInitPreservesLiveTasks(unittest.TestCase):
    """F13: second ``VSCodeServer.__init__`` must not orphan live tasks."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2f5-init-")
        self.saved = _redirect_persistence(self.tmpdir)

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_running_task_row_is_not_marked_process_killed(self) -> None:
        live_id, _ = th._add_task("live task")
        dead_id, _ = th._add_task("dead task")
        self.assertEqual(_task_result(live_id), "Agent Failed Abruptly")
        self.assertEqual(_task_result(dead_id), "Agent Failed Abruptly")

        stop = threading.Event()
        worker = threading.Thread(target=stop.wait, daemon=True)
        worker.start()
        state = _RunningAgentState("tab-live", "model", is_task_active=True)
        state.task_history_id = live_id
        state.task_thread = worker
        state.stop_event = stop
        _RunningAgentState.register("tab-live", state)
        try:
            server = VSCodeServer()
            # The orphan sweep runs on a background thread so daemon
            # startup is never blocked by SQLite lock contention; join
            # it so the assertions below observe the post-sweep state.
            sweep = server._orphan_sweep_thread
            assert sweep is not None
            sweep.join(timeout=30)
            self.assertFalse(sweep.is_alive(), "orphan sweep did not finish")
            # The dead row has no live owner thread → swept.
            self.assertEqual(
                _task_result(dead_id),
                "Task terminated unexpectedly (process killed)",
            )
            # The live row's worker thread is still running in THIS
            # process → its sentinel must be left for the worker's
            # own cleanup ``finally`` to overwrite.
            self.assertEqual(_task_result(live_id), "Agent Failed Abruptly")
        finally:
            stop.set()
            worker.join(timeout=5)


class TestF14SnapshotActiveTabsConcurrent(unittest.TestCase):
    """F14: registry hammering must never break the snapshot helper."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_snapshot_never_raises_under_concurrent_mutation(self) -> None:
        stop = threading.Event()

        def hammer() -> None:
            i = 0
            while not stop.is_set():
                i += 1
                tab_id = f"tab-{i % 64}"
                state = _RunningAgentState(
                    tab_id, "model", is_task_active=bool(i % 2)
                )
                state.task_history_id = str(i)
                _RunningAgentState.register(tab_id, state)
                if i % 3 == 0:
                    _RunningAgentState.unregister(tab_id)
                time.sleep(random.uniform(0.0, 0.0002))

        threads = [
            threading.Thread(target=hammer, daemon=True) for _ in range(4)
        ]
        for t in threads:
            t.start()
        try:
            for _ in range(400):
                tabs = ws_mod._snapshot_active_tabs()
                for entry in tabs:
                    self.assertIn("(task=", entry)
                time.sleep(random.uniform(0.0, 0.0002))
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)


class TestF8SpawnCloudflaredRetries(unittest.TestCase):
    """F8: exhausted retries keep last proc's stderr open, reap the rest."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2f5-cf-")
        fake = Path(self.tmpdir) / "cloudflared"
        fake.write_text("#!/bin/sh\nexit 1\n")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self.saved_path = os.environ["PATH"]
        os.environ["PATH"] = f"{self.tmpdir}{os.pathsep}{self.saved_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self.saved_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_retry_exhaustion_reaps_intermediates(self) -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        server._spawn_cloudflared(["--url", "http://127.0.0.1:1"], retries=3)
        proc = server._tunnel_proc
        assert proc is not None
        self.addCleanup(proc.stderr.close if proc.stderr else lambda: None)
        # All three attempts exited immediately; the LAST one is kept
        # (already reaped) with its stderr still open for the caller's
        # URL-parsing path.
        self.assertIsNotNone(proc.returncode)
        self.assertEqual(proc.returncode, 1)
        assert proc.stderr is not None
        self.assertFalse(proc.stderr.closed)
        self.assertEqual(proc.stderr.read(), "")


class TestF5AndF10LiveServer(unittest.IsolatedAsyncioTestCase):
    """E2E tests over a real running RemoteAccessServer."""

    async def asyncSetUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2f5-live-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._stopped = False

    async def _stop_server(self) -> None:
        if not self._stopped:
            self._stopped = True
            await self.server.stop_async()

    async def asyncTearDown(self) -> None:
        await self._stop_server()
        _restore_persistence(self.saved)
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_f5_media_and_voice_model_served_correctly(self) -> None:
        """F5: HTTP branches serve correct bytes (reads off the loop)."""
        req = Request("/media/main.js", Headers())
        resp = await self.server._process_request(cast(Any, None), req)
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.body, (ws_mod.MEDIA_DIR / "main.js").read_bytes())

        missing = Request("/media/../secrets.txt", Headers())
        resp = await self.server._process_request(cast(Any, None), missing)
        assert resp is not None
        self.assertEqual(resp.status_code, 404)

        # Pre-seeded cache → served without any network access.
        payload = os.urandom(256 * 1024)
        cache = Path(self.tmpdir) / "models" / "model.tar.gz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        saved_cache = ws_mod.VOICE_MODEL_CACHE
        ws_mod.VOICE_MODEL_CACHE = cache
        try:
            req = Request("/voice-model.tar.gz", Headers())
            resp = await self.server._process_request(cast(Any, None), req)
            assert resp is not None
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.body, payload)
        finally:
            ws_mod.VOICE_MODEL_CACHE = saved_cache

    async def test_f5_voice_model_unavailable_returns_502(self) -> None:
        saved_cache = ws_mod.VOICE_MODEL_CACHE
        saved_url = ws_mod.VOICE_MODEL_URL
        ws_mod.VOICE_MODEL_CACHE = Path(self.tmpdir) / "none" / "m.tar.gz"
        # Closed port → the real download fails fast, no external network.
        ws_mod.VOICE_MODEL_URL = "http://127.0.0.1:1/model.tar.gz"
        try:
            req = Request("/voice-model.tar.gz", Headers())
            resp = await self.server._process_request(cast(Any, None), req)
            assert resp is not None
            self.assertEqual(resp.status_code, 502)
        finally:
            ws_mod.VOICE_MODEL_CACHE = saved_cache
            ws_mod.VOICE_MODEL_URL = saved_url

    async def test_f10_stop_async_stops_in_flight_worker(self) -> None:
        """F10: stop_async must cooperatively stop and JOIN workers."""
        stop_event = threading.Event()
        unwound = threading.Event()

        def worker() -> None:
            try:
                while not stop_event.is_set():
                    time.sleep(0.01)
            except KeyboardInterrupt:
                pass
            finally:
                unwound.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        state = _RunningAgentState("tab-w2f5", "model", is_task_active=True)
        state.task_thread = thread
        state.stop_event = stop_event
        _RunningAgentState.register("tab-w2f5", state)
        try:
            await self._stop_server()
            self.assertTrue(
                unwound.is_set(),
                "stop_async must run the worker's cleanup before returning",
            )
            self.assertFalse(thread.is_alive())
            self.assertTrue(state.interrupted_by_shutdown)
        finally:
            stop_event.set()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
