# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the 2026-10-26 ``web_server.py`` sweep fixes.

* The three ``remote-url.json`` writers (tunnel publication, watchdog
  LAN-IP republish, watchdog missing-file repair) used to write a URL
  captured *before* their executor job ran, so a republish queued
  behind a tunnel publication could land last and leave the OLD tunnel
  URL on disk.  Now every write reads the server's current state under
  one lock, and the publication records the new URL before writing.
* ``_error_page`` always carries the administrator ``<details>`` block
  (every caller passes one; the empty-details arm was dead).
* The auth-modal label, the error pages and the start-page link name
  the installed brand (``PRODUCT_NAME``), not the stock product.

Not reproduced here: ``_serve_async`` now runs the cron store adoption
and seeding (``adopt_previous_home_jobs`` / ``seed_default_jobs``,
both under a blocking ``flock``) in ``asyncio.to_thread``.  Only the
real ``kiss-web`` daemon lifecycle (``start()``) reaches that code,
and provoking the stall needs a second process holding the cron
store lock during the daemon's start-up window, which cannot be
timed from a test without a test double.

Everything is real: a ``RemoteAccessServer`` on a free port, plain
HTTPS requests, the real default executor replaced by a one-worker
pool so the executor jobs run in a known order.  No mocks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kiss.core.brand import PRODUCT_NAME
from kiss.server.web_server import RemoteAccessServer, _cancel_task
from kiss.tests.server.test_local_lan_urls import _LiveServerCase
from kiss.tests.server.test_tls_local_ca import _https_get

OLD = "https://old.example.test"
NEW = "https://new.example.test"


def _nothing() -> None:
    """Executor sentinel: runs once every job queued before it has finished."""


class TestUrlFileWritersOrdered(_LiveServerCase):
    """The last ``remote-url.json`` write always holds the newest tunnel URL."""

    def _url_file(self) -> dict[str, object]:
        """Parse the server's ``remote-url.json``."""
        data: dict[str, object] = json.loads(self.server._url_file.read_text())
        return data

    async def test_queued_republish_cannot_leave_a_stale_tunnel_url_in_the_file(self) -> None:
        """A republish queued behind the publication of NEW still writes NEW.

        The default executor is a one-worker pool whose worker is held
        by a gate job, so every write queues behind it in order: first
        the publication's, then the republish's, which is queued while
        the publication is still waiting for its write (the window the
        watchdog's IP-change republish can hit).  Before the fix the
        republish captured the tunnel URL when it was queued (still
        OLD) and its write landed last, so the file ended up with the
        dead OLD URL.
        """
        # The start-up PyPI check would otherwise queue a network fetch
        # on the single worker ahead of the sentinel.
        await _cancel_task(self.server._version_check_task)
        self.server._version_check_task = None
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        gate = threading.Event()
        blocker = loop.run_in_executor(None, gate.wait)
        self.server._active_url = OLD
        publish = asyncio.create_task(self.server._publish_tunnel_url(NEW))
        await asyncio.sleep(0)
        self.assertFalse(publish.done())
        self.server._republish_urls()
        assert self.server._republish_task is not None
        gate.set()
        await blocker
        await publish
        await self.server._republish_task
        await loop.run_in_executor(None, _nothing)
        self.assertEqual(self._url_file()["tunnel"], NEW)
        self.assertEqual(self.server._active_url, NEW)

    async def test_missing_file_repair_writes_the_current_tunnel_url(self) -> None:
        """The watchdog's repair of a deleted URL file reads the live state."""
        self.server._active_url = NEW
        self.server._url_file.unlink()
        await asyncio.to_thread(self.server._watchdog_check_url_file)
        self.assertEqual(self._url_file()["tunnel"], NEW)

    async def test_write_without_a_tunnel_omits_the_tunnel_key(self) -> None:
        """With only the local URL active the file has no ``tunnel`` key."""
        self.server._active_url = self.server._local_url
        await asyncio.to_thread(self.server._write_url_file_sync)
        self.assertNotIn("tunnel", self._url_file())


class TestBrandedPages(_LiveServerCase):
    """Pages built in Python name the installed brand, never the stock one."""

    async def test_auth_modal_label_names_the_brand(self) -> None:
        """The chat page's password label says ``Password for this <brand> server``."""
        status, _headers, body = await asyncio.to_thread(_https_get, self.port, "/")
        self.assertEqual(status, 200)
        self.assertIn(f"Password for this {PRODUCT_NAME} server", body.decode())

    async def test_404_page_has_brand_link_and_admin_details(self) -> None:
        """An unknown path gets the branded start-page link and a ``<details>`` block."""
        status, _headers, body = await asyncio.to_thread(
            _https_get, self.port, "/no-such-page",
        )
        self.assertEqual(status, 404)
        page = body.decode()
        self.assertIn(f"Go to the {PRODUCT_NAME} start page", page)
        self.assertIn(f"open {PRODUCT_NAME}.", page)
        self.assertIn("<details><summary>For the server administrator</summary>", page)
        self.assertIn("No route matches", page)


class TestCmdWorkDir(_LiveServerCase):
    """``_cmd_work_dir`` falls back to the backend work dir and nothing else."""

    async def test_explicit_work_dir_wins_else_backend_work_dir(self) -> None:
        """A string ``workDir`` is used as is; anything else yields the backend's."""
        backend = self.server._agent_server.work_dir
        self.assertTrue(backend)
        explicit = tempfile.mkdtemp()
        self.assertEqual(self.server._cmd_work_dir({"workDir": explicit}), explicit)
        self.assertEqual(self.server._cmd_work_dir({"workDir": ""}), backend)
        self.assertEqual(self.server._cmd_work_dir({"workDir": ["x"]}), backend)
        self.assertEqual(self.server._cmd_work_dir({}), backend)

    async def test_fresh_server_adopts_its_own_work_dir_as_the_backend_one(self) -> None:
        """A server built with a work dir hands it to the backend: the fallback is never empty."""
        own = Path(tempfile.mkdtemp())
        srv = RemoteAccessServer(host="127.0.0.1", port=self.port + 1, work_dir=str(own))
        self.assertEqual(srv._cmd_work_dir({}), str(own))
