# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Startup must not wait forever behind a wedged holder of the UDS lock.

Concurrency audit (F3, reviewer R3 finding 11): ``_setup_server``
serialised the UDS probe/unlink/bind sequence across processes with a
blocking ``flock(LOCK_EX)`` run in the executor.  Cancelling the
coroutine never interrupts that syscall, so a sibling process that
wedged while holding ``sorcar.sock.lock`` stalled the new daemon's
startup indefinitely — before the WSS listener was even bound, i.e.
with no listener at all.  The acquisition is now a ``LOCK_NB`` poll
with a deadline; on expiry the daemon falls back to WSS-only exactly
like any other UDS bind failure.

A real child process holds the lock; the daemon is a real
``RemoteAccessServer`` with a self-signed cert.  No mocks.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)
from kiss.tests.server.test_web_server_uds_owner_wait import (
    _redirect_persistence,
    _restore_persistence,
)


class TestUdsLockDeadline(IsolatedAsyncioTestCase):
    """A held sidecar lock costs the UDS, not the whole daemon."""

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
            uds_owner_wait_s=1.0,
        )
        self.holder: subprocess.Popen[str] | None = None

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self.holder is not None and self.holder.poll() is None:
            self.holder.kill()
            self.holder.wait(timeout=5)
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_startup_falls_back_to_wss_when_lock_is_held(self) -> None:
        lock_path = self.uds_path.with_name(self.uds_path.name + ".lock")
        self.holder = subprocess.Popen(
            [
                sys.executable, "-c",
                "import fcntl, sys, time\n"
                f"f = open({str(lock_path)!r}, 'w')\n"
                "fcntl.flock(f, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "time.sleep(300)\n",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self.holder.stdout is not None
        self.assertEqual(self.holder.stdout.readline().strip(), "locked")
        self.server._uds_lock_timeout_s = 1.0

        started = time.monotonic()
        await asyncio.wait_for(self.server.start_async(), timeout=30.0)
        elapsed = time.monotonic() - started

        self.assertIsNone(self.server._uds_server, "UDS must be skipped")
        self.assertIsNotNone(self.server._ws_server, "WSS must still be bound")
        self.assertFalse(self.uds_path.exists(), "no socket may be bound")
        self.assertLess(elapsed, 15.0, f"startup took {elapsed:.1f}s")
        self.assertIsNone(self.holder.poll(), "the lock holder must be untouched")


if __name__ == "__main__":
    import unittest

    unittest.main()
