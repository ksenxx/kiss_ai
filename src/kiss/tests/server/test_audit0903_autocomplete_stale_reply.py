# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (server, fix round): stale old-workspace file replies.

The ``@``-mention picker's background refresh
(:meth:`_refresh_file_cache`) used to verify AND REMOVE its
per-connection request token under ``_state_lock``, release the lock,
and only then load usage, rank, and emit the ``files`` event.  A newer
``getFiles`` from the SAME connection — same typed prefix, different
tab/work directory — arriving in that window emitted first, and the
superseded old-workspace reply then emitted LAST with no further token
check.  The frontend validates replies only by active tab and prefix
(not by work dir), so the picker ended up showing — and letting the
user insert — paths from the wrong repository (review Finding 5).

The fix computes the scan and ranking outside the lock, then
reacquires ``_state_lock``, re-verifies that the captured token is
still the connection's current request, emits while it is, and removes
the token only after the emission.

The superseded interleaving is made deterministic with a real
:class:`VSCodeServer` subclass whose ``_emit_files`` parks the OLD
workspace's populated reply on its way out (park-then-delegate, the
same production-boundary technique as the parking printers); the newer
request runs on its own thread through the production ``_get_files``.
Because the fixed code emits under ``_state_lock``, the release
condition is two-sided: the parked old reply is released once the
newer reply has been observed (old code — the stale reply then lands
last and the test fails) or after a grace period during which the
newer request is provably lock-blocked (fixed code — the old reply
lands first, while still current, and the newer reply wins).
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter

_CONN = "conn-stale"
_PREFIX = "marker"


class _ParkingEmitServer(VSCodeServer):
    """Real server whose ``_emit_files`` parks one designated reply."""

    def __init__(self, printer: MemoryPrinter) -> None:
        super().__init__(printer=printer)
        self.park_marker = ""
        self.parked = threading.Event()
        self.release = threading.Event()

    def _emit_files(
        self,
        ranked: list[dict[str, Any]],
        conn_id: str,
        loading: bool = False,
        prefix: str = "",
        tab_id: str = "",
    ) -> None:
        """Park the populated reply naming ``park_marker``, then delegate."""
        if (
            self.park_marker
            and not loading
            and not self.parked.is_set()
            and any(self.park_marker in str(f) for f in ranked)
        ):
            self.parked.set()
            if not self.release.wait(timeout=60):
                raise TimeoutError("parked too long in _emit_files")
        super()._emit_files(
            ranked, conn_id, loading=loading, prefix=prefix, tab_id=tab_id,
        )


class TestAutocompleteStaleReply(TestCase):
    """The last ``files`` reply for a connection must be the newest request's."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0903-files-"))
        self.wd_old = self.tmp / "repo-old"
        self.wd_old.mkdir()
        (self.wd_old / "marker-old.py").write_text("old\n", encoding="utf-8")
        self.wd_new = self.tmp / "repo-new"
        self.wd_new.mkdir()
        (self.wd_new / "marker-new.py").write_text("new\n", encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = _ParkingEmitServer(self.printer)
        self.server.work_dir = str(self.tmp)

    def tearDown(self) -> None:
        self.server.release.set()
        agent_state.agent_states.clear()

    def _files_events(self, conn_id: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == "files" and ev.get("connId") == conn_id
        ]

    def _wait(self, predicate: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def _populated(self, ev: dict[str, Any]) -> bool:
        return not ev.get("loading") and bool(ev.get("files"))

    def _names(self, ev: dict[str, Any]) -> str:
        return " ".join(str(f) for f in ev.get("files", []))

    def _warm_cache(self, work_dir: Path) -> None:
        """Populate the file cache for *work_dir* through production code."""
        self.server._get_files(_PREFIX, str(work_dir), "warm-conn", "warm-tab")
        self.assertTrue(
            self._wait(
                lambda: any(
                    self._populated(ev)
                    for ev in self._files_events("warm-conn")
                ),
                30.0,
            ),
            "warming the new workspace's file cache never completed",
        )

    def test_superseded_old_workspace_reply_never_lands_last(self) -> None:
        self._warm_cache(self.wd_new)

        # Request 1: same prefix, OLD workspace — cache miss, so the
        # populated reply comes from the background refresh, which is
        # parked on its way out.
        self.server.park_marker = "marker-old"
        self.server._get_files(_PREFIX, str(self.wd_old), _CONN, "tab-1")
        self.assertTrue(
            self.server.parked.wait(timeout=30),
            "the old workspace's populated reply never reached emission",
        )

        # Request 2: SAME connection and prefix, NEW workspace (cache
        # hit) — the exact supersession the frontend cannot detect.
        req2 = threading.Thread(
            target=self.server._get_files,
            args=(_PREFIX, str(self.wd_new), _CONN, "tab-2"),
            daemon=True,
        )
        req2.start()

        # Two-sided release (see module docstring): observed newer
        # reply (old code) or grace period with request 2 lock-blocked
        # (fixed code).
        self._wait(
            lambda: any(
                self._populated(ev) and "marker-new" in self._names(ev)
                for ev in self._files_events(_CONN)
            ),
            2.0,
        )
        self.server.release.set()
        req2.join(timeout=30)
        self.assertFalse(req2.is_alive(), "the newer getFiles never finished")

        self.assertTrue(
            self._wait(
                lambda: any(
                    self._populated(ev) and "marker-new" in self._names(ev)
                    for ev in self._files_events(_CONN)
                ),
                30.0,
            ),
            "the newer workspace's reply was never emitted at all",
        )
        # Let any straggling stale emission land before the verdict.
        time.sleep(0.3)
        populated = [
            ev for ev in self._files_events(_CONN) if self._populated(ev)
        ]
        self.assertTrue(populated)
        self.assertIn(
            "marker-new",
            self._names(populated[-1]),
            "BUG: the superseded OLD workspace's file list was emitted "
            "LAST and would repopulate the picker with paths from the "
            f"wrong repository: {[self._names(e) for e in populated]}",
        )

    def test_reply_superseded_during_ranking_is_dropped(self) -> None:
        """A token replaced while the old reply ranks is never emitted.

        Covers the fixed re-verification branch: the old refresh has
        already published its scan when the newer request lands, so
        its reply is superseded between the scan and the emission
        (inside the usage-ranking step).  The interleaving is made
        deterministic with the REAL persistence read/write lock the
        ranking step's ``_load_file_usage`` takes: the test holds the
        write side, parking BOTH refreshes inside ranking, installs
        the newer token, and releases — the old reply must then be
        dropped by the emit-time token check, and only the newer
        workspace's reply reaches the connection.
        """
        from kiss.agents.sorcar import persistence as _persistence

        write_lock = _persistence._rw_lock.write_lock()
        write_lock.__enter__()
        released = False
        try:
            # Request 1 (OLD workspace): the miss path emits only the
            # loading placeholder on this thread; its refresh thread
            # scans, publishes the cache, and blocks in ranking.
            self.server._get_files(_PREFIX, str(self.wd_old), _CONN, "tab-1")
            self.assertTrue(
                self._wait(
                    lambda: str(self.wd_old) in self.server._file_cache,
                    30.0,
                ),
                "the old workspace's scan never published its cache",
            )
            # Request 2 (NEW workspace, same connection): replaces the
            # token BEFORE the old reply can reach its emit-time check.
            self.server._get_files(_PREFIX, str(self.wd_new), _CONN, "tab-2")
            self.assertTrue(
                self._wait(
                    lambda: str(self.wd_new) in self.server._file_cache,
                    30.0,
                ),
                "the new workspace's scan never published its cache",
            )
            write_lock.__exit__(None, None, None)
            released = True

            self.assertTrue(
                self._wait(
                    lambda: any(
                        self._populated(ev) and "marker-new" in self._names(ev)
                        for ev in self._files_events(_CONN)
                    ),
                    30.0,
                ),
                "the newer workspace's reply never arrived",
            )
            # The answered request's token is gone; give the old
            # refresh time to (wrongly) emit before the verdict.
            self.assertTrue(
                self._wait(
                    lambda: _CONN not in self.server._files_request_map(),
                    30.0,
                ),
            )
            time.sleep(0.3)
            self.assertFalse(
                any(
                    self._populated(ev) and "marker-old" in self._names(ev)
                    for ev in self._files_events(_CONN)
                ),
                "the superseded old-workspace reply was emitted despite "
                "the emit-time token check",
            )
        finally:
            if not released:
                write_lock.__exit__(None, None, None)

    def test_unsuperseded_refresh_still_answers(self) -> None:
        """A cache-miss refresh with no competing request must reply."""
        self.server._get_files(_PREFIX, str(self.wd_old), "conn-solo", "tab")
        self.assertTrue(
            self._wait(
                lambda: any(
                    self._populated(ev) and "marker-old" in self._names(ev)
                    for ev in self._files_events("conn-solo")
                ),
                30.0,
            ),
            "the lone refresh's populated reply never arrived",
        )
        # Its token was removed after the answer: the map stays empty
        # for idle connections.
        with self.server._state_lock:
            self.assertNotIn(
                "conn-solo", self.server._files_request_map(),
            )
