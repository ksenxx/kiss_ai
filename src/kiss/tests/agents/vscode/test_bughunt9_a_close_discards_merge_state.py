# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (batch A / A4): loop-teardown race in ``_fire_pending_tab_close``.

``_fire_pending_tab_close`` used to pop the tab's ``_WebMergeState``
(via ``_pop_merge_state``, which also drops the per-tab action lock)
BEFORE checking whether the event loop is still available.  When the
``call_later`` callback races loop teardown (``self._loop is None`` or
``not self._loop.is_running()``), the popped state — the only thing
that could ever drive the in-flight review to ``all-done`` — was
silently discarded and nothing was dispatched: the backend tab stayed
``is_merging=True`` forever.

The guard must run BEFORE the pop so that bailing out is side-effect
free.  These tests exercise the real method on a real
:class:`RemoteAccessServer` (no mocks): one with no loop at all, one
with a real-but-not-running loop.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import RemoteAccessServer


def _make_server(tmpdir: str) -> RemoteAccessServer:
    """Build a lightweight (unstarted) server, like the timer race hits."""
    return RemoteAccessServer(host="127.0.0.1", port=0, work_dir=tmpdir)


def _merge_data(work: Path) -> dict:
    """Create one real modified file and its merge entry."""
    current = work / "f.txt"
    base = work / "f_base.txt"
    lines = "".join(f"line{i}\n" for i in range(10))
    current.write_text(lines.replace("line2\n", "edit2\n"))
    base.write_text(lines)
    return {
        "work_dir": str(work),
        "files": [
            {
                "name": "f.txt",
                "current": str(current),
                "base": str(base),
                "target": str(current),
                "hunks": [{"bs": 2, "bc": 1, "cs": 2, "cc": 1}],
            }
        ],
    }


class TestFirePendingCloseLoopGuard(unittest.TestCase):
    """Bailing out on a dead loop must not discard the merge state."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bh9-a4-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.work = Path(self.tmpdir) / "work"
        self.work.mkdir()

    def test_no_loop_keeps_merge_state(self) -> None:
        """``self._loop is None``: the guard must fire before the pop."""
        server = _make_server(self.tmpdir)
        self.assertIsNone(server._loop)
        tab_id = "tab-a4-noloop"
        server._register_merge_state(tab_id, _merge_data(self.work))

        server._fire_pending_tab_close(tab_id)

        with server._merge_states_lock:
            self.assertIn(
                tab_id, server._merge_states,
                "BUG: _fire_pending_tab_close discarded the merge state "
                "even though no loop was available to dispatch all-done — "
                "the review can never finish and the tab stays is_merging",
            )

    def test_stopped_loop_keeps_merge_state(self) -> None:
        """A real, not-running loop: same guard, same requirement."""
        server = _make_server(self.tmpdir)
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        server._loop = loop
        self.assertFalse(loop.is_running())
        tab_id = "tab-a4-stopped"
        server._register_merge_state(tab_id, _merge_data(self.work))

        server._fire_pending_tab_close(tab_id)

        with server._merge_states_lock:
            self.assertIn(
                tab_id, server._merge_states,
                "BUG: merge state popped and dropped while the loop was "
                "not running — the popped state is the only handle that "
                "could drive the review to all-done",
            )


if __name__ == "__main__":
    unittest.main()
