# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: ``files`` events must echo the requested ``prefix``.

The ``@``-mention picker protocol was not staleness-safe: the backend's
``_get_files`` answers a cache miss with an immediate empty
``loading=true`` reply and a SECOND populated ``files`` event once the
background directory scan finishes — potentially seconds later on a
big work dir.  Unlike the ``ghost`` event (which echoes ``query`` so
the frontend can drop stale suggestions), ``files`` events carried no
``prefix``, so the frontend had no way to tell a late reply for an
abandoned ``@``-mention from a fresh one.  The late event re-opened
the picker over the input after the user had moved on, and the
phantom picker swallowed the next Enter keystroke.

Fix: every ``files`` event is stamped with the ``prefix`` it was
ranked for; ``media/main.js`` ignores replies whose prefix no longer
matches the @-mention being typed (see
``test/bughunt6_files_stale.test.js`` for the frontend half).
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _RunningAgentState


class TestFilesEventCarriesPrefix(unittest.TestCase):
    """Backend half: ``_get_files`` must stamp ``prefix`` on replies."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "src").mkdir()
        (root / "src" / "util.py").write_text("x = 1\n")
        (root / "src" / "helpers.py").write_text("y = 2\n")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self._tmp.cleanup()

    def _wait_for_files_events(self, count: int, timeout: float = 10.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            files_events = [e for e in self.events if e.get("type") == "files"]
            if len(files_events) >= count:
                return files_events
            time.sleep(0.02)
        raise AssertionError(
            f"expected {count} files events, got: {self.events!r}",
        )

    def test_cache_miss_replies_carry_prefix(self) -> None:
        """Both the loading reply and the post-scan reply echo the prefix."""
        self.server._get_files("ut", work_dir=self._tmp.name)
        files_events = self._wait_for_files_events(2)
        for ev in files_events:
            self.assertIn(
                "prefix", ev,
                f"files event must echo the requested prefix: {ev!r}",
            )
            self.assertEqual(ev["prefix"], "ut")
        # The post-scan reply must actually contain the matching file.
        populated = [e for e in files_events if e.get("files")]
        self.assertTrue(populated, f"no populated files reply: {files_events!r}")
        texts = [f["text"] for f in populated[-1]["files"]]
        self.assertTrue(any("util.py" in t for t in texts), texts)

    def test_cache_hit_reply_carries_prefix(self) -> None:
        """The synchronous cache-hit reply echoes the prefix too."""
        self.server._get_files("ut", work_dir=self._tmp.name)
        self._wait_for_files_events(2)
        self.events.clear()

        self.server._get_files("helpers", work_dir=self._tmp.name)
        files_events = self._wait_for_files_events(1)
        self.assertEqual(files_events[0].get("prefix"), "helpers")


if __name__ == "__main__":
    unittest.main()
