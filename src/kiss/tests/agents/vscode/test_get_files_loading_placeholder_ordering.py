# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``@``-mention picker must never end up showing the placeholder.

R09-3.  On the first ``getFiles`` for a work_dir the cache is cold, so
``_AutocompleteMixin._get_files`` starts a background scan and answers
immediately with an empty ``loading=true`` list; the scan emits the
real list when it finishes.  The two calls were in the wrong order —
the scan thread was started *first* and the placeholder emitted
afterwards — so whenever the scan won the race its populated reply was
overwritten by the empty placeholder that followed it.

The client cannot defend itself: both events carry the same ``prefix``
and belong to the same request, so neither the prefix guard nor the
per-connection request token in ``media/main.js`` can tell them apart.
The picker flashes the file list and then goes empty, and stays empty
until the next keystroke.

Making the race deterministic
-----------------------------

Nothing here is mocked: a real ``VSCodeServer``, a real work_dir on
disk and the real scan thread are used.  The one arrangement is that
the server is given a printer whose delivery of the *placeholder* is
slow — a real subscriber that is momentarily busy, expressed as a
:class:`JsonPrinter` subclass exactly like the shared
:class:`MemoryPrinter`.  That is enough to decide the race by
construction rather than by luck:

* with the scan started first, the placeholder's slow delivery gives
  the scan all the time it needs, so the populated reply is recorded
  first and the placeholder lands on top of it — the bug; and
* with the placeholder emitted first, its delivery finishes before the
  scan is even started, so the order is correct no matter how fast
  either side is.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.server.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

#: How long the placeholder's delivery is held up.  Only an upper
#: bound: the wait ends as soon as a populated reply is recorded.
_PLACEHOLDER_DELIVERY_SECONDS = 2.0


def _texts(event: dict[str, Any]) -> list[str]:
    """Return the suggestion texts carried by a ``files`` event."""
    return [str(entry.get("text", "")) for entry in event.get("files", [])]


class SlowPlaceholderPrinter(MemoryPrinter):
    """A recorder whose delivery of the ``loading`` reply is slow."""

    def __init__(self) -> None:
        """Initialise the recorder and its populated-reply signal."""
        super().__init__()
        self.populated_delivered = threading.Event()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, delaying an empty ``loading`` file list.

        Args:
            event: The event dictionary to emit.
        """
        is_files = event.get("type") == "files"
        if is_files and event.get("loading"):
            self.populated_delivered.wait(_PLACEHOLDER_DELIVERY_SECONDS)
        super().broadcast(event)
        if is_files and not event.get("loading"):
            self.populated_delivered.set()


class TestGetFilesLoadingPlaceholderOrdering(unittest.TestCase):
    """A cold-cache ``getFiles`` must not answer with an empty list."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-getfiles-order-")
        self.printer = SlowPlaceholderPrinter()
        self.server = VSCodeServer(self.printer)
        self.server.work_dir = self.tmpdir

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_work_dir(self, name: str) -> str:
        """Create a tiny work_dir whose scan finishes almost instantly."""
        wd = Path(self.tmpdir) / name
        wd.mkdir()
        Path(wd, "alpha.py").write_text("x\n")
        return str(wd)

    def _files_events(self, conn_id: str) -> list[dict[str, Any]]:
        """Recorded ``files`` events addressed to *conn_id*."""
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == "files" and ev.get("connId") == conn_id
        ]

    def _await_two_replies(self, conn_id: str) -> list[dict[str, Any]]:
        """Wait for both replies of a cold-cache request."""
        deadline = time.time() + 15
        while time.time() < deadline:
            if len(self._files_events(conn_id)) >= 2:
                break
            time.sleep(0.005)
        return self._files_events(conn_id)

    def test_placeholder_is_never_emitted_after_the_real_list(self) -> None:
        """The last ``files`` event of a request must be the real one."""
        work_dir = self._make_work_dir("cold")
        self.server._get_files(
            "", work_dir=work_dir, conn_id="conn1", tab_id="tab1",
        )
        events = self._await_two_replies("conn1")

        assert len(events) == 2, (
            f"expected a placeholder and a populated reply, got {events}"
        )
        assert events[0].get("loading") is True, (
            "the empty loading placeholder was emitted AFTER the "
            f"populated list, so the picker is left showing nothing: "
            f"{events}"
        )
        assert events[1].get("loading") is None, (
            f"the placeholder must not be the final reply: {events}"
        )
        assert _texts(events[1]) == ["alpha.py"], (
            f"the scan reply lost its files: {events}"
        )

    def test_warm_cache_answers_synchronously_without_a_placeholder(
        self,
    ) -> None:
        """A second request is served from the cache, in one event."""
        work_dir = self._make_work_dir("warm")
        self.server._get_files(
            "", work_dir=work_dir, conn_id="warm1", tab_id="tab1",
        )
        self._await_two_replies("warm1")

        self.server._get_files(
            "", work_dir=work_dir, conn_id="warm2", tab_id="tab1",
        )
        events = self._files_events("warm2")
        assert len(events) == 1, f"expected a single reply, got {events}"
        assert events[0].get("loading") is None
        assert _texts(events[0]) == ["alpha.py"]


if __name__ == "__main__":
    unittest.main()
