# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_cmd_complete`` active-file snapshot semantics (server audit).

The VS Code extension sends ``complete`` commands whose
``activeFileContent`` is ``completeDoc?.getText()`` — ``undefined``
whenever the visible editor's document is not among
``vscode.workspace.textDocuments`` — while ``activeFile`` is still the
on-disk path.  Two bugs are pinned here:

* A connection that reported an ``activeFile`` but never any
  ``activeFileContent`` used to have ``""`` (an "open but EMPTY
  buffer", honoured verbatim) passed as its snapshot instead of
  ``None``, dead-coding the documented on-disk
  ``read_active_file_head`` fallback — no identifier from the active
  file was ever suggested.

* When the window switched to a DIFFERENT ``activeFile`` without
  supplying content, the stored buffer content of the PREVIOUS file
  stayed paired with the new path, so completions offered identifiers
  from a file no longer on screen.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class TestCmdCompleteDiskFallback(unittest.TestCase):
    """End-to-end ``_cmd_complete`` → worker → ``completions`` events."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        self._tmp.cleanup()

    def _completions_for(self, query: str, timeout: float = 5.0) -> list[dict[str, str]]:
        """Return the completions the worker broadcast for *query*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for ev in self.events:
                if ev.get("type") == "completions" and ev.get("query") == query:
                    return list(ev.get("completions", []))
            time.sleep(0.01)
        self.fail(f"no completions event for query {query!r} within {timeout}s")

    def test_file_without_content_falls_back_to_disk(self) -> None:
        """``activeFile`` with no ``activeFileContent`` reads the disk file."""
        path = self.root / "module.py"
        path.write_text("diskOnlyIdentifier = 1\n", encoding="utf-8")
        self.server._cmd_complete({
            "type": "complete",
            "query": "diskOn",
            "activeFile": str(path),
            "connId": "win-1",
        })
        texts = [c["text"] for c in self._completions_for("diskOn")]
        self.assertIn(
            "diskOnlyIdentifier", texts,
            "on-disk fallback was not used for a connection that never "
            "reported an activeFileContent snapshot",
        )

    def test_file_switch_without_content_drops_stale_buffer(self) -> None:
        """Switching ``activeFile`` with no content unpairs the old buffer."""
        file_a = self.root / "a.py"
        file_a.write_text("alphaFromDiskA = 1\n", encoding="utf-8")
        file_b = self.root / "b.py"
        file_b.write_text("betaFromDiskB = 1\n", encoding="utf-8")
        # Window reports file A with a live buffer snapshot.
        self.server._cmd_complete({
            "type": "complete",
            "query": "alphaLi",
            "activeFile": str(file_a),
            "activeFileContent": "alphaLiveBuffer = 1\n",
            "connId": "win-2",
        })
        texts = [c["text"] for c in self._completions_for("alphaLi")]
        self.assertIn("alphaLiveBuffer", texts)
        # Window switches to file B, no buffer snapshot available.
        self.server._cmd_complete({
            "type": "complete",
            "query": "alphaLi",
            "activeFile": str(file_b),
            "connId": "win-2",
        })
        # The stale buffer of A must be gone: the reply for this second
        # request must not offer A's live-buffer identifier...
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            replies = [
                ev for ev in self.events
                if ev.get("type") == "completions" and ev.get("query") == "alphaLi"
            ]
            if len(replies) >= 2:
                break
            time.sleep(0.01)
        else:
            self.fail("no second completions reply for the file switch")
        second_texts = [c["text"] for c in replies[1].get("completions", [])]
        self.assertNotIn(
            "alphaLiveBuffer", second_texts,
            "the previous file's live buffer stayed paired with the new "
            "activeFile",
        )
        # ...and B's on-disk identifiers are served via the fallback.
        self.server._cmd_complete({
            "type": "complete",
            "query": "betaFr",
            "activeFile": str(file_b),
            "connId": "win-2",
        })
        texts = [c["text"] for c in self._completions_for("betaFr")]
        self.assertIn("betaFromDiskB", texts)

    def test_same_file_without_content_keeps_buffer(self) -> None:
        """Re-reporting the SAME file with no content keeps the buffer."""
        path = self.root / "same.py"
        path.write_text("sameDiskIdent = 1\n", encoding="utf-8")
        self.server._cmd_complete({
            "type": "complete",
            "query": "sameLi",
            "activeFile": str(path),
            "activeFileContent": "sameLiveIdent = 1\n",
            "connId": "win-3",
        })
        self._completions_for("sameLi")
        self.events.clear()
        self.server._cmd_complete({
            "type": "complete",
            "query": "sameLi",
            "activeFile": str(path),
            "connId": "win-3",
        })
        texts = [c["text"] for c in self._completions_for("sameLi")]
        self.assertIn(
            "sameLiveIdent", texts,
            "the live buffer snapshot was dropped although the window "
            "re-reported the same activeFile",
        )


if __name__ == "__main__":
    unittest.main()
