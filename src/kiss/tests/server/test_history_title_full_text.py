# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""History sidebar items must carry the full task text so the
``.running-item > .sidebar-item-text`` line-clamp (3 lines) in ``main.css``
can actually wrap multiple lines.

The backend previously truncated ``title`` to 50 characters with an
ellipsis, which restricted every history-panel row to a single visible
line regardless of how much CSS box space was available.  This test
locks in the fix: ``title`` carries the full task text untouched, up
to ``HISTORY_TITLE_CHARS``.  A pasted prompt of tens of kilobytes is
cut there (head and tail kept, a note in between), so one such task
does not make every CHATS page a long wait on a slow network: its
whole text is in the transcript's ``prompt`` event once it is opened.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.server.server import HISTORY_TITLE_CHARS, AgentServer
from kiss.tests.persistence_helpers import close_db

_LONG_TASK = (
    "This is a deliberately very long task description that comfortably "
    "exceeds the previous 50 character truncation threshold and should "
    "wrap across multiple lines in the history sidebar so the user can "
    "skim the first three lines of context before clicking in."
)


class TestHistoryTitleFullText(unittest.TestCase):
    """``getHistory`` returns the full task text in the ``title`` field."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        close_db()
        th._DB_PATH = Path(self._tmp) / "history.db"  # type: ignore[attr-defined]

        self.server = AgentServer()
        self.server.work_dir = self._tmp
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        orig = self.server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(e))
            orig(e)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        close_db()
        th._DB_PATH = self._orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    return list(e["sessions"])
        return []

    def test_long_task_title_is_not_truncated(self) -> None:
        task_id, _ = th._add_task(_LONG_TASK)
        th._save_task_result(result="done", task_id=task_id)

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()

        self.assertEqual(len(sessions), 1)
        s = sessions[0]
        self.assertEqual(s["title"], _LONG_TASK)
        self.assertNotIn("preview", s)  # the text travels once, as title
        self.assertNotIn("...", s["title"])
        self.assertGreater(len(s["title"]), 50)
        self.assertLess(len(_LONG_TASK), HISTORY_TITLE_CHARS)

    def test_short_task_title_unchanged(self) -> None:
        short = "short task"
        task_id, _ = th._add_task(short)
        th._save_task_result(result="done", task_id=task_id)

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["title"], short)

    def test_huge_task_title_is_cut_head_and_tail(self) -> None:
        """A prompt far above the cap travels as its head, a note and its tail."""
        head = "HEAD " + "h" * 3_000
        tail = "t" * 3_000 + " TAIL"
        huge = head + "\nmiddle that the page does not carry\n" + tail
        task_id, _ = th._add_task(huge, chat_id="chat-huge")
        th._save_task_result(result="done", task_id=task_id)

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()

        self.assertEqual(len(sessions), 1)
        text = sessions[0]["title"]
        self.assertLess(len(text), HISTORY_TITLE_CHARS + 100)
        self.assertTrue(text.startswith(huge[: HISTORY_TITLE_CHARS // 2]))
        self.assertTrue(text.endswith(huge[-(HISTORY_TITLE_CHARS // 2):]))
        omitted = len(huge) - 2 * (HISTORY_TITLE_CHARS // 2)
        self.assertIn(f"[{omitted} characters omitted from this row]", text)
        self.assertNotIn("middle that the page does not carry", text)
        # The chat header's text comes from its loader cut to 1000.
        self.assertEqual(sessions[0]["chat_first_task"], huge[:1000])


if __name__ == "__main__":
    unittest.main()
