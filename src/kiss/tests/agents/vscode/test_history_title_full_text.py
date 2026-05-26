"""History sidebar items must carry the full task text so the
``.running-item > .sidebar-item-text`` line-clamp (4 lines) in ``main.css``
can actually wrap multiple lines.

The backend previously truncated ``title`` to 50 characters with an
ellipsis, which restricted every history-panel row to a single visible
line regardless of how much CSS box space was available.  This test
locks in the fix: ``title`` carries the full task text untouched.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.agents.vscode.server import VSCodeServer

_LONG_TASK = (
    "This is a deliberately very long task description that comfortably "
    "exceeds the previous 50 character truncation threshold and should "
    "wrap across multiple lines in the history sidebar so the user can "
    "skim the first four lines of context before clicking in."
)


class TestHistoryTitleFullText(unittest.TestCase):
    """``getHistory`` returns the full task text in the ``title`` field."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        th._close_db()
        th._DB_PATH = Path(self._tmp) / "sorcar.db"  # type: ignore[attr-defined]

        self.server = VSCodeServer()
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
        th._close_db()
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
        self.assertEqual(s["preview"], _LONG_TASK)
        # Ensure the previous 50-char + ellipsis truncation is gone.
        self.assertNotIn("...", s["title"])
        self.assertGreater(len(s["title"]), 50)

    def test_short_task_title_unchanged(self) -> None:
        short = "short task"
        task_id, _ = th._add_task(short)
        th._save_task_result(result="done", task_id=task_id)

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["title"], short)


class TestRunningItemLineClampCSS(unittest.TestCase):
    """The CSS that controls how many lines history rows can show
    still pins ``-webkit-line-clamp`` to 4 on ``.running-item`` rows.
    History items render with ``className = 'sidebar-item running-item'``
    in ``renderHistory`` so this selector applies to them too.
    """

    def test_running_item_line_clamp_is_four(self) -> None:
        css_path = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "media"
            / "main.css"
        )
        css = css_path.read_text(encoding="utf-8")
        # Block:
        #   .running-item > .sidebar-item-text {
        #     ...
        #     -webkit-line-clamp: 4;
        #     line-clamp: 4;
        #     ...
        #   }
        start = css.index(".running-item > .sidebar-item-text")
        block = css[start : start + 400]
        self.assertIn("-webkit-line-clamp: 4;", block)
        self.assertIn("line-clamp: 4;", block)


if __name__ == "__main__":
    unittest.main()
