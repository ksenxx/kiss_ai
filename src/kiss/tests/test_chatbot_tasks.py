"""Tests for chatbot task history features."""

import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.task_history as assistant


def _use_temp_history():
    """Redirect HISTORY_FILE to a temp file, return cleanup function."""
    original = assistant.HISTORY_FILE
    tmp = Path(tempfile.mktemp(suffix=".json"))
    assistant.HISTORY_FILE = tmp
    assistant._history_cache = None
    return original, tmp


def _restore_history(original: Path, tmp: Path) -> None:
    assistant.HISTORY_FILE = original
    assistant._history_cache = None
    if tmp.exists():
        tmp.unlink()


def _entry(task: str, result: str = "") -> dict[str, str]:
    return {"task": task, "result": result}


class TestHistoryFileOps(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

class TestAddTask(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

if __name__ == "__main__":
    unittest.main()
