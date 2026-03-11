"""Tests for chatbot task history features."""

import shutil
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.task_history as assistant


def _use_temp_history():
    """Redirect HISTORY_FILE and _CHAT_EVENTS_DIR to temp locations."""
    original_file = assistant.HISTORY_FILE
    original_events_dir = assistant._CHAT_EVENTS_DIR
    tmp_dir = Path(tempfile.mkdtemp())
    assistant.HISTORY_FILE = tmp_dir / "task_history.jsonl"
    assistant._CHAT_EVENTS_DIR = tmp_dir / "chat_events"
    assistant._history_cache = None
    return original_file, original_events_dir, tmp_dir


def _restore_history(original_file: Path, original_events_dir: Path, tmp_dir: Path) -> None:
    assistant.HISTORY_FILE = original_file
    assistant._CHAT_EVENTS_DIR = original_events_dir
    assistant._history_cache = None
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _entry(task: str, result: str = "") -> dict[str, str]:
    return {"task": task, "result": result}


class TestHistoryFileOps(unittest.TestCase):
    def setUp(self) -> None:
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

class TestAddTask(unittest.TestCase):
    def setUp(self) -> None:
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

if __name__ == "__main__":
    unittest.main()
