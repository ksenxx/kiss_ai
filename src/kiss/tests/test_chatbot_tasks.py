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

    def test_load_empty_history(self) -> None:
        loaded = assistant._load_history()
        assert loaded == assistant.SAMPLE_TASKS
        assert self.tmp.exists()

    def test_load_corrupted_file(self) -> None:
        self.tmp.write_text("not json")
        loaded = assistant._load_history()
        assert loaded == assistant.SAMPLE_TASKS

    def test_load_non_list_json(self) -> None:
        self.tmp.write_text('{"key": "value"}')
        loaded = assistant._load_history()
        assert loaded == assistant.SAMPLE_TASKS


class TestAddTask(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_add_new_task(self) -> None:
        assistant._add_task("Build a REST API")
        history = assistant._load_history()
        assert history[0]["task"] == "Build a REST API"


if __name__ == "__main__":
    unittest.main()
