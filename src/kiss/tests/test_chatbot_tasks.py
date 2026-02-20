"""Tests for chatbot task history features."""

import tempfile
import unittest
from pathlib import Path

import kiss.agents.assistant.assistant as assistant


def _use_temp_history():
    """Redirect HISTORY_FILE to a temp file, return cleanup function."""
    original = assistant.HISTORY_FILE
    tmp = Path(tempfile.mktemp(suffix=".json"))
    assistant.HISTORY_FILE = tmp
    return original, tmp


def _restore_history(original: Path, tmp: Path) -> None:
    assistant.HISTORY_FILE = original
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
        assert assistant._load_history() == []

    def test_save_and_load_history(self) -> None:
        assistant._save_history([_entry("task1"), _entry("task2")])
        loaded = assistant._load_history()
        assert [e["task"] for e in loaded] == ["task1", "task2"]

    def test_load_corrupted_file(self) -> None:
        self.tmp.write_text("not json")
        assert assistant._load_history() == []

    def test_load_non_list_json(self) -> None:
        self.tmp.write_text('{"key": "value"}')
        assert assistant._load_history() == []

    def test_save_truncates_to_max(self) -> None:
        tasks = [_entry(f"task{i}") for i in range(assistant.MAX_HISTORY + 200)]
        assistant._save_history(tasks)
        loaded = assistant._load_history()
        assert len(loaded) == assistant.MAX_HISTORY

    def test_save_overwrite(self) -> None:
        assistant._save_history([_entry("a"), _entry("b")])
        assistant._save_history([_entry("x")])
        loaded = assistant._load_history()
        assert len(loaded) == 1 and loaded[0]["task"] == "x"


class TestAddTask(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_add_new_task(self) -> None:
        assistant._add_task("Build a REST API")
        history = assistant._load_history()
        assert history[0]["task"] == "Build a REST API"

    def test_exact_duplicate_moves_to_top(self) -> None:
        assistant._save_history([
            _entry("old task"),
            _entry("Build a REST API"),
            _entry("another task"),
        ])
        assistant._add_task("Build a REST API")
        history = assistant._load_history()
        assert history[0]["task"] == "Build a REST API"
        assert sum(1 for e in history if e["task"] == "Build a REST API") == 1

    def test_modified_task_kept_as_separate_entry(self) -> None:
        assistant._save_history([
            _entry("Write unit tests for the auth module"),
            _entry("Fix homepage layout bugs"),
        ])
        assistant._add_task("Add tests for the authentication module")
        history = assistant._load_history()
        assert history[0]["task"] == "Add tests for the authentication module"
        task_strs = [e["task"] for e in history]
        assert "Write unit tests for the auth module" in task_strs


if __name__ == "__main__":
    unittest.main()
