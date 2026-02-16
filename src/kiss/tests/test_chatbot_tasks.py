"""Tests for chatbot task deduplication and proposal features."""

import tempfile
import unittest
from pathlib import Path

import kiss.agents.coding_agents.chatbot as chatbot
from kiss.tests.conftest import requires_gemini_api_key


def _use_temp_history():
    """Redirect HISTORY_FILE to a temp file, return cleanup function."""
    original = chatbot.HISTORY_FILE
    tmp = Path(tempfile.mktemp(suffix=".json"))
    chatbot.HISTORY_FILE = tmp
    return original, tmp


def _restore_history(original: Path, tmp: Path) -> None:
    chatbot.HISTORY_FILE = original
    if tmp.exists():
        tmp.unlink()


class TestHistoryFileOps(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_load_empty_history(self) -> None:
        assert chatbot._load_history() == []

    def test_save_and_load_history(self) -> None:
        chatbot._save_history(["task1", "task2"])
        assert chatbot._load_history() == ["task1", "task2"]

    def test_load_corrupted_file(self) -> None:
        self.tmp.write_text("not json")
        assert chatbot._load_history() == []

    def test_load_non_list_json(self) -> None:
        self.tmp.write_text('{"key": "value"}')
        assert chatbot._load_history() == []

    def test_save_truncates_to_max(self) -> None:
        tasks = [f"task{i}" for i in range(200)]
        chatbot._save_history(tasks)
        loaded = chatbot._load_history()
        assert len(loaded) == chatbot.MAX_HISTORY

    def test_save_overwrite(self) -> None:
        chatbot._save_history(["a", "b"])
        chatbot._save_history(["x"])
        assert chatbot._load_history() == ["x"]


class TestFindSemanticDuplicatesEdgeCases(unittest.TestCase):
    def test_empty_existing_tasks(self) -> None:
        assert chatbot._find_semantic_duplicates("any task", []) == []


@requires_gemini_api_key
class TestFindSemanticDuplicates(unittest.TestCase):
    def test_detects_same_task_different_wording(self) -> None:
        existing = [
            "Add unit tests for the login module",
            "Fix the CSS layout on the homepage",
            "Refactor the database connection pool",
        ]
        new_task = "Write tests for the login feature"
        duplicates = chatbot._find_semantic_duplicates(new_task, existing)
        assert 0 in duplicates, f"Expected index 0 to be a duplicate, got {duplicates}"

    def test_no_duplicates_for_unrelated_task(self) -> None:
        existing = [
            "Add unit tests for the login module",
            "Fix the CSS layout on the homepage",
        ]
        new_task = "Set up CI/CD pipeline with GitHub Actions"
        duplicates = chatbot._find_semantic_duplicates(new_task, existing)
        assert duplicates == [], f"Expected no duplicates, got {duplicates}"

    def test_returns_valid_indices(self) -> None:
        existing = ["task A", "task B", "task C"]
        new_task = "completely unrelated quantum physics research"
        duplicates = chatbot._find_semantic_duplicates(new_task, existing)
        for idx in duplicates:
            assert 0 <= idx < len(existing), f"Index {idx} out of range"


@requires_gemini_api_key
class TestAddTaskWithDedup(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_add_new_task(self) -> None:
        chatbot._add_task("Build a REST API")
        history = chatbot._load_history()
        assert history[0] == "Build a REST API"

    def test_exact_duplicate_moves_to_top(self) -> None:
        chatbot._save_history(["old task", "Build a REST API", "another task"])
        chatbot._add_task("Build a REST API")
        history = chatbot._load_history()
        assert history[0] == "Build a REST API"
        assert history.count("Build a REST API") == 1

    def test_semantic_duplicate_removed(self) -> None:
        chatbot._save_history([
            "Write unit tests for the auth module",
            "Fix homepage layout bugs",
        ])
        chatbot._add_task("Add tests for the authentication module")
        history = chatbot._load_history()
        assert history[0] == "Add tests for the authentication module"
        assert "Write unit tests for the auth module" not in history


@requires_gemini_api_key
class TestRefreshProposedTasks(unittest.TestCase):
    def setUp(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def tearDown(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_empty_history_produces_no_proposals(self) -> None:
        chatbot._refresh_proposed_tasks()
        with chatbot._proposed_lock:
            assert chatbot._proposed_tasks == []

    def test_generates_proposals_from_history(self) -> None:
        chatbot._save_history([
            "Add user authentication with JWT",
            "Create REST API for user management",
            "Set up PostgreSQL database schema",
        ])
        chatbot._refresh_proposed_tasks()
        with chatbot._proposed_lock:
            proposals = list(chatbot._proposed_tasks)
        assert len(proposals) > 0, "Expected at least one proposal"
        assert all(isinstance(p, str) and p.strip() for p in proposals)

    def test_proposals_capped_at_five(self) -> None:
        chatbot._save_history([f"Task number {i}" for i in range(10)])
        chatbot._refresh_proposed_tasks()
        with chatbot._proposed_lock:
            assert len(chatbot._proposed_tasks) <= 5


if __name__ == "__main__":
    unittest.main()
