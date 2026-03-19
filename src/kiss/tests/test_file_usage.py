"""Tests for file usage tracking and @ file picker frequency sorting."""

import json
import unittest
from pathlib import Path

from kiss.agents.sorcar import task_history

# ---------------------------------------------------------------------------
# kiss/agents/sorcar/task_history.py — task_history
# ---------------------------------------------------------------------------


class TestFileUsage(unittest.TestCase):
    """Tests for _load_file_usage / _record_file_usage persistence."""

    def setUp(self) -> None:
        self._orig = task_history.FILE_USAGE_FILE
        self._tmp = Path(__file__).parent / "_test_file_usage.json"
        task_history.FILE_USAGE_FILE = self._tmp
        if self._tmp.exists():
            self._tmp.unlink()

    def tearDown(self) -> None:
        task_history.FILE_USAGE_FILE = self._orig
        if self._tmp.exists():
            self._tmp.unlink()

    def test_load_empty_returns_empty(self) -> None:
        """Loading from a non-existent file returns empty dict."""
        assert task_history._load_file_usage() == {}

    def test_load_non_dict_json(self) -> None:
        self._tmp.write_text("[1,2,3]")
        assert task_history._load_file_usage() == {}

    def test_load_corrupt_json(self) -> None:
        """Corrupt JSON file is handled gracefully, returns empty dict."""
        self._tmp.write_text("{bad json!!")
        assert task_history._load_file_usage() == {}

    def test_load_filters_non_int_values(self) -> None:
        """Non-integer values in JSON are filtered out by _int_values."""
        self._tmp.write_text(json.dumps({"a.py": 3, "b.py": "not_int", "c.py": 1.5}))
        usage = task_history._load_file_usage()
        assert usage == {"a.py": 3, "c.py": 1}
        assert "b.py" not in usage

    def test_record_creates_file(self) -> None:
        """Recording to a non-existent file creates it."""
        assert not self._tmp.exists()
        task_history._record_file_usage("x.py")
        assert self._tmp.exists()
        assert task_history._load_file_usage() == {"x.py": 1}

    def test_record_increments_count(self) -> None:
        """Recording the same file increments its count."""
        task_history._record_file_usage("a.py")
        task_history._record_file_usage("a.py")
        task_history._record_file_usage("a.py")
        assert task_history._load_file_usage()["a.py"] == 3

    def test_record_moves_key_to_end(self) -> None:
        """Recording a file moves its key to the end (most recent)."""
        task_history._record_file_usage("a.py")
        task_history._record_file_usage("b.py")
        task_history._record_file_usage("c.py")
        assert list(task_history._load_file_usage().keys()) == [
            "a.py",
            "b.py",
            "c.py",
        ]
        # Re-access a.py — it should move to the end
        task_history._record_file_usage("a.py")
        usage = task_history._load_file_usage()
        assert list(usage.keys()) == ["b.py", "c.py", "a.py"]
        assert usage["a.py"] == 2
        assert usage["b.py"] == 1
        assert usage["c.py"] == 1

    def test_max_entries_no_eviction_at_limit(self) -> None:
        """No eviction when exactly at the limit."""
        orig_max = task_history._MAX_FILE_USAGE_ENTRIES
        task_history._MAX_FILE_USAGE_ENTRIES = 3
        try:
            task_history._record_file_usage("a.py")
            task_history._record_file_usage("b.py")
            task_history._record_file_usage("c.py")
            usage = task_history._load_file_usage()
            assert len(usage) == 3
            assert set(usage.keys()) == {"a.py", "b.py", "c.py"}
        finally:
            task_history._MAX_FILE_USAGE_ENTRIES = orig_max

    def test_max_entries_evicts_oldest(self) -> None:
        """file_usage.json never exceeds _MAX_FILE_USAGE_ENTRIES."""
        orig_max = task_history._MAX_FILE_USAGE_ENTRIES
        task_history._MAX_FILE_USAGE_ENTRIES = 5
        try:
            for i in range(7):
                task_history._record_file_usage(f"file{i}.py")
            usage = task_history._load_file_usage()
            assert len(usage) == 5
            # Oldest two (file0.py, file1.py) should be evicted
            assert "file0.py" not in usage
            assert "file1.py" not in usage
            # Most recent five should remain
            for i in range(2, 7):
                assert f"file{i}.py" in usage
        finally:
            task_history._MAX_FILE_USAGE_ENTRIES = orig_max

    def test_max_entries_reaccess_preserves_entry(self) -> None:
        """Re-accessing a file prevents it from being evicted."""
        orig_max = task_history._MAX_FILE_USAGE_ENTRIES
        task_history._MAX_FILE_USAGE_ENTRIES = 3
        try:
            task_history._record_file_usage("a.py")
            task_history._record_file_usage("b.py")
            task_history._record_file_usage("c.py")
            # Re-access a.py — moves it to end (most recent)
            task_history._record_file_usage("a.py")
            # Now add d.py — should evict b.py (oldest), not a.py
            task_history._record_file_usage("d.py")
            usage = task_history._load_file_usage()
            assert len(usage) == 3
            assert "b.py" not in usage
            assert set(usage.keys()) == {"c.py", "a.py", "d.py"}
            assert usage["a.py"] == 2
        finally:
            task_history._MAX_FILE_USAGE_ENTRIES = orig_max

    def test_max_entries_preserves_counts(self) -> None:
        """Surviving entries keep their original counts after eviction."""
        orig_max = task_history._MAX_FILE_USAGE_ENTRIES
        task_history._MAX_FILE_USAGE_ENTRIES = 2
        try:
            # Build up counts: a=3, b=1
            task_history._record_file_usage("a.py")
            task_history._record_file_usage("a.py")
            task_history._record_file_usage("a.py")
            task_history._record_file_usage("b.py")
            # Add c.py — should evict a.py (oldest, since b.py was accessed more recently)
            # Wait: a.py was last accessed 2nd-to-last (before b.py), but re-accessing
            # a.py 3 times moved it to end each time. Let me trace:
            # After a.py(1): {a:1}
            # After a.py(2): {a:2}  (popped and re-inserted at end)
            # After a.py(3): {a:3}
            # After b.py(1): {a:3, b:1}
            # Now add c.py: {a:3, b:1, c:1} -> excess=1 -> evict a.py (front)
            task_history._record_file_usage("c.py")
            usage = task_history._load_file_usage()
            assert len(usage) == 2
            assert "a.py" not in usage
            assert usage["b.py"] == 1
            assert usage["c.py"] == 1
        finally:
            task_history._MAX_FILE_USAGE_ENTRIES = orig_max

    def test_max_entries_evicts_multiple_at_once(self) -> None:
        """If file pre-loaded with excess entries, eviction catches up."""
        orig_max = task_history._MAX_FILE_USAGE_ENTRIES
        task_history._MAX_FILE_USAGE_ENTRIES = 2
        try:
            # Pre-load 4 entries directly (simulating an old file before limit)
            self._tmp.write_text(json.dumps({"a.py": 1, "b.py": 2, "c.py": 3, "d.py": 4}))
            # Record one more entry — should evict down to limit
            task_history._record_file_usage("e.py")
            usage = task_history._load_file_usage()
            assert len(usage) == 2
            # a, b, c should be evicted (oldest 3), d and e remain
            assert set(usage.keys()) == {"d.py", "e.py"}
        finally:
            task_history._MAX_FILE_USAGE_ENTRIES = orig_max


def _end_dist(text: str, q: str) -> int:
    """Distance from end of path to end of rightmost query match."""
    if not q:
        return 0
    pos = text.lower().rfind(q)
    if pos < 0:
        return len(text)
    return len(text) - (pos + len(q))


def _sort_suggestions(
    file_cache: list[str],
    usage: dict[str, int],
    q: str,
) -> list[dict[str, str]]:
    """Replicate the sorting logic from the /suggestions?mode=files endpoint.

    Matches the actual production code in sorcar.py — all items start with
    type="file", frequent items get type="frequent".
    """
    frequent: list[dict[str, str]] = []
    rest: list[dict[str, str]] = []
    for path in file_cache:
        if q and q not in path.lower():
            continue
        item = {"type": "file", "text": path}
        if usage.get(path, 0) > 0:
            frequent.append(item)
        else:
            rest.append(item)
    # Build recency rank from insertion order in usage dict
    # (last key = most recently used → rank 0).
    _usage_keys = list(usage.keys())
    _recency = {k: i for i, k in enumerate(reversed(_usage_keys))}
    _n = len(_usage_keys)
    frequent.sort(
        key=lambda m: (
            _end_dist(m["text"], q),
            _recency.get(m["text"], _n),
            -usage.get(m["text"], 0),
        )
    )
    rest.sort(key=lambda m: _end_dist(m["text"], q))
    for f in frequent:
        f["type"] = "frequent"
    return (frequent + rest)[:20]


class TestSuggestionsFrequencySort(unittest.TestCase):
    """Tests for /suggestions?mode=files frequency sorting logic."""

    def test_frequent_files_first(self) -> None:
        result = _sort_suggestions(
            ["a.py", "b.py", "c.py", "dir/"],
            {"c.py": 5, "dir/": 2},
            "",
        )
        # dir/ is last in dict → most recent (recency 0), c.py → recency 1
        assert result[0] == {"type": "frequent", "text": "dir/"}
        assert result[1] == {"type": "frequent", "text": "c.py"}
        assert result[2] == {"type": "file", "text": "a.py"}
        assert result[3] == {"type": "file", "text": "b.py"}

    def test_no_frequent_files(self) -> None:
        result = _sort_suggestions(["x.py", "y.py"], {}, "")
        assert len(result) == 2
        assert all(r["type"] == "file" for r in result)

    def test_stable_order_in_rest(self) -> None:
        """Rest items with same end_dist preserve original order."""
        result = _sort_suggestions(
            ["dir1/", "a.py", "dir2/", "b.py", "dir3/"], {}, ""
        )
        assert [r["text"] for r in result] == [
            "dir1/",
            "a.py",
            "dir2/",
            "b.py",
            "dir3/",
        ]

    def test_query_filters_before_sort(self) -> None:
        result = _sort_suggestions(
            ["src/a.py", "lib/b.py", "src/c.py"],
            {"src/c.py": 10, "lib/b.py": 5},
            "src",
        )
        assert len(result) == 2
        assert result[0] == {"type": "frequent", "text": "src/c.py"}
        assert result[1] == {"type": "file", "text": "src/a.py"}

    def test_end_match_priority_in_rest(self) -> None:
        """Paths with query matching closer to end should rank higher."""
        result = _sort_suggestions(
            [
                "src/kiss/agents/sorcar/browser_ui.py",
                "src/kiss/agents/sorcar/sorcar.py",
                "src/kiss/agents/sorcar/",
            ],
            {},
            "sorcar",
        )
        # "sorcar/" ends with "sorcar/" → end_dist=1 (trailing /)
        # "sorcar.py" has "sorcar" at end before ".py" → end_dist=3
        # "browser_ui.py" has "sorcar" in the middle → end_dist=15
        assert result[0]["text"] == "src/kiss/agents/sorcar/"
        assert result[1]["text"] == "src/kiss/agents/sorcar/sorcar.py"
        assert result[2]["text"] == "src/kiss/agents/sorcar/browser_ui.py"

    def test_end_match_priority_in_frequent(self) -> None:
        """Frequent paths also sorted by end-match distance, then recency."""
        result = _sort_suggestions(
            [
                "lib/utils/config.py",
                "src/config.py",
            ],
            {"lib/utils/config.py": 5, "src/config.py": 3},
            "config",
        )
        # "src/config.py" → end_dist=3 (.py)
        # "lib/utils/config.py" → end_dist=3 (.py)
        # Same end_dist → src/config.py is more recent (last in dict)
        assert result[0]["text"] == "src/config.py"
        assert result[1]["text"] == "lib/utils/config.py"

    def test_end_match_mixed_frequent_and_rest(self) -> None:
        """Frequent items come before rest; within each group, sorted by end-match."""
        result = _sort_suggestions(
            [
                "deep/nested/utils/foo.py",
                "foo.py",
                "lib/foo/bar.py",
            ],
            {"deep/nested/utils/foo.py": 3},
            "foo",
        )
        # Frequent: "deep/nested/utils/foo.py" → end_dist=3
        # Rest: "foo.py" → end_dist=3, "lib/foo/bar.py" → end_dist=7
        assert result[0] == {"type": "frequent", "text": "deep/nested/utils/foo.py"}
        assert result[1] == {"type": "file", "text": "foo.py"}
        assert result[2] == {"type": "file", "text": "lib/foo/bar.py"}

    def test_recency_ordering(self) -> None:
        """Most recently used files (last in dict) appear first."""
        # c.py is last in dict → most recent → should appear first
        result = _sort_suggestions(
            ["a.py", "b.py", "c.py"],
            {"a.py": 10, "b.py": 5, "c.py": 1},
            "",
        )
        assert result[0]["text"] == "c.py"  # most recent (last in dict)
        assert result[1]["text"] == "b.py"
        assert result[2]["text"] == "a.py"  # least recent (first in dict)

    def test_empty_query_stable_order(self) -> None:
        """Empty query: all end_dist=0, so original order preserved."""
        result = _sort_suggestions(
            ["z.py", "a.py", "m/", "b/"],
            {},
            "",
        )
        assert [r["text"] for r in result] == ["z.py", "a.py", "m/", "b/"]

    def test_max_20_results(self) -> None:
        """At most 20 results are returned."""
        paths = [f"file{i:03d}.py" for i in range(30)]
        result = _sort_suggestions(paths, {}, "")
        assert len(result) == 20

    def test_no_match_returns_empty(self) -> None:
        """Query that matches nothing returns empty list."""
        result = _sort_suggestions(["a.py", "b.py"], {}, "zzz")
        assert result == []


class TestSelectACSpacing(unittest.TestCase):
    """Test the selectAC space insertion logic."""

    @staticmethod
    def _select_ac(
        value: str,
        cursor: int,
        item_text: str,
    ) -> tuple[str, int]:
        before = value[:cursor]
        import re

        m = re.search(r"@([^\s]*)$", before)
        if not m:
            return value, cursor
        start = len(before) - len(m.group(0))
        after = value[cursor:]
        sep = "" if (not after or after[0].isspace()) else " "
        new_val = before[:start] + "@" + item_text + sep + after
        np = start + 1 + len(item_text) + len(sep)
        return new_val, np

    def test_no_trailing_space_at_end(self) -> None:
        result, pos = self._select_ac("@sr", 3, "src/")
        assert result == "@src/"
        assert pos == 5

    def test_no_double_space_before_existing_space(self) -> None:
        result, pos = self._select_ac("@sr rest", 3, "src/")
        assert result == "@src/ rest"
        assert pos == 5

    def test_adds_space_before_text(self) -> None:
        result, pos = self._select_ac("@srrest", 3, "src/")
        assert result == "@src/ rest"
        assert pos == 6

    def test_mid_sentence(self) -> None:
        result, pos = self._select_ac(
            "check @sr and go",
            9,
            "src/",
        )
        assert result == "check @src/ and go"
        assert pos == 11


if __name__ == "__main__":
    unittest.main()
