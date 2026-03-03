"""Tests for file usage tracking and @ file picker frequency sorting."""

import json
import unittest
from pathlib import Path

from kiss.agents.sorcar import task_history


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

    def test_load_empty(self) -> None:
        assert task_history._load_file_usage() == {}

    def test_record_and_load(self) -> None:
        task_history._record_file_usage("src/foo.py")
        task_history._record_file_usage("src/foo.py")
        task_history._record_file_usage("src/bar.py")
        usage = task_history._load_file_usage()
        assert usage["src/foo.py"] == 2
        assert usage["src/bar.py"] == 1

    def test_load_corrupt_json(self) -> None:
        self._tmp.write_text("not json")
        assert task_history._load_file_usage() == {}

    def test_load_non_dict_json(self) -> None:
        self._tmp.write_text("[1,2,3]")
        assert task_history._load_file_usage() == {}

    def test_load_filters_non_numeric(self) -> None:
        self._tmp.write_text(json.dumps({"a": 5, "b": "x"}))
        usage = task_history._load_file_usage()
        assert usage == {"a": 5}

    def test_increments_existing(self) -> None:
        self._tmp.write_text(json.dumps({"x.py": 3}))
        task_history._record_file_usage("x.py")
        assert task_history._load_file_usage()["x.py"] == 4


class TestSuggestionsFrequencySort(unittest.TestCase):
    """Tests for /suggestions?mode=files frequency sorting logic."""

    def test_frequent_files_first(self) -> None:
        file_cache = ["a.py", "b.py", "c.py", "dir/"]
        usage = {"c.py": 5, "dir/": 2}

        matched: list[dict[str, str]] = []
        for path in file_cache:
            ptype = "dir" if path.endswith("/") else "file"
            matched.append({"type": ptype, "text": path})

        frequent: list[dict[str, str]] = []
        rest: list[dict[str, str]] = []
        for item in matched:
            if usage.get(item["text"], 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)
        frequent.sort(key=lambda m: (m["type"] != "file", -usage.get(m["text"], 0)))
        rest.sort(key=lambda m: m["type"] != "file")
        for f in frequent:
            f["type"] = "frequent_" + f["type"]
        result = (frequent + rest)[:20]

        assert result[0]["text"] == "c.py"
        assert result[0]["type"] == "frequent_file"
        assert result[1]["text"] == "dir/"
        assert result[1]["type"] == "frequent_dir"
        assert result[2]["text"] == "a.py"
        assert result[2]["type"] == "file"
        assert result[3]["text"] == "b.py"
        assert result[3]["type"] == "file"

    def test_no_frequent_files(self) -> None:
        file_cache = ["x.py", "y.py"]
        usage: dict[str, int] = {}
        frequent = []
        rest = []
        for path in file_cache:
            item = {
                "type": "dir" if path.endswith("/") else "file",
                "text": path,
            }
            if usage.get(path, 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)
        result = (frequent + rest)[:20]
        assert len(result) == 2
        assert all(r["type"] == "file" for r in result)

    def test_files_before_folders_in_rest(self) -> None:
        file_cache = ["dir1/", "a.py", "dir2/", "b.py", "dir3/"]
        usage: dict[str, int] = {}
        frequent: list[dict[str, str]] = []
        rest: list[dict[str, str]] = []
        for path in file_cache:
            item = {
                "type": "dir" if path.endswith("/") else "file",
                "text": path,
            }
            if usage.get(path, 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)
        rest.sort(key=lambda m: m["type"] != "file")
        result = (frequent + rest)[:20]
        assert result[0]["text"] == "a.py"
        assert result[1]["text"] == "b.py"
        assert result[2]["text"] == "dir1/"
        assert result[3]["text"] == "dir2/"
        assert result[4]["text"] == "dir3/"

    def test_files_before_folders_in_frequent(self) -> None:
        file_cache = ["dir1/", "a.py", "dir2/", "b.py"]
        usage = {"dir1/": 10, "a.py": 5, "dir2/": 3, "b.py": 1}
        frequent: list[dict[str, str]] = []
        rest: list[dict[str, str]] = []
        for path in file_cache:
            item = {
                "type": "dir" if path.endswith("/") else "file",
                "text": path,
            }
            if usage.get(path, 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)
        frequent.sort(key=lambda m: (m["type"] != "file", -usage.get(m["text"], 0)))
        rest.sort(key=lambda m: m["type"] != "file")
        for f in frequent:
            f["type"] = "frequent_" + f["type"]
        result = (frequent + rest)[:20]
        # Files come before dirs, sorted by usage within each group
        assert result[0]["text"] == "a.py"
        assert result[0]["type"] == "frequent_file"
        assert result[1]["text"] == "b.py"
        assert result[1]["type"] == "frequent_file"
        assert result[2]["text"] == "dir1/"
        assert result[2]["type"] == "frequent_dir"
        assert result[3]["text"] == "dir2/"
        assert result[3]["type"] == "frequent_dir"

    def test_query_filters_before_sort(self) -> None:
        file_cache = ["src/a.py", "lib/b.py", "src/c.py"]
        usage = {"src/c.py": 10, "lib/b.py": 5}
        q = "src"
        frequent = []
        rest = []
        for path in file_cache:
            if q not in path.lower():
                continue
            item = {
                "type": "dir" if path.endswith("/") else "file",
                "text": path,
            }
            if usage.get(path, 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)
        frequent.sort(key=lambda m: (m["type"] != "file", -usage.get(m["text"], 0)))
        rest.sort(key=lambda m: m["type"] != "file")
        for f in frequent:
            f["type"] = "frequent_" + f["type"]
        result = (frequent + rest)[:20]
        assert len(result) == 2
        assert result[0]["text"] == "src/c.py"
        assert result[0]["type"] == "frequent_file"
        assert result[1]["text"] == "src/a.py"


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
