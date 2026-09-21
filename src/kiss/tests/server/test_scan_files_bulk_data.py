# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``@``-mention picker favouring source files.

``_scan_files`` leaves out the contents of *bulk data directories* —
subtrees of at least ``_BULK_DATA_MIN_FILES`` files of which at least
``_BULK_DATA_MIN_FRACTION`` carry a data suffix (JSON/log/text results) —
while keeping the directory's own ``dir/`` entry.  Code-heavy trees of any
size are listed in full.  ``rank_file_suggestions`` then ranks files inside
*wide containers* (directories with at least ``WIDE_DIR_MIN_CHILDREN``
subdirectories, i.e. ``artifacts/<run_id>/``) after every other match, and
orders equally good matches shallowest first, so generated artifacts do not
crowd out the project's own files while remaining reachable.

The ``_SCAN_FILES_CAP`` early exit (1,000,000 entries) is not exercised:
reaching it needs a million filesystem entries, and lowering the constant
would mean patching the module under test.
"""

from __future__ import annotations

from pathlib import Path

from kiss.server.diff_merge import (
    _BULK_DATA_MIN_FILES,
    _bulk_data_dirs,
    _is_under,
    _scan_files,
)
from kiss.server.helpers import WIDE_DIR_MIN_CHILDREN, _wide_dirs, rank_file_suggestions

MANY = _BULK_DATA_MIN_FILES + 50


def _touch_many(directory: Path, count: int, suffix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"f{i:04d}{suffix}").write_text("x")


class TestBulkDataDirectoriesAreSkipped:
    def test_json_dump_contents_dropped_but_entry_kept(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        (wd / "src").mkdir(parents=True)
        (wd / "src" / "main.py").write_text("x")
        (wd / "README.md").write_text("x")
        _touch_many(wd / "runs", MANY, ".json")
        _touch_many(wd / "runs" / "nested" / "deeper", 3, ".log")

        paths = _scan_files(str(wd))

        assert "runs/" in paths
        assert "README.md" in paths
        assert "src/" in paths
        assert "src/main.py" in paths
        assert not any(p.startswith("runs/") and p != "runs/" for p in paths)
        assert len(paths) == 4

    def test_code_heavy_tree_of_same_size_is_listed_in_full(self, tmp_path: Path) -> None:
        """Thousands of ``.py`` files with a few JSON siblings are source, not data."""
        wd = tmp_path / "ws"
        _touch_many(wd / "papers" / "ablation", MANY, ".py")
        _touch_many(wd / "papers" / "ablation", 20, ".json")

        paths = _scan_files(str(wd))

        assert "papers/ablation/f0000.py" in paths
        assert "papers/ablation/f0000.json" in paths
        # 2 dirs + MANY py + 20 json (the json names overwrite nothing: same
        # stem, different suffix).
        assert len(paths) == 2 + MANY + 20

    def test_small_data_directory_is_kept(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        _touch_many(wd / "fixtures", _BULK_DATA_MIN_FILES - 1, ".json")

        paths = _scan_files(str(wd))

        assert "fixtures/f0000.json" in paths
        assert len(paths) == 1 + _BULK_DATA_MIN_FILES - 1

    def test_repository_that_is_itself_a_data_set_stays_browsable(
        self, tmp_path: Path
    ) -> None:
        """The root is never treated as bulk data, only its subdirectories."""
        wd = tmp_path / "ws"
        _touch_many(wd, MANY, ".csv")

        paths = _scan_files(str(wd))

        assert len(paths) == MANY

    def test_bulk_directory_nested_in_a_code_tree(self, tmp_path: Path) -> None:
        """Only the results subtree is dropped; the surrounding code stays."""
        wd = tmp_path / "ws"
        _touch_many(wd / "bench", 5, ".py")
        _touch_many(wd / "bench" / "results" / "run1", MANY // 2, ".jsonl")
        _touch_many(wd / "bench" / "results" / "run2", MANY // 2, ".TXT")

        paths = _scan_files(str(wd))

        assert "bench/" in paths
        assert "bench/f0000.py" in paths
        assert "bench/results/" in paths
        assert "bench/results/run1/" not in paths
        assert "bench/results/run1/f0000.jsonl" not in paths
        assert "bench/results/run2/f0000.TXT" not in paths
        assert len(paths) == 1 + 5 + 1

    def test_data_files_at_the_parent_level_count_towards_the_subtree(
        self, tmp_path: Path
    ) -> None:
        """Files spread over several small subdirectories still add up."""
        wd = tmp_path / "ws"
        for i in range(10):
            _touch_many(wd / "jobs" / f"job{i}", MANY // 10 + 1, ".yaml")

        paths = _scan_files(str(wd))

        assert paths == ["jobs/"]


class TestBulkDataHelpers:
    def test_is_under_excludes_the_directory_itself(self) -> None:
        bulk = {"a/b"}
        assert _is_under("a/b/c.txt", bulk)
        assert _is_under("a/b/c/", bulk)
        assert _is_under("a/b/c/d.txt", bulk)
        assert not _is_under("a/b/", bulk)
        assert not _is_under("a/bc/x.txt", bulk)
        assert not _is_under("a/", bulk)
        assert not _is_under("top.txt", bulk)

    def test_bulk_data_dirs_thresholds(self) -> None:
        counts = {
            ".": [1000, 1000],
            "exact": [_BULK_DATA_MIN_FILES, _BULK_DATA_MIN_FILES],
            "ninety": [1000, 900],
            "below_fraction": [1000, 899],
            "too_small": [_BULK_DATA_MIN_FILES - 1, _BULK_DATA_MIN_FILES - 1],
        }
        assert _bulk_data_dirs(counts) == {"exact", "ninety"}

    def test_bulk_child_is_not_folded_into_its_parent(self) -> None:
        """``bench`` keeps its five scripts; only ``bench/results`` is bulk."""
        counts = {
            ".": [0, 0],
            "bench": [5, 0],
            "bench/results": [0, 0],
            "bench/results/run1": [150, 150],
            "bench/results/run2": [150, 150],
        }
        assert _bulk_data_dirs(counts) == {"bench/results"}


class TestShallowerPathsRankFirstAmongEqualMatches:
    def test_equal_matches_are_ordered_by_depth(self) -> None:
        cache = [
            "artifacts/run_0001/README.md",
            "artifacts/run_0002/README.md",
            "README.md",
            "src/README.md",
        ]
        ranked = [r["text"] for r in rank_file_suggestions(cache, "README", {})]
        assert ranked == [
            "README.md",
            "src/README.md",
            "artifacts/run_0001/README.md",
            "artifacts/run_0002/README.md",
        ]

    def test_match_position_still_beats_depth(self) -> None:
        """A closer-to-the-end match wins even when it is deeper."""
        cache = ["config_loader.py", "src/kiss/core/config.py"]
        ranked = [r["text"] for r in rank_file_suggestions(cache, "config", {})]
        assert ranked == ["src/kiss/core/config.py", "config_loader.py"]

    def test_empty_query_lists_root_entries_before_nested_ones(self) -> None:
        cache = ["a/deep/x.py", "b/", "README.md", "a/y.py"]
        ranked = [r["text"] for r in rank_file_suggestions(cache, "", {})]
        assert ranked == ["b/", "README.md", "a/y.py", "a/deep/x.py"]

    def test_wide_container_contents_rank_last_but_stay_reachable(self) -> None:
        runs = [f"papers/artifacts/run_{i:03d}/" for i in range(WIDE_DIR_MIN_CHILDREN)]
        cache = (
            ["papers/", "papers/artifacts/", "src/", "src/tests/"]
            + runs
            + [r + "tests/" for r in runs]
            + [r + "tests/test_cli.py" for r in runs]
            + ["src/tests/test_long_descriptive_name.py", "papers/notes_test.md"]
        )
        ranked = [r["text"] for r in rank_file_suggestions(cache, "test", {}, limit=100)]
        # Shortest suffix after the match wins among non-artifacts even though
        # ``test_cli.py`` would beat both on that measure.
        assert ranked[:3] == [
            "src/tests/",
            "papers/notes_test.md",
            "src/tests/test_long_descriptive_name.py",
        ]
        assert ranked[3:].count("papers/artifacts/run_000/tests/") == 1
        assert all(p.startswith("papers/artifacts/run_") for p in ranked[3:])
        # A query that only the artifacts satisfy still lists them.
        only = [r["text"] for r in rank_file_suggestions(cache, "test_cli", {})]
        assert only == [r + "tests/test_cli.py" for r in runs][: len(only)]

    def test_wide_dirs_threshold_and_root_exclusion(self) -> None:
        few = [f"pkg/sub{i}/" for i in range(WIDE_DIR_MIN_CHILDREN - 1)]
        assert _wide_dirs(few + ["pkg/"]) == set()
        assert _wide_dirs(few + ["pkg/", f"pkg/sub{WIDE_DIR_MIN_CHILDREN - 1}/"]) == {"pkg"}
        # Many top-level directories do not make the repository root a container.
        assert _wide_dirs([f"top{i}/" for i in range(WIDE_DIR_MIN_CHILDREN)]) == set()

    def test_frequent_files_keep_their_usage_order(self) -> None:
        cache = ["deep/a/b/used.py", "used.py"]
        usage = {"deep/a/b/used.py": 3}
        ranked = rank_file_suggestions(cache, "used", usage)
        assert [r["text"] for r in ranked] == ["deep/a/b/used.py", "used.py"]
        assert ranked[0]["type"] == "frequent"
        assert ranked[1]["type"] == "file"
