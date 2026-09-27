# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``@``-mention picker favouring source files.

``FileIndex.scan`` leaves out the contents of *bulk data directories* —
subtrees of at least ``_BULK_DATA_MIN_FILES`` files of which at least
``_BULK_DATA_MIN_FRACTION`` carry a data suffix (JSON/log/text results) —
while keeping the directory's own ``dir/`` entry.  Code-heavy trees of any
size are listed in full.  The index orders its entries once, at build
time: shallow entries before deep ones, and everything inside a *wide
container* (a directory with at least ``WIDE_DIR_MIN_CHILDREN``
subdirectories, i.e. ``artifacts/<run_id>/``) after every other entry.
``FileView.search`` then ranks matches by how close to the end of the path
the query occurs, ties keeping index order, so generated artifacts do not
crowd out the project's own files while remaining reachable.

The ``_SCAN_FILES_CAP`` early exit is covered by
``test_wave2_merge_bugs.TestScanFilesCapCoversDirectories``.
"""

from __future__ import annotations

from pathlib import Path

from kiss.server.file_index import (
    _BULK_DATA_MIN_FILES,
    WIDE_DIR_MIN_CHILDREN,
    FileIndex,
    _bulk_data_dirs,
)

MANY = _BULK_DATA_MIN_FILES + 50


def _touch_many(directory: Path, count: int, suffix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"f{i:04d}{suffix}").write_text("x")


def _touch(root: Path, *rel: str) -> None:
    """Create empty files at each *rel* path below *root*."""
    for r in rel:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def _search(root: Path, query: str, usage: dict[str, int], limit: int = 20) -> list[str]:
    """Scan *root* and return the picker texts for *query*."""
    view = FileIndex.scan(str(root)).view("")
    return [r["text"] for r in view.search(query, usage, limit=limit)]


class TestBulkDataDirectoriesAreSkipped:
    def test_json_dump_contents_dropped_but_entry_kept(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        (wd / "src").mkdir(parents=True)
        (wd / "src" / "main.py").write_text("x")
        (wd / "README.md").write_text("x")
        _touch_many(wd / "runs", MANY, ".json")
        _touch_many(wd / "runs" / "nested" / "deeper", 3, ".log")

        paths = FileIndex.scan(str(wd)).paths

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

        paths = FileIndex.scan(str(wd)).paths

        assert "papers/ablation/f0000.py" in paths
        assert "papers/ablation/f0000.json" in paths
        # 2 dirs + MANY py + 20 json (the json names overwrite nothing: same
        # stem, different suffix).
        assert len(paths) == 2 + MANY + 20

    def test_small_data_directory_is_kept(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        _touch_many(wd / "fixtures", _BULK_DATA_MIN_FILES - 1, ".json")

        paths = FileIndex.scan(str(wd)).paths

        assert "fixtures/f0000.json" in paths
        assert len(paths) == 1 + _BULK_DATA_MIN_FILES - 1

    def test_repository_that_is_itself_a_data_set_stays_browsable(
        self, tmp_path: Path
    ) -> None:
        """The root is never treated as bulk data, only its subdirectories."""
        wd = tmp_path / "ws"
        _touch_many(wd, MANY, ".csv")

        paths = FileIndex.scan(str(wd)).paths

        assert len(paths) == MANY

    def test_bulk_directory_nested_in_a_code_tree(self, tmp_path: Path) -> None:
        """Only the results subtree is dropped; the surrounding code stays."""
        wd = tmp_path / "ws"
        _touch_many(wd / "bench", 5, ".py")
        _touch_many(wd / "bench" / "results" / "run1", MANY // 2, ".jsonl")
        _touch_many(wd / "bench" / "results" / "run2", MANY // 2, ".TXT")

        index = FileIndex.scan(str(wd))
        paths = index.paths

        assert "bench/" in paths
        assert "bench/f0000.py" in paths
        assert "bench/results/" in paths
        assert "bench/results/run1/" not in paths
        assert "bench/results/run1/f0000.jsonl" not in paths
        assert "bench/results/run2/f0000.TXT" not in paths
        assert len(paths) == 1 + 5 + 1
        assert "bench" in index.dirs
        assert "bench/results" not in index.dirs, "a collapsed directory cannot serve a tab"

    def test_data_files_at_the_parent_level_count_towards_the_subtree(
        self, tmp_path: Path
    ) -> None:
        """Files spread over several small subdirectories still add up."""
        wd = tmp_path / "ws"
        for i in range(10):
            _touch_many(wd / "jobs" / f"job{i}", MANY // 10 + 1, ".yaml")

        paths = FileIndex.scan(str(wd)).paths

        assert paths == ["jobs/"]


class TestBulkDataHelpers:
    def test_bulk_data_dirs_thresholds(self) -> None:
        counts = {
            "": [1000, 1000],
            "exact": [_BULK_DATA_MIN_FILES, _BULK_DATA_MIN_FILES],
            "ninety": [1000, 900],
            "below_fraction": [1000, 899],
            "too_small": [_BULK_DATA_MIN_FILES - 1, _BULK_DATA_MIN_FILES - 1],
        }
        assert _bulk_data_dirs(counts) == {"exact", "ninety"}

    def test_bulk_child_is_not_folded_into_its_parent(self) -> None:
        """``bench`` keeps its five scripts; only ``bench/results`` is bulk."""
        counts = {
            "": [0, 0],
            "bench": [5, 0],
            "bench/results": [0, 0],
            "bench/results/run1": [150, 150],
            "bench/results/run2": [150, 150],
        }
        assert _bulk_data_dirs(counts) == {"bench/results"}


class TestShallowerPathsRankFirstAmongEqualMatches:
    def test_equal_matches_are_ordered_by_depth(self, tmp_path: Path) -> None:
        _touch(
            tmp_path,
            "artifacts/run_0001/README.md",
            "artifacts/run_0002/README.md",
            "README.md",
            "src/README.md",
        )
        assert _search(tmp_path, "README", {}) == [
            "README.md",
            "src/README.md",
            "artifacts/run_0001/README.md",
            "artifacts/run_0002/README.md",
        ]

    def test_match_position_still_beats_depth(self, tmp_path: Path) -> None:
        """A closer-to-the-end match wins even when it is deeper."""
        _touch(tmp_path, "config_loader.py", "src/kiss/core/config.py")
        ranked = _search(tmp_path, "config", {})
        assert ranked == ["src/kiss/core/config.py", "config_loader.py"]

    def test_empty_query_lists_root_entries_before_nested_ones(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a/deep/x.py", "README.md", "a/y.py")
        (tmp_path / "b").mkdir()
        ranked = _search(tmp_path, "", {})
        assert ranked == ["README.md", "a/", "b/", "a/y.py", "a/deep/", "a/deep/x.py"]

    def test_wide_container_contents_rank_last_but_stay_reachable(self, tmp_path: Path) -> None:
        runs = [f"papers/artifacts/run_{i:03d}/" for i in range(WIDE_DIR_MIN_CHILDREN)]
        _touch(
            tmp_path,
            "src/tests/test_cli.py",
            "papers/notes/test_cli.py",
            *[r + "tests/test_cli.py" for r in runs],
        )
        index = FileIndex.scan(str(tmp_path))
        paths = index.paths
        # Every entry of the wide container comes after every entry outside it,
        # however deep the latter is.
        first_artifact = paths.index(runs[0])
        outside = [p for p in paths if not p.startswith("papers/artifacts/run_")]
        assert all(paths.index(p) < first_artifact for p in outside)
        assert "papers/artifacts/" in outside

        ranked = [r["text"] for r in index.view("").search("test_cli", {}, limit=200)]
        # Equally good matches: the project's own files first (index order:
        # breadth-first, alphabetical within a directory), then the runs.
        assert ranked[:2] == ["papers/notes/test_cli.py", "src/tests/test_cli.py"]
        assert ranked[2:] == [r + "tests/test_cli.py" for r in runs]
        # Even a better match position inside the container loses to a
        # project file: ``test`` ends ``runs/.../tests`` but not the
        # project's ``test_long_descriptive_name.py``.
        _touch(tmp_path, "src/tests/test_long_descriptive_name.py")
        index = FileIndex.scan(str(tmp_path))
        ranked = [r["text"] for r in index.view("").search("test", {}, limit=100)]
        assert "src/tests/test_long_descriptive_name.py" in ranked[:4]
        first_run = next(i for i, p in enumerate(ranked) if p.startswith("papers/artifacts/run_"))
        assert first_run > ranked.index("src/tests/test_long_descriptive_name.py")
        # A query that only the artifacts satisfy still lists them.
        only = [r["text"] for r in index.view("").search("run_0", {})]
        assert only
        assert all(p.startswith("papers/artifacts/run_0") for p in only)

    def test_wide_dirs_threshold_and_root_exclusion(self, tmp_path: Path) -> None:
        # ``pkg/sub00/a.py`` is three levels deep, ``other/deep/er/a.py`` four.
        few = tmp_path / "few"
        subs = [f"pkg/sub{i:02d}/a.py" for i in range(WIDE_DIR_MIN_CHILDREN - 1)]
        _touch(few, "other/deep/er/a.py", *subs)
        paths = FileIndex.scan(str(few)).paths
        assert paths.index("pkg/sub00/a.py") < paths.index("other/deep/er/a.py"), (
            "below the threshold pkg/ is not a container: shallow entries stay first"
        )

        wide = tmp_path / "wide"
        _touch(wide, "other/deep/er/a.py", *subs, f"pkg/sub{WIDE_DIR_MIN_CHILDREN - 1:02d}/a.py")
        paths = FileIndex.scan(str(wide)).paths
        assert paths.index("other/deep/er/a.py") < paths.index("pkg/sub00/")
        assert paths.index("pkg/") < paths.index("other/deep/er/a.py"), (
            "the container itself is not demoted"
        )

        # Many top-level directories never make the root a wide container:
        # its entries keep their order and are not demoted below a nested
        # container's contents.
        root = tmp_path / "root"
        tops = [f"top{i:02d}" for i in range(WIDE_DIR_MIN_CHILDREN)]
        _touch(
            root, "top.py", "src/deep/test_cli.py", *[f"{t}/a.py" for t in tops],
            *[f"top00/runs/run{i:03d}/test_cli.py" for i in range(WIDE_DIR_MIN_CHILDREN)],
        )
        index = FileIndex.scan(str(root))
        paths = index.paths
        assert paths[0] == "top.py"
        assert index.wide_from == paths.index("top00/runs/run000/")
        ranked = [r["text"] for r in index.view("").search("test_cli", {})]
        assert ranked[0] == "src/deep/test_cli.py"

    def test_frequent_files_keep_their_usage_order(self, tmp_path: Path) -> None:
        _touch(tmp_path, "deep/a/b/used.py", "used.py")
        usage = {"deep/a/b/used.py": 3}
        ranked = FileIndex.scan(str(tmp_path)).view("").search("used", usage)
        assert [r["text"] for r in ranked] == ["deep/a/b/used.py", "used.py"]
        assert ranked[0]["type"] == "frequent"
        assert ranked[1]["type"] == "file"
