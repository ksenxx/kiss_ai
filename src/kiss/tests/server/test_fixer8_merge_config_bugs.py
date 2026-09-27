# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Fixer-8 findings (real repos, no mocks).

F2  ``_AutocompleteMixin._refresh_files_after_task`` queues a rescan on
    the ``FileIndexRegistry`` worker: a no-op for a root nobody has
    indexed yet, and a pick-up of added/removed files for an indexed
    one, without broadcasting any event.  (The original race between a
    concurrent ``_file_cache`` writer and the post-task scan no longer
    exists: a single worker thread owns every index.)
F4  ``_MergeFlowMixin._main_dirty_files`` must not ``strip()`` porcelain
    paths: filenames with leading/trailing spaces are legal and unquoted.
F5  The porcelain fallback of ``_get_worktree_changed_files`` (extracted
    as ``merge_flow._porcelain_paths``) must not strip paths and must
    split rename entries ``old -> new`` instead of emitting the joined
    string as one bogus file.
F8  (obsolete) ``diff_merge._write_base_copy`` was removed together
    with the interactive diff/merge review workflow.
F9  ``autocomplete._ghost_suffix`` behaviour for the three completion
    kinds actually produced by ``_complete_many`` (guards the removal of
    the unreachable ``else`` arm).
F12 ``vscode_config.sanitize_config`` must reject boolean values for
    numeric keys (``max_budget: true`` used to become ``1.0``).
F17 ``file_index.FileIndex.scan`` must treat root-anchored ``.gitignore``
    entries like ``/build`` as matching at the repo root only, not at
    every depth.

All tests use real git repos / real directories in ``tmp_path`` and call
the production functions directly.  No mocks, patches, or fakes.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.server.autocomplete import _AutocompleteMixin, _ghost_suffix
from kiss.server.diff_merge import _git
from kiss.server.file_index import FileIndex, FileIndexRegistry
from kiss.server.json_printer import JsonPrinter
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.tests.conftest import posix_only


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c", "user.email=test@test",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run_git(repo, "init")
    (repo / "a.txt").write_text("hello\n")
    _run_git(repo, "add", "a.txt")
    _run_git(repo, "commit", "-m", "initial")


class _RecordingPrinter(JsonPrinter):
    """Real JsonPrinter subclass recording broadcast events in a list.

    Mirrors the transport-owning subclass pattern documented on
    :meth:`JsonPrinter.broadcast`, but records into memory instead of
    persisting so tests stay free of database side effects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* in memory instead of persisting it."""
        with self._events_lock:
            self.events.append(event)


class _AC(_AutocompleteMixin):
    """Concrete autocomplete host with the state the mixin expects."""

    def __init__(self, work_dir: str, registry: FileIndexRegistry) -> None:
        self.work_dir = work_dir
        self._state_lock = threading.RLock()
        self._file_index = registry
        self.rec_printer = _RecordingPrinter()
        self.printer = self.rec_printer


def _wait_for(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll *pred* until it holds or fail after *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def _index(registry: FileIndexRegistry, work_dir: str) -> None:
    """Build the index covering *work_dir* and wait for it."""
    done = threading.Event()
    registry.ensure(work_dir, done.set)
    assert done.wait(10.0)
    assert registry.view_for(work_dir) is not None


def _paths(registry: FileIndexRegistry, work_dir: str) -> list[str]:
    """Return the current entries of *work_dir*'s view (``[]`` when unindexed)."""
    view = registry.view_for(work_dir)
    return [] if view is None else view.paths


class _MF(_MergeFlowMixin):
    """Concrete merge-flow host; ``_main_dirty_files`` uses only work_dir."""

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir



class TestRefreshAfterTaskRace:
    """F2: ``_refresh_files_after_task`` against a real ``FileIndexRegistry``."""

    def _registry(self, tmp_path: Path) -> FileIndexRegistry:
        return FileIndexRegistry(home=str(tmp_path / "home"), cache_dir=tmp_path / "cache")

    def test_never_indexed_root_is_left_alone(self, tmp_path: Path) -> None:
        """Refreshing a root nobody asked about queues no scan.

        The worker serves jobs in order, so once a sentinel build of an
        unrelated root has completed, any job the refresh might have
        queued before it would have run too.
        """
        wd = tmp_path / "ws"
        wd.mkdir()
        (wd / "a.txt").write_text("x")
        other = tmp_path / "other"
        other.mkdir()
        registry = self._registry(tmp_path)
        try:
            host = _AC(str(wd), registry)
            host._refresh_files_after_task(str(wd))
            _index(registry, str(other))

            assert registry.view_for(str(wd)) is None
            assert set(registry._indexes) == {str(other)}
            assert host.rec_printer.events == []
        finally:
            registry.stop()

    def test_indexed_root_picks_up_added_and_removed_files(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        wd.mkdir()
        (wd / "gone.txt").write_text("x")
        registry = self._registry(tmp_path)
        try:
            host = _AC(str(wd), registry)
            _index(registry, str(wd))
            assert _paths(registry, str(wd)) == ["gone.txt"]

            # Directory mtimes have a coarse granularity: leave the build
            # behind before changing the tree so the rescan sees a new mtime.
            time.sleep(0.02)
            (wd / "gone.txt").unlink()
            (wd / "new.txt").write_text("x")
            host._refresh_files_after_task(str(wd))

            _wait_for(lambda: _paths(registry, str(wd)) == ["new.txt"])
            assert host.rec_printer.events == [], "no unsolicited files event"
        finally:
            registry.stop()

    def test_empty_work_dir_falls_back_to_the_host_work_dir(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        wd.mkdir()
        (wd / "a.txt").write_text("x")
        registry = self._registry(tmp_path)
        try:
            host = _AC(str(wd), registry)
            _index(registry, str(wd))
            time.sleep(0.02)
            (wd / "b.txt").write_text("x")
            host._refresh_files_after_task()

            _wait_for(lambda: _paths(registry, str(wd)) == ["a.txt", "b.txt"])
        finally:
            registry.stop()



class TestMainDirtyFilesNoStrip:
    def test_leading_space_untracked_filename_survives(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / " padded .txt").write_text("x\n")

        files = _MF(str(repo))._main_dirty_files(str(repo))

        assert " padded .txt" in files
        assert "padded .txt" not in files

    def test_rename_reports_new_side_only(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo)
        _run_git(repo, "mv", "a.txt", "b.txt")

        files = _MF(str(repo))._main_dirty_files(str(repo))

        assert "b.txt" in files
        assert "a.txt -> b.txt" not in files



class TestPorcelainPathsFallbackParser:
    def test_rename_split_and_spaces_preserved(self, tmp_path: Path) -> None:
        from kiss.server.merge_flow import _porcelain_paths

        repo = tmp_path / "repo"
        _make_repo(repo)
        _run_git(repo, "mv", "a.txt", "b.txt")
        (repo / " padded .txt").write_text("x\n")

        status = _git(str(repo), "status", "--porcelain")
        assert status.returncode == 0
        files = _porcelain_paths(status.stdout, rename_both_sides=True)

        assert "a.txt" in files
        assert "b.txt" in files
        assert "a.txt -> b.txt" not in files
        assert " padded .txt" in files

    def test_default_reports_new_side_only(self, tmp_path: Path) -> None:
        from kiss.server.merge_flow import _porcelain_paths

        repo = tmp_path / "repo"
        _make_repo(repo)
        _run_git(repo, "mv", "a.txt", "b.txt")

        status = _git(str(repo), "status", "--porcelain")
        files = _porcelain_paths(status.stdout)

        assert files == ["b.txt"]

    @posix_only('a double quote is not a valid NTFS file-name character')
    def test_quoted_path_unquoted_once(self, tmp_path: Path) -> None:
        from kiss.server.merge_flow import _porcelain_paths

        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / 'we"ird.txt').write_text("x\n")

        status = _git(str(repo), "status", "--porcelain")
        files = _porcelain_paths(status.stdout)

        assert 'we"ird.txt' in files



class TestGhostSuffixKinds:
    def test_task_kind_uses_full_query(self) -> None:
        out = _ghost_suffix(
            "fix", [{"type": "task", "text": "fix the flaky test"}],
        )
        assert out == " the flaky test"

    def test_trick_kind_uses_sentence_partial(self) -> None:
        out = _ghost_suffix(
            "alw", [{"type": "trick", "text": "always run tests"}],
        )
        assert out == "ays run tests"

    def test_identifier_kind_uses_trailing_token(self) -> None:
        out = _ghost_suffix(
            "use foo.ba", [{"type": "identifier", "text": "foo.bar"}],
        )
        assert out == "r"

    def test_mismatched_prefix_returns_empty(self) -> None:
        out = _ghost_suffix(
            "use foo.ba", [{"type": "identifier", "text": "qux"}],
        )
        assert out == ""

    def test_no_completions_returns_empty(self) -> None:
        assert _ghost_suffix("anything", []) == ""






class TestGitignoreAnchoring:
    """F17: ``.gitignore`` anchoring rules applied by ``FileIndex.scan``.

    The unanchored-name case uses ``out`` rather than ``node_modules``:
    the latter is now skipped unconditionally (``JUNK_DIR_NAMES``), so it
    would pass regardless of the ignore file.
    """

    def _tree(self, tmp_path: Path, gitignore: str) -> Path:
        wd = tmp_path / "ws"
        wd.mkdir()
        (wd / ".gitignore").write_text(gitignore)
        for d in ("build", "src/build", "out", "a/out", "src/generated"):
            p = wd / d
            p.mkdir(parents=True)
            (p / "f.txt").write_text("x")
        (wd / "keep.txt").write_text("x")
        return wd

    def test_root_anchored_entry_skips_root_only(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "/build\n")
        paths = FileIndex.scan(str(wd)).paths
        assert "build/f.txt" not in paths
        assert "build/" not in paths
        assert "src/build/f.txt" in paths

    def test_unanchored_name_skips_any_depth(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "out\n")
        paths = FileIndex.scan(str(wd)).paths
        assert "out/f.txt" not in paths
        assert "out/" not in paths
        assert "a/out/f.txt" not in paths
        assert "a/out/" not in paths
        assert "keep.txt" in paths

    def test_path_entry_skips_exact_path_only(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "src/generated\n")
        paths = FileIndex.scan(str(wd)).paths
        assert "src/generated/f.txt" not in paths
        assert "src/generated/" not in paths
        assert "src/build/f.txt" in paths

    def test_trailing_slash_dir_entry_unanchored(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "build/\n")
        paths = FileIndex.scan(str(wd)).paths
        assert "build/f.txt" not in paths
        assert "src/build/f.txt" not in paths
