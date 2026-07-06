# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Fixer-8 findings (real repos, no mocks).

F2  ``_AutocompleteMixin._refresh_files_after_task`` must not clobber a
    fresher ``_file_cache`` entry published by a concurrent writer while
    its background scan was running.
F4  ``_MergeFlowMixin._main_dirty_files`` must not ``strip()`` porcelain
    paths: filenames with leading/trailing spaces are legal and unquoted.
F5  The porcelain fallback of ``_get_worktree_changed_files`` (extracted
    as ``merge_flow._porcelain_paths``) must not strip paths and must
    split rename entries ``old -> new`` instead of emitting the joined
    string as one bogus file.
F8  ``diff_merge._write_base_copy`` no longer takes the ignored
    ``binary`` parameter; committed bytes are preserved exactly for both
    text (CRLF) and binary content.
F9  ``autocomplete._ghost_suffix`` behaviour for the three completion
    kinds actually produced by ``_complete_many`` (guards the removal of
    the unreachable ``else`` arm).
F12 ``vscode_config.sanitize_config`` must reject boolean values for
    numeric keys (``max_budget: true`` used to become ``1.0``).
F17 ``diff_merge._load_gitignore_dirs`` must treat root-anchored
    entries like ``/build`` as matching at the repo root only, not at
    every depth.

All tests use real git repos / real directories in ``tmp_path`` and call
the production functions directly.  No mocks, patches, or fakes.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.vscode.autocomplete import _AutocompleteMixin, _ghost_suffix
from kiss.agents.vscode.diff_merge import _git, _scan_files
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.vscode_config import DEFAULTS, sanitize_config


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

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        self._state_lock = threading.RLock()
        self._file_cache: dict[str, list[str]] = {}
        self.rec_printer = _RecordingPrinter()
        self.printer = self.rec_printer


class _MF(_MergeFlowMixin):
    """Concrete merge-flow host; ``_main_dirty_files`` uses only work_dir."""

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir


# ---------------------------------------------------------------------------
# F2 — post-task refresh must not clobber a fresher concurrent cache entry
# ---------------------------------------------------------------------------


class TestRefreshAfterTaskRace:
    def test_fresher_concurrent_entry_survives_post_task_scan(
        self, tmp_path: Path,
    ) -> None:
        """A cache entry published while the post-task scan runs wins.

        The scan is slowed deterministically by a large tree (2000
        files) so the main thread can publish a fresher entry before
        the background thread reaches its cache write.
        """
        wd = tmp_path / "big"
        for d in range(40):
            sub = wd / f"d{d:02d}"
            sub.mkdir(parents=True)
            for f in range(50):
                (sub / f"f{f:02d}.txt").write_text("x")
        wd_str = str(wd)

        host = _AC(wd_str)
        host._file_cache[wd_str] = ["stale.txt"]

        host._refresh_files_after_task(wd_str)
        # Concurrent writer (e.g. explicit refresh) publishes a fresher
        # scan while the post-task background scan is still walking.
        fresh = ["fresh.txt"]
        with host._state_lock:
            host._file_cache[wd_str] = fresh

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with host._state_lock:
                current = host._file_cache[wd_str]
            assert current is fresh, (
                "post-task scan clobbered the fresher concurrent entry"
            )
            time.sleep(0.05)

    def test_no_cache_entry_is_noop(self, tmp_path: Path) -> None:
        host = _AC(str(tmp_path))
        host._refresh_files_after_task(str(tmp_path))
        time.sleep(0.2)
        assert host._file_cache == {}
        assert host.rec_printer.events == []

    def test_changed_set_updates_cache_and_broadcasts(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "new.txt").write_text("x")
        wd_str = str(tmp_path)
        host = _AC(wd_str)
        host._file_cache[wd_str] = ["gone.txt"]
        host._refresh_files_after_task(wd_str)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with host._state_lock:
                if host._file_cache[wd_str] == ["new.txt"]:
                    break
            time.sleep(0.02)
        assert host._file_cache[wd_str] == ["new.txt"]
        assert any(e.get("type") == "files" for e in host.rec_printer.events)


# ---------------------------------------------------------------------------
# F4 — _main_dirty_files must not strip space-adjacent filenames
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# F5 — porcelain fallback parser: no strip, renames split, both sides
# ---------------------------------------------------------------------------


class TestPorcelainPathsFallbackParser:
    def test_rename_split_and_spaces_preserved(self, tmp_path: Path) -> None:
        from kiss.agents.vscode.merge_flow import _porcelain_paths

        repo = tmp_path / "repo"
        _make_repo(repo)
        _run_git(repo, "mv", "a.txt", "b.txt")
        (repo / " padded .txt").write_text("x\n")

        status = _git(str(repo), "status", "--porcelain")
        assert status.returncode == 0
        files = _porcelain_paths(status.stdout, rename_both_sides=True)

        # The worktree fallback mirrors the primary ``--no-renames``
        # diff, which lists BOTH sides of a rename.
        assert "a.txt" in files
        assert "b.txt" in files
        assert "a.txt -> b.txt" not in files
        assert " padded .txt" in files

    def test_default_reports_new_side_only(self, tmp_path: Path) -> None:
        from kiss.agents.vscode.merge_flow import _porcelain_paths

        repo = tmp_path / "repo"
        _make_repo(repo)
        _run_git(repo, "mv", "a.txt", "b.txt")

        status = _git(str(repo), "status", "--porcelain")
        files = _porcelain_paths(status.stdout)

        assert files == ["b.txt"]

    def test_quoted_path_unquoted_once(self, tmp_path: Path) -> None:
        from kiss.agents.vscode.merge_flow import _porcelain_paths

        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / 'we"ird.txt').write_text("x\n")

        status = _git(str(repo), "status", "--porcelain")
        files = _porcelain_paths(status.stdout)

        assert 'we"ird.txt' in files


# ---------------------------------------------------------------------------
# F8 — _write_base_copy: dead ``binary`` parameter removed; bytes exact
# ---------------------------------------------------------------------------


class TestWriteBaseCopyNoBinaryParam:
    def test_crlf_text_bytes_preserved(self, tmp_path: Path) -> None:
        from kiss.agents.vscode.diff_merge import _write_base_copy

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init")
        committed = b"alpha\r\nbeta\r\n"
        (repo / "crlf.txt").write_bytes(committed)
        _run_git(repo, "add", "crlf.txt")
        _run_git(repo, "commit", "-m", "crlf")

        base = _write_base_copy(
            str(repo), tmp_path / "merge", tmp_path / "ub", "crlf.txt",
            "HEAD",
        )
        assert base.read_bytes() == committed

    def test_binary_bytes_preserved(self, tmp_path: Path) -> None:
        from kiss.agents.vscode.diff_merge import _write_base_copy

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init")
        committed = b"\x00\x01\xffPNG\r\n\x00"
        (repo / "blob.bin").write_bytes(committed)
        _run_git(repo, "add", "blob.bin")
        _run_git(repo, "commit", "-m", "bin")

        base = _write_base_copy(
            str(repo), tmp_path / "merge", tmp_path / "ub", "blob.bin",
            "HEAD",
        )
        assert base.read_bytes() == committed

    def test_missing_blob_writes_empty_base(self, tmp_path: Path) -> None:
        from kiss.agents.vscode.diff_merge import _write_base_copy

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init")
        (repo / "a.txt").write_text("x\n")
        _run_git(repo, "add", "a.txt")
        _run_git(repo, "commit", "-m", "a")

        base = _write_base_copy(
            str(repo), tmp_path / "merge", tmp_path / "ub", "brand-new.txt",
            "HEAD",
        )
        assert base.read_bytes() == b""


# ---------------------------------------------------------------------------
# F9 — _ghost_suffix behaviour for the three produced kinds
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# F12 — sanitize_config: booleans are not numbers
# ---------------------------------------------------------------------------


class TestSanitizeConfigBooleanBudget:
    def test_true_budget_falls_back_to_default(self) -> None:
        out = sanitize_config({"max_budget": True})
        assert out["max_budget"] == DEFAULTS["max_budget"]

    def test_false_budget_falls_back_to_default(self) -> None:
        out = sanitize_config({"max_budget": False})
        assert out["max_budget"] == DEFAULTS["max_budget"]

    def test_finite_numbers_still_accepted(self) -> None:
        assert sanitize_config({"max_budget": 55})["max_budget"] == 55
        assert sanitize_config({"max_budget": 55.5})["max_budget"] == 55.5
        assert sanitize_config({"max_budget": "42"})["max_budget"] == 42.0

    def test_bool_defaults_still_coerce_truthy(self) -> None:
        bool_keys = [k for k, v in DEFAULTS.items() if isinstance(v, bool)]
        for key in bool_keys:
            assert sanitize_config({key: 1})[key] is True
            assert sanitize_config({key: 0})[key] is False


# ---------------------------------------------------------------------------
# F17 — root-anchored .gitignore entries match at the root only
# ---------------------------------------------------------------------------


class TestGitignoreAnchoring:
    def _tree(self, tmp_path: Path, gitignore: str) -> Path:
        wd = tmp_path / "ws"
        wd.mkdir()
        (wd / ".gitignore").write_text(gitignore)
        for d in ("build", "src/build", "node_modules",
                  "a/node_modules", "src/generated"):
            p = wd / d
            p.mkdir(parents=True)
            (p / "f.txt").write_text("x")
        (wd / "keep.txt").write_text("x")
        return wd

    def test_root_anchored_entry_skips_root_only(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "/build\n")
        paths = _scan_files(str(wd))
        assert "build/f.txt" not in paths
        assert "build/" not in paths
        # git does NOT ignore src/build for a root-anchored /build.
        assert "src/build/f.txt" in paths

    def test_unanchored_name_skips_any_depth(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "node_modules\n")
        paths = _scan_files(str(wd))
        assert "node_modules/f.txt" not in paths
        assert "a/node_modules/f.txt" not in paths
        assert "keep.txt" in paths

    def test_path_entry_skips_exact_path_only(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "src/generated\n")
        paths = _scan_files(str(wd))
        assert "src/generated/f.txt" not in paths
        assert "src/build/f.txt" in paths

    def test_trailing_slash_dir_entry_unanchored(self, tmp_path: Path) -> None:
        wd = self._tree(tmp_path, "build/\n")
        paths = _scan_files(str(wd))
        assert "build/f.txt" not in paths
        assert "src/build/f.txt" not in paths
