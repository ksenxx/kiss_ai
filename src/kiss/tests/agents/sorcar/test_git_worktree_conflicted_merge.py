# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the conflicted-merge helpers in ``GitWorktreeOps``.

``begin_conflicted_merge`` re-applies a task branch onto the original
branch leaving the conflict markers in place, ``finish_conflicted_merge``
verifies the resolution and commits it, and ``abort_conflicted_merge``
restores the clean tree.  Every test uses real git in a temp repo.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, MergeResult, _git


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
    )
    return result.stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _run(path, "config", "user.email", "t@t.com")
    _run(path, "config", "user.name", "T")
    (path / "f.txt").write_text("a\nb\nc\n")
    _run(path, "add", ".")
    _run(path, "commit", "-qm", "initial")
    return path


def _conflicting_branch(repo: Path, *, with_baseline: bool) -> tuple[str, str | None]:
    """Create ``task`` from HEAD, edit ``b`` on both sides, return (branch, baseline).

    With a baseline the branch starts with a baseline commit (as a real
    worktree does) and the agent's commits follow it.
    """
    _run(repo, "checkout", "-qb", "task")
    baseline = None
    if with_baseline:
        (repo / "dirty.txt").write_text("user dirty state\n")
        _run(repo, "add", ".")
        _run(repo, "commit", "-qm", "kiss: baseline")
        baseline = _run(repo, "rev-parse", "HEAD").strip()
    (repo / "f.txt").write_text("a\nB-agent\nc\n")
    _run(repo, "commit", "-qam", "agent edit")
    (repo / "f.txt").write_text("a\nB-agent\nc\nd-agent\n")
    _run(repo, "commit", "-qam", "agent second edit")
    _run(repo, "checkout", "-q", "main")
    (repo / "f.txt").write_text("a\nB-user\nc\n")
    _run(repo, "commit", "-qam", "user edit")
    return "task", baseline


class TestBeginConflictedMerge:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_baseline_branch_leaves_markers_and_reports_paths(self) -> None:
        branch, baseline = _conflicting_branch(self.repo, with_baseline=True)
        assert GitWorktreeOps.squash_merge_from_baseline(
            self.repo, branch, baseline or "",
        ) == MergeResult.CONFLICT

        conflicted = GitWorktreeOps.begin_conflicted_merge(self.repo, branch, baseline)

        assert conflicted == ["f.txt"]
        text = (self.repo / "f.txt").read_text()
        assert "<<<<<<< " in text and ">>>>>>> " in text
        assert "B-user" in text and "B-agent" in text
        # The non-conflicting hunk of the agent's net change is applied.
        assert "d-agent" in text
        # The baseline's dirty-state file is part of the agent's net diff
        # only when it changed after the baseline; it did not, so it
        # must not leak onto main.
        assert not (self.repo / "dirty.txt").exists()
        assert GitWorktreeOps.unresolved_paths(self.repo) == ["f.txt"]

    def test_legacy_branch_uses_merge_squash(self) -> None:
        branch, _ = _conflicting_branch(self.repo, with_baseline=False)
        assert GitWorktreeOps.squash_merge_branch(self.repo, branch) == MergeResult.CONFLICT

        conflicted = GitWorktreeOps.begin_conflicted_merge(self.repo, branch, None)

        assert conflicted == ["f.txt"]
        assert "<<<<<<< " in (self.repo / "f.txt").read_text()

    def test_clean_apply_returns_empty_list_and_stages(self) -> None:
        _run(self.repo, "checkout", "-qb", "task")
        (self.repo / "new.txt").write_text("new\n")
        _run(self.repo, "add", ".")
        _run(self.repo, "commit", "-qm", "agent adds file")
        _run(self.repo, "checkout", "-q", "main")

        assert GitWorktreeOps.begin_conflicted_merge(self.repo, "task", None) == []
        assert _git("diff", "--cached", "--name-only", cwd=self.repo).stdout.split() == [
            "new.txt",
        ]

    def test_unknown_branch_with_baseline_returns_none(self) -> None:
        before = GitWorktreeOps.status_porcelain(self.repo)
        head = _run(self.repo, "rev-parse", "HEAD").strip()

        assert GitWorktreeOps.begin_conflicted_merge(self.repo, "no-such", head) is None
        assert GitWorktreeOps.status_porcelain(self.repo) == before

    def test_unknown_branch_without_baseline_returns_none_and_restores(self) -> None:
        assert GitWorktreeOps.begin_conflicted_merge(self.repo, "no-such", None) is None
        assert GitWorktreeOps.status_porcelain(self.repo) == ""


class TestFinishConflictedMerge:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.branch, self.baseline = _conflicting_branch(self.repo, with_baseline=True)
        self.head_before = _run(self.repo, "rev-parse", "HEAD").strip()
        self.conflicted = GitWorktreeOps.begin_conflicted_merge(
            self.repo, self.branch, self.baseline,
        )
        assert self.conflicted == ["f.txt"]

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _finish(self) -> MergeResult:
        return GitWorktreeOps.finish_conflicted_merge(
            self.repo, self.branch, self.conflicted or [], self.head_before,
            user_prompt="do the thing", task_result="did the thing",
        )

    def test_unresolved_path_is_rejected(self) -> None:
        assert self._finish() == MergeResult.CONFLICT

    def test_staged_file_with_markers_is_rejected(self) -> None:
        _run(self.repo, "add", "f.txt")
        assert GitWorktreeOps.paths_with_conflict_markers(self.repo, ["f.txt"]) == ["f.txt"]
        assert self._finish() == MergeResult.CONFLICT

    def test_discarded_merge_is_rejected(self) -> None:
        # A resolver that threw the merge away: clean index, HEAD unmoved.
        _run(self.repo, "reset", "-q", "--hard", "HEAD")
        assert self._finish() == MergeResult.CONFLICT
        assert _run(self.repo, "rev-parse", "HEAD").strip() == self.head_before

    def test_resolved_and_staged_is_committed_with_task_metadata(self) -> None:
        (self.repo / "f.txt").write_text("a\nB-both\nc\nd-agent\n")
        _run(self.repo, "add", "f.txt")

        assert self._finish() == MergeResult.SUCCESS

        assert _run(self.repo, "rev-parse", "HEAD").strip() != self.head_before
        assert GitWorktreeOps.status_porcelain(self.repo) == ""
        message = _run(self.repo, "log", "-1", "--format=%B")
        assert "do the thing" in message and "did the thing" in message
        assert _run(self.repo, "show", "HEAD:f.txt") == "a\nB-both\nc\nd-agent\n"
        assert GitWorktreeOps.paths_with_conflict_markers(self.repo, ["f.txt"]) == []

    def test_resolver_that_committed_itself_is_refused_and_undone(self) -> None:
        # A rogue resolver: throws the merge away and commits something else.
        _run(self.repo, "reset", "-q", "--hard", "HEAD")
        (self.repo / "f.txt").write_text("unrelated\n")
        _run(self.repo, "commit", "-qam", "resolver commit")
        assert _run(self.repo, "rev-parse", "HEAD").strip() != self.head_before

        assert self._finish() == MergeResult.CONFLICT

        GitWorktreeOps.abort_conflicted_merge(self.repo, "", self.head_before)
        assert _run(self.repo, "rev-parse", "HEAD").strip() == self.head_before
        assert (self.repo / "f.txt").read_text() == "a\nB-user\nc\n"
        assert _run(self.repo, "show", f"{self.branch}:f.txt") == "a\nB-agent\nc\nd-agent\n"

    def test_markers_of_a_larger_configured_size_are_detected(self) -> None:
        GitWorktreeOps.abort_conflicted_merge(self.repo, "", self.head_before)
        (self.repo / ".gitattributes").write_text("f.txt conflict-marker-size=12\n")
        _run(self.repo, "add", ".gitattributes")
        _run(self.repo, "commit", "-qm", "wider markers")
        head = _run(self.repo, "rev-parse", "HEAD").strip()
        assert GitWorktreeOps.begin_conflicted_merge(
            self.repo, self.branch, self.baseline,
        ) == ["f.txt"]
        assert "<<<<<<<<<<<< " in (self.repo / "f.txt").read_text()
        _run(self.repo, "add", "f.txt")

        assert GitWorktreeOps.paths_with_conflict_markers(self.repo, ["f.txt"]) == ["f.txt"]
        assert GitWorktreeOps.finish_conflicted_merge(
            self.repo, self.branch, ["f.txt"], head,
        ) == MergeResult.CONFLICT

    def test_path_with_pathspec_magic_is_checked_literally(self) -> None:
        odd = ":(odd).txt"
        (self.repo / odd).write_text("<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> theirs\n")
        _run(self.repo, "add", "--", f":(literal){odd}")

        assert GitWorktreeOps.paths_with_conflict_markers(self.repo, [odd]) == [odd]
        assert GitWorktreeOps.paths_with_conflict_markers(self.repo, ["f.txt", odd]) == [odd]

    def test_git_failure_reports_every_path_as_unresolved(self) -> None:
        missing = Path(self.tmpdir) / "no-such-repo"
        assert GitWorktreeOps.paths_with_conflict_markers(missing, ["f.txt"]) == ["f.txt"]

    def test_deleted_file_resolution_is_accepted(self) -> None:
        _run(self.repo, "rm", "-q", "f.txt")
        assert self._finish() == MergeResult.SUCCESS
        assert not (self.repo / "f.txt").exists()

    def test_marker_check_with_no_paths_is_empty(self) -> None:
        assert GitWorktreeOps.paths_with_conflict_markers(self.repo, []) == []


class TestAbortConflictedMerge:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restores_clean_tree_and_removes_new_untracked_files(self) -> None:
        branch, baseline = _conflicting_branch(self.repo, with_baseline=True)
        head = _run(self.repo, "rev-parse", "HEAD").strip()
        assert GitWorktreeOps.begin_conflicted_merge(self.repo, branch, baseline)
        (self.repo / "scratch.txt").write_text("left behind by the resolver\n")

        GitWorktreeOps.abort_conflicted_merge(self.repo, "", head)

        assert GitWorktreeOps.status_porcelain(self.repo) == ""
        assert not (self.repo / "scratch.txt").exists()
        assert _run(self.repo, "rev-parse", "HEAD").strip() == head
        assert (self.repo / "f.txt").read_text() == "a\nB-user\nc\n"
        # The task branch is untouched.
        assert _run(self.repo, "show", f"{branch}:f.txt") == "a\nB-agent\nc\nd-agent\n"

    def test_keeps_untracked_files_of_a_dirty_tree(self) -> None:
        (self.repo / "mine.txt").write_text("user file\n")
        before = GitWorktreeOps.status_porcelain(self.repo)
        assert before

        GitWorktreeOps.abort_conflicted_merge(self.repo, before, "")

        assert (self.repo / "mine.txt").exists()
