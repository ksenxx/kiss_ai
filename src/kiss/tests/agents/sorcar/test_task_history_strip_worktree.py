# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test that the working directory saved to the task history table
has the ``.kiss-worktrees/kiss_wt-<slug>`` worktree suffix stripped.

BUG: When a task runs inside a git worktree, the agent's ``work_dir``
points at ``<repo>/.kiss-worktrees/kiss_wt-<slug>``.  This worktree
directory is ephemeral — it is removed once the worktree is merged or
discarded.  Persisting that path verbatim in ``task_history.extra``
means later history loads see a workspace path that no longer exists
on disk, breaking the history sidebar's "Workspace" filter (the row
appears to belong to a workspace the user never opened).

Fix: strip the ``.kiss-worktrees/kiss_wt-<slug>[/...]`` suffix before
persisting ``work_dir`` to ``task_history.extra``, leaving the parent
repository path (the user-visible workspace folder) in the database.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import strip_worktree_suffix
from kiss.agents.sorcar.persistence import _add_task, _save_task_extra


def _redirect(tmpdir: str) -> tuple:
    """Redirect the persistence DB to a temp dir. Mirrors test_persistence.py."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestStripWorktreeSuffix(unittest.TestCase):
    """Unit-level tests for the helper used by callers before saving."""

    def test_strips_worktree_root(self) -> None:
        wt = "/Users/alice/proj/.kiss-worktrees/kiss_wt-1782617911-0960938a"
        assert strip_worktree_suffix(wt) == "/Users/alice/proj"

    def test_strips_worktree_subdir(self) -> None:
        wt = "/Users/alice/proj/.kiss-worktrees/kiss_wt-abc-deadbeef/src/foo"
        assert strip_worktree_suffix(wt) == "/Users/alice/proj"

    def test_passthrough_when_not_in_worktree(self) -> None:
        assert strip_worktree_suffix("/Users/alice/proj") == "/Users/alice/proj"

    def test_passthrough_empty_string(self) -> None:
        assert strip_worktree_suffix("") == ""

    def test_strips_at_repo_root(self) -> None:
        wt = "/repo/.kiss-worktrees/kiss_wt-x-12345678"
        assert strip_worktree_suffix(wt) == "/repo"

    def test_trailing_slash_not_kept_on_parent(self) -> None:
        wt = "/Users/alice/proj/.kiss-worktrees/kiss_wt-1-2/"
        assert strip_worktree_suffix(wt) == "/Users/alice/proj"

    def test_only_strips_kiss_worktrees_segment(self) -> None:
        # A path that *contains* the literal string ".kiss-worktrees"
        # only mid-segment should not be touched.
        p = "/Users/alice/some.kiss-worktrees-backup/data"
        assert strip_worktree_suffix(p) == p

    def test_requires_kiss_wt_prefix(self) -> None:
        # The directory under ``.kiss-worktrees/`` must start with
        # ``kiss_wt-`` (the framework's slug format).  Anything else is
        # not a KISS worktree and must be left alone.
        p = "/Users/alice/proj/.kiss-worktrees/something-else/file.txt"
        assert strip_worktree_suffix(p) == p

    def test_root_absolute_worktree(self) -> None:
        """A worktree at the filesystem root must yield ``/``."""
        assert strip_worktree_suffix("/.kiss-worktrees/kiss_wt-x") == "/"

    def test_relative_worktree_path(self) -> None:
        """A relative worktree path must yield ``.`` (current dir)."""
        assert strip_worktree_suffix(".kiss-worktrees/kiss_wt-x") == "."

    def test_windows_backslash_path(self) -> None:
        """Windows-style backslashes must be folded so the suffix is
        still recognised."""
        p = r"C:\Users\alice\proj\.kiss-worktrees\kiss_wt-x-1234"
        assert strip_worktree_suffix(p) == "C:/Users/alice/proj"


class TestChatSorcarAgentBuildExtraStripsWorktree(unittest.TestCase):
    """``ChatSorcarAgent._build_extra_payload`` must produce a payload
    whose ``work_dir`` is the parent repo, not the worktree path."""

    def test_build_extra_payload_strips_worktree(self) -> None:
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        agent = ChatSorcarAgent("Sorcar VS Code")
        wt = "/Users/alice/proj/.kiss-worktrees/kiss_wt-abc-12345678"
        payload = agent._build_extra_payload(
            model="claude-opus-4-7",
            work_dir=wt,
            is_parallel=False,
            is_worktree=True,
        )
        assert payload["work_dir"] == "/Users/alice/proj", (
            f"_build_extra_payload kept worktree suffix: {payload['work_dir']!r}"
        )

    def test_build_extra_payload_passthrough_non_worktree(self) -> None:
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        agent = ChatSorcarAgent("Sorcar VS Code")
        payload = agent._build_extra_payload(
            model="claude-opus-4-7",
            work_dir="/Users/alice/proj",
            is_parallel=False,
            is_worktree=False,
        )
        assert payload["work_dir"] == "/Users/alice/proj"


class TestSaveTaskExtraEndToEnd(unittest.TestCase):
    """End-to-end: a payload whose ``work_dir`` is a worktree path,
    persisted via ``_save_task_extra``, must round-trip with the
    parent repo path stored in ``task_history.extra``."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def tearDown(self) -> None:
        if th._db_conn is not None:
            try:
                th._db_conn.close()
            except Exception:
                pass
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_extra(self, task_id: int) -> dict:
        db = th._get_db()
        row = db.execute(
            "SELECT extra FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        assert row is not None
        result: dict = json.loads(row["extra"])
        return result

    def test_chat_sorcar_agent_persists_stripped_work_dir(self) -> None:
        """End-to-end check via the agent's own payload builder."""
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        agent = ChatSorcarAgent("Sorcar VS Code")
        wt = "/Users/alice/proj/.kiss-worktrees/kiss_wt-abc-12345678"
        payload = agent._build_extra_payload(
            model="claude-opus-4-7",
            work_dir=wt,
            is_parallel=False,
            is_worktree=True,
        )
        task_id, _chat_id = _add_task("repro task", "")
        _save_task_extra(payload, task_id=task_id)

        stored = self._read_extra(task_id)
        assert stored["work_dir"] == "/Users/alice/proj", (
            f"work_dir was stored verbatim with worktree suffix: "
            f"{stored['work_dir']!r}"
        )
        assert stored["model"] == "claude-opus-4-7"
        assert stored["is_worktree"] is True

    def test_task_runner_payload_persists_stripped_work_dir(self) -> None:
        """Mirror the literal payload built in
        ``kiss.agents.vscode.task_runner._run_task_inner`` and assert
        that it persists the parent repo path (the fix must apply at
        the task_runner call site too)."""
        from kiss.agents.vscode.task_runner import build_task_extra_payload

        task_id, _chat_id = _add_task("runner task", "")
        wt = "/repo/.kiss-worktrees/kiss_wt-XYZ-87654321"
        payload = build_task_extra_payload(
            model="claude-opus-4-7",
            work_dir=wt,
            version="test",
            tokens=10,
            cost=0.01,
            steps=3,
            is_parallel=False,
            is_worktree=True,
            auto_commit_mode=False,
            start_ms=1,
            end_ms=2,
        )
        _save_task_extra(payload, task_id=task_id)

        stored = self._read_extra(task_id)
        assert stored["work_dir"] == "/repo", (
            f"task_runner persisted raw worktree path: {stored['work_dir']!r}"
        )
        assert stored["is_worktree"] is True
        assert stored["model"] == "claude-opus-4-7"

    def test_plain_path_passthrough_end_to_end(self) -> None:
        """A non-worktree ``work_dir`` must persist unchanged."""
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        agent = ChatSorcarAgent("Sorcar VS Code")
        plain = "/Users/alice/proj"
        payload = agent._build_extra_payload(
            model="claude-opus-4-7",
            work_dir=plain,
            is_parallel=False,
            is_worktree=False,
        )
        task_id, _chat_id = _add_task("plain task", "")
        _save_task_extra(payload, task_id=task_id)

        stored = self._read_extra(task_id)
        assert stored["work_dir"] == plain


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
