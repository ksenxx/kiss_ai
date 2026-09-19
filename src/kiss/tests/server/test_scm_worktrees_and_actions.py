# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_explorer_scm_commands and is intentionally
#   shadowed by test parameters of the same name)
"""End-to-end tests for worktree-aware Source Control and the sidebar actions.

The remote webapp's Source Control view must show the changes of EVERY
worktree of the repository (``gitStatus.worktrees``) and a graph that
starts from every worktree's HEAD (``gitLog``).  Its commit context menu
drives ``gitShow`` (Open Changes / Open File / Compare with...) and
``gitAction`` (Checkout (Detached) / Create Branch... / Create Tag... /
Cherry Pick); the Explorer's context menu drives ``fsAction``; a
clicked PDF comes back from ``openFile`` as bytes, not as an error.
Every test talks to a REAL :class:`RemoteAccessServer` over ``wss://``
against a REAL git repository with a REAL linked worktree — no mocks.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.server.explorer import git_worktrees
from kiss.server.fs_actions import FIND_MAX_MATCHES, fs_action
from kiss.tests.conftest import IS_WINDOWS, is_root, posix_only, requires_unix_sockets
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    _git,
    _request,
    harness,  # noqa: F401  (module fixture used by param name)
)

_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] >> endobj\n"
    b"trailer << /Root 1 0 R >>\n%%EOF\n\0"
)

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture(scope="module")
def worktree(harness: ExplorerHarness) -> Path:
    """A linked worktree of the harness repo with one own commit and a
    dirty file, beside the repository (so the Explorer of the main
    checkout never lists it)."""
    wt = Path(harness.tmpdir) / "wt-task"
    # The daemon's git runs with a scrubbed environment; a real user's
    # repository carries an identity, which cherry-pick and annotated
    # tags need to create their commit / tag objects.
    _git(harness.work_dir, "config", "user.name", "Test Author")
    _git(harness.work_dir, "config", "user.email", "author@example.com")
    _git(harness.work_dir, "worktree", "add", "-q", "-b", "kiss/wt-task", str(wt))
    (wt / "task.txt").write_text("worktree-commit-sentinel-91af\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "worktree: add task.txt")
    (wt / "README.md").write_text("# readme edited in the worktree\n")
    (wt / "wt-untracked.txt").write_text("worktree-untracked-sentinel-5e2d\n")
    (harness.work_dir / "report.pdf").write_bytes(_PDF_BYTES)
    (harness.work_dir / "dot.png").write_bytes(_PNG_BYTES)
    return wt


class TestWorktreeStatusAndLog:
    def test_status_lists_every_worktree_with_its_own_changes(
        self, harness, worktree,
    ) -> None:
        reply = _request(
            harness, {"type": "gitStatus", "tabId": "t", "token": "1"},
            "gitStatus",
        )
        assert "error" not in reply
        wts = reply["worktrees"]
        assert [w["name"] for w in wts] == ["repo", "wt-task"]
        main, other = wts
        assert main["current"] is True and main["branch"] == "main"
        assert other["current"] is False
        assert other["branch"] == "kiss/wt-task"
        assert Path(other["path"]) == worktree
        # The main checkout's rows are the same list `changes` carries.
        assert main["changes"] == reply["changes"]
        assert {c["path"] for c in main["changes"]} >= {
            "README.md", "dir/nested.py", "untracked.txt", "b.txt",
        }
        other_paths = {(c["path"], c["status"]) for c in other["changes"]}
        assert ("README.md", "M") in other_paths
        assert ("wt-untracked.txt", "U") in other_paths
        # absPath points into the worktree, so a click opens THAT copy.
        readme = next(c for c in other["changes"] if c["path"] == "README.md")
        assert Path(readme["absPath"]) == worktree / "README.md"

    def test_status_from_inside_the_worktree_marks_it_current(
        self, harness, worktree,
    ) -> None:
        reply = _request(
            harness,
            {"type": "gitStatus", "workDir": str(worktree), "token": "2"},
            "gitStatus",
        )
        assert reply["branch"] == "kiss/wt-task"
        current = [w for w in reply["worktrees"] if w["current"]]
        assert len(current) == 1 and current[0]["name"] == "wt-task"
        assert reply["worktrees"][0]["name"] == "repo"  # main first

    def test_log_starts_from_every_worktree_head(self, harness, worktree) -> None:
        reply = _request(
            harness, {"type": "gitLog", "token": "3", "limit": 50}, "gitLog",
        )
        assert "error" not in reply
        subjects = [c["subject"] for c in reply["commits"]]
        assert "worktree: add task.txt" in subjects
        wt_commit = next(
            c for c in reply["commits"] if c["subject"] == "worktree: add task.txt"
        )
        assert "kiss/wt-task" in wt_commit["refs"]
        assert wt_commit["files"] == [{"path": "task.txt", "status": "A"}]
        heads = {w["name"]: w["head"] for w in reply["worktrees"]}
        assert heads["wt-task"] == wt_commit["sha"]
        assert heads["repo"] == reply["head"] == harness.shas["merge"]
        # Date order still holds: the merge (HEAD of main) and the
        # worktree commit are both newer than everything they descend from.
        idx = {c["sha"]: i for i, c in enumerate(reply["commits"])}
        for c in reply["commits"]:
            for parent in c["parents"]:
                assert idx[parent] > idx[c["sha"]]

    def test_worktree_list_skips_vanished_worktree(self, harness, worktree) -> None:
        gone = Path(harness.tmpdir) / "wt-gone"
        _git(harness.work_dir, "worktree", "add", "-q", "--detach", str(gone))
        shutil.rmtree(gone)
        try:
            names = [w["name"] for w in git_worktrees(str(harness.work_dir))]
            assert "wt-gone" not in names
            assert names[:2] == ["repo", "wt-task"]
        finally:
            _git(harness.work_dir, "worktree", "prune")

    def test_worktree_list_outside_a_repo_is_empty(self, harness) -> None:
        # Not a repository: git fails and the fallback names the folder
        # itself as the (only) current worktree.
        rows = git_worktrees(str(harness.plain_dir))
        assert len(rows) == 1 and rows[0]["current"] is True
        assert rows[0]["path"] == str(harness.plain_dir)


class TestGitShow:
    def test_open_changes_returns_the_commit_patch(self, harness, worktree) -> None:
        reply = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["second"], "token": "s1"},
            "gitShow",
        )
        assert "error" not in reply
        assert reply["subject"] == "second: rename a.txt -> b.txt"
        assert reply["mode"] == "patch"
        assert "diff --git" in reply["text"]
        assert "nested-sentinel-4f2a" in reply["text"]
        assert reply["truncated"] is False

    def test_open_changes_of_one_file_restricts_the_patch(self, harness) -> None:
        reply = _request(
            harness,
            {
                "type": "gitShow", "sha": harness.shas["second"],
                "path": "dir/nested.py", "token": "s2",
            },
            "gitShow",
        )
        assert "dir/nested.py" in reply["text"]
        diff_part = reply["text"][reply["text"].index("diff --git"):]
        assert "b.txt" not in diff_part
        assert "1 file changed" in reply["text"]

    def test_merge_commit_patch_is_against_first_parent(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["merge"], "token": "s3"},
            "gitShow",
        )
        assert "feature.txt" in reply["text"]

    def test_open_file_returns_the_file_at_that_revision(self, harness) -> None:
        reply = _request(
            harness,
            {
                "type": "gitShow", "sha": harness.shas["second"],
                "path": "dir/nested.py", "mode": "file", "token": "s4",
            },
            "gitShow",
        )
        assert "error" not in reply
        assert reply["text"] == "x = 1  # nested-sentinel-4f2a\n"

    def test_open_file_of_a_binary_blob_is_refused(self, harness, worktree) -> None:
        _git(harness.work_dir, "add", "dot.png")
        _git(harness.work_dir, "commit", "-q", "-m", "add png")
        sha = _git(harness.work_dir, "rev-parse", "HEAD").strip()
        try:
            reply = _request(
                harness,
                {"type": "gitShow", "sha": sha, "path": "dot.png",
                 "mode": "file", "token": "s5"},
                "gitShow",
            )
            assert "binary" in reply["error"]
        finally:
            _git(harness.work_dir, "reset", "-q", "--soft", "HEAD~1")
            _git(harness.work_dir, "reset", "-q", "dot.png")

    def test_compare_with_a_revision(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["merge"],
             "base": "v1~2", "token": "s6"},
            "gitShow",
        )
        assert "error" not in reply
        assert reply["base"] == "v1~2"
        assert "feature.txt" in reply["text"]

    def test_compare_with_unknown_revision_is_an_error(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["merge"],
             "base": "no-such-branch", "token": "s7"},
            "gitShow",
        )
        assert reply["error"] == "Unknown revision: no-such-branch"
        bad = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["merge"],
             "base": "--output=/tmp/x", "token": "s8"},
            "gitShow",
        )
        assert bad["error"].startswith("Not a revision")

    def test_bad_sha_and_non_repo_are_errors(self, harness) -> None:
        reply = _request(
            harness, {"type": "gitShow", "sha": "zzz", "token": "s9"}, "gitShow",
        )
        assert reply["error"] == "Not a commit id: zzz"
        plain = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["first"],
             "workDir": str(harness.plain_dir), "token": "s10"},
            "gitShow",
        )
        assert plain["error"].startswith("Not a git repository")
        missing = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["first"],
             "workDir": str(harness.plain_dir / "nope"), "token": "s11"},
            "gitShow",
        )
        assert missing["error"].startswith("Directory not found")
        unknown = _request(
            harness,
            {"type": "gitShow", "sha": "f" * 40, "token": "s12"}, "gitShow",
        )
        assert "error" in unknown


class TestGitAction:
    def test_create_branch_checks_it_out_in_the_worktree(
        self, harness, worktree,
    ) -> None:
        reply = _request(
            harness,
            {"type": "gitAction", "action": "createBranch",
             "sha": harness.shas["second"], "name": "from-second",
             "workDir": str(worktree), "token": "a1"},
            "gitActionResult",
        )
        assert reply.get("ok") is True, reply
        assert _git(worktree, "symbolic-ref", "--short", "HEAD").strip() == "from-second"
        assert _git(worktree, "rev-parse", "HEAD").strip() == harness.shas["second"]
        # The dirty README survived (checkout carries it over), and the
        # main checkout is untouched.
        assert (worktree / "wt-untracked.txt").exists()
        assert _git(harness.work_dir, "symbolic-ref", "--short", "HEAD").strip() == "main"
        # Back to the task branch for the tests below.
        _git(worktree, "checkout", "-q", "kiss/wt-task")

    def test_create_branch_with_existing_or_bad_name_fails(
        self, harness, worktree,
    ) -> None:
        dup = _request(
            harness,
            {"type": "gitAction", "action": "createBranch",
             "sha": harness.shas["second"], "name": "main",
             "workDir": str(worktree), "token": "a2"},
            "gitActionResult",
        )
        assert "already exists" in dup["error"]
        bad = _request(
            harness,
            {"type": "gitAction", "action": "createBranch",
             "sha": harness.shas["second"], "name": "bad name",
             "workDir": str(worktree), "token": "a3"},
            "gitActionResult",
        )
        assert bad["error"].startswith("Not a valid name")
        empty = _request(
            harness,
            {"type": "gitAction", "action": "createTag",
             "sha": harness.shas["second"], "name": "",
             "workDir": str(worktree), "token": "a4"},
            "gitActionResult",
        )
        assert empty["error"].startswith("Not a valid name")

    def test_create_tag_plain_and_annotated(self, harness, worktree) -> None:
        plain = _request(
            harness,
            {"type": "gitAction", "action": "createTag",
             "sha": harness.shas["first"], "name": "t-plain", "token": "a5"},
            "gitActionResult",
        )
        assert plain.get("ok") is True, plain
        assert _git(harness.work_dir, "rev-parse", "t-plain").strip() == harness.shas["first"]
        annotated = _request(
            harness,
            {"type": "gitAction", "action": "createTag",
             "sha": harness.shas["first"], "name": "t-note",
             "message": "release note", "token": "a6"},
            "gitActionResult",
        )
        assert annotated.get("ok") is True, annotated
        assert _git(harness.work_dir, "cat-file", "-t", "t-note").strip() == "tag"
        assert "release note" in _git(harness.work_dir, "tag", "-n1", "t-note")

    def test_checkout_detached_and_cherry_pick(self, harness, worktree) -> None:
        # A second, clean worktree so neither the dirty main checkout
        # nor the task worktree is disturbed.
        clean = Path(harness.tmpdir) / "wt-clean"
        _git(harness.work_dir, "worktree", "add", "-q", "--detach", str(clean),
             harness.shas["second"])
        try:
            detached = _request(
                harness,
                {"type": "gitAction", "action": "checkoutDetached",
                 "sha": harness.shas["first"], "workDir": str(clean),
                 "token": "a7"},
                "gitActionResult",
            )
            assert detached.get("ok") is True, detached
            assert _git(clean, "rev-parse", "HEAD").strip() == harness.shas["first"]
            assert not (clean / "dir").exists()
            picked = _request(
                harness,
                {"type": "gitAction", "action": "cherryPick",
                 "sha": harness.shas["feat"], "workDir": str(clean),
                 "token": "a8"},
                "gitActionResult",
            )
            assert picked.get("ok") is True, picked
            assert (clean / "feature.txt").read_text() == "feature-file-sentinel-7c1e\n"
            assert _git(clean, "log", "-1", "--format=%s").strip() == "feature: add feature.txt"
            # Cherry-picking a merge takes its first-parent change.
            merged = _request(
                harness,
                {"type": "gitAction", "action": "cherryPick",
                 "sha": harness.shas["merge"], "workDir": str(clean),
                 "token": "a9"},
                "gitActionResult",
            )
            # The merge's first-parent diff re-adds feature.txt, which
            # already exists identically: git reports nothing to pick
            # (an error) or an empty pick; either way the tree is intact.
            assert "type" in merged
            assert (clean / "feature.txt").exists()
        finally:
            _git(harness.work_dir, "worktree", "remove", "--force", str(clean))

    def test_checkout_detached_refuses_to_clobber_dirty_files(
        self, harness, worktree,
    ) -> None:
        # The task worktree's README.md is modified; checking out the
        # first commit (which changes README? no: it has the same
        # README) succeeds, so use dir/nested.py edits in the main
        # checkout, where the staged change collides.
        reply = _request(
            harness,
            {"type": "gitAction", "action": "checkoutDetached",
             "sha": harness.shas["first"], "token": "a10"},
            "gitActionResult",
        )
        assert "error" in reply
        assert _git(harness.work_dir, "symbolic-ref", "--short", "HEAD").strip() == "main"

    def test_unknown_action_and_missing_dir(self, harness) -> None:
        unknown = _request(
            harness,
            {"type": "gitAction", "action": "rebase",
             "sha": harness.shas["first"], "token": "a11"},
            "gitActionResult",
        )
        assert unknown["error"] == "Unknown git action: rebase"
        missing = _request(
            harness,
            {"type": "gitAction", "action": "cherryPick",
             "sha": harness.shas["first"],
             "workDir": str(harness.plain_dir / "nope"), "token": "a12"},
            "gitActionResult",
        )
        assert missing["error"].startswith("Directory not found")
        plain = _request(
            harness,
            {"type": "gitAction", "action": "cherryPick",
             "sha": harness.shas["first"],
             "workDir": str(harness.plain_dir), "token": "a13"},
            "gitActionResult",
        )
        assert plain["error"].startswith("Not a git repository")


def _fs(harness: ExplorerHarness, **cmd) -> dict:
    # The client sends the Explorer root as ``workDir``; the daemon
    # refuses paths outside it.  Default to the temp dir holding both
    # the repo and the plain folder so tests can act on either.
    payload = {
        "type": "fsAction", "tabId": "t", "token": "f",
        "workDir": harness.tmpdir,
    }
    payload.update(cmd)
    return _request(harness, payload, "fsResult")


class TestFsAction:
    def test_new_file_new_folder_rename_delete(self, harness) -> None:
        root = harness.plain_dir
        created = _fs(harness, action="newFile", path=str(root), name="notes/a.txt")
        assert created.get("ok") is True, created
        assert Path(created["path"]) == root / "notes" / "a.txt"
        assert (root / "notes" / "a.txt").is_file()
        folder = _fs(harness, action="newFolder", path=str(root), name="sub")
        assert folder.get("ok") is True and (root / "sub").is_dir()
        dup = _fs(harness, action="newFolder", path=str(root), name="sub")
        assert dup["exists"] is True and "already exists" in dup["error"]
        bad = _fs(harness, action="newFile", path=str(root), name="../x")
        assert bad["error"].startswith("Not a valid name")
        renamed = _fs(
            harness, action="rename", path=str(root / "notes" / "a.txt"),
            dest=str(root / "notes" / "b.txt"),
        )
        assert renamed.get("ok") is True, renamed
        assert (root / "notes" / "b.txt").is_file()
        assert not (root / "notes" / "a.txt").exists()
        # A relative rename target resolves against the command's workDir.
        rel = _fs(
            harness, action="rename", path=str(root / "notes" / "b.txt"),
            dest="notes/c.txt", workDir=str(root),
        )
        assert rel.get("ok") is True, rel
        assert (root / "notes" / "c.txt").is_file()
        # Rename stays in the folder, like VS Code's: a name with a
        # path separator (the client sends ``parent/<typed name>``) is
        # not a valid file name.
        (root / "sub" / "taken.txt").write_text("taken\n")
        across = _fs(
            harness, action="rename", path=str(root / "notes" / "c.txt"),
            dest=str(root / "sub" / "taken.txt"),
        )
        assert across["error"].startswith("Not a valid name")
        assert (root / "notes" / "c.txt").is_file()
        (root / "notes" / "taken.txt").write_text("taken\n")
        clash = _fs(
            harness, action="rename", path=str(root / "notes" / "c.txt"),
            dest=str(root / "notes" / "taken.txt"),
        )
        assert clash["exists"] is True
        forced = _fs(
            harness, action="rename", path=str(root / "notes" / "c.txt"),
            dest=str(root / "notes" / "taken.txt"), overwrite=True,
        )
        assert forced.get("ok") is True, forced
        assert (root / "notes" / "taken.txt").read_text() == ""
        assert not (root / "notes" / "c.txt").exists()
        # The overwrite left no backup behind.
        assert sorted(p.name for p in (root / "notes").iterdir()) == ["taken.txt"]
        deleted = _fs(harness, action="delete", path=str(root / "notes"))
        assert deleted.get("ok") is True and not (root / "notes").exists()
        gone = _fs(harness, action="delete", path=str(root / "notes"))
        assert gone["error"].startswith("Not found")

    def test_copy_and_move(self, harness) -> None:
        root = harness.plain_dir
        (root / "src.txt").write_text("copy me\n")
        (root / "dest").mkdir(exist_ok=True)
        same = _fs(harness, action="copy", path=str(root / "src.txt"), dest=str(root))
        assert same.get("ok") is True, same
        assert Path(same["path"]) == root / "src copy.txt"
        again = _fs(harness, action="copy", path=str(root / "src.txt"), dest=str(root))
        assert Path(again["path"]) == root / "src copy 2.txt"
        into = _fs(
            harness, action="copy", path=str(root / "src.txt"),
            dest=str(root / "dest"),
        )
        assert into.get("ok") is True and (root / "dest" / "src.txt").read_text() == "copy me\n"
        clash = _fs(
            harness, action="copy", path=str(root / "src.txt"),
            dest=str(root / "dest"),
        )
        assert clash["exists"] is True
        (root / "src.txt").write_text("changed\n")
        forced = _fs(
            harness, action="copy", path=str(root / "src.txt"),
            dest=str(root / "dest"), overwrite=True,
        )
        assert forced.get("ok") is True
        assert (root / "dest" / "src.txt").read_text() == "changed\n"
        moved = _fs(
            harness, action="move", path=str(root / "src copy.txt"),
            dest=str(root / "dest"),
        )
        assert moved.get("ok") is True
        assert (root / "dest" / "src copy.txt").exists()
        assert not (root / "src copy.txt").exists()
        noop = _fs(harness, action="move", path=str(root / "src.txt"), dest=str(root))
        assert noop.get("ok") is True and (root / "src.txt").exists()
        inside = _fs(
            harness, action="move", path=str(root / "dest"),
            dest=str(root / "dest"),
        )
        assert "into itself" in inside["error"]
        # Folders copy recursively.
        tree = _fs(harness, action="copy", path=str(root / "dest"), dest=str(root))
        assert (Path(tree["path"]) / "src.txt").exists()
        missing_dest = _fs(
            harness, action="copy", path=str(root / "src.txt"),
            dest=str(root / "nowhere"),
        )
        assert missing_dest["error"].startswith("Not found")

    def test_find_in_folder_and_compare(self, harness) -> None:
        root = harness.plain_dir
        (root / "hay.txt").write_text("needle here\nno\nneedle again\n")
        (root / ".git").mkdir(exist_ok=True)
        (root / ".git" / "skip.txt").write_text("needle in git dir\n")
        found = _fs(harness, action="findInFolder", path=str(root), query="needle")
        assert found.get("ok") is True, found
        assert found["count"] == 2
        assert "hay.txt:1:needle here" in found["text"]
        assert "hay.txt:3:needle again" in found["text"]
        assert ".git" not in found["text"]
        nothing = _fs(harness, action="findInFolder", path=str(root), query="zzzqqq")
        assert nothing.get("ok") is True and nothing["count"] == 0
        empty = _fs(harness, action="findInFolder", path=str(root), query="")
        assert empty["error"] == "Nothing to search for"
        (root / "one.txt").write_text("a\nb\n")
        (root / "two.txt").write_text("a\nc\n")
        diff = _fs(
            harness, action="compare", path=str(root / "one.txt"),
            dest=str(root / "two.txt"),
        )
        assert diff.get("ok") is True, diff
        assert "-b" in diff["text"] and "+c" in diff["text"]
        same = _fs(
            harness, action="compare", path=str(root / "one.txt"),
            dest=str(root / "one.txt"),
        )
        assert "identical" in same["text"]
        folder = fs_action("compare", str(root), str(root / "one.txt"))
        assert folder["error"].startswith("Not a file")

    def test_find_truncates_huge_result_sets(self, harness) -> None:
        big = harness.plain_dir / "big.txt"
        big.write_text("hit\n" * (FIND_MAX_MATCHES + 5))
        found = fs_action("findInFolder", str(harness.plain_dir), query="hit")
        assert found["truncated"] is True
        assert found["count"] == FIND_MAX_MATCHES + 5
        assert found["text"].count("\n") == FIND_MAX_MATCHES
        big.unlink()

    def test_errors_and_guards(self, harness) -> None:
        unknown = _fs(harness, action="chmod", path=str(harness.plain_dir))
        assert unknown["error"] == "Unknown file action: chmod"
        nopath = _fs(harness, action="delete", path="")
        # An empty path fails the catalog's required-field check and
        # gets no fsResult; the handler itself also refuses it.
        assert nopath is not None
        missing = _fs(harness, action="delete", path=str(harness.plain_dir / "nope"))
        assert missing["error"].startswith("Not found")
        assert fs_action("delete", "/")["error"].startswith("Refusing")
        assert fs_action("newFile", str(harness.plain_dir / "only.txt"), name="x")[
            "error"
        ].startswith("Not a folder")
        assert fs_action("findInFolder", str(harness.plain_dir / "only.txt"), query="x")[
            "error"
        ].startswith("Not a folder")
        assert fs_action("rename", str(harness.plain_dir / "only.txt"), dest="")[
            "error"
        ].startswith("Not a valid name")
        assert fs_action("copy", str(harness.plain_dir / "only.txt"), dest="/nowhere")[
            "error"
        ].startswith("Not a folder")
        assert fs_action("rename", str(harness.plain_dir / "nope"), dest="/x")[
            "error"
        ].startswith("Not found")
        assert fs_action("copy", str(harness.plain_dir / "nope"), dest=str(harness.plain_dir))[
            "error"
        ].startswith("Not found")
        assert fs_action("frobnicate", str(harness.plain_dir))["error"].startswith(
            "Unknown file action"
        )
        # An OSError inside an action becomes an error reply.
        ro = harness.plain_dir / "ro"
        ro.mkdir(exist_ok=True)
        os.chmod(ro, 0o500)
        try:
            if not IS_WINDOWS and not is_root():
                # chmod bits are advisory on Windows: no denial to test.
                denied = fs_action("newFile", str(ro), name="x.txt")
                assert denied["error"].startswith("newFile failed")
        finally:
            os.chmod(ro, 0o700)


class TestOpenBinaryFiles:
    def test_pdf_comes_back_as_bytes_not_an_error(self, harness, worktree) -> None:
        reply = _request(
            harness, {"type": "openFile", "path": "report.pdf", "tabId": "t"},
            "fileContent",
        )
        assert "error" not in reply, reply
        assert reply["binary"] is True
        assert reply["mime"] == "application/pdf"
        assert reply["size"] == len(_PDF_BYTES)
        assert base64.b64decode(reply["base64"]) == _PDF_BYTES
        assert reply["name"] == "report.pdf"
        assert "content" not in reply and "version" not in reply

    def test_image_comes_back_as_bytes(self, harness, worktree) -> None:
        reply = _request(
            harness, {"type": "openFile", "path": "dot.png", "tabId": "t"},
            "fileContent",
        )
        assert reply["binary"] is True and reply["mime"] == "image/png"
        assert base64.b64decode(reply["base64"]) == _PNG_BYTES

    def test_other_binaries_are_still_refused(self, harness, worktree) -> None:
        (harness.work_dir / "blob.bin").write_bytes(b"\0\1\2\3")
        try:
            reply = _request(
                harness, {"type": "openFile", "path": "blob.bin"}, "fileContent",
            )
            assert reply["error"].startswith("Cannot display binary file")
        finally:
            (harness.work_dir / "blob.bin").unlink()

    def test_oversized_pdf_is_refused(self, harness, worktree) -> None:
        from kiss.server import web_server

        big = harness.work_dir / "big.pdf"
        with open(big, "wb") as f:
            f.truncate(web_server._OPEN_BINARY_MAX_BYTES + 1)
        try:
            reply = _request(
                harness, {"type": "openFile", "path": "big.pdf"}, "fileContent",
            )
            assert reply["error"].startswith("File too large")
        finally:
            big.unlink()

    def test_background_flag_is_echoed(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "openFile", "path": "README.md", "background": True},
            "fileContent",
        )
        assert reply["background"] is True
        plain = _request(
            harness, {"type": "openFile", "path": "README.md"}, "fileContent",
        )
        assert "background" not in plain


@requires_unix_sockets
class TestUdsDrop:
    """VS Code windows (UDS peers) never get the remote-only replies."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "gitShow", "sha": "0" * 40},
            {"type": "gitAction", "action": "cherryPick", "sha": "0" * 40},
            {"type": "fsAction", "action": "delete", "path": "/nope"},
        ],
    )
    def test_uds_delivered_commands_produce_no_reply(self, harness, payload) -> None:
        import asyncio

        async def _over_uds() -> list[dict]:
            reader, writer = await asyncio.open_unix_connection(str(harness.uds_path))
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
            # A control command that always answers, to bound the wait.
            writer.write((json.dumps({"type": "getTabsState"}) + "\n").encode())
            await writer.drain()
            got: list[dict] = []
            while True:
                line = await asyncio.wait_for(reader.readline(), 30)
                if not line:
                    break
                msg = json.loads(line)
                got.append(msg)
                if msg.get("type") == "tabs_state":
                    break
            writer.close()
            return got

        got = harness.run(_over_uds())
        assert all(
            m.get("type") not in ("gitShow", "gitActionResult", "fsResult")
            for m in got
        ), got


def test_git_available() -> None:
    """The suite assumes a git binary (every action shells out to it)."""
    assert shutil.which("git")
    subprocess.run(["git", "--version"], check=True, capture_output=True)


@contextlib.contextmanager
def _scratch_worktree(harness: ExplorerHarness) -> Iterator[Path]:
    """A throw-away detached worktree of the harness repo.

    Commits made in it become part of the repository (and its head is
    one of the graph's starts) without disturbing the dirty files the
    module's other tests rely on in the main checkout / task worktree.
    """
    wt = Path(harness.tmpdir) / f"wt-scratch-{uuid.uuid4().hex[:8]}"
    _git(harness.work_dir, "worktree", "add", "-q", "--detach", str(wt))
    try:
        yield wt
    finally:
        _git(harness.work_dir, "worktree", "remove", "--force", str(wt))
        _git(harness.work_dir, "worktree", "prune")


class TestReviewRegressions:
    """Defects a read-only review found in the first implementation.

    Each test reproduces one of them end to end against the real
    daemon / git and must keep failing if the fix is undone.
    """

    def test_delete_and_rename_act_on_the_symlink_not_its_target(
        self, harness,
    ) -> None:
        root = harness.plain_dir
        real = root / "real"
        real.mkdir()
        (real / "keep.txt").write_text("keep\n")
        link = root / "link"
        link.symlink_to(real)
        file_link = root / "file-link.txt"
        file_link.symlink_to(root / "only.txt")
        renamed = _fs(
            harness, action="rename", path=str(file_link),
            dest=str(root / "moved-link.txt"),
        )
        assert renamed.get("ok") is True, renamed
        assert (root / "moved-link.txt").is_symlink()
        assert (root / "only.txt").read_text() == "only\n"
        deleted = _fs(harness, action="delete", path=str(link))
        assert deleted.get("ok") is True, deleted
        assert not link.is_symlink() and not link.exists()
        assert (real / "keep.txt").read_text() == "keep\n", "target must survive"
        # A dangling symlink is an entry too: it can be deleted.
        dangling = root / "dangling"
        dangling.symlink_to(root / "nowhere")
        gone = _fs(harness, action="delete", path=str(dangling))
        assert gone.get("ok") is True, gone
        assert not os.path.lexists(dangling)
        (root / "moved-link.txt").unlink()
        shutil.rmtree(real)

    def test_actions_are_confined_to_the_explorer_root(self, harness) -> None:
        outside = Path(harness.tmpdir) / "outside.txt"
        outside.write_text("outside\n")
        try:
            denied = _fs(
                harness, action="delete", path=str(outside),
                workDir=str(harness.plain_dir),
            )
            assert denied["error"].startswith("Not inside the workspace")
            assert outside.exists()
            escaped = _fs(
                harness, action="rename", path=str(harness.plain_dir / "only.txt"),
                dest=str(harness.plain_dir / ".." / "only.txt"),
                workDir=str(harness.plain_dir),
            )
            assert escaped["error"].startswith("Not inside the workspace")
            moved_out = _fs(
                harness, action="move", path=str(harness.plain_dir / "only.txt"),
                dest=harness.tmpdir, workDir=str(harness.plain_dir),
            )
            assert moved_out["error"].startswith("Not inside the workspace")
            assert (harness.plain_dir / "only.txt").exists()
        finally:
            outside.unlink()

    @posix_only("os.mkfifo")
    def test_failed_overwrite_keeps_the_old_destination(self, harness) -> None:
        root = harness.plain_dir
        src = root / "srcdir"
        src.mkdir()
        os.mkfifo(src / "pipe")  # copytree cannot copy a FIFO
        dest = root / "dest2"
        dest.mkdir()
        (dest / "srcdir").mkdir()
        (dest / "srcdir" / "sentinel.txt").write_text("sentinel\n")
        try:
            result = fs_action("copy", str(src), dest=str(dest), overwrite=True)
            assert "error" in result, result
            assert (dest / "srcdir" / "sentinel.txt").read_text() == "sentinel\n"
            assert sorted(p.name for p in dest.iterdir()) == ["srcdir"]
        finally:
            shutil.rmtree(src)
            shutil.rmtree(dest)

    def test_rename_onto_a_descendant_or_across_folders_is_refused(
        self, harness,
    ) -> None:
        root = harness.plain_dir
        tree = root / "tree"
        (tree / "child").mkdir(parents=True)
        try:
            result = fs_action(
                "rename", str(tree), dest=str(tree / "child"), overwrite=True,
            )
            assert result["error"].startswith("Not a valid name")
            assert (tree / "child").is_dir()
            assert fs_action("rename", str(tree), dest=str(root / "x" / "y"))[
                "error"
            ].startswith("Not a valid name")
        finally:
            shutil.rmtree(tree)

    def test_status_rename_then_modify_marks_only_the_rename(
        self, harness,
    ) -> None:
        from kiss.server.explorer import parse_porcelain_status

        rows = parse_porcelain_status("RM new.txt\0old.txt\0", "/repo")
        abs_path = str(Path("/repo") / "new.txt")  # native separators
        assert rows == [
            {"path": "new.txt", "absPath": abs_path, "status": "R",
             "group": "staged", "origPath": "old.txt"},
            {"path": "new.txt", "absPath": abs_path, "status": "M",
             "group": "changes"},
        ]

    @posix_only("a newline is not a valid NTFS file-name character")
    def test_worktree_paths_with_newlines_and_locked_worktrees(
        self, harness, worktree,
    ) -> None:
        odd = Path(harness.tmpdir) / "wt\nnewline"
        locked = Path(harness.tmpdir) / "wt-locked"
        _git(harness.work_dir, "worktree", "add", "-q", "--detach", str(odd))
        _git(harness.work_dir, "worktree", "add", "-q", "--detach", str(locked))
        _git(harness.work_dir, "worktree", "lock", str(locked))
        shutil.rmtree(locked)  # e.g. unmounted media: locked, folder gone
        try:
            rows = {w["name"]: w for w in git_worktrees(str(harness.work_dir))}
            assert "wt\nnewline" in rows
            assert rows["wt\nnewline"]["detached"] is True
            assert rows["wt\nnewline"]["head"] == harness.shas["merge"]
            assert "wt-locked" in rows, "a locked worktree keeps its head"
            status = _request(
                harness, {"type": "gitStatus", "tabId": "t", "token": "rl"},
                "gitStatus",
            )
            by_name = {w["name"]: w for w in status["worktrees"]}
            assert "error" in by_name["wt-locked"]  # its folder is gone
            assert by_name["wt\nnewline"]["changes"] == []
        finally:
            _git(harness.work_dir, "worktree", "unlock", str(locked))
            _git(harness.work_dir, "worktree", "remove", "--force", str(odd))
            _git(harness.work_dir, "worktree", "prune")

    def test_log_pins_a_worktree_head_the_limit_would_cut(
        self, harness, worktree,
    ) -> None:
        # The worktree head is a start of the walk, but --max-count is a
        # global cap: with limit 1 only the newest commit survives, yet
        # the graph promises every worktree head.
        reply = _request(
            harness, {"type": "gitLog", "token": "lim", "limit": 1}, "gitLog",
        )
        shas = [c["sha"] for c in reply["commits"]]
        heads = {w["head"] for w in reply["worktrees"] if w["head"]}
        assert heads <= set(shas), (heads, shas)
        assert reply["head"] == harness.shas["merge"]
        # The newest commit (date order) leads; the pinned heads follow.
        assert len(shas) == len(set(shas)) and len(shas) >= 2

    def test_log_with_an_unborn_worktree_still_walks_the_others(
        self, harness, worktree,
    ) -> None:
        orphan = Path(harness.tmpdir) / "wt-orphan"
        _git(harness.work_dir, "worktree", "add", "-q", "--detach", str(orphan))
        _git(orphan, "checkout", "-q", "--orphan", "empty")
        try:
            rows = {w["name"]: w for w in git_worktrees(str(harness.work_dir))}
            assert rows["wt-orphan"]["head"] == ""
            # From the committed checkout: the unborn worktree is listed
            # but contributes no start, and the walk succeeds.
            reply = _request(
                harness, {"type": "gitLog", "token": "ub1", "limit": 50}, "gitLog",
            )
            assert "error" not in reply, reply
            assert harness.shas["merge"] in [c["sha"] for c in reply["commits"]]
            # From the unborn checkout: HEAD names nothing, but the other
            # worktrees' heads are still walked.
            reply = _request(
                harness,
                {"type": "gitLog", "workDir": str(orphan), "token": "ub2"},
                "gitLog",
            )
            assert "error" not in reply, reply
            assert reply["head"] == ""
            assert harness.shas["merge"] in [c["sha"] for c in reply["commits"]]
        finally:
            _git(harness.work_dir, "worktree", "remove", "--force", str(orphan))
            _git(harness.work_dir, "worktree", "prune")

    def test_log_carries_the_full_commit_message(self, harness, worktree) -> None:
        with _scratch_worktree(harness) as wt:
            (wt / "msg.txt").write_text("m\n")
            _git(wt, "add", "msg.txt")
            _git(wt, "commit", "-q", "-m", "Title line", "-m", "Body paragraph.")
            reply = _request(
                harness, {"type": "gitLog", "token": "msg", "limit": 50}, "gitLog",
            )
            commit = next(c for c in reply["commits"] if c["subject"] == "Title line")
            assert commit["message"] == "Title line\n\nBody paragraph."

    def test_merge_file_patch_is_against_first_parent(self, harness) -> None:
        # feature.txt was added by the merged-in branch: against the
        # first parent it is an addition; git's default merge output
        # for a path shows nothing for a clean merge.
        reply = _request(
            harness,
            {"type": "gitShow", "sha": harness.shas["merge"],
             "path": "feature.txt", "token": "mf"},
            "gitShow",
        )
        assert "error" not in reply, reply
        assert "+++ b/feature.txt" in reply["text"], reply["text"]

    @posix_only("the file name is not valid on NTFS")
    def test_pathspec_magic_in_a_file_name_is_literal(self, harness) -> None:
        with _scratch_worktree(harness) as repo:
            magic = repo / ":(glob)*.txt"
            magic.write_text("magic\n")
            _git(repo, "add", "--", str(magic))
            _git(repo, "commit", "-q", "-m", "magic name")
            sha = _git(repo, "rev-parse", "HEAD").strip()
            reply = _request(
                harness,
                {"type": "gitShow", "sha": sha, "path": ":(glob)*.txt", "token": "pm"},
                "gitShow",
            )
            assert "error" not in reply, reply
            assert "+++ b/" in reply["text"]
            assert "README.md" not in reply["text"]
            assert "b.txt" not in reply["text"]
            # A legal name that looks like an option is a path after ``--``.
            (repo / "--all").write_text("dash\n")
            _git(repo, "add", "--", "--all")
            _git(repo, "commit", "-q", "-m", "dash name")
            sha2 = _git(repo, "rev-parse", "HEAD").strip()
            dash = _request(
                harness,
                {"type": "gitShow", "sha": sha2, "path": "--all", "token": "pm2"},
                "gitShow",
            )
            assert "error" not in dash, dash
            assert "+++ b/--all" in dash["text"]
            bad = _request(
                harness,
                {"type": "gitShow", "sha": sha, "path": "/etc/passwd", "token": "pm3"},
                "gitShow",
            )
            assert bad["error"].startswith("Not a repository path")

    def test_truncation_counts_bytes_and_marks_once(self, harness) -> None:
        from kiss.server import explorer

        text = "\u00e9" * (explorer.GIT_SHOW_MAX_BYTES // 2 + 10)  # 2 bytes each
        cut, truncated = explorer._truncate(text)
        assert truncated is True
        assert len(cut.encode()) <= explorer.GIT_SHOW_MAX_BYTES
        assert "truncated" not in cut  # the client adds the marker
        kept, flag = explorer._truncate("short")
        assert (kept, flag) == ("short", False)
        # A non-UTF-8 byte git reported (surrogateescape) survives the cut.
        odd = "\udcff" + "x" * explorer.GIT_SHOW_MAX_BYTES
        cut, truncated = explorer._truncate(odd)
        assert truncated is True and cut.startswith("\udcff")

    def test_confinement_follows_symlinked_folders_on_disk(self, harness) -> None:
        root = harness.plain_dir
        outside = Path(harness.tmpdir) / "outside-dir"
        outside.mkdir()
        (outside / "victim.txt").write_text("victim\n")
        portal = root / "portal"
        portal.symlink_to(outside)
        try:
            wd = str(root)
            denied = _fs(
                harness, action="delete", path=str(portal / "victim.txt"), workDir=wd,
            )
            assert denied["error"].startswith("Not inside the workspace"), denied
            assert (outside / "victim.txt").exists()
            created = _fs(
                harness, action="newFile", path=str(portal), name="escaped.txt",
                workDir=wd,
            )
            assert created["error"].startswith("Not inside the workspace")
            assert not (outside / "escaped.txt").exists()
            pasted = _fs(
                harness, action="copy", path=str(root / "only.txt"), dest=str(portal),
                workDir=wd,
            )
            assert pasted["error"].startswith("Not inside the workspace")
            found = _fs(
                harness, action="findInFolder", path=str(portal), query="victim",
                workDir=wd,
            )
            assert found["error"].startswith("Not inside the workspace")
            # The link itself is a workspace entry: deleting it removes
            # the link, not the folder it points to.
            unlinked = _fs(harness, action="delete", path=str(portal), workDir=wd)
            assert unlinked.get("ok") is True, unlinked
            assert not os.path.lexists(portal) and (outside / "victim.txt").exists()
        finally:
            if os.path.lexists(portal):
                portal.unlink()
            shutil.rmtree(outside)

    def test_sha256_repository_unborn_worktree(self, harness) -> None:
        from kiss.server.explorer import git_log

        repo = Path(harness.tmpdir) / "sha256-repo"
        subprocess.run(
            ["git", "init", "-q", "--object-format=sha256", "-b", "main", str(repo)],
            check=True,
        )
        _git(repo, "config", "user.name", "T")
        _git(repo, "config", "user.email", "t@example.com")
        (repo / "f.txt").write_text("f\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-q", "-m", "sha256 root")
        orphan = repo.parent / "sha256-orphan"
        _git(repo, "worktree", "add", "-q", "--detach", str(orphan))
        _git(orphan, "checkout", "-q", "--orphan", "empty")
        try:
            rows = {w["name"]: w for w in git_worktrees(str(repo))}
            assert rows["sha256-orphan"]["head"] == ""
            assert len(rows["sha256-repo"]["head"]) == 64
            reply = git_log(str(repo))
            assert "error" not in reply, reply
            assert [c["subject"] for c in reply["commits"]] == ["sha256 root"]
        finally:
            shutil.rmtree(orphan, ignore_errors=True)
            shutil.rmtree(repo, ignore_errors=True)

    def test_bare_repository_lists_no_worktree(self, harness) -> None:
        bare = Path(harness.tmpdir) / "bare.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        try:
            assert git_worktrees(str(bare)) == []
        finally:
            shutil.rmtree(bare)

    def test_find_and_compare_output_is_capped_while_streaming(self, harness) -> None:
        from kiss.server.fs_actions import TEXT_MAX_BYTES, compare_files

        root = harness.plain_dir
        big = root / "huge.txt"
        # One match per line, 100 bytes each: ~6 MiB of grep output.
        line = "needle " + "y" * 92 + "\n"
        big.write_text(line * (6 * 1024 * 1024 // len(line)))
        try:
            found = fs_action("findInFolder", str(root), query="needle")
            assert found["truncated"] is True
            assert len(found["text"].encode()) <= TEXT_MAX_BYTES
            assert found["text"].endswith("\n")  # no half line
            assert found["text"].count("\n") == FIND_MAX_MATCHES
            assert found["count"] >= FIND_MAX_MATCHES  # matches seen before the cap
            other = root / "huge2.txt"
            other.write_text(line.replace("y", "z") * (6 * 1024 * 1024 // len(line)))
            diff = compare_files(str(big), str(other))
            assert diff["truncated"] is True
            assert len(diff["text"].encode()) <= TEXT_MAX_BYTES
            other.unlink()
        finally:
            big.unlink()
