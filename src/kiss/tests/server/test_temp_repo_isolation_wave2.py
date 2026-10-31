# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Throwaway test repositories must never reach the developer's checkout.

On 2026-10-30 a test run with ``TMPDIR`` inside this repository's
worktree committed that worktree several times.  The "non-git folder"
cases of ``test_autocommit_after_merge.py`` and
``test_manual_commit_notifications.py`` hand ``_commit_main_tree`` a
fresh ``tempfile.mkdtemp()`` folder; git walked up from it into the
enclosing checkout, and ``git add -A`` + commit swept every agent's
work in progress into a commit named after the tests' fake message.
A throwaway repository whose ``git init`` failed (``/tmp`` out of
inodes) walks up the same way, and a leaked ``GIT_DIR`` (pytest run
from a git hook) points every git call at the hook's repository.

Each test builds a scratch "checkout" (a repository with an ignored
``tmp/`` and an uncommitted edit, like a developer's tree) and checks
that it is left exactly as it was.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from kiss.tests.conftest import is_root, posix_only
from kiss.tests.server.conftest import init_repo

_REPO_ROOT = Path(__file__).resolve().parents[4]

# One "non-git folder" test from each file that committed the worktree,
# plus one test whose throwaway repository comes from ``init_repo``.
_CHILD_TESTS = (
    "src/kiss/tests/server/test_autocommit_after_merge.py::TestAutocommitChanges"
    "::test_commit_in_non_git_dir_reports_failure",
    "src/kiss/tests/server/test_manual_commit_notifications.py::TestManualCommitFailure"
    "::test_non_git_folder_fails_with_error_toast",
    "src/kiss/tests/server/test_autocommit_after_merge.py::TestAutocommitChanges"
    "::test_commit_stages_and_commits_tracked_and_untracked",
)


def _git(cwd: Path, *args: str) -> str:
    """Run ``git *args`` in *cwd* with a clean repository scope; return stdout."""
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True,
    ).stdout


def _checkout(root: Path) -> Path:
    """Create a stand-in developer checkout at *root* and return it.

    It has one commit, an ignored ``tmp/`` folder (the agents' scratch
    space, where a misdirected ``TMPDIR`` points) and an uncommitted
    edit that a stray ``git add -A`` + commit would sweep up.
    """
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "dev@example.com")
    _git(root, "config", "user.name", "Developer")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "a.txt").write_text("committed\n", encoding="utf-8")
    (root / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    _git(root, "add", "a.txt", ".gitignore")
    _git(root, "commit", "-q", "-m", "developer commit")
    (root / "a.txt").write_text("work in progress\n", encoding="utf-8")
    (root / "tmp").mkdir()
    return root


def _snapshot(checkout: Path) -> tuple[str, str, str]:
    """Return the checkout's HEAD, short status and local config."""
    return (
        _git(checkout, "rev-parse", "HEAD"),
        _git(checkout, "status", "--porcelain"),
        _git(checkout, "config", "--local", "--list"),
    )


@contextmanager
def _hook_env(checkout: Path) -> Iterator[None]:
    """Set the variables git exports to a hook of *checkout*, restoring them after."""
    names = ("GIT_DIR", "GIT_WORK_TREE")
    saved = {name: os.environ.get(name) for name in names}
    os.environ["GIT_DIR"] = str(checkout / ".git")
    os.environ["GIT_WORK_TREE"] = str(checkout)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.parametrize("hook_env", [False, True], ids=["tmpdir", "tmpdir+basetemp+hook-env"])
def test_session_with_temp_root_inside_a_checkout_leaves_it_alone(
    tmp_path: Path, hook_env: bool,
) -> None:
    """A child session whose temp root sits in a checkout must not touch it.

    The child pytest gets ``TMPDIR`` inside the scratch checkout and no
    inherited ``GIT_CEILING_DIRECTORIES``: the root conftest alone must
    keep git's repository search inside the temp roots.  The second
    case adds an in-checkout ``--basetemp``, an unrelated inherited
    ceiling, and the ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE``
    a git hook of the checkout would export.
    """
    checkout = _checkout(tmp_path / "checkout")
    before = _snapshot(checkout)
    env = {k: v for k, v in os.environ.items() if k != "GIT_CEILING_DIRECTORIES"}
    env["TMPDIR"] = str(checkout / "tmp")
    extra: list[str] = []
    if hook_env:
        env["GIT_DIR"] = str(checkout / ".git")
        env["GIT_WORK_TREE"] = str(checkout)
        env["GIT_INDEX_FILE"] = str(checkout / ".git" / "index")
        env["GIT_CEILING_DIRECTORIES"] = str(tmp_path / "unrelated")
        extra = [f"--basetemp={checkout / 'tmp' / 'basetemp'}"]

    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra, *_CHILD_TESTS],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert _snapshot(checkout) == before
    assert child.returncode == 0, child.stdout[-4000:] + child.stderr[-2000:]
    assert "3 passed" in child.stdout


def test_init_repo_raises_when_git_init_fails(tmp_path: Path) -> None:
    """A folder ``git init`` cannot initialize makes ``init_repo`` raise."""
    target = tmp_path / "broken"
    target.mkdir()
    (target / ".git").write_text("not a gitdir pointer\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="git init"):
        init_repo(target)

    assert not (target / "seed.txt").exists()


@posix_only("a read-only folder blocks git init")
@pytest.mark.skipif(is_root(), reason="root ignores the read-only permission bits")
def test_init_repo_failure_inside_a_checkout_leaves_it_alone(tmp_path: Path) -> None:
    """A failed ``git init`` must stop ``init_repo`` before git walks up.

    Without the check, ``git config user.email ...`` in the folder that
    has no ``.git`` of its own rewrites the enclosing checkout's
    configuration.
    """
    checkout = _checkout(tmp_path / "checkout")
    before = _snapshot(checkout)
    target = checkout / "tmp" / "repo"
    target.mkdir()
    target.chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match="git init"):
            init_repo(target)
    finally:
        target.chmod(0o755)

    assert _snapshot(checkout) == before


def test_init_repo_refuses_a_folder_that_is_not_its_own_repository(tmp_path: Path) -> None:
    """``init_repo`` refuses to configure or commit outside the new folder.

    Inside a git hook of the checkout (``GIT_DIR`` / ``GIT_WORK_TREE``
    exported), ``git init`` "re-initializes" the checkout and succeeds;
    every later step would land in the checkout.
    """
    checkout = _checkout(tmp_path / "checkout")
    before = _snapshot(checkout)
    target = tmp_path / "throwaway"

    with _hook_env(checkout), pytest.raises(RuntimeError, match="not the root of its own"):
        init_repo(target)

    assert _snapshot(checkout) == before
    assert not (target / "seed.txt").exists()


def test_init_repo_builds_its_own_repository(tmp_path: Path) -> None:
    """The happy path: ``main`` checked out, ``seed.txt`` committed, a clean tree."""
    repo = tmp_path / "repo"

    init_repo(repo)

    assert Path(_git(repo, "rev-parse", "--show-toplevel").strip()) == repo.resolve()
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert _git(repo, "log", "--format=%s").strip() == "seed"
    assert _git(repo, "status", "--porcelain") == ""
