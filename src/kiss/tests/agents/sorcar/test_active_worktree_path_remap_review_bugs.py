# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for bugs gpt-5.5 flagged in commit 43a1b70f.

The previous fix (``_active_worktree_remap``) plugs the common case
where an LLM emits a parent-repo absolute path while running inside a
``.kiss-worktrees/kiss_wt-*`` worktree.  A subsequent gpt-5.5 review
identified several escape hatches that re-create the original
"task didn't run in worktree / no commit" failure mode:

1. **Relative paths** — ``Path("README.md").resolve()`` uses the
   *host process's* ``os.getcwd()``, not the agent's ``work_dir``.
   A relative path emitted by the LLM during a worktree task
   therefore lands wherever the host process was started.

2. **Edit of a file present in main but absent from the worktree** —
   the remap is gated on ``remapped.is_file()``; if the worktree
   branch deleted the file, ``is_file()`` is False and the edit
   silently mutates main.

3. **Read of a file deleted in the worktree** — the remap is gated on
   ``remapped.exists()`` and the stale-worktree fallback can walk a
   non-existent worktree path back into main.  Either way Read
   returns stale main content.

4. **Bash** — the previous fix did not touch the Bash tool, so a
   model can still leak edits to the main checkout via shell
   redirection / ``sed -i`` / ``rm`` etc.

5. Successful Read/Write/Edit operations that were silently remapped
   must surface the remap so the model can reason about its own
   filesystem state on follow-up turns.

6. The auto-commit + squash-merge pipeline (the *actual* invariant
   the original bug broke) must advance ``main`` by exactly one
   commit whose tree contains the remapped edit, not just leave the
   worktree's working tree dirty.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.useful_tools import UsefulTools


def _run(*args: str, cwd: Path) -> None:
    """Run a subprocess and raise on non-zero exit."""
    subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _stdout(*args: str, cwd: Path) -> str:
    """Run a subprocess and return its trimmed stdout."""
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a real git repo + a real ``.kiss-worktrees/kiss_wt-*`` worktree.

    Mirrors :func:`test_active_worktree_path_remap.repo_with_worktree`.
    """
    repo = tmp_path / "myrepo"
    repo.mkdir()
    _run("git", "init", "-b", "main", ".", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("MAIN README v1\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("# main app\n")
    _run("git", "add", "-A", cwd=repo)
    _run("git", "commit", "-m", "init", cwd=repo)
    GitWorktreeOps.ensure_excluded(repo)

    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-test-cafebabe"
    _run(
        "git", "worktree", "add",
        "-b", "kiss/wt-test-cafebabe", str(wt_dir),
        cwd=repo,
    )
    return repo, wt_dir


# ────────────────────────────────────────────────────────────────────
# Bug 1 — Relative paths bypass the remap
# ────────────────────────────────────────────────────────────────────

def test_relative_path_edit_lands_in_worktree_even_when_host_cwd_unrelated(
    repo_with_worktree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Edit("README.md", ...)`` from a worktree task must hit the
    worktree's README regardless of the host process's ``os.getcwd()``.

    Reproducer for Bug 1: an LLM that emits a relative path during a
    worktree task expects it to resolve under ``work_dir`` (which the
    system prompt tells it is the worktree).  Currently
    ``Path("README.md").resolve()`` uses ``os.getcwd()`` instead, so
    the resolved path can be ``<unrelated>/README.md`` — completely
    bypassing the worktree.
    """
    repo, wt_dir = repo_with_worktree
    # Host cwd is an unrelated tmp directory — the bug only surfaces
    # when the host's cwd is NOT the parent repo (otherwise the remap
    # accidentally still catches it).
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Edit("README.md", "MAIN README v1", "EDITED v2")

    assert "Successfully replaced" in out, out
    assert (wt_dir / "README.md").read_text() == "EDITED v2\n"
    assert (repo / "README.md").read_text() == "MAIN README v1\n"
    # And it must NOT have created or touched anything in the unrelated dir.
    assert not (unrelated / "README.md").exists()


def test_relative_path_write_lands_in_worktree_even_when_host_cwd_unrelated(
    repo_with_worktree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as above for ``Write`` of a brand-new file."""
    repo, wt_dir = repo_with_worktree
    unrelated = tmp_path / "unrelated2"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Write("notes.md", "hello\n")

    assert "Successfully wrote" in out, out
    assert (wt_dir / "notes.md").read_text() == "hello\n"
    assert not (repo / "notes.md").exists()
    assert not (unrelated / "notes.md").exists()


def test_relative_path_read_returns_worktree_content(
    repo_with_worktree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Read("README.md")`` must return the worktree's content."""
    repo, wt_dir = repo_with_worktree
    (wt_dir / "README.md").write_text("WT VERSION\n")
    unrelated = tmp_path / "unrelated3"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Read("README.md")

    assert out == "WT VERSION\n", out


# ────────────────────────────────────────────────────────────────────
# Bug 2 — Edit on file present in main but absent from worktree
# ────────────────────────────────────────────────────────────────────

def test_edit_does_not_mutate_main_when_file_missing_from_worktree(
    repo_with_worktree,
) -> None:
    """``Edit`` must never fall through to main when the active worktree
    doesn't have the file.

    Reproducer for Bug 2: the worktree branch deletes ``README.md``;
    main still has it.  An LLM that emits the main-repo path currently
    falls through the remap (``remapped.is_file()`` is False) and
    edits main directly.
    """
    repo, wt_dir = repo_with_worktree
    # Worktree's branch deletes README.md and commits.
    (wt_dir / "README.md").unlink()
    _run("git", "add", "-A", cwd=wt_dir)
    _run("git", "commit", "-m", "drop readme", cwd=wt_dir)
    assert not (wt_dir / "README.md").exists()
    assert (repo / "README.md").exists()
    main_before = (repo / "README.md").read_text()
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Edit(str(repo / "README.md"), "MAIN README v1", "PWNED")

    # The main repo's README MUST be untouched.
    assert (repo / "README.md").read_text() == main_before
    # The tool must surface a clear error rather than silently
    # writing to main.
    assert "Error" in out, out


# ────────────────────────────────────────────────────────────────────
# Bug 3 — Read returning stale main content when worktree deleted file
# ────────────────────────────────────────────────────────────────────

def test_read_does_not_return_main_content_when_file_missing_from_worktree(
    repo_with_worktree,
) -> None:
    """``Read`` must report not-found rather than returning main's copy
    when the worktree branch removed the file.

    Reproducer for Bug 3 + ``_stale_worktree_fallback`` interaction.
    """
    repo, wt_dir = repo_with_worktree
    (wt_dir / "README.md").unlink()
    _run("git", "add", "-A", cwd=wt_dir)
    _run("git", "commit", "-m", "drop readme", cwd=wt_dir)
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Read(str(repo / "README.md"))

    # MUST NOT leak main's content.
    assert "MAIN README v1" not in out, out
    # And SHOULD be a not-found error.
    assert "not found" in out.lower() or "Error" in out, out


# ────────────────────────────────────────────────────────────────────
# Bug 4 — Bash absolute-main-repo paths leak edits
# ────────────────────────────────────────────────────────────────────

def test_bash_output_redirection_to_main_repo_is_refused(
    repo_with_worktree,
) -> None:
    """``Bash("... > /abs/main/repo/X")`` from a worktree task must not
    mutate the main repo.

    Reproducer for Bug 4: the LLM bypasses Read/Write/Edit's remap by
    going through the shell.  The fix should reject (or at minimum
    not actually perform) shell writes targeting the parent repo's
    working tree.
    """
    repo, wt_dir = repo_with_worktree
    target = repo / "README.md"
    before = target.read_text()
    tools = UsefulTools(work_dir=str(wt_dir))

    tools.Bash(
        f"echo PWNED > {target}",
        description="evil",
    )

    # Main MUST be untouched.
    assert target.read_text() == before


def test_bash_sed_inplace_against_main_repo_is_refused(
    repo_with_worktree,
) -> None:
    """``sed -i`` against a main-repo absolute path must not succeed."""
    repo, wt_dir = repo_with_worktree
    target = repo / "README.md"
    before = target.read_text()
    tools = UsefulTools(work_dir=str(wt_dir))

    # Use the GNU/BSD-compatible form: -i with an explicit empty backup
    # suffix works on macOS BSD sed; -i alone works on GNU sed.  We
    # just test that the file CONTENT doesn't change either way.
    if subprocess.run(["sed", "--version"], capture_output=True).returncode == 0:
        cmd = f"sed -i 's/MAIN/PWNED/' {target}"
    else:  # BSD sed (macOS)
        cmd = f"sed -i '' 's/MAIN/PWNED/' {target}"
    tools.Bash(cmd, description="evil")

    assert target.read_text() == before


def test_bash_rm_against_main_repo_is_refused(
    repo_with_worktree,
) -> None:
    """``rm /abs/main/repo/X`` must not delete files from the main repo."""
    repo, wt_dir = repo_with_worktree
    sentinel = repo / "src" / "app.py"
    assert sentinel.exists()
    tools = UsefulTools(work_dir=str(wt_dir))

    tools.Bash(f"rm -f {sentinel}", description="evil")

    assert sentinel.exists()


def test_bash_inside_worktree_still_works(repo_with_worktree) -> None:
    """The Bash guard must not break legitimate worktree-scoped commands.

    Operations on the worktree's own working tree must still succeed
    (the guard is only for the parent-repo prefix that is NOT also a
    worktree prefix).
    """
    repo, wt_dir = repo_with_worktree
    tools = UsefulTools(work_dir=str(wt_dir))

    out = tools.Bash(
        f"echo WT_OK > {wt_dir / 'notes.md'}",
        description="legit",
    )

    assert "Error" not in out, out
    assert (wt_dir / "notes.md").read_text() == "WT_OK\n"


def test_bash_on_paths_outside_repo_still_works(
    repo_with_worktree, tmp_path: Path,
) -> None:
    """Commands targeting paths unrelated to the parent repo must succeed."""
    repo, wt_dir = repo_with_worktree
    scratch = tmp_path / "scratch.txt"
    tools = UsefulTools(work_dir=str(wt_dir))

    tools.Bash(f"echo HELLO > {scratch}", description="scratch")

    assert scratch.read_text() == "HELLO\n"


# ────────────────────────────────────────────────────────────────────
# Bug 6 — Auto-commit + squash-merge end-to-end
# ────────────────────────────────────────────────────────────────────

def test_remap_actually_advances_main_via_squash_merge(
    repo_with_worktree,
) -> None:
    """End-to-end: a remapped Edit must produce a main-advancing commit.

    The original failure mode was "no commit was created on main".
    This test invokes the framework's actual auto-commit +
    squash-merge functions and asserts that main's HEAD advances by
    exactly one commit and that the commit's tree contains the
    remapped edit.
    """
    repo, wt_dir = repo_with_worktree
    branch = "kiss/wt-test-cafebabe"
    main_head_before = _stdout("git", "rev-parse", "main", cwd=repo)
    baseline = main_head_before  # worktree was branched from main.
    tools = UsefulTools(work_dir=str(wt_dir))

    # Simulate the LLM's main-repo Edit during a worktree task.
    tools.Edit(str(repo / "README.md"), "MAIN README v1", "MERGED")

    # Auto-commit on the worktree branch.
    assert GitWorktreeOps.has_uncommitted_changes(wt_dir)
    committed = GitWorktreeOps.commit_all(wt_dir, "agent: remapped edit")
    assert committed

    # Squash-merge the branch into main.
    result = GitWorktreeOps.squash_merge_from_baseline(
        repo, branch, baseline,
    )
    # Whatever the enum reports as "applied", main's head must advance.
    main_head_after = _stdout("git", "rev-parse", "main", cwd=repo)
    assert main_head_after != main_head_before, (
        f"main did not advance: result={result}, head={main_head_after}"
    )
    # And the post-merge main must contain the remapped edit.
    assert (repo / "README.md").read_text() == "MERGED\n"
