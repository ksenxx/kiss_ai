# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — symlink bypass of ``_bash_parent_repo_guard``.

The guard matches the command string against the *resolved* parent-repo
prefix only (``Path(work_dir).resolve()``).  When ``work_dir`` contains a
symlinked component (e.g. ``/tmp`` → ``/private/tmp`` on macOS — the very
spelling the model sees in its ``Work dir:`` hint), a Bash command that
targets the parent repo through the *unresolved* spelling never matches
the resolved prefix, bypasses the guard, and mutates the user's main
checkout — the exact failure mode the guard exists to prevent.

Read/Write/Edit are immune because ``_active_worktree_remap`` resolves
both the work_dir and the target path; the Bash guard was the only
inconsistent code path.
"""

from pathlib import Path

from kiss.core.useful_tools import UsefulTools, _bash_parent_repo_guard


def _make_symlinked_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Return (repo_via_symlink, worktree_via_symlink) with a live worktree."""
    real = tmp_path / "real"
    (real / "repo" / ".kiss-worktrees" / "kiss_wt-abc-1").mkdir(parents=True)
    (real / "repo" / "README.md").write_text("MAIN\n")
    link = tmp_path / "link"
    link.symlink_to(real)
    repo = link / "repo"
    return repo, repo / ".kiss-worktrees" / "kiss_wt-abc-1"


def test_guard_catches_unresolved_symlink_spelling(tmp_path: Path) -> None:
    """A parent-repo path spelled through the symlinked (unresolved)
    work_dir prefix must be refused just like the resolved spelling."""
    repo, wt = _make_symlinked_repo(tmp_path)

    err = _bash_parent_repo_guard(f"echo PWNED > {repo}/README.md", str(wt))

    assert err is not None and "worktree" in err, err


def test_bash_symlinked_parent_repo_write_is_refused_e2e(tmp_path: Path) -> None:
    """End-to-end: the Bash tool must not mutate the main checkout via
    the unresolved symlink spelling of the parent-repo path."""
    repo, wt = _make_symlinked_repo(tmp_path)
    target = repo / "README.md"
    tools = UsefulTools(work_dir=str(wt))

    out = tools.Bash(f"echo PWNED > {target}", description="evil")

    assert target.read_text() == "MAIN\n"
    assert "Error" in out, out


def test_guard_still_catches_resolved_spelling(tmp_path: Path) -> None:
    """Regression guard: the resolved spelling stays refused."""
    repo, wt = _make_symlinked_repo(tmp_path)

    err = _bash_parent_repo_guard(
        f"echo PWNED > {repo.resolve()}/README.md", str(wt)
    )

    assert err is not None and "worktree" in err, err


def test_guard_allows_worktree_paths_in_both_spellings(tmp_path: Path) -> None:
    """Commands targeting the worktree itself stay allowed (both spellings)."""
    repo, wt = _make_symlinked_repo(tmp_path)

    assert _bash_parent_repo_guard(f"echo ok > {wt}/f.txt", str(wt)) is None
    assert (
        _bash_parent_repo_guard(f"echo ok > {wt.resolve()}/f.txt", str(wt)) is None
    )
