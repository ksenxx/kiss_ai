# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — stale-worktree fallback inconsistencies in UsefulTools.

``_stale_worktree_fallback`` is documented as applying to a *now-deleted*
``.kiss-worktrees/kiss_wt-*`` directory, and ``Read`` uses it so a model
that remembers a torn-down worktree path transparently reads the parent
repo copy.  Three inconsistencies with that contract:

1. ``Edit`` never applies the fallback: after a successful ``Read`` of a
   stale worktree path, ``Edit`` on the *same* path fails with
   "File not found" — inconsistent sibling code paths.

2. ``Write`` never applies the fallback either: writing to a stale
   worktree path silently *resurrects a zombie worktree directory*
   (``mkdir(parents=True)``) whose contents are never merged — the exact
   failure mode the module's own docstring warns about.

3. The helper never checks that the worktree is actually deleted, so
   ``Read`` on a *live* worktree path whose file was deleted inside the
   worktree silently leaks the parent repo's copy instead of reporting
   not-found (the same leak test_active_worktree_path_remap_review_bugs
   Bug 3 forbids for the parent-path spelling).
"""

from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A fake parent repo with one real file; no worktree on disk."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("REAL CONTENT\n")
    return tmp_path


def _stale_path(repo: Path) -> Path:
    return (
        repo
        / ".kiss-worktrees"
        / "kiss_wt-abc123-1700000000"
        / "src"
        / "module.py"
    )


def test_edit_stale_worktree_path_falls_back_to_repo(repo: Path) -> None:
    """Edit on a stale worktree path must edit the parent-repo file,
    exactly like Read transparently reads it."""
    tools = UsefulTools()
    stale = _stale_path(repo)

    # Read succeeds via the documented fallback...
    assert tools.Read(str(stale)) == "REAL CONTENT\n"

    # ...so Edit on the very same path must succeed too.
    out = tools.Edit(str(stale), "REAL CONTENT", "EDITED CONTENT")

    assert "Successfully replaced" in out, out
    assert (repo / "src" / "module.py").read_text() == "EDITED CONTENT\n"
    # No zombie worktree directory may appear.
    assert not (repo / ".kiss-worktrees").exists()


def test_edit_stale_worktree_path_still_errors_when_repo_file_missing(
    repo: Path,
) -> None:
    """No silent success when neither the worktree nor repo copy exists."""
    tools = UsefulTools()
    stale = repo / ".kiss-worktrees" / "kiss_wt-dead-1" / "nope.py"

    out = tools.Edit(str(stale), "a", "b")

    assert "File not found" in out, out
    assert not (repo / ".kiss-worktrees").exists()


def test_write_stale_worktree_path_does_not_resurrect_zombie_worktree(
    repo: Path,
) -> None:
    """Write to a stale worktree path must target the parent repo, not
    recreate ``.kiss-worktrees/kiss_wt-*`` (contents there are never
    merged and are silently lost)."""
    tools = UsefulTools()
    stale = _stale_path(repo)

    out = tools.Write(str(stale), "NEW CONTENT\n")

    assert "Successfully wrote" in out, out
    # The write must land in the parent repo...
    assert (repo / "src" / "module.py").read_text() == "NEW CONTENT\n"
    # ...and must NOT resurrect a zombie worktree directory.
    assert not (repo / ".kiss-worktrees").exists()


def test_read_live_worktree_deleted_file_does_not_leak_repo_copy(
    repo: Path,
) -> None:
    """A *live* worktree path whose file was deleted in the worktree must
    report not-found — not silently return the parent repo's copy."""
    wt_root = repo / ".kiss-worktrees" / "kiss_wt-live-1"
    (wt_root / "src").mkdir(parents=True)  # worktree exists, file deleted

    out = UsefulTools().Read(str(wt_root / "src" / "module.py"))

    assert "REAL CONTENT" not in out, out
    assert "File not found" in out, out


def test_read_stale_worktree_fallback_still_works(repo: Path) -> None:
    """Regression guard: the documented Read fallback keeps working."""
    out = UsefulTools().Read(str(_stale_path(repo)))

    assert out == "REAL CONTENT\n"
