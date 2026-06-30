# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: merge must not fabricate conflicts from baseline dirty state.

When the user has uncommitted dirty edits at WT-creation time that fall
INSIDE a region the agent later deletes (or modifies) on the worktree
branch, the previous ``cherry-pick --no-commit baseline..branch``
implementation produced a spurious modify/delete conflict on merge.

Mechanism of the spurious conflict:
- ``_do_merge`` stashed the dirty main state before the merge, leaving
  main HEAD == ``baseline^`` (clean).
- ``cherry-pick`` of ``baseline..branch`` performs a 3-way merge per
  commit with **base = each commit's parent**.  For the first commit
  on the worktree branch the parent is ``baseline`` (which captured
  the dirty edits), so cherry-pick saw:

      base   = baseline                  (has dirty edits)
      ours   = main HEAD == baseline^    (no dirty edits)
      theirs = branch tip                (has dirty + agent deletions)

  ``base → ours`` reverts the dirty edits while ``base → theirs``
  deletes the surrounding hunk → modify/delete conflict — even though
  the agent's actual net diff (``baseline..branch``) and main HEAD's
  actual content (``baseline^``) are perfectly compatible.

These tests build the exact scenario with real git in a temp repo and
exercise the real :class:`GitWorktreeOps` API (no mocks, patches, or
test doubles).  Pre-fix they fail with ``MergeResult.CONFLICT``; the
fix in ``squash_merge_from_baseline`` must let them pass.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _git,
)

# Mirrors the actual user file in the chat session that triggered the
# bug (papers/kisssorcar/social/twitter_mimic.md): four large
# "Options" with user wording tweaks inside Option A and Option D, and
# an agent that keeps only Option D.
ORIGINAL = (
    "# Twitter mimic — drafts\n"
    "\n"
    "## Option A — corporate\n"
    "Sentence A1.\n"
    "Sentence A2 about KISS Sorcar.\n"
    "Sentence A3 wrap-up.\n"
    "\n"
    "## Option B — playful\n"
    "Sentence B1.\n"
    "Sentence B2 with emoji.\n"
    "Sentence B3 close.\n"
    "\n"
    "## Option C — technical\n"
    "Sentence C1 architecture detail.\n"
    "Sentence C2 throughput numbers.\n"
    "Sentence C3 link to repo.\n"
    "\n"
    "## Option D — provocateur\n"
    "Sentence D1 sharp hook.\n"
    "Sentence D2 about prompt runners.\n"
    "Sentence D3 closing punch.\n"
)


def _dirty_edits(text: str) -> str:
    """Apply the user's pre-WT wording tweaks inside Option A AND D."""
    text = text.replace(
        "Sentence A2 about KISS Sorcar.\n",
        "Sentence A2 about KISS Sorcar (user-edited).\n",
    )
    text = text.replace(
        "Sentence D2 about prompt runners.\n",
        "Sentence D2 about prompt runners — user wording.\n",
    )
    return text


def _branch_tip_content() -> str:
    """What the agent leaves after 'keep only Option D' — incl. dirty edit."""
    return (
        "# Twitter mimic — drafts\n"
        "\n"
        "## Option D — provocateur\n"
        "Sentence D1 sharp hook.\n"
        "Sentence D2 about prompt runners — user wording.\n"
        "Sentence D3 closing punch.\n"
    )


def _make_repo(path: Path) -> Path:
    """Create a git repo with the unedited ORIGINAL file committed on main."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True,
        check=True,
    )
    target = path / "papers" / "kisssorcar" / "social"
    target.mkdir(parents=True, exist_ok=True)
    (target / "twitter_mimic.md").write_text(ORIGINAL)
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _setup_worktree_with_dirty_baseline(
    repo: Path, branch: str
) -> tuple[Path, str]:
    """Reproduce the production flow: dirty edits captured into baseline."""
    target = repo / "papers" / "kisssorcar" / "social" / "twitter_mimic.md"
    target.write_text(_dirty_edits(ORIGINAL))

    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
    GitWorktreeOps.stage_all(wt_dir)
    GitWorktreeOps.commit_staged(wt_dir, "kiss: baseline from dirty state")
    baseline = GitWorktreeOps.head_sha(wt_dir)
    assert baseline is not None
    GitWorktreeOps.save_baseline_commit(repo, branch, baseline)
    return wt_dir, baseline


def _agent_keeps_only_option_d(wt_dir: Path) -> None:
    """Agent edit: delete Options A, B, C; keep Option D (with dirty edit)."""
    target = wt_dir / "papers" / "kisssorcar" / "social" / "twitter_mimic.md"
    target.write_text(_branch_tip_content())
    GitWorktreeOps.stage_all(wt_dir)
    GitWorktreeOps.commit_staged(wt_dir, "agent: keep only Option D")


class TestMergeWithDirtyBaselineRegion:
    """Real-git scenario from chat session task 3.

    User had small dirty edits inside Option A AND Option D.  Agent
    deleted Options A, B, C (keeping only Option D, with its dirty
    line preserved).  Pre-fix merge failed with CONFLICT; post-fix
    must SUCCEED with the agent's intent committed and the user's
    surviving dirty edits restored.
    """

    def test_merge_succeeds_without_spurious_conflict(self) -> None:
        """Exact bug repro: cherry-pick must not fabricate a modify/delete
        conflict from baseline-captured dirty edits when the agent
        deletes the surrounding region.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            wt_dir, baseline = _setup_worktree_with_dirty_baseline(
                repo, "kiss/wt-conflict-repro"
            )
            _agent_keeps_only_option_d(wt_dir)

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)

            # Production flow: stash any dirty main state, then merge.
            assert GitWorktreeOps.stash_if_dirty(repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, "kiss/wt-conflict-repro", baseline,
            )
            assert result == MergeResult.SUCCESS, (
                "Cherry-pick fabricated a spurious modify/delete conflict "
                "from the baseline-captured dirty edits.  The agent's net "
                "diff (baseline..branch) and main HEAD content (baseline^) "
                "are semantically compatible."
            )

            # The committed HEAD must reflect the agent's intent: Option D
            # only.  The user's dirty wording at Option D is in the stash
            # and will be restored on stash pop if it doesn't conflict
            # with the merge — but that is a stash-pop concern, not a
            # cherry-pick regression.
            committed = _git(
                "show",
                "HEAD:papers/kisssorcar/social/twitter_mimic.md",
                cwd=repo,
            )
            assert committed.returncode == 0
            content = committed.stdout
            assert "Option A" not in content
            assert "Option B" not in content
            assert "Option C" not in content
            assert "Option D" in content

            # No leftover conflict markers in the index.
            assert "<<<<<<<" not in content
            assert "=======" not in content
            assert ">>>>>>>" not in content

    def test_dirty_main_with_non_overlapping_edit_preserved(self) -> None:
        """When user dirty edits do NOT overlap any agent-touched hunk,
        the merge succeeds AND the stash pop cleanly restores them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            # An unrelated file (never touched by the agent) is the user's
            # only dirty state.  twitter_mimic.md is left untouched and
            # only the agent edits it on the worktree branch.
            (repo / "notes.txt").write_text("user scratch notes\n")

            branch = "kiss/wt-conflict-repro-2"
            wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
            assert GitWorktreeOps.create(repo, branch, wt_dir)
            GitWorktreeOps.save_original_branch(repo, branch, "main")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            GitWorktreeOps.save_baseline_commit(repo, branch, baseline)

            # Agent edits twitter_mimic.md only; never touches notes.txt.
            _agent_keeps_only_option_d(wt_dir)

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)

            assert GitWorktreeOps.stash_if_dirty(repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

            assert GitWorktreeOps.stash_pop(repo), (
                "Dirty edits to a file the agent never touched must be "
                "restored cleanly after merge."
            )

            # Unrelated dirty file restored intact.
            assert (repo / "notes.txt").read_text() == "user scratch notes\n"
            # The commit captures only the agent's edit to twitter_mimic.md;
            # notes.txt is not part of the commit (it stays dirty).
            committed_notes = _git("show", "HEAD:notes.txt", cwd=repo)
            assert committed_notes.returncode != 0
            target = (
                repo / "papers" / "kisssorcar" / "social" / "twitter_mimic.md"
            )
            assert "Option A" not in target.read_text()
            assert "Option D" in target.read_text()
