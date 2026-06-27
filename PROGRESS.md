# Progress — Diagnose and fix spurious merge-conflict message

## Task

Investigate why the previous task generated a manual merge-conflict message,
reproduce it with an end-to-end test, fix it, test it using `claude-opus-4-7`,
and then review the work with `gpt-5.5` (non-codex).

## Steps so far

- Read `SORCAR.md` first as required; it is empty.

- Switched to `claude-opus-4-7` for implementation/test work as requested.

- Inspected current branch/status and confirmed this task starts on
  `kiss/wt-1782544196-1298b638` at commit
  `d64fd6fe feat: add start_line parameter to Read tools`.

- Located the manual-conflict message in `WorktreeSorcarAgent.merge()` and
  inspected the previous failed branch `kiss/wt-1782543318-283af1fe`.

- Replayed the previous task's manual cherry-pick in an isolated temporary git
  worktree under `./tmp/repro-prev-conflict`:

  ```text
  git cherry-pick --no-commit a3582790521293a064c1687de0f0e7b671ed77ce..kiss/wt-1782543318-283af1fe
  CONFLICT (content): Merge conflict in src/kiss/INJECTIONS.md
  ```

  The conflict markers showed a whitespace/wording drift between the
  dirty-state baseline and the branch edit in `src/kiss/INJECTIONS.md`; the
  temp worktree was then removed.

- Wrote an end-to-end regression test in
  `src/kiss/tests/agents/sorcar/test_progress_md_merge_conflict.py`:

  ```python
  def test_injections_md_prompt_drift_is_scratch_and_auto_resolves(self) -> None:
      injections = self.repo / "src" / "kiss" / "INJECTIONS.md"
      injections.parent.mkdir(parents=True)
      injections.write_text(INJECTIONS_HEAD)
      _commit_all(self.repo, "add injection scratch prompt")

      injections.write_text(INJECTIONS_BASELINE)
      wt_dir = _create_worktree(self.repo, "kiss/wt-test-injections")
      assert GitWorktreeOps.copy_dirty_state(self.repo, wt_dir)
      GitWorktreeOps.stage_all(wt_dir)
      assert GitWorktreeOps.commit_staged(
          wt_dir, "kiss: baseline from dirty state", no_verify=True
      )
      baseline = GitWorktreeOps.head_sha(wt_dir)
      assert baseline is not None
      _git("checkout", "--", "src/kiss/INJECTIONS.md", cwd=self.repo)

      (wt_dir / "src" / "kiss" / "INJECTIONS.md").write_text(INJECTIONS_BRANCH)
      (wt_dir / "agent_work.py").write_text("y = 2\n")
      _commit_all(wt_dir, "agent task work")
      GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

      result = GitWorktreeOps.squash_merge_from_baseline(
          self.repo, "kiss/wt-test-injections", baseline
      )

      assert result == MergeResult.SUCCESS
  ```

- Confirmed the new test failed before the fix with `MergeResult.CONFLICT` and
  git reporting `could not apply ... agent task work`.

- Fixed `GitWorktreeOps.ensure_scratch_merge_driver()` to register both scratch files:

  ```python
  for scratch_path in ("PROGRESS.md", "src/kiss/INJECTIONS.md"):
      GitWorktreeOps._append_info_line(
          repo, "attributes", f"{scratch_path} merge=kiss-scratch"
      )
  ```

- Updated the merge-driver docstring to describe both `PROGRESS.md` and
  `src/kiss/INJECTIONS.md` scratch semantics.

- Extended the idempotency test to assert exactly one
  `src/kiss/INJECTIONS.md merge=kiss-scratch` attributes line.

- Verified the new test passes after the fix.

- Ran impacted scratch/worktree tests in parallel across 8 splits
  (10 cores → 8 workers): all 92 selected tests passed.

- Switched to `gpt-5.5` for review. Review findings:

  - The root-cause identification is concrete and reproducible: the previous
    branch still exists, its baseline commit is `a3582790`, and a clean replay
    of `git cherry-pick --no-commit a3582790..kiss/wt-1782543318-283af1fe`
    conflicts specifically in `src/kiss/INJECTIONS.md`.
  - The new test is end-to-end enough for this subsystem: it uses a real
    temporary git repo, real worktree branch, real dirty-state baseline commit,
    real cherry-pick merge path through
    `GitWorktreeOps.squash_merge_from_baseline()`, and no mocks/fakes.
  - The failure mode is correctly pinned: before the fix,
    `GitWorktreeOps.ensure_scratch_merge_driver()` only added
    `PROGRESS.md merge=kiss-scratch`, so `src/kiss/INJECTIONS.md` used Git's
    default text merge and conflicted.
  - The fix is minimal and general for the observed bug: it extends the
    existing repo-local scratch merge-driver registration to the second scratch
    file rather than changing merge control flow or suppressing all conflicts.
  - Real source conflicts remain guarded by the existing
    `test_real_source_conflict_still_reported`; adding `INJECTIONS.md` does not
    make arbitrary project files auto-resolve.
  - Idempotency remains covered for both attributes entries.
  - One documentation nit was found: the test module docstring still described
    only `PROGRESS.md`; updated it to describe both scratch files.
