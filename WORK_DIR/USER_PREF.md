# User Preferences

- The build extension script is at `scripts/build-extension.sh` (hyphenated, not underscore).
- VS Code extension ID is `ksenxx.kiss-sorcar`.
- VS Code extension source is at `src/kiss/agents/vscode/`.
- Always uninstall the old extension before installing a new one to avoid stale state in VS Code's extensions.json.
- User is interested in understanding KISS internals and architecture. Provide detailed, code-referenced answers when asked about how things work.
- The canonical SYSTEM.md is at `src/kiss/SYSTEM.md`; the root-level copy is referenced by `scripts/release.sh` and `src/kiss/agents/vscode/copy-kiss.sh`.
- `copy-kiss.sh` now copies from `src/kiss/SYSTEM.md` instead of root `SYSTEM.md`.
- `scm/inputBox` is a proposed API; the extension uses `scm/title` (stable) instead.
- `GitWorktreeOps.commit_staged()` commits already-staged changes without re-staging; used by `_auto_commit_worktree` to avoid redundant `git add -A`.
- `_release_worktree()` returns the branch name it ended on (or None), so callers can reuse it instead of calling `current_branch()` again.
- In `WorktreeSorcarAgent.run()`, `_chat_id` is allocated before `_restore_from_git()` so the branch prefix search is always meaningful.
- Worktree mode now copies user's dirty state (staged, unstaged, untracked) into the worktree and creates a "baseline commit". The baseline SHA is stored in `git config branch.<name>.kiss-baseline`. All downstream operations (merge review, changed-file detection, conflict checking, squash-merge) diff against the baseline to isolate agent-only changes.
- `GitWorktree` dataclass has a `baseline_commit: str | None = None` field.
- `squash_merge_from_baseline()` uses `git cherry-pick --no-commit baseline..branch` to replay only agent commits. For very small files where the user's dirty changes and agent's changes are on adjacent lines, cherry-pick may report a conflict (this is expected behavior since git's merge algorithm can't distinguish adjacent edits).
- For legacy worktrees without baseline (created before the feature), all merge paths fall back to the original `squash_merge_branch()` behavior.
