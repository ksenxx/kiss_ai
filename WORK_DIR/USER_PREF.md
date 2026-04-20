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
