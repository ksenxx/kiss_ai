---
title: 'Release workflow: scripts/release.sh to kiss_ai, GitHub release, PyPI, VS
  Code Marketplace'
uuid: cbc950b6-3176-4d58-a758-915d3efe0a1f
summary: 'scripts/release.sh step by step: preflight, purge, release_needed, calver
  bump, VSIX build, filtered history to ksenxx/kiss_ai, gh release, uv publish, vsce
  publish, local install.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Release workflow (`scripts/release.sh`)

Development happens in the private origin (`ksenxx/kiss`). A release publishes to:
- the public GitHub repo `ksenxx/kiss_ai` (`PUBLIC_REPO_URL`, remote name `public`), with a
  filtered copy of the history (see `dev-release-exclude-and-history-filter`);
- PyPI as `kiss-agent-framework` (`PYPI_PACKAGE_NAME`);
- the VS Code Marketplace as `ksenxx.kiss-sorcar`;
- a GitHub release on kiss_ai with the `.vsix` attached.

## Steps in `main()`
1. Preflight: record the current branch, `ensure_remote` for `public`. `scripts/exclude.json`
   must exist, parse (`read_exclude_paths`) and have no uncommitted changes. Uncommitted edits to
   it would be stashed away below, and the release would silently use stale rules.
2. Stash uncommitted work (`git stash push --include-untracked`). An EXIT trap pops it if the
   script fails. Then `git fetch origin`, `fetch_public_refs` (kiss_ai refs are mirrored under
   `refs/kiss-public-branches/*` and `refs/kiss-public-tags/*`) and
   `git pull --rebase origin <branch>`.
3. `purge_public_history`: rewrite every branch and tag already on kiss_ai so the excluded paths
   never existed. This is a no-op when the history is already clean.
4. `build_filtered_history` of the current commit, then `release_needed(filtered, public_main)`.
   Nothing to release means: kiss_ai main already has the VSIX, its tree minus the VSIX equals
   the filtered tree, the tip is not a merge, and its parent is the filtered head. In that case
   the script runs `run_local_install` and exits 0.
5. `bump_version`: calendar version `YEAR.MONTH.N` (N+1 in the same month, otherwise `.0`).
   It is written to `src/kiss/core/_version.py`, `README.md`, `src/kiss/SYSTEM.md`, and the
   extension's `package.json` and `package-lock.json` (`update_*_version`).
6. `build_vscode_extension`: copy README into the extension dir, then
   `npm ci --ignore-scripts --no-audit --no-fund --omit=optional`, `npm run compile`,
   `npm run copy-kiss`, `npm run package` produces `kiss-sorcar.vsix`, and `out/` and
   `kiss_project/` are removed. `--omit=optional` drops keytar, which is not needed because
   `--pat` is passed explicitly. `*.vsix` is gitignored, so the file is never committed to
   origin.
7. Commit "Version bumped to <version>" and push to origin. The push is `pull --rebase` + `push`
   with up to 3 attempts.
8. Push the rewritten development history to kiss_ai, plus a release tip commit that adds the
   VSIX (`tree_with_vsix`, `create_public_commit`, `push_public_snapshot`), and tag it with the
   version.
9. `gh release create <tag>` on kiss_ai and upload the VSIX asset.
10. `publish_to_pypi`: `uv build`, then `check_pypi_file_sizes`, then `uv publish`. Requires
    `UV_PUBLISH_TOKEN` (see `dev-release-pypi-size-guard`).
11. `publish_vscode_extension`: `npx @vscode/vsce publish --packagePath kiss-sorcar.vsix --pat
    "$VSCE_PAT" --allow-proposed-apis contribSourceControlInputBoxMenu`. Requires `VSCE_PAT`.
12. `run_local_install`: `KISS_SKIP_LAUNCH=1 bash ./install.sh --non-interactive`, so this
    machine runs the version it just released.
13. Pop the stash.

## Why it is shaped this way
- The history pushed to kiss_ai is the full development history, rewritten, not a squashed
  snapshot (commit "publish full filtered development history to kiss_ai instead of squashed
  snapshots"). The rewrite is deterministic for a given git-filter-repo version, so consecutive
  releases share commits.
- The VSIX exists only in kiss_ai's release tip commit and the GitHub release asset. Users who
  install with `scripts/install.sh` clone kiss_ai. `install.sh`'s `guard_vsix_tracking` refuses
  to proceed if origin ever starts tracking the `.vsix`.
- `gh`'s update notice is suppressed during the release.

## Tests
- `scripts/test_release_exclude.sh` (exclude filtering), `scripts/test_release_pypi_size.sh`,
  and the purge suite. Pytest wrappers live in `src/kiss/tests/scripts/test_release_*.py`, and
  `test_release_vscode_publish.py` covers the publish step.

## Sources
- `scripts/release.sh` (`main`, `release_needed`, `bump_version`, `build_vscode_extension`, `publish_to_pypi`, `publish_vscode_extension`, `run_local_install`, `purge_public_history`)
- `install.sh` (`guard_vsix_tracking`)
- `src/kiss/tests/scripts/test_release_exclude_and_vsix.py`, `test_release_purge_history.py`, `test_release_vscode_publish.py`
