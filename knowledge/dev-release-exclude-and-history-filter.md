---
title: scripts/exclude.json and the filtered public history (git-filter-repo purge)
uuid: 1d1c31ac-5c20-4cbe-a423-942f8e29a385
summary: 'How release.sh keeps private paths out of public kiss_ai: exclude.json rules,
  build_filtered_history (git-filter-repo), purge_public_history with leases, history
  verification.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# exclude.json and the filtered public history

## `scripts/exclude.json`
A JSON list of literal repo-relative paths (files or folders, no globs) that must never reach
the public repo `ksenxx/kiss_ai`. They stay tracked in origin. It currently lists
`scripts/exclude.json` itself, `reports`, `marketing`, `papers`, `benchmarkings`, `PWD`, `tmp`,
`PROGRESS.md`, `SCRATCH.md`, `RECIPES.md` and `templates`.

`read_exclude_paths` validates the file:
- it must exist (`[]` means exclude nothing) and be a JSON list of strings;
- no newlines, tabs or NUL; no leading `/` and no `..` component;
- trailing `/` is stripped.

`main()` also refuses to release if the file has uncommitted changes.

Consequence: anything under `benchmarkings/` (and `papers/`, `reports/`, ...) exists only in the
private repo. Code in `src/kiss` that the public package needs must not import from those
paths.

## Rewriting new history: `build_filtered_history`
- It rewrites the full history of the commit being released in a throwaway bare repo, using
  git-filter-repo (`resolve_filter_repo` prefers `git filter-repo`, then
  `python3 -m git_filter_repo`, then `uvx --from git-filter-repo git-filter-repo`). The
  checkout is never touched.
- Authors, dates and messages are kept. Commit-hash references inside messages are remapped.
  Commits whose only content was excluded paths are dropped and their children re-parented. The
  result is stored at `refs/kiss-filtered/main` (`FILTERED_REF`).
- It runs before the version bump and publication: a broken exclude.json or a missing
  filter-repo aborts the release there. It is not the first side effect: `main()` has already
  stashed local changes, fetched and `git pull --rebase`d, and run `purge_public_history`
  (which can force-push to kiss_ai).

## Purging already-published history: `purge_public_history`
Filtering new snapshots is not enough once exclude.json gains a path that an earlier release
already published. So on every run, even when there is nothing to release:
- the public branches and tags (mirrored locally under `refs/kiss-public-branches/`,
  `refs/kiss-public-tags/`, `refs/kiss-public-other/`) are rewritten in a throwaway repo;
- `classify_purged_refs` splits them into force-push refspecs and delete refspecs. A ref the
  rewrite dropped must be deleted, or it would keep pointing at unpurged content;
- the push is atomic, with a `--force-with-lease` per ref recording what each ref pointed at
  when fetched, so anything pushed to kiss_ai in the meantime is not clobbered;
- `verify_public_history_clean` re-checks afterwards.

## Verification
- `verify_no_excluded_paths` checks the tip tree.
- `verify_no_excluded_paths_in_history` checks that no ancestor contains an excluded path
  either. `excluded_paths_in_history` does this with one batched `git cat-file` over the cross
  product of commits and paths.
- `warn_unrewritable_public_refs`: GitHub's `refs/pull/*` are read-only. A pull request opened
  against a commit with excluded content keeps it fetchable until the PR is closed and GitHub
  Support runs a server-side GC. The script can only warn.

## Adding a path
Edit `scripts/exclude.json`, commit it, then run the release. The purge step removes the path
from the existing public history as well. Test with `bash scripts/test_release_exclude.sh`,
which builds a scratch repo and a bare "public" remote and checks that excluded paths never
arrive.

## Sources
- `scripts/exclude.json`
- `scripts/release.sh` (`read_exclude_paths`, `filtered_tree`, `build_filtered_history`, `resolve_filter_repo`, `purge_public_history`, `classify_purged_refs`, `verify_no_excluded_paths_in_history`, `excluded_paths_in_history`, `warn_unrewritable_public_refs`)
- `scripts/test_release_exclude.sh`, `src/kiss/tests/scripts/test_release_purge_history.py`
