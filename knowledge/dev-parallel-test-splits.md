---
title: Running the full test suite in parallel splits (pytest + JS)
uuid: 53eb8c98-ddd8-481a-b246-9dccb5381c17
summary: 'Recipe for running all ~11k pytest ids and ~360 JS suites concurrently:
  collect ids, split, bash runner with mapfile, slow/Docker/process_killer groups,
  worktree prep, flake triage.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Running the full suite in parallel splits

The suite is too large to run in one process. The agent's own rules say: split the test ids
evenly into `min(#tests, max(1, cores - 2))` splits and run them concurrently with
`run_commands_parallel`, one pytest command per split.

## Prep in a worktree
`node_modules` is symlinked into agent worktrees but the extension's compiled `out/` is not.
Run `npm run compile` (or `npx tsc -p .`) in `src/kiss/agents/vscode` once before JS tests,
the Playwright tests that load `out/`, or the extension lint. Never `npm install` in a worktree.

## Python
1. Collect ids:
   `uv run pytest --collect-only -q -p no:cacheprovider | grep '::' > tmp/all_ids.txt`.
   The default `addopts` already drops `slow` and `process_killer`.
2. Split, e.g. `split -n r/<N> -d -a 2 tmp/all_ids.txt tmp/splits/py_`. `r/N` deals lines
   round-robin, so each split gets an equal count; `l/N` balances bytes, not test counts.
3. Run each split from a `#!/bin/bash` script:
   ```bash
   mapfile -t ids < "$1" || exit 2
   [ "${#ids[@]}" -gt 0 ] || exit 2   # an empty list would run the WHOLE suite
   uv run pytest -q -p no:cacheprovider -o addopts="" -rfE \
     --timeout 600 --timeout-method=thread "${ids[@]}"
   ```
   - Ids contain spaces (parametrized ids), so never use `$(cat file)`.
   - The agent's Bash tool runs `/bin/sh` (dash on Linux), which has no `mapfile` or `time`,
     so put the loop in a bash script file.
   - `-o addopts=""` is safe because the ids were collected under the default deselection.
   - A split killed by `--timeout` prints no summary line; grep its log for `+++ Timeout +++`.
   - Have the runner write full logs to `tmp/logs/<split>.log` and print only the summary line
     plus `FAILED`/`ERROR` lines, so the combined report stays small.
4. Slow set: collect with `-m slow -o addopts=""`. Run the Docker-backed ones in one process,
   since they contend for the daemon, and split the rest.
5. `-m process_killer -o addopts=""` last and alone. These tests signal whole process groups.

## JavaScript
`src/kiss/agents/vscode/test/run-all.js` (`npm test` = compile + `node test/run-all.js`) runs
every `*.test.js` and `*.coverage.js` file in `test/` sequentially, each in its own node
process with `--no-maglev --no-concurrent-sparkplug` and a 10-minute per-suite timeout. To
parallelize, split that file list and run each file the same way from inside
`src/kiss/agents/vscode`:
`timeout 900 node --no-maglev --no-concurrent-sparkplug test/<file>`. Resolve list/log paths
with `realpath` before `cd`. See `vscode-js-tests` for why the V8 flags matter.

## Triage
- Re-run each failed id alone. If it passes alone, it is load-dependent: reproduce it under load
  (several concurrent copies plus heavy splits) before calling it a flake. A load-only failure
  can still come from a deterministic ordering bug in the test.
- Compare the `file:line` of each failure, not only the node id.
- After a model-catalog refresh commit (`update_models.py`), expect stale hardcoded model names
  and prices in `src/kiss/tests/core/models/`.
- A renamed test leaves its old id in a split file, and pytest exits 4 ("not found"). This is
  harmless: re-run with the new id.
- `git status` after JS runs may show scratch files under `test/` (gitignored).

## Sources
- `pyproject.toml` (`addopts`, markers)
- `src/kiss/agents/vscode/test/run-all.js` (`V8_FLAGS`, `SUITE_TIMEOUT_MS`, `testFiles`)
- `src/kiss/agents/vscode/package.json` (`test` script)
- `src/kiss/SYSTEM.md` (Testing: parallel-split rule)
