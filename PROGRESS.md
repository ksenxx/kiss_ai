# Progress

- Started new task: fix installation failure triggered from the VS Code settings
  panel update button.
- Read `SORCAR.md` first as required; it is empty.
- Cleared stale `PROGRESS.md` content from a previous task.
- Inspected the initial worktree status and located update/install related VS
  Code files and tests.
- Read `install.sh`, the VS Code update-button path helper, package metadata,
  build scripts, and existing update/install regression tests.
- Observed existing mitigations already present for earlier update failures:
  `npm ci --ignore-scripts --no-audit --no-fund`, daemon restart after extension
  install, robust `/dev/tty` handling, and update-button installer lookup rooted
  at `~/kiss_ai`.
- Reproduced the VS Code extension packaging step locally. It completed, but
  showed that `vsce package` implicitly reruns `npm run vscode:prepublish` after
  the installer has already reached the packaging command. This explains the
  apparent stall at the exact command printed in the user's log.
- Planned changes:
  1. Add a node packaging helper that calls `@vscode/vsce`'s lower-level
     `pack()` API so the final packaging step writes a real VSIX without
     implicit `vscode:prepublish`.
  1. Make `npm run package` use that helper.
  1. Make install/release scripts run `npm run compile` and
     `npm run copy-kiss` explicitly before `npm run package`, preserving the
     build while avoiding hidden duplicate work.
  1. Add an end-to-end node regression test proving the packaging helper does
     not execute `vscode:prepublish`.
  1. Exclude the helper scripts from the packaged VSIX because they are
     build-time only.
- Added failing regression
  `src/kiss/agents/vscode/test/packageVsixNoPrepublish.test.js`; it initially
  failed because the new helper did not exist.
- Added `src/kiss/agents/vscode/scripts/package-vsix.js` and updated
  `src/kiss/agents/vscode/package.json` to use it and include the regression in
  `npm test`.
- Updated `install.sh`, `scripts/release.sh`, and `scripts/release_exp.sh` so
  `npm ci` is followed by explicit `npm run compile`, `npm run copy-kiss`, and
  then the no-prepublish `npm run package`.
- Updated `.vscodeignore` to exclude `scripts/**` from runtime VSIX contents.
- Updated the Python install-script regression to assert the new explicit build
  order.
- Verified targeted regressions: `node test/packageVsixNoPrepublish.test.js`
  passed and
  `uv run pytest -q src/kiss/tests/agents/vscode/test_install_script_npm_ignore_scripts.py`
  passed (3 tests).
- Verified the install build sequence itself by running
  `npm run compile && npm run copy-kiss && npm run package`; it produced
  `kiss-sorcar.vsix` without the implicit `Executing prepublish script` phase.
- Ran `uv run check --full`; code checks and VS Code tests passed, but mdformat
  failed only because this `PROGRESS.md` file was not formatted. Reformatted the
  ordered-list indentation and long lines manually.
