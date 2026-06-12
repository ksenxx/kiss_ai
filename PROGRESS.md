# Task: Fix Update-button failure (install.sh hang at "[5/6] Building VS Code extension...")

## Status — COMPLETE

## Symptom

Clicking the settings Update button runs `~/kiss_ai/install.sh` in an integrated
terminal (`SorcarSidebarView._runUpdate()` / `web_server.py _handle_run_update`).
The update froze with no further output right after npm's deprecation warnings:

```
>>> [5/6] Building VS Code extension...
npm warn deprecated whatwg-encoding@3.1.1: ...
npm warn deprecated prebuild-install@7.1.3: No longer maintained. ...
npm warn deprecated glob@11.1.0: ...
^C
```

## Diagnosis

- Failing machine is the user's other Mac (`Koushiks-MBP`), so diagnosed from the
  paste + repo: output stops during `npm ci`'s reify phase — the only step there
  that can block indefinitely with no output is a dependency *install script*.
- `package-lock.json` has exactly two packages with `hasInstallScript`:
  - `keytar` 7.9.0 — **optional**, lazily-imported dep of `@vscode/vsce`
    (`await import('keytar')` in `out/store.js`, publish-credential storage only;
    never used by `vsce package`). Its install script is
    `prebuild-install || node-gyp rebuild`: downloads a prebuilt binary from the
    **archived** atom/node-keytar GitHub releases, falling back to a native
    node-gyp compile — either can hang forever (no timeout) on network/toolchain
    trouble. Matches the symptom: hang right after the `prebuild-install`
    deprecation warning.
  - `@vscode/vsce-sign` — postinstall used only for VSIX signing; `vsce package`
    never signs.
- Neither script is needed to compile (`tsc`) and package (`vsce package`) the
  VSIX.

## Fix

Pass `--ignore-scripts --no-audit --no-fund` to every `npm ci` that builds the
extension:

- `install.sh` step \[5/6\]: `npm ci --ignore-scripts --no-audit --no-fund`
  (with explanatory comment).
- `scripts/release.sh`: added `--ignore-scripts` to its existing
  `npm ci --no-audit --no-fund` (parity).
- `scripts/release_exp.sh`: `npm ci` → `npm ci --ignore-scripts --no-audit --no-fund` (parity).

## Verification

- Full build verified in dev repo: `rm -rf node_modules && npm ci --ignore-scripts --no-audit --no-fund && npm run package` → `DONE Packaged: kiss-sorcar.vsix (187 files, 1.77 MB)`; rebuilt artifact restored via
  `git checkout -- src/kiss/agents/vscode/kiss-sorcar.vsix`.
- New regression test
  `src/kiss/tests/agents/vscode/test_install_script_npm_ignore_scripts.py`:
  - behavioral test builds a throwaway npm project whose dependency tarball has a
    marker-writing install script, first reproduces the bug (plain `npm ci` runs
    the script), then runs `npm ci` with the exact flags parsed out of
    `install.sh` and asserts the script does NOT run while the dep still
    installs (offline, `file:` dep only);
  - plus source-parity checks that `install.sh`/release scripts pass
    `--ignore-scripts` before `npm run package`.
- `bash -n` on all three edited shell scripts: OK.
- Impacted suites pass: new file (3 passed) + test_install_script_daemon_restart,
  test_installer_path_parity, test_check_active_tasks_script,
  test_web_extension_parity (17 passed).
- `uv run check --full` run; fixed pre-existing mdformat failures
  (PROGRESS.md, RECIPES.md, src/kiss/INJECTIONS.md).
