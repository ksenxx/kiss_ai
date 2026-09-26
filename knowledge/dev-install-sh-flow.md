---
title: 'install.sh: source install flow, 5 steps, non-interactive mode, setsid detachment'
uuid: 01630a87-c956-4eaf-8dc2-8d7842ab47f2
summary: 'Root install.sh: bootstrap git/node/VS Code CLI, build and install kiss-sorcar.vsix,
  copy MODEL_INFO.json, .extension-updated, KISS_SKIP_LAUNCH, --non-interactive, setsid
  re-exec.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# install.sh: source install flow

The repo-root `install.sh` is deliberately small. It bootstraps only the tools needed to build
and install the VS Code extension from a checkout, then launches VS Code. Runtime setup (uv,
Python env, Playwright, and so on) belongs to the extension's `DependencyInstaller`, so a user
gets the same setup whether they run this script or install the VSIX directly. It never
upgrades an already-installed third-party tool.

Usage: `./install.sh [--non-interactive]`. Output goes to the terminal and is appended to
`~/.kiss/install.log`.

## Who runs it
- `scripts/install.sh` (the `curl ... | bash` one-liner) after cloning `~/.kiss/kiss_ai` (see
  `dev-install-bootstrap-and-update-lock`)
- the VS Code settings-panel Update button, and the kiss-web daemon's update endpoint, with
  `--non-interactive`
- `scripts/release.sh` and `./rsorcar` at the end, with `KISS_SKIP_LAUNCH=1 --non-interactive`
- `scripts/docker-startup.sh`, the Docker entrypoint

## Steps
0. macOS only: `ensure_xcode_clt`, `ensure_homebrew` (asks `[Y/n]` in interactive mode).
1. `[1/5]` git: `install_git` if missing. uv is only reported; the extension installs it.
2. `[2/5]` Node.js, npm and npx: `install_node` downloads `NODE_VERSION` (v22.16.0) from
   nodejs.org into `~/.local` when missing.
3. `[3/5]` VS Code CLI: `find_code_cli`, else `install_code_cli`. Then `clear_vscode_cache`,
   so step 5 is not served from stale cached state.
4. `[4/5]` Build the extension in `src/kiss/agents/vscode`:
   `npm ci --ignore-scripts --omit=optional --prefer-offline --no-audit --no-fund` (one retry
   with a clean `node_modules`), `npm run compile`, `apply_brand_overlay`, `npm run copy-kiss`,
   `npm run package` to get `kiss-sorcar.vsix`, then `restore_brand_overlay`, which also runs
   when the build fails (see `dev-brand-overlay`). `--ignore-scripts` exists because npm install
   scripts hung the Update button. Then `install_repo_script_launcher` writes wrappers for
   `rsorcar` and `sorcar-docker` into `~/.local/bin`. They are wrappers, not symlinks, so the
   real script sees the checkout as its own directory.
5. `[5/5]` `code --install-extension kiss-sorcar.vsix --force`, then `guard_vsix_tracking`
   (fails if git tracks the `.vsix`), copy `src/kiss/core/models/MODEL_INFO.json` to
   `${KISS_HOME:-~/.kiss}/MODEL_INFO.json`, write a UTC timestamp to
   `$KISS_HOME/.extension-updated`, and remove the legacy `install_dir` files.
6. Launch: skipped when `KISS_SKIP_LAUNCH` is set, because the caller launches the editor.
   Docker's code-server would otherwise fail with EADDRINUSE. Also skipped when VS Code is
   already running (`vscode_is_running`): the extension watches `out/extension.js` and
   `~/.kiss/.extension-updated` and reloads its window itself. Otherwise `launch_vscode`.

## Interactive vs non-interactive
`_KISS_INTERACTIVE` is 0 when `--non-interactive` or `KISS_NONINTERACTIVE` is given, or when
`/dev/tty` cannot be opened (cron, CI, daemon). Every question then takes its default answer,
Yes.

## Signal immunity (setsid re-exec)
A user's Update run died in the middle of `tsc` from a SIGINT they did not send. A terminal
signal reaches every process in the foreground group, and Node does not reliably keep an
inherited SIG_IGN. The fix, in the `kiss-new-session-reexec` block that tests extract
verbatim: in non-interactive mode, when perl with `POSIX::setsid` is available, the script
`exec`s perl, which calls `setsid()` and re-execs `bash install.sh`. The sentinel
`_KISS_NEW_SESSION=1` prevents a loop. A session with no controlling terminal cannot receive
terminal signals. Interactive mode skips this because `[Y/n]` and sudo prompts need the
terminal. The tee'd output ignores INT/TERM, and `run_with_heartbeat` prints progress every
`KISS_HEARTBEAT_INTERVAL` (15 s) during silent npm/git steps. Tests for this carry the
`process_killer` marker.

## Sources
- `install.sh` (`install_node`, `find_code_cli`, `clear_vscode_cache`, `install_repo_script_launcher`, `guard_vsix_tracking`, `vscode_is_running`, `launch_vscode`, `run_with_heartbeat`, the `kiss-new-session-reexec` and `kiss-interactive-mode` blocks)
- `src/kiss/tests/test_install_model_info_copy.py`, `src/kiss/tests/agents/vscode/test_install_script_new_session_immunity.py`, `src/kiss/tests/scripts/test_install_script_npm_ignore_scripts.py`
