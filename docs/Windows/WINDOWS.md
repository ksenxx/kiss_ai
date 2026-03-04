# Windows Compatibility Notes

## Status

This repository now supports Windows for the standard developer workflow (`uv run check`)
and for core Sorcar file-path handling.

## Shell Behavior (Bash Tool)

The Sorcar `Bash` tool now selects its shell at startup:

1. On Windows, if `bash` is detectable (for example Git Bash), it uses:
   - `bash -lc <command>`
1. On Windows, if `bash` is not available, it falls back to:
   - `cmd.exe /c <command>`
1. On non-Windows, it uses:
   - `$SHELL -c <command>` (or `sh -c` fallback)

This makes Bash-style commands work on Windows whenever Git Bash is installed.

Implementation note:

- The shell selector prefers Git Bash locations and ignores the WSL launcher
  (`C:\Windows\System32\bash.exe`) to avoid accidental WSL routing.

## Path Handling Updates

- Sorcar open-file endpoint now uses OS-aware absolute-path detection (`os.path.isabs`),
  so Windows paths like `C:\repo\file.py` are treated as absolute paths.
- `read_project_file()` now normalizes `\` to `/`, so Windows-style relative paths are accepted.

## Tools You May Need

No extra tool is required for basic Python framework usage.

Install these tools only if you need their related features:

- Git for Windows (recommended): enables Git Bash, which improves Bash tool compatibility.
- code-server: required for Sorcar embedded editor panel.
- Playwright browsers: required for browser automation features.
  - install with: `uv run playwright install`
- Docker Desktop: required for Docker-based agent workflows.

## code-server on Windows

### What it is

`code-server` is VS Code running in a browser.
In this repository, Sorcar uses it to provide the embedded editor panel.

### Official sources

- Project/docs: <https://coder.com/docs/code-server/latest>
- Install docs: <https://coder.com/docs/code-server/latest/install>
- npm package (Windows install path): <https://www.npmjs.com/package/code-server>
- Source repository: <https://github.com/coder/code-server>

### Important Windows note

The official install docs state there are currently no native Windows release binaries;
Windows installation is done through npm.

### Install steps (PowerShell)

1. Install Node.js LTS (includes npm):
   - `winget install --id OpenJS.NodeJS.LTS --exact`
1. Open a new PowerShell window.
1. Verify Node/npm:
   - `node --version`
   - `npm --version`
1. If Git Bash is installed, force npm script execution through Git Bash (recommended on Windows):
   - `$env:npm_config_script_shell = 'C:\Program Files\Git\bin\bash.exe'`
1. Install code-server globally:
   - `npm install --global code-server --unsafe-perm`
1. Verify:
   - `code-server --version`
   - `Get-Command code-server`

### If npm install fails on native module build

Install Windows build prerequisites used by VS Code extensions/native deps:

- VS Code Windows prerequisites:
  <https://github.com/microsoft/vscode/wiki/How-to-Contribute#prerequisites>

Then retry:

- `npm install --global code-server --unsafe-perm`

### If npm install fails with `EBUSY` rename on Windows

On some Windows + npm setups (observed with npm 10.9.x on 2026-03-04), `postinstall.sh`
can fail with:

- `EBUSY: resource busy or locked, rename ...\\node_modules\\code-server -> ...\\.code-server-*`

Reason: during global install, the nested `npm install` runs with inherited global npm
environment and tries to mutate the same global package directory it is currently running from.

Use this workaround (PowerShell), which installs package files first, then runs the
dependency installs locally inside `lib/vscode`:

```powershell
# 1) Install code-server package without postinstall.
$env:npm_config_script_shell = 'C:\Program Files\Git\bin\bash.exe'
npm install --global code-server --unsafe-perm --ignore-scripts

# 2) Resolve the global package path.
$cs = Join-Path (npm root -g) 'code-server'

# 3) Install VS Code server deps locally (not global).
Push-Location "$cs\lib\vscode"
# Critical: remove inherited global npm flags so this install stays local.
Remove-Item Env:npm_config_global -ErrorAction SilentlyContinue
Remove-Item Env:npm_config_prefix -ErrorAction SilentlyContinue
npm install --unsafe-perm --omit=dev
if (-not (Test-Path 'node_modules.asar')) {
  cmd /c "mklink /J node_modules.asar node_modules"
}
Pop-Location

# 4) Install extension deps locally.
Push-Location "$cs\lib\vscode\extensions"
Remove-Item Env:npm_config_global -ErrorAction SilentlyContinue
Remove-Item Env:npm_config_prefix -ErrorAction SilentlyContinue
npm install --unsafe-perm --omit=dev
Pop-Location

# 5) Verify.
code-server --version
```

### Quick run test

Run once to confirm it starts:

- `code-server --bind-addr 127.0.0.1:13338 --auth none .`

Then open:

- <http://127.0.0.1:13338>

### How KISS uses it

KISS/Sorcar checks for `code-server` on `PATH` and uses it automatically when present.
If not found, Sorcar still runs but shows the editor fallback panel.

## Recommended Windows Setup

1. Install Python and `uv`.
1. Install Git for Windows (includes Git Bash).
1. If you want embedded editor support, install `code-server` (section above).
1. Run:
   - `uv venv --python 3.13`
   - `uv sync`
   - `uv run check`

## Notes

- `UsefulTools.Edit` uses a Python fallback path on Windows for robust file editing behavior.
- CI now includes `windows-latest` and runs `uv run check`.
