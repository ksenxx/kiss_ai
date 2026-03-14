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
- Windows drive paths in Git-Bash/WSL forms are normalized before file operations:
  - `/c/repo/file.py`
  - `/mnt/c/repo/file.py`
  map to `C:\repo\file.py`.
- `read_project_file()` now normalizes `\` to `/`, so Windows-style relative paths are accepted.

## Tools You May Need

No extra tool is required for basic Python framework usage.

Install these tools only if you need their related features:

- Git for Windows (recommended): enables Git Bash, which improves Bash tool compatibility.
- Codex CLI (optional fallback): not required for native ChatGPT-plan login in Sorcar;
  still useful as fallback transport/login path.
- code-server: required for Sorcar embedded editor panel.
- Playwright browsers: required for browser automation features.
  - install with: `uv run playwright install`
- Docker Desktop: required for Docker-based agent workflows.

## OpenAI via ChatGPT/Codex Login (No OPENAI_API_KEY)

KISS can run OpenAI-family models using ChatGPT/Codex subscription auth.

Preferred path (no Codex CLI required):

- Open Sorcar auth panel and click `Login plan`.
- Complete browser OAuth flow.
- Sorcar receives callback on `http://localhost:1455/auth/callback` and stores tokens in:
  - `~/.kiss/codex_oauth.json`
- If login cannot start, check whether local port `1455` is already in use.

Optional fallback path:

- If Codex CLI is installed and logged in (`codex login status`), KISS can also use that auth source.

KISS now prefers native OAuth token usage with direct requests to:

- `https://chatgpt.com/backend-api/codex/responses`

and keeps Codex CLI execution as an automatic fallback.

Native token source behavior:

- KISS reads/writes its own cache at `~/.kiss/codex_oauth.json`.
- On first use, KISS bootstraps credentials from Codex CLI auth file:
  - `%USERPROFILE%\.codex\auth.json`
- You can override the source auth file with:
  - `KISS_CODEX_AUTH_FILE=<path-to-auth.json>`

### Setup

1. Verify Codex CLI:
   - `codex --version`
1. Login (browser flow):
   - `codex login`
1. Verify auth:
   - `codex login status`

### Routing behavior in KISS

- KISS treats ChatGPT/Codex auth as a subset of OpenAI model IDs (overlap list), not a full API mirror.
- When both auth sources exist:
  - overlap model IDs use ChatGPT/Codex auth first,
  - API-only model IDs use `OPENAI_API_KEY`.
- If `OPENAI_API_KEY` is not set, only overlap-model IDs are exposed for OpenAI-family routing.
- To force auth mode globally:
  - `KISS_OPENAI_AUTH=codex` (prefer Codex subscription auth)
  - `KISS_OPENAI_AUTH=api` (force API key path)
- To force transport backend when `KISS_OPENAI_AUTH=codex`:
  - `KISS_CODEX_TRANSPORT=native` (native OAuth + direct backend API)
  - `KISS_CODEX_TRANSPORT=cli` (force Codex CLI execution)
- Sorcar UI now includes an **Authentication** panel button. It shows active auth routing for the selected model and supports refresh/logout.

PowerShell examples:

```powershell
$env:KISS_OPENAI_AUTH = "codex"
$env:KISS_CODEX_TRANSPORT = "native"
uv run sorcar .
```

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

### Troubleshooting: Explorer shows `ENOPRO` / no file system provider

If code-server Explorer shows:

- `Unable to resolve workspace folder (ENOPRO: No file system provider found ...)`

on Windows, the folder query must use a leading slash before the drive letter:

- correct: `folder=/C:/github/kiss_ai`
- incorrect: `folder=C:/github/kiss_ai` (can be misparsed as URI scheme `c:`)
- incorrect: `folder=file:///C:/github/kiss_ai`

Sorcar now generates the correct Windows form automatically.

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
- Sorcar code-server restart handling on Windows uses native process/port tooling (`netstat` + `taskkill`),
  so no `lsof` installation is required.
- CI now includes `windows-latest` and runs `uv run check`.
