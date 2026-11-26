# Installing KISS Sorcar

> Install KISS Sorcar with the one-line installer, from a checkout, with the Mac installer package, with the Windows setup program, with the Linux installer, in Docker, or as a Python package. Requires Python 3.13+.

## Full Install (one-liner)

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

The bootstrap script clones or updates `~/.kiss/kiss_ai`, then runs its `install.sh` from the directory you ran the command in, so the daemon serves that directory. This one-line command is for macOS and Linux; on Windows, use the [Windows setup program](#windows-setup-program) or install from a checkout. A Linux without `curl` (an Ubuntu desktop ships `wget` alone) installs with the [Linux installer](#linux-installer).

## Full Install from a Checkout

```bash
git clone https://github.com/ksenxx/kiss_ai.git
cd kiss_ai
./install.sh
```

`install.sh` targets macOS and Linux on `x86_64`, `aarch64`, and `arm64`, and 64-bit Windows 10 and 11. On Windows, open Git Bash (it comes with Git for Windows) in the checkout and run the same `./install.sh`. The script asks no question and needs no administrator password, no compiler, no Homebrew and no git (a copy of the sources without git history installs too): it downloads what is missing (uv, Python, Chromium, cloudflared) into your home directory. It runs five steps:

1. Copies the project into `~/.kiss/kiss` (the library, the SEAs, the web app frontend).
2. Builds the Python environment with `uv`.
3. Byte-compiles the library (SEAs and the few files a runtime reads as scripts stay source).
4. Installs the `sorcar` CLI and the `rsorcar` and `sorcar-docker` launchers into `~/.local/bin` (on Windows also `sorcar.cmd`, and the folder is added to your user PATH), Playwright's Chromium and `cloudflared` (from Cloudflare's GitHub release).
5. Refreshes `~/.kiss/MODEL_INFO.json`, registers `kiss-web` as your user service (launchd on macOS, a systemd user unit on Linux, a per-user "Run" registry entry on Windows that starts it at every logon without a console window, a detached process otherwise), restarts it only when no task is running, sets a fresh random remote-access password (printed in a box of `*` at the end; only its hash is stored), and opens the web app at `http://127.0.0.1:8787` as a standalone app: a window of its own without browser UI (Chromium's app mode, started by `kiss-app` in Google Chrome, Chromium, Edge or Brave when one is installed, else in Playwright's Chromium, with its own profile under `~/.kiss/webapp-profile`), and a launcher in your applications menu (`~/.local/share/applications/kiss-sorcar.desktop` on Linux, `~/Applications/KISS Sorcar.app` on macOS, a Start Menu shortcut on Windows) so you can reopen it like any program; a white-label brand gets its own name and icon there, and the running window shows that icon in the dock or taskbar too (on Linux the launcher names the window's X11 class and Wayland app id; on macOS the bundle runs the browser's binary through a symlink under its own identity). Without such a browser the URL opens in the default browser. On a remote or SSH machine it prints the cloudflared URL instead.

Options: `KISS_SKIP_LAUNCH=1` skips starting the daemon and opening the app; `--non-interactive` is for an automated caller: the run detaches from the terminal so that no terminal signal can cut it short (the installer asks no question in either mode). The installer logs to `~/.kiss/install.log` and tracks its progress in `~/.kiss/.install-progress`.

## Mac Installer Package

[Download KISS-Sorcar.pkg](https://github.com/ksenxx/kiss_ai/raw/main/installers/KISS-Sorcar.pkg) and open it. It runs the same `install.sh` for the logged-in user on macOS 14 or later (Apple silicon and Intel) and needs no Homebrew, no git and none of Apple's developer tools. The Mac must be online: the installer downloads Python, the libraries and Chromium, about 1 GB.

The package is not signed with an Apple Developer ID, so macOS blocks it when you double-click it. Open **System Settings > Privacy & Security**, click **Open Anyway** next to the message about the package, and confirm with an administrator password. Or install it from Terminal, which is not blocked:

```bash
installer -pkg ~/Downloads/KISS-Sorcar.pkg -target CurrentUserHomeDirectory
```

KISS Sorcar opens in its own window when the installation is complete. Enter the API key of a model provider in **Settings** before your first task.

## Windows Setup Program

[Download KISS-Sorcar-Setup.exe](https://github.com/ksenxx/kiss_ai/raw/main/installers/KISS-Sorcar-Setup.exe) and open it. It runs the same `install.sh` for the user who opens it on 64-bit Windows 10 and Windows 11 (ARM PCs run it through their x64 emulation) and needs no administrator rights, no git, no Python and no bash on the PC. The PC must be online: the installer downloads Python, the libraries and Chromium, about 1 GB, which takes a few minutes.

The program is not signed, so Windows SmartScreen shows "Windows protected your PC" for a downloaded copy. Click **More info**, then **Run anyway**. To install without a window, run it from a terminal with `/S`:

```
KISS-Sorcar-Setup.exe /S
```

The setup program unpacks a copy of the sources into `%USERPROFILE%\.kiss\kiss_pkg` and runs `install.sh` through Git Bash. A PC without Git for Windows first gets the portable Git for Windows in `%LOCALAPPDATA%\Programs\Git`. `install.sh` builds the app in `%USERPROFILE%\.kiss\kiss` and starts the `kiss-web` daemon, which also starts at every logon of that user. It adds a **KISS Sorcar** shortcut to the Start Menu, puts the `sorcar` command (`sorcar.cmd`) in `%USERPROFILE%\.local\bin` and adds that folder to the user's PATH.

KISS Sorcar opens in its own window when the installation is complete. Enter the API key of a model provider in **Settings** before your first task.

To uninstall, open **Settings > Apps > Installed apps** (**Apps & features** on Windows 10), find **KISS Sorcar** and click **Uninstall**. This removes the app, the Start Menu shortcut, the logon start and the `sorcar` command. Your settings, API keys and history stay in `%USERPROFILE%\.kiss`.

## Linux Installer

[Download KISS-Sorcar-Linux.run](https://github.com/ksenxx/kiss_ai/raw/main/installers/KISS-Sorcar-Linux.run), open a terminal and run it:

```bash
bash ~/Downloads/KISS-Sorcar-Linux.run
```

It runs the same `install.sh` for the user who runs it on 64-bit Linux (x86-64 and ARM64) and needs no root, no git, no Python and no curl: `wget` is enough, which is what an Ubuntu desktop ships. The machine must be online: the installer downloads Python, the libraries and Chromium, about 1 GB. It is tested on Ubuntu 24.04 LTS, 26.04 LTS and 26.10, Debian 13, Fedora 44 and 45, openSUSE Tumbleweed and Arch Linux.

The installer unpacks a copy of the sources into `~/.kiss/kiss_pkg` and runs `install.sh`, which builds the app in `~/.kiss/kiss`, puts the `sorcar` command in `~/.local/bin` and starts the `kiss-web` daemon as a systemd user service, which also starts whenever the machine starts. The daemon's working folder is the directory you ran the command from, your home directory in a new terminal.

KISS Sorcar opens in its own window when the installation is complete; on a machine without a display (a server reached over ssh) the installer prints the address to open from another device and the remote-access password. Enter the API key of a model provider in **Settings** before your first task.

A desktop install has the system libraries the agent's browser needs. A minimal system (a server, a container) may lack them; the installer installs them when it runs as root or `sudo` asks no password, and otherwise prints the command that does: `sudo ~/.kiss/kiss/.venv/bin/python -m playwright install-deps chromium`.

## Updating

When a new version is available, the web app shows an update toast with **Update** and **Update when idle**. The daemon then runs `scripts/install.sh` (the bootstrap) in `~/.kiss/kiss_ai`, which runs `install.sh`. If the update fails, run the one-line installer again. It will not delete your history.

Windows cannot replace the files of a running program, so on Windows the installer stops `kiss-web` before it updates the app and starts it again afterwards (on macOS and Linux the daemon keeps running during an update). It refuses to update while tasks are running.

## Docker

```bash
./sorcar-docker [PORT] [--rebuild]
```

`sorcar-docker` builds a plain Debian image from the `Dockerfile` and runs a container whose entrypoint (`scripts/docker-startup.sh`) clones the repo (the private `kiss.git` when `GH_TOKEN` is set, otherwise the public `kiss_ai.git`), runs `install.sh`, sets the web app password (`KISS_WEB_PASSWORD`, or a generated one printed in `docker logs`) and starts `kiss-web` on port 8787 (host port 8787 by default). State persists in the `kiss-sorcar-web-home` volume. Open `https://localhost:PORT` on the host; the certificate is self-signed, so accept it once or install `/ca.crt`.

## Python Package Install

If you only want the Python package (the `kiss-web` daemon, the Python client API, and the messaging-agent entry points):

```bash
pipx install kiss-agent-framework
# or
uv tool install kiss-agent-framework
```

KISS Sorcar requires **Python 3.13+**. The PyPI package name is `kiss-agent-framework` and the daemon entry point is `kiss-web`.

## Configure Model Access

Provide at least one model backend. You can use environment variables such as:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
```

You can also set API keys, a custom model endpoint, and custom HTTP headers in the Settings panel of the web app — useful for local or self-hosted models.

You can register your own models (e.g. a local vLLM/Ollama endpoint or a provider model not in the bundled catalog) in the **Custom Models** section of the Settings panel; entries are stored in `~/.kiss/MY_MODELS.json` and appear in the model picker alongside the bundled catalog.

### Default and Fast Model Order

With no explicit model configured, KISS Sorcar picks the first provider with configured credentials in priority order:

1. **Anthropic**: `claude-opus-5-5` (default) and `claude-sonnet-5-5` (fast).
2. **OpenAI**: `gpt-6.1-sol-medium` (default) and `gpt-6-luna` (fast).
3. **Gemini**: `gemini-3.8-flash` (default) and `gemini-3.5-flash-lite` (fast).
4. **OpenRouter**: `openrouter/anthropic/claude-opus-5.5` (default) and `openrouter/anthropic/claude-sonnet-5.5` (fast).
5. **Bedrock**: `bedrock/global.anthropic.claude-opus-5-5` / `bedrock/global.anthropic.claude-sonnet-5-5` (or registered `bedrock/*` profiles via `AWS_BEARER_TOKEN_BEDROCK` or AWS credentials + `AWS_REGION`).
6. **Claude Code CLI (`cc/*`)**: `cc/opus` (default) and `cc/haiku` (fast); bills your Claude subscription by default (`claude setup-token` supported in Settings).
7. **Codex CLI (`codex/*`)**: `codex/default`; bills your ChatGPT subscription by default (`codex login --device-auth` or `CODEX_ACCESS_TOKEN` supported in Settings).

### Router Agents

Six bundled router SEAs appear under the `Router` group in the model picker: `autorouter`, `bestrouter`, `orautorouter`, `orbestrouter`, `ccxbestrouter`, and `brbestrouter`.

## Next Steps

- [Client Interfaces](cli.md) — the `kiss-web` daemon, the web app, and Python API
- [Supported Models](models.md) — pick a model
- [Tips](tips.md) — get the highest-quality results
