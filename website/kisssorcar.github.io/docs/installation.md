# Installing KISS Sorcar

> Install KISS Sorcar with the one-line installer, from a checkout, in Docker, or as a Python package. Requires Python 3.13+.

## Full Install (one-liner)

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

The bootstrap script clones or updates `~/.kiss/kiss_ai`, then runs its `install.sh` from the directory you ran the command in, so the daemon serves that directory.

## Full Install from a Checkout

```bash
git clone https://github.com/ksenxx/kiss_ai.git
cd kiss_ai
./install.sh
```

`install.sh` targets macOS and Linux on `x86_64`, `aarch64`, and `arm64`. It runs five steps:

1. Copies the project into `~/.kiss/kiss` (the library, the SEAs, the web app frontend).
2. Builds the Python environment with `uv`.
3. Byte-compiles the library (SEAs and the few files a runtime reads as scripts stay source).
4. Installs the `sorcar` CLI and the `rsorcar` and `sorcar-docker` launchers into `~/.local/bin`, Playwright's Chromium and `cloudflared`.
5. Refreshes `~/.kiss/MODEL_INFO.json`, registers `kiss-web` as your user service (launchd on macOS, a systemd user unit on Linux, a detached process otherwise), restarts it only when no task is running, sets a fresh random remote-access password (printed in a box of `*` at the end; only its hash is stored), and opens the web app at `http://127.0.0.1:8787` as a standalone app: a window of its own without browser UI (Chromium's app mode, started by `kiss-app` in Google Chrome, Chromium, Edge or Brave when one is installed, else in Playwright's Chromium, with its own profile under `~/.kiss/webapp-profile`), and a launcher in your applications menu (`~/.local/share/applications/kiss-sorcar.desktop` on Linux, `~/Applications/KISS Sorcar.app` on macOS) so you can reopen it like any program; a white-label brand gets its own name and icon there, and the running window shows that icon in the dock or taskbar too (on Linux the launcher names the window's X11 class and Wayland app id; on macOS the bundle runs the browser's binary through a symlink under its own identity). Without such a browser the URL opens in the default browser. On a remote or SSH machine it prints the cloudflared URL instead.

Options: `KISS_SKIP_LAUNCH=1` skips starting the daemon and opening the app; `--non-interactive` runs without prompts. The installer logs to `~/.kiss/install.log` and tracks its progress in `~/.kiss/.install-progress`.

## Updating

When a new version is available, the web app shows an update toast with **Update** and **Update when idle**. The daemon then runs `scripts/install.sh` (the bootstrap) in `~/.kiss/kiss_ai`, which runs `install.sh`. If the update fails, run the one-line installer again. It will not delete your history.

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
export ZAI_API_KEY=...
export MOONSHOT_API_KEY=...
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...
export GEMINI_API_KEY=...
```

You can also set API keys, a custom model endpoint, and custom HTTP headers in the Settings panel of the web app — useful for local or self-hosted models.

You can register your own models (e.g. a local vLLM/Ollama endpoint or a provider model not in the bundled catalog) in the **Custom Models** section of the Settings panel; entries are stored in `~/.kiss/MY_MODELS.json` and appear in the model picker alongside the bundled catalog.

## Next Steps

- [Client Interfaces](cli.md) — the `kiss-web` daemon, the web app, and Python API
- [Supported Models](models.md) — pick a model
- [Tips](tips.md) — get the highest-quality results
