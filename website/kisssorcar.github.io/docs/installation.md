# Installing KISS Sorcar

> Install KISS Sorcar from source, as a Python package, as a VS Code extension, or in Docker. Requires Python 3.13+.

## Full Install from Source

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

The installer targets macOS and Linux on `x86_64`, `aarch64`, and `arm64`. It installs or checks the tools needed to run KISS Sorcar and build/install the VS Code extension.

### Upgrade questions

Run from a terminal, the installer asks before replacing tools you already have. Each of these is a `[Y/n]` question (Enter means yes; "n" keeps what you have and continues):

- installing Homebrew (macOS only, when it is missing)
- upgrading git, uv, Node.js, or VS Code when the installed version is older than the one KISS Sorcar requires

Tools that are missing altogether (git, Node.js, the VS Code CLI, the Xcode Command Line Tools on macOS) are installed without a question, because the installer cannot continue without them. Declining an upgrade is allowed but the installer warns you about the consequence (for example, the extension build may fail with an old Node.js). On Linux a git upgrade goes through `sudo`, which prompts for your password as usual.

To answer every question with its default (yes) without asking, pass the flag or set the variable for the shell that runs the installer:

```bash
~/.kiss/kiss_ai/install.sh --non-interactive
# or
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | KISS_NONINTERACTIVE=1 bash
```

Without a terminal (cron, CI, the daemon) the installer is non-interactive automatically. The **Update** button in the VS Code extension and the web app, the Docker image, and `rsorcar` always run it non-interactively; in those paths a failed upgrade is reported as a warning and the installer continues.

Other switches: `KISS_NO_BREW=1` skips the Homebrew install entirely, `KISS_SKIP_UPDATE=1` installs the checkout as-is without a `git pull`, and `KISS_SKIP_LAUNCH=1` leaves VS Code closed at the end. Everything the installer does is logged to `~/.kiss/install.log`.

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

You can also set API keys, a custom model endpoint, and custom HTTP headers in the Settings panel of the VS Code extension or web app — useful for local or self-hosted models.

## VS Code Extension

To install only the KISS Sorcar extension, open Visual Studio Code, search for **KISS Sorcar** in the extension marketplace, install it, and relaunch VS Code. Press ESC if you do not have a specific API key ready, but configure at least one model backend before running tasks.

## Docker

To run KISS Sorcar in a Docker container (exposes a VS Code interface in the host machine's browser):

```bash
~/.kiss/kiss_ai/sorcar-docker
```

## Next Steps

- [Client Interfaces](cli.md) — the `kiss-web` daemon, chat clients, and Python API
- [Supported Models](models.md) — pick a model
- [Tips](tips.md) — get the highest-quality results
