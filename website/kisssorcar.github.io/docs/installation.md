# Installing KISS Sorcar

> Install KISS Sorcar from source, as a Python package, as a VS Code extension, or in Docker. Requires Python 3.13+.

## Full Install from Source

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

The installer targets macOS and Linux on `x86_64`, `aarch64`, and `arm64`. It installs or checks the tools needed to run KISS Sorcar and build/install the VS Code extension.

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
~/kiss_ai/sorcar-docker
```

## Next Steps

- [Client Interfaces](cli.md) — the `kiss-web` daemon, chat clients, and Python API
- [Supported Models](models.md) — pick a model
- [Tips](tips.md) — get the highest-quality results
