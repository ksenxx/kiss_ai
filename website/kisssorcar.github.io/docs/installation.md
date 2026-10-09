# Installing KISS Sorcar

> Install KISS Sorcar from source, as a Python package, as a VS Code extension, or in Docker. Requires Python 3.13+.

## Full Install from Source

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

The installer targets macOS and Linux on `x86_64`, `aarch64`, and `arm64`. It installs or checks the tools needed to run KISS Sorcar and build/install the VS Code extension.

If the Update button in the settings UI fails, run the full installation command again. It will not delete your history.

## Python Package Install

If you only want the Python package (the `kiss-web` daemon, the Python client API, and the messaging-agent entry points):

```bash
pipx install kiss-agent-framework
# or
uv tool install kiss-agent-framework
```

KISS Sorcar requires **Python 3.13+**. The PyPI package name is `kiss-agent-framework` and the daemon entry point is `kiss-web`.

## Configure Model Access

You can connect a Claude Code or Codex subscription, configure API keys, or register a custom endpoint.

**Subscription connection.** Use Claude Code 2.1.280+ or Codex 0.162.0+ for subscription isolation. Install the official CLI on the machine running the Sorcar daemon, then use **Settings → CLI Connections → Sign in** and **Refresh status**. Existing logins are reused. The fixed terminal commands are:

```bash
claude auth login
codex login --device-auth
```

Choose **Subscription** and select `cc/opus` or `codex/gpt-6.1-sol`. Sorcar verifies the login before execution and isolates the child process from API credentials and alternate-provider overrides. Codex subscription runs refuse a configured `openai_base_url` or `openai` provider override. New connections use Subscription mode; detected API or enterprise configurations remain available under **Existing CLI configuration**, with their billing shown in Settings. Remote users sign in on the daemon machine; Windows web users can copy the command into a terminal there.

Subscription tasks keep this billing constraint through retries, children, model switches, task updates, and commit helpers. Optional API classification is skipped. A task that requests an API model under that constraint fails with guidance; start a separate API task to use API credits. Fable is optional and requires the saved **Allow Fable usage credits** setting for any Claude subscription login, including one kept as the existing CLI configuration: noninteractive Claude requests can bill usage credits without prompting. Without that opt-in, native Claude child agents use the selected model, and Claude settings files cannot remap model aliases, add a fallback model, or call an advisor model for that run. Provider plan limits and extra-usage settings still apply.

**API connection.** Configure keys in Settings or export them before starting the daemon:

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

You can register your own models (e.g. a local vLLM/Ollama endpoint or a provider model not in the bundled catalog) in the **Custom Models** section of the Settings panel; entries are stored in `~/.kiss/MY_MODELS.json` and appear in the model picker alongside the bundled catalog.

## VS Code Extension

To install only the KISS Sorcar extension, open Visual Studio Code, search for **KISS Sorcar** in the extension marketplace, install it, and relaunch VS Code. You can dismiss the API-key prompt and connect a subscription in CLI Connections, or configure an API backend before running tasks.

## Docker

To run KISS Sorcar in a Docker container (exposes a VS Code interface in the host machine's browser):

```bash
~/kiss_ai/sorcar-docker
```

## Next Steps

- [Client Interfaces](cli.md) — the `kiss-web` daemon, chat clients, and Python API
- [Supported Models](models.md) — pick a model
- [Tips](tips.md) — get the highest-quality results
