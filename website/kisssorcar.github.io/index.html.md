# KISS Sorcar — Tell it what to do, in English. It picks the best LLM and ships the work.

> Open-source general-purpose recursively self-evolving AI agent for long-horizon tasks and AI discovery and optimization. Apache 2.0, 993 models. Works as a web/mobile app, a Python API, and a terminal command.

- Website: <https://kisssorcar.github.io/>
- Source: <https://github.com/ksenxx/kiss_ai>
- PyPI: <https://pypi.org/project/kiss-agent-framework/>
- Paper: <https://arxiv.org/abs/2604.23822>
- Full docs: <https://kisssorcar.github.io/docs/index.md>
- LLM index: <https://kisssorcar.github.io/llms.txt>

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

**KISS Sorcar is a free, simple, local-first, bring-your-own-key AI agent framework.** It runs as a browser/mobile web app served by a local daemon, and offers a Python client API for scripting tasks. Your prompts and code go directly to the model provider or local endpoint you configure — *never through our servers*. Multi-model workflows take a paragraph of prompt, not a pipeline: complex AI systems and techniques can be replaced with a paragraph of prompt in KISS Sorcar.

- **Version:** 2026.11.15
- **License:** Apache-2.0
- **Python:** 3.13+

## § 01 — Terminal-Bench 2.0: KISS Sorcar vs Pi, Codex CLI and Claude Code

On the HarnessTax study's 30 Terminal-Bench 2.0 tasks and seven models, KISS Sorcar solved **75.6%** of attempts against 70.0% for Pi, 65.7% for Codex CLI, and 65.1% for Claude Code, with the best point estimate on every model. Paired reruns of Pi on Claude Fable 5 put KISS Sorcar **+11.1 points** ahead on the 30 study tasks (95% CI +3.3 to +20.0) and **+10.5 points** ahead on 57 unsampled tasks (95% CI +2.9 to +18.7). Full benchmark details: [docs/benchmarks.md](docs/benchmarks.md) and [blog/harness-tax-terminal-bench-blog.html](blog/harness-tax-terminal-bench-blog.html).

## § 02 — Installation

### Full install (recommended, macOS and Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

Or from a checkout:

```bash
git clone https://github.com/ksenxx/kiss_ai.git && cd kiss_ai && ./install.sh
```

### Python package install (Python 3.13+)

```bash
pipx install kiss-agent-framework
# or
uv tool install kiss-agent-framework
```

### Docker

```bash
./sorcar-docker [PORT] [--rebuild]
```

### Configure model access

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
```

Default and fast model priority order: Anthropic (`claude-opus-5-5` / `claude-sonnet-5-5`), OpenAI (`gpt-6.1-sol-medium` / `gpt-6-luna`), Gemini (`gemini-3.8-flash` / `gemini-3.5-flash-lite`), OpenRouter, Amazon Bedrock (`bedrock/*`), Claude Code CLI (`cc/opus` / `cc/haiku`), and Codex CLI (`codex/default`). Also configurable via the Settings panel (API keys, Custom Models in `~/.kiss/MY_MODELS.json`, Bedrock, Claude Code `cc/*` subscription billing, Codex `codex/*` subscription billing). Six bundled router SEAs (`autorouter`, `bestrouter`, `orautorouter`, `orbestrouter`, `ccxbestrouter`, `brbestrouter`) appear under `Router` in the model picker. See [docs/installation.md](docs/installation.md).

## § 03 — Using KISS Sorcar

Three client interfaces backed by the `kiss-web` local daemon:

- **Web and mobile app** (`http://127.0.0.1:8787` or remote `cloudflared` HTTPS tunnel): `@` file mentions, automatic git worktree isolation, `checkpoint(name)`/`rollback(name)`, voice chat ("Hey Sorcar") and composer dictation mic button, live steering and `<task>…</task>` queueing, live Browser tab, and `/cron` schedules.
- **Python client API (`kiss.server.sorcar.run`)**:
  ```python
  from kiss.server import sorcar

  result = sorcar.run("Refactor utils.py for clarity.")
  sorcar.run("Summarise this codebase.", model="claude-sonnet-5-5", max_budget=2.0)
  sorcar.run("Now fix the typos you found.", chat_id=result.chat_id)
  ```
- **Standalone `sorcar` terminal command**: `sorcar -t "Summarize README.md"` or `sorcar -f task.txt`.
- **Sorcar Extension Agents (SEAs) & slash commands**: single-file Python agents subclassing `BaseSea`, `WorkerSea`, or `ChannelSea` from `kiss.agents.seas.base.base_sea`, registered via `$KISS_HOME/SEAS.md`, validated with `/<name> check` and `sea lint`, and invoked via `/<name>` or `run_agent()`. Includes bundled commands such as `/ask`, `/sh`, `/sync`, `/discover`, `/gepa`, `/write_paper`, `/review_paper`, `/remember`, `/forget`, `/timeline`, routers, and 44 channel/service commands. See [docs/sea-commands.md](docs/sea-commands.md) and [docs/cli.md](docs/cli.md).
- **Persistent memory, Skills, MCP servers, and promptlet tricks**: seven `memory_*` tools backed by Markdown files under `~/.kiss/memories` and a SQLite vector index; Agent Skills with `skills.sh` security checks; MCP servers with OAuth and `~/.kiss/mcp.lock`; personal promptlet tricks in `~/.kiss/MY_INJECTION.md`.

## § 04 — Messaging & Third-Party Agents

44 third-party agents (32 messaging-channel agents, 10 service agents, plus `kiss-a2a` and `kiss-oai` infrastructure agents) and a Govee smart-home CLI. Sign-in runs on demand via each connector's `check_<service>_auth`/`authenticate_<service>` tools or the Connectors panel (Composio OAuth for the six Google Workspace agents; public OAuth device-code/PKCE for GitHub, Teams, Slack, and Discord; Browser tab completion). On Linux, credentials for the 18 Muse-covered connectors are isolated behind a local auth daemon (`python -m kiss.agents.third_party_agents.muse_auth`; opt out with `KISS_MUSE_AUTH=0`). See [docs/messaging-agents.md](docs/messaging-agents.md).

## § 05 — Models Supported

**993 models** across **7 provider categories**: OpenAI (99), Anthropic (82), Amazon Bedrock (144), Gemini (61), OpenRouter (515), Claude Code CLI (79), Codex CLI (13). Capability totals: **941** generation, **807** function-calling, **6** embedding, **38** decision models. Supports vendor effort aliases (`-low`/`-medium`/`-high`/`-xhigh`/`-max`) and live cost/budget tracking. See [docs/models.md](docs/models.md).

## § 06 — Contributing

Contributions in the form of issues are welcome, and KISS Sorcar can help implement and review them: <https://github.com/ksenxx/kiss_ai/issues>.

## § 07 — License

Apache-2.0 — <https://github.com/ksenxx/kiss_ai/blob/main/LICENSE>.

## § 08 — Citation

```bibtex
@misc{sen2026kisssorcar,
  title         = {KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant},
  author        = {Sen, Koushik},
  year          = {2026},
  eprint        = {2604.23822},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2604.23822}
}
```
