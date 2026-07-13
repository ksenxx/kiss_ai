# KISS Sorcar CLI Reference

> The `sorcar` CLI runs in interactive (Claude-Code-style REPL) and non-interactive (one-shot task) modes, and includes a `sorcar mcp` subcommand for managing Model-Context-Protocol servers.

## Modes

`sorcar` runs in two modes:

- **Interactive** (no `-t/--task` or `-f/--file`) — a Claude-Code-style REPL that connects as a thin terminal client to the local `sorcar web` daemon. Chat-session control (new chat, resume by id, list history) and worktree merge/discard prompts are driven from slash commands. Each task is isolated in a git worktree by default.
- **Non-interactive** (`-t` or `-f` supplied) — runs a plain `SorcarAgent` once on the supplied task and exits. Worktree isolation and chat-session control are unavailable in this mode; display events are still streamed into the chat DB so the run is replayable in the chat webview.

## Examples

```bash
# Launch the interactive Sorcar CLI, similar to Claude Code.
sorcar

# Run a one-shot task (non-interactive).
sorcar -t "What is 2435*234?"

# Use a specific model.
sorcar -m "claude-sonnet-4-6" -t "What is 2435*234?"

# Custom endpoint and headers for a local or self-hosted model.
sorcar -e "http://localhost:8000/v1" --header "Authorization:Bearer xxx" \
       -t "Summarize this codebase."

# Cap spend at $2 and pin the working directory.
sorcar -b 2.0 -w "$HOME/projects/my-repo" -t "Refactor utils.py for clarity."

# Use the contents of a file as the task.
echo "Can you find the cheapest non-stop flight from SFO to JFK on June 15?" > prompt
sorcar -f prompt

# Disable browser/web tools (terminal-only mode).
sorcar --no-web -t "Lint and fix every Python file under src/."

# Disable parallel sub-agents for a deterministic single-thread run.
sorcar --no-parallel -t 'Run pytest and report which tests fail and why.'

# Ask Sorcar to use desktop/browser/messaging tools.
sorcar -t 'Can you send the message "Hello from Sorcar!" to ksen via the desktop Slack app?'

# Ask Sorcar to explain code.
sorcar -t 'Can you show me the detailed step-by-step workflow of gepa.py?'

# Manage MCP servers.
sorcar mcp list --ping

# Print the installed sorcar version and exit.
sorcar --version
```

## CLI Options

| Flag | Description |
|------|-------------|
| `-V`, `--version` | Print `sorcar <version>` (from `kiss.__version__`) and exit |
| `-t`, `--task` | Task description; switches to non-interactive mode |
| `-f`, `--file` | Path to a file whose contents are used as the task; switches to non-interactive mode |
| `-m`, `--model_name` | LLM model name; defaults to the best available model for the configured API keys |
| `-e`, `--endpoint` | Custom base URL for a local or self-hosted model |
| `--header` | Custom HTTP header in `Key:Value` form; may be repeated |
| `-b`, `--max_budget` | Maximum spend in USD for the run |
| `-w`, `--work_dir` | Working directory; defaults to the directory where `sorcar` is launched |
| `-v`, `--verbose` | Print Rich panels to the console (`true` by default; pass `false` for quiet mode) |
| `-p`, `--parallel` / `--no-parallel` | Enable/disable parallel sub-agents (enabled by default) |
| `--worktree` / `--no-worktree` | Interactive only. Isolate each task in a git worktree branch (enabled by default); use `--no-worktree` to run directly in the working tree |
| `--auto-commit` / `--no-auto-commit` | Interactive only. Auto-commit worktree changes when a task finishes (enabled by default); use `--no-auto-commit` to preserve the worktree for manual review |
| `--no-web` | Disable browser/web tools (terminal-only mode) |

`--worktree` / `--no-worktree` / `--auto-commit` / `--no-auto-commit` are rejected with `exit 2` when combined with `-t`/`-f`, since the non-interactive path runs a bare `SorcarAgent` that does not implement them. Argparse prefix abbreviations are disabled, so each flag must be spelled out in full.

## Interactive CLI Features

- `@` file/folder mentions with ranked project-file completion.
- Slash commands: `/help`, `/clear` (alias `/new`), `/resume`, `/model`, `/model list`, `/cost` (aliases `/usage`, `/context`), `/commands`, `/skills`, `/mcp`, `/autocommit`, `/voice` (toggle wake-word voice chat), and `/exit` (alias `/quit`).
- Custom Markdown slash commands loaded from `~/.kiss/commands`, `<project>/.kiss/commands`, `~/.claude/commands`, and `<project>/.claude/commands`.
- Agent Skills loaded from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills`, and bundled Sorcar skills.
- MCP server discovery from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, and `<project>/.mcp.json`.
- VS Code "Tricks" button entries read from `~/.kiss/INJECTIONS.md` (one per `## Trick` section), seeded on install from the bundled `src/kiss/INJECTIONS.md`.
- VS Code welcome-screen sample-task chips are the concatenation of `~/.kiss/MY_TASK_TEMPLATES.md` (your personal tasks) and the bundled `src/kiss/SAMPLE_TASKS.md`.

## `sorcar mcp` Subcommand

Manage Model-Context-Protocol servers used by Sorcar:

| Subcommand | Purpose |
|---|---|
| `sorcar mcp add <name> <cmd…>` | Register a stdio (default) or `--transport http`/`sse` server in `--scope user` (`~/.kiss/mcp.json`) or `--scope project` (`<work_dir>/.kiss/mcp.json`); supports `--env KEY=VALUE` and `--header 'Key: Value'` (repeatable) |
| `sorcar mcp list [--ping]` | List configured servers; `--ping` also connects and reports live status and tool counts |
| `sorcar mcp get <name>` | Print one server's configuration as JSON |
| `sorcar mcp remove <name>` | Delete a server from every writable config file |
| `sorcar mcp auth <name> [--no-browser]` | Run the OAuth 2.1 browser flow (dynamic client registration + PKCE) and persist tokens under `~/.kiss/mcp_auth/` |
| `sorcar mcp logout <name>` | Delete a server's stored OAuth tokens |
| `sorcar mcp debug <name>` | Connect and dump capabilities, tools (with input schemas and permission status), resources, and prompts |
