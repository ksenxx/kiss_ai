# KISS Sorcar Client Interfaces

> KISS Sorcar is used through three client interfaces, all served by one local daemon (`kiss-web`): the VS Code extension, the remote web/mobile app, and the Python client API `kiss.server.sorcar.run`. A fourth interface, the `sorcar` terminal command, runs a SorcarAgent directly in the current directory without the daemon.

## The `sorcar` Terminal Command

The standalone `sorcar` command runs a task without the daemon: `sorcar -t "Summarize README.md"` runs an inline task, and `sorcar -f task.txt` runs the file's content as the task (exactly one of `-t`/`-f` is required; see `sorcar --help` for the model, budget, and work-dir flags).

## The `kiss-web` Daemon

The `kiss-web` daemon hosts the agents, chat sessions, and the web app, and services every client command — including config reads/writes, default-model lookup, and the wake-word listener — over its socket. The VS Code extension starts it automatically; you can also manage it yourself:

```bash
# Start the daemon (serves the web app and the extension).
kiss-web

# Pin the daemon's working directory.
kiss-web --workdir "$HOME/projects/my-repo"

# Print the active remote (cloudflared) URL and exit.
kiss-web --url
```

| Flag | Description |
|------|-------------|
| `--workdir` | Working directory for the daemon |
| `--url` | Print the active remote URL and exit |

## VS Code Extension and Web/Mobile App

Open the KISS Sorcar sidebar in VS Code (or the remote web app in a browser) and type or speak your task. The chat interface provides:

- `@` file/folder mentions with ranked project-file completion.
- Per-task **git worktree isolation** — worktrees are pre-warmed in the background for fast task start, with auto-commit and merge on success, or an interactive merge/discard prompt — toggle both in the Settings panel.
- A pre-run **task classifier** (a single fast non-agentic model call — structured output with a plain-text fallback; skipped for `cc/*` and `codex/*` models) that detects whether the task is a development task — non-development tasks (questions, research, git-only operations) skip worktree isolation, and simple tasks get a lite system prompt for faster starts. Toggleable in the Settings panel.
- A model picker, per-task budget caps, chat history with resume (filtered to the current workspace by default), an agent dashboard (burger menu, bottom-left), and inline rendering of tool-generated images in the chat panels.
- Wake-word voice chat ("sorcar, …") via the mic button, including steering a running agent by voice.
- Live steering: inject a message into a running agent, or switch its model mid-run. Wrapping the message in `<task>…</task>` tags instead queues it as a follow-up task that runs sequentially after the current task finishes.
- Tab mirroring — every VS Code window and web client opened on the same workspace shows the same tabs with the same contents; the tab bar is scoped to the client's workspace directory, and sub-agents dispatched with `run_agent` open their own tab in the calling workspace.
- Scheduled automations: ask in plain language ("every weekday at 9am, summarize my unread Slack messages") and the built-in cron agent (also runnable from the shell as `kiss-cron`) creates, lists, pauses, resumes, or removes the schedule. A job runs an unattended LLM task or a plain shell command and can deliver its result to an authenticated messaging channel (25 of the 32 channels support delivery, e.g. `telegram:123456`, `email:user@example.com`).
- API keys, a custom model endpoint, custom HTTP headers, budget limits, and the remote-access password, all set in the Settings panel.

The remote web app is the same interface served over a cloudflared tunnel: copy the URL and password from the Settings panel and open it on any device.

## Python Client API

Any Python process can run a task on the daemon with `kiss.server.sorcar.run` and block until it finishes:

```python
from kiss.server import sorcar

result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
print(result.text, result.success, result.cost, result.tokens, result.steps)

# Continue the same chat (the agent sees the prior task as context):
follow_up = sorcar.run("Now fix the typos you found", chat_id=result.chat_id)
```

Keyword options:

| Option | Description |
|--------|-------------|
| `work_dir` | Working directory for the task; the daemon's default when empty |
| `scope_work_dir` | Workspace-scope directory for the task's tab; the task's working directory when empty |
| `parent_task_id` | Task-history row id of the calling task; non-empty marks the run as a sub-agent of that task (nested tab and history row) — how the `run_agent` tool dispatches |
| `parent_tab_id` | Frontend tab id of the calling task's tab, so the webview knows which tab spawned the sub-agent |
| `model` | Model name; the daemon's selected default when empty |
| `chat_id` | Existing chat session id to continue; a new chat when empty |
| `tools` | Path to a Python file whose `get_tools()` function returns the functions the daemon registers as extra agent tools |
| `system_prompt` | Replace the default system prompt for the run (and its sub-agents) |
| `append_to_system_prompt` | Append text to the system prompt instead of replacing it |
| `append_to_prompt` | Append text to the task prompt |
| `append_basic_tools` | Set `False` to restrict the agent to `finish` plus your `tools` file, dropping the built-in toolset (default `True`) |
| `extension_agent_path` | Run a full Sorcar Extension Agent (SEA) — a Python file that computes the run's parameters and tools on the daemon (see below) |
| `use_worktree` | Run the task in an isolated git worktree (default `True`) |
| `auto_commit` | Auto-commit the task's changes on success (default `True`) |
| `max_budget` | Per-task budget override in USD |
| `model_config` | Per-task model configuration override (custom endpoint / headers) |
| `use_web_tools` | Per-task browser-tool enablement override (maps to the agent's `web_tools` toggle; `None` uses the daemon's configured default — the settings panel's "Use web tools" checkbox) |
| `classify_tasks` | Per-run override of pre-run task classification: `True` forces it on, `False` skips it, `None` (default) uses the daemon's persisted setting — the settings panel's "Classify tasks before running" checkbox |
| `is_parallel` | Whether the agent may spawn parallel sub-agents (default `True`) |
| `timeout` | Maximum seconds to wait for the task to finish (default `3600`) |
| `stop_on_timeout` | Also stop the task when `timeout` expires; default `False` — the task keeps running |
| `sock_path` | Daemon Unix-domain-socket path override |

The returned `TaskResult` carries `text`, `success`, `cost`, `tokens`, `steps`, `chat_id`, and `task_id`.

## Sorcar Extension Agents (SEAs)

A **Sorcar Extension Agent (SEA)** is a plain Python file whose path you pass as `extension_agent_path` to `sorcar.run()`. The daemon imports the file on every run and calls its top-level `X()` functions — named after `run()`'s parameters — to compute the run's parameters; parameters without a getter keep whatever the caller passed. One file can define the task prompt, system prompt, model, budget, tools, and safety hooks — a complete custom agent.

- **Overridable parameters.** Every `sorcar.run()` parameter except `timeout`, `stop_on_timeout`, `sock_path`, `parent_task_id`, `parent_tab_id`, and `extension_agent_path` itself has a getter named after it: `prompt()`, `work_dir()`, `model()`, `chat_id()`, `system_prompt()`, `tools()`, `use_worktree()`, `auto_commit()`, `max_budget()`, `model_config()`, `if_append_basic_tools()` (overrides `append_basic_tools`), `append_to_system_prompt()`, `append_to_prompt()`, `scope_work_dir()`, `use_web_tools()`, `classify_tasks()`, and `is_parallel()`. `use_web_tools()` and `classify_tasks()` return a bool, or `None` to fall back to the daemon's persisted setting.
- **Atomic, type-checked overrides.** Getters run in the daemon process and are re-imported from source on every run. Each return value is type-checked; overrides apply only after every getter succeeds, and a broken getter fails the task with a diagnostic in `TaskResult.text`.
- **Tools, two ways.** `tools()` may return a list of callables — making the script its own tools file — or the path of a separate Python file whose `get_tools()` (or `tools()`) returns the callables. Either way the tools execute in the daemon process; nothing is serialized over the socket. `tools()` overrides (does not append to) the caller's `tools` argument.
- **Hook getters.** `llm_call_hook()` and `tool_call_hook()` return functions with no `run()` equivalent. `llm_call_hook(new_messages)` runs before every LLM call and its return value replaces the outgoing messages; `tool_call_hook(name, args)` runs before every tool call — returning `"OK"` lets the tool execute, any other string suppresses the call and is given to the model as the tool's result.

```python
# guarded_agent.py — veto dangerous shell commands
def veto_destructive(name, args):
    if name == "Bash" and "rm -rf" in str(args.get("command", "")):
        return "Blocked: destructive command"
    return "OK"

def tool_call_hook():
    return veto_destructive
```

The full authoring guide is in [`src/kiss/server/README.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/server/README.md).

## Skills, MCP Servers, and Customization

- Agent Skills loaded from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills`, and bundled Sorcar skills.
- MCP server discovery from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, and `<project>/.mcp.json`; OAuth tokens are persisted under `~/.kiss/mcp_auth/`. A curated catalog of privacy-first MCP connectors (fetch, time, memory, GitHub, Slack, Google Workspace, WhatsApp, …) ships in [`connectors/`](https://github.com/ksenxx/kiss_ai/blob/main/connectors/README.md) with `enable.py`/`verify.py` CLIs.
- "Tricks" (inject-instruction) entries are the concatenation of two `## Trick`-sectioned Markdown files: `~/.kiss/MY_INJECTION.md` (your personal tricks, auto-created on first read and never overwritten thereafter) and the bundled `src/kiss/INJECTIONS.md`, read directly from the package so every upgrade delivers the latest bundled tricks. Edit `~/.kiss/MY_INJECTION.md` to customize; your tricks are listed first.
- Welcome-screen sample-task chips are the concatenation of `~/.kiss/MY_TASK_TEMPLATES.md` (your personal tasks) and the bundled `src/kiss/SAMPLE_TASKS.md`.
