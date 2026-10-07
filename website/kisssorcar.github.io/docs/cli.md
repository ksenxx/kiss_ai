# KISS Sorcar Client Interfaces

> KISS Sorcar is used through three client interfaces, all served by one local daemon (`kiss-web`): the VS Code extension, the remote web/mobile app, and the Python client API `kiss.server.sorcar.run`. A fourth interface, the `sorcar` terminal command, runs a SorcarAgent directly in the current directory without the daemon.

Paths below written as `$KISS_HOME/...` are under the Sorcar home directory: `~/.kiss` for the stock brand, `~/.s10s` once Seamless Loop is installed, or wherever the `KISS_HOME` environment variable points.

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

- `@` file/folder mentions with ranked file completion over the working directory (`./path` items) and the rest of your home directory (`~/path` items), served from a persistent file index that is refreshed in the background. In a tab that is neither running a task nor waiting for an answer, a one-line message that is just the path of an existing file opens that file in the editor or a content tab instead of starting a task.
- Per-task **git worktree isolation** — worktrees are pre-warmed in the background for fast task start, with auto-commit and merge on success, or an interactive merge/discard prompt — toggle both in the Settings panel.
- A pre-run **task classifier** that detects whether the task could create or modify any file in the current git repository (code, docs, reports, presentations, data — anything that could become git-tracked) and so needs a worktree — a hard rule whose only exception is a git-only task (commit, merge, rebase, resolving merge conflicts, ...); tasks that write no files (questions, Internet answers given in the reply) and git-only tasks skip worktree isolation, and simple tasks get a lite system prompt for faster starts. With an `OPENROUTER_API_KEY` and "Use Jev (decisions model)" ticked it asks the `~typesafe/jev-latest` decisions model one typed question (about 0.2 s and $0.00003 per task; on a 415-prompt benchmark it matched hand labels more often than the LLM classifiers, see `benchmarkings/task_classifier/`); otherwise, or if that call fails, it falls back to one fast non-agentic call on the run's own model (structured output, with one plain-text retry if that fails; skipped for `cc/*` and `codex/*` models). Both are Settings-panel checkboxes: "Classify tasks before running" (`classify_tasks`) and "Use Jev (decisions model)" (`classify_with_decisions`, on by default). The Jev checkbox is the one switch for the decisions model: ticked, the classifier asks Jev first and every agent gets the `decide` tool; unticked, the LLM classifier answers and no task calls Jev.
- A model picker, per-task budget caps, chat history with resume (filtered to the current workspace by default), an agent dashboard (burger menu, bottom-left), and inline rendering of tool-generated images in the chat panels.
- **Image and PDF attachments**: attach files to a task via the picker, paste, or drag-and-drop — images (HEIC/HEIF converted, oversized ones re-encoded) and PDFs are sent to the model along with the prompt.
- **Persistent agent memory** (on by default): standard Sorcar runs get seven `memory_*` tools (search, pull, read, write, list, refresh, delete) and a memory protocol, so agents recall lessons, preferences, and decisions across tasks (not for Docker runs, `cc/*`/`codex/*` models, runs that drop the built-in toolset, or runs whose `model_config` supplies its own `system_instruction`). Pages are Markdown files under `$KISS_HOME/memories` with a SQLite vector index (OpenAI embeddings when an `OPENAI_API_KEY` is available, otherwise a fully offline hashed embedder). Toggle it in the Settings panel or set `KISS_USE_MEMORY=0`; a custom directory goes in the `memory_dir` key of `$KISS_HOME/config.json`. A run whose working directory is inside a git repository also gets that repository's own memory, the sub-directory `$KISS_HOME/memories/<repo>/` named after the repository directory (shared by a main checkout and its linked worktrees; independent clones share it only when their directory names match): its pages are addressed as `<repo>/<page>`, `memory_search` covers every memory unless narrowed with `memory=`, and the protocol tells the agent to keep repository knowledge there and cross-project lessons in the general memory.
- Wake-word voice chat ("Hey Sorcar, …") via the mic button, including steering a running agent by voice.
- Live steering: inject a message into a running agent, or switch its model mid-run. Wrapping the message in `<task>…</task>` tags instead queues it as a follow-up task that runs sequentially after the current task finishes.
- Tab mirroring — every VS Code window and web client opened on the same workspace shows the same tabs with the same contents; the tab bar is scoped to the client's workspace directory. A sub-agent dispatched with `run_agent` or `run_parallel` opens a nested tab of its own on every client viewing the parent task, which is closed everywhere when the sub-agent finishes; closing it by hand while the sub-agent runs closes it on every surface until you reopen it from its history row or by collapsing and expanding the fan-out panel.
- In the remote web app, file links in a transcript open in **content tabs**: text files in a Monaco editor, Markdown and HTML with preview and source views, images in the browser's viewer, and PDFs in a bundled pdf.js viewer (fit-to-width pages, pinch or Ctrl/Cmd + wheel zoom, an editable "Page N of M" field, a Download link and keyboard paging) that the VS Code extension also uses for `.pdf` links.
- Scheduled automations: ask in plain language ("every weekday at 9am, summarize my unread Slack messages") and the built-in cron agent (also runnable from the shell as `kiss-cron`) creates, lists, pauses, resumes, or removes the schedule. A job runs an unattended LLM task or a plain shell command and can deliver its result to an authenticated messaging channel (25 of the 32 channels support delivery, e.g. `telegram:123456`, `email:user@example.com`).
- API keys, a custom model endpoint, custom HTTP headers, budget limits, and the remote-access password, all set in the Settings panel.

The remote web app is the same interface served over a cloudflared tunnel: copy the URL and password from the Settings panel and open it on any device. Its desktop mode adds a docked **Task Info sidebar** next to the chat — live token, cost, step, elapsed-time, machine, work-dir, and budget metrics for the visible tab's running task, plus a **Task update**: a short report, written by the bundled `task_update` agent (`src/kiss/agents/seas/task_update/task_update_sea.py`), of what that task has done so far and its partial results. The agent runs when the panel first shows the task, every 10 minutes after that, and whenever you press the refresh button at the top right of the report; it runs as a sub-agent in the task's own chat and its cost counts towards the task.

## Python Client API

Any Python process can run a task on the daemon with `kiss.server.sorcar.run` and block until it finishes (up to `timeout`, one hour by default):

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
| `system_prompt` | Replace the default system prompt for the run (and its sub-agents) |
| `append_to_system_prompt` | Append text to the system prompt instead of replacing it |
| `append_to_prompt` | Append text to the task prompt |
| `sea_path` | Run a full Sorcar Extension Agent (SEA) — a Python file that computes the run's parameters and tools on the daemon (see below); the only way to give the agent extra tools |
| `use_worktree` | Run the task in an isolated git worktree (default `True`) |
| `auto_commit` | Auto-commit the task's changes on success (default `True`) |
| `max_budget` | Per-task budget override in USD |
| `model_config` | Per-task model configuration override (custom endpoint / headers) |
| `use_web_tools` | Per-task browser-tool enablement override (maps to the agent's `web_tools` toggle; `None` uses the daemon's configured default — the settings panel's "Use web tools" checkbox) |
| `classify_tasks` | Per-run override of pre-run task classification: `True` forces it on, `False` skips it, `None` (default) uses the daemon's persisted setting — the settings panel's "Classify tasks before running" checkbox |
| `use_memory` | Per-run persistent-memory override: `True` gives the run (and its `run_parallel` sub-agents) the `memory_*` tools plus the memory protocol, `False` withholds them, `None` (default) uses the daemon's default — a non-empty `KISS_USE_MEMORY` environment variable on the daemon process, else the settings panel's "Use persistent memory" checkbox. The memory safety gates (stripped basic tools, Docker runs, `cc/*`/`codex/*` models, a caller `system_instruction`) always win |
| `is_parallel` | Whether the agent may spawn parallel sub-agents (default `True`) |
| `tool_profile` | Built-in toolset the run is cut down to: `"full"`, `"review"`, `"assistant"`, `"bash"` (Bash only), `"none"` (no built-in tool: `finish` plus a SEA's `add_to_tools()` only) or `+`-joined tool groups such as `"shell+edit+browser"`; `""` (default) is the daemon's usual choice |
| `docker_image` | Run the task's built-in shell and file tools (`Bash`, `run_commands_parallel`, `Read`, `Edit`, `Write`) inside this Docker image; empty runs them on the host |
| `timeout` | How long the client waits for the result — `3600` seconds by default, `None` waits indefinitely; on expiry the client raises `TimeoutError` while the daemon task keeps running |
| `stop_on_timeout` | Also stop the task when `timeout` expires (default `False`) |
| `sock_path` | Daemon Unix-domain-socket path override |

The returned `TaskResult` carries `text`, `success`, `cost`, `tokens`, `steps`, `chat_id`, and `task_id`.

## Sorcar Extension Agents (SEAs)

A **Sorcar Extension Agent (SEA)** is a Python file that defines one class deriving from `kiss.agents.seas.base.base_sea.BaseSea`; you pass the file's path as `sea_path` to `sorcar.run()`. Each SEA lives in its own folder named after it, `<name>/<name>_sea.py`, next to the helper modules and data files it needs (`src/kiss/agents/seas/sh/sh_sea.py`, `src/kiss/agents/third_party_agents/slack/slack_sea.py`, ...). The daemon executes the file on every run, instantiates the class and calls the methods it defines: `description()` (one sentence that says what the SEA does and how to use it; it is what `/<name> help` prints, so every SEA registered as a slash command needs it), `settings(settings)` (the dict that configures the run), `prompt(task)`, `system_prompt(system_prompt)`, `tools(tools)`, and the two call hooks. Every method has an identity default on `BaseSea`, so anything the class does not define keeps whatever the caller passed. One file can define the task prompt, system prompt, model, budget, tools, and safety hooks — a complete custom agent.

- **`settings(self, settings)`.** Receives the dict the base classes built (`{}` for a direct `BaseSea` subclass) and returns the dict to use, normally `settings | {...}`: a `kind` (`session` or `worker`, defaults only), the `channel` flag (an external-service agent; its behaviours are the channel table of [SEA commands](sea-commands.md)) and any of `run()`'s per-run keywords: `work_dir`, `model`, `chat_id`, `use_worktree`, `auto_commit`, `max_budget`, `model_config`, `use_web_tools`, `auto_classify`, `use_memory`, `allow_fan_out`, `tool_profile`, `docker_image` (no other `run()` keyword is accepted), plus two dispatch-side keys: `timeout` (seconds a `run_agent` call waits for this SEA before returning its `agent_job` id while the sub-task keeps running; unrelated to `run()`'s own `timeout`, which bounds the client's wait) and `locked` (keys an explicit `run_agent` / `run_parallel` argument may not change). `scope_work_dir` has no setting (it is the caller's workspace). A kind is only a dict of defaults laid under the explicit keys: `session` (the default) is empty, the run is an ordinary Sorcar session with the caller's or the user's settings; `worker` is `use_worktree`, `auto_commit`, `auto_classify`, `allow_fan_out`, `use_web_tools`, `use_memory` all `False`, a focused tool-bound run on the caller's tree. `channel: True` is not a kind but a flag with fixed behaviours: the SEA is a `worker` whose `work_dir` defaults to `$KISS_HOME/channel_work`, holds its channel workspace, gets the channel preamble, never inherits from a calling task and is never a `run_parallel` child (every bundled channel agent and the cron agent set it). A relative `work_dir`, whether a SEA's setting or a call's option, is a path under the calling task's directory. A `None` value, or `""` for a string key, means "no override". The bundled `/sh` declares `{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}`; `/write` declares `{"timeout": 3600}`; `/ask` declares `{"kind": "worker", "tool_profile": "none", "locked": ["tool_profile"]}` and appends its task framing in `prompt(task)`.
- **Inheritance.** A SEA builds on another by deriving from its class: import the bundled class, or `class MySea(sea_class("bestrouter"))` with `sea_class` from `kiss.agents.sorcar.sea_commands` (a registered command name, or a `.py` path resolved against the directory passed as `relative_to`, normally `Path(__file__).parent`). The launcher applies every class of the inheritance chain that defines a method, base class first, so a method must not call `super()`: `settings()` receives the base's dict, `prompt(task)` the base's prompt, `system_prompt()`, `tools()` and `llm_call_hook()` the base's result, and the first `tool_call_hook()` refusal wins. A `channel: True` SEA is not meant as a base class; the model picked on the tab (`bestrouter`, `autorouter`) is applied before the SEA of every run there (its routing protocol survives a `system_prompt()` that appends, not one that replaces). A file must define exactly one `BaseSea` subclass of its own.
- **Prompts.** Two methods: `prompt(task)` receives the task text (what follows `/xxx`, the `run_agent` task, the `run()` prompt) and returns the prompt body (every `{task_id}` of the final text is replaced by the calling task's id, which is how `/ask` names the task a question is about); `system_prompt(system_prompt)` receives the run's assembled system prompt (the default or the caller's base prompt, plus any text the caller appended) and returns the one to use: `/sh` returns its own text and so replaces it, `/write` returns `system_prompt + "\n\n" + PROTOCOL` and so appends its writing protocol while leaving the base prompt alone.
- **Tools.** `tools(tools)` receives the run's toolset so far (the built-in toolset of its tool profile plus the tools inherited from the calling task) and returns the list of callables (never a file path) the run gets, normally `tools + [mine]`; with `"tool_profile": "none"` in `settings()` it receives `[]` and the agent has exactly the returned tools plus `finish`. The tools execute in the daemon process; nothing is serialized over the socket.
- **Atomic, type-checked settings.** `settings()` and the other methods run in the daemon process and are re-imported from source on every run. Every value is type-checked against the key it fills; an unknown, renamed or removed key, an unknown kind, a wrong type or a raising `settings()` / `prompt()` fails the task before it starts, with a diagnostic naming the source (`settings()['model'] must be str, got int`) in `TaskResult.text`; `system_prompt()`, `tools()` and the call hooks run inside the agent, and a wrong type or exception there fails the task with the same kind of message.

**One precedence rule** decides every run setting (generated from `sea_settings.PRECEDENCE_RULE`):

<!-- sea-docs: precedence -->
> For every setting of a sub-task: what the call passes explicitly (a `run_agent` / `run_parallel` argument or `options` key) wins, then the SEA's `settings()`, then what the calling task passes on (and, for `/<name>`, the chat panel's persisted settings), then the user's defaults. A SEA may list keys in `locked`: a call that passes a different value for a locked key is refused with an error, never silently overruled.
>
> For example, `/sh` declares `{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}`, so `run_agent(agent="sh", task=..., tool_profile="review")` is refused with `Error: sh: the script locks tool_profile='bash' (asked for 'review')`, while `run_agent(agent="sh", task=..., model="gpt-5")` runs it with that model: `model` is not locked, so the explicit argument wins.
<!-- /sea-docs -->

A `/<name>` run in a tab passes nothing explicitly, so the SEA's settings apply over the tab's.

- **Only the class counts.** Module-level functions and constants are ordinary Python the daemon never reads; `prompt` and `system_prompt` are not `settings()` keys (they are methods), so a class naming them in `settings()` fails as an unknown key. `uv run sea lint` flags the names of the earlier module-level contract (`add_to_tools`, `add_to_system_prompt`, the per-field getters) and `uv run sea lint --fix` rewrites renamed settings keys.
- **Hooks.** `llm_call_hook(new_messages)` runs before every LLM call and its return value replaces the outgoing messages; `tool_call_hook(name, args)` runs before every tool call and returns a `Verdict` (`ALLOW` and `refuse(text)` from `kiss.agents.seas.base.base_sea`): `ALLOW` lets the tool execute, `refuse(text)` suppresses the call and *text* is given to the model as the tool's result (`uv run sea lint --fix` rewrites the older `None` / string spellings). Neither has a `run()` equivalent: the daemon stages them as run-time hooks.

```python
# guarded_agent.py — veto dangerous shell commands
from typing import Any

from kiss.agents.seas.base.base_sea import ALLOW, BaseSea, Verdict, refuse


class GuardedSea(BaseSea):
    def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
        if name == "Bash" and "rm -rf" in str(args.get("command", "")):
            return refuse("Blocked: destructive command")
        return ALLOW
```

The full authoring guide is in [`src/kiss/server/README.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/server/README.md).

### Dispatching SEAs from a task with run_agent and run_parallel

A running agent has two tools for sub-agents, both of which open a nested tab on every client viewing the parent task:

- **`run_agent(task, agent="", model="", tool_profile="", max_budget="", timeout="", options="", wait="")`** runs one agent on `task` as a sub-task and blocks until it finishes, returning its YAML result. `agent` is resolved by three rules: empty or a generic label (`"general"`, `"assistant"`, ...; not `"worker"`, which is a kind) runs a plain Sorcar sub-agent (`"reviewer"` is refused: a reviewer is a plain sub-agent with `tool_profile="review"`); a path (`.py` suffix or a path separator; relative paths are anchored at the calling task's work directory) runs that script; a registered command name (`"write_paper"`, `"sh"`, a channel such as `"slack"`, `"cron"`, a `SEAS.md` folder; case, spaces, hyphens and underscores are ignored) runs that command's SEA. Anything else returns an error naming the closest command. `timeout` (seconds) bounds how long the call blocks: it defaults to the SEA's `timeout` setting, else 3600; on expiry the sub-task is not stopped — it keeps running as an `agent_job` and the call returns its job id (a job still running when the calling task ends is killed). `model` and `max_budget` default to the calling task's model and half of its remaining budget (the daemon defaults for a `channel` SEA or an `inherit: false` option); `tool_profile` (`"review"`, `"bash"`, `"shell+edit"`, ...) cuts the sub-task's toolset down. `options` is a JSON object in the SEA settings vocabulary, each key only to override an inherited value: `work_dir` (relative to the calling task's directory), `chat_id`, `workspace` (the account of a multi-account channel; refused for a SEA that is not a `channel`), `tool_profile` (the same as the argument; the two may not differ), `model_config`, `inherit`, `use_worktree`, `auto_commit`, `auto_classify`, `use_web_tools`, `use_memory`, `allow_fan_out`, `docker_image`, `add_to_prompt`, `add_to_system_prompt`; for example `'{"use_web_tools": false}'`. `wait="false"` returns a job id at once; `agent_job(job_id, action)` then waits for the result (`"wait"`), reports the status (`"tail"`) or stops the sub-task (`"kill"`). A sub-task inherits from the calling task its model (and, when the sub-task runs the model this task was launched with and the SEA picks none, its model configuration), half of its remaining budget, its chat, its system-prompt additions and extra tools, its web-tools, memory, fan-out (`allow_fan_out`), Docker, worktree and auto-commit settings, and its work directory; a `channel` SEA inherits none of these, nor does a call with the `inherit: false` option. Precedence is the one rule above. The result starts with a `ran:` line stating the effective configuration (SEA, kind, model, tool profile, budget, timeout, the inherited keys and the inherited or default values the SEA pinned, e.g. `pinned=use_worktree(True->False)`).
- **`run_parallel(tasks, agent="", model="", tool_profile="", max_budget="", timeout="", max_workers="", options="")`** runs several sub-agents concurrently on a JSON array of task strings and returns their results in order, with the same arguments as `run_agent` (`max_budget` and `timeout` are per child: a child still running when its timeout expires is stopped and reports `success: false`; `options` takes the same keys except those a child, a thread of the calling task on its own tree and chat, cannot honour: `use_worktree`, `auto_commit`, `auto_classify`, `chat_id` and `workspace` are refused); `agent` (a SEA path or slash-command name) runs every child as that SEA, in-process, with the same inheritance table as `run_agent`. A channel SEA, or one pinning `use_worktree`, `auto_commit` or `auto_classify` to `true` or a `chat_id`, is refused with an error that says to use `run_agent`; `inherit: false` makes the children take nothing from the caller but their budget share; a SEA's `timeout` bounds each child when the call passes none.

**Slash commands.** Every SEA folder `xxx/xxx_sea.py` in a scanned folder is also a chat command named after the folder: `/xxx some text` runs that file directly in the tab's run on the task "some text" (the SEA's settings, system prompt and tools apply to that session; no relay through `run_agent` and no nested tab), and `/xxx help` prints the SEA's `description()` without running it. A loose `xxx_sea.py` outside its own folder is not registered. Bundled channel agents are registered automatically (`/slack`, `/gmail`, ...); add the parent folders of your own SEAs to `$KISS_HOME/SEAS.md`. See [Slash Commands for SEAs](sea-commands.md).

## Skills, MCP Servers, and Customization

- Agent Skills loaded from `$KISS_HOME/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills`, and bundled Sorcar skills.
- MCP server discovery from `$KISS_HOME/mcp.json`, `<project>/.kiss/mcp.json`, and `<project>/.mcp.json`; OAuth tokens are persisted under `$KISS_HOME/mcp_auth/`. A curated catalog of privacy-first MCP connectors (fetch, time, memory, GitHub, Slack, Google Workspace, WhatsApp, …) ships in [`connectors/`](https://github.com/ksenxx/kiss_ai/blob/main/connectors/README.md) with `enable.py`/`verify.py` CLIs.
- "Tricks" (inject-instruction) entries are the concatenation of two `## Trick`-sectioned Markdown files: `$KISS_HOME/MY_INJECTION.md` (your personal tricks, auto-created on first read and never overwritten thereafter) and the bundled `src/kiss/INJECTIONS.md`, read directly from the package so every upgrade delivers the latest bundled tricks. Edit `$KISS_HOME/MY_INJECTION.md` to customize; your tricks are listed first.
