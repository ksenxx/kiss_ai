# Writing Sorcar Extension Agents (SEAs) for KISS Sorcar

A **Sorcar Extension Agent (SEA)** is a plain Python file whose path you pass as
`extension_agent_path` to `kiss.server.sorcar.run()`.  The daemon
imports the file, calls its top-level `X()` functions — named after
`run()`'s parameters — and uses
the return values to override the run's parameters.  Parameters
without a getter keep whatever the caller passed (or the default).
The getters execute **in the daemon process** and are re-imported from
source on every run (no `__pycache__` is written).

This tutorial covers every overridable parameter, the tools-file
contract, error handling, and ends with a complete working example.

### Prerequisites

- A running `kiss-web` daemon (start one with `kiss-web`).
- At least one model available to the daemon: an LLM provider API key
  (Anthropic, OpenAI, Google, OpenRouter, etc.) or an installed Claude
  Code / Codex CLI executable (`cc/*`, `codex/*` models).
- Any Python packages your SEA imports must be available
  in the daemon's Python environment.


## Quick start

```python
# weather_agent.py — a minimal SEA

import requests

def prompt() -> str:
    return "Look up the current weather in San Francisco and report it."

def max_budget() -> float:
    return 0.50

def use_worktree() -> bool:
    return False  # no repo changes expected

def if_append_basic_tools() -> bool:
    return False  # only finish + our tools

def system_prompt() -> str:
    return (
        "You are a weather assistant. Use the get_weather tool "
        "to look up weather, then call finish with the result."
    )

# --- tools the agent may call ---

def get_weather(city: str) -> str:
    """Return current weather for a city from wttr.in.

    Args:
        city: City name to look up.
    """
    resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
    resp.raise_for_status()
    return resp.text.strip()

def tools() -> list:
    """Return the tools the agent may call."""
    return [get_weather]
```

Launch it:

```python
from kiss.server import sorcar

result = sorcar.run(
    "placeholder",  # required non-blank; overridden by prompt()
    extension_agent_path="weather_agent.py",
)
print(result.text, result.success, result.cost)
```

The client-side `prompt` argument must be non-empty (the client
validates this before connecting to the daemon), but when the agent
script defines `prompt()`, the script's return value replaces it
on the daemon.


## How it works

```
 Client process                          Daemon process
 ──────────────                          ──────────────
 sorcar.run(                             receive "run" JSON command
   prompt=...,                               │
   extension_agent_path="agent.py"           ▼
 )                                       apply_agent_overrides(cmd)
   │                                         │
   │ validate path exists                    │ import agent.py
   │ resolve to absolute                     │ for each PARAM_FIELDS entry:
   │ send JSON {"agentPath": "…", …}        │   if X defined & callable:
   │ over Unix-domain socket                 │     call X()
   │                                         │     type-check return value
   ▼                                         │     stage override
 block, read events ◄───────────────────     │ apply staged overrides to cmd
                                             ▼
                                         load_tools_file(cmd["toolsFile"])
                                             │
                                             ▼
                                         agent.run(tools=client_tools, …)
```

1. **Client side** — `sorcar.run()` validates that `extension_agent_path`
   points to an existing `.py` file, resolves it to an absolute path,
   and sends it as `"agentPath"` on the wire.  `None` or `""` means
   no agent script; any other non-string value (including a
   `pathlib.Path`), a non-`.py` path, or a nonexistent file raises
   `ValueError` immediately (before any daemon connection).

2. **Daemon side** — `apply_agent_overrides()` (in
   `kiss.server.agent_file`) imports the file, iterates every
   overridable parameter, calls `X()` when defined, type-checks
   the return value, and writes the checked value into the command
   dict.  Overrides are staged: they apply atomically only after every
   getter succeeds.  A broken getter raises `AgentFileError` and the
   task fails with a diagnostic message in `TaskResult.text`.

3. **Tools loading** — After overrides, the daemon reads the
   `toolsFile` field and calls `load_tools_file()` to import it and
   invoke its `get_tools()` (or, for an SEA doubling as its own tools
   file, its `tools()`).  The returned callables become the
   agent's tools.


## Overridable parameters

Every parameter of `sorcar.run()` except `timeout`, `stop_on_timeout`,
`sock_path`, `parent_task_id`, `parent_tab_id`, `parent_reviewer`,
`side_channel`, and `extension_agent_path` itself has a corresponding
getter the SEA may define.  The getter is named `X()` for parameter `X`,
except `append_basic_tools`, whose getter is
`if_append_basic_tools()`.  The table below lists them all.

| Getter function              | Return type                     | `run()` default           | Wire field          |
|------------------------------|---------------------------------|---------------------------|---------------------|
| `prompt()`               | `str` (non-empty)               | (required argument)       | `prompt`            |
| `work_dir()`             | `str`                           | `""` (daemon default)     | `workDir`           |
| `model()`                | `str`                           | `""` (daemon default)     | `model`             |
| `chat_id()`              | `str`                           | `""` (new chat)           | `chatId`            |
| `system_prompt()`        | `str`                           | `""` (daemon-selected)    | `systemPrompt`      |
| `tools()`                | `str`, `Path`, `list`, or `None`| `None` (no extra tools)   | `toolsFile`         |
| `use_worktree()`         | `bool`                          | `True`                    | `useWorktree`       |
| `auto_commit()`          | `bool`                          | `True`                    | `autoCommit`        |
| `max_budget()`           | finite `int`/`float` (not `bool`) or `None` | `None` (daemon default) | `maxBudget`  |
| `model_config()`         | `dict` or `None`                | `None`                    | `modelConfig`       |
| `if_append_basic_tools()` | `bool`                         | `True`                    | `appendBasicTools`  |
| `append_to_system_prompt()` | `str`                        | `""` (append nothing)     | `appendToSystemPrompt` |
| `append_to_prompt()`     | `str`                           | `""` (append nothing)     | `appendToPrompt`    |
| `scope_work_dir()`       | `str`                           | `""` (scope = work dir)   | `tabScopeWorkDir`   |
| `use_web_tools()`        | `bool` or `None`                | `None` (daemon default)   | `webTools`          |
| `classify_tasks()`       | `bool` or `None`                | `None` (daemon default)   | `classifyTasks`     |
| `use_memory()`           | `bool` or `None`                | `None` (daemon default)   | `useMemory`         |
| `is_parallel()`          | `bool`                          | `True`                    | `useParallel`       |
| `tool_profile()`         | `str`                           | `""` (daemon's choice)    | `toolProfile`       |
| `docker_image()`         | `str`                           | `""` (host)               | `dockerImage`       |

When a getter is absent, the caller's value is used (which is the
`run()` default when the caller did not pass one).

The parameters without getters:

- **`timeout`** — bounds the *client's* local wait (`None` waits
  indefinitely); the daemon never sees it.
- **`stop_on_timeout`** — whether a `timeout` expiry also stops the
  task, awaiting the stop's confirmation (default `False`: the task
  keeps running); a client-side choice the script must not override.
- **`sock_path`** — selects which daemon to connect to; the script
  already runs on that daemon.
- **`parent_task_id` / `parent_tab_id` / `parent_reviewer`** — the
  CALLING task's identity (how `run_agent` nests a dispatched run under
  its caller) and whether that caller sits in a reviewer sub-tree
  (so the child's `run_parallel` spawns no further reviewers), which a
  dispatched script must not be able to forge.
- **`side_channel`** — marks the run as a side channel of its parent
  (the `/ask` sub-agent whose answer is delivered into the PARENT's
  transcript, so its own tab closes when the run ends); only
  meaningful with `parent_task_id`, and not forgeable for the same
  reason.
- **`extension_agent_path`** — the script cannot override its own path.

### Getter semantics

- **`model()`** — an empty string `""` means "use the daemon's
  configured default model".  A non-empty string must name a model in
  the daemon's available model list or the task fails.
- **`chat_id()`** — an empty string `""` starts a fresh chat.  A
  non-empty string resumes that chat session.
- **`system_prompt()`** — an empty or blank string leaves the base
  prompt to the daemon: with task classification enabled (the
  default) a task classified as simple runs on the reduced
  `SYSTEM_LITE.md`, everything else on the full `SYSTEM.md`.  A
  non-empty string replaces that base prompt.  A
  `model_config()["system_instruction"]` value, if present, takes
  precedence over the composed prompt (`KISSAgent.run` only
  `setdefault`s it).
- **`tools()`** — **overrides** (does not append to) the caller's
  `tools` argument.  Returning `None` clears any caller-supplied tools.
- **`if_append_basic_tools()`** — overrides the
  `append_basic_tools` parameter; `False` strips the run down to
  `finish` plus the supplied tools.
- **`append_to_system_prompt()`** — extra text **appended** to
  the run's system prompt (the daemon-selected base prompt or the
  `system_prompt()` replacement) when the agent is executed.
  Unlike `system_prompt()`, it does not replace anything.
- **`append_to_prompt()`** — extra text **appended** to the
  executed task prompt.  A multi-`<task>` prompt runs the agent once
  per subtask and the text is appended to each subtask's prompt.  The
  appended text becomes part of the recorded prompt in chat history.
- **`scope_work_dir()`** — the workspace directory the run's tab is
  scoped to in clients' tab bars, when different from the execution
  `work_dir`.  An empty string scopes the tab to the run's work
  directory (the default scoping), like an empty client-sent
  `scope_work_dir`.
- **`use_web_tools()`** — per-run browser-tool enablement.  `None`
  falls back to the daemon's configured default (the settings panel's
  "Use web tools" checkbox, persisted as `use_web_browser`).
- **`classify_tasks()`** — per-run pre-run task classification.
  `None` falls back to the daemon's configured default (the settings
  panel's "Classify tasks before running" checkbox, persisted as
  `classify_tasks`).
- **`use_memory()`** — per-run persistent agent memory
  (`kiss.core.memoryfield`): the seven `memory_*` tools plus the
  memory protocol prompt block.  `True` enables, `False` disables,
  `None` falls back to the daemon's configured default (the settings
  panel's "Use persistent memory" checkbox, persisted as
  `use_memory`, or the daemon process's `KISS_USE_MEMORY` environment
  variable).  The override is forwarded to `run_parallel` sub-agents
  and never bypasses the memory safety gates: a run without the basic
  toolset, a Docker run, a run-to-completion CLI model (`cc/*`,
  `codex/*`), or a caller-supplied
  `model_config["system_instruction"]` stays memory-free even with
  `True`.
- **`is_parallel()`** — whether the agent may spawn parallel
  sub-agents (`run_parallel`).
- **`tool_profile()`** — the name of the tool profile the run's
  built-in toolset is cut down to: `"full"` (everything), `"review"`
  (read and run, no editing, browser or dispatch), `"shell"` (`Bash`,
  `bash_job`, `Read`, `run_commands_parallel`) or `"bash"` (`Bash`
  only — the bundled `/sh` agent's choice); `finish` is always added.
  `""` keeps the daemon's usual choice.  An unknown name fails the
  task when it starts.  Ignored when `if_append_basic_tools()` is
  `False`, which builds no built-in toolset at all.
- **`docker_image()`** — the Docker image the run's shell and file
  tools (`Bash`, `run_commands_parallel`, `Read`, `Edit`, `Write`)
  execute in: an image name starts a fresh container that is removed
  when the task ends, `container:<name-or-id>` attaches to a container
  the caller already runs (commands run in its working directory, it
  is left running afterwards), `""` runs the tools on the host.
  `run_parallel` sub-agents share the task's container; `bash_job`
  and persistent memory are unavailable in a Docker run.

### Hook getters (no `run()` parameter)

The script may also define two hook getters with no corresponding
`sorcar.run()` parameter — a callable cannot be JSON-serialized, so
the hooks exist ONLY as agent-script getters, evaluated in the daemon
process:

| Getter function        | Return type          | Staged command field |
|------------------------|----------------------|----------------------|
| `llm_call_hook()`  | callable or `None`   | `llmCallHook`        |
| `tool_call_hook()` | callable or `None`   | `toolCallHook`       |

The returned functions — `llm_call_hook` and `tool_call_hook` — are
passed to the underlying `KISSAgent.run()` of every task-executor
sub-session of the task's agent (internal helper sessions, e.g. the
failed-session trajectory summarizer, and `run_parallel` sub-agents
are not hooked).  Per `KISSAgent.run()`'s contract:

- **`llm_call_hook(new_messages)`** — called before every LLM call
  with the list of new messages (those added to the conversation
  since the previous LLM call) about to be sent; its return value
  replaces those messages.
- **`tool_call_hook(name, args)`** — called before every tool call
  with the tool's name and arguments dict.  Returning `"OK"` lets the
  tool execute; any other returned string suppresses the call and is
  given to the model as the tool's result.

Returning `None` from a getter means "no hook".  Any other
non-callable return value fails the task with a diagnostic error,
like every wrong-typed getter.

```python
# guarded_agent.py
def veto_destructive(name, args):
    if name == "Bash" and "rm -rf" in str(args.get("command", "")):
        return "Blocked: destructive command"
    return "OK"

def tool_call_hook():
    return veto_destructive
```


## Tools: two contracts

An SEA supplies tools to the LLM agent through one of two
approaches.

### 1. Separate tools file (path return)

`tools()` returns the **path** (string or `pathlib.Path`) of
another Python file.  The daemon imports that file and calls its
`get_tools()` (or `tools()`) to obtain the callable list.  Use an
absolute path; the
daemon does not resolve paths against the client's working directory.

```python
# my_agent.py
import pathlib

def tools():
    return pathlib.Path("/absolute/path/to/my_tools.py")
```

```python
# my_tools.py
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First factor.
        b: Second factor.
    """
    return a * b

def get_tools():
    return [multiply]
```

### 2. Self-contained agent (list return)

`tools()` returns a **list of callables** directly.  The daemon
normalizes this to the agent script's own path and later re-imports
the same file as the tools file, calling `tools()` again.  This
makes the SEA its own tools file — a single file provides
both parameter overrides and tools.

Because the file is imported twice per run (once for parameter
overrides, once for tools loading), module-level side effects execute
twice.  Use guards (e.g. `if __name__ == "__main__"`, lazy
initialization, or idempotent setup) if side effects are expensive.

```python
# self_contained_agent.py

def prompt() -> str:
    return "Double the number 21."

def double(n: int) -> int:
    """Double a number.

    Args:
        n: The number to double.
    """
    return n * 2

def tools() -> list:
    return [double]
```

### Tool function requirements

Each tool function must:

1. Have a **name** — the function name becomes the tool name the LLM
   sees.  Names must not collide with built-in tools (e.g. `finish`,
   `Bash`, `Read`) or with each other.
2. Have a **docstring** with a Google-style `Args:` section describing
   each parameter.
3. Use **type-annotated, keyword-bindable parameters** — the daemon
   builds the tool schema from the annotations.
4. Be **callable** — the daemon calls `callable(tool)` on each entry.

```python
def search_database(query: str, max_results: int = 10) -> str:
    """Search the internal database.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    # implementation
    return results
```


## The `append_basic_tools` parameter

By default (`append_basic_tools=True`) the agent gets `finish` (always
present) and the built-in KISS Sorcar toolset — `Bash` (with
`background=True` for detached jobs), `bash_job` (wait for / tail /
kill a background job), `run_commands_parallel` (several shell
commands at once, no LLM sub-agents), `Read`, `Edit`, `Write`,
`ask_user_question`, `talk`, `set_model`, `summary`, `run_agent`,
browser tools (when `use_web_tools`), `run_parallel` and
`number_of_cores` (when `is_parallel`), `decide` (when
`OPENROUTER_API_KEY` is configured), the `memory_*` tools (when
memory is enabled), and any configured skill and MCP-server tools —
**plus** your extension tools.  A restricted tool profile (a reviewer
sub-agent dispatched with `run_parallel(..., tool_profile="review")`)
filters that built-in set, including the memory tools; extension tools
are still appended.

When `append_basic_tools=False`, the agent's **only** tools are
`finish` and the tools from `tools()`.  This is useful for
building focused, restricted agents.

When restricting tools, the full default system prompt (`SYSTEM.md`)
assumes the full toolset (its workflow rules name `Read`, `Edit`,
`Bash` and the browser tools).  Pass a custom `system_prompt()` that
matches the tools you provide:

```python
def if_append_basic_tools() -> bool:
    return False

def system_prompt() -> str:
    return (
        "You are a weather assistant. Use the get_weather tool "
        "to look up weather, then call finish with the result."
    )
```


## Error handling

Errors fall into two categories depending on where they are caught:

**Client-side errors** (raised as `ValueError` before connecting to
the daemon):
- `prompt` is empty or blank
- `tools` is neither `None` nor a `str`/`Path` to an existing `.py` file
- `extension_agent_path` is neither `None`/`""` nor a string
- The agent-script path is not a `.py` file
- The agent-script file does not exist

**Daemon-side errors** (the task starts, then fails with
`result.success == False` and the diagnostic in `result.text`, which
prefixes the message below with `Task failed: AgentFileError: `):

| Condition | `AgentFileError` message |
|-----------|---------------|
| File deleted between client validation and daemon import | `agent script '...' is not an existing Python (.py) file` |
| File raises at import time | `agent script '...' failed to import: ...` |
| `X` defined but not callable | `X of agent script '...' must be a callable, got ...` |
| `X()` raises an exception | `X() of agent script '...' raised: ...` |
| `X()` returns wrong type | `X() of agent script '...' must return ..., got ...` |
| `X()` returns a value whose type check itself raises (e.g. a `str` subclass with a raising `strip`) | `X() of agent script '...' returned a broken value: ...` |

Overrides are **atomic**: if any getter fails, the command keeps all
its original values (no partial overrides).


## Continuing chat sessions

Pass `chat_id` to continue an existing daemon chat session.  The agent
receives the chat's retained top-level context as a prefix, not
necessarily every prior row: sub-agent rows are excluded, a history
longer than ten entries keeps the first two and the latest eight, and
with the default history digest older entries are shortened and only the
newest two task/result pairs stay whole while the prefix fits in 6,000
characters (an oversized history drops older entries first, then digests
the newest ones too, then hard-truncates).

```python
result1 = sorcar.run(
    "Analyze the codebase",
    extension_agent_path="my_agent.py",
)

# Continue the same chat
result2 = sorcar.run(
    "Now fix the issues you found",
    chat_id=result1.chat_id,
    extension_agent_path="my_agent.py",
)
```

An SEA can also force a specific chat via `chat_id()`.


## Model configuration

Use `model_config()` to pass custom endpoint URLs, headers, or
sampling parameters:

```python
def model() -> str:
    return "my-custom-model"

def model_config() -> dict:
    return {
        "base_url": "http://localhost:8080/v1",
        "api_key": "sk-local-key",
    }
```

When `model_config` contains a `base_url`, the model factory bypasses
its normal provider routing and creates an OpenAI-compatible model
pointing at that URL.  The daemon still runs its model-availability
preflight first: `model()` must return a generation-capable name from
the bundled catalog or from `~/.kiss/MY_MODELS.json` whose provider is
usable (an API key for HTTP providers, the executable on `PATH` for
`cc/*` / `codex/*`), so replace `my-custom-model` above with such a
name; otherwise the task fails with `No model available.  Set at least
one API key in the environment.`


## Complete working example

Below is a self-contained SEA that gives the LLM tools for
managing a SQLite task database.  It uses the full basic toolset
(`append_basic_tools` defaults to `True`), so the LLM can also use
`Bash`, `Read`, `Write`, etc. alongside the custom database tools.

The agent does **not** define `prompt()` or `model()`, so
the caller's prompt reaches the LLM and the daemon's configured
default model is used.

```python
# task_manager_agent.py
"""SEA for managing a SQLite task database.

Gives the LLM three tools — add_task, list_tasks, complete_task — and
a system prompt explaining how to use them.  The agent runs with the
full KISS Sorcar toolset so it can also read files, run commands, etc.
"""

import json
import os
import sqlite3
import threading

# --- Database setup ---
# Guarded with CREATE IF NOT EXISTS so the double-import of a
# self-contained agent (parameter overrides + tools loading) is safe.

_DB_PATH = os.path.expanduser("~/.kiss/task_manager.db")
_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    """Return a connection to the task database, creating it if needed."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


# --- Overridable parameter getters ---
# No prompt() — the caller's prompt is used as-is.
# No model() — the daemon's configured default model is used.

def max_budget() -> float:
    return 1.0


def use_worktree() -> bool:
    return False  # no code changes expected


def system_prompt() -> str:
    return (
        "You manage a personal task list stored in a SQLite database.  "
        "Use the add_task, list_tasks, and complete_task tools to "
        "manipulate the database.  Always call list_tasks after "
        "modifications so the user sees the updated state.  "
        "You also have the standard KISS Sorcar tools (Bash, Read, "
        "Write, etc.) if you need them."
    )


# --- Tool functions ---

def add_task(title: str) -> str:
    """Add a new task to the database.

    Args:
        title: Short description of the task.
    """
    with _lock:
        conn = _get_db()
        conn.execute("INSERT INTO tasks (title) VALUES (?)", (title,))
        conn.commit()
        task_id = conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]
        conn.close()
    return json.dumps({"added": {"id": task_id, "title": title}})


def list_tasks(include_done: bool = False) -> str:
    """List tasks from the database.

    Args:
        include_done: If True, include completed tasks.
    """
    with _lock:
        conn = _get_db()
        if include_done:
            rows = conn.execute(
                "SELECT id, title, done, created_at FROM tasks "
                "ORDER BY created_at"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, done, created_at FROM tasks "
                "WHERE done = 0 ORDER BY created_at"
            ).fetchall()
        conn.close()
    tasks = [
        {"id": r[0], "title": r[1], "done": bool(r[2]),
         "created_at": r[3]}
        for r in rows
    ]
    return json.dumps({"tasks": tasks, "count": len(tasks)})


def complete_task(task_id: int) -> str:
    """Mark a task as completed.

    Args:
        task_id: The numeric ID of the task to complete.
    """
    with _lock:
        conn = _get_db()
        cursor = conn.execute(
            "UPDATE tasks SET done = 1 WHERE id = ? AND done = 0",
            (task_id,),
        )
        conn.commit()
        updated = cursor.rowcount
        conn.close()
    if updated:
        return json.dumps({"completed": task_id})
    return json.dumps({"error": f"Task {task_id} not found or already done"})


def tools() -> list:
    """Return the tools the agent may call."""
    return [add_task, list_tasks, complete_task]
```

Launch it:

```python
from kiss.server import sorcar

# First run — add some tasks (the prompt reaches the LLM directly)
result = sorcar.run(
    "Add three tasks: buy groceries, review PR #42, write tests",
    extension_agent_path="task_manager_agent.py",
)
print(result.text)
print(f"Cost: ${result.cost:.4f}, Steps: {result.steps}")

# Follow-up in the same chat — the agent remembers context
result2 = sorcar.run(
    "Complete 'buy groceries' and show me remaining tasks",
    chat_id=result.chat_id,
    extension_agent_path="task_manager_agent.py",
)
print(result2.text)
```


## Reference: `sorcar.run()` signature

```text
def run(
    prompt: str,
    *,
    work_dir: str = "",
    scope_work_dir: str = "",
    parent_task_id: str = "",
    parent_tab_id: str = "",
    parent_reviewer: bool = False,
    side_channel: bool = False,
    model: str = "",
    chat_id: str = "",
    system_prompt: str = "",
    tools: str | Path | None = None,
    extension_agent_path: str = "",
    use_worktree: bool = True,
    auto_commit: bool = True,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    use_web_tools: bool | None = None,
    classify_tasks: bool | None = None,
    use_memory: bool | None = None,
    is_parallel: bool = True,
    append_basic_tools: bool = True,
    append_to_system_prompt: str = "",
    append_to_prompt: str = "",
    tool_profile: str = "",
    docker_image: str = "",
    timeout: float | None = 3600.0,
    stop_on_timeout: bool = False,
    sock_path: str | Path | None = None,
) -> TaskResult
```

```python
@dataclass(frozen=True)
class TaskResult:
    text: str         # human-readable result summary
    success: bool     # whether the agent reported success
    cost: float       # budget consumed in USD
    tokens: int       # total LLM tokens consumed
    steps: int        # total agent steps taken
    chat_id: str = ""  # daemon chat session id (for continuation)
    task_id: str = ""  # persisted task_history row id
```


## SEA vs. tools file

| Aspect | SEA (`extension_agent_path`) | Tools file (`tools`) |
|--------|------------------------------------------|----------------------|
| **Purpose** | Override run parameters AND supply tools | Supply tools only |
| **Getter functions** | `prompt()`, `model()`, `system_prompt()`, `tools()`, etc. (20 total, plus 2 hooks) | `get_tools()` (or `tools()`) only |
| **Required function** | None — define only the getters you need | Must define `get_tools()` (or `tools()`) |
| **Can be combined** | Yes — `tools()` can point to a separate tools file | N/A |
| **Can be self-contained** | Yes — return a list from `tools()` and the script becomes its own tools file | Always self-contained |


## Tips

- The client-side `prompt` argument must be **non-empty** even when
  `prompt()` overrides it; the client validates before connecting.
- An SEA **may define any subset** of the getters.  Only
  define the ones whose defaults you want to change.
- Omit `model()` to use the daemon's configured default model
  rather than hard-coding one.
- The `tools()` return value of a **list** makes the SEA its
  own tools file.  This is the most common pattern.
- **Use absolute paths** for the `tools()` path return — the
  daemon does not resolve paths against the client's working directory.
- `tools()` **overrides** the caller's `tools` argument; it does
  not append to it.  Returning `None` clears caller-supplied tools.
- The agent script is **re-imported from source** on every run.
  Edits take effect immediately without restarting the daemon.
- A self-contained agent (list-returning `tools()`) is imported
  **twice** per run: once for parameter overrides, once for tools
  loading.  Keep module-level side effects idempotent.
- `max_budget()` must return a **finite** number.  `NaN`,
  `±inf`, or an overflowing value raises `AgentFileError`.
- A getter defined as a non-callable (e.g. a module-level variable
  named `model` or `tools`) is treated as a broken getter and stops
  the task — it is not treated as "absent".  Avoid module-level
  variables that share a getter's name.
- The SEA and its tools run **in the daemon process**
  with the daemon user's privileges and environment.  Any libraries
  your code imports must be installed in the daemon's Python
  environment.
- Name the file `xxx_sea.py` and put its folder in `~/.kiss/SEAS.md`
  (one folder per line; blank lines and `#` comments are ignored) to
  expose it as the chat command `/xxx`; `/xxx some text` runs the SEA
  on "some text" via `run_agent`.  Bundled
  `src/kiss/agents/third_party_agents/*_sea.py` scripts take
  precedence over `SEAS.md` folders, later `SEAS.md` lines beat
  earlier ones, and the bundled Sorcar-extending SEAs in
  `src/kiss/agents/seas/` (`/merge`, `/sh`, `/skillopt`,
  `/task_update`; `dummy_sea.py`, an SEA with no getters, is what
  `run_agent` runs when its `agent` argument is empty) have the
  lowest precedence, so a `SEAS.md` folder can shadow them.  Syntax,
  precedence and the dispatch flow are
  documented in
  [docs/sea-commands.md](https://kisssorcar.github.io/docs/sea-commands.md)
  (source: `website/kisssorcar.github.io/docs/sea-commands.md`).
- The outer run of a `/xxx` command is only a relay that calls
  `run_agent`; before it starts, for each relay setting that is enabled
  (worktree isolation, auto-commit) the daemon imports the resolved SEA
  and checks the matching `use_worktree()` or `auto_commit()` getter.
  An exact `False` demotes that setting on the relay as well (the SEA is
  not handed the relay's worktree, and the relay does not auto-commit
  what the SEA left in the tree); an import or getter exception during
  such a check fails the relay with `SeaScriptError`, so keep
  module-level side effects idempotent.
