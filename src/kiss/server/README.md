# Writing Extension Agents for KISS Sorcar

An **extension agent** is a plain Python file whose path you pass as
`extension_agent_path` to `kiss.server.sorcar.run()`.  The daemon
imports the file, calls its top-level `get_X()` functions, and uses
the return values to override the run's parameters.  Parameters
without a getter keep whatever the caller passed (or the default).
The getters execute **in the daemon process** and are re-imported from
source on every run (no `__pycache__` is written).

This tutorial covers every overridable parameter, the tools-file
contract, error handling, and ends with a complete working example.

### Prerequisites

- A running `kiss-web` daemon (start one with `kiss-web`).
- At least one LLM provider API key configured (Anthropic, OpenAI,
  Google, OpenRouter, etc.).
- Any Python packages your extension agent imports must be available
  in the daemon's Python environment.


## Quick start

```python
# weather_agent.py — a minimal extension agent

import requests

def get_prompt() -> str:
    return "Look up the current weather in San Francisco and report it."

def get_max_budget() -> float:
    return 0.50

def get_use_worktree() -> bool:
    return False  # no repo changes expected

def get_append_basic_tools() -> bool:
    return False  # only finish + our tools

def get_system_prompt() -> str:
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

def get_tools() -> list:
    """Return the tools the agent may call."""
    return [get_weather]
```

Launch it:

```python
from kiss.server import sorcar

result = sorcar.run(
    "placeholder",  # required non-blank; overridden by get_prompt()
    extension_agent_path="weather_agent.py",
)
print(result.text, result.success, result.cost)
```

The client-side `prompt` argument must be non-empty (the client
validates this before connecting to the daemon), but when the agent
script defines `get_prompt()`, the script's return value replaces it
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
   │ send JSON {"agentPath": "…", …}        │   if get_X defined & callable:
   │ over Unix-domain socket                 │     call get_X()
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
   and sends it as `"agentPath"` on the wire.  A non-string, non-`.py`,
   or nonexistent path raises `ValueError` immediately (before any
   daemon connection).

2. **Daemon side** — `apply_agent_overrides()` (in
   `kiss.server.agent_file`) imports the file, iterates every
   overridable parameter, calls `get_X()` when defined, type-checks
   the return value, and writes the checked value into the command
   dict.  Overrides are staged: they apply atomically only after every
   getter succeeds.  A broken getter raises `AgentFileError` and the
   task fails with a diagnostic message in `TaskResult.text`.

3. **Tools loading** — After overrides, the daemon reads the
   `toolsFile` field and calls `load_tools_file()` to import it and
   invoke its `get_tools()`.  The returned callables become the
   agent's tools.


## Overridable parameters

Every parameter of `sorcar.run()` except `timeout`, `sock_path`, and
`extension_agent_path` itself has a corresponding `get_X()` getter the
extension agent may define.  The table below lists them all.

| Getter function              | Return type                     | `run()` default           | Wire field          |
|------------------------------|---------------------------------|---------------------------|---------------------|
| `get_prompt()`               | `str` (non-empty)               | (required argument)       | `prompt`            |
| `get_work_dir()`             | `str`                           | `""` (daemon default)     | `workDir`           |
| `get_model()`                | `str`                           | `""` (daemon default)     | `model`             |
| `get_chat_id()`              | `str`                           | `""` (new chat)           | `chatId`            |
| `get_system_prompt()`        | `str`                           | `""` (default SYSTEM.md)  | `systemPrompt`      |
| `get_tools()`                | `str`, `Path`, `list`, or `None`| `None` (no extra tools)   | `toolsFile`         |
| `get_use_worktree()`         | `bool`                          | `True`                    | `useWorktree`       |
| `get_auto_commit()`          | `bool`                          | `True`                    | `autoCommit`        |
| `get_max_budget()`           | `float` (finite) or `None`      | `None` (daemon default)   | `maxBudget`         |
| `get_model_config()`         | `dict` or `None`                | `None`                    | `modelConfig`       |
| `get_web_tools()`            | `bool` or `None`                | `None` (daemon default)   | `webTools`          |
| `get_is_parallel()`          | `bool`                          | `True`                    | `useParallel`       |
| `get_append_basic_tools()`   | `bool`                          | `True`                    | `appendBasicTools`  |
| `get_append_to_system_prompt()` | `str`                        | `""` (append nothing)     | `appendToSystemPrompt` |
| `get_append_to_prompt()`     | `str`                           | `""` (append nothing)     | `appendToPrompt`    |

When a getter is absent, the caller's value is used (which is the
`run()` default when the caller did not pass one).

The three parameters without getters:

- **`timeout`** — bounds the *client's* local wait; the daemon never
  sees it.
- **`sock_path`** — selects which daemon to connect to; the script
  already runs on that daemon.
- **`extension_agent_path`** — the script cannot override its own path.

### Getter semantics

- **`get_model()`** — an empty string `""` means "use the daemon's
  configured default model".  A non-empty string must name a model in
  the daemon's available model list or the task fails.
- **`get_chat_id()`** — an empty string `""` starts a fresh chat.  A
  non-empty string resumes that chat session.
- **`get_system_prompt()`** — an empty or blank string uses the
  default `SYSTEM.md` system prompt.  A non-empty string replaces it.
- **`get_tools()`** — **overrides** (does not append to) the caller's
  `tools` argument.  Returning `None` clears any caller-supplied tools.
- **`get_web_tools()`** — `None` uses the daemon's configured
  default; `True`/`False` forces browser tools on or off.
- **`get_append_to_system_prompt()`** — extra text **appended** to
  the run's system prompt (the default `SYSTEM.md` prompt or the
  `get_system_prompt()` replacement) when the agent is executed.
  Unlike `get_system_prompt()`, it does not replace anything.
- **`get_append_to_prompt()`** — extra text **appended** to the
  executed task prompt.  A multi-`<task>` prompt runs the agent once
  per subtask and the text is appended to each subtask's prompt.  The
  appended text becomes part of the recorded prompt in chat history.


## Tools: two contracts

An extension agent supplies tools to the LLM agent through one of two
approaches.

### 1. Separate tools file (path return)

`get_tools()` returns the **path** (string or `pathlib.Path`) of
another Python file.  The daemon imports that file and calls its own
`get_tools()` to obtain the callable list.  Use an absolute path; the
daemon does not resolve paths against the client's working directory.

```python
# my_agent.py
import pathlib

def get_tools():
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

`get_tools()` returns a **list of callables** directly.  The daemon
normalizes this to the agent script's own path and later re-imports
the same file as the tools file, calling `get_tools()` again.  This
makes the extension agent its own tools file — a single file provides
both parameter overrides and tools.

Because the file is imported twice per run (once for parameter
overrides, once for tools loading), module-level side effects execute
twice.  Use guards (e.g. `if __name__ == "__main__"`, lazy
initialization, or idempotent setup) if side effects are expensive.

```python
# self_contained_agent.py

def get_prompt() -> str:
    return "Double the number 21."

def double(n: int) -> int:
    """Double a number.

    Args:
        n: The number to double.
    """
    return n * 2

def get_tools() -> list:
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

By default (`append_basic_tools=True`) the agent gets the built-in
KISS Sorcar toolset — `Bash`, `Read`, `Edit`, `Write`,
`ask_user_question`, `talk`, `set_model`, `summary`, and (depending
on `web_tools` and `is_parallel`) browser tools, `run_agent`,
`run_parallel`, `number_of_cores` — **plus** your extension tools.

When `append_basic_tools=False`, the agent's **only** tools are
`finish` and the tools from `get_tools()`.  This is useful for
building focused, restricted agents.

When restricting tools, the default system prompt (`SYSTEM.md`)
assumes the full toolset (it mandates a first `Read("./SORCAR.md")`
call, among other things).  Pass a custom `get_system_prompt()` that
matches the tools you provide:

```python
def get_append_basic_tools() -> bool:
    return False

def get_system_prompt() -> str:
    return (
        "You are a weather assistant. Use the get_weather tool "
        "to look up weather, then call finish with the result."
    )
```


## Error handling

Errors fall into two categories depending on where they are caught:

**Client-side errors** (raised as `ValueError` before connecting to
the daemon):
- `extension_agent_path` is not a string
- The path is not a `.py` file
- The file does not exist

**Daemon-side errors** (the task starts, then fails with
`result.success == False` and the diagnostic in `result.text`):

| Condition | Error message |
|-----------|---------------|
| File deleted between client validation and daemon import | `agent script '...' is not an existing Python (.py) file` |
| File raises at import time | `agent script '...' failed to import: ...` |
| `get_X` defined but not callable | `get_X of agent script '...' must be a callable, got ...` |
| `get_X()` raises an exception | `get_X() of agent script '...' raised: ...` |
| `get_X()` returns wrong type | `get_X() of agent script '...' must return ..., got ...` |

Overrides are **atomic**: if any getter fails, the command keeps all
its original values (no partial overrides).


## Continuing chat sessions

Pass `chat_id` to continue an existing daemon chat session.  The agent
sees all prior tasks and results as context.

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

An extension agent can also force a specific chat via `get_chat_id()`.


## Model configuration

Use `get_model_config()` to pass custom endpoint URLs, headers, or
sampling parameters:

```python
def get_model() -> str:
    return "my-custom-model"

def get_model_config() -> dict:
    return {
        "base_url": "http://localhost:8080/v1",
        "api_key": "sk-local-key",
    }
```

When `model_config` contains a `base_url`, the daemon bypasses its
normal model routing and creates an OpenAI-compatible model pointing
at that URL.


## Complete working example

Below is a self-contained extension agent that gives the LLM tools for
managing a SQLite task database.  It uses the full basic toolset
(`append_basic_tools` defaults to `True`), so the LLM can also use
`Bash`, `Read`, `Write`, etc. alongside the custom database tools.

The agent does **not** define `get_prompt()` or `get_model()`, so
the caller's prompt reaches the LLM and the daemon's configured
default model is used.

```python
# task_manager_agent.py
"""Extension agent for managing a SQLite task database.

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
# No get_prompt() — the caller's prompt is used as-is.
# No get_model() — the daemon's configured default model is used.

def get_max_budget() -> float:
    return 1.0


def get_use_worktree() -> bool:
    return False  # no code changes expected


def get_system_prompt() -> str:
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


def get_tools() -> list:
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
    model: str = "",
    chat_id: str = "",
    system_prompt: str = "",
    tools: str | Path | None = None,
    extension_agent_path: str = "",
    use_worktree: bool = True,
    auto_commit: bool = True,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = True,
    append_basic_tools: bool = True,
    append_to_system_prompt: str = "",
    append_to_prompt: str = "",
    timeout: float = 3600.0,
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
    chat_id: str      # daemon chat session id (for continuation)
    task_id: str      # persisted task_history row id
```


## Extension agent vs. tools file

| Aspect | Extension agent (`extension_agent_path`) | Tools file (`tools`) |
|--------|------------------------------------------|----------------------|
| **Purpose** | Override run parameters AND supply tools | Supply tools only |
| **Getter functions** | `get_prompt()`, `get_model()`, `get_system_prompt()`, `get_tools()`, etc. (15 total) | `get_tools()` only |
| **Required function** | None — define only the getters you need | Must define `get_tools()` |
| **Can be combined** | Yes — `get_tools()` can point to a separate tools file | N/A |
| **Can be self-contained** | Yes — return a list from `get_tools()` and the script becomes its own tools file | Always self-contained |


## Tips

- The client-side `prompt` argument must be **non-empty** even when
  `get_prompt()` overrides it; the client validates before connecting.
- An extension agent **may define any subset** of the getters.  Only
  define the ones whose defaults you want to change.
- Omit `get_model()` to use the daemon's configured default model
  rather than hard-coding one.
- The `get_tools()` return value of a **list** makes the extension
  agent its own tools file.  This is the most common pattern.
- **Use absolute paths** for the `get_tools()` path return — the
  daemon does not resolve paths against the client's working directory.
- `get_tools()` **overrides** the caller's `tools` argument; it does
  not append to it.  Returning `None` clears caller-supplied tools.
- The agent script is **re-imported from source** on every run.
  Edits take effect immediately without restarting the daemon.
- A self-contained agent (list-returning `get_tools()`) is imported
  **twice** per run: once for parameter overrides, once for tools
  loading.  Keep module-level side effects idempotent.
- `get_max_budget()` must return a **finite** number.  `NaN`,
  `±inf`, or an overflowing value raises `AgentFileError`.
- A `get_X = None` (a defined attribute that is not callable) is
  treated as a broken getter and stops the task — it is not treated
  as "absent".
- The extension agent and its tools run **in the daemon process**
  with the daemon user's privileges and environment.  Any libraries
  your code imports must be installed in the daemon's Python
  environment.
