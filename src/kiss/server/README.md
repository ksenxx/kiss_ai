# Writing Sorcar Extension Agents (SEAs) for KISS Sorcar

A **Sorcar Extension Agent (SEA)** is a plain Python file whose path you pass as
`extension_agent_path` to `kiss.server.sorcar.run()`.  The daemon
imports the file, calls its top-level `settings()` function — which
returns a dict of a `kind` and the `run()` parameters the SEA wants
to pin — and writes those values over the run's parameters.
Parameters the dict does not name keep whatever the caller passed (or
the default).  A few more top-level functions shape the run: the three
prompt surfaces — `prompt(task)` (receives the task text, returns the
prompt body), a replacement `system_prompt()` and an additive
`add_to_system_prompt()` — plus `add_to_tools()` and two hook getters.
A SEA can build on another with `settings()["extends"]`.  Everything
the file defines executes **in the daemon process** and is re-executed
from source on every run (no `__pycache__` is written) by the one
loader `kiss.agents.sorcar.sea_settings.execute_python_file`.

Each SEA lives in its own folder named after it, `<name>/<name>_sea.py`,
together with the helper modules and data files it needs (the bundled
`src/kiss/agents/seas/sh/sh_sea.py` and
`src/kiss/agents/third_party_agents/slack/slack_sea.py` are laid out this
way).  Besides `settings()`, every SEA must define
`description()`: a zero-argument function returning one sentence that
says what the SEA does and how to use it.  It is what the chat command
`/<name> help` prints (see the slash-command bullet under [Tips](#tips)).

This tutorial covers every `settings()` key and kind, the additive
functions, how `run_agent` and `/xxx` commands run a SEA, error
handling, and ends with a complete working example.

### Prerequisites

- A running `kiss-web` daemon (start one with `kiss-web`).
- At least one model available to the daemon: an LLM provider API key
  (Anthropic, OpenAI, Google, OpenRouter, etc.) or an installed Claude
  Code / Codex CLI executable (`cc/*`, `codex/*` models).
- Any Python packages your SEA imports must be available
  in the daemon's Python environment.


## Quick start

```python
# weather/weather_sea.py — a minimal SEA

import requests


def description() -> str:
    return (
        "Reports the current weather in San Francisco from wttr.in; "
        "pass its path as extension_agent_path or send `/weather now` "
        "(any text after the command) once its parent folder is listed "
        "in $KISS_HOME/SEAS.md."
    )


def settings() -> dict:
    return {
        "kind": "worker",       # no worktree, auto-commit, classifier, fan-out, browser, memory
        "tool_profile": "none",   # no built-in tool: get_weather + finish only
        "max_budget": 0.50,
    }

def prompt(task: str) -> str:
    # The task is what the caller typed after ``/weather`` or passed
    # to ``run_agent``; the SEA turns it into the prompt body.
    return f"Look up the current weather in {task or 'San Francisco'} and report it."

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

def add_to_tools() -> list:
    return [get_weather]
```

Launch it:

```python
from kiss.server import sorcar

result = sorcar.run(
    "Paris",  # the task; the SEA's prompt(task) turns it into the prompt
    extension_agent_path="weather/weather_sea.py",
)
print(result.text, result.success, result.cost)
```

The client-side `prompt` argument must be non-empty (the client
validates this before connecting to the daemon); it is the task the
SEA's `prompt(task)` receives on the daemon.  A SEA without `prompt()`
runs the task text as its prompt.


## How it works

```
 Client process                          Daemon process
 ──────────────                          ──────────────
 sorcar.run(                             receive "run" JSON command
   prompt=...,                               │
   extension_agent_path="agent.py"           ▼
 )                                       layers = load_layers(cmd, base=<picker SEA>)
   │                                         │ execute agent.py and every script it
   │ validate path exists                    │   extends, ONCE (sea_commands.sea_layers)
   │ resolve to absolute                     │ settings per layer: kind defaults +
   │ send JSON {"agentPath": "…", …}        │   explicit keys, type-checked, merged
   │ over the local WSS endpoint             ▼
   │                                     kind "channel": enter the workspace
   │                                         │ (held until the run ends)
   │                                         ▼
   │                                     apply_agent_overrides(cmd, layers)
   │                                         │ evaluate_sea(layers, task, task_id):
   │                                         │   prompt(task) chain ({task_id} filled),
   │                                         │   system_prompt(), add_to_system_prompt(),
   │                                         │   add_to_tools() union, hooks
   ▼                                         │ stage one wire field per setting,
 block, read events ◄───────────────────     │   prompt, cmd["tools"], hooks
                                             │ apply staged overrides to cmd
                                             ▼
                                         agent.run(tools=cmd["tools"], …)
```

1. **Client side** — `sorcar.run()` validates that `extension_agent_path`
   points to an existing `.py` file, resolves it to an absolute path,
   and sends it as `"agentPath"` on the wire.  `None` or `""` means
   no SEA; any other non-string value (including a
   `pathlib.Path`), a non-`.py` path, or a nonexistent file raises
   `ValueError` immediately (before any daemon connection).

2. **Daemon side** — the task runner executes the file and the
   scripts it `extends` once (`kiss.agents.sorcar.sea_commands.sea_layers`;
   a model-picker SEA chosen on the tab is laid under them as the
   outermost layer), resolves each layer's settings
   (`kiss.agents.sorcar.sea_settings.resolve_settings`: the named
   kind's defaults with the explicit keys of `settings()` on top,
   every value type-checked) and merges them (a later layer's key
   wins).  `apply_agent_overrides()` (in `kiss.agents.sorcar.agent_file`)
   then evaluates the layers on the task (`evaluate_sea`) and writes
   each merged setting into the command dict's wire field, the result
   of the `prompt(task)` chain into `prompt`, `system_prompt()` into
   `systemPrompt`, and appends `add_to_system_prompt()` to
   `appendToSystemPrompt`.
   Overrides are staged: they apply atomically only after everything
   succeeds.  A broken script raises `AgentFileError` and the task
   fails with a diagnostic message in `TaskResult.text`.

3. **Tools** — The list of callables `add_to_tools()` returned is
   staged on the daemon-side command dict's `tools` field (never on
   the wire, like the hooks) and passed straight to the agent, added
   to its built-in toolset; with `"tool_profile": "none"` the daemon
   builds no built-in toolset and the list plus `finish` is the whole
   tool set.  Nothing is re-imported.


## `settings()`: the run parameters

A SEA pins the parameters of its run with ONE optional top-level
function, `settings()`, returning a dict of data: a `kind` (a named
dict of defaults) and any of the keys below (`sea_settings.SETTING_TYPES`).
Every key but `kind`, `extends`, `timeout`, `locked` and `hidden`
(`sea_settings.DISPATCHER_SETTINGS`) is a parameter of `sorcar.run()` and
lands on that parameter's wire field; the wire
name is the keyword in camelCase (`agent_file.wire_field`:
`use_web_tools` → `useWebTools`), so no hand-kept table is needed.
Extra tools have no `run()` parameter at all (a callable cannot travel
the wire): they come from `add_to_tools()` (see below).  The bundled
`/sh` agent is the whole pattern:

```python
def settings() -> dict[str, Any]:
    """A worker with Bash only, on the real checkout, running SYSTEM_PROMPT."""
    return {
        "kind": "worker",
        "tool_profile": "bash",
    }
```

Generated by `uv run sea docs` from `sea_settings.SETTING_TYPES`; the
`run()` default of a key the script leaves out is the daemon's.

<!-- sea-docs: settings -->
| Key | Type | `run` wire field | Meaning |
|---|---|---|---|
| `kind` | `str` | — | What the run is: `session` (the default, an ordinary Sorcar session), `worker` or `channel`; each is a dict of defaults laid under the explicit keys (see the kind table). A `channel` run holds its channel workspace, gets the channel preamble, never inherits from a calling task and is never a `run_parallel` child or an `extends` base. |
| `extends` | `str` | — | A base script (command name or `.py` path) whose layers run under this one: settings merge with the later layer winning, `prompt(task)` functions chain, system-prompt additions concatenate, tools union. |
| `work_dir` | `str` | `workDir` | The directory the run works in; default: the calling task's or the tab's. A relative path is resolved against the script's own folder, not the caller's. |
| `model` | `str` | `model` | The LLM model, a catalogue name or a model-picker SEA; `""` or `None` keeps the caller's. |
| `chat_id` | `str` | `chatId` | The chat the run's events go to; default: a new chat. |
| `use_worktree` | `bool` | `useWorktree` | Run in a git worktree of the project (daemon default `True`). |
| `auto_commit` | `bool` | `autoCommit` | Commit the run's changes when it ends (daemon default `True`). |
| `max_budget` | `int \| float` | `maxBudget` | USD budget of the run, a finite number; default: the caller's share or the daemon's default. |
| `model_config` | `dict` | `modelConfig` | Model configuration dict passed to the LLM (temperature, base URL, ...). |
| `use_web_tools` | `bool` | `useWebTools` | Give the run the browser tools (daemon default: on). |
| `auto_classify` | `bool` | `classifyTasks` | Let the pre-run classifier decide the worktree mode and lite prompt (daemon default: the persisted setting). |
| `use_memory` | `bool` | `useMemory` | Give the run the `memory_*` tools (daemon default: the persisted setting). |
| `allow_fan_out` | `bool` | `isParallel` | Let the run call `run_parallel` (default `True`). |
| `tool_profile` | `str` | `toolProfile` | The run's toolset: `review`, `bash`, `shell+edit`, ... (default: the full toolset). |
| `docker_image` | `str` | `dockerImage` | Run inside this Docker image (default: the host). |
| `timeout` | `int \| float` | — | Seconds the call blocks for the run: the call's `timeout` argument or option wins, then this setting, then the default, which is 3600 for a `run_agent` call (when it expires the run keeps going as an `agent_job` and the call returns its job id; a job still running when the calling task ends is killed) and no limit of its own for a `run_parallel` child (a thread of the calling task, bounded by it); ignored by `/<name>`. |
| `locked` | `list` | — | Keys an explicit `run_agent` / `run_parallel` argument or option may not change: a differing value is an error. |
| `hidden` | `bool` | — | `True`: the script is no `/command` and no `run_agent` agent name (loadable by path and as an `extends` base only). It must be the literal `True` in `settings()` because the command registry reads it from the source without running the script (a computed value is ignored). |
<!-- /sea-docs -->

A key whose value is `None` is dropped and behaves like an absent
key — as is a `model` of `""` — the kind's default applies when the
kind has one for it (`{"kind": "worker", "use_memory": None}`
still runs without memory); otherwise the caller's or the persisted
value stands.  A `bool` key given a non-`bool`, an `int`/`float` key
given a `bool`, an unknown, renamed or removed key, an unknown `kind`, or a
non-finite `max_budget` / `timeout` stops the task with a diagnostic
(see [Error handling](#error-handling)).

The prompt itself is not a setting: `settings()` is data, text and
code come from the getters.  Three optional functions are the only
prompt surfaces:

| Function                  | Returns | Effect                                                                 |
|---------------------------|---------|------------------------------------------------------------------------|
| `prompt(task)`            | `str` (non-empty) | the prompt body for the task text the run was submitted with (the text after `/xxx`, the `run_agent` task, the `sorcar.run()` prompt); `{task_id}` in the result is replaced by the calling task's id (the command's `parentTaskId`, `""` without one); without it the task text is the prompt |
| `system_prompt()`         | `str`   | replaces the base system prompt (`systemPrompt`)                        |
| `add_to_system_prompt()`  | `str`   | appended to the system prompt after the caller's own suffix              |

### Kinds

`settings()["kind"]` says what the run is and picks a dict of defaults
(`sea_settings.kind_defaults()`); explicit keys of the same dict override
it.  A kind is defaults plus, for `channel`, the daemon's and the
dispatcher's channel behaviour described below.

<!-- sea-docs: kinds -->
| Kind | Defaults | Use |
|---|---|---|
| `session` | nothing | The default: an ordinary Sorcar session with the caller's or the user's settings (`/write`, `/write_paper`, `bestrouter`). |
| `worker` | `use_worktree=False`, `auto_commit=False`, `auto_classify=False`, `allow_fan_out=False`, `use_web_tools=False`, `use_memory=False` | A focused tool-bound run on the caller's tree: no worktree, no auto-commit, no classifier, no fan-out, no browser, no memory (`/sh`, `/ask`, `/merge`, `/remember`, `/forget`, `/task_update`). |
| `channel` | `use_worktree=False`, `auto_commit=False`, `auto_classify=False`, `allow_fan_out=False`, `use_web_tools=False`, `use_memory=False`, `work_dir='<home>/channel_work'` | A worker for an external service, in the shared `channel_work` scratch directory under the Sorcar home, never the caller's project; it holds its channel workspace, gets the channel preamble and never inherits from a calling task (every bundled channel agent and `/cron`). |
<!-- /sea-docs -->

A `channel` run's `run_agent` dispatch takes NOTHING from the
calling task (an `inherit: true` option is refused); `work_dir` keeps
the run out of the caller's project (cron's own `work_dir`,
`cron_agent.cron_work_dir()`, wins over the kind's) and is implicitly
locked.  `kind: "channel"` is the one key the daemon acts on beyond
passing a value through: it appends the channel preamble
(`agent_file.CHANNEL_PREAMBLE`, "You are the {name} agent: this
session already has the {name} tools ... Never call run_agent here
...", `{name}` being the script's file stem without `_sea`) to the
system prompt suffix, before the SEA's own `add_to_system_prompt()`,
and holds the run's channel workspace (the `run_agent` option
`workspace`, default `"default"`; the account the channel tools load
credentials for, exported as `KISS_CHANNEL_WORKSPACE`) from before
`add_to_tools()` runs — the tools bind the credentials of the
workspace active at that moment — until the run ends, whether the run
came from `/slack ...`, `run_agent("slack")` or
`run_agent(".../slack_sea.py")`; a run whose workspace differs from a
running channel task's waits (bounded by
`channel_workspace.WORKSPACE_WAIT_TIMEOUT_SECONDS`) instead of handing
that task the wrong credentials.  A `kind: "channel"` SEA can be
neither a base of another SEA nor a `run_parallel` child.  A channel
SEA is three functions:

```python
def settings() -> dict:
    return {"kind": "channel"}

def add_to_system_prompt() -> str:
    return SlackAgent.channel_system_prompt

def add_to_tools() -> list:
    return [...]  # the channel's authenticated API tools
```

### Extending a SEA: `extends`

`settings()["extends"]` names a *base* SEA — a registered command name
(`"bestrouter"`, `"sh"`) or a path (`.py` suffix or a separator;
relative to the SEA's own folder) — whose configuration the SEA
refines.  The daemon executes the base chain and the SEA once each
(`sea_commands.sea_layers`) and applies the layers in order, the SEA
last:

- settings: a later layer's key wins; the effective `kind` is the
  last one other than `session` (`session` changes nothing, so it
  never masks a base's `worker`);
- `prompt(task)`: chained — the SEA's receives what the base's returned;
- `add_to_system_prompt()`: concatenated, base first;
- `system_prompt()`, `llm_call_hook()`, `tool_call_hook()`: the
  innermost layer that defines one wins;
- `add_to_tools()`: the union, a later layer's tool replacing an
  earlier one of the same name.

A model-picker SEA chosen on the tab (`bestrouter`, `autorouter`) is
the outermost layer of every run submitted from that tab: a `/xxx`
command or a `run_agent` child run on that tab runs its own SEA on top
of the picker's model and routing protocol.  A file both chains share
(a SEA that `extends` the tab's picker, or a common ancestor) is one
layer, executed once.  A layer whose `model` setting names a picker
entry is resolved the same way: the run gets the picker's model (the
default model when it names none), never the picker's name.  A cycle,
an unknown base and a `kind: "channel"` base are errors.

### Precedence

One rule decides every setting of a dispatched run
(`sea_settings.PRECEDENCE_RULE`, enforced by `locked_conflicts`; this
block is generated from the constant):

<!-- sea-docs: precedence -->
> For every setting of a sub-task: what the call passes explicitly (a `run_agent` / `run_parallel` argument or `options` key) wins, then the SEA's `settings()`, then what the calling task passes on (and, for `/<name>`, the chat panel's persisted settings), then the user's defaults. A SEA may list keys in `locked`: a call that passes a different value for a locked key is refused with an error, never silently overruled.
>
> For example, `/sh` declares `{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}`, so `run_agent(agent="sh", task=..., tool_profile="review")` is refused with `Error: sh: the script locks tool_profile='bash' (asked for 'review')`, while `run_agent(agent="sh", task=..., model="gpt-5")` runs it with that model: `model` is not locked, so the explicit argument wins.
<!-- /sea-docs -->

Example of a refused call: `Error: sh: the script locks
tool_profile='bash' (asked for 'review')`.  A `/<name>` run in a tab
and a `run()` call with an `extension_agent_path` pass nothing
explicitly (unless the `run()` call marks values `"explicit"` in its
`provenance` argument, as the dispatcher does), so there the SEA's
settings apply over the tab's or the client's values.  The `ran:` line of every sub-task result reports the
outcome: `inherited=` names the keys taken from the calling task,
`pinned=` the inherited or default values the SEA's `settings()`
replaced (`pinned=use_worktree(True->False)` for `/sh` run from a
worktree task); an explicit argument never appears there, because it
either won or was refused.  A plain sub-agent whose worktree default
the pre-run classifier dropped (a non-development task) ends with
`classified=use_worktree(True->False)`; a `tools=review(inferred)`
says nobody named the profile and the sub-agent got `review` because
it is a reviewer (see `run_parallel`'s `tool_profile`).

Inheritance (`agent_dispatch.inherit_from_parent`) applies to every
dispatched script that is not a `channel`, unless the call's `inherit`
option is `false`: the `run_agent`
arguments the caller leaves empty are first filled from the calling
agent, as a `run_parallel` child's would be: its model (and its
`model_config`, but only when the sub-task runs the model the caller
was launched with and the script's `settings()` names no `model`
itself: a script-chosen model runs with default provider routing),
half of its remaining budget (the other half stays reserved for the
caller), its chat id (so the sub-task sees the conversation's earlier
tasks and results), its own replacement system prompt and appended
system-prompt text (the `system_prompt` / `append_to_system_prompt`
its run was given, so a run's extra system instructions constrain its
whole task tree as they do through `run_parallel`; the classifier's
lite-vs-full choice is not among them, the sub-task is classified on
its own), the text appended to its own prompt, the extra tools its
own SEA added (so the tools the inherited system prompt
refers to exist; not into a run on the `none` profile), its web-tools
and memory settings, whether it may fan out itself (`allow_fan_out`),
its live Docker container (`container:<id>`), and
its effective worktree and auto-commit choices after the classifier's
demotion (both `False` when the container is inherited: the sub-task
then works in the caller's tree).  `run_parallel` children are built
from the same table (the precedence rule above decides between the
inherited values and the SEA's).  A `channel` SEA (`agent="slack"`,
`agent="cron"`, ...) inherits none of these: it runs on the daemon's
default model and budget unless the call passes its own, and every
key the kind sets (`work_dir`, `use_worktree`, `auto_commit`,
`auto_classify`, `allow_fan_out`, `use_web_tools`, `use_memory`) is
locked, so a call whose `options` contradict one is refused.  A
`use_worktree` any SEA pins, or a `run_agent` call passes explicitly,
is a decision the task classifier never demotes (it still demotes a
client's or persisted default, and says so with `classified=`).

### `description()` — mandatory, not a run parameter

| Function        | Return type         | Used by                                   |
|-----------------|---------------------|-------------------------------------------|
| `description()` | `str` (one sentence, non-empty) | `/<name> help` in the chat (`kiss.agents.sorcar.sea_commands`) |

Every SEA must define `description()`, a zero-argument function that
returns one sentence saying what the SEA does and how to use it.  It
is not a run parameter: `apply_agent_overrides()` ignores it and it
never reaches the command dict.  The slash-command registry calls it
when the prompt is `/<name> help` (the bare word `help`, any letter
case, nothing after it; `sea_commands.help_text_if_command`): the
daemon does not run the SEA and calls no model, it imports the script,
calls `description()` and posts the returned sentence as the task
result.  A SEA without a callable `description()`, or whose
`description()` raises or returns something other than a non-empty
string, yields a diagnostic instead.

### `timeout` — optional, for SEAs that run longer than an hour

A `run_agent` call waits for its sub-task for a bounded time
(`agent_dispatch.resolve_timeout`): an explicit `timeout`
argument wins; an empty one takes the SEA's `settings()["timeout"]`
when that is positive; with neither,
`DEFAULT_DISPATCH_TIMEOUT_SECONDS`, 3600 s.  When the bound expires
the sub-task is NOT stopped: it keeps running as an `agent_job` and
the call returns the job's id (`agent_job(id, "wait")` collects the
result, `"kill"` stops it; a job still running when the calling task
ends is killed).  A SEA whose runs may take longer than an hour
declares it so a caller is not handed a job id too early (`/write_paper` 6 h, `/review_paper`
2 h, `/revise_and_review_paper` 24 h); `/write` pins the default
explicitly:

```python
DISPATCH_TIMEOUT_SECONDS = 3600.0

def settings() -> dict[str, Any]:
    return {"timeout": DISPATCH_TIMEOUT_SECONDS}
```

The setting is read for every `run_agent` dispatch, whatever spelling
`agent` used (a path, a registered command name, a channel name or
`cron`): the dispatcher resolves the agent to its script and reads
that script's merged `settings()` before it waits.  Not a run parameter: it has no wire
field, and a `/xxx text` command, which runs the SEA directly in the
tab's own run, has no dispatch timer at all.

### Running a SEA: `run_agent` and `/xxx` commands

A running agent dispatches a SEA with the `run_agent` tool
(`agent_dispatch.make_run_agent_tool`):

```text
run_agent(task, agent="", model="", tool_profile="", max_budget="", timeout="", options="", wait="")
run_parallel(tasks, agent="", model="", tool_profile="", max_budget="", timeout="", max_workers="", options="")
agent_job(job_id, action="tail", timeout_seconds="")
```

`task` is the sub-task's task text (the SEA's `prompt(task)`, if
defined, turns it into the prompt); `options` is a JSON object in the
`settings()` vocabulary (`agent_dispatch.OPTION_TYPES`: every settings
key except the ones that describe a script, `kind`, `extends`,
`locked` and `hidden`, plus `inherit`, `workspace`, `add_to_prompt`
and `add_to_system_prompt`); `model`, `tool_profile`, `max_budget` and
`timeout` (number strings) are shortcuts for the options of the same
name (either way, but not differently in both).  Every value given is
explicit: it wins over the SEA's settings, or is refused when the SEA
locks the key.  Each option replaces the inherited value (and the
SEA's, unless locked) of one key: `work_dir` (relative to
the calling task's directory), `chat_id`, `workspace` (the account of
a multi-account channel; refused for any SEA that is not a `kind:
"channel"`), `model_config`, `inherit`,
`use_worktree`, `auto_commit`, `auto_classify`, `use_web_tools`,
`use_memory`, `allow_fan_out`, `docker_image`, `add_to_prompt` and
`add_to_system_prompt`, e.g. `'{"use_web_tools": false}'`.  `wait="false"`
returns with a job id as soon as the sub-task's tab exists (its
initial `status running=true`, so the spawn lands inside the call's
time window on every surface); the dispatch runs in a daemon thread
of the calling process (`agent_dispatch.start_agent_job`), its spend
is bound to the calling run's usage epoch, and only the calling
agent's `agent_job` tool sees the job: `agent_job(job_id, "wait")`
blocks for its result, `"tail"` reports whether it still runs,
`"kill"` sets the job's cancel event, which `daemon_client.run`'s read
loop turns into a stop of the sub-task confirmed by its terminal
status (an unconfirmed stop is reported as such, never as "stopped").
`agent` names WHICH agent to run; every spelling resolves to one
SEA path and takes the same dispatch
(`agent_dispatch.resolve_agent` tries, in order):

1. empty, or a generic label a model may invent (`general`,
   `assistant`, `default`, ...; not `worker`, which is a kind, and
   not `reviewer`, which is refused with "pass `tool_profile="review"`"
   because a reviewer is a toolset, not an agent): the plain
   sub-agent, `seas/sorcar/sorcar_sea.py`;
2. a path (a `.py` suffix or a path separator): that SEA file, a
   relative path resolved against the calling task's work directory;
3. a registered slash-command name (case, spaces, hyphens and
   underscores are ignored, so "Home Assistant" is `homeassistant`):
   a bundled SEA (`write_paper`, `sh`), a channel
   (`third_party_agents/<name>/<name>_sea.py`), `cron`
   (`sea_commands.BUILTIN_COMMANDS`, the scheduled-automations agent
   `kiss.agents.sorcar.cron_agent`) or a `SEAS.md` folder: that
   command's script, located by path.  The calling process executes
   the script and its `extends` bases once to read their merged
   `settings()` (`sea_commands.sea_settings`: the kind, `timeout` and
   `work_dir` the dispatch needs); the run itself, with the getters
   and tools, happens on the daemon.

Anything else returns `Error: unknown agent '...' — not a registered
slash command and not a path to a .py SEA file.` with the closest
command and the registered commands.  The sub-task is reported under
the resolved name: the command name, else the script's file stem
without `_sea`.

A `/xxx text` prompt typed into a chat tab is NOT a dispatch: the
daemon (`task_runner._run_task`, via `sea_commands.slash_command_task`)
runs the SEA `xxx` directly in the tab's own run, with `text` as the
prompt and the SEA's path as the run's `agentPath`, so the SEA's
settings, system prompt and tools apply to that very run; there is no
relay LLM turn and no nested sub-agent tab, and the raw `/xxx text`
line stays what the task panel and the chat history show
(`displayPrompt`).  `/xxx help` answers with `description()` without
running anything; `/xxx check` executes the script and answers with
its layers, effective settings, model, added tools, defined getters
and sample prompt, or the first error (`sea_commands.sea_check`).  A
model-picker SEA (`bestrouter`) selected on the
tab is the outermost layer under `/sh ...`: `/sh`'s own `agentPath`
stays, on the picker's model and with the picker's routing protocol.

`run_parallel(tasks, ..., agent=...)` runs the same SEA in-process for
every fan-out child under the same precedence rule: the call's
arguments and `options` win, then the SEA's settings, then what the
children inherit; its `prompt(task)` shapes each child's prompt, its
system-prompt texts,
tools and hooks apply (`sorcar_agent._sea_run_kwargs`).  The children
inherit what a `run_agent` sub-task inherits (the parent's model and
configuration, chat, system-prompt texts, prompt suffix, extra tools,
web/memory/Docker choices, `allow_fan_out`), the SEA's own `model_config`
replacing the parent's; a `prompt(task)` that raises for one task fails
that child alone.  A child is a thread of the caller on its own tree
and chat, so a SEA — or the call's `options` — pinning
`use_worktree`, `auto_commit` or `auto_classify` to `True`, a
`chat_id` or a `workspace`, and a `kind: "channel"` SEA, are refused
with an error that says to use `run_agent`
(`agent_dispatch.fanout_conflict`); a SEA's `timeout` bounds each
child only when the call passes none, and `inherit: false` makes the
children take nothing from the caller but their budget share.

### `add_to_system_prompt()`, `register_as_model()` and `on_picked_as_model()` — model routing SEAs

| Function                 | Return type | Effect                                                     |
|--------------------------|-------------|------------------------------------------------------------|
| `add_to_system_prompt()` | `str`       | text **added** to `appendToSystemPrompt` after the value already there |
| `register_as_model()`    | `bool`      | `True` lists the SEA in the model picker under its command name |
| `on_picked_as_model(work_dir)` | any   | hook run once per run whose model is the SEA; its result is logged |

`add_to_system_prompt()` carries a SEA's *model routing protocol*: the
returned text is appended to the run's system prompt after whatever
`appendToSystemPrompt` already holds (the caller's text, or the
channel preamble), separated by a blank line.  It never replaces the
caller's value, so the protocol reaches every run of the SEA and the
caller's own additions survive; `run_parallel` workers inherit it like
any system-prompt suffix.  `register_as_model()` is a registry flag,
not a run parameter: `kiss.agents.sorcar.sea_commands.model_seas()`
lists every registered SEA whose `register_as_model()` returns `True`,
the daemon offers them in the model picker (vendor `Router`, once at
least one catalog model is runnable), and a task run with such a pick
becomes a run of the SEA on the model its
`settings()["model"]` names (else the default model) — `/xxx` slash
commands and runs that already carry an `agentPath` keep their agent
and only take that model.  The bundled `autorouter` and `bestrouter`
are such SEAs; `bestrouter` is three functions besides `description()`:

```python
def register_as_model() -> bool:
    return True

def add_to_system_prompt() -> str:
    return SYSTEM_PROMPT  # the routing protocol

def settings() -> dict[str, Any]:
    return {"model": PRIMARY_MODEL}
```

`on_picked_as_model(work_dir)` is the picked SEA's chance to act on
being chosen.  `kiss.agents.sorcar.sea_commands.run_picked_hook` runs it
on a daemon thread at two moments: when the user picks the SEA in the
model picker (`selectModel`, with the tab's registered work directory,
else the daemon's global one, without waiting), and once per run whose model is the SEA (after the
SEA overrides, with the run's effective work directory,
waiting at most `PICKED_HOOK_TIMEOUT_SECONDS`, 15 s, before the task
starts).  It is not a run parameter: the return value is only logged,
a hook that raises is logged as a warning, and one that blocks is
abandoned on its thread; none of them fails the run.  It does not run
for a `/xxx` command's agent or an explicit `agentPath`, only for the
SEA the model pick names.  `autorouter` uses it to make sure an
enabled weekly cron job that runs `/rsi7d autorouter` exists
(`autorouter_sea.schedule_weekly_rsi7d`: an enabled job of that name is
kept as is, a paused one is resumed, and when there is none one is
created — a prompt job in the KISS checkout the work directory
is in, with worktree and auto-commit, or a scratch-directory job that
still refreshes `$KISS_HOME/AUTOROUTER.md` when there is no checkout;
a linked task worktree resolves to its owning checkout).  Hooks from
concurrent picks may overlap — a SEA file is executed afresh on every
call and cannot hold a lock of its own — so the look-up-then-create is
the cron store's atomic `cron_job("ensure")`, not the SEA's.

The `run()` parameters without a `settings()` key (the allowlist is
`sea_settings.SETTING_TYPES`; naming one of these stops the task with
`settings() has an unknown key ...`):

- **`timeout`** — bounds the *client's* local wait (`None` waits
  indefinitely); the daemon never sees it.  (The `timeout` KEY of
  `settings()` is a different thing: the dispatcher's wait for a
  `run_agent` sub-task, see above.)
- **`stop_on_timeout`** — whether a `timeout` expiry also stops the
  task, awaiting the stop's confirmation (default `False`: the task
  keeps running); a client-side choice the script must not override.
- **`record_timeout`** — a bound sent as the wire field `timeout` (so
  the daemon records it in the task's `task_settings` event and checks
  it against a SEA's locked `timeout`) when the client's own wait has
  none: `run_agent` waits without a deadline on a job thread and bounds
  the call by joining that thread, so it passes the call's bound here.
- **`cancel` / `running`** — two `threading.Event`s of the client's
  wait: setting `cancel` from another thread stops the task
  (`agent_job(id, "kill")`, or the calling run's end) and the wait
  continues until the task's terminal status confirms it is dead or
  the confirmation grace expires, then raises `CancelledError` with
  `confirmed` set accordingly; `running` is set when the task's
  initial `status running=true` arrives, the moment its tab exists on
  every surface (a background `run_agent` job returns its notice only
  after it), or when the wait ends without one.
- **`provenance`** — `{setting key: "explicit" | "inherited"}`, where
  each of the command's values came from (`kiss.agents.sorcar.run_config`);
  the dispatcher fills it from what the call passed and what it took
  from the calling agent, the daemon records it in the task's
  `task_settings` event, and the result's `ran:` line reads its
  `inherited=` list from there.
- **`endpoint_file`** — selects which daemon to connect to (default:
  `$KISS_SORCAR_LOCAL`, else `$KISS_HOME/sorcar-local.json`, the file the
  daemon writes with its WSS URL and per-start local token); the script
  already runs on that daemon.
- **`parent_task_id` / `parent_tab_id` / `parent_reviewer`** — the
  CALLING task's identity (how `run_agent` nests a dispatched run under
  its caller) and whether that caller sits in a reviewer sub-tree (a
  marker the child and its own sub-agents inherit; with tool profiles
  enabled, a marked run whose task is not an implementation task and
  that names no explicit profile gets the read-only `review` profile),
  which a dispatched script must not be able to forge.
- **`side_channel`** — marks the run as a side channel of its parent
  (a sub-agent whose result is shown outside its own tab: the `/ask`
  answerer delivers its answer into the PARENT's transcript, the
  periodic in-process `/ask` run of `kiss.server.task_update` fills the
  parent task's task-info panel, and the in-process `/merge` run that
  resolves a conflicting
  auto-merge works in the parent's repository; so its own tab closes
  when the run ends and is not re-opened when the chat is reloaded);
  only
  meaningful with `parent_task_id`, and not forgeable for the same
  reason.
- **`extension_agent_path`** — the script cannot override its own path.
- **`scope_work_dir`** — the calling workspace recorded on the run's
  tab in the daemon's shared tab registry (`tabScopeWorkDir`),
  informational only and meaningless for a sub-agent run, which gets
  no registry tab; the daemon fills it from the caller.
- **`inherit_tools`** — whether the run also gets the extra tools of
  the task `parent_task_id` names (the callables that parent's script
  added through `add_to_tools()`, and those the parent inherited
  itself), added after the run's own `add_to_tools()` tools, skipping
  names it already has; a run on the `none` profile keeps exactly its
  own set.  `run_agent` sets it for every inheriting dispatch (not a
  `channel`, no `inherit: false` option), so a sub-task that inherits the caller's system
  prompt also has the tools that prompt refers to; a script decides
  nothing here.
- **`workspace`** — the multi-account channel workspace a
  `kind: "channel"` run holds for its lifetime (see
  [Kinds](#kinds)); the `run_agent` option `workspace` or a
  channel launcher supplies it, every other run ignores it.
- **`append_to_system_prompt` / `append_to_prompt`** — the caller's
  additions; a SEA adds its own with `add_to_system_prompt()` and
  `prompt(task)`, which never replace the caller's text.

### Setting semantics

- **`model`** — leave the key out to keep the caller's model (the
  tab's pick, or the calling task's model for a `run_agent`
  dispatch).  An explicit empty string `""` or `None` is dropped like
  an absent key (`sea_settings.resolve_settings`), so it keeps the
  caller's model too; a run whose command names no model at all gets
  the tab's selected model, else the daemon's configured default
  (`task_runner._tab_model`).  A non-empty string must name a model in
  the daemon's available model list or the task fails.
- **`chat_id`** — an empty string `""` starts a fresh chat.  A
  non-empty string resumes that chat session.
- **`system_prompt()`** — the function that replaces the base system
  prompt (`/sh`, `/merge`, `/ask` define one; it is not a settings
  key).  Without it, or with an empty or blank return, the base
  prompt is the daemon's: with task classification enabled (the default) a task classified as
  simple runs on the reduced `SYSTEM_LITE.md`, everything else on the
  full `SYSTEM.md`.  Both files are loaded with their `{{IDENTITY}}`
  brand placeholder filled in (`kiss.core.brand.render_brand`); a
  string from `system_prompt()` or `add_to_system_prompt()` is used
  verbatim, with no placeholder rendering.  A non-empty string
  replaces that base prompt.  Either way the agent still appends its
  per-run operational instructions after the prompt
  (`RelentlessAgent.perform_task`): the work directory (omitted when
  the tools run in an attached container that does not mount it), the
  process id, the task settings, and the user's standing instructions
  from `$KISS_HOME/AGENTS.md` (the file the
  bundled `/remember` and `/forget` SEAs maintain) when that file
  exists.  A `model_config["system_instruction"]` value, if present,
  takes precedence over the composed prompt (`KISSAgent.run` only
  `setdefault`s it).
- **`prompt(task)`** — the function that turns the task text into the
  prompt body (`/merge` wraps the user's text with the standing merge
  instructions; the coding harness ignores the runner's label and
  uses its config's instruction).  It must return a non-empty string.
  Without it the task text is the prompt.
- **`{task_id}` in `prompt(task)`** — replaced by the calling task's
  id (the command's `parentTaskId`; the empty string when the run has
  no parent), which is how `/ask` names the task a question is about:

  ```python
  def prompt(task: str) -> str:
      return task + "\n\n" + (
          "The question above is about the task with id {task_id}. "
          "Call task_context with that task id, then answer the question."
      )
  ```

  A run with a SEA executes its prompt as one task: the
  task runner does not split `<task>` blocks into subtasks for an
  `agentPath` run (they are the SEA's to interpret), so the text is
  appended once.  A prompt that is nothing but a filesystem path is
  likewise handed to the SEA as-is rather than turned into an "open
  this file" request, as it is for a path typed into a chat box.  The
  shaped prompt becomes the recorded prompt in chat history, except
  for a `/xxx text` run, whose history keeps the `/xxx text` line the
  user typed.
- **`use_web_tools`** — per-run browser-tool enablement.  `None`
  falls back to the daemon's configured default (the settings panel's
  "Use web tools" checkbox, persisted as `use_web_browser`).  Under
  the daemon the browser tools include `show_browser()`, which moves
  the page the agent is browsing into the Browser tab on every surface
  (the daemon's `BrowserTabService`, passed to the run as
  `live_browser` and forwarded to its sub-agents) so the user can
  watch and act on it; there is no setting for it.
- **`auto_classify`** — per-run pre-run task classification.
  `None` falls back to the daemon's configured default (the settings
  panel's "Classify tasks before running" checkbox, persisted as
  `classify_tasks`; `task_classifier.classification_enabled`).
- **`use_memory`** — per-run persistent agent memory
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
  `True`.  Pages live in the directory named by the `memory_dir`
  setting, or `$KISS_HOME/memories` when unset;
  a run whose work directory is inside a git repository also attaches
  that repository's domain memory, a sub-directory named after the
  repository's main checkout directory (linked worktrees resolve to it
  through `git rev-parse --git-common-dir`, so they share one domain;
  `sorcar_agent._memory_settings`, `_repo_memory_domains`).
- **`use_worktree`** — whether the run executes in a fresh git
  worktree on its own branch (when `work_dir` is inside a git
  repository; the daemon may hand it a spare worktree it prepared in
  advance) instead of the main working tree.  With task classification
  enabled, a task not classified as development work also runs
  without a worktree; the verdict only ever demotes a `True`, so
  `False` stays `False`.
- **`auto_commit`** — whether the run's changes are committed when
  it finishes successfully.  A worktree run's branch is committed and
  squash-merged into the original branch (a conflicting merge is
  handed to the bundled `/merge` SEA, which resolves and stages the
  conflicted files, and the resolver then commits); a main-tree run's
  changes are committed in place.  A run that failed, was stopped, or
  reported `success: False` is not auto-committed, and `False` leaves
  a worktree run pending for the user to review, merge or discard.
  The commit message is generated from the diff and stamped with the
  user's prompt and the task's result (HTML converted to Markdown,
  `kiss.agents.sorcar.commit_message`).  A merge into the main tree
  waits while another non-worktree task is active there and
  `git status --porcelain -uno` reports uncommitted changes to tracked
  files, whoever made them (an unreadable status blocks too); untracked
  scratch files alone do not block it
  (`merge_flow._main_tree_blocks_merge`).
- **`allow_fan_out`** — whether the agent may spawn parallel
  sub-agents (`run_parallel`); `run_agent` stays available either way.
- **`tool_profile`** — the name of the tool profile the run's
  built-in toolset is cut down to: `"full"` (everything), `"review"`
  (the `shell` and `browser` groups, the four read-only memory tools
  `memory_search`, `memory_pull`, `memory_read` and `memory_list`,
  `decide`, `summary` and `talk`; no editing, memory writing or
  dispatch), `"assistant"`
  (the `shell` group plus `ask_user_question`, `talk`, `decide`,
  `summary`, `set_model`), `"bash"` (`Bash` only — the bundled `/sh`
  agent's choice), or a tool group: `"shell"` (`Bash`, `bash_job`,
  `Read`, `run_commands_parallel`), `"edit"` (`Edit`, `Write`),
  `"browser"`, `"memory"`, `"agents"` (`run_agent`, `agent_job`,
  `run_parallel`, `number_of_cores`), `"mcp"` (the configured servers' tools and the
  sign-in pair), `"skills"`, `"user"` (`ask_user_question`, `talk`),
  `"decide"`, `"control"` (`summary`, `set_model`).  Join several with
  `+` to keep the union of their tools (`"shell+edit+memory"`);
  `finish` is always added.  `"none"` builds no built-in toolset at
  all: the run's tools are `finish` and whatever `add_to_tools()`
  returns (the bundled `/ask` agent's choice, so its only tools are
  `task_context` and `finish`).  `""` keeps the daemon's usual choice.
  An unknown name fails the task when it starts.
- **`docker_image`** — the Docker image the run's shell and file
  tools (`Bash`, `run_commands_parallel`, `Read`, `Edit`, `Write`)
  execute in: an image name starts a fresh container that is removed
  when the task ends (the task's `work_dir` is bind-mounted at the
  same path and is the container's working directory),
  `container:<name-or-id>` attaches to a container the caller already
  runs (commands run in its own working directory, nothing extra is
  mounted, it is left running afterwards), `""` runs the tools on the
  host.
  `run_parallel` sub-agents share the task's container; `bash_job`
  and persistent memory are unavailable in a Docker run.  A
  run-to-completion CLI model (`cc/*`, `codex/*`) executes its own
  tools on the host, so a non-empty `docker_image` with such a
  model fails the task with a `KISSError` instead of silently
  bypassing the container.  The bundled `seas/coding/coding_sea.py`'s
  `ContainerHarness` is the reference user of the attach form: its
  `docker_image()` method returns `container:<id>` for the trial
  container it is given, and the generated per-trial SEAs expose it.

### Hook getters (no `run()` parameter)

The script may also define two hook getters with no corresponding
`sorcar.run()` parameter or `settings()` key — a callable cannot be
JSON-serialized, so the hooks exist ONLY as SEA functions,
evaluated in the daemon process:

| Function               | Return type          | Staged command field |
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

Returning `None` from a hook function means "no hook".  Any other
non-callable return value fails the task with a diagnostic error,
like every wrong-typed value.

```python
# guarded_agent.py
def veto_destructive(name, args):
    if name == "Bash" and "rm -rf" in str(args.get("command", "")):
        return "Blocked: destructive command"
    return "OK"

def tool_call_hook():
    return veto_destructive
```

### Only `settings()` configures a run

The per-field getters of the earlier contract (`def use_worktree() ->
bool`, `def model() -> str`, `def max_budget() -> float`, ...,
`dispatch_timeout()`, `append_to_prompt()`, `append_to_system_prompt()`
and the whole-toolset `tools()`) are no longer read: a module-level
function with one of those names is an ordinary function the daemon
ignores.  The dispatcher and the task runner evaluate `settings()`
alone when they need a script's kind, timeout, model or work
directory before the run exists (`sea_commands.sea_settings`), so
`settings()` must be cheap and side-effect-free there; `system_prompt()`,
`add_to_system_prompt()`, `add_to_tools()` and the hooks run only inside
`apply_agent_overrides`, once per run.


## Tools: `add_to_tools()`

An SEA supplies tools to the LLM agent through one function,
`add_to_tools()`, returning a **list of callables** (a file path is
not accepted).  There is no other way to give the agent extra tools:
`sorcar.run()` has no tools parameter, because a callable cannot
travel the wire.

- By default the returned tools are **added** to the built-in toolset
  (`Bash`, `Read`, `Edit`, `Write`, browser tools, ...), or to the
  tool profile the SEA's `tool_profile` setting selects.
- With `"tool_profile": "none"` the returned tools plus `finish` are
  the agent's **entire** tool set; the built-in toolset is not built.
  Pair it with a `system_prompt()` written for those tools.

The daemon calls `add_to_tools()` once, while applying the script's
settings, and hands the returned callables to the agent.  A single
file provides both settings and tools.  The file is re-executed from
source on every run (no `__pycache__`), so keep module-level side
effects cheap or idempotent.

```python
# self_contained_agent.py

def prompt(task: str) -> str:
    return "Double the number 21."

def double(n: int) -> int:
    """Double a number.

    Args:
        n: The number to double.
    """
    return n * 2

def add_to_tools() -> list:
    return [double]  # built-in toolset + double
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


## The built-in toolset

By default — no `tool_profile` setting, with or without
`add_to_tools()` — the agent gets `finish` (always present) and the
built-in KISS Sorcar toolset — `Bash` (with
`background=True` for detached jobs), `bash_job` (wait for / tail /
kill a background job), `run_commands_parallel` (several shell
commands at once, no LLM sub-agents), `Read`, `Edit`, `Write`,
`ask_user_question`, `talk`, `set_model`, `summary`, `run_agent`,
browser tools (when `use_web_tools`), `run_parallel` and
`number_of_cores` (when `allow_fan_out`), `decide` (when
`OPENROUTER_API_KEY` is configured), the `memory_*` tools (when
memory is enabled), and any configured skill and MCP-server tools —
**plus** your extension tools.  A restricted tool profile (a reviewer
sub-agent dispatched with `run_parallel(..., tool_profile="review")`)
filters that built-in set, including the memory tools; extension tools
are still appended.

With `"tool_profile": "none"` the agent's **only** tools are `finish`
and the tools `add_to_tools()` returned; the built-in toolset is not
built, so `use_web_tools` and `allow_fan_out` have nothing to act on,
and a `run_agent` caller's extra tools are not merged in either.
This is useful for building focused, restricted agents (`/ask`).  A
SEA that wants the built-in toolset plus its own tools leaves
`tool_profile` alone (or picks a narrower profile such as `"bash"`,
as `/remember` does).

When restricting tools, the full default system prompt (`SYSTEM.md`)
assumes the full toolset (its workflow rules name `Read`, `Edit`,
`Bash` and the browser tools).  Pass a custom `system_prompt()` that
matches the tools you provide:

```python
def settings() -> dict:
    return {"kind": "worker", "tool_profile": "none"}

def add_to_tools() -> list:
    return [get_weather]  # the whole tool set: get_weather + finish

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
- `extension_agent_path` is neither `None`/`""` nor a string
- The SEA path is not a `.py` file
- The SEA file does not exist

**Daemon-side errors** (the task starts, then fails with
`result.success == False` and the diagnostic in `result.text`, which
prefixes the message below with `Task failed: AgentFileError: `):

| Condition | `AgentFileError` message |
|-----------|---------------|
| File deleted between client validation and daemon import | `agent script '...' is not an existing Python (.py) file` |
| File raises at import time | `agent script '...' failed to import: ...` |
| `settings()` returns something other than a dict | `agent script '...': settings() must return a dict, got ...` |
| `settings()` raises | `agent script '...': settings() raised: ...` |
| unknown key | `agent script '...': settings() has an unknown key 'x'; known keys: kind, extends, ...` |
| renamed or removed key | `agent script '...': settings() key 'preset' was renamed to 'kind'; run `uv run sea lint --fix` to rewrite the script`, `... key 'inherit' was removed: a `channel` run never inherits ...` |
| `extends` names no command or file, forms a cycle, or names a channel SEA | `agent script '...': extends 'x' is not a registered SEA command; ...`, `... extends chain is a cycle: ...`, `... cannot extend the channel agent script '...'` |
| unknown `kind` | `agent script '...': settings()['kind'] must be one of session, worker, channel; got 'x'` |
| wrong-typed value (`bool` for a number, `int` for a `str`, ...) | `agent script '...': settings()['max_budget'] must be int or float, got bool` |
| non-finite `max_budget` / `timeout` | `agent script '...': settings()['max_budget'] must return a finite number or None` |
| `prompt()` returns an empty string | `prompt() of agent script '...' must return a non-empty string` |
| `settings` bound to a non-callable | `agent script '...': settings must be a function returning a dict, got ...` |
| `prompt`, `system_prompt`, `add_to_tools`, `add_to_system_prompt` or a hook `X` bound to a non-callable | `X of agent script '...' must be a callable, got ...` |
| `X()` raises | `X() of agent script '...' raised: ...` |
| `X()` returns the wrong type | `system_prompt() of agent script '...' must return a string, got ...`, `add_to_tools() of agent script '...' must return a list of tool callables (not a file path), got ...`, `add_to_system_prompt() of agent script '...' must return a string, got ...`, `tool_call_hook() of agent script '...' must return a callable or None, got ...` |
| a value whose own methods raise (e.g. a `str` subclass with a raising `__str__`) | `prompt() of agent script '...' returned a broken value: ...`, `agent script '...': settings()['max_budget'] returned a broken value: ...` |

Overrides are **atomic**: if anything fails, the command keeps all
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

An SEA can also force a specific chat via `settings()["chat_id"]`.


## Model configuration

Use the `model_config` setting to pass custom endpoint URLs, headers,
or sampling parameters:

```python
def settings() -> dict:
    return {
        "model": "my-custom-model",
        "model_config": {
            "base_url": "http://localhost:8080/v1",
            "api_key": "sk-local-key",
        },
    }
```

When `model_config` contains a `base_url`, the model factory bypasses
its normal provider routing and creates an OpenAI-compatible model
pointing at that URL.  The daemon still runs its model-availability
preflight first: `model` must name a generation-capable model from
the bundled catalog or from `$KISS_HOME/MY_MODELS.json` whose provider is
usable (an API key for HTTP providers, the executable on `PATH` for
`cc/*` / `codex/*`), so replace `my-custom-model` above with such a
name; otherwise the task fails with `No model available.  Set at least
one API key in the environment.`


## Complete working example

Below is a self-contained SEA that gives the LLM tools for
managing a SQLite task database.  Its tools come from
`add_to_tools()`, so the LLM can also use the built-in `Bash`, `Read`,
`Write`, etc. alongside the custom database tools.

It defines no `prompt(task)` and its `settings()` names no `model`,
so the caller's prompt reaches the LLM and the caller's model (the
tab's pick, or the daemon's configured default) is used.

```python
# task_manager/task_manager_sea.py
"""SEA for managing a SQLite task database.

Gives the LLM three tools — add_task, list_tasks, complete_task — and
a system prompt explaining how to use them.  The agent runs with the
full KISS Sorcar toolset so it can also read files, run commands, etc.
"""

import json
import sqlite3
import threading

from kiss.core.config import kiss_home


def description() -> str:
    """Return the one-sentence help text shown by ``/task_manager help``."""
    return (
        "Manages a personal SQLite task list (add, list, complete) through three "
        "tools; use it as `/task_manager <request>` or pass its path as "
        "extension_agent_path."
    )


# --- Database setup ---
# Guarded with CREATE IF NOT EXISTS: the file is re-executed from
# source on every run.

_DB_PATH = kiss_home() / "task_manager.db"
_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    """Return a connection to the task database, creating it if needed."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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


# --- Run settings ---
# No prompt(task) function — the caller's prompt is used as-is.
# No "model" key — the caller's model is used.

def settings() -> dict:
    return {
        "max_budget": 1.0,
        "use_worktree": False,  # no code changes expected
    }


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


def add_to_tools() -> list:
    """The database tools, added to the built-in toolset."""
    return [add_task, list_tasks, complete_task]
```

Launch it:

```python
from kiss.server import sorcar

# First run — add some tasks (the prompt reaches the LLM directly)
result = sorcar.run(
    "Add three tasks: buy groceries, review PR #42, write tests",
    extension_agent_path="task_manager/task_manager_sea.py",
)
print(result.text)
print(f"Cost: ${result.cost:.4f}, Steps: {result.steps}")

# Follow-up in the same chat — the agent remembers context
result2 = sorcar.run(
    "Complete 'buy groceries' and show me remaining tasks",
    chat_id=result.chat_id,
    extension_agent_path="task_manager/task_manager_sea.py",
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
    extension_agent_path: str = "",
    use_worktree: bool = True,
    auto_commit: bool = True,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    use_web_tools: bool | None = None,
    classify_tasks: bool | None = None,
    use_memory: bool | None = None,
    is_parallel: bool = True,
    append_to_system_prompt: str = "",
    append_to_prompt: str = "",
    tool_profile: str = "",
    docker_image: str = "",
    inherit_tools: bool = False,
    workspace: str = "",
    provenance: dict[str, str] | None = None,
    timeout: float | None = 3600.0,
    stop_on_timeout: bool = False,
    record_timeout: float | None = None,
    endpoint_file: str | Path | None = None,
    cancel: threading.Event | None = None,
    running: threading.Event | None = None,
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


## Tips

- The client-side `prompt` argument must be **non-empty** even when
  the SEA's `prompt(task)` rewrites it; the client validates before
  connecting.
- `settings()` **may name any subset** of the keys, or be left out
  altogether.  Only pin the ones whose defaults you want to change;
  a kind covers the common bundles.
- Omit the `model` key to keep the caller's model (the tab's pick, or
  the daemon's configured default) rather than hard-coding one.
- Extra tools come only from `add_to_tools()`, returning a **list of
  callables**; the built-in toolset plus yours is the most common
  pattern, `"tool_profile": "none"` gives an exact tool set.
- The SEA is **re-imported from source** on every run.
  Edits take effect immediately without restarting the daemon.
- `max_budget` and `timeout` must be **finite** numbers.  `NaN`,
  `±inf`, or an overflowing value raises `AgentFileError`.
- A contract name bound to a non-callable (e.g. a module-level
  variable named `settings`, `prompt` or `add_to_tools`, including
  one set to `None`) is treated as a broken function and stops the
  task — it is never treated as "absent": membership in the module's
  namespace decides, so only a name the file does not define is
  undefined.  Avoid module-level variables that share a contract
  name's spelling (a constant such as `ADD_TO_PROMPT` or
  `DISPATCH_TIMEOUT_SECONDS` is fine; a variable named `model` is
  harmless too, since the per-field getters are no longer read).
- The SEA and its tools run **in the daemon process**
  with the daemon user's privileges and environment.  Any libraries
  your code imports must be installed in the daemon's Python
  environment.  A tool that runs on the task's worker thread can call
  `kiss.server.agent_state.current_agent()` to get the running agent
  (its `work_dir`, model and usage counters); it returns `None` on any
  other thread.  The bundled `seas/autorouter/autorouter_sea.py`,
  `seas/rsi7d/rsi7d_sea.py` and `seas/skillopt/skillopt_sea.py` use it.
- Put the SEA in a folder named after the command, `xxx/xxx_sea.py`,
  and list that folder's parent in `$KISS_HOME/SEAS.md` (one folder per
  line; blank lines and `#` comments are ignored) to expose it as the
  chat command `/xxx`; `/xxx some text` runs the SEA on "some text"
  directly in the tab's run, and `/xxx help` prints its
  `description()` without running it.  The same name also works as
  `run_agent(agent="xxx", ...)`.  The command is the folder name, so it must match
  `[A-Za-z0-9_-]+`; a loose `xxx_sea.py` placed directly in a listed
  folder is not a command.  Bundled
  `src/kiss/agents/third_party_agents/<name>/<name>_sea.py` scripts take
  precedence over `SEAS.md` folders, later `SEAS.md` lines beat
  earlier ones, and the bundled Sorcar-extending SEAs in
  `src/kiss/agents/seas/` have the lowest precedence, so a `SEAS.md`
  folder can shadow them.  Of the 17 bundled SEA folders, 15 register
  a command and two (`coding`, `sorcar`) are hidden: `/ask` (answers a question about the current task
  in two or three sentences from one `task_context` call over its
  status, spend, progress log and digested persisted events; its only
  tools are `task_context` and `finish`, so it never touches the task's
  files or shell; typed into a running task's
  tab it runs as a side channel that always dispatches the bundled
  `seas/ask/ask_sea.py`, even when a `SEAS.md` folder shadows the
  command, and the task-info panel's Task update is the same agent
  run in-process by `kiss.server.task_update`), `/autorouter` (runs a task on the cheapest model tier
  that will finish it), `/bestrouter` (runs a task on
  `claude-fable-5-1` and has `gpt-6-astra` review it), the hidden
  `coding` SEA (unattended coding in a Docker container for benchmark
  trials; its `settings()` is `{"hidden": True}`, so there is no
  `/coding` command: the trial runners load the generated per-trial
  SEAs, which expose its `ContainerHarness` methods, by path), `/forget`
  (removes a standing instruction from `$KISS_HOME/AGENTS.md`),
  `/git_extract_knowledge` (builds and refreshes a repository's
  knowledge memory and can schedule its daily refresh), `/merge`
  (resolves git merge conflicts and stages the resolved files; commits
  only when asked), `/remember` (appends a standing instruction to
  `$KISS_HOME/AGENTS.md`), `/review_paper` (reviews a research paper for
  a venue, scoring seven dimensions from 1 to 10), `/revise_and_review_paper` (writes a paper with
  `/write_paper`, has `/review_paper` review it fresh, and repeats until
  strong accept or no further improvement; task text carries `Writing:`
  and `Review:` instructions), `/rsi7d` (7-day self-improvement of the
  indexed SEAs from their recorded runs; the task text starts with the
  scope: `/rsi7d all`, `/rsi7d <name> [<name> ...]` for those SEAs only,
  or `/rsi7d --seas-dir <folder> [<name> ...]` for the SEAs of that
  folder, which then are the ones it may edit), `/sh`
  (runs the command with the `bash` tool profile), `/skillopt`
  (optimizes the prompt text of a skill or SEA against an eval set),
  `/task_update` (reports what a running task has done so far),
  `/write` (writes prose for a general audience in concise, professional
  American English that reads as human-written; it defines only
  `description()`, `add_to_system_prompt()` and a `settings()` of
  `{"timeout": 3600.0}`), `/write_paper` (writes
  or revises a research paper), and the plain sub-agent
  (`seas/sorcar/sorcar_sea.py`, a hidden SEA whose settings are
  `{"hidden": True}`, is what `run_agent` runs when its `agent` argument
  is empty or a generic label).  The other modules in that package
  (`coding/coding_test_context.py`,
  `git_extract_knowledge/git_knowledge_index.py`,
  `git_extract_knowledge/git_knowledge_store.py`, `rsi7d/sea_tuning.py`,
  the shared `agents_md.py`) are helpers, not commands: only
  `<name>/<name>_sea.py` folders are registered.  Syntax,
  precedence and the dispatch flow are
  documented in
  [docs/sea-commands.md](https://kisssorcar.github.io/docs/sea-commands.md)
  (source: `website/kisssorcar.github.io/docs/sea-commands.md`).
- A `/xxx text` command has no outer relay run: the tab's run IS the
  SEA run (`agentPath` set to the SEA, `text` as the prompt), with the
  tab's model, budget, chat and settings as the caller's values and
  the SEA's `settings()` on top of them, exactly as for a `run_agent`
  dispatch (see [Precedence](#precedence)).  The SEA is loaded by the
  same `apply_agent_overrides` as any `agentPath` run, so a broken
  script fails the run with an `AgentFileError` before the agent
  starts.
