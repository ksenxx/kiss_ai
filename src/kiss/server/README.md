# Writing Sorcar Extension Agents (SEAs) for KISS Sorcar

## What is a SEA?

A **Sorcar Extension Agent (SEA)** is a small Python file that describes
a specialised version of the Sorcar assistant.  Think of Sorcar as a
capable general employee, and a SEA as a job description plus a rulebook
you hand that employee before a particular kind of job.  The file can say:

- **What the job is.**  The base class says what the run is: `BaseSea`
  is an ordinary session, `WorkerSea` a lightweight worker with no git
  worktree, browser or memory, `ChannelSea` a worker that serves an
  external service.  `settings()` sets the run parameters the SEA
  cares about: the model, the tool profile, the working directory, how
  long a caller waits for it.  A caller may still ask for a different
  value unless the SEA marks the key `locked`; whatever the SEA does
  not set keeps what the caller asked for.
- **How to phrase the task.**  `prompt(task)` rewrites the text the user
  typed into the prompt the agent actually receives.
- **What the agent knows and may do.**  `system_prompt()` adjusts the
  standing instructions; `tools()` adds, removes or replaces the tools
  the agent can call, so a SEA can bring its own tools (a weather
  lookup, a task database) or lock the agent down to a bare shell.
- **What to watch.**  `tool_call_hook` and `llm_call_hook` run before
  each tool and model call the agent makes, so a SEA can guard, log or
  veto actions.
- **What it is for.**  `description()` is the one-sentence help text
  shown for the SEA's chat command.

A SEA only writes down what it changes; everything else is inherited
from a base that already does the right thing, and one SEA can build on
another the way one job description can extend another.  Users reach a
SEA by typing `/<name>` in the chat, by picking it on a tab, by calling
`sorcar.run()` with its path, or when an agent delegates to it with
`run_agent`.  The file runs inside the Sorcar daemon and is re-read from
source on every run, so editing it takes effect immediately.  The rest
of this document is the precise contract behind that picture.

## The contract

A **Sorcar Extension Agent (SEA)** is a Python file that defines one
class deriving from `kiss.agents.seas.base.base_sea.BaseSea` —
directly for an ordinary session, through `WorkerSea` for a focused
tool-bound run on the caller's tree, through `ChannelSea` for an agent
that serves an external service; you pass the file's path as
`sea_path` to `kiss.server.sorcar.run()`.
The daemon executes the file, instantiates the class and threads the
run's configuration through its methods: `settings(settings)` returns
the `run()` parameters the SEA pins (the daemon writes them over the
run's parameters; the rest keep what the caller passed), `prompt(task)`
turns the task text
into the prompt,
`system_prompt(system_prompt)` and `tools(tools)` receive the run's
assembled system prompt and toolset and return the ones to use, and
`tool_call_hook(name, args)` / `llm_call_hook(new_messages)` are
called around every tool and LLM call.  Every method on `BaseSea`
passes its input through (the root layer's one rule of its own is the
`summary` cadence, see
[Customizing every run](#customizing-every-run)), so a SEA defines
only what it changes.  A SEA builds on another by Python inheritance.  `BaseSea` is also the root
layer of every run made from the chat: a plain prompt runs the bare
`BaseSea`, a `/xxx` command or a picker tab runs its SEA on top of it,
and `run_agent` / `run_parallel` children go through it too — so
editing `src/kiss/agents/seas/base/base_sea.py` (and restarting the
daemon, which imports it once) customizes every such run at once (see
[Customizing every run](#customizing-every-run)).  Everything the file defines
executes **in the daemon process** and is re-executed from source on
every run (no `__pycache__` is written) by the one loader
`kiss.agents.sorcar.sea_settings.execute_python_file`.

Each SEA lives in its own folder named after it, `<name>/<name>_sea.py`,
together with the helper modules and data files it needs (the bundled
`src/kiss/agents/seas/sh/sh_sea.py` and
`src/kiss/agents/third_party_agents/slack/slack_sea.py` are laid out this
way).  A SEA exposed as a command must define `description()`,
returning one sentence that says what the SEA does and how to use it.
It is what the chat command `/<name> help` prints (see the
slash-command bullet under [Tips](#tips)).

This tutorial covers every `settings()` key and base class, the other
methods, how `run_agent` and `/xxx` commands run a SEA, error
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

from typing import Any

import requests

from kiss.agents.seas.base.base_sea import WorkerSea


def get_weather(city: str) -> str:
    """Return current weather for a city from wttr.in.

    Args:
        city: City name to look up.
    """
    resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
    resp.raise_for_status()
    return resp.text.strip()


class WeatherSea(WorkerSea):  # a worker: no worktree tools, classifier, browser or memory
    def description(self) -> str:
        return (
            "Reports the current weather in San Francisco from wttr.in; "
            "pass its path as sea_path or send `/weather now` "
            "(any text after the command) once its parent folder is listed "
            "in $KISS_HOME/SEAS.md."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {
            "tool_profile": "none",  # no built-in tool: get_weather + finish only
            "max_budget": 0.50,
        }

    def prompt(self, task: str) -> str:
        # The task is what the caller typed after ``/weather`` or passed
        # to ``run_agent``; the SEA turns it into the prompt body.
        return f"Look up the current weather in {task or 'San Francisco'} and report it."

    def system_prompt(self, system_prompt: str) -> str:
        # Replaces the assembled system prompt.
        return (
            "You are a weather assistant. Use the get_weather tool "
            "to look up weather, then call finish with the result."
        )

    def tools(self, tools: list[Any]) -> list[Any]:
        # ``tools`` is the run's toolset so far (empty under ``tool_profile: "none"``).
        return tools + [get_weather]
```

Launch it:

```python
from kiss.server import sorcar

result = sorcar.run(
    "Paris",  # the task; the SEA's prompt(task) turns it into the prompt
    sea_path="weather/weather_sea.py",
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
   sea_path="agent.py"           ▼
 )                                       seas = sea_layers(agent.py, base=<picker SEA>)
   │                                         │ execute agent.py ONCE, instantiate its
   │ validate path exists                    │   BaseSea subclass (sea_commands.load_sea)
   │ resolve to absolute                     │ settings: fold settings() over the class's
   │ send JSON {"seaPath": "…", …}        │   MRO, base first (WorkerSea / ChannelSea lay
   │ over the local ws:// endpoint           │   their defaults there); type-checked
   │                                         ▼
   │                                     ChannelSea: enter the workspace
   │                                         │ (held until the run ends)
   │                                         ▼
   │                                     apply_sea(cmd, seas)
   │                                         │ evaluate_sea(seas, task, task_id):
   │                                         │   prompt(task) chain ({task_id} filled),
   │                                         │   system_prompt / tools / llm_call_hook /
   │                                         │   tool_call_hook / llm_response_hook folded
   │                                         │   into one hook each
   ▼                                         │ stage one wire field per setting, the
 block, read events ◄───────────────────     │   prompt, and the five daemon-side hooks
                                             │ apply staged overrides to cmd
                                             ▼
                                         agent.run(system_prompt_hook=…, tools_hook=…, …)
                                             │ system_prompt(base + suffix) once, at start
                                             │ tools(built-ins + inherited) once, at start
                                             │ llm_call_hook / tool_call_hook per call
```

1. **Client side** — `sorcar.run()` validates that `sea_path`
   points to an existing `.py` file, resolves it to an absolute path,
   and sends it as `"seaPath"` on the wire.  `None` or `""` means
   no SEA; any other non-string value (including a
   `pathlib.Path`), a non-`.py` path, or a nonexistent file raises
   `ValueError` immediately (before any daemon connection).

2. **Daemon side** — the task runner executes the file once and
   instantiates the one `BaseSea` subclass it defines
   (`kiss.agents.sorcar.sea_commands.load_sea`; a model-picker SEA
   chosen on the tab is loaded too and applied first,
   `sea_commands.sea_layers`).  The settings are folded over the
   class's inheritance chain, base first (`sea_commands.base_settings`:
   each class's own `settings(settings)` receives the dict so far and
   returns the next one — `WorkerSea` and `ChannelSea` lay their
   defaults there, under the subclass's keys), then resolved
   (`kiss.agents.sorcar.sea_settings.resolve_settings`: every value
   type-checked, a channel's locked keys enforced).  The record each
   class returns is checked the same way as it is folded
   (`sea_commands.declared_settings`), so an unknown key or an
   ill-typed value is refused even when a later class writes over it,
   and the error names the method that wrote it (`SEA '...':
   settings()['max_budget'] must be int or float, got str (in
   MySea.settings)`).
   `apply_sea()` (in `kiss.agents.sorcar.sea_apply`)
   then evaluates the SEA on the task (`evaluate_sea`), writes each
   setting into the command dict's wire field and the result of the
   `prompt(task)` chain into `prompt`, and stages the five daemon-side
   hooks (`systemPromptHook`, `toolsHook`,
   `llmCallHook`, `toolCallHook`, `llmResponseHook`), each a fold of
   that method over the chain, base first.  Overrides are staged: they apply atomically
   only after everything succeeds.  A broken script raises
   `SeaError` and the task fails with a diagnostic message in
   `TaskResult.text`.

3. **Run-time hooks** — the hooks never travel the wire (a callable
   cannot).  When the agent starts, `systemPromptHook` is called once
   with the assembled system prompt (the daemon's base prompt or the
   caller's `system_prompt`, plus the caller's `add_to_system_prompt`
   and the channel preamble) and its return value is the run's system
   prompt, after which the executor appends its own notes, including
   the list of tools of a profile other than `full`, a note when the
   browser tools are off, the memory protocol of a run with memory
   and the path of the file open in the editor when the surface sent
   one (`SorcarAgent.run`); the generic executor then adds its
   mandatory operational instructions and the `# Task Settings`
   section, plus the user's `AGENTS.md` when it exists
   (`RelentlessAgent.perform_task`).  A run whose `model_config` sets
   `plain_io: true` gets none of these trailers except the memory
   protocol: the hook's return is the system prompt on the wire, plus
   only what the adapter needs to offer the run's tools (see "Model
   configuration").  The daemon's base prompt is
   `SYSTEM.md`, or the reduced `SYSTEM_LITE.md` for a run with a
   fixed toolset — `tool_profile` `none`, or a profile without the
   `edit` group such as `review`, `bash` or `shell`, chosen without
   consulting the classifier's verdict
   (`sorcar_agent.classification_chooses_prompt`; the daemon's task
   runner still classifies the task before the run when
   classification is on, while `ChatSorcarAgent.run` itself skips
   the classifier for such a run) — and for a
   task the pre-run classifier (`auto_classify`) deems
   simple.  The prompt is what the `prompt(task)` chain returned plus
   the caller's `add_to_prompt`; a run with memory also gets a block
   listing the memory pages relevant to the task text (its first and
   last 2,000 characters when longer than 4,000) appended by the
   executor (`MemoryTools.recall`, at most five pages), not by any
   layer, and a task with attachments gets a note that they are
   inline in the message.  `toolsHook` is called once with the
   built-in toolset (of the run's tool profile; empty under `"none"`)
   plus the tools inherited from the calling task, and its return
   value is the run's toolset (`finish` is always added).  `llmCallHook`
   runs before every LLM call, `llmResponseHook` after every reply
   (before its tool calls run) and `toolCallHook` before every tool
   call.  Nothing is re-imported.


## `settings()`: the run parameters

A SEA pins the parameters of its run with ONE optional method,
`settings(self, settings)`, which receives the dict its base classes
built (`{}` for a direct `BaseSea` subclass, the worker defaults for
a `WorkerSea`) and returns the dict to use: any of the keys below
(`sea_settings.SETTING_TYPES`).  What the run *is* — a session, a
worker, a channel — is the base class, not a key.  Return `settings |
{...}` to add or override keys and keep the base's; return a dict
without a key to drop it.  Every key but `timeout`, `locked` and
`hidden` (`sea_settings.DISPATCHER_SETTINGS`) is a parameter of
`sorcar.run()` and lands on that parameter's wire field
(`sea_apply.SETTING_FIELDS`); the wire name is the keyword in camelCase
(`sea_settings.wire_field`: `use_web_tools` → `useWebTools`), with
one alias kept from the wire protocol's earlier vocabulary
(`auto_classify` → `classifyTasks`, `sea_settings._WIRE_ALIASES`), so
no hand-kept table is needed.  (The prompt suffixes `add_to_prompt`
and `add_to_system_prompt` are not settings; `daemon_client` spells
them `appendToPrompt` / `appendToSystemPrompt`.)
Extra tools have no `run()` parameter at all (a callable cannot travel
the wire): they come from the `tools(tools)` method (see below).  The
bundled `/sh` agent is the whole pattern:

```python
class ShSea(WorkerSea):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker with Bash only, on the real checkout, running SYSTEM_PROMPT."""
        return settings | {
            "tool_profile": "bash",
            "locked": ["tool_profile"],
        }
```

Generated by `uv run sea docs` from `sea_settings.SETTING_TYPES`; the
`run()` default of a key the script leaves out is the daemon's.
`sea docs [--check] [pages...]` rewrites every `<!-- sea-docs: <table> -->`
block (`sea_docs.TABLES`: `precedence`, `settings`, `bases`, `channel`,
`options`, `commands`) of the pages it is given, by default those of
`sea_docs.GENERATED_FILES` (`docs/sea-commands.md` and `docs/cli.md` of
the website and this file) that exist, so an installed package without
the website checks only this file; `--check` rewrites nothing and exits
1 when a block is stale, and a page named on the command line that does
not exist is reported as `missing: <path>` and fails the run either way.

<!-- sea-docs: settings -->
| Key | Type | `run` wire field | Meaning |
|---|---|---|---|
| `work_dir` | `str` | `workDir` | The directory the run works in; default: the calling task's or the tab's. A relative path is a path under the calling task's directory (the tab's for `/<name>`), in a SEA's settings and in a call's `work_dir` option alike. |
| `model` | `str` | `model` | The LLM model, a catalogue name or a model-picker SEA; default: the caller's (`""`, like `None`, is no override — true of every string key). |
| `chat_id` | `str` | `chatId` | The chat the run's events go to; default under `run_agent`: the calling task's chat, or a new chat when nothing is inherited (a channel, an `inherit: false` call); a `/<name>` run keeps the tab's chat. |
| `max_budget` | `int \| float` | `maxBudget` | USD budget of the run, a finite number; default: the caller's share or the daemon's default. |
| `model_config` | `dict` | `modelConfig` | Model configuration dict passed to the LLM (temperature, ...). `base_url` (with `api_key`) routes every model call of the run, in a tab and headless alike, to that OpenAI-compatible endpoint (a proxy the SEA starts, say); `model` must still name a catalogued model so the calls are priced. `plain_io: true` leaves the framework's own additions out of what the model reads: the system prompt is the SEA's `system_prompt` return alone (no `# Restricted tool profile` or `# Web tools are off` note, no `# MOST IMPORTANT INSTRUCTIONS`, `# Task Settings` or `AGENTS.md` trailer; the memory protocol stays while the run has the memory tools, and a CLI-backed model's text tool catalog stays) and tool results carry no usage line (`Steps: ..., Context: ...`). `coding_gate: false` makes the run no coding run for the finish gate: no verifier or review stage, whatever its edits (a sync resolving conflicts), so its `finish` is not gated at all. Replaces the caller's `model_config` whole, except that a dict of such framework flags alone is added to the configuration that applies anyway (a settings-panel endpoint stays). |
| `use_web_tools` | `bool` | `useWebTools` | Give the run the browser tools (daemon default: on). |
| `auto_classify` | `bool` | `classifyTasks` | Let the pre-run classifier decide whether the task is simple (lite prompt) (daemon default: the persisted setting). |
| `use_memory` | `bool` | `useMemory` | Give the run the `memory_*` tools (daemon default: the persisted setting). |
| `compaction` | `bool` | `compaction` | Let the run compact its context: old, large tool outputs and arguments become short stubs once the context grows large (default: on). `false` turns it off for the run; the daemon's global Compaction setting (the settings panel's checkbox) off turns it off for every run, whatever a SEA or a call passes. |
| `tool_profile` | `str` | `toolProfile` | The run's toolset: `review`, `bash`, `shell+edit`, ... (default: the full toolset). |
| `docker_image` | `str` | `dockerImage` | Run inside this Docker image (default: the host). |
| `timeout` | `int \| float` | — | Seconds a `run_agent` / `run_parallel` call blocks for the run: the call's `timeout` argument wins, then this setting, then 3600. When it expires the run is not stopped: it keeps going as an `agent_job` and the call returns its job id (a job still running when the calling task ends is killed). Ignored by `/<name>`. |
| `locked` | `list` | — | Keys an explicit `run_agent` / `run_parallel` argument or option may not change: a differing value is an error. |
| `hidden` | `bool` | — | `True`: the script is no `/command` and no `run_agent` agent name (loadable by path and as a base class only). It must be the literal `True` in the class's `settings()` because the command registry reads it from the source without running the script (a computed value is ignored). |
<!-- /sea-docs -->

A key whose value is `None` is dropped and behaves like an absent
key — as is a `model` of `""` — so the caller's or the persisted
value stands (a `WorkerSea` subclass returning `settings |
{"use_memory": None}` writes "no override" over the worker default;
leave the key out to keep the default).  A `bool` key given a
non-`bool`, an `int`/`float` key given a `bool`, an unknown, renamed or
removed key (`kind`, `channel` and `allow_fan_out` among the removed
ones), a non-finite `max_budget` / `timeout`, or a `timeout` of zero
or less stops the task with a diagnostic (see
[Error handling](#error-handling)).

What the source readers — the registry's `hidden` check
(`sea_settings.declares_hidden`) and `sea lint`'s key checks — take
for a settings dict is settled by one rule, `sea_settings.settings_dicts`:
the dict literals of the `settings` method whose entries reach its
return value — returned as they are, merged with `|`, unpacked with
`**`, chosen by a conditional expression, or assigned to (or
`update`d into) a local name the return value is built from in those
same ways.  The statements are read in order: a name holds what the
last assignment at the method's top level gave it (an earlier dict it
held is data then, unless the new value is built from the name, `mine
= mine | {...}`), plus what an assignment under an `if` or a loop,
which may or may not run, an `update` or a `|=` added since, and a
`return` reads the names as they are where it stands; the entries
are laid in the order Python evaluates them (`_settings_entries`), so
of two literal values for one key the last one written wins in
`settings_literals`, as it does at run time, and a computed value
written after a literal drops the key, as it overwrites the literal.
Every `def settings` in a class body of the file is read, a helper
class's too (`sea_settings.sea_methods`), since a mixin may reach the
SEA class through an alias or a call; a `"hidden": True` literal
that one of them writes but the script's folded `settings()` does not
return is the linter's `hidden-not-returned` finding.  A dict inside the *value* of another dict's key
(`model_config`'s contents, however deep) and a dict the method only
reads a value from (`info["model"]`) are data, not settings, so the
linter's rewrites never touch their keys.

The prompt itself is not a setting: `settings()` is data, text and
code come from the other methods.  Two optional methods are the only
prompt surfaces:

| Method                           | Returns | Effect                                                                 |
|----------------------------------|---------|------------------------------------------------------------------------|
| `prompt(self, task)`             | `str` (non-empty) | the prompt body for the task text the run was submitted with (the text after `/xxx`, the `run_agent` task, the `sorcar.run()` prompt); every `{task_id}` of the final text is replaced by the calling task's id (the command's `parentTaskId`, `""` without one); without it the task text is the prompt |
| `system_prompt(self, system_prompt)` | `str` | receives the run's assembled system prompt (the base prompt plus the caller's suffix and the channel preamble) and returns the one to use: `return SYSTEM_PROMPT` replaces it, `return system_prompt + "\n\n" + PROTOCOL` appends to it |

### Base classes

The base class says what the run is.  `BaseSea` lays nothing;
`WorkerSea(BaseSea)` lays `WORKER_DEFAULTS` under its subclass's keys;
`ChannelSea(WorkerSea)` adds the scratch `work_dir` and the behaviours
of the next section (`sea_settings.base_class_defaults()` is the
table).  A subclass that writes one of those keys overrides the
default, as with any base.

<!-- sea-docs: bases -->
| Base class | Lays | Use |
|---|---|---|
| `BaseSea` | nothing | The default: an ordinary Sorcar session with the caller's or the user's settings (`/write`, `/write_paper`, `bestrouter`). |
| `WorkerSea` | `auto_classify=False`, `use_web_tools=False`, `use_memory=False` | A focused tool-bound run on the caller's tree as it is: lays the worker defaults (no classifier, no browser, no memory) under the SEA's own keys and drops the `create_worktree` / `commit_and_merge` tools (`/sh`, `/ask`, `/merge`, `/remember`, `/forget`, `/task_update`). |
| `ChannelSea` | `auto_classify=False`, `use_web_tools=False`, `use_memory=False`, `work_dir='<home>/channel_work'` | A worker that serves an external service (Slack, email, cron), not the caller's project; every behaviour this adds is listed in the channel table. The command registry reads the base name from the source, so a channel derives from `ChannelSea` by that name. |
<!-- /sea-docs -->

### Channels: `class XSea(ChannelSea)`

A SEA that serves an external service (Slack, email, cron) derives
from `ChannelSea`, by that name (the command registry reads the base
name from the source).  Every bundled channel is `class
XSea(ChannelSea)` with no `settings()` method at all; cron is
`CronAgentSea(ChannelSea)` with `work_dir: cron_work_dir()`.  The base
class is the one thing with behaviour beyond a value passed through;
every behaviour is listed in `sea_settings.CHANNEL_BEHAVIOURS`,
rendered here, and the module named in each row is where it is
enforced.

<!-- sea-docs: channel -->
| Behaviour | What deriving from `ChannelSea` does |
|---|---|
| worker | It is a `WorkerSea`, and the worker keys are locked, so no call may give it the classifier, the browser or memory (`resolve_settings`). |
| scratch directory | It runs in the shared `<home>/channel_work` scratch directory unless its own `work_dir` says otherwise (`/cron` runs in `<home>/cron/work`); `work_dir` is locked, so it never works in the caller's project (`resolve_settings`); a cron prompt job that runs a channel command (`/slack ...`) therefore passes no `work_dir` option unless the job names a directory of its own (`cron_agent`: the scratch directory is already the dispatch's default, and an explicit value would be refused by the lock). |
| no inheritance | Its `run_agent` dispatch takes nothing from the calling task: not the chat, model, budget share, container or exec bridge, or prompt suffixes; the `inherit: true` option is refused (`agent_dispatch`). |
| workspace | It holds its channel workspace — the `run_agent` option `workspace`, default `default`, the account its tools load credentials for — from before its `tools()` run until the run ends; `workspace` is refused for any other SEA (`sea_apply.channel_workspace`, `agent_dispatch`). |
| preamble | The channel preamble is appended to its system prompt before its own `system_prompt()` runs (`sea_apply.CHANNEL_PREAMBLE`). |
| listed as a channel | A third-party SEA folder whose class derives from `ChannelSea` by that name is a channel agent: `run_agent(agent="<folder>")` finds it by name and the `run_agent` tool's description lists it among the installed channels (`agent_dispatch.available_channels`, `declares_channel`). |
<!-- /sea-docs -->

The channel preamble (`sea_apply.CHANNEL_PREAMBLE`, "You are the
{name} agent: this session already has the {name} tools ... Never call
run_agent here ...", `{name}` being the script's file stem without
`_sea`) is appended to the system prompt suffix the SEA's
`system_prompt()` receives.  The workspace (the account the channel
tools load credentials for, exported as `KISS_CHANNEL_WORKSPACE`) is
held from before `tools()` runs — the tools bind the credentials of
the workspace active at that moment — until the run ends, whether the
run came from `/slack ...`, `run_agent("slack")` or
`run_agent(".../slack_sea.py")`; a run whose workspace differs from a
running channel task's waits (bounded by
`channel_workspace.WORKSPACE_WAIT_TIMEOUT_SECONDS`) instead of handing
that task the wrong credentials.  A channel is not meant as a base
class (see [Extending a SEA](#extending-a-sea-python-inheritance)).  A
channel SEA is three methods and its base class:

```python
class SlackSea(ChannelSea):
    def description(self) -> str:
        return "Messages, channels, users, reactions and search in a Slack workspace ..."

    def system_prompt(self, system_prompt: str) -> str:
        return system_prompt + "\n\n" + SlackAgent.channel_system_prompt

    def tools(self, tools: list[Any]) -> list[Any]:
        workspace = os.environ.get("KISS_CHANNEL_WORKSPACE", "default") or "default"
        return tools + SlackAgent(workspace=workspace)._get_tools()  # the authenticated API tools
```

### Extending a SEA: Python inheritance

A SEA refines another by deriving from its class.  Import the bundled
class, or ask the registry for it by command name or path with
`kiss.agents.sorcar.sea_commands.sea_class(spec, relative_to="")`
(a registered name such as `"bestrouter"` or `"sh"`, or a path with
a `.py` suffix or a separator, resolved against the directory
`relative_to`, normally `Path(__file__).parent`, else the current
directory):

```python
# strict_sh/strict_sh_sea.py
from typing import Any

from kiss.agents.seas.base.base_sea import ALLOW, Verdict, refuse
from kiss.agents.seas.sh.sh_sea import ShSea          # or: sea_class("sh")


class StrictShSea(ShSea):
    def description(self) -> str:
        return "Runs a shell command like /sh, read-only, on a small budget."

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"max_budget": 0.25}          # ShSea's keys stay

    def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
        if name == "Bash" and "rm " in str(args.get("command", "")):
            return refuse("Blocked: read-only run")
        return ALLOW
```

The launcher applies every class of the inheritance chain that defines
a method, base class first (`sea_commands._chain` walks the MRO), so a
method must **not** call `super()`: the base's `settings()` has already
run when the subclass's receives the dict.  Per method:

- `settings(settings)`: each class receives the dict so far and returns
  the next; a later class's key wins and a key it leaves out of its
  return is dropped.  Return `settings | {...}` to keep the base's keys.
  Every class's return is checked as the final record is (known keys,
  typed values), so an ill-typed value is refused even when a later
  class overwrites it.
- `prompt(task)`: chained; the subclass's receives what the base's returned.
- `system_prompt(system_prompt)`, `tools(tools)`, `llm_call_hook(new_messages)`:
  folded the same way, base first; each class receives the previous
  result and returns the next.
- `tool_call_hook(name, args)`: the classes are asked in order and the
  first refusal (`refuse(text)`) wins; `ALLOW` allows, and the notes
  of the allowing verdicts (`allow_with_note(text)`) are joined in
  chain order onto the result.  The model never
  reads a refusal bare: the agent returns `REFUSED: the <tool> call did
  NOT run and had no effect. <text>` (`tool_verdict.refusal_result`),
  and the next allowed call of another tool gets a reminder appended
  to its result to re-issue the refused call (a call of the refused
  tool itself settles it silently; `finish` carries no reminder).
- `description()`, `register_as_model()`, `on_picked_as_model(work_dir)`:
  ordinary Python inheritance; the most derived definition is the one
  called.

`BaseSea` heads every chain, so its methods run first on every run.
They pass their input through, with one rule of the root layer's own,
the `summary` cadence (`base_sea.SUMMARY_EVERY_STEPS`, 10): `tools()`
resets the run's step count and records whether the toolset holds the
`summary` tool (`self.has_summary_tool`), `llm_response_hook` reads
each reply's `step` (the `Steps:` count of the tool-result footer, which
restarts with every continuation session) and marks a summary due at
every 10th step of a run that has the tool, and `tool_call_hook` then
lets every call but `summary` and `finish` run with
`base_sea.SUMMARY_DUE_NOTE` appended to its result ("A summary is due (one
is owed at every step that is a multiple of 10): make your next tool call
summary(description=...), ...") until `summary` runs (an allowing
verdict with a note: `allow_with_note`).  A SEA whose own `tools()` drops `summary` should set
`self.has_summary_tool = False` there (every bundled SEA only appends
tools).  `system_prompt` is folded exactly
like `prompt`: each class receives the text so far and what it returns
is the text, whether it appended to it or replaced it; the last
return is the run's system prompt, with the executor's own notes and
its operational / Task Settings instructions appended after it (see
"Run-time hooks" above) unless the SEA's `model_config` sets
`plain_io: true`, which appends only the memory protocol (while the
run has the memory tools) and, for a CLI-backed `cc/*` / `codex/*`
model, the `## Available Tools` text catalog (a native adapter sends
the tool schemas in the request instead).  Neither method's return reaches a
sub-agent: a `run_agent` / `run_parallel` child inherits only the
caller-supplied `system_prompt` and `add_to_system_prompt` options and
its own layers build its prompt, so a rule a layer appends is stated
once per run without any deduplication.

### Customizing every run

Every run made from the chat — a plain prompt, a `/xxx` command, a run
on a model-picker tab, a `run_agent` or `run_parallel` child — goes
through `src/kiss/agents/seas/base/base_sea.py` as its first layer.  To
change what all of them do, edit that file's methods and restart the
daemon (the file is imported once, unlike SEA files, which are
re-executed on every run):

```python
class BaseSea:
    def settings(self, settings):
        return settings | {"use_memory": True}   # subject to sea_settings.PRECEDENCE_RULE

    def system_prompt(self, system_prompt):
        return system_prompt + "\n\nAnswer in British English."

    def tools(self, tools):
        return tools + [my_house_tool]

    def tool_call_hook(self, name, args):
        if name == "Bash" and "rm -rf" in str(args.get("command", "")):
            return refuse("Blocked: destructive command")
        return ALLOW
```

(The file's real `tools` and `llm_response_hook` bodies reset and
read the run's step, `BaseSea.step`, from each reply's `step`, and its
`tool_call_hook` enforces one rule only, the ten-step `summary`
cadence described below; that is all the logic in them, so keep
those lines when you edit the bodies and add yours around them.)
A plain chat prompt still names no SEA and runs the bare `BaseSea`; a
`run_agent` or `run_parallel` call with an empty `agent` runs the
hidden `seas/sorcar/sorcar_sea.py` and is reported as `sorcar`.
`/xxx check` lists the methods a SEA adds on top of `BaseSea`, not the
base's own.

A model-picker SEA chosen on the tab (`bestrouter`, `autorouter`) is
applied before the SEA of every daemon-run task whose model names the
picker (`sea_commands.sea_layers`, `task_runner._picker_sea`): a
`/xxx` command typed into that tab, or a `run_agent` call that passes
the picker's name as `model`, runs its own SEA on top of the picker's
model, and the picker's methods are folded first, so a
SEA whose `system_prompt()` appends keeps the picker's routing
protocol while one that replaces the prompt (`/sh`) drops it.  A
`run_agent` or `run_parallel` child that inherits the tab's model
inherits the real model the picker resolved to, not the picker layer.
A class both share (a SEA that derives from the
tab's picker class, or a common ancestor) contributes once, in the
picker's place.  A `model` setting that names a picker entry is
resolved the same way: the run gets the picker's model (the default
model when it names none), never the picker's name.  A file must
define exactly one subclass of `BaseSea` of its own (an imported base
class does not count); none or several is an error (`load_sea`).  Do
not derive from a bundled channel SEA: the subclass would inherit the
channel's base class, workspace handling and credential-bound tools.

### Precedence

One rule decides every setting of a dispatched run
(`sea_settings.PRECEDENCE_RULE`, enforced by `locked_conflicts`; this
block is generated from the constant):

<!-- sea-docs: precedence -->
> For every setting of a sub-task: what the call passes explicitly (a `run_agent` / `run_parallel` argument or `options` key) wins, then the SEA's `settings()`, then what the calling task passes on (and, for `/<name>`, the chat panel's persisted settings), then the user's defaults. A SEA may list keys in `locked`: a call that passes a different value for a locked key is refused with an error, never silently overruled.
>
> For example, `/sh` declares `{"tool_profile": "bash", "locked": ["tool_profile"]}`, so `run_agent(agent="sh", task=..., tool_profile="review")` is refused with `Error: sh: the script locks tool_profile='bash' (asked for 'review')`, while `run_agent(agent="sh", task=..., model="gpt-5")` runs it with that model: `model` is not locked, so the explicit argument wins.
<!-- /sea-docs -->

Example of a refused call: `Error: sh: the script locks
tool_profile='bash' (asked for 'review')`.  A `/<name>` run in a tab
and a `run()` call with an `sea_path` pass nothing
explicitly (unless the `run()` call marks values `"explicit"` in its
`provenance` argument, as the dispatcher does), so there the SEA's
settings apply over the tab's or the client's values.  The `ran:` line of every sub-task result reports the
outcome: `inherited=` names the keys taken from the calling task,
`pinned=` the inherited or default values the SEA's `settings()`
replaced (`pinned=use_web_tools(True->False)` for `/sh` run from a
task with the browser); an explicit argument never appears there,
because it either won or was refused.  A `tools=review(inferred)`
says nobody named the profile and the sub-agent got `review` because
it is a reviewer (see `run_parallel`'s `tool_profile`).

Inheritance (`agent_dispatch.inherit_from_parent`) applies to every
dispatched script that is not a channel, unless the call's `inherit`
option is `false`: the `run_agent` or `run_parallel`
arguments the caller leaves empty are first filled from the calling
agent: its model (and its
`model_config`, but only when the sub-task runs the model the caller
was launched with and the script's `model` setting does not pick the
model, because the script names none or the call passes `model`
itself: a script-chosen model runs with default provider routing),
its remaining budget, less what the still-running sub-tasks of its
*other* `run_agent` / `run_parallel` calls hold, divided by N+1 for
N sub-tasks (half for one `run_agent` call; the last share stays
reserved for the caller; a `wait="false"` child or a job left behind
by an expired `timeout` keeps the budget it was given reserved until
it finishes, and one whose dispatch ended with the sub-task possibly
still running — a Stop the daemon did not confirm, or a daemon
connection lost once the `run` was sent (`AgentJob.left_running`,
`agent_dispatch.may_still_run`) — keeps what it had not spent
reserved for the rest of the caller's run, since nothing charges its
later spend, and stays the caller's job even once its result is read,
among the ones the caller's finish gate names before a merge removes
the worktree under them (`live_agent_jobs`): an explicit `max_budget`, else the SEA's own
`max_budget` setting (the value the daemon pins the inherited share
to, `sea_apply.apply_run`, so that value, not the share, is what the
sub-task can spend, up to the caller's remainder less what the live
jobs of its other calls hold: the dispatch sends that bound as
`budgetCap` (`agent_dispatch.budget_cap_of`,
`daemon_client.run_command`; none for an `inherit: false` call) and
the daemon lowers the effective `maxBudget` to it after the SEA's
settings have applied, `sea_apply.cap_budget`, whoever named the
budget, the call, the SEA or the share, the SEA's settings, the
provenance marks and the `ran:` line untouched; the job's
reservation is lowered with it and bounds the cap in turn, never
more than what the job holds, since a remainder grown meanwhile may
already be granted to a call registered since; the budget a call or
its SEA names is admitted in the critical section that registers the
job, `_register_and_start`, lowered to the remainder or the job
refused there before its thread starts, finished with `Error: the
<name> agent task could not run: <BudgetExceededError>`, so a request
never counts against another call's admission before its own and two
calls registered before either ran cannot refuse each other over one
remainder; the children of one
`run_parallel` call, which do not reserve against each other, are
each capped at the same remainder), is on its `AgentJob.budget` from registration
(`new_agent_job`), an inherited share once the job's thread resolves
it, and `agent_dispatch._inherited_budget` reads the live jobs, sums
their reservations (`reserved_budget`) and records the new share all
under the job registry's lock, so two concurrent children never each
get the whole remainder, the second one sees the first one's share,
and a sub-task that finishes in between is re-read rather than counted
twice or dropped; the children of one `run_parallel` call share an
`AgentJob.call_id` and never reserve against each other, as the split
already divides the remainder among them; a call made while running
children hold all of it is refused with `BudgetExceededError`; a
read-only review the call gives no `max_budget` — `tool_profile="review"`,
or no profile and a task that reads as a review without asking for
changes — gets at most `fanout_guard.review_budget(work_dir)`, $5
plus $5 per thousand lines changed in its work directory against
`HEAD`, untracked files included (`changed_lines`: only the files
under the work directory count, also when it is a subdirectory of
its repository, and before the first commit the diff is against the
empty tree), a cap on the inherited share
(`budget_cap` of `inherit_from_parent`, applied only when the budget
is inherited) that never raises it and that the dispatcher applies
before the SEA's own `max_budget` setting, which still takes
precedence; a review dispatched to another `machine` is not capped
this way at all — its `work_dir` is a path on that machine, whose
diff cannot be measured here, so it keeps the ordinary inherited
share), its
chat id (so the sub-task sees the conversation's earlier
tasks and results), its own replacement system prompt and appended
system-prompt text (the `system_prompt` / `add_to_system_prompt`
its run was given, so a run's extra system instructions constrain its
whole task tree; the classifier's
lite-vs-full choice is not among them, the sub-task is classified on
its own), the text appended to its own prompt, the extra tools its
own SEA added (so the tools the inherited system prompt
refers to exist; not into a run on the `none` profile), its web-tools
and memory settings,
and its live Docker container (`container:<id>`) or exec bridge
(`exec:<url>`), so a sandboxed task's children never reach the
host.  A sub-task starts
in the caller's current directory (the caller's worktree once it has
called `create_worktree`) and decides on its own whether to call
`create_worktree` before editing; the Task Settings block of its
system prompt names a scratch directory of its own, a private folder
outside the work tree (`<system temp dir>/kiss-sub-…`,
`SorcarAgent._scratch_dir`; see [Setting semantics](#setting-semantics)),
so siblings in one work directory do not share a scratch file; only a
sub-agent whose container cannot hold such a folder gets the work
tree's scratch directory instead.  `run_parallel` children are
`run_agent` sub-tasks started by the same code
(`agent_dispatch.plan_run_agent`), so this is their table too (the
precedence rule above decides between the
inherited values and the SEA's).  A channel SEA (`agent="slack"`,
`agent="cron"`, ...) inherits none of these: it runs on the daemon's
default model and budget unless the call passes its own, and
`ChannelSea` locks its `work_dir` and the worker keys (`auto_classify`,
`use_web_tools`, `use_memory`), so a call whose `options` contradict
one is refused.

The model of the sub-task is judged against the caller's before it
starts (`agent_dispatch.family_verdict`).  A sub-agent on the caller's
own model at the caller's own reasoning effort
(`model_families.same_model_and_effort`: the route does not make a
model another one, so `openrouter/anthropic/claude-fable-5.2-max` is
`claude-fable-5-2-max` and a Bedrock model or inference-profile id is
the Claude model behind it; the effort does: a `reasoning_effort` in
the model configuration, else the catalog effort of the name, else
the one its suffix names) is the same model with less context (the benchmark
trials of 2026-10-31 that fanned out passed 29% against 37% without),
so inside a task it is refused (`fanout_guard.same_model_conflict`,
prefixed with the resolved SEA's name, `sorcar` for an empty `agent`:
`Error: sorcar: a sub-agent on <model> is this task's own model at its
own effort, the same model with less context, and <shortfall>. Ways
out: do the work inline in this task; or name the disjoint
files/commands in each task of a run_parallel; or pass another model
or effort (model="<name>").`; `<name>` is the first of the strongest
configured model of another family, the caller's model at the nearest
other effort and the strong models of the caller's own family that
the rule lets through as the sub-task would run it
(`model_families.other_worker_model`: with its own `model_config`,
else with the caller's when the name is the caller's launch model),
so no effort alias where that configuration sets the effort, never
the caller's own name, and the clause ends at `effort.` where
nothing configured differs) unless (i) the call is a fan-out of two or more tasks
over disjoint named pieces of work (`undivided_work` /
`named_pieces`: every task names a slash path, a directory ending in
`/`, a dotted module or file name, a URL or a backticked command, and
no piece is named by two tasks; pieces under `tmp/` or under a
sub-agent's private `kiss-sub-…` folder (`is_scratch_piece`) and, in a
fan-out of three or more, pieces every task names (shared
instructions) do not count; a directory a task also names a more specific piece under
is context for that piece, not a claim on the directory, so `src/x/`
with `src/x/a.py` in one task and `src/x/` with `src/x/b.py` in the
other is no overlap, while a piece two tasks claim, and a directory
one task claims whole while the other claims a piece under it
(`src/x/` against `src/x/b.py` or `src/x/nested/`), are; the
shortfall names the task that names nothing of its own, or every pair
of tasks sharing a piece and up to three of the pieces each, in one
clause), (ii) the caller's own
task asks for sub-agents in so many words (`asks_for_sub_agents`:
"sub-agent", "sub-task", `run_parallel`, `run_agent`, "parallel
agents", "adversarial testing" / "training", "second opinion" or
"reviewer", case-insensitive), (iii) the call or the SEA's `model`
setting names another model or another effort of the caller's model,
whatever its vendor (the rule compared vendors until 2026-11-11 and
refused 29 launches of Fable 5.2 max by an Opus 5.5 medium caller), or
(iv) the sub-task is a review.  A family is the vendor that trained
the model whatever the route (`model_families.family_of`: a Claude
model called directly, through Bedrock, OpenRouter or the `cc/` CLI
is one family).  A review that would run on the caller's family and that the
call did not pin (`tool_profile="review"`, or no profile and a task
that reads as a review without asking for changes) runs on the
strongest configured model of another family instead
(`second_opinion_model`, `model_families.other_family_model`, chosen
among `SECOND_OPINION_MODELS` — `gpt-6-astra`, `claude-opus-5-5`,
`gemini-3.1-pro-preview`, their `openrouter/...` twins,
`openrouter/deepseek/deepseek-v4-pro`, `moonshotai/Kimi-K3`,
`deepseek-ai/DeepSeek-V4-Pro`, `cc/opus`, `codex/default`, `glm-4.7`,
`kimi-k3` and, last, the strongest `bedrock/` model the catalog
registers (the bare `BEDROCK_PREFIX` entry, resolved by
`registered_bedrock_model(cheapest=False)` at lookup time) — by the
first of another family whose provider has a credential; same-model
reviewers were right 36-37% of the time
in the audit), and the `ran:` line says so after `model=`:
`(another family than the caller's, for a second opinion)`, or `(no
other family configured)` when the review had to stay on the
caller's family, also under a daemon that pins every task to its
model.  Nothing is judged for a channel, an `inherit: false` call
(neither inherits a model), a sub-task on another `machine` (work
the caller cannot do inline here), or when either model is unknown.

### `description()` — mandatory, not a run parameter

| Method              | Return type         | Used by                                   |
|---------------------|---------------------|-------------------------------------------|
| `description(self)` | `str` (one sentence, non-empty) | `/<name> help` in the chat (`kiss.agents.sorcar.sea_commands`) |

Every SEA exposed as a command must define `description()`, returning
one sentence saying what the SEA does and how to use it (the `BaseSea`
default returns `""`, which is fine for a hidden base class).  It
is not a run parameter: `apply_sea()` ignores it and it
never reaches the command dict.  The slash-command registry calls it
when the prompt is `/<name> help` (the bare word `help`, any letter
case, nothing after it; `sea_commands.help_text_if_command`): the
daemon does not run the SEA and calls no model, it loads the class,
calls `description()` and posts the returned sentence as the task
result.  A `description()` that raises or returns something other than
a non-empty string yields a diagnostic instead
(`sea_commands.sea_description`).

A plain task the user wrote (typed, or an in-chat cron job's prompt)
also gets the sentence of every command it names, so a task told to
"use /discover" knows what the command does and how to call it: a
`/<name>` in its text or in the chat's earlier tasks (the ones
`build_chat_prompt` loads, even one the history digest drops for size;
at a text's start or after whitespace, an opening bracket, a quote or
Markdown emphasis; `~/x`, `a/x`, `/x/`, `/x.py`, `/x?q=1`, `/x#a` and
`[t](/x)` do not count) adds a ``- `/<name>`: <description()>`` line
under `# Slash commands the user named` after the task
(`sea_commands.named_commands_help`, called by
`ChatSorcarAgent.build_chat_prompt`).  A bare `/x` that fell back to a
plain prompt gets no block, so its SEA file stays executed once, and a
SEA whose file or `description()` is broken is left out.  A SEA run and
a `run_agent` sub-task get no block either: they see the same chat, and
a `/discover` run told about `/discover` could call itself.

### `timeout` — optional, for SEAs that run longer than an hour

A `run_agent` or `run_parallel` call waits for its sub-task(s) for a
bounded time (`agent_dispatch.resolve_timeout`): an explicit `timeout`
argument wins; an empty one takes the SEA's `settings()["timeout"]`
when that is positive; with neither,
`DEFAULT_DISPATCH_TIMEOUT_SECONDS`, 3600 s.  When the bound expires
the sub-task is NOT stopped: it keeps running as an `agent_job` and
the call returns the job's id (`agent_job(id, "wait")` collects the
result, `agent_job(ids, "wait_any")` the result of whichever of several
finishes first, `"kill"` stops it; a job still running when the calling task
ends is killed).  A SEA whose runs may take longer than an hour
declares it so a caller is not handed a job id too early (`/write_paper` and `/write_patent`
6 h, `/review_paper` and `/review_patent` 2 h, `/revise_and_review_paper` 24 h); `/write`
pins the default
explicitly:

```python
DISPATCH_TIMEOUT_SECONDS = 3600.0


class WriteSea(BaseSea):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"timeout": DISPATCH_TIMEOUT_SECONDS, "model": "bestrouter"}
```

The setting is read for every `run_agent` and `run_parallel` dispatch,
whatever spelling `agent` used (a path, a registered command name, a
channel name or `cron`): the dispatcher resolves the agent to its
script and reads that script's merged `settings()` before it waits.
The timeout has this one meaning everywhere — the seconds the call
blocks before returning the job id(s) — and never stops a child.  Not
a run parameter: it has no wire field, and a `/xxx text` command,
which runs the SEA directly in the tab's own run, has no dispatch
timer at all.

### Running a SEA headless: `run_sea()`

`kiss.agents.sorcar.headless.run_sea(sea, task, ...)` runs a SEA in
the calling process — no daemon, no UI — through the daemon's SEA
evaluation and run path (`sea_layers` + `evaluate_sea` +
`SorcarAgent.run` with the SEA's settings, prompt and hooks) but
without a tab's settings: the web tools are off unless the SEA sets
`use_web_tools: true`, and a `tool_profile` of `none` strips the run
to `finish` plus what its `tools()` returns, as on the daemon.  It first imports `$KISS_HOME/api_keys.env` into
`os.environ` without writing anything (`load_api_keys_readonly`; a
missing store is empty, so keys already in the environment serve),
then applies the settings saved in `$KISS_HOME/config.json` as the
daemon does at start-up (`apply_config_to_env(load_config())`: the
default `max_budget`, the Claude Code / Codex subscription switches
and the global Compaction switch, `compaction`), so a headless run
honours what the user saved instead of the declared defaults.
It returns a `SeaRunResult`: the
`result` text, `success`, the `error` that ended the run (`""` when it
returned), `budget_used`, `total_tokens_used`, `total_steps`, and
`trajectory_paths`, every trajectory file the run saved — the agent's
own and those of the helper agents it created (session executors,
the summarizer, the task classifier, ...), collected as they were
written, not guessed from their names; `save_trajectories=False`
writes none.  The collector is `kiss.core.base.collecting_trajectories`:
`with collecting_trajectories() as scope:` records in `scope.paths`,
in write order, every trajectory saved by a `Base` agent constructed
in the block — on the entering thread or on a tool thread running
under a copy of its context, so helper agents are included — and
`collecting_trajectories(save=False)` makes those agents save none;
scopes nest and the innermost applies; a child dispatched through
the daemon (`run_agent`, `run_parallel`) is another task, outside the
scope.  A benchmark harness or a training loop uses it in place
of the `SorcarAgent` / `evaluate_sea` internals:

```python
from kiss.agents.sorcar.headless import run_sea

outcome = run_sea("weather", "What is the weather in Paris?", max_budget=0.5)
print(outcome.success, outcome.result)
for path in outcome.trajectory_paths:
    path.unlink()
```

`uv run python scripts/run_sea.py <sea> "<task>" [--work-dir DIR]
[-m MODEL] [-b USD] [--json] [--no-trajectories]` is the command-line
form: `<sea>` is a registered SEA name or a SEA file path; `-m`,
`--work-dir` and `-b` are the fallbacks for a SEA whose `model`,
`work_dir` or `max_budget` setting names none (the `run_sea()`
arguments of those names are the same fallbacks, as its docstring
says: there is no calling task here to pass anything explicitly);
`--work-dir`, the current directory by default, is also what a
relative `work_dir` setting is resolved against
(`sea_settings.anchored_work_dir`), as the calling task's directory
is under `run_agent`;
`--json` prints the whole `SeaRunResult`
as JSON (`trajectory_paths` as strings) where the default prints the
`finish` YAML (or the error), and the exit status is 0 when the
result's `success` is true, 1 otherwise; a SEA that fails to load
(`SeaError`) exits with the message.  The `run_agent` /
`run_parallel` tools need the daemon, so a SEA meant for headless runs
leaves the `agents` tool group out of its `tool_profile`.
`examples/rl_episode/rl_episode_sea.py` is a worked example: an
RL-style environment SEA (a `@tool_schema` `guess` action with
integer bounds; `tool_call_hook` ends the run with `end_run(<outcome
JSON>)` on the winning guess and refuses `finish` while the episode
is open, `llm_response_hook` ends it with `success=False` at
`MAX_TURNS`, 10, replies; `settings()` fixes `tool_profile: none`,
`max_budget: 0.25` and `model_config: {"plain_io": True}`, takes the
model from `RL_POLICY_MODEL` and points `base_url` at `RL_POLICY_URL`
when that is set, so a trainer's OpenAI-compatible policy server
serves the episode; `episode_outcome(result)` reads the outcome dict
(`solved`, `reward`, `attempts`, `secret`, `guesses`, `turns`,
`output_tokens`) back from the run's `result` and raises `ValueError`
when there is none; `output_tokens` sums the replies
`llm_response_hook` had seen when the outcome was built, which on a
provider that runs the winning `guess` natively inside the turn
(Codex) excludes that turn, so `SeaRunResult.total_tokens_used` is
the authoritative total) that a trainer runs through `run_sea()` or that
command line; `src/kiss/tests/agents/seas/test_rl_episode_example.py` runs it
both ways against the scripted local model.
`examples/rl_episode/wire_check.py [episode] [-m MODEL] [--upstream
ORIGIN] [--sea FILE] [--record FILE]` (the episode id defaults to
`episode-3`, `--sea` to the example SEA next to it) checks the prompt
on the wire: a recording proxy in front of a live model (`https://api.openai.com`,
signed with `OPENAI_API_KEY`, or `--upstream` for a policy server)
runs one episode as a trainer would (`scripts/run_sea.py --json` as a
subprocess with `RL_POLICY_URL` at the proxy and `RL_POLICY_MODEL`
from `-m`), records every request, prints the lengths and SHA-256
digests and the first differing byte offset on a mismatch, and exits
0 only when the episode succeeded and every request carried exactly
one system message equal to the SEA's `SYSTEM_PROMPT` byte for byte
(`--record FILE` keeps the requests as JSON lines).

### Running a SEA: `run_agent`, `run_parallel` and `/xxx` commands

A running agent dispatches a SEA with the `run_agent` tool
(`agent_dispatch.make_run_agent_tool`), or several at once with
`run_parallel` (`agent_dispatch.make_run_parallel_tool`), which is
`run_agent` made once per task:

```text
run_agent(task, agent="", model="", tool_profile="", max_budget="", timeout="", options="", wait="", machine="", description="")
run_parallel(tasks, agent="", model="", tool_profile="", max_budget="", timeout="", options="", description="")
agent_job(job_id, action="tail", timeout_seconds="")
```

`task` is the sub-task's task text (the SEA's `prompt(task)`, if
defined, turns it into the prompt); `description` is a title for the
call that the chat shows on the folded tool panel (`run_agent <description>`,
as the Bash panel does; `run_commands_parallel` takes the same argument)
and is ignored at run time; `options` is a JSON object in the
`settings()` vocabulary (`agent_dispatch.OPTION_TYPES`: every settings
key except the ones that describe a script, `locked` and `hidden`
(`sea_settings.META_SETTINGS`), and except the
four the tool takes as arguments, plus `inherit`, `workspace`, `add_to_prompt`
and `add_to_system_prompt`, the caller's text appended to the prompt
and the system prompt); `model`, `tool_profile`, `max_budget` and
`timeout` (number strings; a JSON number is read like its text,
`agent_dispatch._parse_number`, as is `agent_job`'s `timeout_seconds`,
`parse_wait_seconds`) are arguments only, and `options` refuses
those keys with a message naming the argument.  Every value given is
explicit: it wins over the SEA's settings, or is refused when the SEA
locks the key.  Each option replaces the inherited value (and the
SEA's, unless locked) of one key: `work_dir` (relative to
the calling task's directory), `chat_id`, `workspace` (the account of
a multi-account channel; refused for any SEA that is not a channel),
`model_config`, `inherit`, `auto_classify`, `use_web_tools`,
`use_memory`, `compaction` (`false`: the sub-task never compacts its
context; unset, it takes the caller's), `docker_image`,
`add_to_prompt` and `add_to_system_prompt`, e.g. `'{"use_web_tools":
false}'`.  `wait="false"`
returns with a job id as soon as the sub-task's tab exists (its
initial `status running=true`, so the spawn lands inside the call's
time window on every surface), the dispatch has already failed (then
the error is the answer), or a 30 s start grace
(`agent_dispatch._JOB_START_GRACE_SECONDS`) has passed; the dispatch
runs in a daemon thread
of the calling process (`agent_dispatch.launch_agent_job`), its spend
is bound to the calling run's usage epoch (a finished child's cost
is folded into the caller's task, and the `Budget:` figure of the
usage line the caller's model reads reports the task's cumulative
spend, earlier sessions and sub-agents included, against the task's
cap; `RelentlessAgent._budget_status`, installed as
`KISSAgent.budget_status_hook`, recomputes it after every tool call,
so a SEA's prompt must not add a child's reported cost again; the
spend of the helper calls made on the run's behalf — the coding
gate's reviewer, the vision model that describes an attachment for a
model that cannot see — is attributed the same way,
`sorcar_agent._attribute_sub_usage` keyed per attempt and
`Model.helper_usage` keyed per call, so a retry or a stop never
bills one twice and a session hand-off carries the record along; a
child run on another machine, `machine=<name>`, keeps its row there,
and what that row spends after the dispatch returned — an `/ask`
answer, a Task update finishing after it — is fetched from the machine
(`getTaskUsage`, `task_update.sync_remote_subtask_usage`, never
periodic: run only when a first-page CHATS refresh, a Spend refresh or
another daemon's usage request starts one, `start_remote_usage_sync`,
at most one at a time, over the sub-agents recorded in the last
`REMOTE_USAGE_WINDOW_S`, 7 days, widened to every recorded one when
the machine's last full sweep is `REMOTE_USAGE_FULL_SWEEP_S`, an hour,
or more ago)
and the growth over the caller's record of it,
`persistence._charge_remote_subtask`, charged to the caller and its
finished ancestors once: the share a still-running row owes is kept
in `remote_charges` and banked on its live agent under the key
`remote:<id>` (`task_update._bank_remote_charge`), one that arrived
while no live agent held the row is banked by the row's final save
(`ChatSorcarAgent._bank_pending_remote_charges`), and the final save
or orphan recovery settles what the row's totals do not hold yet into
the row exactly once, `persistence._settle_remote_charges`, a repeated
snapshot of the ledger re-settling the retained charges rather than
overwriting them), the budget
it was given stays reserved against the caller's remainder until it
finishes (`AgentJob.budget`, see [Precedence](#precedence)), and only
the calling agent's `agent_job` tool sees the job: `agent_job(job_id, "wait")`
blocks for its result, `"tail"` reports whether it still runs,
`"kill"` sets the job's cancel event, which `daemon_client.run`'s read
loop turns into a stop of the sub-task confirmed by its terminal
status (an unconfirmed stop is reported as such, never as "stopped"),
and `agent_job(ids, "wait_any")` (`agent_dispatch.wait_any_agent_job`;
`ids` is several job ids separated by commas or spaces, or empty for
every job of the task) blocks until any of the named jobs whose result
the tool has not returned yet finishes, or `timeout_seconds` pass, and
returns the results of all that have — a lead line `Finished: <ids>;
running: <ids or none>.` then one section per job opened by `Job <id>
(<name> agent task) finished:` — so a follow-up (a reviewer, a re-fix)
starts as soon as its job returns instead of after the slowest of a
wave; the unread-results note of `Bash(until=)` and `bash_job` is the
same call.  A result any action has returned is marked `AgentJob.reported`
and never returned by `wait_any` again (a `wait_any` on jobs all
reported says so; one that times out says `None of <ids> finished
within <N>s; they are still running. agent_job('', 'wait_any') waits
again.`; an unknown id, or a task with no
job, is an `Error:`), and the mark is set only once the text to return
exists, so an interrupt before that leaves the result for the next
call.  The poll runs in `_JOB_WAKE_SECONDS` slices like a plain wait, so
a Stop lands; the job notice a timed-out `run_agent` returns names the
action.
When the `run_agent` or `run_parallel` call itself is interrupted,
by its Stop button or because the calling task stops, every child it
started is cancelled too (`agent_dispatch._abandon_job`); the call
does not wait for them, so the Stop stays prompt, and the run's end
(`kill_jobs_of`) collects what is left.
`agent` names WHICH agent to run; every spelling resolves to one
SEA path and takes the same dispatch
(`agent_dispatch.resolve_agent` tries, in order):

1. empty, or a generic label a model may invent (`general`,
   `assistant`, `default`, ...; not `worker`, which is a base class, and
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
   the script once, reads its class (a `ChannelSea` or not) and folds
   `settings()` over its class chain to read the effective settings
   (`sea_commands.sea_settings`: the `timeout`, `work_dir` and `locked`
   the dispatch needs); the run itself, with the prompt, system-prompt
   and tool methods, happens on the daemon.

Anything else returns `Error: unknown agent '...' — not a registered
slash command and not a path to a .py SEA file.` with the closest
command and the registered commands.  The sub-task is reported under
the resolved name: the command name, else the script's file stem
without `_sea`.

A `/xxx text` prompt typed into a chat tab is NOT a dispatch: the
daemon (`task_runner._apply_sea`, via `sea_commands.slash_command`)
runs the SEA `xxx` directly in the tab's own run, with `text` as the
prompt and the SEA's path as the run's `seaPath`, so the SEA's
settings, system prompt and tools apply to that very run; there is no
relay LLM turn and no nested sub-agent tab, and the raw `/xxx text`
line stays what the task panel and the chat history show
(`displayPrompt`).  A bare `/xxx` runs the SEA on the empty task
when its `prompt()` turns that into text (a default task, as `/sync`
has); when the SEA passes the empty task through (the stock
`BaseSea`), the bare command stays an ordinary prompt without the
SEA, on the tab's prior SEA or picker SEA if any.  The SEA's file is
executed once either way: `slash_command` only splits the prompt,
and the daemon reads whether a default task exists off the layers it
has already loaded (`load_layers(..., loaded=layers)` reuses them for
the fallback).  The run's start status (`status running=true`, with
`startTs` and the client's run token) is broadcast BEFORE the
script's overrides run: they are untrusted code of unbounded
duration and the chat is already busy, so every surface shows the
tab running from there on, and a script that fails to load fails the
task through the ordinary handler, result then end status.
`/xxx help` answers with `description()` without
running anything (and the `<task>` follow-ups typed while
`description()` ran are run next, see `_end_run_before_loop`
below); `/xxx check` loads the class and answers with the
classes of its inheritance chain, effective settings, model,
the tools `tools()` adds, the methods it defines and the prompt for a
sample task, or the first error (`sea_commands.sea_check`).  A
model-picker SEA (`bestrouter`) selected on the
tab is applied before `/sh` under `/sh ...`: `/sh`'s own `seaPath`
stays, on the picker's model (`/sh` replaces the system prompt, so
the picker's routing protocol does not reach it; a SEA that appends
keeps it).

`run_agent(..., machine="box")` runs the sub-task on a registered
remote machine (`/machine add`; `machine` is its name or its host/IP
address, `machines.find_machine`; this machine's own name, `localhost`
or one of its addresses, `machines.is_local_machine`, is an ordinary
local call), with that machine's tools, files and models.  The call is the same dispatch: the SEA is resolved and
its settings checked here, and the `run` command goes to the local
daemon with a `machine` field, which `server.remote_tabs` forwards to
the machine's daemon over the registered link and whose frames it
relays back, as it does for a chat tab bound to a machine (a `submit`
or `appendUserMessage` typed into a bound tab, `remote_tabs.TYPED_COMMANDS`,
is also written to this daemon's `steer_inputs` table,
`persistence._record_steer_input`, before it is handed to the machine:
the task row and the steer echo are the machine's, and the composer
history that `inputHistory` builds (ArrowUp/ArrowDown, ghost
completions, the `/` popup's order) reads only this daemon's database
(leaving out texts longer than `persistence.INPUT_HISTORY_TEXT_CHARS`,
4,000 characters, which stay persisted, a task prompt in its Chats
row, a steering message in `steer_inputs` alone: a pasted document
is not something the arrow keys recall, and a hundred of them made the
list a fresh page downloads two megabytes),
so a prompt or a `/ask` line typed there would otherwise be forgotten;
a `run` and a `userAnswer` are not recorded, and a refused local write
is logged as a warning rather than failing the command).  What
differs is what travels: the SEA runs as the machine's own copy (a
registered command by its name, the run's `seaCommand`, which the
machine's daemon resolves to its own copy in `task_runner._apply_sea` and
refuses when it has none; the default agent as the machine's plain
run; a `.py` path at the same path there), the
`work_dir` option is a path on the machine as given (empty: its
configured folder; the SEA's own `work_dir` setting, a path here, is
not sent), the task's paths are not rewritten, and the caller's chat,
container and extra tools, which name nothing there, are not
inherited (its model, budget share, prompt suffixes and model
configuration are).  The forwarded `parentTaskId`/`parentTabId` make
it a sub-agent run on the remote daemon too: no top-level tab on
either daemon, the `new_tab` it broadcasts is relayed (once the run's
own frames have told its task id, since the remote announces every
sub-agent run to every client) under the caller's tab, so every
surface here opens the nested sub-agent tab,
streams its transcript (a forwarded `resumeSession`) and closes it on
the relayed `subagentDone` (a link lost meanwhile, or the caller's
tab leaving the stream on a `newChat`, a re-bind or a resume into it,
ends those tabs with a `subagentDone` of this daemon's, so none spins
for good, and a lost link refuses the bound tab's own run with a
`status running:false` stamped `lost: <machine>` before the
`prompt_refused` error, since the task may still be running there,
which `daemon_client.run` raises as `ConnectionLostError` rather than
taking for the task's end; one a client bound to a stream of its own
meanwhile is left to it); a surface
connecting while it runs gets
the same `new_tab` from `RemoteTabs.replay_to`, and one connecting
after the sub-agent finished gets its `subagentDone` again
(`RemoteTab.closed_tabs`: kept per sub-agent tab from the relayed end,
through the `closeTab` a page online answers it with, until the tab
shows something else — a resume into it, a new chat, a bind — or the
stream retires), since a page that was away when the child finished
still has its tab and would otherwise show it spinning for good.  The result's `ran:`
line ends with `machine=<name>`; an unregistered name or host is
refused with the registered names before anything runs, as is a run
whose tab is not already bound to that machine when the thread that
warms the machine's view stream cannot start (`<machine>: the stream could not start: out
of threads`, `RemoteTabs._bind`).

The Terminal and Browser buttons of a chat bound to a machine open
the shell, or the page, on that machine (`server.remote_views`): the
webapp stamps `terminalOpen`/`browserOpen` with `machine`, and
`RemoteViews` keeps one authenticated stream per machine in use
(opened on the first such command or when a chat tab is bound to the
machine, `RemoteViews.bind` from `RemoteTabs._bind`, which starts the
stream's handshake in a background thread, `RemoteViews.warm`, as
the bind begins, alongside the tab's own stream rather than after it,
so a chat opened from the machine waits for one round of TLS and
login handshakes, not two; `_bind` joins that thread and `release`s
the stream, which closes it only when the bind was refused or stood
down and left it idle and unbound, before `open_for`; closed once
it carries no terminal or page and no tab is bound there), forwarding
the terminal and browser commands without
`connId`/`machine` and relaying the machine's `terminal*` and browser
events back stamped `machine`.  A terminal's events go to the
connection that opened it (a dropped connection's shell is hung up
after the terminal grace period unless a reload claims it first —
`SIGHUP` to the shell's process group, and one still alive 2 s later
is killed together with its foreground job, which runs in a process
group of its own that an ignored `SIGHUP` never reached and would
otherwise keep running, holding the pty, reachable from no tab; a
reopen that arrives once the hang-up has started waits for that shell
to be reaped and starts a fresh one, and the old shell's exit is not
reported under the new page); a page's
tab id is namespaced `browser__N@machine`, so a tab-addressed browser
command finds its machine from the id, its frames go to the local
connections whose last `browserViewport` for the tab was visible
(each frame acknowledged to the machine as it arrives,
`browserFrameAck`, and paced per viewer by
`server.frame_flow.FrameFlow`, which the local `BrowserTabService`
uses for its own viewers too: at most `MAX_IN_FLIGHT` (2) frames
await a viewer's acknowledgement, the newest frame produced meanwhile
is held back and goes out on the acknowledgement, a resize keeps the
viewer's frames in flight on its account, and a viewer that never
acknowledges, an older surface, has a frame counted as delivered after
`ACK_GRACE` (5 s), so a slow link carries the latest picture at its
own pace instead of a backlog that delayed the link's keepalive and
lost the stream), and
its `openBrowserTab` goes to every client (the focused reply to a
`browserOpen` to its requester alone; a focused one nobody asked for
is the machine's agent calling for the user, `show_browser`, and
every surface here switches to it as the machine's own would).  A
stream that opens sends `browserList`, answered with the machine's
`browserTabs` snapshot (or an `error` by an older daemon): the pages
open there before the stream are shown here unfocused.  `replay_to`
tells a connecting client the known remote pages in one
`browserTabs {remote: true, tabs}` snapshot, each tab stamped
`machine`; a lost stream ends its terminals with a `terminalError`
naming the machine and closes its pages everywhere, and is opened
again after `REOPEN_DELAY` while a tab is still bound to the machine.
`RemoteViews.bind` only records the binding (under `RemoteTabs`'
registry lock, in the registry's order; `_announce` drops a binding
the registry has moved past, and a `ready` is told every binding
under the same lock); `RemoteTabs`
then opens the stream (`open_for`, before the run is forwarded, so
the page its agent opens finds the stream subscribed) or releases the
previous one (`release`); `close_all` fences every later registration.

`run_parallel(tasks, ..., agent=...)` is N `run_agent` calls
(`agent_dispatch.run_agents_parallel`; `tasks` is a JSON array of
non-empty strings, `fanout_guard.parse_tasks_json`, which
`run_commands_parallel` shares for `commands`; the decoder is lenient
(`_decode_json_leniently`): raw control characters and newlines inside
a string are accepted (`strict=False`), and when decoding still fails
it retries after doubling every lone backslash that begins no JSON
escape (grep's `\|`, `\(`); the error of one that still does not
decode says where decoding stopped, as a rule a double quote
inside a string not written `\"`, that a shell substitution such as
`$(cat file)` is never expanded, or that text that is no array must
be one, and an array that is empty, nests past the decoder's depth or
holds a non-string or blank member is refused too): one `plan_run_agent` per
task with the same `agent`, arguments and `options`, so every child is
a daemon sub-task in its own tab, exactly what `run_agent` would
start, under the same precedence rule and the same inheritance table
(the caller's remaining budget is shared among the N children plus the
caller); its `prompt(task)` shapes each child's prompt, its
system-prompt texts, tools and hooks apply on the daemon.  Every
`options` key `run_agent` takes is accepted here too — `auto_classify`,
`chat_id`, `workspace` and a channel SEA included — and
`inherit: false` makes the children take nothing
from the caller, their budget included (each then gets the explicit
`max_budget`, else the SEA's, else the daemon default, exactly like a
`run_agent` child).  The call blocks for the one
`timeout` (argument > the SEA's setting > 3600) and returns a YAML
list with one entry per task: the child's result text or, for a child
still running at that point, its `agent_job` notice; no child is
stopped by the timeout.  A list longer than the daemon's
`tool_output_max_chars` is written to `tmp/tool-out/<tag>-agents.txt`
under the caller's work dir and the call returns its digest instead
(`agent_dispatch.digest_agent_results`: one `[i/N] ok|FAILED: <first
200 characters of the summary>` line per child, `running (job <id>)`
for one not finished, `FAILED: <the error>` for one whose dispatch
could not run and so has neither a result nor a job).  The `summary`
of every sub-task result
(`run_agent`, `agent_job` and each `run_parallel` entry alike,
`agent_dispatch.format_dispatch_result`) of a child that ran several
sessions (a hand-off, see [Refusals and session
hand-off](#refusals-and-session-hand-off)) is the final session's
summary alone, opened with `(the last of N sessions; the earlier
sessions' handoff notes are in the task's stored result)`
(`agent_dispatch.result_summary`): the `<h3>Previous Session N</h3>`
notes stay in the stored result for the user.

### `system_prompt()`, `register_as_model()` and `on_picked_as_model()` — model routing SEAs

| Method                         | Return type | Effect                                                     |
|--------------------------------|-------------|------------------------------------------------------------|
| `system_prompt(self, system_prompt)` | `str` | the run's system prompt; `system_prompt + "\n\n" + PROTOCOL` **adds** the protocol after what is already there |
| `register_as_model(self)`      | `bool`      | `True` lists the SEA in the model picker under its command name |
| `on_picked_as_model(self, work_dir)` | `str` | hook run once per run whose model is the SEA; the returned text is logged |

A model-routing SEA's `system_prompt()` carries its *model routing
protocol*: it appends the protocol to the system prompt it receives
(the base prompt plus the caller's `appendToSystemPrompt` text, or the
channel preamble), separated by a blank line.  Because it returns the
received text plus a suffix, the caller's own additions survive.  The
result is this run's prompt only: a sub-agent gets the protocol when
its own layers include the picker (a `/xxx` command typed into the
picker's tab, or a `run_agent` call that names the picker as its
`model`), not by inheritance: a child that inherits the tab's model
gets the real model the picker resolved to.
`register_as_model()` is a registry flag,
not a run parameter: `kiss.agents.sorcar.sea_commands.model_seas()`
lists every registered SEA whose `register_as_model()` returns `True`
(the verdict is cached per script, stamped with the mtime, size and
inode of the script and of every SEA file its class derives from —
a base named by command, `sea_class("name")`, as it was read before
the load ran and with the script the name resolved to, an imported
base as found once the load has returned — so
an edit of a custom base alone, `class X(sea_class(...))`, one of a
command-named base made while the load was in flight, or a
re-registration of the command is re-read; a base the script imports
the ordinary way, `from parent_sea import Parent`, which Python would
execute once per process, is re-executed from its current
source once it changed, `sea_settings.refresh_imported_module` — not
`importlib.reload`, which would run the bytecode cached for an
earlier revision of the same size and second; each run in a fresh
`types.ModuleType` that is named in `sys.modules` while its source
runs, under the module's import lock with its spec marked
initializing, so an `import` on another thread waits for the run
instead of reading a half-run module (`sea_settings._run_imported`);
a run that raises puts the old module back, so it publishes nothing
and the globals of a class another task is running are never
half-rewritten under it; for a submodule, the attribute of its package
that `from package import parent` reads is rebound only once every
module ran through; a module another thread is still importing is
not waited for — it is left as that import leaves it under a revision
no file has (`_MID_IMPORT`), and the next call tries again;
never a module of the
kiss package itself — before the script runs, then every base of the
loaded class root first (`_refresh_bases`), and the script is loaded
again when its class still derives from a superseded one
(`_superseded`), so the stamp describes the class the script derives
from;
only a script whose source mentions `register_as_model` or derives
from something other than the plain bases is imported for it,
`_registers_as_model`),
the daemon offers them in the model picker (vendor `Router`, once at
least one catalog model is runnable), and a task run with such a pick
becomes a run of the SEA on the model its
`settings()["model"]` names (else the default model) — `/xxx` slash
commands and runs that already carry an `seaPath` keep their agent,
with the picker's methods folded in first (see
[Extending a SEA](#extending-a-sea-python-inheritance)).  The bundled
`autorouter`, `bestrouter`, their OpenRouter-only twins `orautorouter` and
`orbestrouter`, `ccxbestrouter` (the bestrouter on `cc/` and `codex/`
models) and `brbestrouter` (the bestrouter on Amazon Bedrock models) are
such SEAs; `bestrouter` is three methods besides `description()`:

```python
class BestrouterSea(BaseSea):
    def register_as_model(self) -> bool:
        return True

    def system_prompt(self, system_prompt: str) -> str:
        return system_prompt + "\n\n" + SYSTEM_PROMPT  # the routing protocol

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"model": SIMPLE_MODEL}
```

`on_picked_as_model(work_dir)` is the picked SEA's chance to act on
being chosen.  `kiss.agents.sorcar.sea_commands.run_picked_hook` runs it
on a daemon thread at two moments: when the user picks the SEA in the
model picker (`selectModel`, with the tab's registered work directory,
else the daemon's global one, without waiting; a pick stamped `machine`
never reaches this handler, it is forwarded to that machine's daemon,
`sidebar_panels.machine_models_reply`), and once per run whose model is the SEA (after the
SEA overrides, with the run's effective work directory,
waiting at most `PICKED_HOOK_TIMEOUT_SECONDS`, 15 s, before the task
starts).  It is not a run parameter: the return value is only logged,
a hook that raises is logged as a warning, and one that blocks is
abandoned on its thread; none of them fails the run.  It does not run
for a `/xxx` command's agent or an explicit `seaPath`, only for the
SEA the model pick names.  `autorouter` uses it to make sure an
enabled weekly cron job that runs `/rsi7d autorouter` exists
(`autorouter_sea.schedule_weekly_rsi7d`: an enabled job of that name is
kept as is, a paused one is resumed, and when there is none one is
created — a prompt job in the KISS checkout the work directory
is in, or a scratch-directory job that
still refreshes `$KISS_HOME/AUTOROUTER.md` when there is no checkout;
a linked task worktree resolves to its owning checkout).  A pick whose
hook is still running from an earlier pick of the same model, directory
and script revision shares that run, so a hook that never returns holds
one thread, not one per pick.  The revision is a digest of the text the
SEA ran from, taken from the bytes its load read once and executed (a
run picks the SEA it loaded; a pick in the model picker is keyed by the
text the registered script holds then, and runs under the revision its
thread loads), so the hook of a script fixed since runs beside the stuck
hook of its old text, however long that load waited for another load of
the script, and a script saved again unchanged is the same hook.  The
text is the picked script's own: a fix in a file its class derives from
is no new revision of it.  The hook's thread is entered as that run
before it is started, and runs the hook only while the entry is its own,
so a Stop that lands while a run starts it leaves no hook that later
picks cannot find.  Hooks of picks from different directories may
overlap — a SEA file is executed afresh on every call and cannot hold a
lock of its own — so the look-up-then-create is
the cron store's atomic `cron_job("ensure")`, not the SEA's.

The `run()` parameters without a `settings()` key (the allowlist is
`sea_settings.SETTING_TYPES`; naming one of these stops the task with
`settings() has an unknown key ...`, or `settings() key ... was
removed: ...` for `add_to_system_prompt`, `add_to_prompt` and
`is_parallel`, which `sea_settings.REMOVED_SETTINGS` names):

- **`system_prompt`** — a replacement base system prompt the caller
  supplies (what `run_agent` forwards from the calling run's own
  replacement prompt); the SEA's `system_prompt()` method receives it,
  with the caller's `add_to_system_prompt` already appended, and
  returns the run's.  The method is the SEA's way to set the prompt;
  `settings()` has no key for it.
- **`timeout`** — bounds the *client's* local wait (`None` waits
  indefinitely; the default is 3600 s).  The daemon does not enforce
  it.  The wire field `timeout` carries `record_timeout` when one is
  given, else this value when `stop_on_timeout` is `True`, else
  nothing; a sent value is recorded in the task's `task_settings`, and
  it is checked against a SEA's locked `timeout` only when the call's
  `provenance` marks `timeout` explicit, which `run_agent` does for
  an explicit `timeout` argument (it sends that as `record_timeout`
  with `timeout=None`).  (The `timeout` KEY of `settings()` is a
  different thing: the dispatcher's wait for a `run_agent` sub-task,
  see above.)
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
  every surface, and not by a wait that ends without one; the
  `run_agent` job thread sets it itself once it has recorded the
  dispatch's outcome.  A background `run_agent` call waits for that
  signal for up to the 30 s start grace, then answers with the
  finished result when there is one and with a job notice otherwise
  (also when the grace expired before the tab existed).
- **`provenance`** — `{setting key: "explicit" | "inherited"}`, where
  each of the command's values came from (`kiss.agents.sorcar.run_config`);
  the dispatcher fills it from what the call passed and what it took
  from the calling agent, the daemon records it in the task's
  `task_settings` event, and the result's `ran:` line reads its
  `inherited=` list from there.
- **`endpoint_file`** — selects which daemon to connect to (default:
  `$KISS_SORCAR_LOCAL`, else `$KISS_HOME/sorcar-local.json`, the file the
  daemon writes with its plain-loopback `ws://` URL and per-start local
  token); the script
  already runs on that daemon.
- **`parent_task_id` / `parent_tab_id` / `parent_reviewer`** — the
  CALLING task's identity (how `run_agent` nests a dispatched run under
  its caller) and whether that caller sits in a reviewer sub-tree (a
  marker the child and its own sub-agents inherit; `run_agent` also
  sets it when the submitted task itself reads as a review,
  `fanout_guard.is_review_task`, and the daemon re-checks the
  effective prompt after the SEA's `prompt()` has shaped it, so a
  `prompt()` override that turns a dispatch into a review marks it
  too; with tool profiles enabled, a marked run whose task is not an
  implementation task and that names no explicit profile gets the
  `review` profile, the one without edit and dispatch tools (its
  `Bash` is unrestricted: not a sandbox, see
  [Setting semantics](#setting-semantics)); the task is judged once
  per run, with its `{arguments}` substituted, and that one verdict
  picks the restricted-profile note, the base prompt and the toolset,
  `SorcarAgent._run_profile`), which a dispatched script must not be
  able to forge.
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
- **`sea_path`** — the script cannot override its own path.
- **`scope_work_dir`** — the calling task's directory, sent as
  `tabScopeWorkDir`: `run_agent` fills it from the caller (or with
  `<home>/agent_work` when the calling agent has no directory, as one
  built through the Python API: the base `plan_run_agent` itself
  anchored the relative `work_dir` at and compared the lock against,
  so a call the dispatcher admits is one the daemon admits and the run
  lands in `agent_work/x`, not `agent_work/x/x`), and it is
  the directory a relative `work_dir` (the SEA's setting or the call's
  option) is anchored at and compared against a locked `work_dir`
  (`sea_apply.calling_work_dir`; the command's own `workDir` when it
  is absent, as for a `/<name>` run).  It is also recorded on the
  run's tab in the daemon's shared tab registry.
- **`inherit_tools`** — whether the run also gets the extra tools of
  the task `parent_task_id` names (the callables that parent's SEA
  kept or added beyond the built-ins in its `tools()`, and those the
  parent inherited itself), appended to the built-in toolset under
  names it does not have yet before the run's own `tools()` sees the
  list; a run on the `none` profile keeps exactly what its `tools()`
  returns.  `run_agent` sets it for every inheriting dispatch (not a
  channel, no `inherit: false` option), so a sub-task that inherits the caller's system
  prompt also has the tools that prompt refers to; a script decides
  nothing here.
- **`isolated_repos`** — `{repository root: worktree directory}` for
  every git worktree the calling agent holds (its `isolated_repos()`:
  its own and the ones it inherited, so a sub-agent of a sub-agent
  still sees the top task's worktrees).  The run never creates a
  worktree for one of these repositories: `create_worktree` reports
  that the caller already isolates it, and the run edits the caller's
  worktree, which the caller merges.  `run_agent` and `run_parallel`
  fill it in; a caller outside the daemon (the CLI) passes it on the
  wire because the daemon has no registered state to read it from.
- **`workspace`** — the multi-account channel workspace a
  `ChannelSea` run holds for its lifetime (see
  [Channels](#channels-class-xseachannelsea)); the `run_agent` option
  `workspace` or a channel launcher supplies it, every other run
  ignores it.
- **`add_to_system_prompt` / `add_to_prompt`** — the caller's
  additions; the SEA's `system_prompt()` receives the system prompt
  with the caller's suffix already in it, and the caller's
  `add_to_prompt` is appended after what `prompt(task)` returned.
- **`is_parallel`** — whether the run's built-in toolset includes
  `run_parallel` and `number_of_cores` (the user's parallel-mode
  toggle, wire field `isParallel`, default `True`); a client/UI
  choice with no SEA key (`is_parallel` is a removed settings key):
  a SEA that wants no fan-out picks a `tool_profile` without the
  `agents` group.
- **`machine`** — the registered remote machine the run executes on
  (`/machine add`; its name or host/IP, `machines.find_machine`): the
  local daemon forwards the `run` command, stamped `machine`, to that
  machine's daemon over the registered link and relays its frames
  back (`server.remote_tabs`), so the client's wait is unchanged.
  `work_dir` is then a path on that machine.  Empty (the default)
  runs on the local daemon.  `run_agent(..., machine=...)` fills it;
  see [Running a SEA](#running-a-sea-run_agent-run_parallel-and-xxx-commands).
- **`sea_command`** — the name of a registered command, sent as the
  wire field `seaCommand` in place of `sea_path`: the executing daemon
  resolves it to its own copy of that command's SEA
  (`task_runner._apply_sea`, which refuses a name it has no
  command for).  How a run on a `machine` names a SEA the machine has
  under its own paths; `run_agent` sets it when `agent` is a command
  name and `machine` is given.  Empty (the default) sends `sea_path`.
- **`cron_job`** — whether the cron scheduler launched the run (an
  in-chat job; a prompt job is recognised by its unattended preamble
  in the submitted text, and a cron run's sub-tasks say so on the
  wire), sent as `cronJob` and persisted as the history row's
  `is_cron` flag, which the CHATS list hides unless its Cron chip is
  on.  The unattended verdict is decided on the text as submitted
  (plus `appendToPrompt`), before the SEA a `/xxx` prompt turns into
  can replace the text (the scheduled `/recommendations now` does),
  and stamped on the
  agent as `is_unattended` (read by `cron_agent.is_unattended`, which
  makes `ask_user_question` refuse); `is_cron_job` is stamped from the
  wire flag.  No SEA setting.
- **`attachments`** — the files sent with the prompt (an image or
  PDF the user attached), each `{"mimeType", "data"}` with base64
  `data`, the webapp's `submit` wire shape; sent as the wire field
  `attachments` only when non-empty.  The `mimeType` is advisory:
  the bytes' own header decides the type the model is told when it
  is one of `image/png`, `image/jpeg`, `image/gif` or `image/webp`
  (`Attachment.from_bytes`, `model.image_mime_type`), since a vision
  API rejects the whole request when the declared type and the bytes
  disagree (a `.png` file holding WebP bytes), and an iPhone HEIC
  photo is transcoded to JPEG when the host has a HEIF decoder; the
  declared type is used only for bytes the header does not identify.  The `/ask` side channel passes
  the files attached to the question on to its answering run this
  way.  A message with files sent to a running task is queued for
  that task instead of starting a run: a steering message's files
  reach the live agent with its text, a `<task>` follow-up's become
  the follow-up run's attachments (one entry per block), and leftover
  prompts re-submitted after the run keep theirs
  (`task_runner._redispatch_leftover_prompts`: consecutive plain
  messages become one prompt joined by blank lines with their files
  pooled, a `<task>` message is a re-run of its own with its own
  files, the first re-run is the tab's next run and the rest are
  submitted to it one by one as if typed while it runs; a run that returned before its
  subtask loop — a `/xxx help` answer, a run refused for want of a
  model or during a merge, a setup failure — closes its follow-up
  queue and acknowledges any Stop at once (`_end_run_before_loop`)
  and hands the `<task>` follow-ups it never drained to that
  re-dispatch, one re-tagged re-run each, instead of dropping them;
  a loop that a failure or a Stop cut short drops them).

### Setting semantics

- **`None` and `""`** — a key whose value is `None` is dropped like an
  absent key (`sea_settings.resolve_settings`); so is a string key
  (`work_dir`, `model`, `chat_id`, `tool_profile`,
  `docker_image`) whose value is `""`, exactly as a blank `run_agent`
  option is.  Either means "no override": the caller's or the
  persisted value stands — also when the key is one a base class lays
  (a `WorkerSea` subclass returning `settings | {"use_memory": None}`
  drops the worker default; leave the key out to keep it).
- **`model`** — leave the key out (or write `""` / `None`) to keep the
  caller's model (the tab's pick, or the calling task's model for a
  `run_agent` dispatch); a run whose command names no model at all gets
  the tab's selected model, else the daemon's configured default
  (`task_runner._tab_model`).  A non-empty string must name a model in
  the daemon's available model list or the task fails, unless the
  effective `model_config` has a `base_url` and the name is one the
  bundled catalog or `$KISS_HOME/MY_MODELS.json` knows (so it can be
  priced): such an endpoint run skips the availability check.
- **`chat_id`** — a non-empty string resumes that chat session.  A
  fresh chat comes from not inheriting one (a channel run, a
  `run_agent` call with `inherit: false`), not from `chat_id: ""`,
  which is dropped like every empty string.
- **`system_prompt(system_prompt)`** — the method that receives the
  assembled system prompt and returns the run's (`/sh`, `/merge`,
  `/ask` replace it, `bestrouter` and `/write` append to it; it is not
  a settings key).  Without it the prompt is the daemon's: a run
  whose built-in tool profile lacks the `edit` group —
  `tool_profile: "none"` (no built-in toolset), `review`, `shell`,
  `bash`, ... (not a sandbox: `review`'s `Bash` is unrestricted) —
  runs on the reduced `SYSTEM_LITE.md` whatever the classifier's
  verdict; otherwise, with task classification enabled (the default), a
  task classified as simple runs on `SYSTEM_LITE.md` and everything
  else (a failed or disabled classification included) on the full
  `SYSTEM.md`.  Both files are loaded with their `{{IDENTITY}}`,
  `{{HOME_DIR}}` and `{{VERSION}}` placeholders filled in
  (`kiss.core.brand.render_brand`: the brand's identity sentence, which
  for the stock brand also names the repo and website, the brand's state
  directory, and the package version from `_version.py`); the
  string `system_prompt()` returns is used verbatim, with no
  placeholder rendering.  Either way the agent still appends its
  per-run operational instructions after the prompt
  (`RelentlessAgent.perform_task`): the work directory (omitted when
  the tools run in an attached container that does not mount it), the
  process id, the task settings (model, budget, start time in local
  and UTC, parallel mode, the `Interpreter` the work dir's Python
  runs on (`project_interpreter.project_interpreter`: the `.venv` of
  the work dir, for a task worktree that of the main checkout and
  only without one its own, else the
  one on `PATH`, with its version; left out when the tools run in a
  container; every pytest run prints the same line first, from the
  root `conftest.py`, so a fixer, a reviewer and the full run can be
  seen to share one Python), the `Scratch dir` the prompt sends temporary
  files to — `./tmp/<task-id>/` under a chat, `./tmp/` without one,
  and for a sub-agent a private folder `<system temp dir>/kiss-sub-…`
  (`SorcarAgent._private_dir`; when its commands run in a container,
  the container's `/tmp/kiss-sub-…`, `DockerManager.make_private_dir`)
  that is also the `TMPDIR` of its
  commands and holds its job logs and spilled output: when the
  sub-agent's run ends, its background jobs and every process whose
  environment still carries `KISS_SUBAGENT_DIR=<folder>` are killed
  (`processes.kill_env_marked`, which finds a process that left its
  group through `setsid` or `nohup`; in a container its twin
  `docker_manager.remove_private_dir_script`) and the folder is deleted, so a
  sub-agent leaves no process, no temp file and nothing under the
  work tree's `tmp/`; only a sub-agent whose container cannot hold
  the folder (a read-only one) has none of this and scratches in the
  work tree as any other task does —
  chat and task ids, ...), and the user's standing instructions
  from `$KISS_HOME/AGENTS.md` (the file the
  bundled `/remember` and `/forget` SEAs maintain) when that file
  exists.  A `model_config["system_instruction"]` value, if present,
  takes precedence over the composed prompt (`KISSAgent.run` only
  `setdefault`s it).
- **`prompt(task)`** — the method that turns the task text into the
  prompt body (`/merge` wraps the user's text with the standing merge
  instructions; the coding harness ignores the runner's label and
  uses its config's instruction).  It must return a non-empty string.
  Without it the task text is the prompt.
- **`{task_id}` in `prompt(task)`** — every `{task_id}` of the final
  prompt, written by a `prompt()` method or present in the task text
  itself, is replaced by the calling task's
  id (the command's `parentTaskId`; the empty string when the run has
  no parent), which is how `/ask` names the task a question is about:

  ```python
  def prompt(self, task: str) -> str:
      return task + "\n\n" + (
          "The question above is about the task with id {task_id}. "
          "Call task_context with that task id, then answer the question."
      )
  ```

  A run with a SEA executes its prompt as one task: the
  task runner does not split `<task>` blocks into subtasks for an
  `seaPath` run (they are the SEA's to interpret), so the text is
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
  watch and act on it; there is no setting for it.  The browser the
  tab launches is the one `core.default_browser.resolve_browser`
  picks — `$KISS_BROWSER`, else the machine's default browser when it
  is Chromium-based, else the first installed Chromium-family
  browser, else Playwright's bundled Chromium (`ResolvedBrowser`:
  name, executable, whether it is the default, and a `note` saying
  why a fallback was taken, which the tab's badge shows on every
  surface, `openBrowserTab.note`) — and one that is installed but
  cannot start (a Chrome whose update replaced the launcher but lost
  the framework it loads, say) is replaced by the bundled Chromium
  (`BrowserTabService._launch_resolved`), the badge's note naming the
  browser that failed and why in the lines `launch_failure_summary`
  keeps of Playwright's message: the headline without the API name
  in front, the first line of the browser's own stderr (`[err]`, cut
  at 300 characters) and the `<process did exit ...>` line, rather
  than the whole command line and call log; a launch that fails
  outright (the bundled Chromium itself) reaches the client that
  asked for the tab as a `browserError` with the same summary.  The toolset
  (`WebUseTool.get_tools`, fifteen tools, `sorcar_agent.BROWSER_TOOL_NAMES`)
  is `go_to_url` (a URL, `tab:list` or `tab:N`), `click`, `type_text`,
  `press_key`, `scroll`, `screenshot` (its `file_path`, default
  `screenshot.png`, is resolved against the task's work dir, parents
  created), `get_page_content`, `show_browser`, `close_browser`, and
  six developer tools — four DevTools inspection tools over the page's
  CDP session and two routes into the user's own Chrome: `evaluate_script(script)`
  runs a function (invoked, and awaited when it returns a promise) or
  a bare expression through one `Runtime.evaluate` and returns a
  string as is, anything else as JSON cut at 20,000 characters (the
  page gets 10 s, the call 3 s more; `_cdp_call(..., timeout_secs=3.0)`
  is the deadline every CDP round trip gets so a renderer that never
  answers cannot hang the tool); `list_console_messages(clear=False)`
  lists the page's console output and uncaught errors as `[level] text
  (url:line)` lines, recorded per page from the moment the tool
  adopts it (`_observe_page`; a tab that existed before an
  `attach_to_chrome` is observed from the attach on), the
  last 200 kept, texts cut at 500 characters; `list_network_requests(url_filter="",
  clear=False)` lists `METHOD status URL (type)` lines (`...` while
  pending, `FAILED: <reason>`), the last 300 per page, URLs cut at
  300 characters; `clear=True` empties the buffer after listing so the
  next call shows only what the next action caused; `wait_for(text,
  timeout=15, disappear=False)` polls the page's rendered text (runs
  of white space read as one space) until the substring appears or is
  gone — `timeout` is clamped to 0.5–120 s — and returns the
  accessibility tree, or `Timed out after Ns waiting for text ...`.
  The `[N]` ids of that tree are one snapshot of the page
  (`_get_ax_tree`, `_number_interactive_elements`): every read takes
  Playwright's `mode="ai"` ARIA snapshot, which gives each reachable
  element a reference to that very DOM node (`aria-ref`), and replaces
  the ids of the read before it; `click` and `type_text` resolve an id
  to that node wherever the page has moved it since, and an id the
  last tree does not have, or whose node left the page (removed,
  replaced, or of a document the tab has navigated away from), is
  refused with `Element with ID N has changed since the page was read.
  Call get_page_content() to read the page again and use the IDs it
  shows.` — never handed to a neighbouring element.
  Two routes lead into the user's own logged-in Chrome instead of the
  tool's Chromium.  `attach_to_chrome(cdp_url="http://localhost:9222")`
  connects over CDP to a Chrome the user started with
  `--remote-debugging-port` and adopts its newest tab; from then on
  every browser tool drives that browser, `close_browser` detaches
  without closing it (the built-in Chromium takes over on the next
  call), a Chrome that goes away is reported with the way back
  (`Reconnect with attach_to_chrome(...)`, or `close_browser()`)
  rather than silently replaced, and a `show_browser` relaunch carries
  only the current URL out of it — the cookie jar and localStorage are
  the user's profile and stay there (`_capture_session`).
  `use_my_chrome(action, tab_id, url, script, uid, text, timeout=30)`
  drives the user's everyday Chrome — where Chrome ignores
  `--remote-debugging-port` on the default profile since Chrome 136 —
  through the bundled "KISS Sorcar Bridge" MV3 extension
  (`chrome_bridge.py`): `list_tabs`, `open_tab`, `navigate` (what
  is no URL on its own, which Chrome would open as a file of the
  extension, fails with `'example.com' is not a complete URL: write it
  in full, as in https://example.com`; in a Chrome running with no
  window, as on a Mac once the last one is closed, `open_tab` gives the
  tab a window of its own),
  `activate_tab`, `close_tab`, `evaluate` (a script whose page goes
  away under it, by a link, a reload or a closed tab, fails at once
  with `the page went away (it navigated, reloaded or closed) before
  the script finished` instead of at its deadline), `screenshot` (the PNG is
  decoded into `tmp/chrome-shot-<ns>.png` under the work dir — a
  sub-agent's into its private `kiss-sub-…` folder, which goes with
  the sub-agent — and its path returned; the extension takes its
  captures a second apart,
  since Chrome refuses a third `captureVisibleTab` within one second),
  `snapshot` (an interactive-element listing whose
  numbers feed `click` and `type_text`; what a password field holds
  is never listed, nor repeated by `click`; a control its fieldset
  disables is marked `[disabled]` like one disabled itself; a number is stable across
  snapshots and never reused for another element of the page, and the
  element owns it through the document's `kissUids` registry, which
  `cloneNode` and an `innerHTML` round trip do not copy into, so a copy of the element, or a
  replacement that took its place, never answers to it — `click` and
  `type_text` on a number whose element is gone fail with `element [N]
  no longer exists — the page changed; take a fresh snapshot`;
  `click` on a disabled control fails with `element [N] is disabled:
  nothing was clicked`, and on a link into another tab (`_blank` or a
  named target), which Chrome's pop-up blocker opens for a user's click
  only, its reply adds `the link targets _blank, and Chrome blocks a
  new tab for a scripted click: if none opened, open_tab <address>`;
  `type_text` replaces what an input, a text area or an editor holds
  and changes nothing else: a checkbox, a button or a list fails with
  `element [N] is a <input type=checkbox>, not a text field`, and a
  field whose type drops the text, letters in a number field, with
  `element [N] <input type=number> did not take the text and was left
  unchanged`, and a disabled or read-only field with `element [N]
  <input type=text> is read-only and was left unchanged` (the `readonly`
  of a slider or a colour well is no lock)), each waiting `timeout`
  seconds for the extension's answer (`ChromeBridge.call`; the frame
  carries a `timeout_ms` a tenth short of that, at most
  `_REPLY_MARGIN_SECS` (1 s) short, so the extension's own deadline
  normally answers first with an ordinary error, `command exceeded its
  Ns deadline`, and keeps the connection — `COMMAND_DEADLINE_MS`, 30 s,
  applies only to a frame without `timeout_ms`; a command the
  extension has still not answered by the tool's timeout drops the
  connection, shutting the socket down before the close so the close
  returns at once instead of waiting on a stuck sender, and the
  extension reconnects.  A tab with no page loaded in it (one Chrome's
  Memory Saver discarded, one not loaded yet after a session restore,
  one whose page crashed) is loaded again where it is before
  `evaluate`, `snapshot`, `click`, `type_text`, `screenshot` or
  `activate_tab` reads or shows it: a script there would read an empty
  page and a capture show a blank one.  A result of more bytes than the bridge takes
  in one frame (`MAX_MESSAGE_BYTES`, 64 MiB, which would close the
  connection with every other command on it) is answered by the
  extension as an error instead, `the result is larger than the 64 MB a
  reply may be: return less of it`.  The write itself is bounded too: a watchdog
  bound to that connection's socket shuts it down when the deadline
  passes mid-`send` (a peer that stopped draining its socket would
  otherwise block `sendall` for good), the call fails with `the
  extension stopped reading; '<cmd>' could not be sent within the Ns
  left of its timeout`, and the watchdog is cancelled once the write
  returned so it never cuts a later, healthy connection.  The
  bridge's locks are reentrant and taken through
  `kiss.core.locked.locked_call`, so a Stop landing right after an
  acquire cannot leave a lock held by a dead thread.  On the
  extension's side the deadline travels into every stage of an
  `evaluate`/`snapshot`/`click`/`type_text`: a command whose deadline
  passed while it waited in a queue (`command waited out its Ns
  deadline in the queue`), on a busy renderer (`command exceeded its
  deadline before the page ran it`, checked by the injected script
  itself against the page's clock) or before the CSP fallback's
  debugger attach refuses to run, since KISS was already told it
  failed, and the deadline detaches only the debugger attachment the
  timed-out command itself made, never the next command's on the same
  tab).  The bridge is a loopback
  WebSocket server on an ephemeral `127.0.0.1` port with a mutual
  challenge–response handshake (`challenge` nonce → `hello` with
  `sha256(token:nonce)` → `hello_ok` with `sha256(nonce:token)`; the
  token never crosses the wire, a wrong proof gets `hello_error` and
  a 4001 close, a `hello` whose `version` is not the installed
  manifest's (`extension_version()`) gets `hello_error` and a 4004
  close, a second Chrome profile connecting replaces the first
  with a 4002 close — sent from a thread of its own, since the close
  waits up to websockets' 10 s close timeout for the old peer's close
  frame and a wedged old profile must not hold the new connection's
  reads for that long — `ping`/`pong` frames keep the service worker
  alive, and a `chrome.alarms` tick every 30 s, armed from the first
  connection attempt until another profile replaces this one, starts
  the next worker after Chrome ended one).  The extension files
  (`manifest.json`, with the product
  name filled in so a white-label install never shows "KISS Sorcar",
  and the worker it names, `background-<version>.js`) are installed
  into `$KISS_HOME/chrome_extension/`
  on the first bridge start and refreshed when the bundled copy
  changes; the user loads that directory once as an unpacked
  extension (`chrome://extensions` → Developer mode → Load unpacked),
  and the tool's error says so while no extension is connected.
  The installed manifest's `version` names the build (`1.` and three
  numbers of a hash of both bundled files), and every build's worker
  has a file of its own: Chrome reads a worker's size and, about a
  millisecond later, its bytes, and a worker replaced in between by a
  longer one was cut off at the old size, leaving an extension whose
  worker did not parse.  So no worker file is ever replaced or
  removed: an install writes its worker, then replaces the manifest
  atomically, and leaves the workers of other builds (the
  `background.js` of earlier installs among them, some 50 KB each)
  where they are, since a Chrome that loaded an earlier manifest
  starts that manifest's worker from disk whenever the extension is
  switched on.
  Chrome runs the worker it loaded until the extension is reloaded
  (or Chrome starts again), so a refreshed install reaches a running
  Chrome by a reload, which the worker does itself: a `hello` of
  another version is refused, and a worker so refused (close code
  4004) that finds another version on disk than the one Chrome loaded
  calls `chrome.runtime.reload()`; one that finds its own version
  there asks again three seconds later.  A worker from before that
  answer keeps running: the tool's error asks for one click on the
  extension's reload button on `chrome://extensions` (`Chrome loaded
  its copy of the ... Bridge extension before the one in <folder> was
  installed`).  Chrome runs an unpacked extension only in Developer
  mode: with the mode off, an extension that reloads (by itself or by
  that button) or a Chrome that starts shows it switched off until
  the mode is on again.  No worker can tell the mode, so the Settings
  label says to leave it on and both errors of the tool name it.  Next
  to them `bridge-config.json` (mode 600) names the live port and
  token; the extension re-reads it on every reconnect attempt, so a
  restarted daemon on a new port is found without touching Chrome,
  and `stop` removes the file only while it still holds this server's
  token.  The `chrome_bridge` key of `config.json` (default `false`;
  the settings panel's "Drive my Chrome" checkbox, whose label gives
  the install steps and names the folder to load: the
  `chrome_extension_dir` of the `configData` reply, so the folder of
  the daemon the sheet is for, wherever its `$KISS_HOME` is) makes the
  daemon own the bridge: `chrome_bridge.start_if_enabled()` runs at daemon
  start and after every settings save (`takeover=True`, so its port
  and token replace whatever the file named; a start failure is
  logged and the first tool call retries lazily; switching the key
  off takes effect at the next daemon start) and
  `ChromeBridge.shutdown()` runs when the daemon stops.  With the key
  off, or in a CLI run, the first `use_my_chrome` call starts the
  bridge itself and waits up to 5 s for the extension to find the new
  config; a lazy start that finds another process's bridge still
  answering on the file's port (recognised by the `challenge` frame it
  sends, so a reused port does not count) does not steal the
  extension but returns `Error starting the Chrome bridge: another
  KISS process already serves the Chrome bridge on port N ... run this
  task through that process`.  That probe, the start of the server
  thread and the write of `bridge-config.json` are one critical
  section under the file's lock (`_config_lock`, given up after
  `_CONFIG_LOCK_TIMEOUT_SECS`, 5 s), so two processes starting
  together cannot both find no live bridge and both publish
  (`ChromeBridge._publish`): the socket is bound after the probe, the
  file is written atomically next, and the serving thread starts
  last, still under the lock, so the next starter's probe (which
  needs the server's challenge frame) waits for the lock and finds
  the bridge answering, while an extension connection arriving
  between the write and the thread's start queues on the bound
  socket; a start that the probe, the lock, the write or the thread's
  start refuses is stopped (`stop()`: the bound socket closed, a file
  written by then removed) and publishes nothing.  The bundled `/devtools` SEA
  (`devtools_sea.py`) packages this workflow — act, then verify the
  effect with `wait_for`, `list_console_messages`,
  `list_network_requests` and `evaluate_script` instead of re-reading
  the tree blindly, plus the two routes into the user's Chrome — as a
  system-prompt addendum and forces `use_web_tools` on; it replaced
  the former `browser-devtools` skill.  A browser the
  agent runs without a window is a headed Chromium on a private Xvfb
  display when one can be started (headless mode is the first thing
  bot-protection scripts detect); real headless mode is the fallback
  (macOS, Windows, Linux without Xvfb), and there the browser is
  given its own headed user agent through Chrome's `--user-agent`
  switch rather than a page script (which leaves `navigator.appVersion`
  disagreeing and failed every Cloudflare managed challenge): the
  first headless launch of a Chrome build in the process reads
  `navigator.userAgent`, turns `HeadlessChrome/<v>` into `Chrome/<v>`
  (`web_stealth.headed_user_agent`), remembers it per channel and
  relaunches with the switch; later launches get it up front, and a
  remembered string whose major version no longer matches the browser
  (Chrome updated under the daemon) is forgotten and relearned once
  (`WebUseTool._reconcile_headless_user_agent`).  A page that
  answers with a bot-protection interstitial is recognised by
  `web_stealth.challenge_vendor` (Cloudflare, Anubis, Akamai,
  Imperva, PerimeterX/HUMAN, Google's `/sorry/`, generic CAPTCHA
  pages, from the block page's own title and body wording; AWS WAF's
  blank token challenge and its "Human Verification" CAPTCHA puzzle
  also from the page's `*.awswaf.com` script URLs) and given
  `web_stealth.challenge_wait_secs(vendor)` to clear on its own: 12 s,
  45 s for Anubis's CPU-bound proof of work, 3 s for AWS WAF's puzzle
  (which only a person solves; its "Begin" button is never pressed),
  re-derived when one interstitial gives way to another's; a vendor with a widget
  gets it pressed like a person after a 1.2 s reaction time —
  Turnstile's box once, a 60-140 ms click; HUMAN's `#px-captcha`
  press-and-hold button (walmart, fiverr) with an 8-10 s hold under a
  drifting pointer (`press_hold_secs`, `hold_tremor`,
  `PRESS_AND_HOLD_BOX_JS`; the button itself is pressed when it has a
  usable box, found by its ARIA role in the widget's visible frame,
  `press_and_hold_box`, else the visible frame's own box, since
  fiverr's widget is a 600 px column with a 310 px button at its left
  edge and a hold at the widget's middle missed it; hidden honeypot
  copies of the frame have no usable box and are skipped) and, when
  the bar is still showing 5 s after release, a second hold
  (`widget_presses`); each press extends the wait by the vendor's
  allowance.  A page still blocked is
  reported to the model as a `Note:` naming the host, the vendor and
  the seconds actually waited, with the advice to call
  `show_browser()` so the user can complete the check or to use
  another source (`WebUseTool._settle_challenge`; a Google search
  that got the `/sorry/` page is re-opened on Bing instead, with a
  note saying so, and a semanticscholar.org paper or search page kept
  behind AWS WAF's puzzle — the site's policy for this network's IP
  address, which an un-automated Chrome meets too — is looked up on
  Crossref's open works API, `web_stealth.content_fallback`, rendered
  as a page by `CROSSREF_RENDER_JS` with a note naming the terms and
  the count, unless the user can see the page and may be solving the
  puzzle; when Crossref gives no works list the puzzle is reopened
  and reported as before).  A Playwright driver
  that died under the tool is detected before every call
  (`web_use_tool._driver_is_dead`, `web_stealth.DRIVER_EXITED_MESSAGE`)
  and discarded instead of waited on (`WebUseTool._discard_dead_driver`):
  no graceful close goes to it, every reference is dropped and the
  driver stopped; a Chromium the tool launched itself is killed by its
  recorded pid and the next call launches a fresh driver and browser,
  while in the Browser tab, where the browser is the daemon's and no
  pid is recorded, only the dead driver's tab is closed through the
  daemon (a popup followed just before is found by the target id
  recorded at adoption, `BrowserTabService.tab_for_target`) and the
  next call starts a new driver that reconnects to the daemon's
  browser and opens a new tab (`_attach_live`).  A driver that is
  alive but no longer answers is bounded the same way: the page reads
  whose `timeout` only the driver enforces (`wait_for_function`,
  `aria_snapshot`, `inner_text`) get the loop's own deadline a round
  trip (`_CDP_CALL_TIMEOUT_SECS`, 3 s) beyond the driver's
  (`web_use_tool._page_read`, failing with `Timeout <N>ms exceeded: the
  browser did not answer`); a call whose cancel the driver does not
  acknowledge either has the driver killed a round trip after its
  deadline (`_cdp_call`'s loop timer, `_kill_driver_unless_done`, left
  inert once the call ended), so the call fails at once and the next
  call relaunches; and a graceful close unanswered within its watchdog
  (`_CLOSE_WATCHDOG_SECS`, 15 s) plus a round trip, or a detach from an
  attached Chrome unanswered within a round trip, discards the driver
  whole (`_discard_dead_driver(why=...)`) instead of waiting on it.  The raw-input watchdog and the one on the
  driver's stop are armed and cancelled by `web_use_tool._under_timer`
  (`try`/`finally` around the guarded call, not a `with` block), so a
  Stop landing as a timer starts cannot leave it armed to kill the
  browser or driver of whatever runs next.
- **`auto_classify`** — per-run pre-run task classification.
  `None` falls back to the daemon's configured default (the settings
  panel's "Classify tasks before running" checkbox, persisted as
  `classify_tasks`; `task_classifier.classification_enabled`).
- **`use_memory`** — per-run persistent agent memory
  (`kiss.core.memoryfield`): the seven `memory_*` tools plus the
  memory protocol prompt block, and a recall the run makes for free
  before its first model call: the whole task text (its head and
  tail past 4,000 characters) is the search query, and the five best
  pages are appended to the task message under the heading
  `# Memory pages relevant to this task`, one line each with name,
  last update date and summary, for the model to `memory_read` /
  `memory_pull` as it needs (`MemoryTools.recall`; a failed search
  appends nothing).  A search (the recall's or `memory_search`'s)
  syncs and searches each selected memory on its own worker thread
  while the task thread waits in 0.1 s slices without joining the
  pool (`MemoryTools._search`, `tool_interrupt.result_interruptible`), so a Stop pressed
  during a first-time index build lands at once; the abandoned worker
  starts no further embedding batch once the cancel event is set
  (`VectorIndex.sync(cancel=)`, checked between batches), finishing
  the scan, the batch under way, a query embedding already in flight
  and the stale-row cleanup in the
  background, which is all process exit then waits for, and skips the
  search itself once the cancel is set (`MemoryTools._search_one`:
  nobody reads an abandoned worker's hits).  `True`
  enables, `False` disables,
  `None` falls back to the daemon's configured default (the settings
  panel's "Use persistent memory" checkbox, persisted as
  `use_memory`, or the daemon process's `KISS_USE_MEMORY` environment
  variable).  The override is inherited by `run_agent` / `run_parallel`
  sub-tasks and never bypasses the memory safety gates: a run without the basic
  toolset, a tool profile that keeps no memory tool (`shell`,
  `assistant`, `bash`, ...; the tools would be dropped anyway, so the
  recall search and the protocol block are skipped too), a Docker
  run, or a caller-supplied
  `model_config["system_instruction"]` stays memory-free even with
  `True` (`sorcar_agent._memory_root_for_run`).  Pages live in the directory named by the `memory_dir`
  setting, or `$KISS_HOME/memories` when unset;
  a run whose work directory is inside a git repository also attaches
  that repository's domain memory, a sub-directory named after the
  repository's main checkout directory (linked worktrees resolve to it
  through `git rev-parse --git-common-dir`, so they share one domain;
  `sorcar_agent._memory_settings`, `_repo_memory_domains`).
- **Worktrees are not a setting.**  A run starts in the user's
  checkout.  When `work_dir` is inside a git repository the agent
  calls the `create_worktree` tool before its first edit of a path
  inside the repository that is not git-ignored (tracked or not),
  which moves the run into a fresh worktree on its own branch (the
  checkout's uncommitted, staged and untracked files are mirrored
  into it, `GitWorktreeOps.copy_dirty_state`, from `git status
  --porcelain -z`, whose NUL-delimited, unquoted output keeps a
  pathname with spaces at an end, a newline, a quote or non-UTF-8
  bytes exact and lists both sides of a rename, as the merge flow's
  changed-file and conflict lists do, `merge_flow._git_paths` and
  the shared `_porcelain_entries` parser; only a rename, `R`, vacates
  its old name in the worktree, a copy, `C`, leaves its source; a dirty
  tracked submodule refuses the snapshot, `cannot snapshot the dirty
  submodule '<path>' of <repo> into a worktree`, since its content
  mirrored into the worktree's uninitialized gitlink would be files
  git never sees there, so no worktree is made and the tool answers
  `No worktree could be created for <repo> (detached HEAD, no commits,
  or a git failure; see the log). Edit its files in place.`),
  and `commit_and_merge` when done, which commits the branch and
  squash-merges it into the original branch: the branch's net change
  since its baseline commit (the one that captured the checkout's
  dirty state when the worktree was made, `branch.<branch>.kiss-baseline`
  in the git config) is applied as one `git cherry-pick --no-commit`
  of a synthetic commit holding the branch's tree with the baseline as
  parent (`GitWorktreeOps._squashed_commit`), one three-way merge
  whatever the branch's history, so an intermediate edit the task
  later reverted cannot conflict and a merge commit on the branch
  cannot stop it (a worktree without a baseline gets `git merge
  --squash`; the manual command a failed merge's result tells the user
  to retry with, `git_worktree._manual_merge_cmd`, is the same one
  cherry-pick of that synthetic commit, created inline by `git
  commit-tree <branch>^{tree} -p <baseline>`, where replaying
  `baseline..branch` commit by commit would refuse a merge commit on
  the branch); a conflicting merge is
  resolved by the bundled `/merge` SEA run as a sub-agent.  A merge
  that would write over a file git ignores — one on disk at a path the
  branch starts tracking, or an ignored directory in the way of an
  incoming file (`GitWorktreeOps.ignored_collisions`, what `git
  ls-files --others` lists at or under the path) — is refused before
  anything is applied, since both `merge --squash` and `cherry-pick`
  silently replace such a file and no stash holds a copy: the result
  reads as a merge conflict, the branch is kept, and the log names the
  files to move away; the checkout of the original branch before a
  merge runs with `--no-overwrite-ignore` for the same reason.  A
  repository nested inside the worktree (a clone with a `.git` of
  its own) is a checkout of its own that the enclosing worktree does
  not isolate (`sorcar_agent._in_checkout_of`): a tracked, non-ignored
  edit there needs its own `create_worktree(path)` first.  The commit
  message is generated from the diff and stamped with the user's
  prompt and the task's result (`kiss.agents.sorcar.commit_message`);
  the model call that writes it is billed to the task it commits for
  (`bill_to`), and when a worktree is retired after that task's row
  is already saved (a new task, a new chat, a tab disposal) the spend
  is added to the row and its finished ancestors with a persisted
  `usage_info` event (`_retire_after_task_saved`).  A Stop accepted on
  the `commit_and_merge` call while the `/merge` resolver runs inside
  it ends the resolver at its next safe point — before its next model
  request and once a step's tool results are in its trajectory
  (`KISSAgent._execute_step` calls `tool_interrupt.raise_if_interrupted`
  at both) — so the call ends as interrupted instead of the resolver
  requesting on, or returning a result, after the Stop.  A Stop during the
  merge takes back a half-applied squash when it safely can
  (`_undo_if_applied`, `_undo_merge`): the files the merge wrote go
  back to `HEAD` by git's own per-file check (`read-tree -m -u`), so a
  file an editor saved meanwhile is never overwritten, and the index
  is reset after them; a file that is no longer what the merge wrote,
  or one found where the move would create one, refuses the whole
  move and nothing is written, and when git then cannot reset the
  index the files are put back as the merge left them: either way the
  partial merge, conflicts included, stays in the tree and the index
  for the user to settle, with a warning in the log (an index that
  holds conflicts refuses the next merge).  The stash
  the merge made of the user's uncommitted changes is then popped
  (when that pop fails, the task's result carries a warning telling
  the user to run `git stash pop`), and a `git status` that fails
  keeps the worktree instead of deleting it.
  Both tools belong to the `edit` tool group; `WorkerSea.tools()`
  strips them, so a worker edits the caller's tree in place.  A
  worktree the agent created but never merged is presented to the
  user with the Merge / Discard bar; one with no changes against its
  base (no file its commits changed, no edits on disk or staged in
  its index, no untracked files) is discarded instead (announced
  after the task; silently for a stale resumed session), with any
  git-ignored output in it, but only when git could list its
  changes (`merge_flow._changed_files_of` answers `None`, not an empty
  list, when the comparison fails: a missing object, an unreadable
  index, a base that no longer resolves, or a `git ls-files --others`
  of the untracked files that failed or timed out walking a huge or
  hung tree, which is unknown rather than "no new files"), since the branch may hold
  the only copy of the task's work — such a worktree is kept and
  presented, with a warning in the log.  `use_worktree` and `auto_commit`
  are removed settings: `sea_settings.REMOVED_SETTINGS` refuses them
  and `uv run sea lint` flags a docstring that still mentions their
  getters (`sea_lint.REMOVED_GETTERS`).
- **`tool_profile`** — the name of the tool profile the run's
  built-in toolset is cut down to: `"full"` (everything), `"review"`
  (the `shell` and `browser` groups, the four read-only memory tools
  `memory_search`, `memory_pull`, `memory_read` and `memory_list`,
  `decide`, `summary` and `talk`; no editing, memory writing or
  dispatch), `"assistant"`
  (the `shell` group plus `ask_user_question`, `talk`, `decide`,
  `summary`, `set_model`), `"bash"` (`Bash` only — the bundled `/sh`
  agent's choice), or a tool group: `"shell"` (`Bash`, `bash_job`,
  `Read`, `run_commands_parallel`), `"edit"` (`Edit`, `Write`,
  `create_worktree`, `commit_and_merge`, `checkpoint`, `rollback`),
  `"browser"` (the fifteen `WebUseTool` tools listed under
  `use_web_tools`, `sorcar_agent.BROWSER_TOOL_NAMES`; the group
  only names them, the setting still decides whether they are
  built), `"memory"`, `"agents"` (`run_agent`, `agent_job`,
  `run_parallel`, `number_of_cores`), `"mcp"` (the configured servers' tools and the
  sign-in pair), `"skills"`, `"user"` (`ask_user_question`, `talk`),
  `"decide"`, `"control"` (`summary`, `set_model`).  Join several with
  `+` to keep the union of their tools (`"shell+edit+memory"`);
  `finish` is always added.  `"none"` builds no built-in toolset at
  all: the run's tools are `finish` and whatever `tools([])`
  returns (the bundled `/ask` agent's choice, so its only tools are
  `task_context` and `finish`).  `""` keeps the daemon's usual choice.
  An unknown name fails the task when it starts.
- **`docker_image`** — the Docker image the run's shell and file
  tools (`Bash`, `run_commands_parallel`, `Read`, `Edit`, `Write`)
  execute in: an image name starts a fresh container that is removed
  when the task ends (the task's `work_dir` is bind-mounted at the
  same path and is the container's working directory; the container
  is created, then started, under a `kiss-<12 hex>` name chosen
  beforehand, so a Stop landing inside the create call leaves nothing
  unreachable: cleanup force-removes whatever was made by handle or by
  name, a stop that fails at `close()` falls back to the same force
  removal, and a removal that still fails keeps the name on the
  manager so a later `close()` retries by name instead of forgetting
  a running container and deleting the shared directory it mounts;
  that shared directory, a host temporary directory mounted at
  `/testbed`, is emptied from inside the container before the removal
  by a `find -delete` that spares every bind mount nested under it,
  `docker_manager._sweep_command`, so a work dir under `/testbed` is
  the caller's still (best effort: the sweep prints a line every 5 s
  while it gets somewhere and is waited for as long as its stream
  speaks, but given up after `_SWEEP_TIMEOUT_S`, 30 s, of silence, so
  a wedged daemon or a stuck `find` cannot hold `close()` for good;
  what it had not deleted stays root-owned on the host), and then
  deleted on the host, the `rmtree`
  retried for `_RMTREE_RETRY_S`, 2 s, since Docker Desktop on macOS
  releases the mount points of nested bind mounts a moment after the
  container's removal returns),
  `container:<name-or-id>` attaches to a container the caller already
  runs (commands run in its own working directory, nothing extra is
  mounted, it is left running afterwards; a timed-out command whose
  exec the daemon delayed is still killed when it eventually starts,
  since the reaper of each exec (`docker_manager._ExecRun.reap`)
  watches the exec's own container, which outlives the `close()` that
  drops the attached manager's handle), `exec:http://127.0.0.1:<port>`
  runs them on a remote environment that has no local container at
  all through a local HTTP exec bridge (`exec_bridge.ExecBridge`
  stands in for the Docker manager: `Bash` and
  `run_commands_parallel` POST `{command, timeout, token}` to
  `/exec` and get `{exit_code, output}` back, whole, not streamed; a
  Stop POSTs `/kill` for the token from a helper thread waited on
  for at most `docker_manager._KILL_WAIT_S` (2 s), also when it
  reaches the waiting thread
  as an injected `KeyboardInterrupt`, so the kill is on its way
  before the call returns (a bridge that stalls on it keeps the
  helper thread waiting, not the task; a `/kill` the bridge
  refuses is swallowed); the wait as a whole ends
  `exec_bridge._REPLY_MARGIN_S` (60 s) past the command's deadline —
  a bridge that has not delivered the reply by then, stalled or
  trickling a body that never completes, gets the same `/kill` and
  the command is reported as timed out (exit code `None`; the request
  itself ends at that deadline too, a timer shutting its socket down
  wherever the reply is, status line, headers or body, since the
  socket's idle timeout never fires on a peer that trickles a byte at
  a time, `exec_bridge._shut_down`, the failure reading `the reply did
  not complete within the deadline`), and a
  request that failed after the bridge may have started the command
  (the reply connection dropped, an invalid reply) posts the same
  kill before its error propagates; the run fails at its start when
  `/ping` does not answer; the file tools are the same `Bash`-backed
  `DockerTools` a container run uses), `""` runs the tools on the
  host.
  `run_agent` / `run_parallel` sub-tasks share the task's container
  (`container:<id>`) or its bridge (`exec:<url>`), never the host;
  persistent memory is unavailable in a Docker or bridged run, while
  `bash_job` and `Bash(background=True)` work there through
  `container_jobs.ContainerJobs` (see "Background jobs" below).  The bundled `seas/coding/coding_sea.py`'s
  `ContainerHarness` is the reference user of both forms: its
  `settings()` returns `docker_image: "container:<id>"` for the trial
  container it is given, or the `exec:<url>` of the bridge a remote
  sandbox is served through, and the generated per-trial SEAs merge
  that dict into their own `settings()`.

### Call hooks (no `run()` parameter)

The class may also define three per-call hooks with no corresponding
`sorcar.run()` parameter or `settings()` key — a callable cannot be
JSON-serialized, so the hooks exist ONLY as SEA methods, called in
the daemon process (`sea_apply.apply_run` stages them on the command
fields below, together with `systemPromptHook` and `toolsHook`):

| Method                                 | Returns      | Staged command field |
|----------------------------------------|--------------|----------------------|
| `llm_call_hook(self, new_messages)`    | `list`       | `llmCallHook`        |
| `tool_call_hook(self, name, args)`     | `Verdict`    | `toolCallHook`       |
| `llm_response_hook(self, reply)`       | `Verdict`    | `llmResponseHook`    |

The hooks are passed to the underlying `KISSAgent.run()` of every
task-executor sub-session of the task's agent (internal helper
sessions, e.g. the tool-less summarizer that writes a failed
session's hand-off, are not hooked; a `run_agent` / `run_parallel` child gets the hooks of its own
layers — the `agent` the call names, or the hidden `sorcar` SEA when it
names none — not the parent's).
Per `KISSAgent.run()`'s contract:

- **`llm_call_hook(new_messages)`** — called before every LLM call
  with the list of new messages (those added to the conversation
  since the previous LLM call) about to be sent; its return value
  replaces those messages.  The `BaseSea` method returns them
  unchanged.
- **`tool_call_hook(name, args)`** — called before every tool call
  with the tool's name and arguments dict and returns a typed verdict
  (`kiss.core.tool_verdict.Verdict`, re-exported by `base_sea`):
  `ALLOW` lets the tool execute; `refuse(text)` suppresses the call
  and *text* is given to the model as the tool's result;
  `allow_with_note(text)` (also re-exported) lets the call run as
  asked and appends *text* to its result, after the agent's result
  hook has read the plain result — for a rule the model should act on
  at its next step without losing this one (a blank *text* is a
  `ValueError`; `finish` carries no note).  The `BaseSea` method
  returns `ALLOW` for every call except, at every tenth step of a run
  that has the `summary` tool, a call other than `summary` or `finish`
  while the summary is owed, which gets
  `allow_with_note(SUMMARY_DUE_NOTE)` (see [Customizing every
  run](#customizing-every-run)).
  The model never reads a refusal bare: the executor hands it
  `kiss.core.tool_verdict.refusal_result(name, text)` — `REFUSED: the
  <name> call did NOT run and had no effect. <text>` — and the next
  allowed call of a *different* tool gets `REISSUE_REMINDER` appended
  to its result ("your earlier <name> call was REFUSED and never ran;
  re-issue it now if it is still needed"); a call of the refused tool
  itself settles the reminder silently and `finish` never carries
  one.  A literal `None`
  or string (the
  allow / refuse spellings of older hooks) is the `verdict` finding of
  `uv run sea lint`, and `--fix` rewrites it.
  `end_run(result, success=True)` is the third verdict: the call does
  not run, no later call of the same turn runs, and the run ends with
  *result* as its result (the `summary` of the `finish` YAML, with
  that `success`) — for a rule of the SEA's own that says the task is
  over (the report was accepted, the episode's turn budget is spent)
  instead of waiting for the model to call `finish`.  The implicit
  finish of a model that only talks probes the hook with
  `("finish", {})`, so an `end_run` there ends the run too.
- **`llm_response_hook(reply)`** — called after every model reply,
  before any tool call of the reply runs, with a
  `kiss.core.llm_reply.LLMReply` (re-exported by `base_sea`): the
  reply's `text`, its `tool_calls` (`{"name", "arguments"}` each),
  its `usage` (`input_tokens`, `output_tokens`, `cache_read`,
  `cache_write`, `cost`, `stop_reason`), the `step`, and the run's
  `budget_used` / `total_tokens_used` so far.  Returns `ALLOW` to act
  on the reply or `end_run(result)` to end the run (none of the
  reply's tool calls run); a `refuse(...)` here is a broken SEA.  The
  `BaseSea` method allows every reply and reads its `step` into
  `self.step` (the `Steps:` count of the tool-result footer), marking
  a summary due at every `SUMMARY_EVERY_STEPS`-th step of a run that
  has the `summary` tool; a step that does not grow is a continuation
  session restarting the count, and the previous session's debt is
  dropped with it.
  With `llm_call_hook` (rewrite the next request) this is every
  exchange with the model — what used to need a proxy between the
  agent and the model to count an episode's tokens or end it.  A tool
  Codex ran natively during the turn has run already, and an `end_run`
  the `tool_call_hook` gave on one of those calls is the run's result
  even if the hook here says `ALLOW` (the hook still sees the reply)
  and even if the turn fails after it.

A hook that returns the wrong type (a non-list, a non-`Verdict`)
fails the task with a diagnostic error, like every wrong-typed value.
In an inheritance chain the `llm_call_hook` results are threaded base
first, the first `tool_call_hook` refusal wins (an allowance carries
the notes of the allowing verdicts joined in chain order, and is
`ALLOW` itself when none has one; `sea_commands._first_refusal`), and
the first `llm_response_hook` `end_run` wins.

```python
# guarded_agent.py
from typing import Any

from kiss.agents.seas.base.base_sea import ALLOW, BaseSea, Verdict, refuse


class GuardedSea(BaseSea):
    def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
        if name == "Bash" and "rm -rf" in str(args.get("command", "")):
            return refuse("Blocked: destructive command")
        return ALLOW
```

### Only `settings()` configures a run

The dispatcher and the task runner evaluate `settings()` alone when
they need a script's timeout, model or work directory before
the run exists (`sea_commands.sea_settings`), so `settings()` must be
cheap and side-effect-free there.  `prompt(task)` runs inside
`apply_sea`, once per run; `system_prompt()` and `tools()`
run once when the agent starts, the three call hooks on every LLM
call, reply and tool call.  A method named differently from the contract
(`def model(self)`, `def add_to_tools(self)`, ...) is an
ordinary method the daemon never calls; `uv run sea lint` does not
flag such a definition, only a docstring that still describes the
earlier contract's getters (`stale-docstring`), so check the method
names yourself.  A `settings()` that reads a key its bases lay
(`not settings["use_web_tools"]`) is fine in a run, where the bases
are folded first; `sea lint`, which also runs the script's own
`settings()` over `{}` for its `redundant-key` comparison
(`sea_commands.own_settings`), takes the failure there as "no own
dict to compare", not as a `broken` finding, and logs nothing for it.
`sea lint --fix` rewrites the fixable findings first
(`sea_lint.fix_sea`, each rewrite printed as `fixed <path>:<line>:
<note>`), replacing the file atomically (`atomic_write_text`, so a
daemon loading the script meanwhile reads the old or the new text,
never a truncated one) and never saving a rewrite that no longer
parses: that is the codemod's own failure, printed as `not fixed
<path>: the rewrite no longer parses and was not saved: …`, and the
script is left as it was.


## Tools: `tools(tools)`

An SEA supplies tools to the LLM agent through one method,
`tools(self, tools)`, which receives the run's toolset so far (a copy
of the built-in toolset of the run's tool profile plus the tools
inherited from the calling task) and returns the **list of callables**
the run gets (a file path is not accepted).  There is no other way to
give the agent extra tools: `sorcar.run()` has no tools parameter,
because a callable cannot travel the wire.

- `return tools + [mine]` **adds** to the built-in toolset (`Bash`,
  `Read`, `Edit`, `Write`, browser tools, ...), or to the tool profile
  the SEA's `tool_profile` setting selects; the method may also drop
  or replace entries.
- With `"tool_profile": "none"` the method receives `[]` and what it
  returns plus `finish` is the agent's **entire** tool set; the
  built-in toolset is not built.  Pair it with a `system_prompt()`
  written for those tools.

The daemon calls `tools()` once, when the agent starts, and hands the
returned callables to the agent; whatever the list holds beyond the
built-ins is the run's extra toolset, which its own `run_agent`
sub-tasks inherit.  A single file provides both settings and tools.
The file is re-executed from source on every run (no `__pycache__`),
so keep module-level side effects cheap or idempotent.  Executions of
one file are serialized per file, not globally (`sea_settings._executing`):
a thread that finds the file executing on another thread waits for it,
while a slow import of some other SEA stalls nobody; the wait is
re-entrant for a file that loads itself at import time, two files that
load each other at import time on two threads raise `SeaError` in the
thread that closes the cycle, and a claim left behind by a thread the
task's Stop or a per-tool Stop interrupted mid-execution (the thread
may be alive and carrying on with its task after a per-tool Stop) is
dropped by the next waiter, which re-checks once a second
(`_EXECUTING_RECHECK_SECONDS`) whether the owner's stack still holds
an `execute_python_file` call of that file (`_in_execution`).

```python
# self_contained_agent.py
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea


def double(n: int) -> int:
    """Double a number.

    Args:
        n: The number to double.
    """
    return n * 2


class DoubleSea(BaseSea):
    def prompt(self, task: str) -> str:
        return "Double the number 21."

    def tools(self, tools: list[Any]) -> list[Any]:
        return tools + [double]  # built-in toolset + double
```

### Tool function requirements

Each tool function must:

1. Have a **name** — the function name becomes the tool name the LLM
   sees.  The run keeps one tool per name, the later entry of the list
   `tools()` returns winning, so a tool named like a built-in (`Bash`,
   `Read`) silently replaces it and two of your own with one name
   leave only the second; `finish` is added by the run itself.  Use
   distinct names unless replacing a built-in is the intent.
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

The derived schema maps Python types to JSON-schema types and loses
what a schema can state beyond that — an enumeration of allowed
values, an array of objects with required fields.  A tool whose
arguments are constrained declares its own schema with
`@tool_schema` (re-exported by `base_sea`); the model then sees that
schema verbatim, whether the provider takes native tool definitions
or reads the text catalog of tools, and the function keeps its name,
signature and behaviour:

```python
from kiss.agents.seas.base.base_sea import tool_schema

@tool_schema(
    parameters={
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["low", "high"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"file": {"type": "string"}, "line": {"type": "integer"}},
                    "required": ["file", "line"],
                },
            },
        },
        "required": ["severity", "findings"],
    },
)
def report(severity: str, findings: list[dict[str, Any]]) -> str:
    """File the findings of the review."""
    ...
```


## The built-in toolset

By default — no `tool_profile` setting, with or without a `tools()`
method — the agent gets `finish` (always present) and the
built-in KISS Sorcar toolset — `Bash` (with
`background=True` for detached jobs), `bash_job` (wait for / tail /
kill a background job), `run_commands_parallel` (several shell
commands at once, no LLM sub-agents), `Read`, `Edit`, `Write`,
`create_worktree` and `commit_and_merge` (the last two not in a
Docker run, whose file tools work inside the container),
`checkpoint` and `rollback` (named snapshots of the git working tree,
see below; a top-level host run only — a sub-agent edits its caller's
tree in place, so a rollback of its own would undo the caller's and
its siblings' work, and a Docker run's file tools are
container-backed, so the pair is not registered there),
`ask_user_question` (each ask is broadcast as an `askUser` with a
fresh `questionId`, `task_runner._publish_ask`, and a `userAnswer`
naming that id is delivered only while that ask is the pending one,
`commands._cmd_user_answer`: every surface shows the same question, so
one surface's answer arriving after another's was taken, when the agent
may already be asking again, is dropped instead of answering the next
question; an answer naming only the `question` text is held to the
text, one naming neither, an older client, is delivered as before, and
a replay of a tab with a pending ask carries the id), `talk`,
`set_model`, `summary`, `run_agent`,
`agent_job`, `run_parallel` and `number_of_cores` (the last two only
when the run's `is_parallel` is on, the user's parallel-mode toggle),
the fifteen browser tools (when `use_web_tools`; listed under that
setting), `decide` (when
`OPENROUTER_API_KEY` is configured, the `classify_with_decisions`
key in `$KISS_HOME/config.json` is on (the default; it has no settings-panel toggle)
and the decisions model is in the catalog,
`decide_tool.decisions_tool_available`; `decide(state, questions)`
takes `questions` as a JSON string, a non-empty object `{name:
question}` where a question holds only `type` (`noul`, `choice` or
`score`), a non-blank `instructions` and `criteria` — a non-empty
`{option_id: description}` map or list of non-blank strings for
`choice`, at least two ordered non-blank rubric levels for `score` —
and is refused as `Error: ...` otherwise (`parse_questions`): the
error of JSON that does not decode says where decoding stopped and,
when the text ends before its JSON does, that a closing brace is
probably missing, and a field a question does not take is refused as
another question probably nested in it), the `memory_*` tools (when
memory is enabled), and any configured skill and MCP-server tools —
**plus** your extension tools.  A restricted tool profile (a reviewer
sub-agent dispatched with `run_parallel(..., tool_profile="review")`)
filters that built-in set, including the memory tools; extension tools
are still appended.  An MCP server's tools are named
`<server>_<tool>` and built last, after the profile filter: a
synthesized name that collides with a tool the run already has (the
built-ins actually built, the static built-in list
`mcp_servers._RESERVED_TOOL_NAMES` — every name above including
`checkpoint`/`rollback`, the fifteen browser tools, the `memory_*`
tools, the MCP sign-in pair and `cron_job`, whether or not the run
actually has them, so a profile change never un-reserves a name — the
tools passed to `run(tools=)`,
the tools inherited from the calling task and the MCP wrappers built
before it, passed as `make_mcp_tools(work_dir, taken_names)`) gets a
numeric suffix (`run_parallel_2`) instead of aborting the tool
registration; a name longer than the 64 characters the providers'
tool grammars accept (`mcp_servers._MAX_TOOL_NAME`) is cut and ends
in eight hex digits of the whole name's SHA-1, the collision suffix
kept within the bound (`_bounded_tool_name`), so two long names
sharing a prefix still differ; the `mcp_permissions` deny rules are judged on the
un-suffixed `<server>_<tool>` name, before the suffix is chosen, so a
denied tool never reserves a suffix.  A SEA's `tools()` hook runs after the MCP wrappers are
built and is not in that set: a tool it adds under an MCP wrapper's
name replaces the wrapper in the final name-keyed toolset.  A stdio
server configured in a project file (`<work_dir>/.mcp.json`,
`<work_dir>/.kiss/mcp.json`) is started with the run's absolute work dir as its working
directory (`load_mcp_servers`, `MCPServerConfig.cwd`); one from the
user's `$KISS_HOME/mcp.json` keeps its own.  A wrapper's parameter
schema is reduced from the tool's `inputSchema`
(`mcp_servers._declared_schema`): a node is read through itself, its
local `$ref` target and its `allOf` members, the first part to carry
a keyword giving it while `properties` and `required` are joined;
`type`, `description`, string `enum`, `items`, `anyOf`/`oneOf`
members are kept and every validation-only keyword (`title`,
`default`, `format`, `additionalProperties`, bounds) dropped; a
reference already on its own path (a cycle), or any reference once
the declaration has grown to `_MAX_DECLARED_NODES` (5,000), gives its
target's type only, and a node left without a type (a remote or
dangling reference) declares an object when it has properties, an
array when it has items, a string otherwise.  A call's result is the
text of its text blocks and embedded text resources joined by
newlines (`_result_text`); an image, audio, binary-resource or
resource-link block becomes `[<type> content omitted: <URI>]` (no
URI: `[<type> content omitted]`), `structuredContent` is rendered as
JSON only when the result has no content block at all, an empty
result reads `(empty result)`, and `isError` prefixes `Error: `.

Every MCP tool built this way is defined in every model request.  A
user can have servers deferred instead (`"alwaysLoad": false` in a
server's `mcp.json` entry, or an `mcp_upfront_tool_chars` budget in
`$KISS_HOME/config.json`; neither is set by default, see
`connectors/README.md`).  `make_mcp_tools` then leaves the deferred
servers' wrappers out and appends `mcp_tool_search` and
`mcp_tool_call` (`mcp_tool_search.make_deferred_tools`; numbered like
any synthesized name on a clash): the first lists the deferred tools'
names in its description and returns the definitions a query asks
for, the second runs a deferred tool whose definition the calling
agent's model received in an earlier response.  `KISSAgent._execute_tool`
stamps each call's `ToolCallToken.response` with the agent's id and
the step of the response the call came in, which is how
`DeferredTools` tells a definition returned by the same response (the
call was written without it and is not run) and one given to an
earlier session of the task (a new agent with a new conversation,
which gets the definition again) from one the model has read.  A
model that runs its tools inside its own turn (Codex) has no step, so
there only the agent is checked.  It also appends one tool per
deferred tool under that tool's own name, declaring that tool's
schema and marked `unlisted` (`kiss.core.tool_schema`): no model
request and no text catalog of tools carries a definition of an
unlisted tool, and the agent still runs it when it is called, so a
model that calls a deferred tool by the name it read in the list is
answered as `mcp_tool_call` would answer (the declared schema types
the arguments of a call a CLI model wrote as text).  Codex answers a
native call to a name it was given no definition of by itself, so
there a deferred tool is reached through the two listed tools only.
The run's toolset is the same in every request, which keeps the
providers' prompt caches valid.  A search that matches nothing and a
call that names a tool which is not deferred (the agent's own
`finish`, say) are answered with a note that such a tool needs no
lookup and is called directly, under its own name: a small model was
seen looking up `finish` among the deferred tools and, told only to
try other keywords, calling server tools nobody had asked for.
Deferral costs accuracy on some models and did not improve reasoning
in the evaluation made for it; `reports/mcp-tool-deferral.html` has
the measurements.

### What the shell and file tools enforce

The host tools (`kiss.agents.sorcar.useful_tools.UsefulTools`, with
the guards of `shell_guards.py` and `edit_guards.py`) carry rules a
SEA's own `tool_call_hook` need not restate:

- **Secrets.**  A value of the key store (`$KISS_HOME/api_keys.env`)
  of at least 8 characters, whatever its variable is called, is
  shown as its variable's name in tool output before the output is
  stored or sent.  Two things are left alone: a shorter value, which
  ordinary output would hold by chance, and the region (`AWS_REGION`),
  because `us-east-1` is a public word of source files and an edit
  made from the replaced text would fail or write the name into the
  file.  The replacement happens in three places:
  `KISSAgent._execute_tool` replaces it in every tool result, a
  refusal of the tool-call hook included, before the `tool_result`
  event is printed and the result joins the conversation
  (`secret_redaction.redact_secrets`); the output shown while a
  command runs passes a `StreamRedactor`, which holds back an end
  that may open a secret until the next piece settles it (one per
  stream in a container, where stdout and stderr interleave); and a
  `Bash` result is redacted before it is cut to its budget, so a cut
  inside a key leaves no part of it.  The store is read again when
  the file changes, so a key saved mid-run is covered from the next
  output on, and a value once read stays covered while the daemon
  runs: a key rotated or removed mid-run, or a store that goes
  missing, does not uncover what a running command still prints.
  Not covered: the spill files and job logs on disk
  (read back through a tool, they are redacted), a transformed copy
  of a value (base64, URL-encoding) and a value the model itself
  types into a tool call.
- **Output budgets.**  A `Bash` result keeps `max_output_chars`
  characters, head and tail around a marker; the default is the
  daemon's `tool_output_default_chars` (8,000, `KISS_TOOL_OUTPUT_DEFAULT_CHARS`)
  and a request is capped at `tool_output_max_chars` (50,000,
  `KISS_TOOL_OUTPUT_MAX_CHARS`).  Output over the budget is written in
  full to `tmp/tool-out/<tag>-<n>-bash.txt` under the work dir and
  the marker names the file (`UsefulTools._new_spill` names it and
  `_Capture` writes the output to it as it arrives, keeping only the
  first and last million characters in memory; the tag is per
  tools instance, so a parent and its reviewer never overwrite each
  other's spills; the path is shown in POSIX form on every OS,
  `useful_tools.SPILL_DIR`, since the model pastes it into `Bash`,
  which is bash on Windows too, and the file holds the output
  verbatim, written with `newline=""`; a sub-agent's spills — and
  its `run_commands_parallel` and `run_parallel` reports below — go
  to `tool-out/` of its private folder instead, shown by their
  absolute path, `useful_tools._tools_dir`, so nothing of them is left
  in the work tree).  Only the output is cut to the budget: the
  `Error (exit code N):` line of a host result, the `[exit code: N]`
  line of a container or bridged one and the `Job <id>: <status>` line
  of a `bash_job` report stand whole whatever the budget
  (`useful_tools._format_bash_result`, `_format_job_report`), since
  cut with the output a small budget dropped the status and a failed
  command read as one that passed.  A host `run_commands_parallel`
  report longer than the ceiling (`tool_output_max_chars`) is written
  whole to `tmp/tool-out/<tag>-parallel.txt` (`useful_tools.spill_report`,
  the tag six random hex digits) and the result is its digest
  (`digest_parallel_report`): the tally line, one `[i/N] exit 0 in
  1.2s: <command>` line per command (the command cut to 160
  characters), then for each command that did not succeed the last
  `DIGEST_TAIL_CHARS` (1,200) characters of its output under a `###
  [i/N] ...` header naming the section of the file that has all of it;
  the container and bridged `run_commands_parallel` pass no `spill`
  and return the whole report.  A `run_parallel` result over the same
  ceiling goes to `tmp/tool-out/<tag>-agents.txt` the same way
  (`agent_dispatch.digest_agent_results`: `N sub-tasks; the whole
  result is in <path>, this is its digest:` and one `[i/N] ok|FAILED:
  <first 200 characters of the summary as text>` line per child, or
  `[i/N] running (job <id>)` for one still running and `[i/N] FAILED:
  <the error>` for one whose dispatch could not run).  The coding gate
  reads sub-task outcomes from the digests as it does from the full
  reports; how a shell command ended it does not read from the text
  at all: `Bash` and `run_commands_parallel` (host, container and
  bridge) hand the exit of each command over as data on the calling
  thread (`shell_exit.hand_over`, in the gate ledger's words — the
  exit status, `timeout`, `background` with the job id, `refused`,
  `error`, and for a fan-out command `killed by signal N` — which the
  gate's result hook takes back, `shell_exit.take`), since a command's
  own output may hold an `Error (exit code 7):` line or the section of
  a later command of the report; a `Bash` command made as a
  `Write`/`Edit` (below) ended as the last call made for it did, and
  only a result no tool returned — a refusal, a raise, a Stop, a job's
  report, which opens with the tool's own lines — is read as text.
  `memory_list(max_output_chars=...)` has the same
  default and ceiling, and a `Read` with no range keeps the default
  budget too, ending in a note that names the `start_line` to
  continue from.  A host `Bash` result of a command that only read
  (`grep`, `cat`, `ls`, `sed -n`, `head`, `tail`, `git log`/`status`/
  `diff`/`show`, `find` without an action, `wc`, no output
  redirection; `useful_tools._is_read_only`), took under 2 s and
  returned under 2,000 characters ends with the one-line
  `[hint: this turn made one short read-only call; ...]`
  (`BATCHING_HINT`), at most once per `BATCHING_HINT_EVERY` (20) `Bash`
  calls of the tools instance (`UsefulTools.batching_hint`).
- **`Read` windows.**  A `Read` of a file longer than
  `read_outline_lines` (2,000) with no range returns an outline of
  the file instead of its text when at least five definitions or
  headings are recognised in it (otherwise the first budgeted window,
  as above); a window whose text the model has already been
  shown — keyed by a hash of the window's lines, so the same text
  under another range or another path counts too — returns the
  one-line `Unchanged since your earlier Read at step N (lines a-b of
  <path>): the content is already in your context above. Pass
  force=True to re-read it.` unless `force=True` (`N` is the step
  count the model sees in its footers, `SorcarAgent._executor_step`,
  the `Read`'s ordinal without an agent; the container and bridged
  `Read`, `DockerTools.Read` (`_dedupe`), returns the bare `Unchanged since
  your earlier Read at step N (lines a-b of <path>)` without that
  suffix; the executor clears that
  bookkeeping, `UsefulTools.forget_reads` and the container tools'
  `DockerTools.reset_reads`, the run's `context_reset_hook`, when a
  compaction or a new session drops the earlier copy); `start_line`
  past EOF is an explicit error, not empty content, and does not count
  as a read of the file (it shows nothing of it).  A `\r\n` is shown
  as `\n` and a lone `\r` stays in its line, which ends where `grep`
  and `sed` end it; the numbered regions an `Edit` result or refusal
  shows are cut the same way (`edit_guards.numbered_region`), so a
  line number is the same in all of them.  A `Read` of an image, PDF, audio or video file (a
  MIME type in `model.READ_TOOL_BINARY_MIME_TYPES`) of at most 20 MiB
  (`useful_tools._MAX_BINARY_READ_BYTES`) returns `Read binary file
  <path> as <mime> (<n> bytes); content attached below.` with the
  bytes attached, which a vision model sees as the file (one of those
  types over the limit is refused by its size, `Error: Binary file
  <path> is too large to embed inline …`, before the text attempt
  would read it whole); the
  container and bridged `Read` do the same, fetching the bytes with
  one exec that prints the file's size and its base64
  (`DockerManager.read_bytes`, `ExecBridge.read_bytes`,
  `docker_manager._FETCH_SCRIPT`: `base64 < path | tr -d '\n'`, POSIX
  tools only, so a BSD `base64` without GNU's `-w0` works too; a
  fetch whose base64 does not decode to the announced size is an
  `OSError`, `cannot read <path>: fetched N of its M bytes`, since the
  pipeline's exit code is `tr`'s and would hide an encoder that died
  mid-file), so neither the text decoding nor
  the output cap of the container `Bash` touches them.
- **Read before modify.**  `Edit` and `Write` of an existing file the
  model has not been shown in this session are refused — except of a
  scratch file (`useful_tools._is_scratch_path`, under a `tmp`
  directory inside the work dir, under `$KISS_HOME/cron/work` or in a
  sub-agent's private folder, `<system temp dir>/kiss-sub-…`),
  which `Write` overwrites and `Edit` edits unread, a note of an
  earlier session say, the `Edit` result showing the region changed —
  and a refused `Edit` returns the numbered
  region around `old_string`, after which the file counts as read, so
  the next `Edit` goes through without a separate `Read`.  A `Bash`
  (or `run_commands_parallel`) command whose last `cat`, `sed -n`,
  `grep -n`, `nl`, `head` or `tail` command names exactly one existing
  file by a literal operand (the command is read as bash reads it,
  `shell_scan.simple_commands`: quotes, comments, subshells and `cd`s
  are followed, and an operand holding an expansion or a glob does
  not count), exited 0 and printed a line of it counts as a read of
  that file (`edit_guards.shell_view_target`, `output_shows_file`,
  which reads only the first and last 4,000,000 bytes of a longer
  file, where a `head` or `tail` of it shows).  The
  bookkeeping is keyed by repository-relative path
  (`edit_guards.repo_read_key`), so a file read at the main checkout
  stays read at its `.kiss-worktrees/` path and vice versa.
- **`Edit`.**  `Edit(file_path, edits='[{"old_string": ..., "new_string": ...,
  "replace_all": false}, ...]')` applies several replacements in one
  call, in order and all or none (`old_string` / `new_string` are then
  ignored; a `replace_all` that is not a JSON boolean is refused,
  `Error: edit N: replace_all must be true or false`, since read by
  its truth the string `"false"` replaced every occurrence); every
  `Edit` result shows the numbered lines around each
  replacement in the file as it is after the whole batch (an earlier
  edit's region is reported where a later edit moved it; the region
  stops at the file's last line, and a file the edit emptied says `The
  file is empty after the edit.`), so no
  verifying `Read` follows.  An `old_string` that is not in the file
  verbatim is matched ignoring whitespace (`edit_matching.tolerant_matches`:
  whole lines compared with every run of blanks collapsed to one and
  trailing whitespace dropped, so a copy whose spacing the model
  normalised, or indented by a different depth, still lands — an
  unindented copy of an indented line does not, its leading blank
  being part of the line): one such match is replaced in place, the
  replacement re-indented by the extra indentation the file's lines
  share (`reindent`); several are refused with
  `Error: String not found verbatim; ignoring whitespace it matches N
  places (not unique). Include more surrounding lines in old_string.`;
  none with `Error: String not found in file.` followed by the
  numbered lines of the closest region (`edit_matching.nearest_region`;
  in a long file only the windows sharing words with `old_string` are
  compared, within a budget of 200,000 characters,
  `edit_matching._BUDGET`, where comparing every window took minutes)
  and `Re-issue the Edit with the exact text of those lines as
  old_string.`  `replace_all` replaces the exact text only, and
  `Edit(..., exact=True)` matches every `old_string` verbatim, with
  neither the whitespace-tolerant retry nor the CRLF one (an LF
  `old_string` missing from a CRLF file is retried as CRLF, and the
  `new_string` of a one-line `old_string` found as it is takes the
  file's CRLF too)
  (`useful_tools._edit_locked`): the `Edit` a shell rewrite is applied
  as carries the flag (`useful_tools.edit_args`), so a `sed -i` or a
  python `.replace` that would have matched nothing in the file's text
  changes nothing when the call is made, and is remembered with its
  pairs until that `Edit` lands (`UsefulTools._pending_edit`; for the
  container and the bridge on the manager,
  `docker_manager.PendingEdit`, taken by `DockerTools._edit_as_rewrite`):
  the pairs are written only while they are still the ones the
  command stands for against the text the file holds then
  (`MechanicalEdit.edits_for`; the agent's hook runs in between, and
  a writer meanwhile may leave a line holding a `sed` pattern twice),
  else `Error: <file> changed since the <python heredoc | \`sed -i\`
  call> was read as this Edit, which the command no longer stands
  for: nothing was written. Re-issue the command.`; any other call
  forgets it.  The host
  `Write` replaces the file atomically (a sibling temporary file,
  `.<hex>.tmp` — a name that does not carry the file's, which near the
  file system's limit leaves no room for a longer sibling — moved
  over it with `os.replace`, its mode, owner and group kept,
  `useful_tools._write_file`),
  so a reader sees the old text or the new, never a half-written file;
  a file with other hard links, or one whose directory refuses the
  temporary file, is written in place instead (the rename would cut the
  file from its other names), without that guarantee; so is a file the
  user may not write (a read-only file is refused, as the shell's `>`
  refuses it), one the rename over is refused as busy (a mount point: a
  single file bound into a sandbox or a container) or whose owner and
  group the temporary file cannot be given; any other failure (a full
  disk) leaves the old text.  `Write`
  refuses a path that exists and is not a regular file (`Error:
  <path> exists and is not a regular file (directory/FIFO/device/socket);
  refusing to write to it.`; `Edit` reports one as `Error: File not
  found: <path>`, and a `Read` of one returns `Error: <path> is
  not a regular file (FIFO/device/socket); reading it could block
  forever. Use Bash (which has a timeout) if you really need its
  contents.`), and every read, append and in-place write the host
  tools make checks the type again on the descriptor it opens,
  non-blocking (`useful_tools._open_regular`), so a path swapped for a
  FIFO between the check and the open is refused instead of blocking
  in `open` under the lock every `Write` and `Edit` of the daemon
  shares.  The container
  and bridged `Write` store the content in base64 pieces, one exec
  each, into a uniquely named staging file beside the file the target
  resolves to (`readlink -f`, so a symlink is written through and
  stays) and publish it with one `mv -fT`, the replaced file's mode,
  owner and group kept, a file with other hard links, a read-only one,
  one whose owner the staged file cannot be given or one the `mv` is
  refused over (a file mounted into the container, as Docker mounts
  `/etc/hosts`) getting the staged bytes copied into it instead, and a
  directory that takes no sibling staging the pieces under `/tmp`
  (`docker_tools._store_commands`; a target that exists and is not a
  regular file is refused in the host `Write`'s words); when more than
  one piece follows, the first exec notes the resolved path under
  `/tmp` (`<staging name>.note`) and the later ones read it there, so
  a link repointed between two execs cannot send the later pieces, the
  publication or the discard of a failed store to another file.  A
  store or an edit script that ended with anything but its success
  line (a traceback, a failed payload store) is reported as `Error:
  <that output>` (`docker_tools._failure`), so the rest of a compound
  command is not made after it.  The host
  `Edit`'s read-modify-write and
  the host `Write`'s overwrite or append run under one process-wide
  lock (`useful_tools._FILE_WRITE_LOCK`), so the tool instances of a
  task and its sub-agents — threads of one process editing the same
  tree — cannot undo each other's change, both reporting success,
  when one's write lands between another's read and write of the
  same file.  The
  Docker `Edit` accepts the same `edits` list but applies it
  pair by pair: the first failure stops the batch and the error names
  the edits already applied, and a task Stop ends it at once with the
  stopped edit's result; its whitespace-tolerant retry runs on the
  file's bytes fetched with `read_bytes` (a tools instance without a
  `read_bytes` stops at the exact match; the matched lines are then
  replaced at their byte offset in the file, so the same text inside a
  longer line is not the match; `exact=True`, the flag the
  `Edit` a shell rewrite is applied as carries, skips that retry and
  the CRLF retry of an LF `old_string` in the container as on the
  host; the replacement itself is a python script, or a perl one in a
  container without python, that publishes as the host `Write` does —
  a sibling renamed over the file with its mode, owner and group, in
  place for a hard-linked, read-only or mounted file), its messages `Error: String
  not found in file; ignoring runs of blanks and trailing whitespace
  it matches N places, so add surrounding lines to old_string to make
  it unique` and `Error: String not found in file. The closest region
  is lines A-B of <path>; pass its text verbatim as old_string:` with
  the region's text.
- **Shell rewrites applied as `Write` / `Edit`.**  A `Bash` command
  that is the shell's way of making one `Write` or `Edit` is made as
  that call instead of being refused, so the change is logged and
  reviewable and the model pays no turn to re-issue it
  (`UsefulTools._rewrite_instead`).  Recognised, when the
  conversion is exact (`edit_guards.heredoc_copy`, `mechanical_edit`):
  a `cat` (no option) or `tee` (`-a`/`--append`/`-i`/`--ignore-interrupts` at most)
  here-document copied into one literal file (`cat > f <<'EOF'`,
  `cat >> f <<EOF`, `tee -a f <<EOF`, a `~/x` target resolved with the
  home directory unless more than the tilde may expand in it, a
  variable, a pattern or braces; a target that can only name a
  directory, `f/`, `.` or `..`, is no copy and no edit below, the
  shell opens no file by it, `edit_guards._names_a_directory`; an
  unquoted tag only when the body holds no `$` or
  backtick), which stands for `Write(file_path, content=<the body>)`,
  the current text in front of the body for an append; a `python -` or
  `python3 -` heredoc that reads one file by a string literal (`open(p).read()`,
  `Path(p).read_text()`, a `with` block), applies `.replace(<str>,
  <str>)` one or more times and writes the text back to the same file
  (`edit_guards.python_edit`; `re.sub`, a loop, two files or an
  `encoding`/`newline` keyword that would change the bytes are not
  mechanical, nor is an import that rebinds a name the pattern reads,
  `import os as p`, `from gzip import open`, a `Path` that is not
  `pathlib`'s or a `*` import, nor any import but one of a module
  every python has and whose import does nothing, `os`, `sys`, `re`,
  `json`, `pathlib`, `shutil`, ... (`edit_guards._INERT_MODULES`), or
  `from pathlib import Path` — another module may be missing or act
  before the write — `_PythonRewrite.imports`, nor a
  program python itself would not run as far as its write: one that
  parses but does not compile, a relative import, `read_text` /
  `write_text` on a string rather than a `Path(...)`, a `Path` called
  before `from pathlib import Path`, or `open`/`Path` bound to
  something else in the program; and for a program that imports
  anything, the interpreter the command names is asked, run in the
  command's directory as the command runs it, whether one of those
  modules, or one they import in turn, would load from elsewhere than
  its own library — a `json.py` in that directory is what `import
  json` runs — `edit_guards.shadow_command`, on the host
  `useful_tools._host_shadows` with 10 s to answer,
  `_SHADOW_PROBE_SECONDS`, in the container and over the bridge one
  exec, `DockerManager._shadows`, `ExecBridge._shadows`; an answer
  nobody got counts as a shadow, and a shadow makes the rewrite not
  mechanical); and a `sed -i` (`-i` or `--in-place` exactly: a backup
  suffix, `-i.bak`, is an option the conversion does not take) with at
  most `-e`/`--expression`/`-z`/`--null-data` whose one script
  is `s<d>old<d>new<d>` or `s<d>old<d>new<d>g` with literal text on
  both sides and a one-byte delimiter (`edit_guards.sed_edit`; an
  escaped delimiter that is a BRE metacharacter, `s.a\.b.c.`, is no
  literal text, as sed unescapes it to the metacharacter, and an `-e`
  that ends the command has no script; `g` becomes `replace_all`, and a
  `sed` without `g`, which replaces the first match of every record
  (a line, or a NUL-separated record with `-z`), is converted only
  while no record holds the pattern twice —
  with several matching records the `Edit` gets `replace_all`,
  `MechanicalEdit.edits_for`, which judges the pairs in order, each
  against the text the earlier ones leave, as `Edit` applies them: a
  pair matching nothing is left out, as `str.replace` and sed leave
  the text where `Edit` would refuse the string it did not find, and
  when no pair matches the pairs stand as written, so the `Edit`
  reports the first as not found — unless that first pair, replacing
  one match, is found in the text whitespace aside, when nothing is
  converted, since `Edit` would retry it ignoring runs of blanks and
  trailing whitespace, `edit_matching.tolerant_matches`, and change
  what sed and `str.replace` leave alone; a file holding `\r` the python read would
  translate is not converted either, nor is a rewrite of a name the
  host does not take as written, which runs and fails as the shell
  fails it (`useful_tools._fails_as_written`: a `..` after what is no
  directory, `nosub/../f`, where `Path.resolve` would name `f`; a name
  too long, not encodable, or in a directory the user may not search),
  nor is a `sed -i` on a symlink or
  a hard-linked file, which GNU sed would replace with a new regular
  file where `Edit` writes through the link,
  `useful_tools._sed_replaces_link`, nor one on a file the
  user may not write, which sed replaces all the same where `Edit`,
  writing in place, is refused; the container and the bridge ask
  both questions with one exec, `docker_manager.probe_command(path,
  "link")`, and a probe that cannot answer counts the file as a link,
  so no `Edit` stands for a `sed` nobody checked; nor is a rewrite
  whose program bash does not find, which fails with status 127 and
  leaves the file alone (`useful_tools._host_runs`: a bare name on
  `PATH` — not when `PATH` holds a relative entry, which names another
  directory after the command's `cd` — or an absolute path;
  `probe_command(path, "run")` in the container and over the bridge,
  where a probe that cannot answer counts the program as not
  found)).  The rewrite may be the whole
  command or the first step of `[cd DIR &&] <rewrite> [&& | ; |
  newline] <rest>` on one line with no subshell, substitution or
  backtick, at the top level (not in a subshell, group, pipe or
  background list, not negated by `!`, which inverts a status no
  `Edit` has) and with only the redirections a `Write`/`Edit`
  reproduces (`edit_guards._exact_redirections`: the here-document on
  standard input, standard output to a file through `>`, `>>`, `>|`,
  `>&` with a file operand, `&>` or `&>>`, `2>&1`, `2> /dev/null` and
  `< /dev/null`; an input file, `1>&2`, a standard-error file or a
  descriptor of the command's own leaves the command to run as
  written or to the guards below) (`edit_guards._rewrite_segment`): the rest then runs as a
  second `Bash` call, `cd DIR && <rest>`, with the original call's
  `description`, `timeout_seconds`, `max_output_chars`, `background`
  and `on_timeout` (`useful_tools.followup_call`), only once the
  `Write`/`Edit` went through, and a `cd` bash does not walk to the
  directory it names (`shell_scan.walks`: a `..` is dropped only after
  a directory) is refused up front (`useful_tools.cd_refusal`): one to
  a missing directory (`CD_VERDICT`), as bash would run nothing after
  it, and one bash enters only through a symbolic link
  (`CD_LINK_VERDICT`), where only the shell knows the directory.
  A rewrite the call would poll with `until=` is refused before any
  edit, since `until` re-runs a command as written and the call stands
  for a change made once (`UsefulTools._apply`: `Error: until= re-runs
  a command as written, but this one is applied as <call> instead of
  running: make that call, then poll <rest> with until= in a Bash call
  of its own.`, the `then poll` part only when a rest follows).
  Under an agent whose run has both `Write` and `Edit` to make
  (`UsefulTools.redirect`, set from the run's final toolset in
  `SorcarAgent`) the tool raises the call as a
  `tool_verdict.ToolRedirect` and the agent makes it through its own
  tool path (`KISSAgent._execute_redirected`), so the
  `tool_call_hook`, the guard, the `tool_call`/`tool_result` events and
  the coding gate see the `Write`/`Edit` exactly as if the model had
  issued it (only the original call counts in `tool_calls_made`; the
  redirected call is registered under the original call's interrupt
  token, `tool_interrupt.register_tool_call`, so a Stop on the `Bash`
  panel reaches whichever redirected call is running, and a nested
  call that starts under an already stopped one is not run); a
  profile without one of the two tools (`bash`, `shell`, `review`) or
  a SEA `tools()` that dropped it has the tool apply the `Write`/`Edit`
  itself, follow-up `Bash` included; the
  `Bash` result opens with `Applied as Write(file_path=..., content=...),
  whose result follows:` (`Appended through ...` for a `>>` copy;
  strings cut to 80 characters), then the call's result or its
  refusal, then `Then made Bash(command=...), whose result follows:`
  and the rest's result, or `Bash(...) was not made: the Write did not
  go through.` (`tool_verdict.followup_result`).  On the host the
  file must have been shown (or be a scratch file under `tmp/`, which
  `Write` overwrites unread): a copy into a file not shown is refused
  with the `Read` to make before the same command, or the `Write`; a
  mechanical edit of a file not shown is refused with the exact `Edit`
  call and the region around its first `old_string`, after which the
  file counts as read, so the same command or that `Edit` goes through
  next (a scratch file counts as shown: the edit is applied at once);
  a copy into a file that does not exist yet, or into `/dev/*`,
  runs as the shell command it is; a command that is no such rewrite,
  or whose conversion is not exact, falls to the rewrite and
  here-document guards below.  A foreground container or bridged
  `Bash` applies the same conversions (`docker_manager._rewrite_instead`: the
  current text of an appended file and the text a `sed`/python
  conversion depends on are fetched with `read_bytes`, the `cd` is
  checked with `test -d`), with no shown-file rule — the container
  tracks no reads — and a copy creating a new file redirected to
  `Write` too, but with no home of its own to resolve a `~` target
  (`heredoc_copy(..., home=None)`: the container's or the bridge
  environment's home is not this process's), so such a copy is refused
  by the here-document guard asking for the literal path
  (`HEREDOC_UNRESOLVED_VERDICT`) (the bundled coding SEA's trials run on that container
  `Bash`, so its ledger gate sees the edit).  A redirected `>>` copy
  is remembered on the manager as a `docker_manager.PendingAppend`
  (the path, the file's bytes when the redirect was raised, the
  appended body) for the one `Write` it stands for: when that `Write`
  lands with exactly the snapshot plus the body, `DockerTools.Write`
  appends the body through the shell's `>>`
  (`docker_tools._append_commands`) instead of storing the snapshot,
  so what another writer appended meanwhile is kept, as `cat >>`
  would have kept it; the next `DockerTools` `Read`, `Edit` or
  `Write`, or the next `Bash` of the manager (the ones that start
  and control a container background job included), takes or
  drops the pending append (only `run_commands_parallel` leaves it
  alone), so a redirect the
  agent's hook refused never turns a later `Write` into an append.
  `run_commands_parallel`
  (host, container or bridge) and a `Bash(..., machine=)` job apply no
  conversion: there the rewrite is refused as below.  A container or
  bridged `Bash` started as a job bypasses the foreground path:
  `ContainerJobs.start` embeds the command unchanged in a generated
  launch script, and that script, not the command, is what
  `_docker_bash` runs, so the command gets neither conversion nor
  rewrite guard.  `background=True` hands the command to
  `ContainerJobs.start` as written; `on_timeout="background"` first
  applies the sleep rule (`wait_instead_of_sleep`: a `sleep` longer
  than the deadline is refused, a shorter one outside a loop waits on
  the running job instead, and a `sleep` with no job running or
  pacing a loop stays as it is), then starts what is left as a job and
  waits for it through `bash_job`.
- **Refused shell commands.**  `Bash` itself returns the refusal as
  its result (these are tool-side guards, not `tool_call_hook`
  verdicts, so no re-issue reminder follows): a command that would
  delete the work dir or `/` or kill every process
  (`shell_guards.destructive_command_guard`, `REFUSED: the Bash call did NOT run: ...`;
  the `rm -rf` is matched on the work dir itself, by its path or from
  inside it — `rm -rf *`, `./*`, `"$PWD"`, which only a `cd <dir> &&`
  right before it moves elsewhere, `<dir>` a literal path that neither
  is the work dir nor holds it, `shell_guards._deleted_from_inside` —
  on its parent
  through `<work dir>/..` or `/..`, and through a wrapper — `sudo`,
  `env`, `exec`, `nohup`, `nice`, `time`, `command`, `builtin`,
  `xargs`, `timeout` and their options, a wrapper option spelled like
  a guarded command included, `timeout -s kill 5 rm -rf /` — inside a
  `bash -c '...'` (`sh`, `zsh`, `dash`, `ksh` too; the shell's options
  may go on after the `-c` cluster, `bash -c -e '...'`, `bash -co posix
  '...'`, `bash -c -- '...'`) or after a backquote,
  `shell_guards.destructive_pattern`, one prefix per guarded command
  so the match stays linear in the command's length); inside
  a worktree, a command naming the main checkout's path outside its
  `.venv` (`Error: command references the parent-repo path ...`, with
  the command rewritten against the worktree to copy as is) and a
  `uv sync`/`uv add`/`uv remove`, with or without global options
  before the subcommand (`uv --directory . run`, `uv -q sync`;
  `useful_tools.uv_in_worktree`,
  ``REFUSED: `uv run` did NOT run: the worktree '<path>' has no
  environment of its own and shares the main checkout's
  (<checkout>/.venv); ...``: a worktree has no environment
  of its own, so uv would build it one or, through `UV_PROJECT_ENVIRONMENT`
  or a `.venv` link, install the worktree as the main checkout's editable
  project; the refusal names the main checkout's environment binary to
  run with the worktree's `src/` and `projects/` first on `PYTHONPATH`).
  A `uv` the tool cannot read as one (`bash -c "uv run ..."`, a
  script, a Makefile) is stopped by uv itself: the worktree's `.venv`,
  and that of every directory with a tracked `pyproject.toml`,
  is a file (`GitWorktreeOps.block_own_environment`, written when the
  worktree is made and kept out of `git add -A` through
  `info/exclude`; where the repository un-ignores the name, a
  directory instead, which holds its own `.gitignore` of `*` and the
  text as a file named `bin`, so git stages nothing of it), which uv
  refuses as a project environment in every form and never replaces,
  so a worktree's code
  runs on one Python (on 2026-11-11 a full test run and 43 helpers ran
  on Python 3.14.7 beside the project's 3.13.5).  Where the main
  checkout has no `.venv` there is nothing to share: no file is
  written and `uv` is left alone in every form, so it builds the
  worktree's own environment.
  A `uv run <program> ...` there is not refused but rewritten to that
  binary and run, every `uv run` of the command, the result followed
  by `Note: the worktree has no environment of its own ... run it so
  yourself next time: <rewritten command>` (`useful_tools.UV_REWRITTEN_NOTE`,
  after the result and the batching hint, as every note a `Bash`
  result carries now is, so a reader of the result's head — `Error
  (exit code N):`, `Error: Command timed out`, a `bash_job` report —
  parses what it looks for; also in `run_commands_parallel`, after
  the command's output; not for a `Bash(..., machine=)` job, whose
  files this host cannot see; `uv run -m MODULE` / `--module` becomes
  the environment's `python -m MODULE`, a wrapper before `uv`, `timeout
  5 uv run x`, gets the `PYTHONPATH=` assignment through `env` so it
  stays an assignment there, and the paths are shell-quoted), unless
  the rewrite is ambiguous: a global option with a value, `--directory
  x`, `--with x`, before or after `run` would be dropped, a flag after
  which the next word is no program of the environment (`--script`,
  `-s`, `--gui-script`, `--help`, `--version`), or the
  environment lacks the program, which is refused as above; a shell rewrite of a
  file that the bullet above did not apply as a `Write`/`Edit`: a
  `sed` editing in place (`-i`, `-i.bak`, `-Ei`, `--in-place`) with
  `\n` in one of its words and a `python -` or `python3 -` heredoc that reads a
  file, `.replace(` / `re.sub(` on it and writes it back
  (`edit_guards.rewrite_guard`, ``REFUSED: the `sed -i` call did NOT
  run: ...`` / `REFUSED: the python heredoc did NOT run: ...`, naming
  the `Edit` call to make instead, with the file and the
  `old_string`/`new_string` pairs filled in when the command names
  them literally, each string cut to `edit_guards.PREVIEW_CHARS`, 80,
  `edit_guards.edit_call`; a `python3 - <<` that is only text of a
  here-document a `cat`/`tee` copies into named files and nowhere
  else — `cat > tmp/x.sh <<'EOF'`, `cat <<'EOF' | tee f > /dev/null`,
  a script being written, `_copies_to_files` — is data and not
  refused, while one whose copy also reaches a pipe, a shell, a
  `"$OUT"` or the command's own standard output keeps the refusal); a `cat` or `tee` here-document
  whose text reaches an existing file — through `>`, `>>`, a `tee`
  operand or `tee -a`, directly, down a pipe of `cat`/`tee` copies or
  into a group that copies it (`| { cat > f; }`)
  — (`edit_guards.heredoc_guard`, `REFUSED: the command did not run:
  it writes the existing file ...`, pointing at `Edit`/`Write`), and
  one whose target only the shell can name — an expansion such as
  `"$OUT"` or `~/x`, or a relative path after a `cd` the guard cannot
  resolve — since it may be an existing file (`HEREDOC_UNRESOLVED_VERDICT`);
  a here-document that creates a new file at a literal path, one aimed
  at `/dev/*`, and one fed to an interpreter (`python3 - <<EOF >
  out.json` writes the program's output, not the text) run.  On the
  host the three guards exempt scratch
  files (`useful_tools._is_scratch_path`: under a `tmp` directory
  inside the work dir, anywhere under `$KISS_HOME/cron/work` or in
  a sub-agent's private folder, `<system temp dir>/kiss-sub-…`; a
  checkout that itself sits under `/tmp` is not scratch), as `Write`
  overwrites one unread: a multi-line `sed -i` whose file operands all
  resolve to scratch or missing files, a python heredoc every written
  path of which (`edit_guards.python_written_paths`: `open(p, "w")`,
  `p.write_text(...)`, a literal or a name bound exactly once in the
  program; a path built at run time keeps the refusal, as does an
  `os.chdir`/`os.fchdir` in the program, after which a relative path
  names a file the guard cannot place) resolves so,
  and a here-document into a scratch file run as written; the
  `guarded(path)` predicate the guards take
  (`useful_tools._host_guarded`) says which absolute paths are still
  refused.  A container or bridged `Bash` passes its plain `exists`
  (`docker_manager._refuse_or_lift`: no scratch exemption there, and
  the rewrite guard reads the command alone), as does the host
  `rewrite_guard` of a `Bash(..., machine=)` job, whose files this
  host cannot see.  The
  command is read by `shell_scan.simple_commands`, a quote-aware
  scanner that follows `cd`s, pipes, `&&`/`||`/`;` chains, groups
  and arithmetic commands (`((...))`, whose only commands are its
  command substitutions) the way bash does (where bash versions
  disagree on a `${...}` operand, as bash 3.2, the macOS system
  bash, reads it, so a substitution only that bash would run is
  still scanned; groups or expansions nested past `MAX_NESTING` (50)
  levels end the scan: for a guard with `TooDeepError`, so the `Bash`
  / `bash_job` / `run_commands_parallel` command is refused with
  `TOO_DEEP_VERDICT`, "REFUSED: the command did not run: it nests
  groups or expansions more than 50 levels deep, past what the shell
  guards read. Split it into simpler commands." — in a fan-out
  that one command's `exit -1` section, not the whole call's; from
  `Bash` the error leaves the tool, so the result reads `Failed to
  call Bash with …: REFUSED: …` — since a guard
  must not let a write it cannot read through; for the gate's
  `optimistic` scan with the commands found so far), drops wrapper prefixes
  with their option values (`sudo -u root`, `doas`, `env`, `nice`, `timeout
  -s KILL 5`, `stdbuf`, `xargs -I {}`) and reads a Windows drive path (`C:/x`) as
  the absolute path Git Bash spells `/c/x`, so a redirect inside a
  quoted string or a heredoc body is not a write (the body a
  here-document hands on is kept on the command reading it,
  `SimpleCommand.heredoc_body`, as written after a quoted delimiter,
  after an unquoted one only when it holds no `$` or backtick:
  `handed_text`; in the body after an unquoted delimiter a line ending
  in a backslash is joined with the next before the delimiter is
  looked for, as the shell does; and the delimiter itself gets quote
  removal only, so a `$(...)` or backtick in it is text of the
  delimiter, not a command the scanner reads).  A redirection on a
  descriptor of two or more digits or on bash's `{name}` (`11>file`,
  `{fd}>file`) is read as both shells would: the redirection counts
  as the operator's default descriptor's, so the file is an output of
  the command (bash truncates it for the descriptor it opens, sh
  writes the standard output there), and the `11` or `{fd}` is kept as
  a word of the command, the argument sh would pass;
  `SimpleCommand.redirections` keeps the descriptor as written.  `pushd DIR` is followed as a `cd`; `pushd -n DIR`
  stays where it is, and a bare `pushd` or `pushd +N` goes to a
  directory only the shell knows.  A `cd` that may not run — after
  `||`, or in an `if`, a loop or a `case` — leaves the directory
  unknown, and so does one after `&&` once its and-list ends any other
  way (the rest of the list runs only when the `cd` ran; the gate's
  `optimistic` scan keeps it, as for a variable, below), and so does
  an `eval` or `source`, whose text may change directory.  The literal
  text an `eval 'text'` or a child shell's `bash -c 'text'` (`sh`,
  `dash`, `zsh`, `ksh`; the shell's arguments read as bash reads them,
  `shell_scan.shell_script_index`: its long options first, `--norc`,
  `--rcfile file`, then letter clusters, `-lc`, `-ec`, `-co posix`,
  an `o` or `O` taking the next word, until `--` or `-` ends them or
  an operand starts, which is the text only when a `c` was among the
  letters — `bash -a file -c '...'` runs the file; `shell_scan.command_text`) runs
  is scanned for its commands, in the command's directory and with
  its redirections (`_Scanner.run_text`), so a write
  inside it is a write; `eval`'s commands know the variables this
  shell knows, a child shell's only the ones `export`ed for sure (an
  `export` after `||` or in an `if` exports nothing here, one after
  `&&` holds for the rest of its and-list, `unset` drops the export
  with the value) and the assignments before the command (`T=x bash
  -c`, `env T=x bash -c`; `_Scanner.environment`), read left to
  right with every variable known so far dropped at a wrapper that
  may reset the environment (any `env`, plain or `-i`/`-u T`, `sudo`,
  `doas`), so only the assignments after the last such wrapper reach
  the child; text the shell expands first is only the
  shell's to read.  A function definition (`f() { ...; }`, `function
  f ...`) runs nothing, and its body is scanned as commands that may
  not run; a comment after `&&`, `||` or `|` is skipped as bash skips
  it; a bare `exec > file` makes that file the output of every later
  command that does not redirect its own — the file as the `exec`
  opened it (a later `cd` does not move it), in the shell that ran it
  (a pipeline stage's or a backgrounded `exec` ends with its subshell),
  and after an `exec` that may not have run (after `&&` or `||`, in a
  compound command or a function body) the earlier output stays
  possible too.  A brace expansion
  (`{a,b}`, `{1..3}`), a `~` after the `=` or a `:` of an assignment,
  an unquoted variable once the command set `IFS`, and the operands
  of a program a wrapper runs in another directory (`env -C dir`,
  `sudo -D dir`; `_PREFIX_CHDIR_OPTIONS`) are, like a glob, not
  literal: the shell alone resolves them.  Each simple command records
  whether it follows `||` (`conditional`), follows `&&` (`after_and`),
  ends in `&` (`background`) or is negated by `!` (`negated`), for
  readers that need the command's exit status to mean something
  (the coding gate's verifier reuse).  The scanner also
  follows a variable the command itself assigns a literal value
  (`T=tmp/x; cat > $T/f`, `export T=...`, `${T}`; `_Scanner.variables`;
  `$PWD` is the directory it knows the command runs in, unless the
  command assigned it)
  into the words after it, in the same scope, which is why such a
  target resolves to a literal path instead of being refused as one
  only the shell knows — not after `||`, after
  `&&` only to the rest of the same and-list (`cd d && T=tmp/x && cat
  > $T/f`), forgotten when the list ends any other way, as is one
  assigned in a brace group after `&&`, except under the gate's
  `optimistic` scan, which takes every list to have run through
  (`simple_commands(..., optimistic=True)`: a `cd` or an assignment
  after `&&` or `||` holds for the rest of the command, since the gate
  reads what the agent changed rather than keeping a write from a path
  that may not have been set);
  not in a compound command that may not run, not `T+=`, not an
  assignment before a special builtin or before a wrapper with no
  command after it (`T=x export Y`, `T=x exec >f`: sh keeps the
  variable, bash does not, so it is unknown; `_SPECIAL_BUILTINS`); a subshell's,
  a pipeline stage's and a backgrounded list's assignments end with
  them (a brace group's included); after a brace group that may not
  have run, after `read` (`read T`, `read -aT`), `unset`, `declare`,
  `readonly`, `let`, `printf -v`, a loop over the variable, an
  arithmetic command or expansion assigning it (`((T=1))`, `$((i++))`)
  or an array assignment (`T[0]=x`, `T=(x y)`, `T+=(z)`; the
  substitutions in an array value run, so their commands are scanned)
  the variable is unknown (as are `REPLY` after any
  `read` or `select` and `MAPFILE` after any `mapfile`, the variables
  those fill when no operand names one, and `T` after an assigning
  expansion `${T=x}` / `${T:=x}`), and after `eval`, `source`, a
  `${!T=x}` or a name reference (`declare -n`) every variable is —
  as is every variable bash sets itself (`RANDOM`, `LINENO`, `_`,
  `PIPESTATUS`, ...; `_SHELL_SET`), the one a `coproc` names, whose
  command is read as a pipeline stage, and the one a `printf -v T` or
  `wait -p T` assigns; and after an `alias`, an `enable`, a `trap` on
  anything but the shell's end or an outside signal (`EXIT`, `HUP`,
  `INT`, `QUIT`, `TERM`; a `DEBUG`, `ERR` or `RETURN` trap runs
  between the commands that follow it) or a `set`/`shopt` of
  `lastpipe`, `cdable_vars` or `-k`/`keyword`, bash no longer
  runs the text as written, so no variable and no directory is known
  from there on, `_Scanner.unfollowed` — a child shell started with
  such an option (`bash -O lastpipe`, `-k`), or with `BASH_ENV`,
  `BASHOPTS` or `SHELLOPTS` among the variables it gets, is read the
  same way (`_UNFOLLOWED`, `_SHELL_STARTUP`).
  And a foreground
  `sleep N` with N over 30 (`FOREGROUND_SLEEP_MAX_SECONDS`; N is the
  sum of the literal operands with GNU sleep's `s`/`m`/`h`/`d`
  suffixes, `inf`/`infinity` is forever, and an operand only the
  shell knows, `"$N"`, counts as nothing; `useful_tools.long_sleep`
  finds the first such sleep) that is also longer than the call's
  `timeout_seconds` (`useful_tools.wait_instead_of_sleep`,
  `sleep_refusal`: the wait belongs in `bash_job`, and the refusal
  names the running background job to wait on when there is one).  A
  long sleep within the deadline is spent on a background job of the
  task instead, when one is running: `Bash` waits on the job started
  last for the sleep's length, then runs the command with that sleep
  replaced by `true` (`sleep 60; tail log` waits, then runs `tail
  log`), and the result ends with `Waited on job <id> for up to N s
  instead of sleeping (<status>).` (after the output, like every
  note of a `Bash` result, host or container); with no running job, when the
  sleep paces a `for`/`while`/`until`/`select` loop (taking it out
  would spin the loop) or when the text search (`_without_sleep`)
  does not find the sleep, the command runs as written — with no
  running job the result then ends with `[note: the sleep of N s ran
  as written: no background job of this task is running. Wait in one
  call instead: Bash(command, until="<regex>", poll_seconds=N,
  timeout_seconds=M) ...]` (`useful_tools.sleep_alternative`), the
  polling call described under Background jobs.  The host
  `run_commands_parallel` refuses a sleep over 30 s in any of its
  commands outright (`sleep_guard`), since each is a foreground wait
  for the caller.  A `sleep N &` counts, since the child keeps the
  call's output pipe and the call blocks until it ends anyway — only
  a `sleep` inside a heredoc body the shell never runs (a quoted tag,
  `<<'EOF'`, with no `$(...)` or backtick in the body) is let
  through.  The
  destructive-command, rewrite and here-document guards also run on a
  container `Bash` (`docker_manager._refuse_or_lift`, which asks the
  container whether the target exists with `test -e`,
  `DockerManager._path_test`) and on an exec bridge
  (`ExecBridge._path_test` runs the same `test -e` through the bridge;
  a probe that times out or is stopped fails closed either way: it
  counts as existing, so the copy is refused rather than run on a
  target nobody checked, and as no directory, so a `cd` nobody checked
  authorizes no rewrite after it); the
  host tool checks with `os.path.lexists` against the directory the
  command runs in (`UsefulTools._resolve_cwd`: the work dir, or the
  main checkout when the task's worktree vanished mid-task; resolved
  before the guards run and that same value used to launch the
  command, so the guards and the launch see the same directory), and
  a `Bash(..., machine="<name>")` job, whose
  files this host cannot see, skips the here-document guard and the
  rewrite-to-`Write`/`Edit` step.  The container and bridged `Bash`
  apply the same sleep rule (`wait_instead_of_sleep` with
  `ContainerJobs.last_running` and `ContainerJobs.wait`, each one
  script run in the container); their `run_commands_parallel` does
  not (`docker_manager._refuse_or_lift` calls no `sleep_guard`), and
  the bundled coding SEA has no sleep rule of its own: its trials
  rely on the container `Bash`'s.  A
  package install or build (`apt`, `pip`, `uv`, `npm`, `cargo`,
  `make`, ...) gets at least `INSTALL_TIMEOUT_SECONDS` (900 s)
  whatever `timeout_seconds` asked, on every path.  A
  `run_commands_parallel` whose `commands` is not a JSON list of
  strings, or whose `max_workers` is negative, returns `Error: ...`
  without running anything (`useful_tools.run_commands_pool`, the
  one engine behind the host, container and bridge variants).  The
  host `Bash` runs its command with bash itself wherever one is found
  (`useful_tools._BASH`, `_popen_kwargs`: `shell=True` would run
  `/bin/sh`, dash on Debian, which expands no braces and has no
  arrays); a host without bash gets the platform shell, Windows
  without Git Bash PowerShell.
- **Polling in one call.**  `Bash(command, until="<regex>",
  poll_seconds=5, timeout_seconds=N)` re-runs a cheap read-only
  command (`grep DONE build.log; tail -3 build.log`) every
  `poll_seconds` (at least 1) until the regular expression matches its
  combined output or the deadline passes, in this one call, so a wait
  for a file, a port or a log line costs one turn however long it
  takes (`UsefulTools._poll_until`; each run is a foreground run, so
  its output counts as the view of the file it shows).  The result is
  the last run's, ending in `[until '<pattern>' matched|did not match
  after N run(s) and S s]`, the latter followed by `: the call's
  deadline (timeout_seconds) passed` when it was, or `[until ...
  was not searched after ...: the output is too long to keep whole (N
  characters dropped); poll with a command that prints less (grep,
  tail)]` when a run's output outgrew the capture (a match across its
  gap would be wrong); the search runs in the same matcher child
  process as `bash_job`'s wait, ended at the deadline or a Stop, and
  its exit is seen as it happens (`useful_tools._run_log_matcher`
  waits on the child in 0.2 s slices, checking the deadline and the
  Stop between them), so no poll waits out a tick for its verdict.
  The poll also ends, with `; the poll ended early: every sub-agent
  job of this task has finished and the result of <ids> has not been
  read yet: agent_job('', 'wait_any') returns it` after `did not
  match`, once every sub-agent job of the task has finished with a
  result unread (`UsefulTools.idle_children`, as for a `bash_job`
  wait: a poll for what a child was to produce can then only time
  out; the 2026-11-08 sweep spent 2.4 hours in three such polls, each
  counting a folder of review reports after every reviewer had
  finished).
  `until` with `background=True` is refused (`Error: until= polls a
  foreground command; with background=True follow the job's log with
  bash_job(id, "wait", until=<regex>) instead.`), as is a pattern
  that does not compile (`Error: until is not a valid regular
  expression: ...`), both before anything runs.  The container and
  bridged `Bash` take the same two arguments (`docker_manager._run_until`,
  `until_refusal`, which also refuses a `poll_seconds` that is no
  number; the pattern is searched with `^`/`$` matching at line ends,
  as `grep` does), their verdict line `[until '<pattern>': matched on
  run N after S s]` or `[until '<pattern>': not matched after N runs
  and S s of the T s timeout_seconds]`; the container and bridged
  `Bash` of `SorcarAgent` refuses `until` with `background=True`
  itself, with its own line (`Error: until cannot be combined with
  background=True; start the job and wait on its log with
  bash_job(job_id, 'wait', until=...).`), and forwards both arguments
  otherwise.
- **Background jobs.**  `Bash(command, background=True)` starts the
  command detached (`nohup ... > tmp/bash_jobs/<id>.log 2>&1 &`; a
  sub-agent's log is `bash_jobs/<id>.log` of its private folder,
  named by its absolute path, `useful_tools._tools_dir`) and
  returns a job id; `Bash(..., background=True, machine="<name>")`
  runs it over `ssh` on a machine registered with `/machine add` (an
  SSH one, not a URL one), its output streaming into the same local
  log; `Bash(..., on_timeout="background")` starts the command as a
  job up front, waits `timeout_seconds` for it and, instead of
  killing it at the deadline, leaves it running under its job id
  (`machine=` without `background=True` is an error, `Error: machine=
  applies to background=True only.`; a foreground host, container or
  bridged call with an `on_timeout` other than `"kill"` /
  `"background"` is an error too, `Error: on_timeout must be "kill"
  or "background", got ...`, while a `background=True` call ignores
  the value: the job starts before the check; and an `until` poll
  wins over `on_timeout="background"` on every path).  A plain foreground
  host call whose `timeout_seconds` is over the daemon's
  `bash_detach_seconds` (120, `KISS_BASH_DETACH_SECONDS`; 0 turns this
  off) runs as a job from the start and is followed for that grace
  (`UsefulTools._run_then_detach`): one that finishes within it reads
  like a foreground result (output, exit code, truncation, spill, the
  view it counts as) and leaves no job or log behind; one still
  running is left running under its job id, and the result says so
  (`Still running after 120s; left running as background job <id>
  (pid N). Log: ...`), names the `bash_job(<id>, action="wait",
  until=<regex>, timeout_seconds=<the rest of the deadline>)` to
  make and ends with the output so far.  A tool-panel Stop during the
  grace kills the command as it kills a foreground shell; during an
  `on_timeout="background"` wait it leaves the job running, as
  `bash_job` does.  `bash_job(job_id,
  action="wait", timeout_seconds=N, until="<regex>")` returns as
  soon as the log matches the pattern (scanned once a second,
  `_log_matches`: a log of at most `_LOG_SCAN_BYTES`, 8,000,000, is
  searched whole, a longer one only in its last 8,000,000 bytes from
  the first complete line in them; the search runs in a matcher child
  process, `_start_log_matcher`, that exits 0 on a match and is killed
  when the wait ends at its deadline or a Stop, so a pattern that
  backtracks for minutes cannot hold the task thread past either), so a
  long run is one turn to start and one to collect; `"tail"` and
  `"kill"` are the other actions.  The wait also ends early once
  every sub-agent job of the task has finished with a result its
  `agent_job` tool never returned (`UsefulTools.idle_children`, set by
  the agent to `agent_dispatch.unread_results_note`: what the wait is
  for can then only be something a child was to leave behind, and the
  child's result says what it did instead); the report then ends with
  `[wait ended early: every sub-agent job of this task has finished
  and the result of <ids> has not been read yet: agent_job('',
  'wait_any') returns it]`, and a sleep that became such a wait
  (`wait_instead_of_sleep`) carries the same note after the job's
  status.  Each unread result ends one wait: the note names a job
  once (`AgentJob.nudged`), so a later wait the agent starts without
  reading it runs its course.  A host job is owned by the daemon
  thread that spawned it (`useful_tools._spawn_owned`, `_own_child`,
  `_watch_job`), the only thread that signals or reaps the shell: on
  the task's Stop, or when `bash_job(action="kill")` sets the job's
  `kill` event, it kills the job's process group
  (`processes.kill_process_tree`), and it reaps the shell only once
  the watch is over, so every signal names the group of an unreaped
  shell and never a stranger's; the watch goes on after the shell
  exited while its group still has members (`_job_group_alive` lists
  `/proc` for another process whose group id is the exited shell's
  pid, or asks `ps` where there is no `/proc`,
  `processes._other_group_members`; the process table is read one poll
  after the exit, then after pauses doubling up to a minute), so the
  Stop reaches a `server &` the shell left behind.  A Docker or bridged run has the
  same `background=True`, `on_timeout="background"` and `bash_job`
  (`container_jobs.ContainerJobs`, one instance per agent, so a later
  prompt in the chat can still wait on the job), done entirely with
  shell commands in the container: the job runs under `setsid nohup`
  with the exec token removed from its environment (so the deadline
  sweep of a timed-out foreground command leaves it alone), and its
  state is `/tmp/kiss-jobs/<id>.log` (a sub-agent's job keeps its
  files in `kiss-jobs/` of the sub-agent's private folder instead,
  `ContainerJobs._new_job_dir`, so they are deleted with the folder
  when the sub-agent ends), `<id>.pid` (the session leader,
  whose process group is the job), `<id>.exit` (written when the
  command ends; the shell opens it before it writes the exit code, so
  only a non-empty marker is a finished job; a started job whose
  shell is gone — no process, a zombie, or a stranger wearing the pid
  — with no live process left in its group gets `(no exit code:
  stopped from outside bash_job)` written there by the next `wait`,
  `tail` or running-jobs check, `container_jobs.LOST_STATUS`,
  `SETTLE_FN`, so a `pkill -f` of the job's command no longer leaves
  it counted as running for ever; a job with neither an exit code nor
  a pid file — its files cleared from the jobs dir — reports `status:
  unknown (the job files are gone; the process may still run)`,
  `container_jobs.GONE_STATUS`, which nothing waits on or counts as
  running)
  and `<id>.cmd`; the kill scripts signal a pid only while it is still
  the job's (`container_jobs.OWNED_FN`: the command line names the pid
  file, or the process is a zombie of it), so a recycled pid is never
  a stranger's death; a process a foreground command left running
  (`nohup server &`) can be adopted as a job, `ContainerJobs.adopt`,
  with the same files plus `<id>.owner` (the process's start time as
  adopted, field 22 of `/proc/<pid>/stat`, `container_jobs.IDENT_FN`,
  which with the pid identifies the process through any `exec` and
  tells a later process wearing the pid apart) and `<id>.group` when
  the adopted process leads its own process group (so once it has
  ended the group stays the job's and a child it left there is still
  killed) and `<id>.token` (the exec token of the process as adopted,
  which every process the command that left it running started
  carries in its environment: a kill of the job last ends whatever
  still carries that token, `container_jobs.TOKEN_FN`, so a child the
  adopted process started and left behind when it ended by itself,
  which no parent pid and no process group leads to any more, is
  ended too — unless another job has the same token, since of two
  processes one command left running the token does not say whose
  child an orphan was; a job file that cannot be written removes the
  ones written before it, so a half-registered job is no job), the log linked to
  the file its redirection named and
  a detached watcher writing `(exit code unknown: adopted process)` to
  the exit file once the pid no longer belongs to that start time (an
  `exec` is not an exit); a kill of an adopted job, which leads no
  process group as a rule, signals the process and the descendants
  found by their parent pids in `/proc/*/stat` before the TERM and
  again before the KILL, each by pid and start time
  (`container_jobs.TREE_FN`), so a pid reused meanwhile is never
  signalled — the bundled coding SEA adopts every
  process a trial's shell leaves behind this way and lists it as `left
  running as job <id> (<command>, log <path>)`; a
  `wait` is one polling script run in the container (the deadline plus
  `WAIT_SLACK_SECONDS`, 15 s), `until` is a `grep -E` pattern checked
  before the wait (`Error: until is not a valid grep -E regex`; only a
  `wait` reads it, so a `tail` or `kill` given one checks nothing), and a
  job id this registry never saw is looked up by its `.cmd` file, so a
  job started by an earlier session of the task can still be followed.
  `machine=` has no container form.
- **`commit_and_merge`** removes each worktree with its git-ignored
  files: `tmp/` is git-ignored, so the worktree's `tmp/PROGRESS.md`
  (the progress log the system prompt asks for) and the task's
  verifier (`<scratch dir>/verify.sh`) are in no commit, and nothing
  copies them into the main checkout.  A merge the main
  checkout's uncommitted changes block (`git stash push` failed)
  returns the blocking files from `git status --porcelain` and leaves
  the committed worktree branch in place.
- **`checkpoint(name)` / `rollback(name)`** (`checkpoints.py`) are
  the task-local undo of a top-level host run (not offered to a
  sub-agent or a Docker run, see above).  `checkpoint` snapshots every
  non-ignored file under the top level of the git working tree that
  holds the work dir (the same directory `Bash` runs in,
  `_resolve_cwd`; git runs with the repo-scoping variables of the
  daemon's environment, `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`
  and the like, dropped, as every git call of the worktree code drops
  them, `git_worktree._REPO_SCOPED_GIT_ENV`, so a daemon started with
  them set still snapshots and restores that tree and no other) as a
  git tree object, written through a
  private copy of the index (`copy2`, so the copy keeps the real
  index's mtime and git's "racy git" re-read still catches a
  same-size rewrite made in the second the index was written) so
  `HEAD`, the branch, the real index and
  the stash are untouched: untracked files are included (`git add
  --all` on the private index), untracked ignored files are not,
  tracked files stay in even when an ignore pattern matches them,
  submodules and embedded repositories are gitlink entries whose
  contents are not captured (and whose changes a rollback skips, as it
  skips every file under such a path; an embedded repository with no
  commit checked out yet, a fresh `git init`, which fails the whole
  `git add`, is left out of the snapshot, `checkpoints._unborn_repositories`), and
  files flagged `assume-unchanged`/`skip-worktree` are read as they
  are on disk.  Names match `[A-Za-z0-9_-]{1,40}`; `before-rollback`
  is reserved — a `rollback` whose target differs from the current
  tree snapshots the current tree under that name before it changes
  the first file (a rollback refused because something no checkpoint
  holds is in the way keeps the previous one), so a rollback can
  itself be rolled back — and re-using a name replaces
  the earlier snapshot.  The result names the tree (`Saved checkpoint
  'x' of <root> (git tree 1a2b…; git show 1a2b…:<path> prints a file
  as it was)`); the name table lives in the tool object of the task
  (`Checkpoints.saved`, so a new task starts with none), and each
  snapshot is kept from `git gc` by a ref of its own under
  `refs/kiss-checkpoints/<task namespace>/<id>` in the repository's
  common git directory (`Checkpoints._keep`; a tree, not a commit, so
  `git log --all` meets nothing new): the ref is deleted when another
  snapshot takes the name (`_release`; a `before-rollback` ref only
  once the restore is done, since a rollback to `before-rollback`
  restores from the snapshot it replaces), and `Checkpoints.close()`
  deletes every ref of the task's namespace at the run's end
  (`UsefulTools.close`, called last of the run's cleanup from
  `SorcarAgent._release_run_hooks` with every hook already dropped, so
  a Stop landing in its git commands keeps no ref; a ref git cannot
  delete, its repository gone, stays behind without an error).
  `rollback` rewrites the files that differ
  (working tree only, the branch and the index are left alone; git
  writes them into a `kiss-rollback-…` directory of the git directory
  first and one `rename` moves each over its old file, so the path
  holds the old file or the new at any moment, a Stop included, and a
  directory put there since the plan is not replaced; a file that
  cannot be moved so — on native Windows, or where the git directory
  takes no new directory or is on another filesystem — has its old
  file deleted and is written by git where nothing is,
  `checkpoints._replace`),
  deletes the non-ignored files added since along with the directories
  that leaves empty, and lists both with counts, the first 10 paths
  each (`Rolled <root> back to checkpoint 'x': rewrote 2 files (a.py,
  b.py); deleted 1 file (c.py). The replaced state is checkpoint
  'before-rollback'`); it refuses an unknown name (listing the known
  ones), a
  snapshot of another working tree (one taken before `create_worktree`
  moved the work dir, for example), a no-op (`Nothing to roll back`)
  and an obstacle — an ignored file or the work dir itself standing
  where a restored file must go — in which case nothing is changed.
  The agent's guard (`SorcarAgent._guard_rollback`) refuses a
  `rollback` while the work dir is in the user's own repository
  checkout rather than a worktree, since it would also undo what the
  user changed there meanwhile (`Error: rollback refused: ... Call
  create_worktree() first`; a repository whose worktree setup failed
  is edited in place and the rollback passes).  Every git command of
  a snapshot or restore is watched in 0.2 s slices, so the task's Stop
  or the tool panel's Stop ends it at once instead of after
  `GIT_TIMEOUT_SECONDS`: the task's Stop raises a `CheckpointError`
  naming the interrupted git step (`CheckpointError: git add was
  stopped`, `git write-tree was stopped`, ...), the tool panel's Stop
  terminates the git command and re-raises `ToolCallInterrupted`.
  Neither tool is a SEA setting: the coding SEA's container trials have
  their own `checkpoint`/`rollback` pair over the container's tree.

With `"tool_profile": "none"` the agent's **only** tools are `finish`
and the tools `tools([])` returned; the built-in toolset is not
built, so `use_web_tools` has nothing to act on,
and a `run_agent` caller's extra tools are not merged in either.
This is useful for building focused, restricted agents (`/ask`).  A
SEA that wants the built-in toolset plus its own tools leaves
`tool_profile` alone (or picks a narrower profile such as `"bash"`,
as `/remember` does).

When restricting tools, the full default system prompt (`SYSTEM.md`)
assumes the full toolset (its workflow rules name `Read`, `Edit`,
`Bash` and the browser tools).  Return a `system_prompt()` that
matches the tools you provide:

```python
class WeatherSea(WorkerSea):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"tool_profile": "none"}

    def tools(self, tools: list[Any]) -> list[Any]:
        return [get_weather]  # the whole tool set: get_weather + finish

    def system_prompt(self, system_prompt: str) -> str:
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
- `sea_path` is neither `None`/`""` nor a string
- The SEA path is not a `.py` file
- The SEA file does not exist

**Daemon-side errors** (the task starts, then fails with
`result.success == False` and the diagnostic in `result.text`, which
prefixes the message below with `Task failed: SeaError: `):

| Condition | `SeaError` message |
|-----------|---------------|
| File deleted between client validation and daemon import | `SEA '...' is not an existing Python (.py) file` |
| File raises at import time | `SEA '...' failed to import: ...` (a file that is not parsable Python — a syntax error, undecodable bytes or a NUL byte, nesting too deep for the parser; `sea_settings.PARSE_ERRORS` — is reported the same way, and `sea lint` reports it as a `broken` finding, `cannot parse: ...`) |
| The task's Stop, or a per-tool Stop (`ToolCallInterrupted`), lands while the file executes, the class is instantiated or a method runs | no `SeaError`: the Stop is re-raised unchanged (`sea_settings.is_stop`), so the run ends as stopped, not as a broken SEA |
| No or several `BaseSea` subclasses defined in the file | `SEA '...' must define exactly one subclass of BaseSea (kiss.agents.seas.base.base_sea); found none` / `...; found OneSea, TwoSea` |
| The class cannot be instantiated without arguments | `SEA '...': MySea() raised: ...` |
| `settings()` returns something other than a dict | `settings() of SEA '...' must return a dict, got ...` |
| unknown key | `SEA '...': settings() has an unknown key 'x'; known keys: work_dir, model, chat_id, ...` |
| renamed key | `SEA '...': settings() key 'classify_tasks' was renamed to 'auto_classify'; run `uv run sea lint --fix` to rewrite the script` |
| removed key (`kind`, `preset`, `channel`) | `SEA '...': settings() key 'kind' was removed: what a SEA is became its base class: derive from `WorkerSea` (a tool-bound run on the caller's tree) or `ChannelSea` (an external-service agent) instead of `BaseSea` ...; run `uv run sea lint --fix` to rewrite the script` |
| removed key (`allow_fan_out`, `is_parallel`) | `SEA '...': settings() key 'allow_fan_out' was removed: `run_parallel` is N `run_agent` calls, so there is nothing to allow or forbid separately; `tool_profile` chooses the toolset` |
| removed key (`inherit`, `add_to_prompt` / `append_to_prompt`, `add_to_system_prompt` / `append_to_system_prompt`, `append_basic_tools`, `extends`) | `SEA '...': settings() key 'inherit' was removed: a channel never inherits from the calling task and every other SEA always does; the caller's `inherit` option opts out of inheriting`, `... key 'add_to_prompt' was removed: a SEA shapes the task text in its `prompt(task)` method; `add_to_prompt` is a `run_agent` option, not a setting`, ... (`sea_settings.REMOVED_SETTINGS` holds every entry and its explanation) |
| wrong-typed value (`bool` for a number, `int` for a `str`, ...) | `SEA '...': settings()['max_budget'] must be int or float, got bool (in MySea.settings)` |
| wrong-typed value or unknown key written by one class of the chain, even when a later class overwrites it | `SEA '...': settings()['max_budget'] must be int or float, got str (in MySea.settings)` (`sea_commands.declared_settings` checks each class's return with `sea_settings.resolve_settings` and is the one place these key and value errors are raised, so every row of this table from `unknown key` to `locked` ends in ` (in <Class>.settings)`, naming the method that wrote the entry; `base_settings` only folds the channel locks on top of the checked record) |
| non-finite `max_budget` / `timeout` | `SEA '...': settings()['max_budget'] must return a finite number or None` |
| `timeout` of zero or less | `SEA '...': settings()['timeout'] must be a positive number of seconds, got 0` |
| `locked` naming something other than a run setting (`locked`, `hidden`, an unknown name) | `SEA '...': settings()['locked'] may only name settings keys (work_dir, model, chat_id, ...); got ['hidden']` |
| `description()` of a `/<name>` command returns an empty or blank string (a non-string is the `must return a string` row above) | `SEA '...': description() must return a non-empty string` (raised by `/<name> help`, `sea_commands.sea_description`; `/<name> check` reports it as its text) |
| `prompt()` returns an empty string | `prompt() of SEA '...' must return a non-empty string` |
| a contract method is not a plain Python function: declared `@staticmethod`, or the name bound to a class, a builtin (`tools = len`), a `functools.partial` or a non-callable (`description = "..."`) | `description of SEA '...' must be a method, got staticmethod` (`got builtin_function_or_method`, `got partial`, `got str`, ...; `sea_commands._bound` checks `inspect.isfunction` before the attribute is called, for `description`, `settings` and the hooks alike) |
| a method `X` raises | `X() of SEA '...' raised: ...` |
| a method returns the wrong type | `system_prompt() of SEA '...' must return a string, got ...`, `tools() of SEA '...' must return a list of tool callables (not a file path), got ...`, `llm_call_hook() of SEA '...' must return a list, got ...`, `tool_call_hook() of SEA '...' must return a Verdict (ALLOW, or refuse(text), from kiss.agents.seas.base.base_sea), got ...` |
| a value whose own methods raise (e.g. a `str` subclass with a raising `__str__`) | `prompt() of SEA '...' returned a broken value: ...`, `SEA '...': settings()['max_budget'] returned a broken value: ...` |
| a `sea_class(spec)` base that does not exist | `sea_class('x'): not a registered SEA command; known commands: ...`, `sea_class('x.py'): not an existing Python (.py) file` (raised while the file executes, so reported as an import failure) |

Overrides are **atomic**: if loading the class, `settings()` or
`prompt()` fails, the command keeps all its original values (no
partial overrides).  `system_prompt()`, `tools()` and the three call
hooks run later, inside the agent (once at start, or per call): a
failure there raises the same `SeaError` message, which ends the
task as a failed result (`success == False`) after the overrides
have been applied.  The wrapper around the message depends on the
stage: the setup stage (`task_runner`) reports `Task failed:
SeaError: ...`; a `system_prompt()` or `tools()` failure at agent
start is reported as `Task failed: ...` with the bare message; and a
per-call hook failure is an ordinary step error to the executor
(`kiss_agent`): the step is retried unless the message reads as
non-retryable (an API-key, `unauthorized`, `permission denied` or
`is not available` phrase, which fails at once or switches to a
fallback model), and three consecutive failures
(`kiss_agent.MAX_CONSECUTIVE_ERRORS`) end the run with
the failed result `KISSError: KISS Error: Agent ... failed with 3
consecutive errors. Last error: llm_call_hook() of SEA '...' raised:
...` (no `Task failed:` prefix: the agent itself reports it).  A
stalled stream — the watchdog aborted a response that sent nothing
for `stream_stall_timeout` seconds (`stream_abort.stall_error`,
`is_stall_error`; the Anthropic adapter's message is prefixed with
`Anthropic model <name>: ` so the log line names the model asked) —
is a transport hiccup, not a model failure: it
is retried on its own counter, and only twelve of them
(`MAX_CONSECUTIVE_STALLS`) end the run (`... failed with 12
consecutive stalled streams`).  Both counters are reset by a
step that completes (a switch to the fallback model resets the error
count only), so a stall between two ordinary errors does not reset
the error count, nor an ordinary error the stall count.  A Chat
Completions reply the provider cut short is an error
rather than a parsed result: a `finish_reason` of `length` (the
output-token budget; the error advises raising `max_tokens` or
lowering `reasoning_effort`), `content_filter` or `error` (a gateway
such as OpenRouter reporting a provider that failed mid-generation,
with the upstream error quoted when the choice or the response
carries one) raises `Chat Completions response was
truncated (finish_reason=...)`, as the Responses transport refuses
a `status` of `incomplete` or `failed`, since a tool call cut
mid-JSON would otherwise run with no arguments; the cut-short reply
was still billed.  None of the three is retried as a step error
(the consecutive-error count is for exceptions that are no
`KISSError`); the error's type decides what the loop does with it
(`openai_compatible_model._terminal_error`): `length` is an
`OutputCapError` and `content_filter` a `ContentFilterError`, which
take the failover to another model family described below (an
output cap after its one retry at low effort), and `error` is a plain `KISSError`, which
ends the session (see [Refusals and session
hand-off](#refusals-and-session-hand-off) for the session that
follows it).  A stream the gateway closed cleanly (EOF or
`[DONE]`) before any choice carried a `finish_reason` is refused too: it raises
`KISSError: Chat Completions stream ended without a finish_reason,
so its text and any tool-call arguments may be incomplete.` as a
Responses stream without its `response.completed` event does,
rather than letting the partial text become the turn; the usage
already seen is kept for the agent's partial-usage drain
(`take_partial_usage_response`).  The other transports refuse the
same cut: an Anthropic stream whose body ends before its
`message_stop` event raises `KISSError: Anthropic stream ended
without a message_stop event, so its text and any tool-call
arguments may be incomplete.`, and a Gemini stream that delivered
chunks but none carrying the candidate's `finishReason` raises
`KISSError: Gemini stream ended without a finish reason, so its text
and any tool calls may be incomplete.` (a Gemini stream that
delivered nothing at all is an empty turn, not an error).  Every
Anthropic request that carries tools asks for fine-grained tool
streaming (the `fine-grained-tool-streaming-2025-05-14` beta header,
added next to `interleaved-thinking-2025-05-14` when thinking is on),
so a large tool argument (a long `finish` summary, a `Write` of a big
file) streams as it is written instead of being buffered until the
model has finished it, which would trip the stall watchdog on every
retry; since the API then sends tool input unvalidated, the raw JSON
accumulated for each `tool_use` block (every one that is not blank) is
parsed strictly once the message ends, and a malformed one is the
retryable step error
`Anthropic model <name> streamed malformed JSON for a tool call's
arguments: ...` (the block a `max_tokens` stop cut off is skipped,
since it is dropped anyway).  A
non-streamed (callback-less) OpenAI-compatible request has no stall
clock, since a reasoning turn may legitimately send nothing for
minutes, but it is still stoppable: the SDK call runs on a worker
thread while the task's stop event is watched, and a Stop shuts the
pending socket and drops the client so the SDK does not re-send the
request (`OpenAICompatibleBase._blocking_request`; a reply that
arrived anyway is billed as a rejected response; the client's one
silent retry, `_MAX_RETRIES`, kept for a transient 429/5xx or
connect failure, also re-sends a request whose headers never came,
so a provider that never answers is reported after two header
timeouts).  A streamed OpenAI-compatible request's opening call is
bounded the same way (`_open_stream`): until the response headers
are complete there is no socket the stream watchdog could shut, so
the call runs under a `RequestAbortWatchdog` that shuts the pool's
sockets down at the thread's Stop or at the request's deadline
(`_request_deadline`: the read timeout for every attempt the SDK may
make plus the backoff between them), retires the client for the next
turn and reports the cut as the Stop, or as the stall the stream
watchdog would have raised, instead of letting a peer that trickles
header bytes hold the turn past any Stop.  The shutdown
(`stream_abort.shutdown_pool_sockets`) reaches every httpcore pool
the client may send through — the default transport's and each
proxy mount's (`HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` or explicit
`mounts`) — and a request still connecting or in its TLS handshake:
the pool of every model adapter's client (the OpenAI-compatible,
Anthropic, Gemini and decisions clients alike) gets an
`AbortableNetworkBackend` when the client is built
(`make_pool_abortable`), which keeps every socket it opens, the one a
TLS upgrade yields included, and once aborted shuts them all down
before httpcore writes the request on any of them, so the provider
never sees the stopped request; a TCP connection not yet up at the
abort — the wrapped backend connects on a thread of its own
(`_Connecting`), since the address lookup takes no timeout and
nothing else would end it — fails its request at the abort and is
closed when it comes up, and the aborted backend connects nothing
(`the request was aborted before its connection was up`), so the
SDK's retries of the cut request are refused, until the watchdog
that aborted it re-arms the backend once its request is over
(`AbortableNetworkBackend.rearm`), so a pool kept after an abort may
carry the next request.  Its
streams read in 0.5 s slices (`_READ_SLICE_SECONDS`), so a read that
was already waiting when the shutdown landed fails within the next
slice instead of waiting out the SDK's read timeout (30 minutes on
the OpenAI clients).  Both steps go through private httpx/httpcore
attributes and are best-effort: a transport without a reachable
`_pool` (a custom or mock transport, the `None` a `NO_PROXY` pattern
maps to) is skipped and a failed backend swap is only logged, in
which case the shutdown reaches the connections already open and
nothing more.  A safety refusal
(`stop_reason="refusal"`, `ModelRefusalError`), a provider content
filter (`ContentFilterError`, a subclass: OpenAI's `incomplete_details.reason
== "content_filter"` or `finish_reason == "content_filter"`; also a
request the provider's classifier refused as a plain HTTP error naming
its content policy, OpenAI's 400 "This content was flagged for
possible cybersecurity risk" or Azure's "content management policy"
(`kiss_agent.is_content_policy_error`), which the loop types as a
`ContentFilterError(stop_reason="content_policy")` instead of counting
it toward the three consecutive-error retries; on Gemini, a blocked
prompt — `promptFeedback.blockReason`, raised as `Gemini blocked the
prompt (block_reason=<reason>); the same request will be blocked
again.` — or a candidate that stopped on one
of the filter `finishReason`s, `gemini_model._FILTER_FINISH_REASONS`:
`SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`,
`IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`, with
that name as its `stop_reason`), an
output-token cap hit before a usable answer (`OutputCapError`:
Anthropic `stop_reason="max_tokens"` or Gemini
`finish_reason="MAX_TOKENS"` with no complete tool call on a
tool-bearing turn, Responses `max_output_tokens`, Chat Completions
`finish_reason == "length"`; the cut-off request is still billed,
and the error, like a `ContentFilterError`, has a `.text` attribute
for the text the response held, which the OpenAI-compatible
transports fill, `_terminal_error`, and the `/gepa` reflect tool
reads as a cut reply; the Anthropic and Gemini adapters, which raise
it only on a tool-bearing turn and hand a tool-less caller the cut
text as the reply, leave it empty)
or two consecutive empty responses (no text, no tool call) fail
over to a model of *another family* (`kiss_agent._try_switch_to_fallback`
with `other_family=True`; `model_families.fallback_of_another_family`:
the catalog's declared fallback when it is of another family, else
the strongest configured model of another family among
`SECOND_OPINION_MODELS`, else that `get_fallback_model` result anyway
as the last resort — an explicit catalog `fallback` or, with none,
the configured OpenRouter twin, same family or not — since a
refusal, a filter or a cap stops a model's OpenRouter twin too), once
per run, through `KISSAgent.switch_model`, the path the `set_model`
tool takes as well: the conversation is carried over
(`model.hand_off_conversation`, helper spend record and limit check
included, and the `llm_call_hook` boundary moved to where it lands,
so the switching step's tool results stay new to the hook), a
Gemini-to-Gemini switch keeps its thought signatures, the tool
schema is rebuilt for the new provider, and a `system_prompt`
warning names the old and the new model, the stop reason, and says
so when the same family was the only choice.  An output cap is first
retried once at reasoning effort `low` (`kiss_agent.RETRY_EFFORT`,
`_retry_at_low_effort`) when the model takes an effort level and is
not already at `low` / `minimal` / `none`.  With no fallback, a
refusal is retried once with `kiss_agent.REFUSAL_RETRY_MSG` appended
as a user message (the task is the user's own authorized work in a
sandbox; do the legitimate parts and say in the summary which part
was declined), and a second refusal ends the run; a non-retryable
provider error (no such model, no credit) still switches to
`get_fallback_model`'s choice whatever its family — the catalog's
explicit `fallback` when there is one, else the OpenRouter twin when
`OPENROUTER_API_KEY` is configured — and with neither the error is
raised.  The `stop_reason` of every reply is in its `llm_call` event
(`kiss_agent.stop_reason_of`: Anthropic's `stop_reason`, a Chat
Completions `finish_reason`, the Responses `status` or
`incomplete_details.reason`, and a Gemini candidate's `finish_reason`
by its wire name, `STOP`, `MAX_TOKENS` — which the `/gepa` reflect
tool reads to recognise a reply cut at the output limit,
`gepa_sea.OUTPUT_LIMIT_STOPS`), so a session lost to a filter or a
cap is visible in the event log.
A model that cannot see (`model_families.model_supports_images`: a
family outside `VISION_FAMILIES`, anthropic / openai / google / xai,
with no `vl`,
`vision`, `omni` or `image` segment in its name, or a model of such a
family whose name carries a `TEXT_ONLY_NAME_MARKS` fragment —
`gpt-oss`, `gemma-2-`, `gpt-audio`) does not get the
images and PDFs a tool returns as raw base64: `vision_bridge.describe_attachments`
has the first configured model of `VISION_MODELS` (`claude-sonnet-5-5`,
`gpt-6-luna`, `gemini-3.5-flash-lite`, then the OpenRouter twins of
the first and the third)
describe each image or PDF (`vision_bridge.describable`; an audio or
video clip stays an attachment, since a model that cannot see may
still hear) in one non-agentic call, the description
replaces the attachment in the tool message (a missing vision model
or a failed call becomes a line saying why there is no picture), and
the spend is credited to the run's model as *helper usage*
(`Model.helper_usage`, a running total credited once per call key,
of which the agent bills only the part above its mark, so an
interrupted call is neither billed twice nor lost) and billed by the
agent before every limit
check (`KISSAgent._bill_helper_usage`), so the budget cap covers it;
the agent's limit check also runs before each description call
(`Model.limit_check`, handed to the bridge), so a batch of pictures
stops at the cap instead of paying for every one.

How the task runner labels a run that did not finish on its own
(`task_runner`): the `KeyboardInterrupt` injected into the task
thread — bare, or wrapped in the SEA loader's diagnostic error — is
the run's cancellation (`_is_cancellation`) and is reported as `Task
stopped by user` with a `task_stopped` event, or as `Task interrupted
by server restart/shutdown` with `task_interrupted` when the daemon's
own shutdown injected it (`AgentState.interrupted_by_shutdown`;
`_cancel_outcome`).  Either raise replaces whatever outcome the
sub-task finished before it had reached, so a run stopped or crashed
between sub-tasks is not persisted — and auto-merged — as a
completed success: any other exception escaping the setup or the
bookkeeping between sub-tasks is `Task failed: <ExceptionName>:
<message>` with a `task_error` event.

An install never interrupts a run that way: `install.sh` asks the
daemon for its tasks in flight (`scripts/check-kiss-web-active-tasks.py`,
which prints the `restartPendingFile` a busy daemon names) and, when
there are any, installs the app without restarting, records the
restart it skipped in `$KISS_HOME/.kiss-web.restart-pending` (the
installer's path, the state directory and the work dir to serve) and
exits.  The daemon's `_deferred_restart_loop` (`web_server.py`) looks
for that record every 5 s; with one there, no task in flight and no
installer of its own still running, it raises the run-admission
barrier so no new task starts, claims the record
(`kiss.server.deferred_restart.claim` validates and removes it, so a
broken record is not retried) and runs the recorded `install.sh
--restart-only`, which waits out an install still holding the update
lock, restarts the daemon onto the app already on disk and, apart
from retiring a legacy app directory the new one has replaced, does
nothing else.  So a SEA run may itself run `./install.sh`: the run
completes, and the daemon moves to the new build afterwards, taking
the brand's `$KISS_HOME/services.d/` helpers
(`kiss.server.services.Services`, started once the local endpoint
file is written and stopped by process group at shutdown) down and
up with it.

### Refusals and session hand-off

What the executor (`kiss.core.kiss_agent.KISSAgent`, driven by
`kiss.agents.sorcar.relentless_agent.RelentlessAgent`) does around a
SEA's hooks, so a hook author knows what the model reads:

- **Refusals.**  A `refuse(text)` verdict from any `tool_call_hook`
  layer, or a refusal from the framework's own guard
  (`SorcarAgent._guard_tool_call`, first reason wins: `finish` while
  a user message queued during the step is still undrained, every
  time; `finish` while `run_agent` jobs are still running, once per
  run, and `commit_and_merge` with such jobs, once per run as well
  (`_block_finish_with_live_jobs`: `Error: finish|commit_and_merge
  rejected — run_agent jobs are still running: <ids>. ...`, ending
  `finishing again kills every job still running.` or `merging again
  removes the worktree under every job still running and loses what
  it writes there afterwards.`, since a sub-task inherits its
  caller's worktrees; a second call passes); `finish` while a
  worktree is unmerged, once per run;
  `Write` / `Edit` of a non-git-ignored path inside a repository —
  when a worktree the work dir is not in already isolates that
  repository the refusal names the path to use inside it, otherwise
  it asks for `create_worktree` first; `rollback` while the work dir
  is in a repository checkout rather than a worktree
  (`_guard_rollback`, see `checkpoint` / `rollback` above); and, in a run whose toolset
  has `Edit`, the coding gate on `finish` and `commit_and_merge`
  (`coding_gate.CodingGate`, built per run by
  `SorcarAgent._new_coding_gate`; a read-only run such as a reviewer
  is never gated because it reports rather than claims work).  The
  gate observes every tool call and reads every tool result: the exit
  codes of every shell command (`Bash`, `run_commands_parallel`, a
  `bash_job` wait, host or container) go into a ledger keyed by the
  command as written (`result_exit_codes`: a `bash_job` report counts
  only its `status: exited N` / `Job <id>: exited with code N` line,
  not the command echo or the log lines, and a `Bash` call made with
  `on_timeout="background"` whose report, `Job <id>: <command>` with
  its status and `log:` lines, shows the job already exited gets
  that exit, `job_report`; a `run_commands_parallel`
  report gives each command the exit of its `### [i/N]` section, and
  a digest of one the `[i/N] exit N in 1.2s:` lines of its preamble,
  `PARALLEL_DIGEST_STATUS`; a `run_parallel` digest is read for its
  `[i/N] ok` / `FAILED` lines, `dispatch_successes`), the commands of a
  refused call are `refused`,
  and a call the user's Stop interrupted is no evidence.  An `Edit` /
  `Write` under a `tmp` directory inside the work dir
  (`CodingGate.is_scratch`: the progress log, a probe, the verifier)
  is not observed at all: it changes no work and arms nothing.
  Nothing in the `finish` call itself is refused: the `evidence`
  argument (a JSON list of commands, or one string) is shown to the
  reviewer with the exit code of each command's last run, `not run`
  for one that never ran (`CodingGate.evidence_report`; a command
  that never ran, failed, or tests nothing — `echo`, `cat`, the words
  of the task — proves nothing to it, the `REVIEW_PROMPT` says), and
  the wording of `summary_in_html` is the reviewer's to judge.  A
  benefit the task expects from the change ("prevents attention
  degradation") is motivation, not a requirement: the reviewer does
  not ask for it to be measured unless the task asks for that
  measurement or names a target value.  The ledger's reading check
  (`READING_PROMPT`), the container harness's review
  (`coding_sea.REVIEW_PROMPT`) and its ledger reading check
  (`coding_sea.READING_CHECK_PROMPT`) carry the same sentence.  The
  refusals of earlier gates on the finish call alone — a summary that
  said the result was guessed or unverified, a file change no shell
  command checked, an `evidence` command that never ran or failed, a
  summary the decisions model found leaving a case of the task open —
  were answered by a
  rewording far more often than by a check, and the measurement of
  2026-11-04 found no pass that depended on them, so they are gone;
  `evidence_gate` keeps the readers of what a shell command changes
  (`file_changes`, `writes_files`, `is_scratch_path`) and nothing that
  refuses.  The run
  becomes a *coding run* — the
  gate is armed — once an `Edit` / `Write` targets a code, build or
  configuration file (`coding_gate.is_code_file`: `CODE_SUFFIXES`
  from `.py` to `.toml`, `.yaml`, `.json`, `.css`, and
  `CODE_FILE_NAMES` such as `Makefile`, `Dockerfile`,
  `CMakeLists.txt`), a shell command changes one
  (`CodingGate.observe`, read from the files the call changes,
  `evidence_gate.file_changes`, described with the verifier reuse
  below, so a
  `rm`, `mv` or `git checkout` of a code file arms it too, while a
  command that reads one, `cat install.sh`, names one in text, `git
  commit -m "rm stale"`, or writes one under a `tmp` directory, `gh api
  ... > tmp/x/pages.json`, arms nothing), the
  verifier exists, or a sub-task that was not a review ran — or a
  `rollback` rewrote or deleted a file (`CodingGate.restored`: told,
  as for a sub-task, by the work's `fingerprint` before and after the
  call, so a rollback that failed or restored nothing changes nothing;
  a passing verifier run before it is not reused) — and the
  diff since the run's base commit shows a code file changed; a run
  whose `model_config` has `coding_gate: false` (`CodingGate.coding`,
  a framework flag of `core.models.model.FRAMEWORK_ONLY_CONFIG_KEYS`;
  `/sync` sets it, since what it changes is the merge of commits made
  elsewhere and the resolution of their conflicts, which no reviewer
  of the run's diff can tell from the pulled commits) is never a
  coding run, whatever it changes, and a `finish` on a run that is no
  coding run is not gated at all (`finish_verdict` returns `None`).  On a
  coding run a `finish(success=True)` runs its stages at once in
  threads, each recorded as a `gate_stage` event with its duration
  and outcome (`core.event_sink.publish` through the run's printer):
  the agent's own verifier `verify.sh` in the task's scratch
  directory (`tmp/<task-id>/verify.sh`, the path the system prompt's
  How to Work rule 3 names; run with the run's own shell, a subprocess
  under `bash -c` found through `processes.find_bash` — Git for
  Windows keeps `bash.exe` off `PATH` — in the work dir, or in the
  task's container through the raw exec, `DockerManager.run`, not
  the `Bash` tool, whose output cap would cut the gate's file lists
  and diffs, `VERIFIER_TIMEOUT` 900 s;
  no verifier means no verdict — and the gate's own last run of it is
  reused instead of a new one, the stage recorded as `reused`, while
  the tree is the one it ran on — the same work dir, change count and
  `fingerprint` of the work — and the script's digest is unchanged
  (`git hash-object --no-filters` of the whole file, the bytes Bash
  runs rather than what Git would store, so an edit past the prefix a
  review reads is a new script and a `text` attribute dropping a CR
  hides nothing), `CodingGate.verifier_memo`, so a round in which
  nothing changed runs nothing, and `verifier_stale` forgets that run;
  never outside git, where there is no fingerprint.  A merge that went
  through carries the worktree's last run into the checkout
  (`CodingGate.carry_verifier_run`, from `merged`) when nothing changed
  since it and the checkout's fingerprint is the one it saw, so a
  finish there that finds the same script reuses it; a file another
  merge brought makes the finish run it again.  The agent's own
  passing run of the verifier counts as the gate's run, and is reused
  the same way, only when it is clean: a shell command whose exit code 0 was the
  verifier's (`CodingGate.verifier_decides`, the module function
  `verifier_decides(command, work_dir, verifier_path)` the coding SEA
  shares, read from the scanned command: `bash`/`sh
  ./tmp/<id>/verify.sh` with no option but a switch that keeps the
  first operand the script (`SHELL_SWITCH`: `-x`, `-e`, `-u`, `-v`,
  `-l`, `-p`, `-E`, `-T` and their clusters, `--posix`, `--norc`,
  `--noprofile`, `--login`, `--verbose`; not `-n` or `-D`, which run
  nothing, `-s`, whose script comes from stdin, `-c`, or `-o` / `-O`
  / `--rcfile`, whose next word is their value), a `--` or `-` ending
  the options so that the word after it is the script whatever it
  starts with, or the script itself as
  the command, from the work dir or after a `cd` the scanner follows,
  `verifier_command`, at the top level of the command, not
  negated, its output not piped, not after `||`, not in the
  background, nothing before it that leaves the shell — `exit`,
  `return`, an `exec` of a program, a `set -n` / `set -o noexec`
  left on, `leaves_shell` — and everything after it after
  `&&`, so `bash v.sh && echo ok` counts and `bash v.sh | tail`,
  `bash v.sh; echo $?` and `bash v.sh || true` do not, since their
  exit shows nothing about the verifier's; and only when the call
  writes no file itself, since in `bash v.sh && printf x > f` the
  write follows the run; never a `bash_job` report of a background
  run) keeps its output as the verifier's last output a review is
  shown (`last_verifier_run`); when that command is the call's only
  one, nothing runs after the verifier (`verifier_ends`) and
  `verifier_key` (work dir, change count, fingerprint, script digest)
  is the same before and after the call, it becomes the gate's memo,
  else it forgets the memo (a program in the command may have changed
  a file no scan of it shows), so the gate runs the script itself at
  the finish; an `Edit` /
  `Write` of `verify.sh` or a shell command naming it forgets the
  gate's run as well, as does any later change of the work — an edit
  outside `tmp/`, or a shell command that changes files, which
  changes the memo's change count — and, when a reviewer is set, a
  review of the diff.
  What a shell command changes, `evidence_gate.file_changes`, is read
  from its simple commands, `shell_scan.simple_commands(optimistic=True)`:
  the command as it was meant to run, an assignment or a `cd` after
  `&&` or `||` holding for the rest of the command, `cd d && T=tmp/x
  && a; cp b $T/`, rather than the guards' reading, which lets no
  write through on a path that may not have been set.  A simple
  command changes the targets of its output redirections, its own and
  its groups', other than `/dev/*` and scratch files — a path under
  any `tmp` directory, `_SCRATCH`, resolved in the directory the
  command runs in when the scanner knows it, so `cd /w/tmp/x && cat >
  f` is scratch and `tmp/../a.py` is not, a word the shell still
  expands, `$OUT`, `~/x`, judged as written — and what its program
  changes, `_program_changes`, a program counting in command position
  only, wrappers dropped: `cp`, `rsync` and `install` their last
  operand, for `cp` and `install` the `-t DIR` / `--target-directory
  DIR` destination instead when given; `tee`, `mv` (its `-t DIR`
  target as well),
  `rm`, `rmdir`, `unlink`, `touch`, `mkdir`, `shred`, `truncate` and
  `patch` every path operand, `patch < fix.diff` with none a change
  all the same, `dd` its `of=` operands, `chmod`, `chown` and `chgrp`
  every operand but the first, or every operand when a
  `--reference[=| ]RFILE` supplies the mode or owner (`RFILE` is read,
  not changed; `_REFERENCE_OPTION`) (`WRITING_PROGRAMS`), a call naming
  scratch files only no change; `sed` with `-i` / `--in-place` in any
  position, no change when its files — the plain operands after the
  script, which an `-e` / `-f` option may supply (`_sed_files`) — are
  all scratch, otherwise listed with every plain operand; `make`,
  `cmake` and `ninja` (`BUILD_TOOLS`) unless one argument starts with
  `test` or `check` and every argument does, is an option or has an
  `=`, `make test`, `make -k check V=1` (`cmake --build . --target
  test` is a build: its `.` is neither); any program
  with `--fix`; the writing subcommands of the package tools
  (`runs_subcommand`, `WRITING_SUBCOMMANDS` — `git checkout`, `pip
  install`, `uv sync`, `npm ci`, `cargo build` — the tool in command
  position or as `python -m pip`, `uv pip install` read as `pip
  install`, and `uv run`'s next plain word put in command position
  after the options of `UV_VALUED_OPTIONS` take their values, so `uv
  run pip install x` installs, `uv run pytest` checks and `uv run sed
  -i s/a/b/ f` changes `f`, the program's changes being the
  command's); a nested shell's `-c` string, `bash -c "echo x > f"`,
  and the literal text of an `eval` are read as commands by the
  scanner itself (`shell_scan.command_text`; a text that runs no
  command, a comment or blanks, carries nothing, so the redirections
  stay the outer command's and still truncate), a shell named as an
  argument, `echo bash -c ...`, not; `find`'s
  `-delete` and `-exec` / `-execdir` / `-ok` / `-okdir` the program
  run on the matches, `{}` standing for the start paths, each clause
  up to its `;` or `+` and every clause of one `find` counted (an
  `-execdir` clause runs in each match's directory, so its operands
  count as scratch only when every start path's parent is scratch and
  no operand holds `..`; otherwise they are judged as written); a
  here-document is input to the program reading it, not a write).
  Then the review's findings (or, once the run's review is spent, the
  summary's answers to its open findings, `unanswered_verdict`) are
  judged, and every failed stage is refused together
  (`CodingGate.refusal`, the text of `gate_refusal`, whose header
  says a later round is refused only while the verifier fails) under
  one rule, `CodingGate.refusable`: the first refusal buys one round
  whatever refused it, a later round is refused only when the
  verifier stage refused at that call, and no round past
  `MAX_GATE_ROUNDS` (3) per run is refused at all — the pass rate was
  flat against the number of refused rounds while each round cost a
  model turn at full context; a verifier that fails or times
  out counts against `MAX_VERIFIER_REFUSALS` (3), past which it is not
  run again.  A
  finish with open verdicts that is not refused goes through with them
  appended to the summary as caveats
  (`deliver_with_findings`, `DELIVERED_CAVEATS`; `UNVERIFIED_CAVEAT`
  is appended to a finish that goes through, whatever let it through,
  only when the verifier exists but
  was skipped because its three refusals were already spent,
  `verifier_skipped`; a missing verifier adds no caveat), recorded as
  a `delivered` stage.
  The reviewer (`gate_reviewer`, resolved from the `coding_gate_reviewer`
  key of `$KISS_HOME/config.json`: a model name, `off`, or empty for the
  strongest configured model of another family than the run's,
  `model_families.other_family_model`, none when every configured
  model is of the run's family) reads the task, the summary, the
  verifier and its last output, the evidence named and the diff
  (`diff_report`: each file cut at `REVIEW_ONE_FILE_BYTES` 30,000
  characters, and no further file started once the accumulated text
  has reached `REVIEW_FILE_BYTES` 120,000, so the last file shown may
  push the total past it; files the agent edited in another
  repository's worktree included), runs no commands, answers one
  `VERDICT:` line and numbered `NOT DONE` findings.  The review
  starts with one non-agentic model call (`CodingGate.side_call`) at
  the provider's default effort — the run's `model_config` without its
  `reasoning_effort`, `default_effort`, so a `base_url` or `api_key`
  still applies — with `REVIEW_DEADLINE` (240) seconds per attempt: the
  call runs under a stop event of its own (`call_stop_event`: a
  watcher thread, `_watch_call`, sets it at the deadline or when the
  run's own stop event is set, and the model clients end the
  streaming call on it), a deadline hit becomes the error `no answer
  within 240 s` while a Stop of the run propagates; the attempt's
  spend is charged as it ends, however it ends.  A review that came
  back empty, cut by the output cap (`OutputCapError`), refused
  (`ModelRefusalError`, `ContentFilterError`) or past the deadline
  (`retryable_review_failure`) is retried once (`low_effort`): at
  `reasoning_effort` `REVIEW_RETRY_EFFORT` (`low`) when the model's
  `MODEL_INFO` entry carries a `thinking` level, otherwise with the
  same settings, since a provider rejects the parameter on a model
  that takes none; the stage's `retried` field names the first
  attempt's error; any other failure is not
  retried.  A review without a verdict, past the deadline or whose
  call failed (the stage's outcome `error`, its `error` field the
  call's error, `empty answer` or `no verdict`)
  refuses nothing: it keeps the previous findings open and counts
  under `MAX_FAILED_REVIEWS` (2), a completed review under
  `MAX_REVIEWS` (1); reaching either cap stops further review
  (`review_possible`), so the one review reads the whole diff and
  every later finish of the run is judged on its findings,
  `unanswered_verdict`, which the summary must answer with a change
  or a quoted command and its output, as the decisions model judges,
  `answered_findings`; without the decisions model, or when its call
  fails, the findings are unjudged — `answered_findings` returns
  `None` — and the finish is not refused on them, since a refusal
  must not depend on an optional service, though they stay open.  (A
  review is asked for again only after one that failed — it does not
  count under `MAX_REVIEWS` — and then only when a file changed since
  the last review, `review_wanted`; it is shown the same prompt as the
  first, not its previous findings.)  A
  read-only reviewer the agent dispatched
  itself (`run_agent` / `run_parallel` with `tool_profile="review"`
  or a task that reads as a review) counts as the review once a
  sub-task of it succeeded and no file changed since.  Any other
  local sub-task counts as a change of the work only when the work
  differs after it: a digest of every path that differs from the base,
  tracked or untracked, with its bytes on disk and executable bit, or
  as deleted (`CodingGate.fingerprint`; an untracked file renamed with
  its contents kept is a change, and committing a file changes
  nothing, so the worktree before its merge and the checkout after it
  have one digest when they hold the same files) is
  taken at the dispatch and compared when the result, or the
  `agent_job` report of a job left running, is read (a `wait_any`
  report is read job by job, each `Job <id> (<name> agent task)
  finished:` section its own result, `agent_dispatch.wait_any_results`;
  `CodingGate.subtask_done` records each one; outside git,
  where none can be taken, the sub-task is a change); a sub-task sent
  to another machine (`run_agent(machine=...)`) touches no file here
  and reviews nothing here, so it is neither.  A tool the gate does
  not model (a SEA's own, an MCP server's: `custom_tools`, the run's
  `_extra_tools`) may move HEAD — a sync tool that committed and
  pulled — so after each of its results the gate takes HEAD as its
  base again (`follow_head`; once a base is recorded, and when `git
  rev-parse HEAD` answers) unless the run moved HEAD itself
  (`moved_head`: a `git commit`, `merge`, `pull`, `rebase`,
  `cherry-pick`, `revert`, `am`, `reset`, `checkout` or `switch` among
  its shell commands, `HEAD_MOVING`, or the merge of its worktree), whose
  commits are its work and stay in the diff.  A
  `finish(success=False)` on a coding run runs the verifier too
  (counting no refusal) and appends its outcome to the summary
  (`GIVEUP_NOTE`); the first give-up whose verifier passes is answered
  once with `GIVEUP_RECONSIDER_VERDICT`, the next ends the run.  A
  text-only turn on a coding run (the implicit finish below) is vetoed
  once with a request for an explicit `finish`.  A `commit_and_merge`
  on a coding run runs the verifier and the review first (the
  reviewer is told the summary is not written yet,
  `MERGE_SUMMARY_PLACEHOLDER`, and that the evidence is not named
  yet either, `MERGE_EVIDENCE_PLACEHOLDER`, so it judges the diff and
  the verifier and reports neither as a gap), a refusal there counting
  as a gate round under the same rule (`CodingGate.refusable`: the
  first merge refusal whatever refused it, a later one only when the
  verifier failed at that call, none past `MAX_GATE_ROUNDS`), and a
  merge asked for again with findings open and nothing changed is
  refused with them (`MERGE_UNANSWERED_VERDICT`); the gate follows the
  agent into its worktree (`CodingGate.move`) and back to the checkout
  at the merge.  Every merge refusal ends with `MERGE_CAP_NOTE` ("A
  merge asked for again goes through with its open findings attached
  to the result (and to the finish summary) unless your verifier
  fails, so fix them now."), and the merge the rule does not refuse
  goes through, so no run ends with its work on a branch nobody
  merges: recorded as a `delivered` stage with `call:
  commit_and_merge`, its result ending with the open verdicts
  (`MERGED_WITH_FINDINGS`, "Merged after N refused rounds with these
  findings open: ..."), which the finish summary then carries as
  caveats (`deliver_merge_caveats`, `MERGED_CAVEATS`, a "Merged after
  N refused merge rounds; findings open at the merge" heading and a
  list) except those it answers by quoting the command and output
  that shows the finding wrong, as the decisions model judges — every
  one without it.  A merge that went through resets the round count and drops
  the open findings (`CodingGate.merged`, called by the agent once the
  repository has no worktree left): the rounds the merge spent and the
  findings its review raised stay with the merge, the finish after it
  is refused on rounds of its own and, the run's one review spent at
  the merge, judged on no review, so no finding older than the merge
  ends as its caveat other than those the merge was let through
  with.  When the agent first writes its interpretation ledger
  (`Write` of `LEDGER_NAME`, `ledger.md`, in the task's scratch
  directory, `CodingGate.is_ledger`; the system prompt's How
  to Work rule 1 asks for it; once per run, and only when a reviewer
  is configured), the reviewer model reads the task and
  the ledger once (`CodingGate.reading_note`: one `side_call` with
  `READING_PROMPT`, the ledger cut at `REVIEW_ONE_FILE_BYTES`) and
  names the sentences of the task it would read differently or finds
  unsettled; the answer is parsed with `numbered_items` unless it is
  the single line `NONE` (`READING_NONE`) or the call failed, the
  call is recorded as the `reading` gate stage (`pass`, `fail` with
  the number of `items`, or `error`, with the model and cost), and the
  items are appended to the `Write`'s result (`READING_NOTE`: settle
  each in the ledger before the first edit, by adopting the reading
  or by keeping yours next to the words of the task that rule the
  other out) — the agent's guard returns an allowing
  `Verdict` with that note (`SorcarAgent._guard_tool_call`,
  `tool_verdict.allow_with_note`; the `KISSAgent.tool_call_guard` may
  return a `Verdict` as well as a refusal text, and
  `KISSAgent._tool_call_verdict` passes an allowing one through) when
  the run has the `Edit` tool, and nothing is refused: a reading only
  the user can settle is theirs to answer.  A
  sub-agent gets the same gate without a reviewer: its
  caller's gate reviews the whole diff.  A review's model spend is
  charged to the run that asked for it (`_charge_gate_usage`, bound
  to the run's ledger epoch and persisted row when the gate is built,
  `_new_coding_gate`, since the review runs on its own thread and a
  Stop lets the run end without waiting for it): a run with a row and
  a server printer is charged through the printer's late-charge
  bridge (`charge_task_usage`, `task_update.charge_side_channel_usage`)
  — banked on the live ledger while the row is unfinished, and once
  the row is finished (a review a Stop left running) added to the row
  and its finished ancestors with a persisted `usage_info`, rather
  than landing on the agent's next task — a run with a row but no
  such printer (a CLI run) gets the same row-or-ledger policy from
  `_charge_persisted_usage`, under `TASK_USAGE_LOCK` as the bridge
  is, a finished row then broadcasting `tasks_updated` so the history
  lists show its new total, and a run without a row goes to the
  ledger epoch directly; keyed per attempt (the ledger and the
  finished row count a key once, `persistence._add_keyed_task_usage`)
  so a retry or a stop neither bills it twice nor loses it.  The
  reviewer emits no `llm_call`, so a row charge also records the
  reviewer model, spend and tokens under the row
  (`persistence._record_model_charge`, a `gate_stage` event of stage
  `charge`, unique by its `key` under the row: a second copy, queued
  again because the first still waited in the event journal where
  the recorder's lookup cannot see it, is dropped by the event writer
  inside the database's write lock, `_write_event_batch_locked`) for
  the Spend subpanel's split by model, however the
  review ends; a model call made for the run outside its agent loop
  — the gate's `decide` call on the findings
  (`CodingGate._charge_decide`, under the decisions model), the
  commit message, a `talk` synthesis, a `decide` call — is charged
  the same way under a key of its own (`side_call_binding`, the run's
  epoch and row taken as the call starts, so a reply that lands after
  the task ended still charges the task it was made for, and
  `_attribute_side_call_usage`) and filed under the model it ran on.  The
  classifier fold and the row's final usage save are each retried
  once when an exception, a Stop's injected interrupt above all,
  cuts them short (`_retried_cleanup_step`), the
  retry of a save that already committed writing nothing
  (`_save_final_usage(unless_finished=True)`, `persistence.task_end_ms`)
  so it does not overwrite a late charge on the finished row.  A
  verifier or review stage
  that raises refuses the finish like a failed stage (`run_stage`
  logs the error and records the stage as `error`; its verdict reads
  `Not finished: a check of the gate could not run (...)`), counting
  a round like any refusal; only an exception that escapes the gate
  itself (`finish_verdict` / `merge_verdict`, caught in
  `SorcarAgent._guard_evidence`) lets the call through and logs the
  error rather than ending the run on the gate's bug.  The refusal reaches the model as
  `tool_verdict.refusal_result(name, text)`: `REFUSED: the <name>
  call did NOT run and had no effect. <text>`.  The executor
  remembers the refused tool; the next allowed call of a different
  tool gets `tool_verdict.REISSUE_REMINDER` appended to its result,
  a call of the refused tool itself clears the reminder silently, and
  `finish` never carries one.  The implicit finish of a second
  consecutive text-only turn (`kiss_agent.MAX_CONSECUTIVE_NO_TOOL_CALLS`,
  2) stands in for a `finish` call the model never made, so it faces
  the same hook and guard, consulted as `finish` with no arguments;
  a refusal there is the user message the model reads next (a
  one-shot veto such as the unmerged-worktree gate is spent on it),
  and only an unrefused one ends the session with the model's last
  text as its result (`success=True`, through the registered `finish`
  contract).  `set_model` refuses itself, outside the guard, under a
  daemon that pins every task tree to its launch model
  (`KISS_PIN_TASK_MODEL`, set by the benchmark daemons so a score
  measures one model; `agent_dispatch.task_model_pinned`, which also
  refuses a `run_agent` naming another `model` or a `model_config`,
  one whose SEA's `settings()` names another `model` or a
  `model_config` (`pinned_model_conflict` checks the SEA's effective
  settings too, since the daemon applies a SEA's `model` when the
  call names none: "Call the plain sub-agent (agent='') or a SEA that
  names no model"), and a sub-task starting with a slash command,
  whose SEA may pick its own model).
- **Context compaction.**  Before a model call, once the context has
  grown past `compaction_start_tokens` (100k tokens, then every
  `compaction_step_tokens`, 100k, of growth) or past the steady-state
  target (`compaction_target_fraction`, 0.3, of the model's window:
  150k of 500k, never below `compaction_start_tokens`,
  `compaction_target_tokens`; past it every step is a candidate),
  whichever comes first, old large tool outputs and the large
  arguments of old tool calls (the content of a `Write`, the strings
  of an `Edit`, a long shell command) are considered for replacement
  with stubs
  (`kiss.core.context_compaction`; the stub of an output says how to
  get the text back, the stub of a `Write` / `Edit` argument names the
  file it went to, which is one `Read` away since the compaction
  forgets the Read tool's dedupe), but never within
  `COMPACTION_MIN_STEP_GAP` (10) steps of the previous compaction:
  every compaction is a full prompt-cache miss, so two of them a few
  steps apart would pay twice for one batch of stale text.  The
  newest six results (with their calls' arguments), results and
  arguments of 500 characters or fewer, results of `Edit` / `Write`
  / `finish` and arguments of `finish` are never stubbed, and the
  compaction is skipped (re-planned on a later step) unless it drops
  at least 25% of the context (10% once past the target), at least
  the share the model's cache discount needs to pay the miss back
  within `PAYBACK_STEPS` (50; 28% on a 5% cache-read price, 15% on
  10%, `payback_floor`), while the context is still under 80% of the
  hand-off limit.  A model whose API requires every earlier turn's
  reasoning back whenever tools are present
  (`Model.compacts_whole_turns`: the `deepseek` family,
  `model.REASONING_PASSBACK_FAMILIES` — DeepSeek answers 400
  otherwise, whatever route serves it) cannot have a text stubbed inside a turn,
  so at the same thresholds it loses whole old turns instead
  (`context_compaction.plan_turn_removal`): every model turn but the
  newest `KEEP_RECENT_TURNS` (10) goes entire — reasoning, text, tool
  calls, their results and the user notes between them; the messages
  before the first model turn, the task, stay — and one user message
  stands where they were (`TURN_REMOVAL_NOTE`: `[Harness: the first N
  turns of this session were removed from the context to keep it
  small. ...]`, an earlier removal's note absorbed so one note counts
  them all), the same payback test deciding whether the removal is
  worth its cache miss.  A Claude model on the Anthropic API (not
  Bedrock, and only while `context_edit_trigger_tokens` is positive,
  `AnthropicModel.clears_tool_uses_in_bulk`) is compacted
  by a rule of its own instead of these thresholds
  (`KISSAgent._maybe_clear_tool_uses`): nothing is edited until the
  context passes `context_edit_trigger_tokens` (250k); then the
  tool calls older than the newest `context_edit_keep_tool_uses`
  (20) go through the ordinary planner (`plan_compaction`), which
  stubs only their eligible large results and arguments (a value of
  at most `MIN_COMPACT_CHARS` (500) characters and a protected tool's
  stay), and the old thinking blocks are removed
  (`Model.drop_stale_reasoning`) in one edit made only when the
  stubbed text and the removed reasoning together reach
  `context_edit_clear_at_least` (50k) tokens and at least
  `COMPACTION_MIN_STEP_GAP` (10) steps have passed since the previous
  compaction, so the provider's prompt cache is re-written only
  then.  The two settings hold for a
  model whose cache write costs fifty cache reads (Claude Fable 5.2
  and 5.1); `context_compaction.clear_thresholds` multiplies both by
  the square root of the model's own ratio over fifty
  (`model_info.get_cache_write_ratio`: 177k and 35k at 25 reads a
  write, Opus and Sonnet 5.5; 125k and 25k at 12.5, the other Claude
  models) and holds the trigger at 80% of the hand-off limit or
  less, which is 112k on a 200k window.  The minimum counts the
  reasoning with the stubbed text: a reply's reasoning is its output
  tokens less its visible text and tool calls
  (`context_compaction.reasoning_tokens`), since the provider returns
  a summary of it.  Claude 5.1 and later, which drop a thinking
  block sent back after an earlier message changed or (Fable 5.2)
  refuse the request, lose every thinking block at a clear; an
  earlier model keeps those of its newest reply, which the API
  requires before the pending tool results.
  Each compaction applied is a `compaction` event
  (`step`, `context_tokens`, `dropped_tokens`, `results`, `arguments`,
  `reasoning_blocks`, `target_tokens`; a whole-turn removal reports
  `turns` with zero `results` and `arguments`), persisted with the task's
  events like the gate's `gate_stage` records and the
  `context_edits` events (`step`, `edits`, `cleared_input_tokens`)
  an Anthropic model's server-side context editing reports when the
  provider cleared reasoning turns (`clear_thinking_20251015`, the
  one edit the agent asks for, with `reasoning_keep_turns`) or, for
  an edit a run's own `model_config` asked for, tool uses
  (`clear_tool_uses_20250919`), from the context before a call
  (`anthropic_model.context_edits`, `json_printer.RECORD_EVENT_TYPES`).
  Two switches, both of which must be on for a step to compact
  (`KISSAgent._maybe_compact_conversation`): the daemon's global
  **Compaction** setting (the Settings panel's checkbox, `compaction`
  in `config.json`, applied as `DEFAULT_CONFIG.tool_output_compaction`
  by `apply_config_to_env`; its default is seeded from
  `KISS_TOOL_OUTPUT_COMPACTION`, so `=0` in the environment opts out
  until the switch is saved from the panel), which off stops every run
  on the daemon; and the run's own `compaction` option (the `run()`
  keyword, the SEA setting — a headless `run_sea()` and the in-process
  Task update read it from the SEA's settings the same way — and the
  `run_agent` option), which a run turns off for itself alone.
- **Routine turns.**  The turn after a refused call, a `summary`,
  `memory_search`, `memory_pull`, `memory_read`, `memory_list`,
  `memory_write` or `memory_refresh` call (`kiss_agent.ROUTINE_TOOLS`;
  not `memory_delete`), a `bash_job` / `agent_job` wait (or `agent_job`
  `wait_any`) that returned with the job still running, or an unchanged
  `Read` is a routine turn
  (`kiss_agent.is_routine_turn`).  The `routine_turn_effort` config
  key (`KISS_ROUTINE_TURN_EFFORT`, empty by default) names a lower
  reasoning effort (`low`, `medium`, ...) for such turns on a model
  that supports thinking; `cost_report` counts routine calls and
  their output tokens.  Empty changes nothing.
- **Session hand-off.**  When a session of a task ends before the
  task does — the executor's context reached its hand-off limit
  (`ContextWindowExceededError`), or it hit its step limit
  (`StepLimitExceededError`, a `KISSError` of its own so a caller can
  tell exhaustion from a failure at the last step) or another
  uncaused `KISSError` after the first step (a budget overrun instead
  ends the task: a sub-agent returns a partial result quoting what it
  did, a top-level task fails as budget-exceeded; three consecutive
  step errors, or any other exception, end it with a failed result)
  — the next session starts from a summary of the ended one.  A
  session the model ends itself with `finish(success=False,
  is_continue=True)` continues from the summary the model wrote; on
  the exception path the summary is written by
  a tool-less helper (`finish` only, at most `SUMMARIZER_MAX_STEPS`
  = 3 steps; it is the task's live session while it runs,
  `_current_executor`, so the task-wide budget check and the live
  usage readers see its spend, which is banked once on every exit,
  a Stop landing inside the bank included) from a scripted digest of
  the trajectory, `kiss.agents.sorcar.handoff_digest.handoff_digest`:
  the task text (3,000 characters), the last three `summary` texts
  the model wrote, the files written with
  their top-level symbols (up to 40), the last five failing tool
  results and the last 40 tool calls with the first 300 characters of
  each result (a tool call is read off the one-line `name(k=v, ...)`
  repr the trajectory records; a multi-line python block, one with
  positional arguments, or one whose text before the `(` is not a
  single whitespace-free name, such as `x = f(...)`, is a snippet
  the model wrote and is skipped, `handoff_digest._parse_call`).  The
  continuation's first message also carries a
  `# Current state` block (`SorcarAgent._continuation_state`): every
  active worktree, named as unmerged so `commit_and_merge` is still
  owed and edits go under the worktree path, and the files the task's
  `Write` / `Edit` tools wrote so far (up to 40 listed), so the resumed
  session neither opens with a refused main-checkout command nor
  re-reads what it wrote.  The SEA's call hooks are not applied to
  the helper (it makes no tool call but `finish`).


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
    sea_path="my_agent.py",
)

# Continue the same chat
result2 = sorcar.run(
    "Now fix the issues you found",
    chat_id=result1.chat_id,
    sea_path="my_agent.py",
)
```

An SEA can also force a specific chat via `settings()["chat_id"]`.


## Model configuration

Use the `model_config` setting to pass custom endpoint URLs, headers,
or sampling parameters.  A `base_url` routes every model call of the
run — in a chat tab, under `run_agent` and headless alike — to that
OpenAI-compatible endpoint, a local proxy the SEA starts included; the
`model` must still be a catalogued name so the calls are priced.
`"plain_io": True` leaves the framework's own additions out of what
the model reads: the system prompt on the wire is the SEA's
`system_prompt` return without Sorcar's framework trailers — no
`# Restricted tool profile` or `# Web tools are off` note, no
editor-file line, no `# MOST IMPORTANT INSTRUCTIONS` block (work dir,
PID), no `# Task Settings` section and no `AGENTS.md`; only the
memory protocol stays, and only while the run has the memory tools
(`use_memory: false` drops both), as does
the way the adapter offers the run's tools (a native tool schema in
the request, or the `## Available Tools` text catalog a CLI-backed
`cc/*` / `codex/*` model appends to its system instruction) — and no
usage line (`Steps: ..., Context: ..., Budget: ...`) follows a tool
result or a later user message (`KISSAgent` sets
`usage_info_for_messages` to the empty string).  The user prompt is
the SEA's `prompt` return plus what `SorcarAgent.run` appends in
every mode: the recalled-memory block when the memory tools are on
and their search of the task text finds any, and a `# Important`
attachment notice when files are attached.  So
the prompt and every observation read the same in every run mode (a
chat tab, a headless `scripts/run_sea.py` evaluation, a training
episode) and the SEA controls the system prompt; the printer and the
trajectory keep the usage.  A plain-I/O run that keeps the
default `SYSTEM.md` with a restricted `tool_profile` gets no note
reconciling that prompt's rules with its toolset, so such a run
supplies its own `system_prompt`.  A SEA's `model_config` replaces
the caller's whole.

```python
class LocalModelSea(BaseSea):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {
            "model": "my-custom-model",
            "model_config": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "sk-local-key",
            },
        }
```

When `model_config` contains a `base_url`, the model factory bypasses
its normal provider routing and creates an OpenAI-compatible model
pointing at that URL.  The daemon's model preflight then only asks
that `model` be a name the bundled catalog or `$KISS_HOME/MY_MODELS.json`
knows (`model_info.is_known_model`, so the run can be priced); the
provider's own API key or CLI executable is not required for such an
endpoint run.  Without a `base_url`, `model` must be in the daemon's
available model list (an API key for HTTP providers, the executable
on `PATH` for `cc/*` / `codex/*`), or be the `-xhigh` alias of a
`codex/*` model in it (`codex/gpt-6-astra-xhigh`; the catalog lists no
effort aliases for Codex): the Codex adapter runs the base model at
that effort, sized and priced as the base
(`model_info._strip_thinking_alias`).  So replace `my-custom-model` above
with a catalogued or `MY_MODELS.json` name; otherwise the task fails
with `No model available under the name "my-custom-model".  Pick another
model or configure its provider (API key or CLI).` (on a daemon with no
usable model at all: `No model available.  Set at least one API key in
the environment.`).

`reasoning_effort` in `model_config` is the framework-wide spelling
of a model's effort level (`low`, `medium`, `high`, `xhigh`, `max`,
as the vendor allows).  Without it the adapters send the catalog
entry's `thinking` level: a generated `-<level>` alias
(`claude-opus-5-5-medium`, `gpt-5.6-sol-high`, marked `alias_of` its
base) carries its own level and a base entry the level the catalog
runs it at (`claude-opus-5-5` and `gpt-5.6-sol` are `high`), so the
request matches the catalog's figures rather than the vendor's
default (Opus 5.5 and Haiku 5.5 run at `medium` when the field is
omitted).  The OpenAI-compatible adapters default `reasoning_effort`
on their copy of `model_config` unless it or a native
`reasoning.effort` is set (`openai_compatible_model._model_thinking_level`);
the Anthropic adapter sends it as `output_config.effort`
(`anthropic_model._effort_level_for`: an explicit `reasoning_effort`
wins, since it is the key the agent sets when it lowers the effort
for one turn, then a caller's own `output_config.effort`, then the
catalog level; an entry that rejects the parameter,
such as Haiku 4.5 or Sonnet 4.5, carries no `thinking` and gets
none).  The Claude Code adapter resolves the level the same way
(`ClaudeCodeModel._effort_level`, which calls `_effort_level_for`)
and passes it to the `claude` CLI as `--effort`: a generated
`cc/claude-opus-5-5-xhigh` alias has its `-xhigh` stripped from the
`--model` argument (`_strip_thinking_alias`) and runs `--model
claude-opus-5-5 --effort xhigh`, a base `cc/claude-opus-5-5` runs at
the level the catalog lists for it (`high`), and the CLI's own short
names `cc/opus`, `cc/sonnet` and `cc/haiku` carry no `thinking` and
send no flag, so the CLI's default applies; the CLI runs a level the
model lacks at the highest one below it.  A turn that sends the flag
runs without `CLAUDE_CODE_EFFORT_LEVEL` in the CLI's environment
(`_EFFORT_ENV_VAR`), the one setting that outranks the flag in the
CLI's resolution order.  Because the CLI downgrades an unsupported
level instead of rejecting it, a `cc/claude-*` entry is never probed:
the updater mirrors the level and effort aliases of its direct
`claude-*` twin (`update_models._sync_claude_code_thinking`, resolved
like a Bedrock id's twin, the dated key first), and
`_thinking_scale_for` gives a `cc/claude-*` key the Anthropic ladder.
`_model_thinking_level` uses the loaded catalog entry when
the name is present there and otherwise reads
`$KISS_HOME/MY_MODELS.json` as it is now, so an entry saved in the
settings panel under a new name after the daemon started carries its
`thinking` level before the next restart; a custom endpoint whose
name matches neither the loaded catalog nor the file gets no effort
parameter.  The OpenRouter Claude twins
(`openrouter/anthropic/claude-*`) carry the same ladder: OpenRouter
forwards `reasoning.effort` to Anthropic unchanged, so the catalog
updater (`update_models._thinking_scale_for`) gives such a slug, like
a `bedrock/` id, the ladder of the Claude model behind it and
generates the same effort aliases as the direct twin, up to that
family's top level (`_anthropic_effort_scale`: Opus 4.5 stops at
`-high`, Opus 4.6 and Sonnet 4.6 at `-max` without `-xhigh`, Fable
5.1 has the full ladder, so `openrouter/anthropic/claude-fable-5.1-max`
exists and is `alias_of` its base).  Their level cannot be probed,
since OpenRouter answers 200 at every level for every Claude, so
`_sync_openrouter_claude_thinking` mirrors the direct twin's level (a
twin newly probed this run hands over its probed level, a twin
retested this run its new level, any other the maximum recorded on
disk) and a slug with no direct twin (`claude-opus-4.1`, the
`:thinking` variants) keeps none.  The same twin mapping
(`model_info.openrouter_twin`) is the automatic fallback when the
direct provider rejects a request for a billing or availability
reason — provided an `OPENROUTER_API_KEY` is configured and the entry
declares no `fallback` of its own, which takes precedence
(`get_fallback_model`); the switch is one-shot per run: a direct
name is matched by its last path
segment with dots and dashes treated alike, an effort alias keeps its
level when OpenRouter's side has the matching generated alias and
falls back to the bare twin otherwise.  A refusal, a content filter,
an output cap or empty responses, which the twin shares with its
direct model, fail over to another family instead
(`model_families.fallback_of_another_family`: the declared fallback
when it is of another family, else the strongest configured model of
another family, else the declared fallback or the twin as the last
resort, same family or not; see "Error handling").  The switch (the
fallback's or `set_model`'s; both go through `KISSAgent.switch_model`,
and `set_model` switches the live executor, `_current_executor`,
the session running at the time, and with no live executor only
records the name for the next session, `Model deferred-changed to
...`) hands the live
conversation over.  `set_model` copies the old adapter's
`model_config` onto the new one, minus the keys the target's
provider family cannot accept (`_sanitize_model_config_for_switch`)
and minus a `reasoning_effort` equal to the old model's own default
level (`_model_thinking_level`), so the new model gets its own
default while an effort that differs from that default is kept; a
switch to
a settings-panel custom model (a `MY_MODELS.json` entry with an
`endpoint`, `model_info.custom_model_config`) drops the run's
`base_url`, `api_key` and `extra_headers` and uses that entry's own,
as the task runner starts it, and a switch away from one drops them
and gives the new model its normal routing and credentials, so a
different model never inherits a custom entry's endpoint (nor does a
sub-agent, `agent_dispatch._parent_model_config`); otherwise a
custom `base_url` and its key carry over to the new
model, a standard provider endpoint carries over only when the new
model routes to the same provider default, and a Gemini-to-Gemini or
Claude-to-Claude switch (direct and Bedrock kept apart) keeps the
task's own key, so a per-task key is not silently replaced by the
process-global one.  The routing the switch computed is kept in the
agent's `model_config`, so the next session's executor (a
continuation, the summarizer; `_executor_model_config`) runs the
switched model with that endpoint, key and headers rather than
sending its name to the launch endpoint (the executor's per-session
`work_dir` and `system_instruction` are kept only when the launch
config set them).  The conversation handed over
(`model.hand_off_conversation`, which also moves
the `llm_call_hook` boundary, the helper spend record and the
agent's limit check) is kept verbatim when both models talk
to the same endpoint (`base_url`), so an OpenAI model keeps another
OpenAI model's reasoning items, and converted to plain chat messages
(`responses_items_to_chat_messages`, which every adapter reads) across
endpoints, since Responses API items belong to the endpoint that
produced them (a Chat Completions model taking over an Anthropic
conversation gets each `tool_result` block as a `role="tool"`
message, and the images or PDFs nested in those results — which a
tool message cannot carry — in one user message that follows them:
the media parts first, then the text `[attachments from previous
tool result(s)]` (`TOOL_RESULT_ATTACHMENT_NOTE`), then whatever else
that user message held); and a hop between two Responses endpoints (Together's
`openai/gpt-oss-120b` back to an OpenAI model) strips from each
request, not from the conversation, what the other vendor issued —
a `reasoning` item without the endpoint's own id prefix is dropped
and any other such item loses its `id` — as the vendor's
`responses_item_id_prefixes` in `model_info` declare (`rs`, `fc`,
`msg` for OpenAI, which otherwise answers `Missing required
parameter: 'input[8].summary'`; the vendor is found by its host in
the base URL, compared without regard to case).  A hop into a Gemini
model sends each earlier function call the new model never signed —
another provider's, or an unsigned later call of a parallel batch —
with the documented dummy thought signature
(`gemini_model.FOREIGN_CALL_SIGNATURE`,
`skip_thought_signature_validator`), which Gemini 3 and 4 would
otherwise reject with a 400, while a Gemini-to-Gemini switch carries
the real signatures over; the media an earlier model was sent
(`gemini_model._media_block_to_part`: Anthropic `image` and
`document` blocks, OpenAI `image_url` and `file` data URLs, and
OpenAI `input_audio` parts, whose `format` name is mapped back to its
MIME type) become inline Gemini parts, and a block that cannot be
inlined, such as a remote URL, is dropped with a warning.  The usage line
appended to the messages carries over.  A `codex/*` turn reports one
usage sum for every request the turn made, so the context it leaves
is estimated between the per-request average and that sum from the
size of what the next turn sends, counted with GPT-5's tokenizer
(`CodexModel.context_tokens_from_response`; `tiktoken` over the
`o200k_base` ranks litellm ships, built once per process) rather than
a characters-per-token ratio that holds for neither prose nor JSON,
CJK or base64 — without the tokenizer the text's UTF-8 byte count,
an upper bound, stands in, with one warning in the log.  A `cc/*`
turn's context is the largest per-request token total of the turn's
`request_usages` (`ClaudeCodeModel.context_tokens_from_response`;
the largest request holds the latest state of the conversation), and
the turn's aggregate usage stands in when it lists no request
(`KISSAgent`: a `context_tokens_from_response` that returns no `int`
leaves the call's own token count as the context).

Long prompts are priced by one of two rules in
`model_info.calculate_cost`.  A catalog entry with `prompt_tiers`
(`openrouter/*` entries only: the Claude Sonnet 4.5 and Haiku 5.5
twins, Gemini 2.5 and 3.1 Pro, GPT-5.4 through GPT-6.1, Grok 4.x and
`grok-build`, Qwen Plus, Flash and Coder Flash, Seed 2.0 Lite and
Mini, Sakana Fugu Ultra, and the rolling `~*-latest` aliases that
track one of them — `~openai/gpt-*-latest`,
`~anthropic/claude-haiku-latest`, `~x-ai/grok-latest`; no
direct or `bedrock/` entry carries any) is priced by those tiers
alone.  Each tier is an
integer `min_prompt_tokens` threshold plus any of the seven per-1M
prices (`PROMPT_TIER_PRICE_FIELDS`: input, output, cache read, cache
write, 1-hour cache write, audio input, audio output); a single request whose prompt (input,
cache read, cache write and audio input tokens together) is strictly
longer than the threshold takes the tier's prices in place of the
entry's own, a later tier wins per price when several apply (Qwen
3.7 Flash steps at 32K and again at 256K), a price a tier omits
keeps the base value, and a tier that sets its own input price gets,
derived from that price, the same cache prices the load-time rules
derive for the base (`_with_derived_tier_prices`).  The field
is parsed leniently (`_parse_prompt_tiers`: a non-list value, or a
tier with an unknown key, no price field at all, a missing,
non-integer or non-positive threshold, or a price that is neither
zero nor a finite number from `1e-12` through `1e12` (`_is_price`:
a figure near either end of the float range would overflow the
derived costs to `inf`), is skipped with a warning, since the
catalog is read at import time); the tiers are mirrored onto the
generated `-<level>` aliases of a base that `MY_MODELS.json`
overrides (`_ALIAS_MIRRORED_FIELDS`), copied into a new registry entry seeded
from such a model (`_seed_new_custom_entry`) and scaled with the
other prices by a Bedrock profile's premium (`_scaled_prices`).
`update_models.py` imports them from each OpenRouter listing's
`pricing.overrides` (`openrouter_prompt_tiers`): an override whose
only condition is `min_prompt_tokens` becomes a tier with its
`prompt`, `completion`, `input_cache_read`, `input_cache_write`,
`input_cache_write_1h`, `audio` and `audio_output` prices converted
to per-1M; one with any other
condition (the `utc_*` time-of-day windows, or a key the importer
does not know) is skipped, so the base price applies to it, and so
is one with a non-numeric or negative price; on a later run a
changed tier list replaces the stored one and a list OpenRouter no
longer publishes is dropped from the entry (`_prompt_tier_changes`,
`_REMOVABLE_PRICE_FIELDS`).

An entry without tiers falls back to the name-based rule
(`_long_context_uplift`), which bills the whole request at the
family's multiples of its own short-context rates: Claude Sonnet 4.5
bills a single request whose prompt exceeds 200,000 tokens at 2x the
input and cache rates and 1.5x the output rate, Haiku 5.5 one over
100,000 at 5x every rate, GPT-5.4 through GPT-6.1 (not the 5.4 mini
and nano) one over 272,000 and Gemini 2.5 and 3.1 Pro one over
200,000 at 2x input and cache and 1.5x output.  Neither rule applies
to summed multi-request counts (`single_request=False`), which bill
every token at the short-context rates, and the name-based rule is
not applied to a decisions-only entry (`dec` without `gen`, such as
`openrouter/openai/gpt-6-luna-decisions`), whose flat listed price
is not the generation model's tiered one.  The name-based rule is
keyed on the model name after the Bedrock id, provider prefix,
OpenRouter `anthropic/` prefix and effort alias are stripped and
matched by prefix, so `claude-sonnet-4-5`, `claude-sonnet-4.5`, their
dated ids and the `bedrock/` profiles all carry the Sonnet 4.5 rule,
though their 200K context cannot reach the threshold; the OpenRouter
twin, `openrouter/anthropic/claude-sonnet-4.5`, whose 500K context
can, holds the same figures as a tier instead ($6 input, $22.50
output, $0.60 cache read, $7.50 and $12 cache writes).  Sonnet 4.6
and every other Claude model bill long prompts at their standard
rates.

`$KISS_HOME/MY_MODELS.json` is the custom-model registry the
settings panel's Add / Edit / Delete buttons maintain
(`model_info.save_custom_model`, `delete_custom_model`); an entry is
keyed by model name and holds the endpoint fields the panel edits
(`endpoint`, `api_key`, `headers`, each dropped when blank) plus the
loader-required `context_length`, `input_price_per_1M` and
`output_price_per_1M` (an entry missing one, or whose context length
or any price is not a finite, non-negative number — a string, a
boolean, a negative value, `NaN`, `1e309`, an integer beyond the
float range or a positive figure outside `1e-12` to `1e12`, which
would bill a negative or `NaN` cost or overflow
the cache-price derivation — is skipped with a warning at load,
`_require_number`; the panel never writes such a value, so this
guards hand-edited files).  A brand-new entry is seeded
(`_seed_new_custom_entry`) from the catalog model its name resolves
to — the exact name, a harbor-prefixed form such as
`openai/gpt-6.1-sol`, or a `bedrock/<id>` profile of a `claude-*`
entry, the one rule of `_lookup_model_info` — copying that model's
context length, its `thinking` level (without it the row would stop
requesting the model's reasoning effort after the next start), its
capability flags (`fc`, `emb`, `gen`, `dec`: a dropped flag would turn
a decisions-only or embedding model into a generation one), its
`use_responses_api` transport flag (dropped, the entry and, through
`_sync_alias_transport_flags`, its aliases would lose the Responses
transport), its `fallback`, its `extended_thinking` and
`adaptive_thinking` overrides, and
every per-million price field
(`_PER_1M_PRICE_FIELDS`: input, output, cache read, cache write, 1-hour
cache write, audio input, audio output, image output) and its
`prompt_tiers`, if any (a Bedrock profile's prices, tiers included,
are scaled by `bedrock_price_factor`), so the entry
that replaces the catalog one at the next load bills every token
kind as the catalog did; a name no catalog model matches gets the
generic defaults (128,000 context, $0 prices,
`_CUSTOM_MODEL_DEFAULTS`).  An edit merges into the existing entry,
so hand-written keys such as `thinking` or real prices survive an
endpoint change; a name not yet in the loaded catalog is looked up
in the file as it is now, so a model saved in the panel is priced by
the running daemon without a restart.  An entry named after a
generated effort alias (`gpt-5.6-sol-xhigh` given its own endpoint)
keeps the catalog's `alias_of` marker when it loads, since the panel
writes none, and the alias's `thinking` level unless the entry
carries its own, so its requests still go
out under the base id (`_strip_thinking_alias`) at the alias's effort
while its own context length and prices stay in force (a catalog
entry of that name which is not a mapping, a hand-edited list or
string, contributes nothing and does not stop the user entry from
loading); a `-xhigh` name that is an entry of its
own, with no marker (`_is_registered_model_id`, which reads the file
live, so a router model such as `s10s-scout-xhigh` added after
start-up counts), is kept whole, since that is the name its endpoint
expects — the one case where `-xhigh` is not stripped.  A custom
entry may keep a
harbor-style provider prefix in its name: the model factory strips a
redundant `openai/`, `anthropic/` or `google/` prefix only when the
bare name is a known model or the full name is not, so
`openai/gpt-5-custom` registered in `MY_MODELS.json` is routed and
priced under that full name rather than as an unknown
`gpt-5-custom`; and an entry registered under such a prefixed name
WITH an endpoint (`openai/gpt-6-astra` for a gateway that uses that
id) keeps its full name whenever the call routes to an endpoint,
even though the bare `gpt-6-astra` is a bundled model: the gateway
receives the id it was registered under and the entry's own prices
bill the call, hot-added or loaded (`_is_registered_endpoint_model`,
`_lookup_model_info`).  A generated `-{level}` thinking alias of a
registered endpoint model (`gpt-6-astra-high` when only
`gpt-6-astra` is registered) runs against the base's endpoint, key
and headers (`custom_model_config` resolves the alias to its base
unless the alias has an entry of its own), since the loader mirrors
the base's prices onto the alias.


## Complete working example

Below is a self-contained SEA that gives the LLM tools for
managing a SQLite task database.  Its `tools()` adds the database
tools to the built-in toolset, so the LLM can also use `Bash`, `Read`,
`Write`, etc. alongside them.

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

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.core.config import kiss_home


# --- Database setup ---
# Guarded with CREATE IF NOT EXISTS: the file is re-executed from
# source on every run.

_DB_PATH = kiss_home() / "task_manager.db"
_lock = threading.Lock()


@contextlib.contextmanager
def _get_db() -> Iterator[sqlite3.Connection]:
    """Yield a connection to the task database, creating it if needed.

    The block's changes are committed when it succeeds and rolled back
    when it raises (a locked database), and the connection is closed
    either way: a failed call leaves no write transaction open to lock
    out the next one.
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(_DB_PATH)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        yield conn


# --- Tool functions ---

def add_task(title: str) -> str:
    """Add a new task to the database.

    Args:
        title: Short description of the task.
    """
    with _lock, _get_db() as conn:
        conn.execute("INSERT INTO tasks (title) VALUES (?)", (title,))
        task_id = conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]
    return json.dumps({"added": {"id": task_id, "title": title}})


def list_tasks(include_done: bool = False) -> str:
    """List tasks from the database.

    Args:
        include_done: If True, include completed tasks.
    """
    with _lock, _get_db() as conn:
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
    with _lock, _get_db() as conn:
        cursor = conn.execute(
            "UPDATE tasks SET done = 1 WHERE id = ? AND done = 0",
            (task_id,),
        )
        updated = cursor.rowcount
    if updated:
        return json.dumps({"completed": task_id})
    return json.dumps({"error": f"Task {task_id} not found or already done"})


# --- The SEA ---
# No prompt(task) method — the caller's prompt is used as-is.
# No "model" key — the caller's model is used.

class TaskManagerSea(BaseSea):
    def description(self) -> str:
        """Return the one-sentence help text shown by ``/task_manager help``."""
        return (
            "Manages a personal SQLite task list (add, list, complete) through three "
            "tools; use it as `/task_manager <request>` or pass its path as "
            "sea_path."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"max_budget": 1.0}

    def system_prompt(self, system_prompt: str) -> str:
        return (
            "You manage a personal task list stored in a SQLite database.  "
            "Use the add_task, list_tasks, and complete_task tools to "
            "manipulate the database.  Always call list_tasks after "
            "modifications so the user sees the updated state.  "
            "You also have the standard KISS Sorcar tools (Bash, Read, "
            "Write, etc.) if you need them."
        )

    def tools(self, tools: list[Any]) -> list[Any]:
        """The database tools, added to the built-in toolset."""
        return tools + [add_task, list_tasks, complete_task]
```

Launch it:

```python
from kiss.server import sorcar

# First run — add some tasks (the prompt reaches the LLM directly)
result = sorcar.run(
    "Add three tasks: buy groceries, review PR #42, write tests",
    sea_path="task_manager/task_manager_sea.py",
)
print(result.text)
print(f"Cost: ${result.cost:.4f}, Steps: {result.steps}")

# Follow-up in the same chat — the agent remembers context
result2 = sorcar.run(
    "Complete 'buy groceries' and show me remaining tasks",
    chat_id=result.chat_id,
    sea_path="task_manager/task_manager_sea.py",
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
    cron_job: bool = False,
    model: str = "",
    chat_id: str = "",
    system_prompt: str = "",
    sea_path: str = "",
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    use_web_tools: bool | None = None,
    auto_classify: bool | None = None,
    use_memory: bool | None = None,
    compaction: bool | None = None,
    is_parallel: bool = True,
    add_to_system_prompt: str = "",
    add_to_prompt: str = "",
    tool_profile: str = "",
    docker_image: str = "",
    inherit_tools: bool = False,
    isolated_repos: dict[str, str] | None = None,
    workspace: str = "",
    provenance: dict[str, str] | None = None,
    timeout: float | None = 3600.0,
    stop_on_timeout: bool = False,
    record_timeout: float | None = None,
    endpoint_file: str | Path | None = None,
    cancel: threading.Event | None = None,
    running: threading.Event | None = None,
    machine: str = "",
    sea_command: str = "",
    attachments: list[dict[str, str]] | None = None,
    budget_cap: float | None = None,
    reconnect_seconds: float = 300.0,
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
    settings: dict[str, Any] = field(default_factory=dict)  # the run's task_settings event payload
    reconnects: int = 0  # daemon connections lost mid-run and re-established
```

`settings` is the run's `task_settings` event (its effective model,
work directory, budget, SEA, kind, tool profile, timeout, inherited
and overridden values; `kiss.agents.sorcar.run_config`), empty when
the run ended before the daemon emitted it.

`budget_cap` is the bound a `run_agent` dispatch puts on the run's
budget — the caller's unreserved remainder and no more than the job's
reservation (`agent_dispatch.budget_cap_of`; wire field `budgetCap`,
sent only when not `None`); the daemon lowers the effective
`maxBudget` to it after the SEA's settings apply.

What `run()` raises (all in `kiss.agents.sorcar.daemon_client`):
`ValueError` for an empty prompt or a `sea_path` that is not an
existing `.py` file; `ConnectionError` when no daemon answers at the
endpoint or the `run` cannot be sent (no task was started), and its
subclass `ConnectionLostError` when the daemon drops the connection
after the `run` went out and before the task started, or while a
stop awaits its confirmation, or when a connection lost mid-run is
not re-established within `reconnect_seconds` (or the daemon, once
back, reports the task neither running nor finished), when it loses
its own connection to the registered `machine` running the task (a
terminal status marked `lost`) or when a stop cannot be written on
the broken connection even after one reconnect — the task may still
be running and spending.  A connection the daemon drops while the
task runs (after its `status running=true`, no stop pending) is
re-established with backoff — at once, then 1 s, 2 s, 4 s, ... up to
30 s between attempts (`_RECONNECT_FIRST_BACKOFF_SECONDS`,
`_RECONNECT_MAX_BACKOFF_SECONDS`) — for the `reconnect_seconds`
window (300 s by default): the client re-attaches to its tab with a
`resumeSession` flagged `recovery: true` — a transport recovery, not
a user asking for the tab, so the daemon restores this connection's
subscription and transcript but a tab the user closed while the run
went on stays closed on every surface — naming the run by its task
id or, before the first stamped event gave it one, by the
client-minted run token the `run` carried as `taskId`
(`client_run_token`), so the daemon replays this run and not the
chat's latest row; the daemon replays the tab's events, so a
`result` or terminal status published meanwhile is still delivered
and the wait goes on (`TaskResult.reconnects` counts the reconnects;
the replayed transcript restarts the spend totals, so nothing is
counted twice).  A web app closing the run's `api-…` tab while the
task runs hides it (the tab leaves every surface) but does not end
it: the daemon keeps the task's subscription of the tab for the local
client that addressed it (`AgentServer._headless_client_task`,
`WebPrinter.local_client_addressed`), marked headless so its
copies reach no web app and the tab counts as no viewer, and the
client still receives the `result`; a recovery re-attach leaves that
tab closed as the user left it.  The window opens at the first drop and closes once
the run is replayed; `timeout` keeps running through it: a timeout
that expires while the client is still trying ends the wait on time,
as it would on a live connection (the attempts stop; with
`stop_on_timeout` the stop is still sent after one reconnect of its
own, within the same window); a run the daemon has not named yet (no
chat or task id announced) is not resumed, and `reconnect_seconds=0`
raises `ConnectionLostError` at the first close as before;
`TimeoutError` when the task outlives
`timeout` (the client disconnects and the task keeps running; its
tab is closed, `closeTab`, only once the task is known to be over —
its terminal `status` arrived (the run returned or its stop was
confirmed; a `result` alone is no proof, since the daemon broadcasts
it before the task thread persists and cleans up, and a session
handoff's `finish(success=False, is_continue=True)` result is
answered by the executor with the next session while the task goes
on, so the client keeps waiting) or the run was refused
(`daemon_client._run_ended`) — so a task that may still run keeps
its tab on every surface until a user closes it, while a sub-agent
run's tab is closed on every exit since no surface shows it), or, with
`stop_on_timeout`, its subclasses `StoppedOnTimeoutError` (the daemon
confirmed the stop; `.result` carries the stopped task's spend) and
`StopUnconfirmedTimeoutError` (the stop was sent but never confirmed
within the 20 s grace, so the task may still be running; `.result`
carries what it reported so far); and `RunRefusedError` when the
daemon turns the `run` down with an `error` event of `code:
"prompt_refused"` (a self-update being installed is the case an API
client can hit; so is `Cannot start task: the tab limit is
reached.`, a run whose tab the shared registry would not register
because `tab_registry._MAX_TABS`, 512, are open, the admitted task
unregistered again before it starts): nothing ran or was spent, so the wait ends at once
with the daemon's reason instead of lasting the whole `timeout`;
and `CancelledError` when the caller sets the `cancel` event: the
client sends the `stop` and waits for the task's terminal status
within the same 20 s grace, `.confirmed` tells whether it arrived
and `.result` carries the spend reported so far (no second `stop`
is cascaded for it, nor for the timeout and refusal errors above; a
`cancel` set while the client sleeps out a reconnect back-off ends
the pause at once, `_wait_interruptible`, and the stop goes through
the stop path's own reconnect).  Whatever the error, the sub-task's
`task_history` row id is recorded on the job as `AgentJob.task_id`
(`agent_dispatch._attribute_dispatch_usage`), so a stop the daemon
never confirmed or a lost connection still names the task that may
be running (a cron prompt job, which runs its prompt as a `run_agent`
job it drives itself — `plan_run_agent`, `new_agent_job`,
`launch_agent_job`, `join_agent_job`, not the tool's text, which
never carries the id of a run that ended with an error — and kills
the still-running job at its timeout, raises it as
`cron_agent.UnresolvedRunError.task_id` for a stop never confirmed
and for a lost connection alike, keeps it on the stored job as
`running_task_id` even when the job was edited while it ran, and is
not launched again by a tick or a "run now" while that row is
unfinished, `_claimed_task_id`).
Every other abort of the wait — above all the `KeyboardInterrupt`
injected when the calling task is stopped inside a `run_agent`
dispatch — also sends the daemon a `stop` for the dispatched task's
tab, and the exception carries a `task_result` attribute with the
spend reported so far, so the caller can still charge it.

`cost` is the exact
figure the daemon's final `result` event carries as `cost_usd` next
to its four-decimal `cost` display string (`kiss.core.printer.event_cost`;
a later `usage_info` event, say a side channel's spend or a finish
gate review's that lands after the row was saved — a review a Stop
left running is charged to the finished row of the run that asked
for it, `SorcarAgent._charge_gate_usage` —
updates the totals, and a delta the daemon
has already banked on a waiting ancestor, `ancestor_charged`, is not
folded in twice — a keyed charge's delta once per `charge_key`
(`daemon_client._net_totals`), since a keyed charge that a Stop cut
between its commit and its publication is published again by its
retry, which finds the row's receipt and charges nothing more but
re-sends the totals the rows' latest usage events still miss
(`persistence._unpublished_keyed_charge`); a remote sub-agent's late spend reported while the
caller's row ran but no live agent held it — a bridge-owned run is
unregistered before its final save, a run outside the daemon is never
registered — is banked on the caller's ledger at that final save,
`ChatSorcarAgent._bank_pending_remote_charges`, as the live path
`task_update._bank_remote_charge` banks one, each charge counted once
by its key, and the final totals are then persisted as the row's
latest `usage_info` event before the row is finished); the daemon persists the task's final cost unrounded
as well, so a parent row sums its children's exact spend.  The same
write records `model_costs`, the task's own `llm_call` events summed
per model as `{"model": [cost, tokens]}`
(`persistence._own_model_costs`; a model call made for the task
outside its agent loop — a finish-gate review attempt, the commit
message, a `talk` synthesis, a `decide` call, a `/gepa` reflection or
a `/skillopt` analyst call (charged as a side call of the task like
`decide`, `side_call_binding` and `_attribute_side_call_usage`, filed
under the reflection or optimizer model and counted as no step), the pre-run classifier
(its `decide` call on the decisions model, or the LLM fallback on the
run's model: `ClassifierRun.charges`, filed per model when the fold
banks the spend, `SorcarAgent._file_classifier_charges`, idempotent
by key so a fold retried after a Stop records each once) and a vision
bridge description made for a model that cannot see (billed into the
totals by `KISSAgent._bill_helper_usage` and filed under the vision
model through the run's `helper_charge_hook`,
`SorcarAgent._file_helper_charge`) — emits no `llm_call`,
and the `gate_stage` record with `stage` `charge` that each such
charge leaves, `persistence._record_model_charge` with its `model`,
`cost_usd` and `tokens`, is summed under that model the same way; a
row already finished when the charge lands has its split refreshed
then, and a row the orphan sweep finishes for a process that died
mid-run, `persistence._backfill_orphan_progress`, gets the same
record from its persisted events, its totals those of the newest
parseable `usage_info` event: a text-less ledger publication (the
bank of a side channel, a gate charge or a sub-agent's fold, the
run's closing event, a late charge's), whose structured fields are
the task's totals after the bank, or a per-step one whose text
carries `Steps:` (`persistence._recovered_progress_from_events`
scans the events backward and skips the live monitor's cross-task
usage events, foreign-format and malformed ones; its end time is the
newest persisted event of any type) — a bank publication, stamped
`banked` with the ledger, holds exactly the remote charges banked
before it was published, `_charges_held_by_usage_event` (an unstamped
one from the intermediate, unreleased build before the stamp is read
conservatively from its totals and can lose a cost-only charge once,
which the recovery logs), the others settled on top — and its `cost`
then raised to at least what its records show
(`_floor_orphan_costs_at_records`): its own `llm_call` and charge
records plus its sub-agents' spend, a local sub-agent's row cost and
a remote one's folded `remote_subtasks` record, the rows floored
deepest first so a parent counts the floored cost of a sub-agent
recovered with it, the floor never lowering a cost (it covers a
terminal `s10s -t` run that printed no usage event and the local
sub-agents that died with the process; a record replayed from the
event journal after the sweep floors the killed row and its killed
ancestors again, `_refloor_killed_chain`), so the sidebar's Spend subpanel
files a run that switched models with `set_model` under each model it
actually called rather than under its launch model
(`persistence._split_own_share`: spend no call accounts for stays
with the row's model, calls that add up to more than the row's own
share are scaled down to fit, and a row without the record is
credited to its model); the launch model's provider prefix
(`anthropic/claude-sonnet-4-6`) is stripped there so its spend and
task count merge with the bare name the `llm_call` events carry.


## Tips

- The client-side `prompt` argument must be **non-empty** even when
  the SEA's `prompt(task)` rewrites it; the client validates before
  connecting.
- `settings()` **may name any subset** of the keys, or be left out
  altogether.  Only pin the ones whose defaults you want to change;
  `WorkerSea` covers the common bundle.
- Omit the `model` key to keep the caller's model (the tab's pick, or
  the daemon's configured default) rather than hard-coding one.
- Extra tools come only from the `tools(tools)` method, returning a
  **list of callables**; `tools + [yours]` (the built-in toolset plus
  yours) is the most common pattern, `"tool_profile": "none"` gives
  an exact tool set.
- The SEA is **re-imported from source** on every run.
  Edits take effect immediately without restarting the daemon.
- `max_budget` and `timeout` must be **finite** numbers, and `timeout`
  **positive**.  `NaN`, `±inf`, an overflowing value, or a `timeout`
  of zero or less raises `SeaError`.
- The file must define **exactly one** subclass of `BaseSea` of its
  own; an imported base class (`from ...sh_sea import ShSea`) does not
  count, a second name for the class (`Main = Sea`) is not a second
  class (`load_sea` keys the candidates by identity), and helper
  classes that do not derive from `BaseSea` are free.  A method is applied only when the class (or a base) defines
  it, so leave out what you do not change rather than returning the
  argument unchanged; never call `super()` in a chained method
  (`settings`, `prompt`, `system_prompt`, `tools`, the three hooks), the
  launcher already ran the base's version (`description()`,
  `register_as_model()` and `on_picked_as_model()` are ordinary Python
  inheritance, where `super()` is fine).  Module-level constants
  (`SYSTEM_PROMPT`, `DISPATCH_TIMEOUT_SECONDS`) and functions are
  ordinary Python: the daemon reads only the class.
- The SEA and its tools run **in the daemon process**
  with the daemon user's privileges and environment.  Any libraries
  your code imports must be installed in the daemon's Python
  environment.  A tool that runs on the task's worker thread can call
  `kiss.server.agent_state.current_agent()` to get the running agent
  (its `work_dir`, model and usage counters); it returns `None` on any
  other thread.  The bundled `seas/autorouter/autorouter_sea.py`,
  `seas/gepa/gepa_sea.py`, `seas/rsi7d/rsi7d_sea.py` and
  `seas/skillopt/skillopt_sea.py` use it.
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
  earlier ones, the bundled Sorcar-extending SEAs in
  `src/kiss/agents/seas/` come next, so a `SEAS.md` folder can shadow
  them, and the built-in `cron` (`sea_commands.BUILTIN_COMMANDS`) has
  the lowest precedence of all.  The daemon rebuilds the registry
  every two seconds (`sea_commands._WATCHER_POLL_SECONDS`) and on a
  lookup miss (`get_command`), so a new folder is a command within
  two seconds; a rescan reads the folders under no lock, so one stuck
  on a slow folder delays no other, and every rescan is numbered when
  it starts and published only if no later-started one was published
  first, so an older scan never replaces a newer registry
  (`_publish`; an unchanged list is not re-broadcast to the clients,
  except after a Stop interrupted a broadcast before every subscriber
  had it, when the next rescan broadcasts whatever it finds,
  `_forget_broadcast`).  Of the 32 bundled SEA folders (the
  `base/` folder of the base classes is never a command), 30 register
  a command and two (`coding`, `sorcar`) are hidden: `/ask` (answers a question about the current task
  in two or three sentences from one `task_context` call over its
  status, spend, progress log and digested persisted events — the log
  is rebuilt from the task's own persisted `Write` / `Edit` calls to
  its `PROGRESS.md` (`task_digest.progress_log_tail`; a call whose
  result was an error or that a Stop interrupted counts as having
  written nothing, and an `Edit` is replayed by `task_digest._replay`
  as an approximation of the tool: the first occurrence of the old
  text verbatim (every one with `replace_all`), else with CRLF line
  endings swapped, else, for a single replacement, a unique
  whitespace-tolerant match with the new text re-indented; when none
  of these finds a unique match, a nonempty new text is appended at
  the end instead, so the tool's refusals of an ambiguous or empty
  old text, and its `exact=True`, are not reproduced by the replay),
  not read from
  disk, since a work dir shared by several tasks may hold another
  task's text in the same file; its only
  tools are `task_context` and `finish`, so it never touches the task's
  files or shell; typed into a running task's
  tab it runs as a side channel that always dispatches the bundled
  `seas/ask/ask_sea.py`, even when a `SEAS.md` folder shadows the
  command, with the files attached to the question forwarded as the
  answering run's `attachments`; its `ask_answer` is stamped with the
  task's persisted row id alone (`commands._broadcast_ask_answer`), so
  it reaches the tabs showing that task when it lands — the asking tab
  may have been closed and the task reopened in another meanwhile —
  and is filed under that row even when the task has finished and its
  agent is gone (`JsonPrinter._persistence_task_id` files an
  `ask_answer` under its own stamp), while an answer with no task row
  to file it under is stamped with the asking `tabId` instead and
  shown there once, never replayed, and the task-info panel's Task update is the same agent
  run in-process by `kiss.server.task_update`: it asks the SEA
  `task_update.UPDATE_QUESTION` ("What has this task done so far, and
  what are its partial results?") on the parent's model (unless the ask
  script's `settings()` names a `model`) in the parent's work dir, with the
  `task_context` tool and `finish` only, capped at
  `UPDATE_BUDGET_USD` ($1) per run and charged to the task as
  side-channel usage; it runs on its own schedule while a surface shows
  the panel — the first run once the task is `FIRST_UPDATE_DELAY_S`
  (60 s) old, then whenever a poll finds the last answer older than
  `UPDATE_INTERVAL_S` (10 min), or at once on the panel's refresh
  button — and never twice at once; the child runs under a stop event
  of its own that observes the task's (`task_update._ChildStopEvent`,
  re-checking the parent's every 0.1 s while blocked), so Stop on the
  task's tab ends the update together with the task, the panel
  reporting "The task was stopped before this update finished.",
  while a Stop on the update's own nested tab ends only the update
  and the task and its other children keep running),
  `/autorouter` (runs a task on the cheapest model tier
  that will finish it; its `settings()` fixes the routing session's model to
  `autorouter_sea.ROUTER_MODEL`, `claude-opus-5-5-medium`, with
  `auto_classify` off and `use_web_tools` and `use_memory` on; the
  candidates of each tier are the ordered `(model, note)` pairs of
  `autorouter_sea.TIERS` (small: `zai-org/GLM-5.3-Flash`,
  `gpt-6-luna-xhigh`, `deepseek-ai/DeepSeek-V4.1-Flash`,
  `openrouter/xiaomi/mimo-v2.6-flash`, `openrouter/qwen/qwen3.8-flash`,
  `claude-haiku-5-5-max`; medium: `openrouter/meta/muse-wonder`,
  `gemini-4-argon`, `openrouter/~z-ai/glm-latest`,
  `openrouter/deepseek/deepseek-v4.1-flash-1017`, `openrouter/x-ai/grok-4.7`,
  `gpt-6.1-sol-high`, `claude-sonnet-5-5-max`,
  `openrouter/moonshotai/kimi-k3.1-pro`; frontier: `gpt-6-astra`,
  `claude-fable-5-2-max`, `claude-opus-5-5-max`; most direct entries are
  followed by their `openrouter/...` twin), and `pick_model` returns the
  first candidate whose provider key is configured, skipping the models
  named in its `exclude` argument; its
  `log_decision` tool appends every dispatch and outcome as one row of
  the installation-wide ledger `$KISS_HOME/MODEL_DECISIONS.md`, created
  whole once and appended under an exclusive file lock held for that
  one line, so the sub-agents of one run cannot overwrite each other's
  rows, Windows's seek-then-write emulation of `O_APPEND` included), `/bestrouter` (its
  `settings()` starts every task of the tab on `claude-opus-5-5-medium`
  (`bestrouter_sea.SIMPLE_MODEL`), which answers simple questions and
  queries itself; its routing protocol (`SYSTEM_PROMPT`, shared with
  `orbestrouter` through `routing_prompt(router, simple, primary,
  review)`) tells the agent to use
  `claude-fable-5-2-max` (`PRIMARY_MODEL`, a `set_model` call away) for
  every other task, software development, analysis and writing included, and to have
  `gpt-6-astra-xhigh` (`REVIEW_MODEL`) review it read-only, dispatched
  with a non-blocking `run_agent(model=..., tool_profile="review",
  wait="false")` on at most 75% of the budget and collected with
  `agent_job`), `/orautorouter` and `/orbestrouter`
  (the same two routers with every candidate taken from the OpenRouter vendor:
  `orautorouter_sea.ROUTER_MODEL` is `openrouter/anthropic/claude-opus-5.5-medium`
  and its `TIERS` hold the `openrouter/...` twins in the autorouter's
  order, and `orbestrouter` starts on
  `openrouter/anthropic/claude-opus-5.5-medium`, works on
  `openrouter/anthropic/claude-fable-5.2-max` and is reviewed by
  `openrouter/openai/gpt-6-astra-xhigh`), `/ccxbestrouter` (the bestrouter on
  the Claude Code and Codex CLIs, so a Claude subscription and a ChatGPT
  subscription run it without API tokens: `ccxbestrouter_sea.SIMPLE_MODEL`
  is `cc/claude-opus-5-5-medium`, `PRIMARY_MODEL` `cc/claude-fable-5-2-max`
  and `REVIEW_MODEL` `codex/gpt-6-astra-xhigh`; its `SYSTEM_PROMPT` is the
  bestrouter's `routing_prompt(...)` with the "(not codex)" warning
  dropped, since the reviewer is the Codex model on purpose), `/brbestrouter`
  (the bestrouter on Amazon Bedrock, so one set of AWS credentials runs it:
  `brbestrouter_sea.SIMPLE_MODEL` is
  `bedrock/global.anthropic.claude-opus-5-5-medium`, `PRIMARY_MODEL`
  `bedrock/global.anthropic.claude-opus-5-5-max` and `REVIEW_MODEL`
  `bedrock/global.anthropic.claude-fable-5-1-max`; its `SYSTEM_PROMPT` drops
  the "(not codex)" warning too, which means nothing for a Claude reviewer), the hidden
  `coding` SEA (unattended coding in a Docker container, or on a
  remote sandbox through an `exec:<url>` bridge, for benchmark
  trials; its `settings()` is `{"hidden": True}`, so there is no
  `/coding` command: the trial runners load the generated per-trial
  SEAs, which expose its `ContainerHarness` methods, by path.  Its
  `tool_call_hook` is the model of the coding gate every Sorcar run
  with the edit tools now has (`coding_gate.CodingGate`, see
  "Refusals" under "Error handling"; the harness keeps its own,
  container-bound copy of the stages and shares the readers of shell
  results, `evidence_commands`, `gate_refusal`, `verifier_decides`,
  `call_stop_event`, `default_effort` and the verdict texts
  with it): with the trial config's `finish_gate` on, every
  `finish(success=True)` runs its checks at once in threads and is
  refused with every failed one together under the gate's round rule
  (inline in `ContainerHarness.finish_verdict`: the first refusal
  buys one round whatever refused it, a later round is refused only
  when a script — the verifier, or the independent check when one is
  configured — refused at that call, and no round past
  `max_gate_rounds` (3) per trial is refused), the finish otherwise
  delivered with the open verdicts appended as caveats (logged as
  `delivered_with_findings`, and as `unverified_finish` when the
  checks did not pass).  The checks: the agent's own verifier script
  (`/tmp/kiss-verify.sh` in the container must exist and exit 0
  within `verifier_timeout`, 900 s by default; the script is read once
  for the run and that text is what runs, while the concurrent review
  reads `/tmp/kiss-verify.sh` as it is when the review starts, so the
  reviewer can see a script edited since; the agent's own run of it is
  reused instead when it passed and nothing changed since — the
  harness counts the changes of the work, `ContainerHarness.changes`,
  advanced by `track_change` for an `Edit` / `Write` under the workdir
  (`under_workdir`), a shell command that writes files
  (`evidence_gate.writes_files`) and a `rollback`, and
  `record_exit_codes` sets `verifier_passed_at` to the change count
  at a foreground shell call whose exit 0, read by the gate's
  `verifier_decides`, was the verifier's — its `turn_commands` are
  `(name, args, changes)` triples for that —
  while an edit of the verifier or a shell command naming it makes
  that run stale; `verifier_verdict` then runs nothing, sets
  `last_verifier_run` to the script, exit 0 and the note "passed when
  you ran it; nothing changed since", logs the `verifier` event with
  `reused: true` and records the stage's `reused` field; the gate's
  own last run of a script is reused the same way, `script_run`,
  while neither the files under the workdir — a snapshot taken as
  that run started, `file_snapshot` — nor the script's text changed
  since, whatever its exit code was, the output then ending in the
  `REUSED_NOTE`, "This is the result of the gate's last run of
  <path>, reused: neither a file under <workdir> nor the script
  changed since. ..."), an
  independent check written once per trial by the config's
  `blind_verifier` model (`""`, the harness's default, for none —
  the launcher's default too, since its verdicts did not predict the
  hidden tests and it cost a sixth of the run; `"same"` for the
  trial's own model, `"other"` for the strongest configured model of
  another family, `model_families.other_family_model`, or a model
  name, `resolve_side_model`) from the task statement alone,
  without seeing the agent's work (authored with a shell in the
  container, 40 steps and a tenth of the trial's budget, told to keep
  the script within a quarter of `verifier_timeout`; an author whose
  run ends without a script by its own doing, answering in text or
  spending its steps or budget, is retried once (`blind_retried`;
  the `blind_verifier_written` event of the second run says `retry:
  true`) on a model of another family, or on the configured author
  model again when this daemon pins every task to its model
  (`agent_dispatch.task_model_pinned`: the retry then switches to no
  model the configuration did not name), while an author run that
  fails with an error, no such model, no credential, a provider down,
  is not retried; the
  script is kept on the host and the container's copy at
  `/tmp/kiss-blind-verify.sh`, the agent's to read and run, is
  rewritten before each run, so an edit or deletion there changes
  nothing; a blind check its author produced nothing for is skipped,
  not refused; the two verifiers share one refusal count,
  `MAX_VERIFIER_REFUSALS` (3): each tests it before it runs and
  neither runs once it has reached three, so the finish then counts as
  unverified, but each counts its own failure after its run, so two
  checks that start together at two can leave it at four).  Both
  scripts run as the argument of `<shell> -c` (`run_script`), with a
  host-saved copy of the
  container's original bash (taken at the first model call, its
  SHA-256 logged as `shell_baseline`) copied into the container under
  a fresh private path right before each run and removed after it, so
  nothing the agent left at the shell's path or in a file stands
  between the copy and the exec; a bridged trial, or a copy that
  failed (logged `shell_copy_failed`), runs the container's shell.
  The `evidence` argument of the finish (a JSON list of commands, or
  one string; an empty field, or a parsed JSON value that is neither
  a string nor a list, is no evidence, and a non-empty string that is
  not JSON is one literal command) refuses nothing: the reviewer is
  shown each command named with the exit code of its last run as a
  shell command of this trial (`evidence_report`, from the exit codes
  the results showed; `never ran` for one that did not run), and the
  summary's wording is the reviewer's to judge as well.  (Two earlier
  gates refused on the finish call itself — the string comparison of
  the summary's quotes against tool output, `coding/quoted_evidence.py`,
  no longer wired in, then a refusal of a never-run or failing
  evidence command, of a summary that said the result was guessed or
  unverified, of a file change no shell command ran after, and of a
  summary the decisions model found leaving a case of the statement
  open; the measurement of 2026-11-04 found those refusals answered
  by a rewording far more often than by a check and no pass that
  depended on them.)  The last check is the review: the open findings
  of the trial's review that the summary answers neither with a
  change nor with a quoted command and its output
  (`unanswered_verdict`, as the decisions model judges,
  `answered_findings`, its spend logged as a `findings_answered`
  event with `cost_usd` and `tokens` and counted in the trial's cost
  like the reviewer's, `SIDE_AGENT_EVENTS`; without the decisions
  model the findings are
  unjudged and refuse nothing), or, while the trial's review is
  still to come, a read-only review by the configured `reviewer`
  model, `max_reviews` (`MAX_REVIEWS`, 1) per trial — a re-review,
  when the config allows one, sees its previous findings and only
  the files changed since — whose `NOT DONE` findings are the
  refusal.  The review starts with one model-call attempt at the
  provider's default effort (`review_call`: the trial's model overrides without their
  `reasoning_effort`, the gate's `default_effort`, since at the
  highest effort the reviewer thought for a median 399 s, past the
  prompt cache's life, and its verdicts were no better) with
  `REVIEW_DEADLINE` (240) seconds per attempt, through a stop event of
  its own that the gate's `call_stop_event` fires at the deadline or
  on the run's Stop; a review without a `VERDICT:` line, past the
  deadline, or whose model call raised is logged as failed, refuses
  nothing, keeps the previous findings open and does not use up a
  review (one that came back empty, cut by the output cap, refused or
  past the deadline is first retried once, at low reasoning effort
  when the model's catalog entry has a `thinking` level and unchanged
  otherwise, the gate's `retryable_review_failure` and `low_effort`),
  and the
  second failed review (`MAX_FAILED_REVIEWS`, 2) ends
  the reviewing for the trial.  The `reviewer` setting takes `"same"`
  and `"other"` like `blind_verifier` (`resolve_side_model`), and a
  review is run only when a file under the workdir changed since the
  previous one (a finish that changed nothing is judged on that
  review's findings instead; one that repeats the claim the review
  judged — the same summary and evidence, `claim_text` — on the same
  files gets that review's verdict again without a model call,
  `reused_review`, the stage recorded as reused).  A `finish(success=False)` is checked the same
  way (`giveup_verdict`, logged as `giveup_check`): both verifiers run
  at once without counting a refusal, their outcome is appended to
  the summary next to the agent's verdict (`GIVEUP_NOTE`), a tree
  that passes both is checkpointed as verified so the runner submits
  it, and the first give-up whose checks pass is answered once with
  `GIVEUP_RECONSIDER_VERDICT` (name the unmet requirement, or finish
  with success); the next ends the run.  As soon as
  the verifiers that ran pass, the work tree is archived in the
  container as the `verified` checkpoint (`/tmp/kiss-checkpoints/`,
  at most 4096 MB), its SHA-256 recorded in the `checkpoint` event
  and host copies of the scripts that passed kept next to the
  trajectory (`verified.kiss-verify.sh`,
  `verified.kiss-blind-verify.sh`); when the run ends without an
  accepted finish the trial runner (`submit_best_checkpoint`) re-runs
  those host copies with the original shell and, if the tree on disk
  fails them, restores the checkpoint, provided the archive still has
  the recorded digest (an archive replaced or deleted in the container
  restores nothing, as for `rollback`).  With the
  gate on, the first `Edit` or `Write` under the workdir is refused
  (at most twice, `MAX_LEDGER_REFUSALS`) until the agent has written
  its interpretation ledger, `/tmp/kiss-ledger.md` (`ledger_verdict`),
  and then, when the config's `reading_check` names a model
  (`"other"` for the strongest configured model of another family
  than the trial's, `"same"`, a model name; `""` for none, and none
  either when `"other"` finds no other family configured, since a
  reader of the trial's own family shares its reading), until the
  ledger settles the independent reading of the statement
  (`reading_verdict`): when the ledger first exists the model reads
  the statement and the ledger once (`READING_CHECK_PROMPT`, one
  `review_call`, the ledger cut at `REVIEW_ONE_FILE_BYTES`) and
  names the sentences it would read differently, parsed with
  `numbered_items` unless the answer is the single line `NONE`
  (`READING_NONE`) or the call failed, logged as a `reading_check`
  event (its spend added to the trial's cost: `SIDE_AGENT_EVENTS`,
  `review_usage`) and recorded as the `reading` gate stage (`pass`,
  `fail` with the number of `items`, or `error`); the items refuse the
  edit with `READING_VERDICT` (adopt the reading, or keep yours and
  write next to it the words of the statement that rule the other
  out) until the ledger's text changed since they were raised, at
  most `MAX_READING_REFUSALS` (2) times, after which they are
  dropped.  The system prompt names the reading only when a model is
  configured for it (`READING_NOTE` in the ledger step) and the
  independent check only when one is (`BLIND_NOTE` in the verifier
  step, `BLIND_MENTION` in the finish step); the Terminal-Bench
  launcher (`benchmarks/terminal_bench/harbor_agent.py`, its
  `SeaOptions`: `reviewer` and `reading_check` default to `other`,
  `blind_verifier` to `none`, `same` standing for the trial's model
  and `none` for `""`; `run.py` takes them as `--reviewer`,
  `--reading-check` and `--blind-verifier`, `harbor_command` passing
  each as an `--ak` option) follows those defaults.  A text-only turn, the implicit
  finish that consults the hook with no arguments, is vetoed once with
  a request for an explicit `finish`, and a second one ends the run
  (logged as `implicit_finish`); a `finish` delivered once the
  verifier refusals are spent, or past the round cap with its checks
  failing, is logged as `unverified_finish`; either
  way `ended_unchecked` is true for the run and the trial runner
  counts it as a failure, then `submit_best_checkpoint` re-runs the
  checks saved with the latest verified checkpoint against the
  current tree: a tree that still passes is kept, one that fails a
  saved check is replaced by the checkpoint when its archive digest
  still matches, and a trial with no verified checkpoint is left as it
  is (`restored` / `kept` / `none` in the metrics).  A context
  restart (a continuation session) gets the statement and an inventory
  the harness writes in a fixed form, `ContainerHarness.inventory` —
  the files changed since the run started, the verifier's last verdict
  (its first `VERDICT_LINES`, 20, lines of output), the ledger's open
  checklist items and the running background jobs — rather than the
  last session's narrative; from the `restart_budget`-th restart on
  (`RESTART_BUDGET`, 4) a hand-off that comes with no new edit and no
  change of the verifier's verdict since the restart before it gets no
  session (`restart_denied`, raising `RestartBudgetError` at the
  session's first model call): the run ends as `finish(success=False)`
  with `RESTART_BUDGET_CAVEAT` ("Ended by the restart budget") and the
  state block appended.  A process a trial's shell leaves running is
  adopted as a container background job (`survivors_note`, through
  `ContainerJobs.adopt`, at most `MAX_LISTED_SURVIVORS`, 8, listed)
  and recorded in `background_jobs`, so a later `bash_job` wait on it
  reports an exit code the gate can read.
  Outside the gate, the harness's toolset drops `decide`
  and `summary` and adds `checkpoint(name)` / `rollback(name)`
  snapshots of the workdir, answers `ask_user_question`, `talk`,
  `run_agent` and `set_model` itself, gives `Bash` a 120 s default
  timeout (`DEFAULT_SHELL_TIMEOUT`) and refuses a command that would
  kill every process or delete the workdir (`DESTRUCTIVE_VERDICT`; a
  long foreground `sleep` and a here-document rewrite are the `Bash`
  tool's own guards, see "What the shell and file tools enforce")), `/devtools`
  (a `BaseSea` that drives a web app the way a developer does in
  DevTools: its `system_prompt()` appends `devtools_sea.SYSTEM_PROMPT`,
  the `## Browser DevTools workflow (devtools)` section — act, then
  verify the effect rather than the appearance with `wait_for`,
  `list_console_messages(clear=True)` and
  `list_network_requests(url_filter=..., clear=True)` before and after
  the action, `evaluate_script` recipes for inspecting or repairing
  page state, and the user's own Chrome through `attach_to_chrome` or
  `use_my_chrome` — once, skipping a prompt that already carries the
  heading; its `settings()` forces `use_web_tools: True` and nothing
  else, so the caller's model and tool profile stay; `/devtools <page
  or app, what to do or what is broken>` in the chat or
  `run_agent(agent="devtools", task=...)`), `/discover` (runs the AI discovery
  loop in a git repository (`git init` one around a lone artifact) so
  that every evaluated idea is a commit, each held by a ref
  `refs/discover/<task id>/<id>` that outlives the resets and the
  squash merge of the task branch: the dev / held-out split, the dev
  evaluation command and its noise floor (for a deterministic command
  bootstrap resamples of one run's per-item dev results, or 0 when it
  yields one number; metrics closer than that are equal) are fixed
  first in `<scratch dir>/split.md` and never change (the held-out
  part is off limits from the moment it exists; only the held-out
  check opens it — and, when the task names no held-out part, the
  split command, run before anything else reads the data — not even
  profiling), the Pareto frontier lives in `<scratch
  dir>/pareto.md` (one line per live frontier point: id, parent id,
  commit, idea, aspect, dev metrics; the committed baseline is point
  B), candidate ideas in `ideas.md` and every evaluated one in
  `explored-ideas.md` (one row appended per idea, in order, the
  baseline first and its status alone changed later in place, with
  the draw that led to it, its status and observations; the status
  lives there alone, an evicted point leaves `pareto.md`); the first idea
  comes from a web search for
  SOTA approaches, papers, repos and issues, the candidates appended
  to `ideas.md` with their aspect, parent frontier point and
  rationale and the unexplored ones judged pairwise for a winner; a
  round is `git reset --hard <parent commit>`,
  implement the idea, run the dev evaluation, commit, and log the
  status: `KEPT` (no frontier point dominates it; the points it
  dominates become `DOMINATED`), `TIE` or `FAILED` (neither is ever
  retried), and a frontier or `TIE` point that meets the goal on the
  dev metrics gets the held-out check once, passed when the held-out
  metrics meet the goal too and are no worse than the baseline's on
  the task's first-priority metric, `OVERFIT` and evicted when it fails (the baseline
  re-inserted, `KEPT` again, when that empties the frontier); the
  next operator is one random draw from a Bash call, recorded in
  `explored-ideas.md`: a variant of a frontier point, reasoned from
  its logged observations, for r < 0.33; a composition of two
  frontier points for 0.33 ≤ r < 0.66 (an eligible pair has neither
  the baseline nor a shared aspect, neither commit is an ancestor of
  the other and no logged idea already merged them; the compose is a
  `git merge` commit of the second into the one better on the task's
  first metric, which is the parent; when no pair is eligible, a fork
  of the oldest frontier point that is an ancestor of another, with
  an idea on an aspect that other point lacks, so the two can compose
  later; a fresh draw when there is no such point either); a fresh
  idea from an extensive internet search or `gpt-6-astra` or
  `gemini-4-argon` (a second draw at the same thresholds picks
  which), pairwise-judged against the other unexplored ideas, for
  r ≥ 0.66 (variant, fork and fresh ideas go to `ideas.md` too); the loop stops when a round passes
  the held-out check or when the budget or the round limit is hit
  (the task's, else 100 rounds), then `git reset --hard`s the working
  tree to the best point's commit (the one that passed the check,
  else the first frontier or `TIE` point in the task's metric
  priority, its first-named metric by default, that is no worse than
  the baseline held out on that metric and not dominated or equalled
  by it there, else the baseline) and reports it
  with its dev and held-out metrics, the status counts and the refs; a
  task that asks only for research, ideas or a ranking still gets the
  loop, ranked by measured dev metrics, and one that forbids
  changing its repository gets it in a lab (the artifact's current files
  copied to `<scratch dir>/lab`, `git init`, `create_worktree` on it)
  with the best point handed back as `<scratch dir>/best.patch`; its
  `description()`, which a task naming `/discover` sees, tells that
  task to run its improvement work through the SEA, again while the
  goal is unmet — on
  the `bestrouter` with the full toolset; the loop,
  `discover_sea.DISCOVERY_INSTRUCTIONS`, was the `## AI discovery, auto
  research, and optimization` section of `SYSTEM.md` and is now the
  SEA's alone: its `system_prompt()` appends it at the end of the
  assembled prompt, cutting out a leftover copy first, and its
  `settings()` is `{"model": "bestrouter", "tool_profile": "full"}`;
  `/revise_and_review_paper` and `/rsi7d` do not copy that text: the
  writer task of the first ends with `_DISCOVER_RULES`, the SEA's
  `description()` plus how to call `run_agent(agent="discover", ...)`
  with a task that ends in "Do not change this repository" and how to
  `git apply` the returned `best.patch`; the second's
  `improve_sea_with_discover` tool dispatches the `/discover` child
  itself, as an agent job of the rsi7d task in its own chat), `/forget`
  (removes a standing instruction from `$KISS_HOME/AGENTS.md`),
  `/gepa` (runs GEPA, the reflective text optimizer, as the loop of
  the official implementation: `gepa_sea.GEPA_INSTRUCTIONS`,
  pseudocode written from `gepa-ai/gepa` (main at
  `gepa_sea.OFFICIAL_COMMIT`) and no longer than the `/discover` loop,
  is appended to the assembled prompt by `system_prompt()`, once; the
  agent scores the seed texts on the validation set, then per
  iteration draws a candidate from the per-example Pareto front,
  evaluates it on a minibatch of 3 training examples with records
  (Inputs, Generated Outputs, Feedback), has one component (round
  robin per lineage) rewritten, and adds the rewrite, scored on the
  whole validation set, only when the minibatch total strictly
  improves; with `use_merge` it also merges two frontier candidates
  through their common ancestor.  The reflection LM is not the agent
  but its `reflect(text_file, records_file, new_text_file[, model])`
  tool, which `tools()` adds to the run: one fresh call of the task's
  reflection model (the run's own without one), with no tools and no
  system prompt, on the official reflection prompt
  (`gepa_sea.REFLECTION_PROMPT`, the current text plus every record of
  the minibatch whole), its reply read as the official parser reads it
  (`proposal`: a reply cut at the output limit proposes only a complete
  fenced block) and written to `new_text_file`, or `no proposal: ...`
  when the component has no records or the reply was refused, cut
  short or empty; the call's spend is charged to the task as a side
  call, filed under the reflection model in the Spend panel.  The
  agent never writes or edits a candidate text itself, and only the
  driver evaluates: everything but the reflection is a script the agent
  writes once from the pseudocode, `<scratch dir>/gepa/gepa.py`, over
  `state.json`, with one `log.jsonl` object per iteration, the
  minibatch records under `records/<i>.json` and the texts under
  `candidates/<k>/`; the script's seeded draws are the engine's own
  calls (`rng.choice` over the front list, `rng.shuffle`, Python's set
  order) and it must reproduce the `SELFTEST` answers of the
  pseudocode before the first evaluation; a trial of the evaluation
  command made to learn its output is not counted.  The run stops at the task's
  metric-call budget, `10 * |valset|` without one, and reports the
  candidate with the best validation average; `settings()` is
  `{"model": "bestrouter", "tool_profile": "full"}`
  (`gepa_sea.MODEL`, `TOOL_PROFILE`)),
  `/git_extract_knowledge` (builds and refreshes a repository's
  knowledge memory and can schedule its daily refresh), `/good_morning`
  (a `WorkerSea` whose only tools are `finish` and `stored_recommendations`:
  it greets you, lists the suggestions the last `/recommendations` run
  stored, never generating any itself, and asks what to do today; the
  8:00 default cron job, an `in_chat` job, submits `/good_morning now` as
  a typed message in a new chat), `/machine`
  (a `WorkerSea` whose only tools are `finish`, `machine_add`, `machine_remove` and
  `machine_list` over the registry in `$KISS_HOME/machines.json`
  (`kiss.agents.sorcar.machines`): `/machine add [name] <address> [password]`
  takes either `user@host[:port]`, an SSH login used once to enroll
  this daemon on the machine (one ssh login finds the machine's daemon
  home by its `sorcar-local.json` endpoint file, looking in
  `$KISS_HOME`, then the local brand's home directory, then every other
  `~/.*` directory there, so the machine may run another brand of the
  daemon; copies this daemon's local CA, `$KISS_HOME/tls/ca.pem`, into
  that home's `tls/trusted-clients/<host>-<fingerprint>.pem`, and reads
  back the daemon port and the machine's own CA; the machine is then
  registered at `https://<host>:<port>` with its CA pinned, and every
  later connection goes through one persistent `ssh -N -L` port-forward
  per login and daemon port (`machines.SshTunnel`, from a local loopback
  port to `127.0.0.1:<port>` on the machine, kept while the registry
  holds a machine at that login and port — `close_tunnels` sweeps the
  forwards of the endpoints a registry write dropped, so two daemons of
  one login have a forward each and a machine registered again after
  its daemon moved gets a forward to the new port — and reopened on
  the next connection once it is no longer alive; ssh
  is started with `ExitOnForwardFailure=yes`, `ClearAllForwardings=no`
  and `LocalCommand=echo kiss-forward-up`, which ssh runs only once the
  local port is bound, and that line on ssh's output, not the port
  answering, is what `open` waits for, and what `alive()` requires
  besides a running ssh process and a port that accepts: another
  program may take the port between `_free_port`'s pick and ssh's bind,
  and the link would then have handed its TLS handshake to a stranger;
  a port ssh reports it could not bind is given up for a freshly picked
  one, up to `FORWARD_ATTEMPTS` (3) ports within `SSH_TIMEOUT`, and an
  opening cut short never vouches for a port someone else may hold), so
  the daemon port need not be reachable from here, only the ssh port;
  ssh runs in a process group of its own, killed whole, so a
  `ProxyCommand` of the user's ssh configuration goes down with it
  instead of keeping ssh's output pipe open (the enrolling ssh too:
  one that has not answered within `SSH_TIMEOUT`, 30 s, or that a
  Stop cuts short is killed whole, `ProxyCommand` included, before
  `ssh <login> did not answer in time` is reported);
  through it this daemon presents its client certificate,
  `$KISS_HOME/tls/client.pem`, and the machine's daemon, which sniffs
  the first byte of every connection, serves the forwarded TLS like a
  direct one and admits the link as it admits a browser on its own
  loopback, whatever its remote password; the daemon picks a new
  `trusted-clients` file up within two seconds (`ENROLL_TIMEOUT`, 30 s,
  bounds the first handshake's retries), a daemon without a
  `trusted-clients` directory predates enrollment and is reported as
  such, and since the link arrives as a loopback peer, revoking the
  machine means removing its ssh access there (deleting the
  `trusted-clients` file only refuses the certificate on direct,
  non-loopback connections); the `ssh` client's own configuration
  applies, or `$KISS_SSH_CONFIG` names the config file to use, an alias
  of it resolves through `ssh -G`, the login must work non-interactively
  (`BatchMode`), allow TCP forwarding and land in a POSIX shell, and an
  unknown host key is accepted on first use), or `https://host:port` with the
  machine's remote password, which pins the remote `kiss-web`'s TLS
  certificate (`pin_certificate`, `PIN_TIMEOUT` 30 s in all, the lookup
  of the host's name included) and checks the password with one
  handshake before saving
  (for either form the pin is staged in a sibling file, written as the
  PEM's bytes so Windows gives it no CRLF line ends, verified by the
  first handshake and only then published together with the registry
  entry, so a wrong address or password changes nothing; a Stop or a
  registry-write error between the two puts the previous pin back, or
  removes the new one for a new name, so the entry the registry still
  holds keeps the certificate it was registered with, and the staging
  file is removed however the call ends);
  the name defaults to the host; `/machine remove <name>` forgets it,
  `/machine list` shows each machine's address and reachability; once a
  machine is registered the Chats panel lists its tasks next to the
  local ones, de-duplicated by task id, in the same order as the local
  list (running first, then latest finished first) and filterable by
  machine name (the merge, `machines.merged_history`, waits
  `HISTORY_FANOUT_WAIT` seconds (2) for the machines' pages, fetched on
  a shared pool with one fetch per registration and view at a time, a
  reply that arrives while one is under way joining it rather than
  starting another; a machine that has not answered by then
  contributes the page it last answered that view with, as one that
  fails does (`machines._last_page`, remembered per registration —
  name, URL, password and ssh login, `machines._registration` — and
  per view, the unfiltered one and the latest filtered one, a query
  or a tag, answered for that registration, so a filtered view whose
  fetch outlasts the wait still shows once it lands; an empty page for
  a view the machine as registered has never answered, hence for a
  name registered again at another address, login or password), and when
  the rows its fetch then brings differ from those served in its
  place, `AgentServer._history_landed_late`
  broadcasts `tasks_updated` so every page refreshes its rows; a task
  a `/sync_history_and_memory` run copied into this daemon's
  `history.db` is listed under the machine it ran on, not as a local
  one: `task_history.machine` holds the short host name of that
  machine (`''` for this database's own rows; `kiss.scripts.sync_db`
  stamps the source's rows with its host on extract, keeps a third
  machine's name, and clears the stamp on rows that ran on the
  target's own host), the page carries it as each row's `origin`, and
  `machines.label_synced_rows` turns an origin that is a registered
  machine into the row's `machine` and `machine_url`, as
  `fetch_history` stamps the rows it brings, so the panel shows and
  resumes it there (`machine_on_host` matches the host name the
  machine's daemon reported with its pings — kept across restarts in
  `$KISS_HOME/machine_hosts.json`, `note_host`, which rewrites the
  file under `machine_hosts.lock` from the names it holds plus the new one
  (`HOSTS_LOCK_TIMEOUT` 10 s), never from one process's whole
  mapping, since the fan-out workers, the sidebar's machine rows and a
  `/sync_history_and_memory` process note names at once, and a name
  whose save failed is kept in memory and saved at its next
  observation, and also noted by the
  sync's own ssh probe of each peer's host name, so a machine whose
  daemon never answered maps too — else the machine's name or a host
  of its address; a row of an unregistered host stays this daemon's);
  the local page of the Local filter or of a `workDir`-scoped query
  holds this machine's own rows only (`_load_history(own_only=True)`:
  unstamped rows), so two synced daemons never echo each other's
  copies through the fan-out, and once per database a background
  thread started at daemon start (`persistence._backfill_task_machines`,
  marker file `task-machines-backfilled` next to `history.db`) stamps
  the rows a sync copied in before the column existed from their
  first events, the `task_settings` event's `host` or the system
  prompt's `Machine info:` line (`history_columns.task_origins`); the
  favourite star and delete button of a remote machine's row, and the
  rename pencil and delete button of a chat whose rows are all that
  machine's, act on that machine's `history.db`: `setFavorite`,
  `setChatSummary`, `deleteTask` and `deleteChat` stamped `machine`
  (`sidebar_panels.MACHINE_HISTORY_WRITES`, `machine_history_write`)
  are forwarded on the machine's default link lane, in the order
  clicked per connection and machine, and followed by a
  `tasks_updated` to every page (`ServerApi._forwarded_reply` with
  `refetch_history`), a
  delete waiting for the machine's `historyDeleted` reply (echoing
  the command's `generation`; its `refused` carries the machine's
  refusal, `Cannot delete <what>: its task is still running. Stop it
  first.`), which, like an unreachable machine, is shown as a
  `notice` on the requesting tab; the
  link to each machine keeps its sockets alive with
  WebSocket pings, so an idle hour does not drop it, and a request
  whose previously kept socket has closed is sent once more over a
  fresh one; a kept socket whose ssh forward is no longer the login's
  live one (`MachineLink._forward_gone`: its ssh process exited, or
  another lane reopened the forward on a fresh local port) is dropped
  before the request is sent, so the request dials afresh, which
  reopens the forward, instead of reporting the machine down; every
  request is stamped with a fresh `generation` and a reply that
  echoes one must echo this one, so a frame still on its way for an
  earlier request is skipped rather than taken as the answer, and the
  daemon announces `echoesGeneration` in its `auth_ok` (its replies
  echo the stamp, read by `commands.request_generation` for
  `myModelsData` and `machines` and by the `models` and `history`
  handlers themselves; a page stamps
  some of its own requests the same way — its machine adds, whose
  answers the Machines subpanel tells apart by generation, and its
  history and model-list requests — and a command that carries none
  gets its reply unstamped (a `history` reply says `generation: 0`);
  the stamp is echoed to the
  requester only, links and pages each counting from 0), after which a
  `models`, `history`, `myModelsData`, `machines` or `historyDeleted`
  reply without a
  `generation` (`machines.CORRELATED_REPLIES`: a list the daemon
  pushed to every connection unasked, for another client's edit or a
  registry change) is not read as the answer either — a daemon
  without the flag is read as before, the first frame of the type
  being the answer; its sockets ask for permessage-deflate and for `parts`
  in their `auth` frame, so a transcript or a picture above
  `local_endpoint.PART_CHARS` (64 KiB) comes from the machine as
  `part` frames (`seq`-numbered, every one but the last flagged
  `more`) that `local_endpoint.FrameJoiner` joins, as this daemon's
  `WebPrinter._locked_send` sends such a frame to every peer that
  asked for `parts` (the webapp's `transport.js`, a link,
  `daemon_client`) and whole to any other (a later part is not written
  until the peer's unsent bytes are down to about one part,
  `_PART_ROOM_BYTES`, looked at every 5 ms by `_send_when_room`, since
  the loop's write buffer, the kernel's send queue and the peer's
  receive window hold many parts and a sender that wrote part after
  part ran megabytes ahead of the wire; a loopback `daemon_client`
  peer is not held), and between two parts lets through a sidebar
  panel's reply waiting for the connection — `URGENT_REPLY_TYPES`:
  `history`, `machines`, `cronJobs`, `appsStatus`, `spendReport`,
  `recommendations`, which `WebPrinter.broadcast` and
  `RemoteAccessServer._reply_direct` queue on the ahead lane of the
  endpoint's `FifoSendLock` (`acquire(ahead=True)`, served before the
  waiters in turn; the owner of the lock sees them in `ahead_waiting`
  and `let_ahead` hands them the lock and takes it back before the
  waiters in turn); an urgent reply bigger than a part to a peer
  that joins parts, a `history` page of fifty long rows, takes the
  ordinary lane instead and goes out in parts, since the parts of two
  frames must never interleave and a frame sent whole was minutes of
  silence to a slow peer's stale check — dropping a peer
  whose socket took nothing of a frame for `_SEND_TIMEOUT` (30 s; a
  `_DrainWatch` timer reads the peer's unsent bytes once a second,
  `_unsent_bytes`: the transport's write buffer plus, where it can be
  read, the kernel's send queue, `TIOCOUTQ` on Linux and other POSIX,
  `SO_NWRITE` on macOS, the write buffer alone on Windows) but never one that
  merely reads slowly; a stream socket's handshake (`MachineLink.open_connection`,
  a chat tab's or the machine's view stream) runs outside the link's
  lock, so it connects beside a request in flight on the shared socket
  and two streams opened together by a bind connect at the same time,
  while `retire` waits for the handshakes in flight before the removed
  machine's tunnels are swept; every dial runs on a thread of its own
  (`local_endpoint.dial_off_thread`), so a Stop of the calling task
  ends the wait at once, and a dial that has not reported
  `machines.DIAL_GRACE` (5 s) past its handshake's deadline — the
  lookup of the machine's name takes no timeout — is given up on; a
  connection the dial reports after the caller unwound (the timeout,
  a Stop) is closed by the helper thread itself (`ws.close()`, since
  the caller never took it), and one a Stop lands on after the caller
  has received it is hung up by the caller (`local_endpoint.hang_up`,
  a socket shutdown that takes none of the socket's locks), so no
  socket is left with no owner; after a failure the link reports the machine down without
  dialling again for `machines.FIRST_BACKOFF` seconds (5), doubling with
  every failure in a row up to `DOWN_BACKOFF` (30), a rejected password
  waiting the full 30 s from the first failure so the remote daemon's
  brute-force lockout is not tripped, and the ladder resets on the next
  successful handshake; `machines.forget_after_sleep`, run before a link
  uses a cached socket and from the daemon's watchdog, compares the wall
  and monotonic clocks and, once the host has slept `SLEEP_GAP` seconds
  (30) or more, shuts down every cached socket, kills the ssh forwards
  that are not mid-login and clears the back-offs, so the first request
  after a wake dials afresh instead of hanging for its whole budget on a
  dead socket; the Machines subpanel's report is cached for
  `sidebar_panels.MACHINES_STATUS_TTL_SECONDS` (60), or
  `MACHINES_DOWN_TTL_SECONDS` (10) while any machine is unreachable,
  and the next panel request after that, the page's 30 s poll or the
  panel's refresh button, re-pings them),
  and clicking one resumes that chat in a local tab bound to the machine,
  the transcript area saying `Loading the chat from <machine>…` under
  the task panel until the tab's first `status`, its transcript or an
  error arrives (`main.js` `showRemoteChatLoader`; the line is removed
  from a parked tab too; the bind publishes the tab in the shared
  registry with no chat or task of this daemon's, `RemoteTabs._publish`,
  dropping the local chat the tab showed before with its chat view, task
  pin and printer subscriptions, so a local task it was streaming is no
  longer fanned out to it and opening that chat in another tab displaces
  nothing, and an unbind from another connection, a `closeTab`,
  `newChat` or local-chat resume, lands wholly before or wholly after
  the publication, never half-way; a machine's chat shows in one tab,
  as a local chat does: a `resumeSession` of it into a second tab, or
  a bound tab's run whose first frame tells which chat the machine
  ran it in, displaces the tab that showed that chat
  (`RemoteTabs._displace_locked`, decided in the critical section that
  claims the chat, so two resumes racing each other leave exactly one
  tab), which ends its stream as an unbound one does and leaves the
  registry as a `closeTab` would, every surface closing it — unless
  something republished the tab meanwhile, in which case the close
  stands down; a stream carrying a sub-agent run of a local task has
  no chat tab and neither displaces nor is displaced)
  (its own web UI is never opened; a sub-agent tab that joins the
  task's stream, through `resumeSession` or a replayed
  `openSubagentTab`, has the binding announced to the surfaces
  (`tabMachine`) before its transcript, as the bound tab's is, so the
  drop-down shows the machine for the sub-agent tab as for its parent
  (the announcement is delivered under the stream's lock that made the
  tab one of its own, to every client, so no frame of the tab can
  overtake it; a sub-agent tab that leaves the stream for a chat, a
  `resumeSession` with a `chatId`, has the binding announced gone the
  same way, while a resume of a task the stream never saw keeps it on
  the stream and is answered here with nothing running),
  and a surface's `ready` is told every binding, the bound tabs' and
  their sub-agent tabs', in one `tabMachines` frame before any
  transcript (the page drops a binding it holds that the frame does
  not name: it ended while the page was away), and then the
  transcript of the one bound tab the `ready` names as shown (its
  `tabId`), with the `openSubagentTab` of each sub-agent under it and
  a `replay_deferred` offer in place of the sub-agent's transcript,
  which the surface fetches with a `scoped` `resumeSession` of that
  tab when the user opens it, as a local replay does since
  `server._open_persisted_subagent_tabs` stopped sending the
  children's `task_events` (a chat of ninety sub-agents held the loop
  for some twelve seconds per viewer): every other bound tab is
  deferred as a local tab is, the surface
  getting the `new_tab` or `openSubagentTab` announcement of each
  sub-agent still running under it (`RemoteTab.children`, the live
  announcements the stream relayed, without their replay stamps) with
  that sub-agent's last `status`, then the tab's own last `status`
  (`RemoteTab.last_status`) and a `replay_deferred` offer, and the
  page's `scoped` `resumeSession` of that chat, sent when the user
  opens the tab, is forwarded to the machine, which replays the chat
  to that connection alone while the tab stays bound
  (`RemoteTabs.replay_to` and `_resume`; replaying every bound tab
  used to send each connecting page every machine's transcripts,
  sub-agents included, ahead of its CHATS rows), the replays queued
  only once the CHATS page the surface asked for before its `ready`
  has been answered (`ServerApi.await_history_reply`,
  `HISTORY_FIRST_WAIT` 3 s, a second past the fan-out's own wait,
  after which the replays go out anyway; a connection with no CHATS
  request pending, a phone with its drawer closed, does not wait),
  so the rows are on the wire first; the chat header then shows the
  machine and that machine's working directory, with the directories
  picked there before in the drop-down, so a pick or the folder picker
  changes the machine's working directory, never this daemon's --
  `getMachineWorkDirs`, and `setWorkDir` with `machine`, which the
  daemon proxies to the machine (these round trips, and `machineAction`'s
  add/remove with its handshakes to an unreachable host, run as
  background tasks serialised per machine by `ServerApi._machine_locks`,
  so a slow machine never holds up the connection's `submit`, `stop` or
  `ping`, nor another machine's requests; a subpanel poll, a `getModels`
  or a CLI status probe repeated while its predecessor still waits on
  the machine is dropped, `ServerApi._spawn_panel_reply`, so the
  executor threads a slow machine holds stay bounded); the Explorer and Source Control views
  show the machine's working directory and repository, and a file
  opened from them is read and saved there: every workspace command
  (`listDir`, `gitStatus`, `gitLog`, `gitShow`, `gitAction`,
  `fsAction`, `openFile`, `saveFile`, `checkPaths`) stamped `machine` is forwarded to
  the machine's daemon over the link and its reply, stamped `machine`,
  sent back -- `sidebar_panels.MACHINE_WORKSPACE_REPLIES` (a command
  carrying a `token` gets it echoed on every reply, success, binary or
  typed error, as does the local reply for a machine that is unknown or
  unreachable: `media/main.js` sends a tokened `openFile` for each
  `<img>` whose `src` is a file path in a result, answer, Markdown or
  task-update panel, resolved against the task's work dir, or in a
  previewed `.html`/`.md` file, resolved against the file's directory,
  shows the `fileContent` reply (`base64` bytes of a raster image, the
  UTF-8 `content` of an SVG) as a `data:` URI and, on an `error`, in a
  panel the alt text in a dashed box, in a preview the original `src`,
  the tokened reply settling the request without opening a tab; a
  transcript replayed into a tab, local or bound, leaves each picture
  embedded in a `tool_result` out as a `lazy: {taskId, index}` handle
  (`server._lazy_replay_images`, *index* counting the transcript's
  pictures in event order, since base64 does not compress and the
  pictures were most of a replay's bytes), and `main.js` sends
  `getReplayImage` with the handle once the `<img>` comes within
  400 px of the viewport (an `IntersectionObserver` with a 400 px
  `rootMargin`; a browser without the observer fetches at once),
  stamped `machine` for a bound tab
  (`sidebar_panels.machine_replay_image_reply`, on the `panels` lane),
  the `replayImage` reply echoing the `token` with the `mime` and
  `b64`, or with an `error` for a picture the daemon no longer has);
  the share export of a bound tab asks that machine's history the
  same way (`shareChatTasks` stamped `machine`,
  `sidebar_panels.machine_share_tasks_reply` on the `panels` lane,
  answered in the background like a user's own action since the
  `share_tasks` reply may be megabytes of transcripts; a machine that
  is unknown or unreachable answers an empty reply carrying the
  `error`, and the page reports the share failed instead of exporting
  this daemon's history of the chat — it also refuses a reply whose
  `machine` stamp is not the tab's binding); a
  `submit` whose prompt is nothing but the path of an existing file,
  sent from a tab that is not running a task, starts no task on
  either daemon: the machine answers `promptOpened` and the file's
  `fileContent` (`web_server._open_path_only_prompt`), and
  `RemoteTabs` relays the content stamped `machine`, as the reply to a
  forwarded `openFile` is, so the editor tab saves and reloads the
  file on the machine, and gives the tab back the title the prompt
  displaced (`TabRegistry.retitle_if_generation`, keyed on the
  `promptId` the forwarded submit was stamped with and echoed on
  `promptOpened`, and on the registry generation, so a late answer
  never undoes a newer title); the
  settings sheet opens for the machine too, so its commands
  (`getConfig`, `saveConfig`, `getMyModels`, `getClaudeCodeStatus`,
  `getCodexStatus`, `startCodexLogin`, `saveMyModel`, `deleteMyModel` --
  `sidebar_panels.MACHINE_SETTINGS_REPLIES`) stamped
  `machine` are forwarded the same way, their replies stamped `machine`
  and `forwarded`, a machine's `error` frame coming back as the reply's
  `error`, and a `saveConfig` that changes a password-registered
  machine's remote password to another non-empty one re-registers the
  machine with it (an emptied password is not adopted); an open sheet
  whose drop-down machine changes (a tab switch, the daemon's binding,
  another surface's pick) drops an unsaved in-place custom-model
  edit, saves the edited settings fields to the machine they were
  made for, then loads the new machine's settings (`main.js`
  `syncSettingsMachine`); the chat input's model picker is that
  machine's picker too: `getModels` and `selectModel` stamped `machine`
  (`sidebar_panels.machine_models_reply`; `selectModel` is no longer
  in `remote_tabs.FORWARDED_COMMANDS`, which holds the tab commands a
  bound tab sends to its machine instead of running here: `submit`,
  `run`, `stop`, `userAnswer`, `appendUserMessage`, `interruptTool`,
  `worktreeAction`, the Merge/Discard bar of a bound tab acting on the
  worktree its task left on the machine, whose `worktree_result` comes
  back stamped with the tab, and `getAdjacentTask`, the chat's task
  before or after the one on screen, which only the machine's history
  can answer) bring the machine's own model
  list and default model and make a pick the machine's default (and
  the tab's there), this daemon's default untouched, the `models`
  reply echoing the command's `generation` so the page keeps the
  list of a reply asked before its latest pick but ignores its
  `selected` value (`main.js` keeps every machine's last
  list and default in `machineModels`, re-syncs the picker whenever
  the drop-down's machine changes, and shows a running agent's model
  override only on the machine its task runs on), and a `submit` to a
  registered machine carries no `useWebTools` override, so the machine
  applies its own setting; the `@` mention picker of a bound tab lists
  the machine's files, its `getFiles` (with the tab's browse folder as
  `workDir`; every successful `files` reply, local or forwarded, echoes
  the directory it was resolved for as `workDir`, the daemon-wide one
  for a request that named none (the empty `files` list built for a
  machine that is not registered or unreachable carries the `error`,
  `prefix` and `tabId` but no `workDir`,
  `sidebar_panels.machine_files_reply`), and the page drops a list for a
  directory the tab no longer browses, so a workspace pick on another
  surface that lands while a cold index builds leaves none of the old
  workspace's files in the picker, `autocomplete._emit_files`) and
  `recordFileUsage` stamped `machine`
  (`sidebar_panels.machine_files_reply`, `machine_file_usage`): a
  folder the machine has not indexed is answered `loading` at once, so
  the forwarder asks again at growing intervals (0.2 s doubling to
  2 s) for up to `FILES_INDEX_WAIT` (60 s), a list echoing another
  `prefix` or `tabId` (an earlier request's, sent after that one had
  returned `loading`) asked past as well, the replies produced in the
  background in the order asked per connection, tab and machine
  (`ServerApi._spawn_after`, `_pending_files`), and `recordFileUsage`
  has no reply (`MachineLink.request` with an empty reply type returns
  `{}` once the command is sent; a lost count is logged); the sidebar's
  Schedule, Machines, Apps, Spend and Recommendations subpanels show
  the machine too, their commands (`getCronJobs`, `cronAction`,
  `getMachines`, `machineAction`, `getAppsStatus`, `getSpendReport`,
  `getRecommendations` -- `sidebar_panels.MACHINE_PANEL_REPLIES`)
  forwarded the same way on a link lane of their own
  (`machines.link_for(machine, "panels")`, so the up-to-45-second app
  probe holds up no submit, settings or Explorer request; a `machines`
  reply is waited for `SSH_TIMEOUT + ENROLL_TIMEOUT +
  DNS_TIMEOUT_SECONDS` longer than the link's usual budget, since the
  machine pings its own machines first and an `add` enrolls the new
  one), an unreachable or refusing machine answering with an empty
  reply of the expected type plus the `error` (a `cronAction`'s or
  `machineAction`'s reply, this daemon's own or a forwarded one, also
  echoes the command's `generation`, 0 when it had none: the Schedule
  subpanel stamps each save with one, so it closes its in-place form
  only on the reply to its own create or update whatever the order
  two saves are answered in), so the Machines subpanel
  shows and edits the registry of the machine it shows while the chat
  header's drop-down keeps listing this daemon's registry (the page
  asks an unstamped `getMachines` for it beside the forwarded one);
  the "..." menu's Git Commit (`autocommitAction` stamped `machine`,
  `sidebar_panels.machine_autocommit`) commits the machine's folder:
  its daemon stages, writes the message and commits there and sends the
  link its own copy of the tab's `autocommit_done` (`replyTo`), which
  this daemon re-broadcasts to its clients as the toast and the event
  that re-arms the button, on a `commit` lane with a 900 s wait
  (`MACHINE_COMMIT_TIMEOUT`); a daemon's
  link is never a local client of the machine, whatever token it
  presents, so the machine answers these as it would a browser), the chat header's machine drop-down
  offers it (a machine registered by its address is shown by the host
  name its daemon reports with every ping; a focus on the drop-down
  asks the daemon for a fresh `machines` list, and `renderChatMachine`
  rebuilds its `<option>`s only when their values or labels differ
  from what is shown and sets its value only when it differs from the
  chosen machine, since iOS dismisses a native picker whose options
  are replaced under it, or keeps showing the old ones, so on a phone
  the list opened by the first tap after a reload would never show),
  and `run_agent(machine=...)`
  can run a sub-task on it; with the `https://` form the password may be left out only for a
  daemon that has no remote password set and is reached on a loopback
  port), `/merge`
  (resolves git merge conflicts and stages the resolved files; commits
  only when asked), `/recommendations` (a `WorkerSea` whose tools are
  `finish`, `recent_tasks`, `connected_apps`, read-only memory and
  `save_recommendations`: it reads your own top-level tasks
  (`persistence.user_tasks_since`; the last 24 hours and, further back
  within 30 days (`MAX_DAYS`), enough older tasks to review at least 20
  completed ones, 100 tasks at most; cron runs and its own runs are
  dropped), asks the auth-status probe which third-party
  apps you are signed in to and memory what you did in them, picks ten
  concrete follow-ups, at least three of them combining the signed-in
  apps, or ten starter tasks for a new project when there are none,
  and stores them in `$KISS_HOME/recommendations.json`, which the
  sidebar's RECOMMENDATIONS subpanel shows (`save_recommendations`
  takes exactly ten non-empty one-sentence prompts of at most
  `MAX_SENTENCE_CHARS`, 160, characters each; the longer ones are
  named with their lengths, `Error: shorten these to at most 160
  characters and call again with all 10: #3 (171 chars).`, so the
  model never counts characters itself); `/recommendations now` runs it
  once, and the 6-hourly default cron job keeps the file fresh; both
  default jobs come from `cron_agent.DEFAULT_JOBS`, seeded once per home
  by `seed_default_jobs` at daemon start so your edits and removals
  stand (a job already stored under a default's name, or a former
  name of it, paused or not, is left as it is: the check is made under
  the store lock in the same step as the create, so a job created
  meanwhile is never doubled, and a store that cannot be read seeds
  nothing); before the seeding, `cron_agent.adopt_previous_home_jobs`
  records the daemon's home in `~/.kiss-web-home` and, when that
  marker named a different brand home (a brand install or an
  `rsorcar` deploy moved the daemon between `~/.kiss` and
  `~/.<brand>`), copies every job of the old home's
  `cron/jobs.json` (`load_jobs(path=...)`) whose name this store
  lacks, with a clean failure streak, a fresh next run for a
  repeating job and the old deadline for a one-shot; a finished job
  stays finished, and a store that cannot be read or written is left
  alone while the marker still moves on), `/remember` (appends a standing instruction to
  `$KISS_HOME/AGENTS.md`), `/review_paper` (reviews a research paper for
  a venue, scoring seven dimensions from 1 to 10; its checks and rating
  caps are calibrated on the ICLR 2027 reviews of six submissions: the
  review first states the paper in one sentence and checks the causal
  evidence for the central claim, a controlled comparison, an ablation,
  spread, denominators and a human check of automated judgments, and a
  central claim with no controlled comparison, or a headline number
  behind an unvalidated automated judgment, caps Support for the
  claims at 3/10 or below and the venue rating at reject whatever the
  writing; it reads the
  appendix and the artifact, ends the Detailed review with at most five
  numbered questions for the authors, and its `check_review` tool lists
  the causal-evidence words the review uses (`baseline`, `ablation`,
  `seeds`, `spread`, `denominators`, `single run`, ...), `CHECK` rather
  than a failure when it uses none),
  `/review_patent` (reviews a patent
  application draft or a published patent like an examiner: prior-art search, eligibility,
  novelty, obviousness and definiteness claim by claim, scoring seven dimensions and the
  allowance likelihood, with `read_patent`, `check_patent` and `check_patent_review`
  tools), `/revise_and_review_paper` (writes a paper with
  `/write_paper`, has `/review_paper` review it fresh, and repeats until
  strong accept or no further improvement; task text carries `Writing:`
  and `Review:` instructions; a review that wants a stronger method or
  result sends the writer to `/discover`, run as a sub-agent on a lab
  copy that hands back `best.patch`, and a weakness the allowed
  evidence rules out is stated in the paper as a limitation that
  narrows the claim, a missing control for the central claim cutting
  that claim to what the evidence tests rather than being confessed
  and left standing), `/rsi7d` (7-day self-improvement
  of the indexed SEAs from their recorded runs, each SEA's improvement
  run through a `/discover` child on a lab copy of its script and test
  file with a replay of a past run as the dev evaluation, the best point
  coming back as a patch through the gated editors: its
  `improve_sea_with_discover(name, task, max_budget, replay_task_id,
  replay_budget, ...)` tool snapshots the SEA folder to
  `tmp/rsi7d/snapshots/<name>` and the test file, when it exists, to
  `tmp/rsi7d/snapshots/test_<name>_sea.py.before` (a restore deletes a
  test file that did not exist then), dispatches the child (round limit
  `DISCOVER_ROUNDS`, 2, and `DISCOVER_TIMEOUT_SECONDS`, 4 h, by
  default; each of its replays times out after `REPLAY_TIMEOUT_SECONDS`,
  1 h), applies the child's `best.patch` to the script and test file
  only (a patch touching any other path is refused) through the same
  compile/load/`sea lint` gate as `patch_sea_code`, and restores the
  snapshot when the child returned no patch or the gate refused it;
  `replay_findings(<child task id>)` measures the child's replays; the
  task text starts with the scope: `/rsi7d all`, `/rsi7d <name> [<name>
  ...]` for those SEAs only, or `/rsi7d --seas-dir <folder> [<name>
  ...]` for the SEAs of
  that folder, which then are the ones it may edit), `/sh`
  (runs the command with the `bash` tool profile), `/skillopt`
  (optimizes the prompt text of a skill or SEA against an eval set),
  `/sync` (commits the uncommitted changes of the tab's working
  directory on its current branch — `git add -A`, a commit whose
  message names their number and the host — then fetches `origin/main`,
  merges it into that branch and pushes, through its `git_sync` tool
  (`seas/sync/git_sync.py`: commit, fetch, merge, push, with the round
  retried when the remote moves during the push; a tree with unmerged
  paths left from an earlier merge is refused rather than committed);
  its `settings()` adds `coding_gate: False` to the run's
  `model_config`, so the run is no coding run for the finish gate —
  what it changes is the merge of commits made elsewhere and the
  resolution of their conflicts, which no reviewer of its diff can
  tell from the pulled commits — and the gate, to which `git_sync` is a
  tool it does not model, takes HEAD as its base again after each
  `git_sync` result (once a base is recorded and while the run has
  not moved HEAD with a git command of its own); a `model_config` of
  such framework flags alone
  is added to the configuration that applies anyway, so a
  settings-panel endpoint stays (`model.framework_flags_over`, applied
  by `task_runner`);
  a merge that stops on
  conflicts is left in progress for the agent to resolve with the
  `shell+edit` tools and finish with a second `git_sync` call; its
  `prompt()` supplies a default task, so a bare `/sync` runs it, and
  `/sync <remote>/<branch>` merges another remote branch; when machines
  are registered the `prompt()` also lists them and the agent syncs
  each one in turn before this checkout — the repository's URL is the
  one the task names after the ref (`/sync origin/main of <url>`),
  else what its `remote_url` tool returns (the ref's remote URL
  without its credentials) — through `run_agent(agent="sync", machine=<name>,
  task="<ref> of <that URL> this machine only", max_budget="2")`, with
  no `work_dir` option, waiting for each before the next, then calls
  `git_sync` here, then syncs each machine once more in the same way,
  so every checkout ends up with what the others pushed (a machine
  whose second sync pushes commits of its own is noted in the summary,
  since this checkout gets them at the next `/sync`); the sub-task
  runs in the machine's own folder and its `git_sync(repo, ref, url)` syncs
  that folder (committing what is uncommitted there too) when it is a
  clone of the URL, else the clone among the
  directories the machine's daemon has opened (`config.json`'s
  `work_dir` and `recent_work_dirs`), since the checkout lives
  wherever that machine keeps it (forcing this machine's path on it
  used to fail with `Permission denied: '/Users'` on a Linux box);
  the words "this machine only" in a task
  (what the sub-task sent to a machine says, or `/sync this machine
  only` typed by the user) keep that run on its own checkout, so the
  machines registered there are not synced in turn; the budget is
  $2 per sync, one for this checkout and two per machine, since a
  sub-task's spend counts in the caller's),
  `/install` (`seas/install/`, a `WorkerSea` with `tool_profile:
  "none"` locked, `max_budget: 1.0` and a `timeout` of the one-hour
  install limit plus ten minutes, whose one tool
  `install_on_machines(machines="")` reaches every machine registered
  with `/machine add` by ssh concurrently, finds the clone of the
  private repository `https://github.com/ksenxx/kiss.git` in the
  login's home (`~/kiss` first, then any `~/<dir>` holding an
  `install.sh`, matched by `origin` through `git_sync.repo_key`) and
  runs `./install.sh --non-interactive` there, reporting per machine
  ok or FAILED with the reason (no answer to the checkout search
  within `SEARCH_TIMEOUT_SECONDS`, 60 s; ssh that cannot connect; no
  clone of the repository with an `install.sh` in the home;
  `install.sh` running past `INSTALL_TIMEOUT_SECONDS`, 3600 s, or
  exiting non-zero) and the last 15 non-blank lines of output, uncoloured, the
  rest of any `password:` line redacted; a machine registered by URL
  is reported as skipped; `/install <name>[,<name>]` installs only
  those, and a name `/machine list` does not know fails the call
  before anything is installed; its `system_prompt()` replaces
  `SYSTEM.md` and tells the agent to call the tool once (`machines`
  as the names the message gives, else `""`) and then `finish`, with
  `success=true` only when every machine reports ok, never retrying;
  its `prompt()` appends "Run ./install.sh in the private kiss
  checkout of every registered machine." to every task, so a bare
  `/install` runs on that text alone; it declares no `work_dir`, so
  it runs in the caller's directory),
  `/sync_history_and_memory` (`seas/sync_history_and_memory/`, a
  `WorkerSea` that brings `$KISS_HOME/history.db` — tasks, events,
  deletion records — and `$KISS_HOME/memories` of this machine and of
  every machine registered with `/machine add` by ssh (a machine
  registered by URL is reported as skipped) together in one run: a
  thin wrapper around its one deterministic tool
  `sync_history_and_memory(machines="", dry_run=False)`, which its
  `system_prompt()` — a replacement for `SYSTEM.md` — tells the agent
  to call exactly once (`machines` as a comma-separated subset only
  when the message names machines, `dry_run=True` only when it asks
  for a dry run or a preview, never ssh, sqlite3 or rsync itself)
  and then to `finish` with a short HTML summary repeating the tool's
  per-machine report; its `prompt()` appends "Sync the task history
  and the memory of this machine with every registered machine." to
  every task, so a bare `/sync_history_and_memory` runs on that text
  alone; `settings()` sets `tool_profile: "none"` and locks it, so
  the run's only tools are `finish` and the sync tool, and
  `max_budget: 1.0`; it declares no `work_dir` or `timeout`, so as a
  worker it runs in the caller's directory (the tab's for a slash
  command) with the default timeout.  The tool
  (`peer_sync.PeerSync`) reaches every machine at once over one
  multiplexed ssh connection each (`ssh_peer.SshPeer`) in three
  waves — connect, find each peer's KISS home and read its clock and
  memory fingerprints, pull everything from every peer into this
  machine, then push this machine's union to every peer — so one run
  converges every peer that completes all three waves (a peer that
  fails in any wave is left out of the later ones and its report says
  why it was skipped
  or left half-synced), apart from the held-back tasks and kept
  conflicts below; history rows travel through `kiss.scripts.sync_db` as
  a union of task rows by `task_history.id` and event rows by
  `(task_id, seq)`, with the `deleted_tasks` and `deleted_chats`
  records carried over and applied, so a deleted task stays deleted
  everywhere; memory pages by newest `updated` stamp with tombstones
  for deletions and only the pages whose content hash differs
  (`memory_delta`); two differing writes of one page whose stamps are
  no further apart than the measured clock spread
  (`merge_memory_pages.compare`, `<= tolerance`) are reported as a
  conflict and kept as they are on each side; a task whose row is in
  the outgoing delta and still running — the later of its newest
  event and its start younger than `RUNNING_WINDOW` (300 s) by its own
  machine's clock, or still carrying the web app's start placeholder
  with that time younger than `ABANDONED_AFTER` (24 h) — is held back
  on the side it runs on and travels once it finishes (events alone
  for a row the peer already has are never held); a machine that
  fails is reported with the reason and not retried.  A peer's home
  is not assumed to be `~/.kiss`: a stock install keeps its state
  there, a branded one in the directory its brand's `home_dir` names,
  and one machine may run several daemons, so wave one runs one
  remote shell (`peer_sync.home_probe`) that prints the home of the
  daemon the machine was registered with — the directory under the
  peer's `$HOME` whose `sorcar-local.json` endpoint file names the
  port of the machine's registered URL, looked for in this install's
  brand home and the stock `.kiss` (`HOME_CANDIDATES`, in that order)
  first and then in every other `~/.*` — and, when that daemon is not
  running (a stopped daemon removes its endpoint file) or the port is
  unknown, the first candidate holding a `history.db`, else the first
  that exists, else the first; the peer's report says `KISS home
  there: ~/<dir>`, and an answer that is empty, `.`, `..` or holds a
  `/` fails the peer in wave one),
  `/task_update` (reports what a running task has done so far),
  `/write` (writes prose for a general audience in concise, professional
  American English that reads as human-written; it defines only
  `description()`, a `system_prompt()` that appends its style guide
  and a `settings()` of `{"timeout": 3600.0, "model": "bestrouter"}`), `/write_paper` (writes
  or revises a research paper; it plans the contribution first, the
  research question, the central claim and the controlled comparison
  that tests it, narrows the claim to what the evidence tests rather
  than confessing an untested one in Limitations (when the sources
  hold no comparison that tests the claim, it tells the user before
  writing and narrows the claim), reruns baselines on the same model,
  budget and hardware for the headline comparison, ablates every
  claimed component, reports runs or seeds with their spread and the
  test used, the full pool and selection rule with denominators, and
  a human-labelled subset with its agreement number behind every
  automated judgment, compares directly to the 3 to 5 closest
  systems, places an overview figure before the component details and
  a results figure in the evaluation, and
  keeps engineering trivia out of the abstract and introduction; its
  `check_paper` tool fails on more than one concession in the
  abstract, on line counts, module counts, prompt word counts, file
  paths, dollar costs, `wall-clock` and `steering messages` in the
  abstract or introduction, and on a hedged, ironic or question heading
  (a `?`, `not`, `what it` or a comma that is no enumeration), and
  lists the concessions in the body for review), `/write_patent` (writes or revises a utility patent
  application as a Markdown draft in the 37 CFR 1.77 layout, with `check_patent`,
  proofreader gates for sections, claims, antecedent basis, figures and prose, and
  `build_patent`, which renders the draft to DOCX, a specification PDF and a drawings
  PDF), and the plain sub-agent
  (`seas/sorcar/sorcar_sea.py`, a hidden SEA whose settings are
  `{"hidden": True}`, is what `run_agent` runs when its `agent` argument
  is empty or a generic label).  The other modules in that package
  (`coding/coding_test_context.py`,
  `git_extract_knowledge/git_knowledge_index.py`,
  `git_extract_knowledge/git_knowledge_store.py`, `rsi7d/sea_tuning.py`,
  `sync/git_sync.py`, the shared `agents_md.py`) are helpers, not
  commands: only `<name>/<name>_sea.py` folders are registered.  Syntax,
  precedence and the dispatch flow are
  documented in
  [docs/sea-commands.md](https://kisssorcar.github.io/docs/sea-commands.md)
  (source: `website/kisssorcar.github.io/docs/sea-commands.md`).
- A `/xxx text` command has no outer relay run: the tab's run IS the
  SEA run (`seaPath` set to the SEA, `text` as the prompt), with the
  tab's model, budget, chat and settings as the caller's values and
  the SEA's `settings()` on top of them, exactly as for a `run_agent`
  dispatch (see [Precedence](#precedence)).  The SEA is loaded by the
  same `apply_sea` as any `seaPath` run, so a broken
  script fails the run with an `SeaError` before the agent
  starts.
