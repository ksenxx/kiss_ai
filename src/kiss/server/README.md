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
   │                                         │   tool_call_hook folded into one hook each
   ▼                                         │ stage one wire field per setting, the
 block, read events ◄───────────────────     │   prompt, and the four daemon-side hooks
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
   type-checked, a channel's locked keys enforced).
   `apply_sea()` (in `kiss.agents.sorcar.sea_apply`)
   then evaluates the SEA on the task (`evaluate_sea`), writes each
   setting into the command dict's wire field and the result of the
   `prompt(task)` chain into `prompt`, and stages the four daemon-side
   hooks (`systemPromptHook`, `toolsHook`,
   `llmCallHook`, `toolCallHook`), each a fold of that method over
   the chain, base first.  Overrides are staged: they apply atomically
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
   (`RelentlessAgent.perform_task`).  The daemon's base prompt is
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
   value is the run's toolset (`finish` is always added).  `llmCallHook` and
   `toolCallHook` run around every LLM and tool call.  Nothing is
   re-imported.


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

<!-- sea-docs: settings -->
| Key | Type | `run` wire field | Meaning |
|---|---|---|---|
| `work_dir` | `str` | `workDir` | The directory the run works in; default: the calling task's or the tab's. A relative path is a path under the calling task's directory (the tab's for `/<name>`), in a SEA's settings and in a call's `work_dir` option alike. |
| `model` | `str` | `model` | The LLM model, a catalogue name or a model-picker SEA; default: the caller's (`""`, like `None`, is no override — true of every string key). |
| `chat_id` | `str` | `chatId` | The chat the run's events go to; default under `run_agent`: the calling task's chat, or a new chat when nothing is inherited (a channel, an `inherit: false` call); a `/<name>` run keeps the tab's chat. |
| `max_budget` | `int \| float` | `maxBudget` | USD budget of the run, a finite number; default: the caller's share or the daemon's default. |
| `model_config` | `dict` | `modelConfig` | Model configuration dict passed to the LLM (temperature, base URL, ...). |
| `use_web_tools` | `bool` | `useWebTools` | Give the run the browser tools (daemon default: on). |
| `auto_classify` | `bool` | `classifyTasks` | Let the pre-run classifier decide whether the task is simple (lite prompt) (daemon default: the persisted setting). |
| `use_memory` | `bool` | `useMemory` | Give the run the `memory_*` tools (daemon default: the persisted setting). |
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
- `prompt(task)`: chained; the subclass's receives what the base's returned.
- `system_prompt(system_prompt)`, `tools(tools)`, `llm_call_hook(new_messages)`:
  folded the same way, base first; each class receives the previous
  result and returns the next.
- `tool_call_hook(name, args)`: the classes are asked in order and the
  first refusal (`refuse(text)`) wins; `ALLOW` allows.  The model never
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
`summary` tool (`self.has_summary_tool`), `llm_call_hook` counts one
step per LLM call and marks a summary due at every 10th step of a run
that has the tool, and `tool_call_hook` then refuses every call but
`summary` and `finish` with `base_sea.SUMMARY_DUE_REFUSAL` ("Step 10
is a multiple of 10, so a summary is owed before any other tool runs:
call summary(description=...) now, ..., then re-issue this exact Bash
call.") until `summary` runs; the `summary` call's own result then
carries the re-issue reminder.  A SEA whose own `tools()` drops `summary` should set
`self.has_summary_tool = False` there (every bundled SEA only appends
tools).  `system_prompt` is folded exactly
like `prompt`: each class receives the text so far and what it returns
is the text, whether it appended to it or replaced it; the last
return is the run's system prompt, with the executor's own notes and
its operational / Task Settings instructions appended after it (see
"Run-time hooks" above).  Neither method's return reaches a
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

(The file's real `tools` and `llm_call_hook` bodies reset and count
the run's steps, `BaseSea.step`, one per LLM call, and its
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
it finishes: an explicit `max_budget` is on its `AgentJob.budget`
from registration, an inherited share once the job's thread resolves
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
`HEAD`, untracked files included, a cap on the inherited share
(`budget_cap` of `inherit_from_parent`, applied only when the budget
is inherited) that never raises it and that the dispatcher applies
before the SEA's own `max_budget` setting, which still takes
precedence; for a `machine` dispatch the
diff is measured by local git at the call's `work_dir` option, so
a remote review gets the $5 base unless that path is also a
repository here), its
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
system prompt names a scratch directory of its own, `./tmp/<task
id>/` under that directory (`ChatSorcarAgent`), so siblings in one
work directory never share a scratch file.  `run_parallel` children are
`run_agent` sub-tasks started by the same code
(`agent_dispatch.start_run_agent`), so this is their table too (the
precedence rule above decides between the
inherited values and the SEA's).  A channel SEA (`agent="slack"`,
`agent="cron"`, ...) inherits none of these: it runs on the daemon's
default model and budget unless the call passes its own, and
`ChannelSea` locks its `work_dir` and the worker keys (`auto_classify`,
`use_web_tools`, `use_memory`), so a call whose `options` contradict
one is refused.

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

### `timeout` — optional, for SEAs that run longer than an hour

A `run_agent` or `run_parallel` call waits for its sub-task(s) for a
bounded time (`agent_dispatch.resolve_timeout`): an explicit `timeout`
argument wins; an empty one takes the SEA's `settings()["timeout"]`
when that is positive; with neither,
`DEFAULT_DISPATCH_TIMEOUT_SECONDS`, 3600 s.  When the bound expires
the sub-task is NOT stopped: it keeps running as an `agent_job` and
the call returns the job's id (`agent_job(id, "wait")` collects the
result, `"kill"` stops it; a job still running when the calling task
ends is killed).  A SEA whose runs may take longer than an hour
declares it so a caller is not handed a job id too early (`/write_paper` and `/write_patent`
6 h, `/review_paper` and `/review_patent` 2 h, `/revise_and_review_paper` 24 h); `/write`
pins the default
explicitly:

```python
DISPATCH_TIMEOUT_SECONDS = 3600.0


class WriteSea(BaseSea):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"timeout": DISPATCH_TIMEOUT_SECONDS}
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
`timeout` (number strings) are arguments only, and `options` refuses
those keys with a message naming the argument.  Every value given is
explicit: it wins over the SEA's settings, or is refused when the SEA
locks the key.  Each option replaces the inherited value (and the
SEA's, unless locked) of one key: `work_dir` (relative to
the calling task's directory), `chat_id`, `workspace` (the account of
a multi-account channel; refused for any SEA that is not a channel),
`model_config`, `inherit`, `auto_classify`, `use_web_tools`,
`use_memory`, `docker_image`, `add_to_prompt` and
`add_to_system_prompt`, e.g. `'{"use_web_tools": false}'`.  `wait="false"`
returns with a job id as soon as the sub-task's tab exists (its
initial `status running=true`, so the spawn lands inside the call's
time window on every surface), the dispatch has already failed (then
the error is the answer), or a 30 s start grace
(`agent_dispatch._JOB_START_GRACE_SECONDS`) has passed; the dispatch
runs in a daemon thread
of the calling process (`agent_dispatch.start_agent_job`), its spend
is bound to the calling run's usage epoch (a finished child's cost
is folded into the caller's task, and the `Budget:` figure of the
usage line the caller's model reads reports the task's cumulative
spend, earlier sessions and sub-agents included, against the task's
cap; `RelentlessAgent._budget_status`, installed as
`KISSAgent.budget_status_hook`, recomputes it after every tool call,
so a SEA's prompt must not add a child's reported cost again), the budget
it was given stays reserved against the caller's remainder until it
finishes (`AgentJob.budget`, see [Precedence](#precedence)), and only
the calling agent's `agent_job` tool sees the job: `agent_job(job_id, "wait")`
blocks for its result, `"tail"` reports whether it still runs,
`"kill"` sets the job's cancel event, which `daemon_client.run`'s read
loop turns into a stop of the sub-task confirmed by its terminal
status (an unconfirmed stop is reported as such, never as "stopped").
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
has already loaded (`_reload_layers` reuses them for the fallback).
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
relays back, as it does for a chat tab bound to a machine.  What
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
for good; one a client bound to a stream of its own meanwhile is
left to it); a surface
connecting while it runs gets
the same `new_tab` from `RemoteTabs.replay_to`.  The result's `ran:`
line ends with `machine=<name>`; an unregistered name or host is
refused with the registered names before anything runs.

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
after the terminal grace period unless a reload claims it first; a
reopen that arrives once the hang-up has started waits for that shell
to be reaped and starts a fresh one, and the old shell's exit is not
reported under the new page); a page's
tab id is namespaced `browser__N@machine`, so a tab-addressed browser
command finds its machine from the id, its frames go to the local
connections whose last `browserViewport` for the tab was visible, and
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
registry lock, in the registry's order; `_announce` and the `ready`
replay drop a binding the registry has moved past); `RemoteTabs`
then opens the stream (`open_for`, before the run is forwarded, so
the page its agent opens finds the stream subscribed) or releases the
previous one (`release`); `close_all` fences every later registration.

`run_parallel(tasks, ..., agent=...)` is N `run_agent` calls
(`agent_dispatch.run_agents_parallel`): one `start_run_agent` per
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
stopped by the timeout.

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
lists every registered SEA whose `register_as_model()` returns `True`,
the daemon offers them in the model picker (vendor `Router`, once at
least one catalog model is runnable), and a task run with such a pick
becomes a run of the SEA on the model its
`settings()["model"]` names (else the default model) — `/xxx` slash
commands and runs that already carry an `seaPath` keep their agent,
with the picker's methods folded in first (see
[Extending a SEA](#extending-a-sea-python-inheritance)).  The bundled
`autorouter`, `bestrouter` and their OpenRouter-only twins `orautorouter` and
`orbestrouter` are such SEAs; `bestrouter` is three methods besides `description()`:

```python
class BestrouterSea(BaseSea):
    def register_as_model(self) -> bool:
        return True

    def system_prompt(self, system_prompt: str) -> str:
        return system_prompt + "\n\n" + SYSTEM_PROMPT  # the routing protocol

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"model": PRIMARY_MODEL}
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
for a `/xxx` command's agent or an explicit `seaPath`, only for the
SEA the model pick names.  `autorouter` uses it to make sure an
enabled weekly cron job that runs `/rsi7d autorouter` exists
(`autorouter_sea.schedule_weekly_rsi7d`: an enabled job of that name is
kept as is, a paused one is resumed, and when there is none one is
created — a prompt job in the KISS checkout the work directory
is in, or a scratch-directory job that
still refreshes `$KISS_HOME/AUTOROUTER.md` when there is no checkout;
a linked task worktree resolves to its owning checkout).  Hooks from
concurrent picks may overlap — a SEA file is executed afresh on every
call and cannot hold a lock of its own — so the look-up-then-create is
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
  `agent_dispatch.is_review_task`, and the daemon re-checks the
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
  `tabScopeWorkDir`: `run_agent` fills it from the caller, and it is
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
  `attachments` only when non-empty.  The `/ask` side channel passes
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
  and UTC, parallel mode, the `Scratch dir` the prompt sends temporary
  files to — `./tmp/<task-id>/` under a chat, `./tmp/` without one —
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
  watch and act on it; there is no setting for it.  A browser the
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
  pages, from the block page's own title and body wording) and given
  `web_stealth.challenge_wait_secs(vendor)` to clear on its own: 12 s,
  45 s for Anubis's CPU-bound proof of work; a vendor with a widget
  gets it pressed like a person after a 1.2 s reaction time —
  Turnstile's box once, a 60-140 ms click; HUMAN's `#px-captcha`
  press-and-hold bar (walmart, fiverr) with an 8-10 s hold under a
  drifting pointer (`press_hold_secs`, `hold_tremor`,
  `PRESS_AND_HOLD_BOX_JS`) and, when the bar is still showing 5 s
  after release, a second hold (`widget_presses`); each press extends
  the wait by the vendor's allowance.  A page still blocked is
  reported to the model as a `Note:` naming the host, the vendor and
  the seconds actually waited, with the advice to call
  `show_browser()` so the user can complete the check or to use
  another source (`WebUseTool._settle_challenge`; a Google search
  that got the `/sorry/` page is re-opened on Bing instead, with a
  note saying so).  A Playwright driver
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
  browser and opens a new tab (`_attach_live`).
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
  pool (`MemoryTools._search`, `tools._result`), so a Stop pressed
  during a first-time index build lands at once and the worker
  finishes its current embedding batch in the background.  `True`
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
  which moves the run into a fresh worktree on its own branch,
  and `commit_and_merge` when done, which commits the branch and
  squash-merges it into the original branch; a conflicting merge is
  resolved by the bundled `/merge` SEA run as a sub-agent.  The commit
  message is generated from the diff and stamped with the user's
  prompt and the task's result (`kiss.agents.sorcar.commit_message`);
  the model call that writes it is billed to the task it commits for
  (`bill_to`), and when a worktree is retired after that task's row
  is already saved (a new task, a new chat, a tab disposal) the spend
  is added to the row and its finished ancestors with a persisted
  `usage_info` event (`_retire_after_task_saved`).  A Stop during the
  merge rolls back a half-applied squash and pops the stash the merge
  made of the user's uncommitted changes (when that pop fails, the
  task's result carries a warning telling the user to run
  `git stash pop`), and a `git status` that fails keeps the worktree
  instead of deleting it.
  Both tools belong to the `edit` tool group; `WorkerSea.tools()`
  strips them, so a worker edits the caller's tree in place.  A
  worktree the agent created but never merged is presented to the
  user with the Merge / Discard bar.  `use_worktree` and `auto_commit`
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
  `create_worktree`, `commit_and_merge`),
  `"browser"`, `"memory"`, `"agents"` (`run_agent`, `agent_job`,
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
  name, and a removal that fails keeps the name on the manager so a
  later `close()` retries),
  `container:<name-or-id>` attaches to a container the caller already
  runs (commands run in its own working directory, nothing extra is
  mounted, it is left running afterwards), `exec:http://127.0.0.1:<port>`
  runs them on a remote environment that has no local container at
  all through a local HTTP exec bridge (`exec_bridge.ExecBridge`
  stands in for the Docker manager: `Bash` and
  `run_commands_parallel` POST `{command, timeout, token}` to
  `/exec` and get `{exit_code, output}` back, whole, not streamed; a
  Stop POSTs `/kill` for the token; the run fails at its start when
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

The class may also define two per-call hooks with no corresponding
`sorcar.run()` parameter or `settings()` key — a callable cannot be
JSON-serialized, so the hooks exist ONLY as SEA methods, called in
the daemon process (`sea_apply.apply_run` stages them on the command
fields below, together with `systemPromptHook` and `toolsHook`):

| Method                                 | Returns      | Staged command field |
|----------------------------------------|--------------|----------------------|
| `llm_call_hook(self, new_messages)`    | `list`       | `llmCallHook`        |
| `tool_call_hook(self, name, args)`     | `Verdict`    | `toolCallHook`       |

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
  unchanged after counting the step (`self.step`).
- **`tool_call_hook(name, args)`** — called before every tool call
  with the tool's name and arguments dict and returns a typed verdict
  (`kiss.core.tool_verdict.Verdict`, re-exported by `base_sea`):
  `ALLOW` lets the tool execute; `refuse(text)` suppresses the call
  and *text* is given to the model as the tool's result.  The
  `BaseSea` method returns `ALLOW` for every call except, at every
  tenth step of a run that has the `summary` tool, a call other than
  `summary` or `finish` while the summary is owed (see [Customizing
  every run](#customizing-every-run)).
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

A hook that returns the wrong type (a non-list, a non-`Verdict`)
fails the task with a diagnostic error, like every wrong-typed value.
In an inheritance chain the `llm_call_hook` results are threaded base
first, and the first `tool_call_hook` refusal wins.

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
run once when the agent starts, the two call hooks on every LLM and
tool call.  A method named differently from the contract
(`def model(self)`, `def add_to_tools(self)`, ...) is an
ordinary method the daemon never calls; `uv run sea lint` does not
flag such a definition, only a docstring that still describes the
earlier contract's getters (`stale-docstring`), so check the method
names yourself.


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
so keep module-level side effects cheap or idempotent.

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


## The built-in toolset

By default — no `tool_profile` setting, with or without a `tools()`
method — the agent gets `finish` (always present) and the
built-in KISS Sorcar toolset — `Bash` (with
`background=True` for detached jobs), `bash_job` (wait for / tail /
kill a background job), `run_commands_parallel` (several shell
commands at once, no LLM sub-agents), `Read`, `Edit`, `Write`,
`create_worktree` and `commit_and_merge` (the last two not in a
Docker run, whose file tools work inside the container),
`ask_user_question`, `talk`, `set_model`, `summary`, `run_agent`,
`agent_job`, `run_parallel` and `number_of_cores` (the last two only
when the run's `is_parallel` is on, the user's parallel-mode toggle),
browser tools (when `use_web_tools`), `decide` (when
`OPENROUTER_API_KEY` is configured, the `classify_with_decisions`
key in `$KISS_HOME/config.json` is on (the default; it has no settings-panel toggle)
and the decisions model is in the catalog,
`decide_tool.decisions_tool_available`), the `memory_*` tools (when
memory is enabled), and any configured skill and MCP-server tools —
**plus** your extension tools.  A restricted tool profile (a reviewer
sub-agent dispatched with `run_parallel(..., tool_profile="review")`)
filters that built-in set, including the memory tools; extension tools
are still appended.  An MCP server's tools are named
`<server>_<tool>` and built last, after the profile filter: a
synthesized name that collides with a tool the run already has (the
built-ins actually built, the static built-in list
`mcp_servers._RESERVED_TOOL_NAMES`, the tools passed to `run(tools=)`,
the tools inherited from the calling task and the MCP wrappers built
before it, passed as `make_mcp_tools(work_dir, taken_names)`) gets a
numeric suffix (`run_parallel_2`) instead of aborting the tool
registration; the `mcp_permissions` deny rules are judged on the
un-suffixed `<server>_<tool>` name, before the suffix is chosen, so a
denied tool never reserves a suffix.  A SEA's `tools()` hook runs after the MCP wrappers are
built and is not in that set: a tool it adds under an MCP wrapper's
name replaces the wrapper in the final name-keyed toolset.

### What the shell and file tools enforce

The host tools (`kiss.agents.sorcar.useful_tools.UsefulTools`, with
the guards of `shell_guards.py` and `edit_guards.py`) carry rules a
SEA's own `tool_call_hook` need not restate:

- **Output budgets.**  A `Bash` result keeps `max_output_chars`
  characters, head and tail around a marker; the default is the
  daemon's `tool_output_default_chars` (8,000, `KISS_TOOL_OUTPUT_DEFAULT_CHARS`)
  and a request is capped at `tool_output_max_chars` (50,000,
  `KISS_TOOL_OUTPUT_MAX_CHARS`).  Output over the budget is written in
  full to `tmp/tool-out/<tag>-<n>-bash.txt` under the work dir and
  the marker names the file (`UsefulTools._spill`; the tag is per
  tools instance, so a parent and its reviewer never overwrite each
  other's spills; the path is shown in POSIX form on every OS,
  `useful_tools.SPILL_DIR`, since the model pastes it into `Bash`,
  which is bash on Windows too, and the file holds the output
  verbatim, written with `newline=""`).  `memory_list(max_output_chars=...)` has the same
  default and ceiling, and a `Read` with no range keeps the default
  budget too, ending in a note that names the `start_line` to
  continue from.
- **`Read` windows.**  A `Read` of a file longer than
  `read_outline_lines` (2,000) with no range returns an outline of
  the file instead of its text when at least five definitions or
  headings are recognised in it (otherwise the first budgeted window,
  as above); a window the model has already been
  shown, unchanged since, returns a one-line "already in your
  context" note unless `force=True` (the executor clears that
  bookkeeping, `UsefulTools.forget_reads`, when a compaction drops
  the earlier copy); `start_line` past EOF is an explicit error, not
  empty content.
- **Read before modify.**  `Edit` and `Write` of an existing file the
  model has not been shown in this session are refused — except a
  `Write` under `tmp/` — and a refused `Edit` returns the numbered
  region around `old_string`, after which the file counts as read, so
  the next `Edit` goes through without a separate `Read`.  A `Bash`
  (or `run_commands_parallel`) command whose last `cat`, `sed -n`,
  `grep -n`, `nl`, `head` or `tail` segment names exactly one existing
  file, exited 0 and printed a line of it counts as a read of that
  file (`edit_guards.shell_view_target`, `output_shows_file`).  The
  bookkeeping is keyed by repository-relative path
  (`edit_guards.repo_read_key`), so a file read at the main checkout
  stays read at its `.kiss-worktrees/` path and vice versa.
- **`Edit`.**  `Edit(file_path, edits='[{"old_string": ..., "new_string": ...,
  "replace_all": false}, ...]')` applies several replacements in one
  call, in order and all or none (`old_string` / `new_string` are then
  ignored); every `Edit` result shows the numbered lines around each
  replacement in the file as it is after the whole batch (an earlier
  edit's region is reported where a later edit moved it), so no
  verifying `Read` follows.  The
  Docker `Edit` accepts the same `edits` list but applies it
  pair by pair: the first failure stops the batch and the error names
  the edits already applied, and a task Stop ends it at once with the
  stopped edit's result.
- **Refused shell commands.**  `Bash` itself returns the refusal as
  its result (these are tool-side guards, not `tool_call_hook`
  verdicts, so no re-issue reminder follows): a command that would
  delete the work dir or `/` or kill every process
  (`shell_guards.destructive_command_guard`, `Blocked: ...`); inside
  a worktree, a command naming the main checkout's path (`Error:
  command references the parent-repo path ...`, with the command
  rewritten against the worktree to copy as is); a `sed -i` whose
  script contains `\n` and a `python3 -` heredoc that reads a file,
  `.replace(` / `re.sub(` on it and writes it back
  (`edit_guards.rewrite_guard`, `REFUSED: ... did NOT run`, naming
  the `Edit` call to make instead); and a `sleep N` with N over 30
  (`useful_tools.sleep_guard`: the wait belongs in `bash_job`; the
  host `run_commands_parallel` refuses it in any of its commands too,
  since each is a foreground wait for the caller), a
  `sleep N &` included, since the child keeps the call's output pipe
  and the call blocks until it ends anyway — only a `sleep` inside a
  heredoc body the shell never runs (a quoted tag, `<<'EOF'`, with
  no `$(...)` or backtick in the body) is let through.  The
  destructive-command and rewrite guards also run on a container
  `Bash` (`docker_manager._refuse_or_lift`) and the destructive one
  on an exec bridge; the sleep guard is the host tool's only (the
  container paths, `_refuse_or_lift`, do not call `sleep_guard`; the
  bundled coding SEA adds its own, `coding_sea.SLEEP_POLL`, to its
  trials).  A
  package install or build (`apt`, `pip`, `uv`, `npm`, `cargo`,
  `make`, ...) gets at least `INSTALL_TIMEOUT_SECONDS` (900 s)
  whatever `timeout_seconds` asked, on every path.  A
  `run_commands_parallel` whose `commands` is not a JSON list of
  strings, or whose `max_workers` is negative, returns `Error: ...`
  without running anything (`useful_tools.run_commands_pool`, the
  one engine behind the host, container and bridge variants).
- **Background jobs.**  `Bash(command, background=True)` starts the
  command detached (`nohup ... > tmp/bash_jobs/<id>.log 2>&1 &`) and
  returns a job id; `Bash(..., background=True, machine="<name>")`
  runs it over `ssh` on a machine registered with `/machine add` (an
  SSH one, not a URL one), its output streaming into the same local
  log; `Bash(..., on_timeout="background")` starts the command as a
  job up front, waits `timeout_seconds` for it and, instead of
  killing it at the deadline, leaves it running under its job id
  (`machine=` without `background=True`, or any `on_timeout` other
  than `"kill"` / `"background"`, is an error).  `bash_job(job_id,
  action="wait", timeout_seconds=N, until="<regex>")` returns as
  soon as the log matches the pattern (scanned once a second), so a
  long run is one turn to start and one to collect; `"tail"` and
  `"kill"` are the other actions.  A Docker or bridged run has the
  same `background=True`, `on_timeout="background"` and `bash_job`
  (`container_jobs.ContainerJobs`, one instance per agent, so a later
  prompt in the chat can still wait on the job), done entirely with
  shell commands in the container: the job runs under `setsid nohup`
  with the exec token removed from its environment (so the deadline
  sweep of a timed-out foreground command leaves it alone), and its
  state is `/tmp/kiss-jobs/<id>.log`, `<id>.pid` (the session leader,
  whose process group is the job), `<id>.exit` and `<id>.cmd`; a
  `wait` is one polling script run in the container (the deadline plus
  `WAIT_SLACK_SECONDS`, 15 s), `until` is a `grep -E` pattern checked
  before the wait (`Error: until is not a valid grep -E regex`), and a
  job id this registry never saw is looked up by its `.cmd` file, so a
  job started by an earlier session of the task can still be followed.
  `machine=` has no container form.
- **`commit_and_merge`** copies the worktree's `tmp/PROGRESS.md` (the
  progress log the system prompt asks for; `tmp/` is git-ignored, so
  the merge itself does not carry it) into the main checkout after a
  successful merge and says so in its result.  A merge the main
  checkout's uncommitted changes block (`git stash push` failed)
  returns the blocking files from `git status --porcelain` and leaves
  the committed worktree branch in place.

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
| File raises at import time | `SEA '...' failed to import: ...` |
| No or several `BaseSea` subclasses defined in the file | `SEA '...' must define exactly one subclass of BaseSea (kiss.agents.seas.base.base_sea); found none` / `...; found OneSea, TwoSea` |
| The class cannot be instantiated without arguments | `SEA '...': MySea() raised: ...` |
| `settings()` returns something other than a dict | `settings() of SEA '...' must return a dict, got ...` |
| unknown key | `SEA '...': settings() has an unknown key 'x'; known keys: work_dir, model, chat_id, ...` |
| renamed key | `SEA '...': settings() key 'classify_tasks' was renamed to 'auto_classify'; run `uv run sea lint --fix` to rewrite the script` |
| removed key (`kind`, `preset`, `channel`) | `SEA '...': settings() key 'kind' was removed: what a SEA is became its base class: derive from `WorkerSea` (a tool-bound run on the caller's tree) or `ChannelSea` (an external-service agent) instead of `BaseSea` ...; run `uv run sea lint --fix` to rewrite the script` |
| removed key (`allow_fan_out`, `is_parallel`) | `SEA '...': settings() key 'allow_fan_out' was removed: `run_parallel` is N `run_agent` calls, so there is nothing to allow or forbid separately; `tool_profile` chooses the toolset` |
| removed key (`inherit`, `add_to_prompt` / `append_to_prompt`, `add_to_system_prompt` / `append_to_system_prompt`, `append_basic_tools`, `extends`) | `SEA '...': settings() key 'inherit' was removed: a channel never inherits from the calling task and every other SEA always does; the caller's `inherit` option opts out of inheriting`, `... key 'add_to_prompt' was removed: a SEA shapes the task text in its `prompt(task)` method; `add_to_prompt` is a `run_agent` option, not a setting`, ... (`sea_settings.REMOVED_SETTINGS` holds every entry and its explanation) |
| wrong-typed value (`bool` for a number, `int` for a `str`, ...) | `SEA '...': settings()['max_budget'] must be int or float, got bool` |
| non-finite `max_budget` / `timeout` | `SEA '...': settings()['max_budget'] must return a finite number or None` |
| `timeout` of zero or less | `SEA '...': settings()['timeout'] must be a positive number of seconds, got 0` |
| `locked` naming something other than a run setting (`locked`, `hidden`, an unknown name) | `SEA '...': settings()['locked'] may only name settings keys (work_dir, model, chat_id, ...); got ['hidden']` |
| `description()` of a `/<name>` command returns an empty or blank string (a non-string is the `must return a string` row above) | `SEA '...': description() must return a non-empty string` (raised by `/<name> help`, `sea_commands.sea_description`; `/<name> check` reports it as its text) |
| `prompt()` returns an empty string | `prompt() of SEA '...' must return a non-empty string` |
| a method `X` raises | `X() of SEA '...' raised: ...` |
| a method returns the wrong type | `system_prompt() of SEA '...' must return a string, got ...`, `tools() of SEA '...' must return a list of tool callables (not a file path), got ...`, `llm_call_hook() of SEA '...' must return a list, got ...`, `tool_call_hook() of SEA '...' must return a Verdict (ALLOW, or refuse(text), from kiss.agents.seas.base.base_sea), got ...` |
| a value whose own methods raise (e.g. a `str` subclass with a raising `__str__`) | `prompt() of SEA '...' returned a broken value: ...`, `SEA '...': settings()['max_budget'] returned a broken value: ...` |
| a `sea_class(spec)` base that does not exist | `sea_class('x'): not a registered SEA command; known commands: ...`, `sea_class('x.py'): not an existing Python (.py) file` (raised while the file executes, so reported as an import failure) |

Overrides are **atomic**: if loading the class, `settings()` or
`prompt()` fails, the command keeps all its original values (no
partial overrides).  `system_prompt()`, `tools()` and the two call
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
`is_stall_error`) — is a transport hiccup, not a model failure: it
is retried on its own counter, and only twelve of them
(`MAX_CONSECUTIVE_STALLS`) end the run (`... failed with 12
consecutive stalled streams`).  Both counters are reset by a
step that completes (a switch to the fallback model resets the error
count only), so a stall between two ordinary errors does not reset
the error count, nor an ordinary error the stall count.  A
non-streamed (callback-less) OpenAI-compatible request has no stall
clock, since a reasoning turn may legitimately send nothing for
minutes, but it is still stoppable: the SDK call runs on a worker
thread while the task's stop event is watched, and a Stop shuts the
pending socket and drops the client so the SDK does not re-send the
request (`OpenAICompatibleBase._blocking_request`; a reply that
arrived anyway is billed as a rejected response).  A safety refusal
(`stop_reason="refusal"`, `ModelRefusalError`) or repeated empty
responses switch to the fallback model when one is configured; with
none, a refusal is retried once with `kiss_agent.REFUSAL_RETRY_MSG`
appended as a user message (the task is the user's own authorized
work in a sandbox; do the legitimate parts and say in the summary
which part was declined), and a second refusal ends the run.

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

### Refusals and session hand-off

What the executor (`kiss.core.kiss_agent.KISSAgent`, driven by
`kiss.agents.sorcar.relentless_agent.RelentlessAgent`) does around a
SEA's hooks, so a hook author knows what the model reads:

- **Refusals.**  A `refuse(text)` verdict from any `tool_call_hook`
  layer, or a refusal from the framework's own guard
  (`SorcarAgent._guard_tool_call`, first reason wins: `finish` while
  a user message queued during the step is still undrained, every
  time; `finish` while `run_agent` jobs are still running, once per
  run; `finish` while a worktree is unmerged, once per run;
  `Write` / `Edit` of a non-git-ignored path inside a repository —
  when a worktree the work dir is not in already isolates that
  repository the refusal names the path to use inside it, otherwise
  it asks for `create_worktree` first; and, in a run whose toolset
  has `Edit`, a `finish(success=True)` whose `summary_in_html`
  says the result was guessed or not verified — the evidence gate,
  `evidence_gate.EvidenceGate`, once per run: the refusal quotes the
  `HEDGE_PHRASES` it found ("educated guess", "unverified", "could
  not test", "should work", "no answer key", and the phrases that
  leave a reading of the task open at the end, "if the hidden test",
  "judgement call", "left open", "remains ambiguous", ...) and asks for a
  checklist of requirements each with the command that checked it, or
  a `success=False` that names what is unverified; a second `finish`
  always passes, and a read-only run such as a reviewer is never
  gated because it reports rather than claims work), reaches the model as
  `tool_verdict.refusal_result(name, text)`: `REFUSED: the <name>
  call did NOT run and had no effect. <text>`.  The executor
  remembers the refused tool; the next allowed call of a different
  tool gets `tool_verdict.REISSUE_REMINDER` appended to its result,
  a call of the refused tool itself clears the reminder silently, and
  `finish` never carries one.
- **Context compaction.**  Before a model call, once the context has
  grown past `compaction_start_tokens` (100k tokens, then every
  `compaction_step_tokens` of growth), old large tool outputs are
  considered for replacement with stubs
  (`kiss.core.context_compaction`), but never within
  `COMPACTION_MIN_STEP_GAP` (30) steps of the previous compaction:
  every compaction is a full prompt-cache miss, so two of them a few
  steps apart would pay twice for one batch of stale outputs.  The
  newest six results, results of 500 characters or fewer and those of
  `Edit` / `Write` / `finish` are never stubbed, and the compaction
  is skipped (re-planned on a later step) unless it drops at least
  25% of the context while the context is still under 80% of the
  hand-off limit.  `KISS_TOOL_OUTPUT_COMPACTION=0` disables it.
- **Routine turns.**  The turn after a refused call, a `summary`,
  `memory_search`, `memory_pull`, `memory_read`, `memory_list`,
  `memory_write` or `memory_refresh` call (`kiss_agent.ROUTINE_TOOLS`;
  not `memory_delete`), a `bash_job` / `agent_job` wait that returned with
  the job still running, or an unchanged `Read` is a routine turn
  (`kiss_agent.is_routine_turn`).  The `routine_turn_effort` config
  key (`KISS_ROUTINE_TURN_EFFORT`, empty by default) names a lower
  reasoning effort (`low`, `medium`, ...) for such turns on a model
  that supports thinking; `cost_report` counts routine calls and
  their output tokens.  Empty changes nothing.
- **Session hand-off.**  When a session of a task ends before the
  task does — the executor's context reached its hand-off limit
  (`ContextWindowExceededError`), or it hit its step limit or another
  uncaused `KISSError` after the first step (a budget overrun instead
  ends the task: a sub-agent returns a partial result quoting what it
  did, a top-level task fails as budget-exceeded; three consecutive
  step errors, or any other exception, end it with a failed result)
  — the next session starts from a summary of the ended one.  A
  session the model ends itself with `finish(success=False,
  is_continue=True)` continues from the summary the model wrote; on
  the exception path the summary is written by
  a tool-less helper (`finish` only, at most `SUMMARIZER_MAX_STEPS`
  = 3 steps, its spend billed to the task) from a scripted digest of
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
or sampling parameters:

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
on `PATH` for `cc/*` / `codex/*`).  So replace `my-custom-model` above
with a catalogued or `MY_MODELS.json` name; otherwise the task fails
with `No model available.  Set at least one API key in the
environment.`

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
(`anthropic_model._effort_level_for`; a caller's own
`output_config.effort` wins, and an entry that rejects the parameter,
such as Haiku 4.5 or Sonnet 4.5, carries no `thinking` and gets
none).  A custom endpoint whose name no catalog entry matches gets no
effort parameter.

`$KISS_HOME/MY_MODELS.json` is the custom-model registry the
settings panel's Add / Edit / Delete buttons maintain
(`model_info.save_custom_model`, `delete_custom_model`); an entry is
keyed by model name and holds the endpoint fields the panel edits
(`endpoint`, `api_key`, `headers`, each dropped when blank) plus the
loader-required `context_length`, `input_price_per_1M` and
`output_price_per_1M` (an entry missing one is skipped with a
warning at load).  A brand-new entry is seeded
(`_seed_new_custom_entry`) from the catalog model its name resolves
to — the exact name, a harbor-prefixed form such as
`openai/gpt-6.1-sol`, or a `bedrock/<id>` profile of a `claude-*`
entry, the one rule of `_lookup_model_info` — copying that model's
context length and every per-million price field
(`_PER_1M_PRICE_FIELDS`: input, output, cache read, cache write, 1-hour
cache write, audio input, audio output, image output; a Bedrock
profile's prices are scaled by `bedrock_price_factor`), so the entry
that replaces the catalog one at the next load bills every token
kind as the catalog did; a name no catalog model matches gets the
generic defaults (128,000 context, $0 prices,
`_CUSTOM_MODEL_DEFAULTS`).  An edit merges into the existing entry,
so hand-written keys such as `thinking` or real prices survive an
endpoint change; a name not yet in the loaded catalog is looked up
in the file as it is now, so a model saved in the panel is priced by
the running daemon without a restart.


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

import json
import sqlite3
import threading
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.core.config import kiss_home


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
```

`settings` is the run's `task_settings` event (its effective model,
work directory, budget, SEA, kind, tool profile, timeout, inherited
and overridden values; `kiss.agents.sorcar.run_config`), empty when
the run ended before the daemon emitted it.

What `run()` raises (all in `kiss.agents.sorcar.daemon_client`):
`ValueError` for an empty prompt or a `sea_path` that is not an
existing `.py` file; `ConnectionError` when no daemon answers at the
endpoint or it drops the connection before the task ends;
`TimeoutError` when the task outlives `timeout` (the client closes
the task's tab and disconnects, the task keeps running), or, with
`stop_on_timeout`, its subclasses `StoppedOnTimeoutError` (the daemon
confirmed the stop; `.result` carries the stopped task's spend) and
`StopUnconfirmedTimeoutError` (the stop was sent but never confirmed
within the 20 s grace, so the task may still be running; `.result`
carries what it reported so far); and `RunRefusedError` when the
daemon turns the `run` down with an `error` event of `code:
"prompt_refused"` (a self-update being installed is the case an API
client can hit): nothing ran or was spent, so the wait ends at once
with the daemon's reason instead of lasting the whole `timeout`;
and `CancelledError` when the caller sets the `cancel` event: the
client sends the `stop` and waits for the task's terminal status
within the same 20 s grace, `.confirmed` tells whether it arrived
and `.result` carries the spend reported so far (no second `stop`
is cascaded for it, nor for the timeout and refusal errors above).
Every other abort of the wait — above all the `KeyboardInterrupt`
injected when the calling task is stopped inside a `run_agent`
dispatch — also sends the daemon a `stop` for the dispatched task's
tab, and the exception carries a `task_result` attribute with the
spend reported so far, so the caller can still charge it.

`cost` is the exact
figure the daemon's final `result` event carries as `cost_usd` next
to its four-decimal `cost` display string (`daemon_client._event_cost`;
a later `usage_info` event, say a side channel's spend that lands
after the row was saved, updates the totals, and a delta the daemon
has already banked on a waiting ancestor, `ancestor_charged`, is not
folded in twice); the daemon persists the task's final cost unrounded
as well, so a parent row sums its children's exact spend.  The same
write records `model_costs`, the task's own `llm_call` events summed
per model as `{"model": [cost, tokens]}`
(`persistence._own_model_costs`), so the sidebar's Spend subpanel
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
  count, and helper classes that do not derive from `BaseSea` are
  free.  A method is applied only when the class (or a base) defines
  it, so leave out what you do not change rather than returning the
  argument unchanged; never call `super()` in a chained method
  (`settings`, `prompt`, `system_prompt`, `tools`, the two hooks), the
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
  earlier ones, the bundled Sorcar-extending SEAs in
  `src/kiss/agents/seas/` come next, so a `SEAS.md` folder can shadow
  them, and the built-in `cron` (`sea_commands.BUILTIN_COMMANDS`) has
  the lowest precedence of all.  Of the 27 bundled SEA folders, 25 register
  a command and two (`coding`, `sorcar`) are hidden: `/ask` (answers a question about the current task
  in two or three sentences from one `task_context` call over its
  status, spend, progress log and digested persisted events; its only
  tools are `task_context` and `finish`, so it never touches the task's
  files or shell; typed into a running task's
  tab it runs as a side channel that always dispatches the bundled
  `seas/ask/ask_sea.py`, even when a `SEAS.md` folder shadows the
  command, with the files attached to the question forwarded as the
  answering run's `attachments`, and the task-info panel's Task update is the same agent
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
  button — and never twice at once; the child is bound to the task's
  stop event, so Stop ends the update together with the task, the
  panel reporting "The task was stopped before this update finished."),
  `/autorouter` (runs a task on the cheapest model tier
  that will finish it; its `settings()` fixes the routing session's model to
  `autorouter_sea.ROUTER_MODEL`, `openrouter/moonshotai/kimi-k3`, with
  `auto_classify` off and `use_web_tools` and `use_memory` on), `/bestrouter` (runs a task on
  `claude-fable-5-1` and has `gpt-6-astra` review it read-only, dispatched
  with a non-blocking `run_agent(model=..., tool_profile="review",
  wait="false")` on at most 75% of the budget and collected with
  `agent_job`), `/orautorouter` and `/orbestrouter`
  (the same two routers with every candidate taken from the OpenRouter vendor:
  `openrouter/...` tiers, and `openrouter/anthropic/claude-fable-5.1` reviewed by
  `openrouter/openai/gpt-6-astra`), the hidden
  `coding` SEA (unattended coding in a Docker container, or on a
  remote sandbox through an `exec:<url>` bridge, for benchmark
  trials; its `settings()` is `{"hidden": True}`, so there is no
  `/coding` command: the trial runners load the generated per-trial
  SEAs, which expose its `ContainerHarness` methods, by path.  Its
  `tool_call_hook` is the model of a strict finish gate: with the
  trial config's `finish_gate` on, every `finish(success=True)` goes,
  up to the first refusal, through the agent's own verifier script
  (`/tmp/kiss-verify.sh` in the container must exist and exit 0
  within `verifier_timeout`, 900 s by default), an independent check
  written once per trial by the config's `blind_verifier` model (or
  the trial's own model with `"same"`) from the task statement alone,
  without seeing the agent's work (`/tmp/kiss-blind-verify.sh`,
  authored with a shell in the container, 40 steps and a tenth of the
  trial's budget; a blind check its author produced nothing for is
  skipped, not refused; the two verifiers share at most three refusals
  per trial, after which neither is run and the finish counts as
  unverified), the SEA's own `EvidenceGate(track_changes=True)` — the
  hedge check of the executor's guard (see "Refusals" above) plus a
  second reason: a file was changed (`Edit`, `Write`, or a shell
  command that writes, `evidence_gate.WRITING_COMMAND`) and no shell
  command ran since, so nothing verified covers the final state; one
  refusal per trial, lifted when the verifier passes — a once-per-trial
  refusal of a summary that the decisions model finds leaving a case
  of the statement open (`leaves_a_case_open`, threshold 0.7), the
  open findings of the last review that the summary answers neither
  with a change nor with a quoted command and its output
  (`answered_findings`), and a read-only review by the configured
  `reviewer` model, up to `max_reviews` (3) per trial, a re-review
  seeing its previous findings and only the files changed since; its
  `NOT DONE` findings are the refusal, a review without a `VERDICT:`
  line is retried once without the run's per-model overrides, so at
  the reviewer's catalog thinking level, and a review that still has
  none, or whose model call raised (no retry), counts as failed and
  does not use up a review: the first failed review refuses the
  finish, a second one (`MAX_FAILED_REVIEWS`, 2) ends the reviewing
  for the trial and lets the finish through.  As soon as
  the verifiers that ran pass, the work tree is snapshotted
  as the `verified` checkpoint, which the trial runner restores
  (`submit_best_checkpoint`) when the run ends without an accepted
  finish and the tree on disk no longer passes those checks.  With the
  gate on, the first `Edit` or `Write` under the workdir is refused
  (at most twice) until the agent has written its interpretation
  ledger, `/tmp/kiss-ledger.md`.  A text-only turn, the implicit
  finish that consults the hook with no arguments, is vetoed once with
  a request for an explicit `finish`, and a second one ends the run
  (logged as `implicit_finish`; the trial runner counts it as a
  failure).  Outside the gate, the harness's toolset drops `decide`
  and `summary` and adds `checkpoint(name)` / `rollback(name)`
  snapshots of the workdir, answers `ask_user_question`, `talk`,
  `run_agent` and `set_model` itself, gives `Bash` a 120 s default
  timeout, refuses a foreground `sleep` of 10 s or more
  (`SLEEP_POLL`), a here-document that rewrites an existing file, and
  a command that would kill every process or delete the workdir), `/discover` (runs the AI discovery
  loop — baseline and held-out split, search, ideas, pairwise judging,
  implement and evaluate, keep a Pareto frontier of the winners and
  compose them, fresh ideas from `gpt-6-astra` or `gemini-4-argon`,
  stop on a goal met on the held-out split or on the budget or round
  limit — on the `bestrouter` with
  the full toolset; the loop, `discover_sea.DISCOVERY_INSTRUCTIONS`, was
  the `## AI discovery, auto research, and optimization` section of `SYSTEM.md`
  and is now the SEA's alone: its `system_prompt()` appends it at the
  end of the assembled prompt, cutting out a leftover copy first, and its
  `settings()` is `{"model": "bestrouter", "tool_profile": "full"}`;
  `/revise_and_review_paper` puts the same text at the end of every
  `writer_task`), `/forget`
  (removes a standing instruction from `$KISS_HOME/AGENTS.md`),
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
  per login (`machines.SshTunnel`, from a local loopback port to
  `127.0.0.1:<port>` on the machine, kept for the daemon's lifetime and
  reopened on the next connection once its ssh process has died), so
  the daemon port need not be reachable from here, only the ssh port;
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
  certificate and checks the password with one handshake before saving;
  the name defaults to the host; `/machine remove <name>` forgets it,
  `/machine list` shows each machine's address and reachability; once a
  machine is registered the Chats panel lists its tasks next to the
  local ones, de-duplicated by task id, in the same order as the local
  list (running first, then latest finished first) and filterable by
  machine name (the link to each machine keeps its sockets alive with
  WebSocket pings, so an idle hour does not drop it, and a request
  whose previously kept socket has closed is sent once more over a
  fresh one; a stream socket's handshake (`MachineLink.open_connection`,
  a chat tab's or the machine's view stream) runs outside the link's
  lock, so it connects beside a request in flight on the shared socket
  and two streams opened together by a bind connect at the same time,
  while `retire` waits for the handshakes in flight before the removed
  machine's tunnels are swept; after a failure the link reports the machine down without
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
  from a parked tab too)
  (its own web UI is never opened; a sub-agent tab that joins the
  task's stream, through `resumeSession` or a replayed
  `openSubagentTab`, has the binding announced to the surfaces
  (`tabMachine`) before its transcript, as the bound tab's is, so the
  drop-down shows the machine for the sub-agent tab as for its parent,
  and a surface's `ready` replays every binding, the bound tab's and
  then its sub-agent tabs', before any transcript; the chat header then shows the
  machine and that machine's working directory, with the directories
  picked there before in the drop-down, so a pick or the folder picker
  changes the machine's working directory, never this daemon's --
  `getMachineWorkDirs`, and `setWorkDir` with `machine`, which the
  daemon proxies to the machine (these round trips, and `machineAction`'s
  add/remove with its handshakes to an unreachable host, run as
  background tasks serialised by `ServerApi._machine_lock`, so a slow
  machine never holds up the connection's `submit`, `stop` or
  `ping`); the Explorer and Source Control views
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
  the tokened reply settling the request without opening a tab); a
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
  machine with it (an emptied password is not adopted); the sidebar's
  Schedule, Apps, Spend and Recommendations subpanels show the machine
  too, their commands (`getCronJobs`, `cronAction`, `getAppsStatus`,
  `getSpendReport`, `getRecommendations` --
  `sidebar_panels.MACHINE_PANEL_REPLIES`) forwarded the same way on a
  link lane of their own (`machines.link_for(machine, "panels")`, so
  the up-to-45-second app probe holds up no submit, settings or
  Explorer request), an unreachable or refusing machine answering with
  an empty reply of the expected type plus the `error`, while the
  Machines subpanel is this daemon's registry and is never forwarded;
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
  name its daemon reports with every ping), and `run_agent(machine=...)`
  can run a sub-task on it; with the `https://` form the password may be left out only for a
  daemon that has no remote password set and is reached on a loopback
  port), `/merge`
  (resolves git merge conflicts and stages the resolved files; commits
  only when asked), `/recommendations` (a `WorkerSea` whose tools are
  `finish`, `recent_tasks`, `connected_apps`, read-only memory and
  `save_recommendations`: it reads your own top-level tasks
  (`persistence.user_tasks_since`; the last 24 hours and, further back,
  enough older tasks to review at least 20 completed ones; cron runs and
  its own runs are dropped), asks the auth-status probe which third-party
  apps you are signed in to and memory what you did in them, picks ten
  concrete follow-ups, at least three of them combining the signed-in
  apps, or ten starter tasks for a new project when there are none,
  and stores them in `$KISS_HOME/recommendations.json`, which the
  sidebar's RECOMMENDATIONS subpanel shows; `/recommendations now` runs it
  once, and the 6-hourly default cron job keeps the file fresh; both
  default jobs come from `cron_agent.DEFAULT_JOBS`, seeded once per home
  by `seed_default_jobs` at daemon start so your edits and removals
  stand; before the seeding, `cron_agent.adopt_previous_home_jobs`
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
  a venue, scoring seven dimensions from 1 to 10), `/review_patent` (reviews a patent
  application draft or a published patent like an examiner: prior-art search, eligibility,
  novelty, obviousness and definiteness claim by claim, scoring seven dimensions and the
  allowance likelihood, with `read_patent`, `check_patent` and `check_patent_review`
  tools), `/revise_and_review_paper` (writes a paper with
  `/write_paper`, has `/review_paper` review it fresh, and repeats until
  strong accept or no further improvement; task text carries `Writing:`
  and `Review:` instructions), `/rsi7d` (7-day self-improvement of the
  indexed SEAs from their recorded runs; the task text starts with the
  scope: `/rsi7d all`, `/rsi7d <name> [<name> ...]` for those SEAs only,
  or `/rsi7d --seas-dir <folder> [<name> ...]` for the SEAs of that
  folder, which then are the ones it may edit), `/sh`
  (runs the command with the `bash` tool profile), `/skillopt`
  (optimizes the prompt text of a skill or SEA against an eval set),
  `/sync` (commits the uncommitted changes of the tab's working
  directory on its current branch — `git add -A`, a commit whose
  message names their number and the host — then fetches `origin/main`,
  merges it into that branch and pushes, through its `git_sync` tool
  (`seas/sync/git_sync.py`: commit, fetch, merge, push, with the round
  retried when the remote moves during the push; a tree with unmerged
  paths left from an earlier merge is refused rather than committed);
  a merge that stops on
  conflicts is left in progress for the agent to resolve with the
  `shell+edit` tools and finish with a second `git_sync` call; its
  `prompt()` supplies a default task, so a bare `/sync` runs it, and
  `/sync <remote>/<branch>` merges another remote branch; when machines
  are registered the `prompt()` also lists them and the agent then
  syncs each one in turn through `run_agent(agent="sync",
  machine=<name>, task="<ref> of <the remote's URL> this machine only",
  max_budget="2")`, with no `work_dir` option, waiting for each before
  the next, and calls `git_sync` here once more; the sub-task runs in
  the machine's own folder and its `git_sync(repo, ref, url)` syncs
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
  $2 per checkout, this one and one per machine, since a sub-task's
  spend counts in the caller's),
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
  waves — connect and read each peer's clock and memory fingerprints,
  pull everything from every peer into this machine, then push this
  machine's union to every peer — so one run converges every peer
  that completes all three waves (a peer that fails in any wave is
  left out of the later ones and its report says why it was skipped
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
  fails is reported with the reason and not retried),
  `/task_update` (reports what a running task has done so far),
  `/write` (writes prose for a general audience in concise, professional
  American English that reads as human-written; it defines only
  `description()`, a `system_prompt()` that appends its style guide
  and a `settings()` of `{"timeout": 3600.0}`), `/write_paper` (writes
  or revises a research paper), `/write_patent` (writes or revises a utility patent
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
