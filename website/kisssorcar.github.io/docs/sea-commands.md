# Slash Commands for Sorcar Extension Agents

> Every Sorcar Extension Agent (SEA) is a folder `xxx/` that holds the script `xxx_sea.py` plus the helper modules and data files the agent needs. Every such folder the `kiss-web` daemon can see becomes a chat command `/xxx`, named after the folder. The bundled channel agents (`/slack`, `/gmail`, `/github`, ...) are registered automatically; you add your own by listing the parent folders in `$KISS_HOME/SEAS.md`. `/xxx help` prints the SEA's one-sentence `description()`. This page describes the layout and what happens between typing `/xxx` and the SEA running, so you can plug in your own agents without reading the source.

## What a slash command does

Typing this in the chat box of the VS Code extension or the web app:

```
/slack tell #eng that the deploy is done
```

runs the `slack/slack_sea.py` agent directly, in the tab's own run, on the task "tell #eng that the deploy is done": the daemon makes that file the run's agent script and your text its prompt, so the SEA's `settings()`, system prompt and tools apply to the very session you are looking at. There is no relay turn in which a model is told to call `run_agent`, no nested sub-agent tab, and no chance for the model to pick a different agent or explore the code first. A slash command is therefore the predictable way to invoke a specific SEA; from inside a running task, `run_agent(agent="slack", task="...")` dispatches the same SEA as a sub-task.

Any file that is a valid SEA works: a bundled channel agent, or a file of your own whose `settings()` dict and `description()`, `prompt(task)`, `system_prompt()`, `add_to_system_prompt()` and `add_to_tools()` functions configure the run. `prompt(task)` receives the text after the command and returns the prompt body; `settings()["extends"]` lays another SEA (by command name or path) under yours, and the model picked on the tab (`bestrouter`, `autorouter`) is the outermost layer of every run there. The SEA file format is described in [Client Interfaces: Sorcar Extension Agents](cli.md#sorcar-extension-agents-seas).

## Where commands come from

The daemon builds the command list from three sources:

1. **Bundled channel agents** in `src/kiss/agents/third_party_agents/` of the installed package. These always win a name clash.
2. **Bundled Sorcar-extending agents** in `src/kiss/agents/seas/` of the installed package: `/sh`, `/ask`, `/merge`, `/write_paper`, `/review_paper`, `/remember`, `/forget`, `/task_update`, `/rsi7d` and the rest of the [bundled commands table](#bundled-commands) below, which is generated from each SEA's `description()`. Any `SEAS.md` folder may shadow these.
3. **Your folders** listed in `$KISS_HOME/SEAS.md` (`$KISS_HOME` is the Sorcar home: `~/.kiss` for the stock brand, `~/.s10s` once Seamless Loop is installed, or the `KISS_HOME` environment variable).

The command name is the SEA's folder name: `deploy/deploy_sea.py` is `/deploy`, `pr-review/pr-review_sea.py` is `/pr-review`. A folder is registered only when it directly contains a regular file named `<folder>_sea.py` and the folder name uses ASCII letters, digits, `_`, or `-` only. `release.notes/` (a dot) or `my agent/` (a space) is skipped silently because the command could not be typed, and so is a folder whose script has another name (`deploy/main_sea.py`). A leading underscore is allowed: `_scratch/_scratch_sea.py` is `/_scratch`. A loose `xxx_sea.py` placed directly in a listed folder is not a command: every SEA needs its own folder, which is also where its helper modules, prompts and data files live.

## SEAS.md syntax

`$KISS_HOME/SEAS.md` is a plain text file with one folder per line. Despite the `.md` suffix, it is not rendered Markdown; the rules are:

| Rule | Example |
| --- | --- |
| One folder path per line | `/opt/team-agents` |
| Blank lines are ignored | |
| A line starting with `#` is a comment | `# personal agents` |
| ` #` (a space, then `#`) starts an inline comment | `~/my-seas   # personal` |
| `~` and `$VARIABLE` are expanded | `$WORK/agents` |
| A relative path is resolved against your home directory, not the current project | `my-seas` means `~/my-seas` |
| Spaces inside a path work verbatim; no quoting or escaping | `/Users/alice/team seas` |
| Missing or unreadable folders are skipped | `/does/not/exist` has no effect |

Do not shell-quote paths. `"~/my seas"` would look for a folder whose name starts with a literal quote. Backslashes are kept as typed, so Windows paths such as `C:\Users\alice\seas` need no escaping.

A complete example:

```
# $KISS_HOME/SEAS.md
~/my-seas              # personal agents
$WORK/agents           # agents shared by the team
/opt/agents/experimental
```

### Precedence when two folders define the same name

Later lines override earlier ones: the folder at the bottom of `SEAS.md` beats the folder at the top. The bundled `third_party_agents/` folder beats every `SEAS.md` folder, and every `SEAS.md` folder beats the bundled `seas/` folder (the Sorcar-extending agents such as `/merge`, `/sh` and `/task_update`). In the example above, if `~/my-seas` and `/opt/agents/experimental` both contain `deploy/deploy_sea.py`, `/deploy` runs the experimental one; if either contains `slack/slack_sea.py`, `/slack` still runs the bundled Slack agent; if either contains `merge/merge_sea.py`, `/merge` runs your copy instead of the bundled one.

### Changes take effect while the daemon runs

The daemon rescans `SEAS.md` and every listed folder every 2 seconds. Adding a line, dropping a new `xxx/xxx_sea.py` into a listed folder, or deleting one updates the command list without a restart, and every open chat receives the new list, so the autocomplete popup updates too. A command that has just appeared is also resolved on first use even if the scan has not run yet.

## The slash-command flow

```
  you type "/deploy ship v2.3" and press Enter
        |
        v
  chat client (VS Code extension / web app)
    - shows a "Commands" popup while you type the command word
    - submits the prompt unchanged
        |
        v
  kiss-web daemon
    - prompt starts with "/deploy" and "deploy" is registered?
        |                       |
        | yes                   | no  -> handled as an ordinary prompt
        v
    - the tab's run becomes an agent-script run:
        agent script = /abs/path/to/deploy/deploy_sea.py
        prompt       = ship v2.3
        |
        v
  daemon imports deploy_sea.py, applies its settings(), system prompt
  and tools to this run, and the model works on "ship v2.3"
        |
        v
  the result is the task result in your tab
```

Details worth knowing:

- **Autocomplete.** As soon as the first character in the box is `/`, a popup lists the matching commands (substring match, case-insensitive). Pick one with the mouse, or move with the arrow keys and press Tab or Enter; the client inserts `/name ` with a trailing space and closes the popup. Once you type a space after the command word, the popup hides and normal `@file` mentions and ghost-text suggestions resume.
- **Only at position 0.** The command must be the first character of the prompt with no leading whitespace or blank line. `/deploy` on a later line of a multi-line prompt is ordinary text.
- **Exact word match.** `/deployx ship` looks up a command named `deployx`; it does not match `/deploy`.
- **Text is required.** `/deploy` alone, or a `/name` that is not registered, is not a command: it is sent to the model as a normal prompt and answered like any other question.
- **`/deploy help` shows the description.** One of the two reserved sub-tasks (any letter case, nothing after it): the daemon does not run the SEA or call a model; it imports `deploy_sea.py`, calls its `description()` and posts the returned sentence as the task result. A SEA without a callable `description()` returning a non-empty string gets a diagnostic instead.
- **`/deploy check` shows what a run would use.** The other reserved sub-task executes the script (and what it `extends`) without running a task and posts: the layers, the effective settings after the kind's defaults, the model a run takes, the names of the tools `add_to_tools()` adds, which getters and hooks are defined, and the prompt `prompt(task)` yields for a sample task (`{task_id}` shown as `<task id>`). A broken script yields the first error in the words the daemon would use, so a settings typo is found before the first run.
- **`<task>` blocks stay intact.** A prompt such as `/deploy <task>build</task><task>publish</task>` reaches the SEA as one sub-task with the tags in place; the daemon does not split it into two Sorcar tasks.
- **History keeps what you typed.** The task panel, the task list, the chat history, and the frequent-task chips record `/deploy ship v2.3`; the model sees `ship v2.3` as its task.
- **Where the run happens.** The SEA runs in the tab's working directory with the tab's settings (model, worktree, auto-commit, ...) unless its own `settings()` say otherwise: `/sh` declares `{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}`, so it runs with the Bash tool alone, directly on the checkout, without a worktree or auto-commit. A tab passes nothing explicitly, so for every run setting the SEA's `settings()` apply over the tab's values (see the precedence rule below).
- **How long it may run.** A slash-command run is the tab's own task and runs as long as any chat task would. The `timeout` setting matters when another task dispatches the SEA through `run_agent`: the call waits that many seconds (`/write_paper` 6 h, `/review_paper` 2 h, `/revise_and_review_paper` 24 h, `/write` 1 h), else 3600 seconds, then stops the sub-task; an explicit `timeout` argument to `run_agent` overrides both.

## Running a command's SEA from another task

A running task reaches the same SEAs through its `run_agent` tool: `run_agent(agent="sh", task="git status --short")` runs the `/sh` SEA as a sub-task (own tab, own history row, result returned as YAML). `agent` is resolved by three rules: empty or a generic label such as `"general"` or `"assistant"` runs a plain Sorcar sub-agent (`"reviewer"` is refused: a reviewer is a plain sub-agent with `tool_profile="review"`); a path (a `.py` suffix or a path separator) runs that script; a registered command name runs that command's SEA (`"write_paper"`, `"sh"`, a channel such as `"slack"`, `"cron"`, a `SEAS.md` folder; case, spaces, hyphens and underscores are ignored). Anything else is an error naming the closest command. The arguments are the same for `run_agent(task, agent, model, tool_profile, max_budget, timeout, options)` and `run_parallel(tasks, agent, model, tool_profile, max_budget, timeout, max_workers, options)` (`run_parallel`'s `timeout` bounds each child); `options` is a JSON object in the SEA settings vocabulary (`work_dir`, `workspace` for a multi-account channel, `add_to_prompt`, `add_to_system_prompt`, `inherit`, the booleans, ...) such as `{"use_web_tools": false}`. `run_agent(..., wait="false")` returns a job id at once instead of blocking; `agent_job(job_id, "wait" | "tail" | "kill")` then returns the result, reports the status or stops the sub-task.

A SEA runs in one of three ways; the table says what differs:

| | `/<name> task` in a tab | `run_agent(agent=<name>)` | `run_parallel(agent=<name>)` |
|---|---|---|---|
| Where | the tab's own run | a daemon sub-task with its own tab | a thread of the calling task |
| Settings honoured | all but `timeout` | all | all; a pinned `use_worktree`, `auto_commit` or `auto_classify: true` or a `chat_id` is refused with an error |
| Inherits from the caller | nothing (the tab's settings apply) | model, budget share, chat, prompt suffixes, tools, container — unless the SEA is a `channel` or the call passes `inherit: false` | the same; the budget is shared among the children |
| `timeout` | none | argument, else the setting, else 3600 s | per child: argument, else the setting, else none |
| `kind: "channel"` SEAs | allowed | allowed | refused |

The sub-task's settings follow one rule, generated here from `sea_settings.PRECEDENCE_RULE`:

<!-- sea-docs: precedence -->
> For every setting of a sub-task: what the call passes explicitly (a `run_agent` / `run_parallel` argument or `options` key) wins, then the SEA's `settings()`, then what the calling task passes on (and, for `/<name>`, the chat panel's persisted settings), then the user's defaults. A SEA may list keys in `locked`: a call that passes a different value for a locked key is refused with an error, never silently overruled.
>
> For example, `/sh` declares `{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}`, so `run_agent(agent="sh", task=..., tool_profile="review")` is refused with `Error: sh: the script locks tool_profile='bash' (asked for 'review')`, while `run_agent(agent="sh", task=..., model="gpt-5")` runs it with that model: `model` is not locked, so the explicit argument wins.
<!-- /sea-docs -->

The example is generated from the bundled `/sh` by `sea docs`, and `sea lint` checks every sentence of the form "/name declares {...}" in these pages against the script's `settings()`. A `/<name> task` run in a tab passes nothing explicitly, so there the SEA's settings apply over the tab's. See [Client Interfaces: Dispatching SEAs from a task](cli.md#dispatching-seas-from-a-task-with-run_agent-and-run_parallel).

Every `run_agent` and `run_parallel` result starts with what the sub-task actually ran with, so the calling model does not have to guess:

```yaml
ran: sh (worker) model=gpt-5 tools=bash budget=$1.00 timeout=3600s inherited=model,chat_id,max_budget pinned=use_worktree(True->False)
success: true
summary: ...
```

`inherited` lists the settings the sub-task took over from the calling task; `pinned` lists each inherited or default value the SEA's `settings()` replaced, as `key(before->pinned)` (here `/sh` ran without a worktree from a task that uses one). An explicit argument never appears under `pinned`: it either won or the call was refused. A plain sub-agent whose worktree default the pre-run classifier dropped (a non-development task) ends with `classified=use_worktree(True->False)`; an explicit `use_worktree` option is never dropped. `tools=review(inferred)` says nobody named the profile: the sub-agent is a reviewer (a child of one, or one whose task reads as a review and asks for no changes) and got the read-only toolset. A SEA reached by its path that is also a registered command ends the line with `(also agent="name")`, the shorter spelling for the next call. The same record is persisted in the sub-task's `task_settings` event (keys `sea`, `kind`, `tool_profile`, `tool_profile_inferred`, `timeout`, `inherited`, `pinned`, `classified` next to `model`, `work_dir` and `max_budget`), which the task panel shows and `rsi7d` mines.

## The settings vocabulary

The tables below are generated from the code by `uv run sea docs` (`uv run check` keeps them current), so they are the normative list: a key that is not here is an error. `settings()` returns a dict of these keys; `run_agent` / `run_parallel` `options` take the same keys (less the ones a sub-task cannot take as data, plus the call-only `inherit`, `workspace`, `add_to_prompt` and `add_to_system_prompt`).

### `settings()` keys

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
| `timeout` | `int \| float` | — | Seconds the run may take: the call's `timeout` argument or option wins, then this setting, then the default, which is 3600 for a `run_agent` call (the sub-task is stopped when it expires) and no limit of its own for a `run_parallel` child (a thread of the calling task, bounded by it); ignored by `/<name>`. |
| `locked` | `list` | — | Keys an explicit `run_agent` / `run_parallel` argument or option may not change: a differing value is an error. |
| `hidden` | `bool` | — | `True`: the script is no `/command` and no `run_agent` agent name (loadable by path and as an `extends` base only). It must be the literal `True` in `settings()` because the command registry reads it from the source without running the script (a computed value is ignored). |
<!-- /sea-docs -->

### Kinds

`kind` says what the run is; each kind is only a dict of defaults laid under the explicit keys.

<!-- sea-docs: kinds -->
| Kind | Defaults | Use |
|---|---|---|
| `session` | nothing | The default: an ordinary Sorcar session with the caller's or the user's settings (`/write`, `/write_paper`, `bestrouter`). |
| `worker` | `use_worktree=False`, `auto_commit=False`, `auto_classify=False`, `allow_fan_out=False`, `use_web_tools=False`, `use_memory=False` | A focused tool-bound run on the caller's tree: no worktree, no auto-commit, no classifier, no fan-out, no browser, no memory (`/sh`, `/ask`, `/merge`, `/remember`, `/forget`, `/task_update`). |
| `channel` | `use_worktree=False`, `auto_commit=False`, `auto_classify=False`, `allow_fan_out=False`, `use_web_tools=False`, `use_memory=False`, `work_dir='<home>/channel_work'` | A worker for an external service, in the shared `channel_work` scratch directory under the Sorcar home, never the caller's project; it holds its channel workspace, gets the channel preamble and never inherits from a calling task (every bundled channel agent and `/cron`). |
<!-- /sea-docs -->

### `options` keys of `run_agent` and `run_parallel`

<!-- sea-docs: options -->
| Option | Type | Meaning |
|---|---|---|
| `work_dir` | `str` | The directory the sub-task works in; a relative path is resolved against the calling task's directory (a SEA's own `work_dir` setting is relative to the SEA's folder instead). |
| `model` | `str` | The LLM model, a catalogue name or a model-picker SEA; `""` or `None` keeps the caller's. |
| `chat_id` | `str` | The chat the run's events go to; default: a new chat. |
| `use_worktree` | `bool` | Run in a git worktree of the project (daemon default `True`). |
| `auto_commit` | `bool` | Commit the run's changes when it ends (daemon default `True`). |
| `max_budget` | `int \| float` | USD budget of the run, a finite number; default: the caller's share or the daemon's default. |
| `model_config` | `dict` | Model configuration dict passed to the LLM (temperature, base URL, ...). |
| `use_web_tools` | `bool` | Give the run the browser tools (daemon default: on). |
| `auto_classify` | `bool` | Let the pre-run classifier decide the worktree mode and lite prompt (daemon default: the persisted setting). |
| `use_memory` | `bool` | Give the run the `memory_*` tools (daemon default: the persisted setting). |
| `allow_fan_out` | `bool` | Let the run call `run_parallel` (default `True`). |
| `tool_profile` | `str` | The run's toolset: `review`, `bash`, `shell+edit`, ... (default: the full toolset). |
| `docker_image` | `str` | Run inside this Docker image (default: the host). |
| `timeout` | `int \| float` | Seconds the run may take: the call's `timeout` argument or option wins, then this setting, then the default, which is 3600 for a `run_agent` call (the sub-task is stopped when it expires) and no limit of its own for a `run_parallel` child (a thread of the calling task, bounded by it); ignored by `/<name>`. |
| `inherit` | `bool` | `false`: the sub-task takes nothing from the calling task (no model, chat, prompt suffixes, tools or container; a `run_parallel` child still gets its budget share); default `true`. A `channel` run never inherits, so `true` is refused there. |
| `workspace` | `str` | The account a `kind: channel` agent's run holds (its channel workspace); refused for any other kind and by `run_parallel`. |
| `add_to_prompt` | `str` | Text appended to the task after the SEA's `prompt(task)`. |
| `add_to_system_prompt` | `str` | Text appended to the system prompt after the SEA's `add_to_system_prompt()`. |
<!-- /sea-docs -->

## Add your own command in three steps

1. Create a parent folder, a folder named after the command, and the SEA file inside it. Every SEA must define `description()`, one sentence that says what it does and how to use it (this is what `/standup help` prints). The smallest useful SEA adds a system prompt; everything else keeps the daemon's defaults:

   ```python
   # ~/my-seas/standup/standup_sea.py
   def description():
       return (
           "Turns your text into a three-bullet daily stand-up note "
           "(done, next, blocked); use it as /standup <what happened>."
       )


   def system_prompt():
       return (
           "You write terse daily stand-up notes from the user's text. "
           "Three bullets: done, next, blocked. No preamble."
       )
   ```

   Helper modules, prompt files and data the SEA needs go into the same `standup/` folder. To change how the run is configured, add a `settings()` function returning a dict: a `kind` plus any of the per-run settings listed in [Client Interfaces](cli.md#sorcar-extension-agents-seas) (`model`, `max_budget`, `tool_profile`, `timeout`, ...), for example `{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}` (what `/sh` declares) or `{"timeout": 3600}` (`/write`). A kind is only a dict of defaults under your keys: `session` (the default) is empty, `worker` turns worktree, auto-commit, classifier, fan-out, browser and memory off, and `channel` is `worker` plus a `work_dir` of `$KISS_HOME/channel_work`, for an agent of an external service (it holds its channel workspace, gets the channel preamble and never inherits from a calling task). `settings()` is data; text and code come from the getters: `system_prompt()`, `add_to_system_prompt()`, `prompt(task)` (its result may contain `{task_id}`, replaced by the calling task's id), `add_to_tools()` and the hooks. For a SEA that carries its own tools, return a list of callables from `add_to_tools()`; they are added to the built-in toolset, or become the agent's only tools besides `finish` when `settings()` also sets `"tool_profile": "none"` (what `/ask` does). The bundled channel agents are a good template (see [`src/kiss/agents/third_party_agents/`](https://github.com/ksenxx/kiss_ai/tree/main/src/kiss/agents/third_party_agents), one folder per agent).

2. Register the folder:

   ```bash
   echo '~/my-seas' >> $KISS_HOME/SEAS.md
   ```

3. Within two seconds, type `/st` in the chat box: the popup offers `/standup`. Send `/standup help` to see the description you wrote, then `/standup finished the docs page, next is the release, blocked on review` and the note comes back in the chat.

If the command does not appear, check that the script is `<folder>/<folder>_sea.py` (`standup/standup_sea.py`, not `standup_sea.py` at the top of `~/my-seas`), that the folder name uses only `A-Z a-z 0-9 _ -`, and that the folder line in `SEAS.md` resolves to the folder you expect (remember that relative paths are anchored at `~`, not at the project). A broken SEA (import error, `settings()` or another function raising, wrong return type, unknown, renamed or removed settings key, unknown kind) is still listed as a command; the failure surfaces as a diagnostic in the task result when you run it.

## Bundled commands

Every command the package ships, with the first sentence of its `description()` (what `/<name> help` prints). Commands in your `SEAS.md` folders are added to this list on your machine.

<!-- sea-docs: commands -->
| Command | Script | Description |
|---|---|---|
| `/a2a` | `agents/third_party_agents/a2a/a2a_sea.py` | Speaks the Agent-to-Agent (A2A) protocol: discovers a peer agent's card and calls it over JSON-RPC 2.0 (`message/send` / `tasks/get`), and serves this agent's own card and inbound `message/send` requests from an embedded HTTP server; use it as `run_agent(agent="a2a", task="Discover the agent at http://host:port and greet it")` or through the `kiss-a2a` CLI. |
| `/ask` | `agents/seas/ask/ask_sea.py` | Answers a question about the currently running task in two or three plain sentences, from its status, progress log and latest transcript entries, without editing files, running commands or using the internet; type `/ask <question>` into the task's chat tab. |
| `/autorouter` | `agents/seas/autorouter/autorouter_sea.py` | Splits a task into units, runs each on the cheapest model tier (small, medium, frontier) that passes its acceptance check, escalating on failure and logging every decision to ~/.kiss/MODEL_DECISIONS.md; pick `autorouter` in the model picker, use `/autorouter <task>` in the chat, or run_agent(agent="autorouter", task="..."). |
| `/bestrouter` | `agents/seas/bestrouter/bestrouter_sea.py` | Runs every task on claude-fable-5-1 and has gpt-6-astra review and debug the result read-only through run_parallel on at most 75% of the task budget; pick `bestrouter` in the model picker, use `/bestrouter <task>` in the chat, or run_agent(agent="bestrouter", task="..."). |
| `/bluebubbles` | `agents/third_party_agents/bluebubbles/bluebubbles_sea.py` | Reads and sends iMessages through a BlueBubbles server running on a local Mac (macOS only; server URL and password in ~/.kiss/third_party_agents/bluebubbles/config.json); use it with `run_agent(agent="bluebubbles", task=...)` or the `kiss-bluebubbles -t "<task>"` CLI (`kiss-bluebubbles --channel <chat guid>` polls a chat and answers new messages). |
| `/brave` | `agents/third_party_agents/brave/brave_sea.py` | Runs web, news, image and video searches through the Brave Search REST API with a stored subscription token; use it with `run_agent(agent="brave", task="Find recent news about quantum computing")` or the `kiss-brave -t '<task>'` CLI (outbound-only, no --channel poll mode). |
| `/cron` | `agents/sorcar/cron_agent.py` | Scheduled automations: create, list, pause, resume, remove or run now the cron jobs of this KISS home (a polled messaging gateway is a cron job too). |
| `/dingtalk` | `agents/third_party_agents/dingtalk/dingtalk_sea.py` | Sends messages to a DingTalk group through a custom-robot webhook and receives messages from an outgoing robot through an embedded HTTP callback server; use it as run_agent(agent="dingtalk", task="...") or the `kiss-dingtalk` CLI (`-t <task>` for one task, `--channel <id>` for a poll tick). |
| `/discord` | `agents/third_party_agents/discord/discord_sea.py` | Channel agent for Discord that signs in with click-Allow OAuth (or a bot token for reading, polling and managing messages), posts to the authorized channel through its webhook and lists servers and channels via the REST API v10; use `run_agent(agent="discord", task=...)` or the `kiss-discord` CLI. |
| `/email` | `agents/third_party_agents/email/email_sea.py` | Reads, lists, marks and sends mail in any IMAP/SMTP mailbox (unread mail over IMAP4_SSL, replies over SMTP, automated senders skipped); use it with `run_agent(agent="email", task="...")` or the `kiss-email -t '...'` CLI, starting with `kiss-email -t 'authenticate'` to store the mailbox credentials. |
| `/feishu` | `agents/third_party_agents/feishu/feishu_sea.py` | Channel agent for Feishu/Lark that authenticates with an app_id/app_secret and can list chats, get chat and user info, and list, send, reply to and delete messages; use run_agent(agent="feishu", task="...") or the `kiss-feishu -t '<task>'` CLI (start with the task `authenticate`). |
| `/firecrawl` | `agents/third_party_agents/firecrawl/firecrawl_sea.py` | Scrapes pages, maps sites, searches the web and manages crawls through the Firecrawl v2 REST API (cloud or self-hosted, authenticated with an API key); use it as `run_agent(agent="firecrawl", task="Scrape https://example.com and summarize it")` or through the `kiss-firecrawl` CLI. |
| `/forget` | `agents/seas/forget/forget_sea.py` | Removes a standing instruction that /remember stored in ~/.kiss/AGENTS.md so later tasks stop following it; use it as `/forget <instruction text>` in the chat or `run_agent(agent="forget", task="<instruction text>")`. |
| `/gcal` | `agents/third_party_agents/gcal/gcal_sea.py` | Lists the signed-in user's Google Calendars and lists, searches, creates, updates and deletes their events through the Calendar REST API (signed in via Composio); use run_agent(agent="gcal", task="...") or the `kiss-gcal` CLI. |
| `/gdocs` | `agents/third_party_agents/gdocs/gdocs_sea.py` | Creates, reads, edits and lists Google Docs through the Docs and Drive REST APIs with sign-in handled by Composio (outbound only, no message polling); use it with `run_agent(agent="gdocs", task=...)` or the `kiss-gdocs -t "<task>"` CLI. |
| `/gdrive` | `agents/third_party_agents/gdrive/gdrive_sea.py` | Lists, searches, reads, uploads, moves and shares files in the user's Google Drive through the Drive v3 REST API with sign-in and calls proxied by Composio; use it with `run_agent(agent="gdrive", task="Find my spreadsheets modified this week")` or the `kiss-gdrive -t '<task>'` CLI (outbound-only, no --channel poll mode). |
| `/git_extract_knowledge` | `agents/seas/git_extract_knowledge/git_extract_knowledge_sea.py` | Indexes every tracked file and commit of a git repository into its domain memory (curated Markdown pages plus a full-text block store) and schedules a daily incremental refresh; use `/git_extract_knowledge <repo path or clone URL>`, `/git_extract_knowledge update <repo>`, `/git_extract_knowledge ask <question>` or run_agent(agent="git_extract_knowledge", task=...). |
| `/github` | `agents/third_party_agents/github/github_sea.py` | Works with GitHub repositories, issues, pull requests, commits and files through the GitHub REST API after a device-flow sign-in (outbound only, no message polling); use it as run_agent(agent="github", task="...") or the `kiss-github -t <task>` CLI. |
| `/gmail` | `agents/third_party_agents/gmail/gmail_sea.py` | Channel agent for Gmail that reads, searches, sends, labels and trashes email through a Composio-connected Google account (the user connects with a Composio Connect Link and Composio's proxy holds the token); use `run_agent(agent="gmail", task=...)` or the `kiss-gmail` CLI. |
| `/googlechat` | `agents/third_party_agents/googlechat/googlechat_sea.py` | Lists, gets and creates Google Chat spaces, lists their members, and reads, posts, edits and deletes messages as the signed-in user (via Composio) or as a Chat bot with a service account; use `run_agent(agent="googlechat", task="...")` or the `kiss-gchat -t '...'` CLI. |
| `/gsheets` | `agents/third_party_agents/gsheets/gsheets_sea.py` | Google Sheets agent (signed in through Composio) that creates and lists spreadsheets, adds sheets, and reads, updates, appends, clears and batch-updates cell ranges; use run_agent(agent="gsheets", task="...") or the `kiss-gsheets -t '<task>'` CLI (outbound-only, no message polling). |
| `/homeassistant` | `agents/third_party_agents/homeassistant/homeassistant_sea.py` | Controls a Home Assistant instance through its REST API with a long-lived access token (reading entity states, calling services and posting persistent notifications); use it as `run_agent(agent="homeassistant", task="Turn off all the lights in the kitchen")` or through the `kiss-ha` CLI. |
| `/imessage` | `agents/third_party_agents/imessage/imessage_sea.py` | Sends iMessages and attachments and reads conversations through the macOS Messages app via AppleScript (macOS only); use run_agent(agent="imessage", task="...") or the `kiss-imessage` CLI. |
| `/irc` | `agents/third_party_agents/irc/irc_sea.py` | Joins IRC channels and sends and reads messages on the IRC server configured in ~/.kiss/third_party_agents/irc/config.json (server, nick, optional TLS); use it with `run_agent(agent="irc", task=...)` or the `kiss-irc -t "<task>"` CLI (`kiss-irc --channel <#channel>` polls a channel and answers new messages). |
| `/line` | `agents/third_party_agents/line/line_sea.py` | Sends text and image push/reply messages, reads profiles and quota and leaves groups on LINE through the Messaging API with a stored channel access token, receiving inbound messages via a local webhook queue; use it with `run_agent(agent="line", task="Send 'Hello!' to user U123456789")` or the `kiss-line -t '<task>'` CLI (`kiss-line --channel <id>` runs one inbound poll tick). |
| `/matrix` | `agents/third_party_agents/matrix/matrix_sea.py` | Sends and reads messages in Matrix rooms through matrix-nio, signing in with the homeserver's OAuth 2.0 device grant or a hand-supplied access token; use it as run_agent(agent="matrix", task="...") or the `kiss-matrix` CLI (`-t <task>` for one task, `--channel <room>` for a poll tick). |
| `/mattermost` | `agents/third_party_agents/mattermost/mattermost_sea.py` | Channel agent for Mattermost that lists teams, channels and users and reads, posts and manages messages through the REST API with a personal access token stored under ~/.kiss/third_party_agents/mattermost; use `run_agent(agent="mattermost", task=...)` or the `kiss-mattermost` CLI. |
| `/merge` | `agents/seas/merge/merge_sea.py` | Resolves the git merge conflicts left in the current repository by reading both sides of every conflict block, writing a resolution that keeps the intent of both, and staging the resolved files without committing; use it as `/merge <task>` in the chat (e.g. |
| `/msteams` | `agents/third_party_agents/msteams/msteams_sea.py` | Lists teams, channels, chats and members and reads or posts channel and chat messages in Microsoft Teams through Microsoft Graph as the user signed in with the device code flow; use it with `run_agent(agent="msteams", task="...")` or the `kiss-msteams -t '...'` CLI. |
| `/nextcloud` | `agents/third_party_agents/nextcloud/nextcloud_sea.py` | Channel agent for Nextcloud Talk that signs in with Login Flow v2 (or a username and app password) and can list, create and rename rooms, list participants, and list, post and delete messages; use run_agent(agent="nextcloud", task="...") or the `kiss-nextcloud -t '<task>'` CLI (start with the task `authenticate`). |
| `/nostr` | `agents/third_party_agents/nostr/nostr_sea.py` | Publishes notes and replies, sends encrypted DMs, reads and sets profiles and manages relays on the Nostr decentralized protocol via pynostr; use it as `run_agent(agent="nostr", task="Post a note saying hello")` or through the `kiss-nostr` CLI. |
| `/notion` | `agents/third_party_agents/notion/notion_sea.py` | Searches Notion, reads and queries databases, reads, creates and updates pages and blocks, and reads and adds comments through the Notion REST API with an internal-integration token; use run_agent(agent="notion", task="...") or the `kiss-notion` CLI. |
| `/ntfy` | `agents/third_party_agents/ntfy/ntfy_sea.py` | Publishes notifications to and reads messages from the ntfy topic configured in ~/.kiss/third_party_agents/ntfy/config.json (optional self-hosted server and access token); use it with `run_agent(agent="ntfy", task=...)` or the `kiss-ntfy -t "<task>"` CLI (`kiss-ntfy --channel <topic>` polls the topic and answers new messages). |
| `/overleaf` | `agents/third_party_agents/overleaf/overleaf_sea.py` | Lists, reads and edits Overleaf projects and their files through the editor's web routes using the user's pasted `overleaf_session2` browser cookie (outbound only, no message polling); use it as run_agent(agent="overleaf", task="...") or the `kiss-overleaf -t <task>` CLI. |
| `/phone` | `agents/third_party_agents/phone/phone_sea.py` | Channel agent that controls an Android phone through its companion REST app (configured by device IP under ~/.kiss/third_party_agents/phone) to send and read SMS, make and end calls, read the call log and list, dismiss or reply to notifications; use `run_agent(agent="phone", task=...)` or the `kiss-phone` CLI. |
| `/postgres` | `agents/third_party_agents/postgres/postgres_sea.py` | Queries a PostgreSQL database given by a postgresql:// URI (read-only by default, with schema, table, index and EXPLAIN inspection and optional write statements); use it with `run_agent(agent="postgres", task="...")` or the `kiss-postgres -t '...'` CLI. |
| `/qq` | `agents/third_party_agents/qq/qq_sea.py` | Channel agent for the official QQ bot platform that sends group and C2C (private) messages with a bot appid/secret and receives inbound events on an embedded Ed25519-verified webhook server; use run_agent(agent="qq", task="...") or the `kiss-qq -t '<task>'` CLI (start with the task `authenticate`). |
| `/remember` | `agents/seas/remember/remember_sea.py` | Stores the prompt as a standing instruction in ~/.kiss/AGENTS.md so every future Sorcar task follows it; use `/remember <instruction>` in the chat or run_agent(agent="remember", task="<instruction>"), and `/forget` to remove it. |
| `/review_paper` | `agents/seas/review_paper/review_paper_sea.py` | Reviews a research paper (PDF, .tex, .md or .txt) for a venue like a careful human reviewer, searching the related work and writing a Summary/Strengths/Weaknesses/Detailed review/Scores (seven dimensions, 1 to 10) review that passes the word-limit and AI-slop gates; use it as `/review_paper Review <paper path> for <venue>; <word limit> words; to <output path>` or `run_agent(agent="review_paper", task=...)`. |
| `/revise_and_review_paper` | `agents/seas/revise_and_review_paper/revise_and_review_paper_sea.py` | Writes a research paper with /write_paper from your writing instructions, has /review_paper review it as a fresh reviewer under your review instructions, and repeats the revise-and-review loop (with experiments, ablations or AI discovery when the review asks for evidence) until the review says strong accept or the paper cannot improve further; use `/revise_and_review_paper Writing: <...> Review: <...>`. |
| `/rsi7d` | `agents/seas/rsi7d/rsi7d_sea.py` | Mines the last 7 days of the indexed SEAs' runs in ~/.kiss/history.db for agentic mistakes, cost sinks and quality problems, applies and evaluates improvements to each SEA (its own included): instructions in its prompt, settings() limits tuned from the runs, tools and guardrail hooks in its code, through gated editors that keep the script loading and lint-clean (file-modifying tasks are replayed in a clone at the task's commit); mines eval sets for skillopt; refreshes the autorouter SEA's model evidence and, with the user's permission (asked for, unless the task text grants it), improves KISS Sorcar itself: src/kiss/SYSTEM.md, ~/.kiss/AGENTS.md and its code. |
| `/sh` | `agents/seas/sh/sh_sea.py` | Runs the shell command given in the prompt directly on the tab's working directory with only the Bash tool and returns its verbatim output; use it as `/sh <command>` in the chat (e.g. |
| `/signal` | `agents/third_party_agents/signal/signal_sea.py` | Sends and receives Signal messages and attachments and lists contacts and groups through the signal-cli subprocess, linking to your phone's Signal account with a QR code like Signal Desktop; use it as `run_agent(agent="signal", task="Send 'Hello!' to +14155238886")` or through the `kiss-signal` CLI. |
| `/simplex` | `agents/third_party_agents/simplex/simplex_sea.py` | Sends and receives SimpleX Chat messages and lists contacts through a locally running `simplex-chat` CLI's WebSocket API (default ws://127.0.0.1:5225); use run_agent(agent="simplex", task="...") or the `kiss-simplex` CLI. |
| `/skillopt` | `agents/seas/skillopt/skillopt_sea.py` | Optimizes the prompt text of a skill (SKILL.md), of a SEA (its system_prompt() constant) or of a named module constant against a JSON eval set, writing accepted text next to the target as `<target>.proposed`; use `/skillopt optimize <target> with <evals.json> for one epoch`, `run_agent(agent="skillopt", task=...)`, or `uv run python -m kiss.agents.seas.skillopt.skillopt_sea --target ... |
| `/slack` | `agents/third_party_agents/slack/slack_sea.py` | Messages, channels, users, reactions and search in a Slack workspace the user authorizes once in the browser (OAuth user token, refreshed automatically); use it with `run_agent(agent="slack", task=...)` or the `kiss-slack -t "<task>"` CLI (`kiss-slack --channel <channel>` polls a channel and answers new messages, `--workspace <name>` selects among several signed-in workspaces). |
| `/sms` | `agents/third_party_agents/sms/sms_sea.py` | Sends and lists SMS, MMS and WhatsApp messages and places or lists voice calls through a Twilio account (account SID, auth token and from-number stored in ~/.kiss/third_party_agents/sms/config.json); use it with `run_agent(agent="sms", task="Send 'Hello!' to +14155238886")` or the `kiss-sms -t '<task>'` CLI (`kiss-sms --channel <number>` runs one inbound poll tick). |
| `/synology` | `agents/third_party_agents/synology/synology_sea.py` | Sends messages to Synology Chat through an incoming webhook and receives messages from an outgoing webhook through an embedded HTTP server; use it as run_agent(agent="synology", task="...") or the `kiss-synology` CLI (`-t <task>` for one task, `--channel <id>` for a poll tick). |
| `/task_update` | `agents/seas/task_update/task_update_sea.py` | Reports what a running or finished Sorcar task has done so far and its partial results by reading its persisted transcript from ~/.kiss/history.db; use it as `/task_update <task_id>` in the chat or `run_agent(agent="task_update", task="<task_id>")`. |
| `/telegram` | `agents/third_party_agents/telegram/telegram_sea.py` | Channel agent for Telegram that uses a @BotFather bot token (stored under ~/.kiss/third_party_agents/telegram) to send, edit, forward, pin and delete messages, photos, documents and polls, read updates and inspect or moderate chat members through the Bot API; use `run_agent(agent="telegram", task=...)` or the `kiss-telegram` CLI. |
| `/tlon` | `agents/third_party_agents/tlon/tlon_sea.py` | Lists groups and channels, reads and posts messages, reads profiles and runs pokes and scries on an Urbit ship (Tlon) through its Eyre HTTP server; use it with `run_agent(agent="tlon", task="...")` or the `kiss-tlon -t '...'` CLI. |
| `/twitch` | `agents/third_party_agents/twitch/twitch_sea.py` | Channel agent for Twitch (Helix API and chat, signed in with the OAuth2 device code grant) that gets stream, channel and user info, lists chatters, sends chat messages, bans users, and lists or creates clips; use run_agent(agent="twitch", task="...") or the `kiss-twitch -t '<task>'` CLI (start with the task `authenticate`). |
| `/webhook` | `agents/third_party_agents/webhook/webhook_sea.py` | Runs an embedded HTTP server that accepts HMAC-SHA256-signed `POST /hook/<route>` webhooks (GitHub or generic scheme) and turns each event into a prompt rendered from the route's template, with tools to add, remove and list routes; use it as `run_agent(agent="webhook", task="Add a webhook route for GitHub pushes")` or through the `kiss-webhook` CLI. |
| `/wecom` | `agents/third_party_agents/wecom/wecom_sea.py` | Posts text and markdown messages to a WeCom (WeChat Work) group through a group-robot incoming webhook (outbound only, no inbound messages); use run_agent(agent="wecom", task="...") or the `kiss-wecom` CLI. |
| `/weixin` | `agents/third_party_agents/weixin/weixin_sea.py` | Sends customer-service messages from a WeChat Official Account (appid and appsecret in ~/.kiss/third_party_agents/weixin/config.json) and receives inbound messages on an embedded callback server; use it with `run_agent(agent="weixin", task=...)` or the `kiss-weixin -t "<task>"` CLI (`kiss-weixin --channel <openid>` polls for new messages and answers them). |
| `/whatsapp` | `agents/third_party_agents/whatsapp/whatsapp_sea.py` | Sends messages, files and audio and searches contacts, chats and message history on a personal WhatsApp account through the locally built lharries/whatsapp-mcp Go bridge, pairing once by scanning a QR code from the phone; use it with `run_agent(agent="whatsapp", task="Send 'Hello!' to +1234567890")` or the `kiss-whatsapp -t '<task>'` CLI (`kiss-whatsapp --channel <chat>` runs one poll tick). |
| `/write` | `agents/seas/write/write_sea.py` | Writes concise, professional American English for a general audience that reads as if a person wrote it, with the vocabulary and sentence patterns of machine text banned; use `/write <what to write, its sources and, optionally, the output path>` in the chat or run_agent(agent="write", task="..."). |
| `/write_paper` | `agents/seas/write_paper/write_paper_sea.py` | Writes or revises a LaTeX research paper from your sources and results, running AI-slop/consistency gates (`check_paper`) and a pdflatex+bibtex build (`build_paper`) with a read-only reviewer model; use `/write_paper <venue, .tex path, topic, sources, options>` or run_agent(agent="write_paper", task=...). |
| `/zalo` | `agents/third_party_agents/zalo/zalo_sea.py` | Sends and receives Zalo Official Account messages through the Zalo OA API with an access token and an embedded webhook server; use it as run_agent(agent="zalo", task="...") or the `kiss-zalo` CLI (`-t <task>` for one task, `--channel <id>` for a poll tick). |
<!-- /sea-docs -->
