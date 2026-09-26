---
title: 'Prompt rewrites: /xxx slash commands (SEAs) and bare-path tasks'
uuid: 08a25d76-eb07-45cd-a97e-970bcc86ebdf
summary: 'sea_commands: every *_sea.py becomes /<stem> (third_party_agents > SEAS.md
  > seas/, watcher, run_agent rewrite, /ask); bare_path_task: a path-only task is
  opened with the OS opener.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Slash commands and bare-path tasks

Two modules rewrite what the LLM sees before a task runs. In both cases the task history keeps the raw user text.

## Slash commands (`sea_commands.py`)
Every visible `*_sea.py` Sorcar Extension Agent script becomes a command `/<stem>`, where the stem is the filename
minus `_sea.py` (`sh_sea.py` -> `/sh`, `write_paper_sea.py` -> `/write_paper`).

### Discovery and precedence (highest first)
1. `src/kiss/agents/third_party_agents/` (found through the package `kiss.agents.third_party_agents`): channel
   agents such as `/slack` and `/gmail`, plus `/ask`.
2. Folders listed one per line in `~/.kiss/SEAS.md` (`$KISS_HOME/SEAS.md`, `seas_md_path`). Later lines beat
   earlier lines. Blank lines and `#` lines are ignored, as is an inline comment tail after whitespace + `#`
   (a bare `#` inside a name is kept). `~` and env vars are expanded; relative paths resolve against the home
   directory. Missing or unreadable folders (or an unreadable `SEAS.md`) are skipped silently.
3. `src/kiss/agents/seas/`, the bundled SEAs (`/merge`, `/sh`, `/task_update`, `/autoroute`, ...). Lowest, so a
   `SEAS.md` folder can replace e.g. `/merge`.

`_scan_folder` accepts only regular files ending in `_sea.py` whose command name matches `^[A-Za-z0-9_-]+$`;
`foo.bar_sea.py` or `space name_sea.py` are skipped. Underscore-prefixed stems are valid.

Lookups (`get_command`, `list_commands`) return cached hits without rescanning; the registry is refreshed on a miss,
when it is empty, and, in the daemon, by
`start_registry_watcher` polling every 2 s (bounded to 0.1-60 s), so edits to `SEAS.md` or new files apply without
a restart. Subscribers (`subscribe`) receive the command list for autocomplete and are notified outside the lock.
Lock order is always `_notify_lock` then `_lock`.

### Prompt rewrite
`rewrite_prompt_if_command(prompt)` is called by `server/task_runner.py`. It fires only when the prompt starts
with `/` at character 0 (no leading whitespace), the name is registered, and the trailing text is non-empty
(`_split_slash_command` strips it). It then returns `(rewritten, sea_path)`, where the rewritten prompt orders the
agent to call **`run_agent` immediately** with `agent="<abs path>"` and the text verbatim, without exploring code,
and then relay the result. It returns `None` for unknown commands or a bare `/sh`, since an empty `run_agent` task
would be rejected. `ChatSorcarAgent.run(_history_prompt=...)` keeps the raw `/xxx ...` for history.

`/ask` is special: the directive also passes `append_to_prompt` read from `ask_sea.py`'s `APPEND_TO_PROMPT`, which
contains a literal `<task_id>` placeholder; `agent_dispatch._dispatch` substitutes the calling task's id right
before sending. `/ask` on a running tab goes through a side channel in `kiss.server.commands` instead.

### Outer relay run
The tab where `/xxx` was typed runs a relay task that calls `run_agent`. `task_runner` uses
`sea_getter_is_false(sea_path, "use_worktree")` and `"auto_commit"` so an SEA that declares it works on the real
checkout (`/sh`, `/merge`) is not handed the relay's worktree and the relay does not auto-commit what it left.
An SEA that fails to import raises `SeaScriptError`, failing the relay with the diagnostic.

## Bare-path tasks (`bare_path_task.py`)
When the whole prompt (optionally quoted) is an existing path, relative to work_dir with `~` expanded,
`with_open_directive` appends: open it with the platform opener (`opener_command`: `open` on macOS, `rundll32.exe
url.dll,FileProtocolHandler` on Windows, otherwise `xdg-open`, with the path shell-quoted) through Bash, then
finish with one sentence. Do not read, summarize, edit, or ask. Without this, SYSTEM.md's "ask rather than guess"
made the agent ask what to do with the file. `ChatSorcarAgent.run` applies it after capturing `history_prompt`.

## Sources
- `src/kiss/agents/sorcar/sea_commands.py` (`seas_md_path`, `_scan_folder`, `_read_seas_md_folders`, `refresh_registry`, `get_command`, `list_commands`, `subscribe`, `_split_slash_command`, `rewrite_prompt_if_command`, `sea_getter_is_false`, `SeaScriptError`, `start_registry_watcher`)
- `src/kiss/agents/sorcar/bare_path_task.py` (`bare_path`, `opener_command`, `with_open_directive`)
- `src/kiss/server/task_runner.py` (caller of `rewrite_prompt_if_command`, relay use of `sea_getter_is_false`)
- `src/kiss/agents/sorcar/agent_dispatch.py` (`_dispatch` `<task_id>` substitution)
- `src/kiss/agents/third_party_agents/ask_sea.py` (`APPEND_TO_PROMPT`)
- `src/kiss/server/commands.py` (`/ask` side channel, `_ASK_COMMAND_PREFIX`)
