---
title: How to write and run a SEA (agent script)
uuid: 122efe76-59ca-4814-a99b-9fa437620e9d
summary: How to write a *_sea.py agent script (getters, tool functions, self-contained
  file) and run it via /name, run_agent with a .py path, or kiss.server.sorcar.run(extension_agent_path).
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# How to write and run a SEA

## Minimal SEA

A SEA is any `.py` file; name it `<name>_sea.py` if it should also be a slash command.

```python
"""Changelog agent: summarizes git log since a tag."""
SYSTEM_PROMPT = "You summarize git history. Run git log with Bash, then call finish."

def system_prompt() -> str:          # replaces SYSTEM.md
    return SYSTEM_PROMPT

def tool_profile() -> str:           # full | review | shell | assistant | bash
    return "shell"

def use_worktree() -> bool:          # read-only work: run on the real checkout
    return False

def auto_commit() -> bool:
    return False

def use_web_tools() -> bool:
    return False
```

The bundled `sh_sea.py` is the model example: a module constant for the prompt plus
`tool_profile() -> "bash"` and every side feature (worktree, commit, classification,
fan-out, browser, memory) turned off. `dummy_sea.py` is empty: no getters means a plain Sorcar
session, and it is the default agent of `run_agent`.

## Adding tools

Define plain functions with typed parameters and docstrings (they become the tool schema) and
return them from `tools()`:

```python
def count_commits(since: str) -> str:
    """Count commits since a git ref. Args: since: the ref."""
    ...

def tools() -> list:
    return [count_commits]
```

A list return makes the script its own tools file. `tools()` may instead return the absolute
path of a separate tools file exposing `get_tools()`. Tools run in the daemon process.

Guidelines drawn from the bundled SEAs:
- Keep the prompt in a module-level string constant returned by `system_prompt()`, so
  SkillOpt can optimize it (see `seas-skillopt`). SkillOpt's default extraction reads only
  `system_prompt()`; for an `append_to_system_prompt()`-only SEA you must name the constant
  explicitly (`load_target(path, constant=...)`).
- Keep the file self-contained: SkillOpt copies candidates to scratch dirs, so imports of
  sibling modules or files next to `__file__` break it.
- Getters run on every task start; keep them fast and side-effect free.
- Use `append_to_system_prompt()` when the agent should keep the full Sorcar rules and tools
  (`write_paper_sea.py`, `review_paper_sea.py`); use `system_prompt()` for a narrow agent.
- Set `max_budget()` for bounded helpers (`task_update_sea.py`, `merge_sea.py` do).

## Running it

1. **Slash command**: put the file in `src/kiss/agents/seas/` or in a folder listed in
   `~/.kiss/SEAS.md`, then type `/<name> <task text>` in an idle tab (see
   `sorcar-slash-commands-and-bare-paths`). Text after the command is required.
2. **From an agent**: `run_agent(agent="path/to/name_sea.py", task="...")`. `run_agent`
   recognizes a script by a `.py` suffix or a path separator; relative paths resolve against the
   calling task's work dir and must exist. A bare name such as `"skillopt"` is looked up as a
   channel and fails with `unknown agent` (the `skillopt_sea.py` docstring shows
   `run_agent(agent="skillopt")`, which does not work; pass the path). Path-named scripts default
   to `use_worktree=true`, `auto_commit=true` unless the script's getters say otherwise.
3. **From Python**: `kiss.server.sorcar.run(prompt, extension_agent_path="name_sea.py", ...)`
   against the running daemon; the client validates the path with `resolve_agent_path`, the
   daemon applies the getters, and a broken script returns `TaskResult(success=False)`.

In-process (no daemon), SkillOpt's `SeaTarget.rollout_kwargs` shows how getters map to
`SorcarAgent.run` keyword arguments (`system_prompt` -> `base_system_prompt`,
`use_web_tools` -> `web_tools`, `append_to_system_prompt` -> `system_prompt`, ...).

## Sources
- `src/kiss/agents/seas/sh_sea.py`, `src/kiss/agents/seas/dummy_sea.py`
- `src/kiss/agents/sorcar/agent_dispatch.py` (`run_agent`, `DEFAULT_AGENT_PATH`, `_unknown_agent_error`)
- `src/kiss/agents/sorcar/daemon_client.py` (`resolve_agent_path`)
- `src/kiss/server/agent_file.py` (`apply_agent_overrides`)
- `src/kiss/agents/seas/skillopt_sea.py` (`SeaTarget.rollout_kwargs`)
