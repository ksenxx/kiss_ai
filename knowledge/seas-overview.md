---
title: SEAs (Sorcar Extension Agents / agent scripts) area overview
uuid: 4116ad00-204b-4f11-b085-955018aa76aa
summary: What a SEA (Sorcar Extension Agent, *_sea.py agent script) is, where bundled
  SEAs live (agents/seas, ask_sea.py), and links to contract, slash command, SkillOpt
  and eval pages.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# SEAs (Sorcar Extension Agents) area overview

SEA stands for **Sorcar Extension Agent** (renamed from "extension agent" in commit
`5082fe611`). It is a plain Python file, usually named `<name>_sea.py`, that defines optional
zero-argument top-level getters such as `system_prompt()`, `tools()`, `tool_profile()` or
`use_worktree()`. When a task runs with that file as its agent script, the daemon imports it and
each defined getter overrides the matching `run` parameter. Nothing is subclassed; a SEA is
configuration plus optional tool functions.

## File map

| Path | Role |
|---|---|
| `src/kiss/server/agent_file.py` | The contract: `PARAM_FIELDS`, `HOOK_FIELDS`, `apply_agent_overrides`, `_check_override`, `AgentFileError` |
| `src/kiss/server/task_runner.py` | Calls `apply_agent_overrides(cmd)` when a `run` command carries `agentPath` |
| `src/kiss/agents/sorcar/sea_commands.py` | Slash-command registry: every `*_sea.py` becomes `/<name>` |
| `src/kiss/agents/sorcar/agent_dispatch.py` | `run_agent` tool; default agent `DEFAULT_AGENT_PATH` = `seas/dummy_sea.py` |
| `src/kiss/agents/seas/` | Bundled SEAs that drive Sorcar's own workflows |
| `src/kiss/agents/seas/evals/` | Eval sets for SkillOpt (`sh_sea_evals.json`, `sh_sea_heldout_evals.json`) |
| `src/kiss/agents/third_party_agents/ask_sea.py` | `/ask` side-channel Q&A SEA |
| `API.md` | `extension_agent_path` documentation of `kiss.server.sorcar.run` |

Third-party channel agents (`slack_sea.py`, ...) share the `_sea.py` suffix and the getter
contract but are a separate area.

## Detail pages

- `seas-agent-script-contract`: every getter, types, validation, hooks, where it runs.
- `sorcar-slash-commands-and-bare-paths`: `/name` commands, `~/.kiss/SEAS.md`, precedence, prompt rewrite.
- `seas-writing-and-running`: how to write a SEA and the three ways to run one.
- `seas-bundled-agents`: what each bundled SEA does and which getters it sets.
- `seas-skillopt`: the SkillOpt prompt optimizer SEA.
- `seas-eval-set-format`: eval JSON format, grading rules, sh_sea eval sets.

## Sources
- `src/kiss/agents/seas/__init__.py` (package docstring)
- `src/kiss/server/agent_file.py` (`PARAM_FIELDS`, `apply_agent_overrides`)
- `src/kiss/agents/sorcar/sea_commands.py`
- `src/kiss/agents/sorcar/agent_dispatch.py` (`DEFAULT_AGENT_PATH`)
