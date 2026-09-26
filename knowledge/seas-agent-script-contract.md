---
title: SEA agent scripts and tools files loaded by the daemon
uuid: e0f9995c-a238-4cdc-bf51-c1b3c63d5d72
summary: SEA agent-script getters (PARAM_FIELDS), types, wire fields, llm/tool call
  hooks, atomic validation in apply_agent_overrides, and tools-file loading via get_tools()/tools().
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# SEA agent scripts and tools files

Clients never serialize Python callables. They send file PATHS (`agentPath`, `toolsFile`) and the
daemon imports the files itself, so getters, hooks and tools run in the daemon process like
native code. The client resolves and validates the paths (`daemon_client.resolve_agent_path`,
`resolve_tools_file`); the daemon still treats the wire fields as untrusted.

## Shared loader

`tools_file.execute_python_file(raw_path, error_cls, label)` compiles and executes the file from
source on every run (no `__pycache__` read or write), so each run sees the file's current
contents and the caller's directory stays free of bytecode. A non-string value, a missing or
non-`.py` path, or an import-time exception raises `error_cls`.

## Getters (PARAM_FIELDS)

Each optional zero-argument top-level function overrides one `kiss.server.sorcar.run` parameter.
Getter -> wire field, accepted return type:

| Getter | Wire field | Type |
|---|---|---|
| `prompt()` | `prompt` | non-empty str |
| `work_dir()`, `model()`, `chat_id()`, `system_prompt()`, `append_to_system_prompt()`, `append_to_prompt()`, `scope_work_dir()`, `tool_profile()`, `docker_image()` | `workDir`, `model`, `chatId`, `systemPrompt`, `appendToSystemPrompt`, `appendToPrompt`, `tabScopeWorkDir`, `toolProfile`, `dockerImage` | str |
| `tools()` | `toolsFile` | tools-file path (str or PathLike), a list of tool callables, or None |
| `use_worktree()`, `auto_commit()`, `if_append_basic_tools()`, `is_parallel()` | `useWorktree`, `autoCommit`, `appendBasicTools`, `useParallel` | bool |
| `use_web_tools()`, `classify_tasks()`, `use_memory()` | `webTools`, `classifyTasks`, `useMemory` | bool or None (None = daemon default) |
| `max_budget()` | `maxBudget` | finite number (normalized to float) or None |
| `model_config()` | `modelConfig` | dict or None |

Notes:
- The only renamed getter is `if_append_basic_tools()` (parameter `append_basic_tools`). The old
  `get_` prefix was dropped in commit `ead4bf9b2`.
- `system_prompt()` replaces the base system prompt (`SYSTEM.md`) entirely; use
  `append_to_system_prompt()` to keep the default prompt and add rules.
- `tools()` returning a path writes that path to `toolsFile` (a PathLike is normalized to str).
  Returning a list makes the script its own tools file: the script's path is written to
  `toolsFile` and the task runner later calls its `tools()` (or `get_tools()`).
- `tool_profile()` must name a key of `sorcar_agent.TOOL_PROFILES` (`full`, `review`, `shell`,
  `assistant`, `bash`) or `""`; an unknown name is rejected when the task starts.
- `docker_image()` may be an image name or `container:<name-or-id>` to attach to a running
  container.
- No getters exist for `timeout`, `stop_on_timeout`, `sock_path` (client-side transport) or
  `parent_task_id`/`parent_tab_id` (the caller's identity, which a script must not forge to
  re-parent itself).

## Hooks (HOOK_FIELDS)

`llm_call_hook()` and `tool_call_hook()` return a callable or None. They are staged on the
daemon-side fields `llmCallHook` / `toolCallHook` (never on the wire, since a callable cannot be
serialized) and passed through `SorcarAgent.run` to `KISSAgent.run` of every executor
sub-session. `llm_call_hook(new_messages)` may rewrite messages before each LLM call;
`tool_call_hook(name, args)` must return `"OK"` for the tool to run, any other string becomes the
tool result. They do not apply to `run_parallel` sub-agents.

## How overrides are applied

`TaskRunner._run_task` calls `apply_agent_overrides(cmd)` for a `run` command with `agentPath`,
inside its outer try/finally because it executes untrusted code of unbounded duration. It runs
on the task's worker thread and returns the set of overridden fields.
1. `execute_python_file` imports the script.
2. For each getter present in the namespace: a non-callable (even `X = None`) raises
   `AgentFileError`; the getter is called; `_check_override` type-checks and normalizes the
   result, inside `BaseException` guards (untrusted values, e.g. a `str` subclass with a
   raising `strip`, must become a diagnostic, not kill the thread).
3. Overrides are staged and written into `cmd` only after every getter succeeds, so a broken
   script leaves the command untouched.

A broken script (malformed field, missing file, import error, raising or non-callable getter,
wrong type) raises `AgentFileError`; the task fails and the client gets
`TaskResult(success=False)` with the diagnostic text. Getters the script does not define keep
the value the client sent.

In `_run_task_inner`, the reviewer marker in `agent._subagent_info` is re-checked on the
EFFECTIVE prompt after overrides, because a `prompt()` override can turn a dispatch into a review
task after the caller-side check in `agent_dispatch._dispatch` passed.

## Tools file (`toolsFile`; `tools=` in `sorcar.run`)

`load_tools_file(raw_path)` calls the module's top-level `get_tools()`, or `tools()` when
`get_tools` is absent (a module defining both uses `get_tools()`), so an SEA can double as its own
tools file. It must return a list/tuple of callables (nothing is scanned). Tools should be plain synchronous functions with type-annotated,
keyword-bindable parameters and Google-style docstrings. Any problem raises `ToolsFileError` and
the task fails loudly instead of running without the tools. `_run_task_inner` loads it before
`agent.run`.

## Sources
- `src/kiss/server/agent_file.py` (`PARAM_FIELDS`, `HOOK_FIELDS`, `apply_agent_overrides`, `_check_override`, `AgentFileError`)
- `src/kiss/server/tools_file.py` (`execute_python_file`, `load_tools_file`, `ToolsFileError`)
- `src/kiss/server/task_runner.py` (`_run_task`, `_run_task_inner`)
- `src/kiss/agents/sorcar/daemon_client.py` (`resolve_tools_file`, `resolve_agent_path`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent.run` hook forwarding, `TOOL_PROFILES`)
- `API.md` (`extension_agent_path` of `kiss.server.sorcar.run`)
