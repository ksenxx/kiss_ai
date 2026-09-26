---
title: Sorcar agent class hierarchy (RelentlessAgent, SorcarAgent, ChatSorcarAgent)
uuid: debedf37-9dd4-4398-8aad-36c3af706fbf
summary: Class chain Base -> RelentlessAgent -> SorcarAgent -> ChatSorcarAgent ->
  WorktreeSorcarAgent, what each layer adds, and the per-session KISSAgent executor
  that actually talks to the model.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Sorcar agent class hierarchy

```
kiss.core.base.Base
  └─ RelentlessAgent        relentless_agent.py   continuation loop, budget ledger, docker
       └─ SorcarAgent       sorcar_agent.py       tools, profiles, fan-out, classifier, memory
            └─ ChatSorcarAgent   chat_sorcar_agent.py   chat_id, task_history rows, chat prefix
                 └─ WorktreeSorcarAgent  worktree_sorcar_agent.py (git area)
```

None of these classes calls the LLM directly. `RelentlessAgent.perform_task` creates a **fresh
`kiss.core.kiss_agent.KISSAgent` per sub-session** (`KISSAgent(f"{name} Session-{n}")`) and runs it
with the assembled system prompt and tools. The step loop, tool execution and `_check_limits` belong
to that executor (see `core-agent-run-loop`).

## What each layer owns
- **RelentlessAgent**: `run()` (resolves settings through `_reset`, substitutes the prompt template, and
  optionally wraps everything in a `DockerManager` context), `perform_task()` (the multi-session
  `is_continue` loop), the usage ledger (`budget_used`, `total_tokens_used` and `total_steps` are
  properties backed by `_UsageLedger`), `_check_total_budget`, and `_system_prompt_task_settings`.
  Defaults: `DEFAULT_MAX_BUDGET = 200.0`, `DEFAULT_MODEL_NAME = "claude-opus-4-6"`, and 10000 for both
  max_steps and max_sub_sessions.
- **SorcarAgent**: builds tools (`_get_tools`), tool profiles (`TOOL_PROFILES`, `_tool_profile`),
  `run_parallel` (`_run_tasks_parallel` plus the module-level `run_tasks_parallel` engine), pre-run task
  classification (`_classify_task_once`), the lite/full system prompt choice, memory tools, steering
  hooks (`_drain_pending_user_messages`, `_block_finish_when_user_message_pending`), and
  `set_model`/`talk`/`ask_user_question`. `_reset` resolves the model through `_resolve_model_name`
  (caller's model, then the last-selected model, then `get_default_model()`), so the relentless
  default model is rarely what runs. `run()` adds `web_tools`, `is_parallel`, `base_system_prompt`,
  `append_basic_tools`, `use_memory`, `tool_profile` and `ask_user_question_callback`.
- **ChatSorcarAgent**: multi-turn chat state (`chat_id`, `build_chat_prompt`), persistence of each
  run as a `task_history` row, the `_subagent_info` marker, and the `new_tab` broadcast for
  sub-agents. Every `run_parallel` child is a `ChatSorcarAgent`. See `sorcar-chat-session-agent`.
- **WorktreeSorcarAgent**: git worktree isolation and auto-commit. Belongs to the git area (`git-overview`).

## Entry points
- The CLI `sorcar` (`pyproject.toml` script `kiss.agents.sorcar.sorcar_agent:main`) accepts
  `-t/--task` or `-f/--file`, plus `-m/--model`, `-b/--max-budget` and `--work-dir` (default
  `KISS_WORKDIR`, else the cwd). It runs a plain `SorcarAgent` and exits 0 only if the result has
  `success: true`.
- The daemon/VS Code server runs `WorktreeSorcarAgent`. Python scripts go through `kiss.server.sorcar.run`.

## Gotcha
`_get_tools` must run after `docker_manager` is set, which is why `SorcarAgent.perform_task` (not
`run`) builds the tools, and why the Docker variants of Bash/Read/Edit/Write only appear inside
`RelentlessAgent.run`'s `DockerManager` block.

## Sources
- `src/kiss/agents/sorcar/relentless_agent.py` (`RelentlessAgent`, `_reset`, `run`, `perform_task`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent`, `_reset`, `_resolve_model_name`, `perform_task`, `main`)
- `src/kiss/agents/sorcar/chat_sorcar_agent.py` (`ChatSorcarAgent`)
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` (`WorktreeSorcarAgent`)
