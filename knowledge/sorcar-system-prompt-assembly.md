---
title: How the system prompt is assembled and delivered (SYSTEM.md vs SYSTEM_LITE.md,
  brand tokens, SORCAR.md, Task Settings)
uuid: 37f72def-699b-463b-ab65-896f6cc9d3e9
summary: 'SYSTEM.md/SYSTEM_LITE.md loaded in base.py via render_brand; order: base,
  suffix, profile note, memory protocol, IMPORTANT_INSTRUCTIONS, Task Settings, SORCAR.md;
  sent as system_instruction.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# System prompt assembly

## The prompt files
- `src/kiss/SYSTEM.md` (about 24 KB) is the full prompt, in XML-tagged sections: `<identity>` (with Rule
  Precedence), `<visibility_constraint>`, `<tool_rules>` (tool usage, context continuation, the mandatory
  `summary` cadence, `talk` voice rules), `<web_research>` (the 10-site rule), `<code_style>`, `<workflow>`,
  `<testing>`, `<pre_finish_verification>` and `<sorcar_specific>`.
- `src/kiss/SYSTEM_LITE.md` (about 5 KB) keeps `<identity>`, `<visibility_constraint>`, a short `<tool_rules>`
  and `<sorcar_specific>`: same identity and output contract, no development/testing/web-research workflow,
  which saves tokens on chores and questions.
- Both are loaded **once at import** of `kiss.core.base`; editing them changes every Sorcar run.

## Loading (`src/kiss/core/base.py`)
- `SYSTEM_PROMPT = render_brand(SYSTEM.md text)`, `SYSTEM_PROMPT_LITE = render_brand(SYSTEM_LITE.md text)`.
- `kiss.core.brand.render_brand` replaces `{{PRODUCT_NAME}}`, `{{SHORT_NAME}}`, `{{TAGLINE}}` and `{{IDENTITY}}`
  with values from `src/kiss/agents/vscode/media/brand.json` (merged over `DEFAULT_BRAND`). Unknown `{{...}}`
  tokens are left as they are.
- On Windows, `_WINDOWS_SUFFIX` (`## Windows Environment`) is appended to both prompts: it says Git Bash is
  available, or asks for PowerShell syntax when `bash` is not on PATH.

## Layer 1: `SorcarAgent.run`
1. **Base**: the `base_system_prompt` argument if non-blank; otherwise `SYSTEM_PROMPT_LITE` when the pre-run
   classifier (`kiss.agents.sorcar.task_classifier`) says `is_simple`; otherwise `SYSTEM_PROMPT`. A failed or
   disabled classification keeps the full prompt. `is_simple` means "neither software development nor
   Internet search" (see `sorcar-task-classifier`).
2. `+ system_prompt`: an append-only suffix (the `append_to_system_prompt` of `run_agent` /
   `kiss.server.sorcar.run`).
3. `+ RESTRICTED_PROFILE_NOTE` when the tool profile is not `full` and `append_basic_tools` is true. It lists
   the offered tools (dropping `bash_job` under Docker) and says rules needing other tools (Write/Edit,
   PROGRESS.md, browser, memory writes, run_parallel, run_agent) do not apply.
4. `+ MEMORY_PROTOCOL` (`kiss.core.memoryfield.tools`, the `## Memory` section) when `_memory_root_for_run`
   enables persistent memory (see `memory-agent-tools`).
5. `+ "- The path of the file open in the editor is ..."` when `current_editor_file` is given.

Attachments do not touch the system prompt; they add an "# Important" note to the user prompt telling the
model not to open a browser to view them.

## Layer 2: `RelentlessAgent.perform_task`
6. `+ IMPORTANT_INSTRUCTIONS` (`# MOST IMPORTANT INSTRUCTIONS`): the `is_continue` instruction, HTML summary
   format, `- Work dir: ...` (`WORK_DIR_LINE`, omitted when tools run in a container that does not mount the
   work dir), and "Current process PID: N — NEVER kill this process".
7. `+ "# Task Settings"` from `_task_settings_section()`. Each value is collapsed onto one line so host
   strings cannot inject headings. Labels: Model name, Max budget (USD), Starting time, plus `_host_settings`
   (user id, IP address, OS, machine info). SorcarAgent adds "Parallel mode"; ChatSorcarAgent adds
   "Worktree mode", "Chat id", "Task id", "Is subagent" and "Parent task id".
8. `+ ~/.kiss/SORCAR.md` (`config_module.kiss_home() / "SORCAR.md"`), inlined verbatim when it exists (the
   user's preferences file). The repo-root `SORCAR.md` is **not** read by the agent; it is an example note.
   `SYSTEM.md` no longer asks for `Read("./SORCAR.md")` as a first step: the cost-levers work removed that
   wasted step, since the home copy is already inlined.

## Propagation to sub-agents
`run_parallel` forwards `_base_system_prompt` and `_system_prompt_suffix` to every child, so custom prompts
constrain the whole task tree. Each child re-runs its own classification (lite vs full).

## Delivery to the model (`KISSAgent.run`)
- A non-empty `system_prompt` is copied into `model_config["system_instruction"]` with **setdefault**, on a
  copy of the dict. If the caller already set `system_instruction`, that value wins and `system_prompt` is
  only printed.
- Each provider adapter maps `system_instruction` to its own system field. CLI models (`cc/*`, `codex/*`)
  receive it inside the prompt after `CLI_SYSTEM_PROMPT_HEADER`.
- The printer gets a `system_prompt` event when `print_prompts` is true.

## Prompt templates vs brand tokens
The user prompt is a template filled by `substitute_prompt_args` (`{key}` placeholders, one pass, unknown
braces left alone). Brand `{{...}}` rendering is separate and applies only to the prompt files.

## Sources
- `src/kiss/SYSTEM.md`, `src/kiss/SYSTEM_LITE.md`
- `src/kiss/core/base.py` (`SYSTEM_PROMPT`, `SYSTEM_PROMPT_LITE`, `_WINDOWS_SUFFIX`)
- `src/kiss/core/brand.py` (`render_brand`, `BRAND_FILE`)
- `src/kiss/core/kiss_agent.py` (`KISSAgent.run`)
- `src/kiss/core/models/model.py` (`CLI_SYSTEM_PROMPT_HEADER`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent.run`, `RESTRICTED_PROFILE_NOTE`, `_system_prompt_task_settings`)
- `src/kiss/agents/sorcar/relentless_agent.py` (`IMPORTANT_INSTRUCTIONS`, `WORK_DIR_LINE`, `TASK_SETTINGS_HEADER`, `_task_settings_section`, `_host_settings`, `perform_task`)
- `src/kiss/agents/sorcar/chat_sorcar_agent.py` (`_system_prompt_task_settings`)
- `src/kiss/core/memoryfield/tools.py` (`MEMORY_PROTOCOL`)
- `projects/cost-levers-implementation-plan.md` (removal of the mandatory SORCAR.md read)
