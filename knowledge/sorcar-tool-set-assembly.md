---
title: How SorcarAgent builds its tool list (_get_tools)
uuid: 74506729-85fd-45af-b91a-d197a40364d9
summary: 'SorcarAgent._get_tools order: Bash/Read/Edit/Write (or Docker), browser,
  memory, skill, MCP, run_agent, ask_user_question, talk, set_model, decide, summary,
  run_parallel.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Tool set assembly (`SorcarAgent._get_tools`)

`SorcarAgent.perform_task` calls `_get_tools()` (only when `append_basic_tools` is true), appends the
caller's `tools`, and `RelentlessAgent.perform_task` prepends `finish`. With
`append_basic_tools=False` the agent gets only `finish` plus the caller's tools.

## Order
1. **Shell/file tools**
   - Host: `UsefulTools(stream_callback, stop_event, work_dir, jobs=self._background_jobs)` provides
     `Bash`, `bash_job`, `run_commands_parallel`, `Read`, `Edit` and `Write`. It also sets
     `self.context_reset_hook = useful_tools.forget_reads`. The background-job registry lives on the agent,
     so later prompts in the same chat can still reach earlier jobs. See `sorcar-useful-tools`.
   - Docker (`docker_manager` set): a `Bash` shim over `_docker_bash` that refuses `background=True`,
     `docker_manager.run_commands_parallel`, and `DockerTools` Read/Edit/Write. There is no `bash_job`.
2. **Browser**: `WebUseTool(...).get_tools()`. Only for the `full` profile with `web_tools=True`. Sub-agents
   get an `ephemeral` browser.
3. **Memory**: `self._memory_tools.tools()` when memory is enabled for the run.
4. **Restricted profile exit**: when the profile is not `full`, add `ask_user_question`, `talk`,
   `set_model`, `summary` and (if available) `decide`, then return only the tools whose `__name__` is in
   the profile's set. The skill, MCP, run_agent and run_parallel tools are never even built.
5. **Full profile extras**, in order: the `skill` tool (only when user or project skills exist), MCP tools
   (`make_mcp_tools`; failure is logged and skipped), `run_agent` (`agent_dispatch.make_run_agent_tool`),
   `ask_user_question`, `talk`, `set_model`, `decide` (if `decisions_tool_available()`), `summary`,
   and `run_parallel` + `number_of_cores` only when `is_parallel`.

## Inline tools defined in `_get_tools`
- `ask_user_question`: calls `_ask_user_question_callback` (the webview or CLI `_ask_user_in_terminal`).
  In unattended cron runs (`cron_agent.is_unattended`) it returns an error telling the model to proceed.
- `talk(language, text, emotion)`: broadcasts a `talk` event with optional synthesized audio
  (`synthesize_talk_audio`); the TTS cost goes to the task through `_attribute_tts_usage`.
- `set_model(model_name)`: swaps the live executor's model in place, keeping the conversation,
  usage info and thought signatures. It sanitizes `model_config` across providers and keeps custom
  `base_url`/`api_key`. It refuses cc/codex CLI models when `docker_image` is set. The change is
  display-only in the picker (`_show_model_in_picker`) and never persists `last_model`.
- `run_parallel` and `number_of_cores` (`os.process_cpu_count()`): see `sorcar-run-parallel-fanout`.
- `summary` is a module-level no-op (see `sorcar-summary-and-steering`).

## Why profiles cut tools
Every tool schema is re-sent on every step, so a reviewer carrying ~30 unused schemas pays for them on
every step. See `sorcar-tool-profiles`.

## Sources
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent._get_tools`, `perform_task`, `summary`)
- `src/kiss/agents/sorcar/useful_tools.py` (`UsefulTools`)
- `src/kiss/agents/sorcar/skills.py` (`make_skill_tool`), `mcp_servers.py` (`make_mcp_tools`), `decide_tool.py` (`make_decide_tool`)
- `src/kiss/agents/sorcar/agent_dispatch.py` (`make_run_agent_tool`)
