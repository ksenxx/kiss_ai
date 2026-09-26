---
title: KISSAgent run loop, steps and per-step sequence
uuid: de88f941-54a8-421a-85fd-6432b89c784c
summary: How KISSAgent.run works - _reset, _setup_tools, _set_prompt, _run_agentic_loop,
  _execute_step order (pre_step_hook, compaction, llm_call_hook, generate, tools),
  max_steps, text-only retries.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# KISSAgent run loop, steps and per-step sequence

## Entry: `KISSAgent.run(model_name, prompt_template, arguments, system_prompt, tools, ...)`
1. Stores `llm_call_hook` and `tool_call_hook`. If `system_prompt` is non-empty, it is copied into
   `model_config["system_instruction"]` with `setdefault`. **Gotcha:** if the caller's
   `model_config` already has `system_instruction`, that value wins and `system_prompt` is ignored,
   although it is still printed.
2. `_reset(...)` clears per-run state (`messages`, counters, `function_map`, `_fallback_used`, compaction
   threshold, trajectory stamp) **before** it builds the model. `model()` raises for an unknown name, and the
   `finally: self._save()` must not overwrite the previous run's trajectory file.
   The model object is reused (via `reset_conversation()` + `rebind_callbacks()`) when both name and config
   are unchanged. Otherwise it is rebuilt with `kiss.core.models.model_info.model(...)`.
   Defaults are `max_steps=10000` and `max_budget=10.0`.
3. A non-agentic run that is passed `tools` raises `KISSError`.
4. `_setup_tools` appends `self.finish` when no tool is named `finish`, registers the tools
   (`_add_functions` raises on duplicate names) and caches the schema once in `_cached_tools_schema`
   (see `core-tool-schema-from-python-functions`).
5. `_set_prompt` fills `{key}` placeholders with `kiss.core.utils.substitute_prompt_args`. This is a
   single regex pass that leaves unknown braces alone, so JSON in prompts is safe and a placeholder
   with no matching key stays literal. The prompt then goes to `model.initialize(prompt, attachments)`.
6. `is_agentic=False` leads to `_run_non_agentic` (a single `_generate_once`). Otherwise `_run_agentic_loop`.
7. `finally: self._save()` always writes the trajectory.

## `_run_agentic_loop`
- `while self.step_count < self.max_steps`. The step bound lives only here. Running out raises
  `KISSError("... exceeded N steps.")`.
- Each iteration first checks `self.model.runs_task_to_completion`. Sorcar's `set_model` can swap to a CLI
  model mid-run; see `models-cli-backed-cc-codex`.
- `step_count += 1`, `_check_limits()`, then `_execute_step()`. A non-None return value is the result.
  It is printed as a `result` event and returned.
- The error handling is described in `core-errors-retry-and-model-fallback`.

## `_execute_step` order (one model call)
1. `pre_step_hook(model)`. Sorcar uses it to drain queued user messages.
2. `_maybe_compact_conversation()` (see `core-context-compaction`).
3. `llm_call_hook(new_messages)` receives `model.conversation[_llm_hook_conversation_index:]`, and its return
   value replaces that slice. The index moves forward both before and after the model call, so a failed
   call never hands already-hooked messages back to the hook, and the assistant turn does not count as new.
4. `_prompt_cache_touched_at = time.time()`, then
   `model.generate_and_process_with_tools(function_map, tools_schema=_cached_tools_schema)` returns
   `(function_calls, response_text, response)`.
5. Usage accounting, `model.set_usage_info_for_messages(usage_info)`, and a `usage_info` printer event.
   The base `Model` appends the usage string (`Steps: a/b, Context: x/y tokens, Total tokens: n, Budget: $u/$m,`)
   to the next user or tool-result message, so this is how the model sees its step count and budget.
   When a turn has any call other than `finish`, `_check_limits()` also runs before the tools do.
6. **No tool calls:** `_consecutive_no_tool_calls += 1`. From the second consecutive turn
   (`MAX_CONSECUTIVE_NO_TOOL_CALLS = 2`), an empty turn raises `_EmptyModelResponseError`, and a text turn
   finishes implicitly (see `core-finish-contract-and-implicit-finish`). Otherwise a user nudge is added:
   "Your response MUST have at least one function call...".
7. **Tool calls:** each call goes through the hooks and guard and is then executed (see
   `core-tool-execution-and-hooks`). After every non-finish tool, `_check_limits()` runs. A limit error stops the remaining calls in the turn; it is
   held and raised only **after** the step is recorded in `messages`, because the executed tools really ran.
8. Two entries go into the trajectory: a `model` message (the text plus a python repr of each call plus
   usage) and a `user` message (`[name]: result` joined).
9. When `finish` ran unblocked, its result is returned. Otherwise
   `model.add_function_results_to_conversation_and_return(function_results)` runs and the loop continues.

Several `finish` calls or mixed calls in one turn: calls execute in order; limits are checked
after every call except an unblocked `finish`, and a limit error breaks the loop and is raised; otherwise the step returns the result of the last unblocked `finish`.

## Sources
- `src/kiss/core/kiss_agent.py` (`KISSAgent.run`, `_reset`, `_setup_tools`, `_set_prompt`, `_run_agentic_loop`, `_execute_step`)
- `src/kiss/core/utils.py` (`substitute_prompt_args`)
