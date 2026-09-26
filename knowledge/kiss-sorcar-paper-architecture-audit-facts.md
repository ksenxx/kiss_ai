---
title: KISS Sorcar code facts that the architecture section of kiss_sorcar.tex must
  match
uuid: 9ae7bcbc-e610-4adb-a36e-70960a01bdcc
summary: 'Verified code facts (as of v2026.9.24) for auditing the Agent Architecture
  section of papers/kisssorcar/kiss_sorcar.tex: compaction, 70% handoff, classifier
  benchmark regeneration, chat history digest, tool inventory.'
created: '2026-09-26T20:00:20Z'
updated: '2026-09-26T20:00:20Z'
---
# Code facts for the kiss_sorcar.tex architecture section (verified 2026-09-26, v2026.9.24)

- KISSAgent DOES compact in place: `_maybe_compact_conversation` (src/kiss/core/kiss_agent.py) replaces old large tool outputs with stubs when `DEFAULT_CONFIG.tool_output_compaction` (default True, `KISS_TOOL_OUTPUT_COMPACTION`) and context >= 100k tokens, then every further 100k (`compaction_start_tokens`, `compaction_step_tokens` in src/kiss/core/config.py). Added in commit 756ddf4c0 (2026-09-19). Any sentence saying "instead of compacting the context in place" is wrong.
- Context hand-off is at 70% of the window, not at overflow: `CONTEXT_LIMIT_FRACTION` (kiss_agent.py ~line 75, config `context_limit_fraction`, env `KISS_CONTEXT_LIMIT_FRACTION`); `_check_limits` raises `ContextWindowExceededError`, RelentlessAgent then runs the trajectory summarizer.
- Non-retryable errors: KISSAgent first tries `_try_switch_to_fallback` (OpenRouter twin) and raises only if no fallback. `MAX_CONSECUTIVE_ERRORS = 3` is a module constant, not configurable.
- RelentlessAgent stops after `MAX_ZERO_PROGRESS_SESSIONS = 2` zero-progress sessions or `max_sub_sessions`.
- Sorcar default tool list (non-Docker): Bash, bash_job, run_commands_parallel, Read, Edit, Write, web tools (go_to_url, click, type_text, press_key, scroll, screenshot, get_page_content, show_browser, close_browser), memory_* (7), run_agent, run_parallel (if parallel), number_of_cores, ask_user_question, talk, set_model, decide (if OPENROUTER key + catalog), summary, MCP/skill tools. `TOOL_PROFILES` (full/review/shell/assistant/bash) filter it.
- Chat history: `MAX_TASKS = 10` keeps first two + latest eight, BUT `chat_history_digest` (default True since 756ddf4c0) keeps only the newest `DIGEST_FULL_RESULTS = 2` results in full, digests older ones (600/300 chars, tags stripped) and drops oldest-first when the prefix exceeds `DIGEST_MAX_PREFIX_CHARS = 6_000` (src/kiss/agents/sorcar/chat_sorcar_agent.py).
- Chat session ops: `new_chat`, `resume_chat_by_id`, `resume_from_task_id` (seed from a task's parent chain). There is no "resume by task description".
- Classifier benchmark: benchmarkings/task_classifier/results.json was regenerated 2026-09-20 (commit 6f2845ca4). Current: jev-decisions 88.7% (95.1% of 344 certain), median 0.25 s, mean $0.000027; claude-fable-5-1 84.3%, median 3.1 s, mean $0.00614 (226x). The paper's 88%/94%/0.22 s/$0.000025/80%/3.5 s/200x are from the 2026-09-18 file.
- Read-before-modify guard in Edit/Write shipped in v2026.9.19; the tmp/ + cron-dir Write exemption (`_is_scratch_path`) shipped in v2026.9.22.
- Worktree branch: `kiss/wt-<int(time.time())>-<uuid4().hex[:8]>` (worktree_pool.new_task_branch); dir `.kiss-worktrees/kiss_wt-...`.
- JsonPrinter: only the current block type / task binding is thread-local; recordings and bash buffers are dicts keyed by task_id (src/kiss/server/json_printer.py ~405-413). Sub-agent events are persisted too (persistence.py ~4060).
