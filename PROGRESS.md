# PROGRESS — invalid-byte filename repro for current diff (this session, COMPLETE)

REPRO-ONLY task; no source edits. Built ./tmp/repro_harness.py + ./tmp/consume_manifest.js
(temp, deleted at finish) exercising git_worktree/\_porcelain_entries, \_diff_name_only,
merge_flow.\_porcelain_paths, diff_merge.\_parse_diff_hunks/\_capture_untracked/\_prepare_merge_view
in isolated temp git repos, run twice: current worktree code vs HEAD code
(git archive HEAD src/kiss → temp tree, sys.path swap). Findings:

- APFS rejects invalid-UTF8-byte filenames (EILSEQ errno 92) → on-disk invalid-byte
  tracked-modified/untracked cases impossible on macOS; index/history-only states
  (tracked deleted via update-index --cacheinfo; staged modified) fully reproduced.
- Special-char combos (quote, backslash, newline, tab, leading/trailing space, café,
  " -> " in name, U+2028): CURRENT code exact end-to-end (porcelain, \_diff_name_only -z,
  merge manifest names/paths, Node fs consumption all clean, rename both sides exact).
- HEAD defect FIXED by current diff #1: HEAD GitWorktreeOps.\_diff_name_only used
  strip().splitlines() → 'line\\u2028sep.txt' split into 'line'+'sep.txt', ' leading.txt'
  mangled to 'leading.txt'. Current -z version byte-exact.
- HEAD defect FIXED by current diff #2: HEAD diff_merge.\_git strict utf-8 decode →
  UnicodeDecodeError crash in \_parse_diff_hunks/\_prepare_merge_view on any invalid-byte
  path in git output. Current surrogateescape decode parses byte-exact ('caf\\udce9-latin1.txt',
  round-trips via os.fsencode).
- RESIDUAL (NOT newly introduced — HEAD crashed earlier in same scenario): current
  \_prepare_merge_view raises uncaught OSError EILSEQ at diff_merge.py:958
  (deleted_placeholder.write_text) on macOS for tracked-deleted invalid-byte paths
  (realistic: cloning a Linux-created repo with such a path leaves it ' D' on APFS).
- Platform limitation (both variants): manifest JSON stores lone-surrogate names
  ("\\udce9", json.dumps fine); Node JSON.parse accepts, but Node fs converts lone
  surrogate → U+FFFD (efbfbd) → ENOENT on Linux-resident raw-byte files. VS Code/Node
  can never open such paths from JSON; unchanged by the diff.

______________________________________________________________________

# PROGRESS — consistency analysis web_server.py vs server.py/task_runner.py/media JS

ANALYSIS ONLY task. No source edits, no tests. Output: ./tmp/candidates-E.md (numbered findings, >85% confidence, with grep proof). Then finish.

## Done so far (session 1)

- web_server.py dispatch `_dispatch_client_command` (line 4442): handles cliEvent(4501), cliTabHello(4511), cliTaskStart(4524), cliTaskEnd(4538), setWorkDir(4551), openFile(4557, WSS only), voiceTranscribe(4569), \_VSCODE_ONLY_COMMANDS(4579), activeTasksQuery(4581), ready(4584), submit(4610), getWelcomeSuggestions(4613), runUpdate(4616), serverReset(4619), mergeAction(4622), closeTab(4641, web only), else -> \_translate_webview_command (3509; translates userActionDone(3531), resumeSession id->chatId(3540)) -> \_run_cmd -> server.py:\_handle_command(636) -> commands.py \_HANDLERS (1452-1479): run, stop, getModels, selectModel, getHistory, getFrequentTasks, deleteTask, deleteFrequentTask, setFavorite, getFiles, recordFileUsage, userAnswer, appendUserMessage, resumeSession, mergeAction, closeTab, newChat, complete, getInputHistory, getAdjacentTask, generateCommitMessage, worktreeAction, autocommitAction, setWorkDir, getConfig, saveConfig, cliInfo. Unknown -> error broadcast.
- \_VSCODE_ONLY_COMMANDS (web_server.py:900): focusEditor, webviewFocusChanged, openFile, resolveDroppedPaths, pickFolder, sizeReport.
- JS postMessage commands (main.js): notificationAction(318), closeTab(1124), newChat(1662), getWelcomeSuggestions(1663), complete(2398), resumeSession(2848,5499,8512), getAdjacentTask(3857), getHistory(4143...), getInputHistory(4821), stop(5088,6890), sizeReport(5111), runUpdate(6157,7017), worktreeAction(6407,6416), autocommitAction(6476,7115), mergeAction(6718), ready(6747), getConfig(6754,8962), webviewFocusChanged(6760,6763), focusEditor(6773), serverReset(7068,7080), openFile(7515), resolveDroppedPaths(7572), appendUserMessage(7651), submit(7666), userAnswer(7831), selectModel(7938), setFavorite(8230), deleteTask(8275), saveConfig(8912), setWorkDir(8925,9280), getFrequentTasks(8983), deleteFrequentTask(9140), getFiles(9357), recordFileUsage(9488).
- voice.js posts: voiceAck(525, gated cfg.mode==='webview' only), voiceTranscribe(629, both modes but web_server handles), voiceSensitivity(1009, webview-gated), voiceToggle(1030, webview-gated). => voice cmds properly gated, NOT an inconsistency.
- notificationAction: only extension sends notifications WITH actions (src/WebviewNotifications.ts:58,61,76); web_server notifications carry no actions, and main.js removeNotification only posts notificationAction when notifyOnClose (actions present, no local onClick). So currently consistent (fragile but not >85% bug). Extension handles it at SorcarSidebarView.ts:1553. NOT handled by web_server dispatch/\_VSCODE_ONLY set — POSSIBLE finding: unhandled-if-ever-sent-from-web ("Unknown command" banner). Confidence borderline; mark as maybe.
- CANDIDATE FINDING A (server-cmd-never-handled?): "pickFolder" is in \_VSCODE_ONLY_COMMANDS but no JS grep hit for pickFolder in media/\*.js yet — verify who sends pickFolder (extension settings panel? grep media + src/).
- CANDIDATE FINDING B: "generateCommitMessage" and "cliInfo" and "run" and "getModels" in \_HANDLERS — verify some client sends them (extension src/, main.js sends selectModel but getModels? grep 'getModels' in main.js/src). Possibly server-handled commands never sent by JS (extension may send them via UDS).
- Events broadcast by web_server.py+json_printer.py ("type": X): error, system_output, fileContent, update_available, text_end, status, notification, merge_nav, mergeAction(?? broadcast as event—check), activeTasksResponse, voiceSpeech, usage_info, tool_result, tool_call, thinking_start, thinking_end, text_delta, tasks_updated, setTaskText, run(??), resumeSession, result, remote_url, notice, merge_started, merge_data, focusInput, closeTab, auth_required, auth_ok. NOTE: this grep only covered web_server.py+json_printer.py with exact `"type": "x"` pattern; server.py/task_runner.py/commands.py also broadcast (status, result, error, clear, askUser, task_done, task_error, task_stopped, task_interrupted, tasks_updated, worktree_result, openSubagentTab, new_tab...). JS listens (case ''): full list captured — notable JS-listened events to verify senders: adjacent_task_events, appendToInput, askUserDone, autocommit_done, autocommit_prompt, clearChat, commitMessage, daemonStatus, droppedPaths, ensureChat, followup_suggestion, ghost, prompt, showWelcome, subagentDone, system_prompt, talk, taskDeleted, triggerStop, updateSetting, welcome_suggestions, worktree_done, thinking_delta. Check which of auth_required/auth_ok/activeTasksResponse/voiceSpeech(handled voice.js msg.type==='voiceSpeech' line1077 OK)/update_available are listened by JS: JS has update_available case ✓, but 'auth_required','auth_ok','activeTasksResponse' NOT in JS case list — check who consumes (likely CLI kiss client / task_runner? activeTasksQuery came from cliTabHello clients). Verify before claiming.
- Constants found (web_server.py): VOICE_MODEL_URL(100-102, vosk-model-small-en-us-0.15.tar.gz, alphacephei URL — need exact line), \_WS_PING_TIMEOUT=10(245), \_TUNNEL_STARTUP_GRACE=120(257), \_MAX_RESTORED_TABS=32(325), \_MAX_ATTACHMENTS=32(327), \_MAX_PROMPT_BYTES=1_000_000(369), \_MAX_LINE_BYTES=64MiB(378), \_MAX_VOICE_AUDIO_B64=4MiB(384), \_TAB_CLOSE_GRACE=10.0(397), \_OPEN_FILE_MAX_BYTES=2_000_000(898). voice.js uses cfg.modelUrl (729) — server injects it; vosk.js takes modelUrl param — so model URL likely single-sourced from web_server VOICE_MODEL_URL via page template (verify chat.html/template injection of modelUrl + ackAudioUrl). Check main.js for hardcoded 32 attachment limit / 1MB prompt cap duplicates; check voice.js hardcoded audio cap vs 4MiB; check ping interval in main.js (WS keepalive) vs \_WS_PING_TIMEOUT; check \_TAB_CLOSE_GRACE duplicated in server.py/task_runner.py.

## Session 2 verified results

- CONFIRMED FINDING 1 (redundancy, dead listener): main.js:4753 `case 'ensureChat':` (body 4754-4757: `if (tabs.length === 0) createNewTab();`) has NO sender anywhere in src/kiss (py/ts/js). Only other repo ref: src/kiss/tests/agents/sorcar/test_vscode_tabs.py:699 which merely indexes the JS source (`end = self.js_src.index("case 'ensureChat':", idx)`). grep -rn "ensureChat" src/kiss → only those 2 hits. Fix: remove the dead case (and adjust that structural test). Risk low. Confidence 95%.
- CONFIRMED FINDING 2 (redundancy, duplicated helper): `_broadcast_to_conn` identical logic in web_server.py:~4667 (`if conn_id: event["connId"] = conn_id; self._printer.broadcast(event)`) and server.py:670 (`if conn_id: event["connId"] = conn_id; self.printer.broadcast(event)`). Same semantics, duplicated docstring intent. Fix: module-level shared helper broadcast_to_conn(printer, event, conn_id). Risk low. Confidence 90%.
- Voice cmds voiceAck(voice.js:525)/voiceSensitivity(1009)/voiceToggle(1030) are gated `cfg.mode === 'webview'` → NOT inconsistencies. voiceTranscribe handled web_server.py:4569. Model URL single-sourced: server serves "/voice-model.tar.gz" (web_server.py:2817 injects cfg.modelUrl; vosk.js takes modelUrl param, no hardcoded URL). ackAudioUrl injected web_server.py:2820. NO constant mismatch here.
- No JS duplication found for \_MAX_ATTACHMENTS/\_MAX_PROMPT_BYTES/\_MAX_RESTORED_TABS/\_MAX_VOICE_AUDIO_B64 (grep found no 32/1_000_000/4*1024*1024 caps in main.js/voice.js; main.js:6739 restoredTabs uncapped client-side = server-side defense only, fine).
- pid-liveness (\_is_pid_alive web_server.py:1033) and version reading (\_read_version web_server.py:2883 etc.) exist ONLY in web_server.py — no py duplication. (Note: \_find_install_script web_server.py:932 + \_KISS_AI_ROOT web_server.py:918 are DOCUMENTED Python twins of extension src/installerPath.js findInstallScript()/kissAiRoot() — cross-language duplication, candidate FINDING 3; verify installerPath.js values match: root = Path.home()/"kiss_ai", probe "install.sh".)
- Events: subagentDone sent by server.py ✓; updateSetting sent by main.js? only main.js+types.ts have it → check who SENDS updateSetting (extension? types.ts suggests ToWebview). auth_required/auth_ok/activeTasksResponse consumed by CLI/scripts/tests not main.js case switch (main.js:4464 comment says auth_required handled in pre-auth dispatch) — verify auth_ok consumer before claiming anything.
- server.py/task_runner.py constants: no GRACE/_MAX_/\_TIMEOUT module constants (only \_REPLAY_STRIPPED_EXTRA_KEYS server.py:103). So no Python-Python constant mismatch for grace/ping.

## Session 3 — COMPLETE. ./tmp/candidates-E.md written with 6 findings (notificationAction not exempted; pickFolder dead exemption; ensureChat dead listener; updateSetting dead listener+type; \_broadcast_to_conn duplicated web_server.py:4663/server.py:670; installer path Python/JS twins values match) + non-findings section. Task done; finish() next.

## TODO (session 3 — done)

1. Verify FINDING 3: Read src/kiss/agents/vscode/src/installerPath.js (or .ts) — confirm kissAiRoot()=~/kiss_ai and findInstallScript probes install.sh; note exact lines.
1. updateSetting: grep -rn "updateSetting" src/kiss/agents/vscode/src/\*.ts main.js — if no sender → dead listener finding like ensureChat. Also check 'triggerStop', 'appendToInput', 'clearChat', 'ghost', 'daemonStatus' senders exist (each had ≥2 py/ts files so likely fine; only updateSetting(1) suspicious).
1. Check web_server.py broadcast `"type": "mergeAction"` and `"type": "run"` and `"type": "resumeSession"` and `"type": "closeTab"` events (from earlier grep) — these overlap COMMAND names; find their line numbers + context: are they server→client events that main.js listens for? main.js has no case 'mergeAction'/'run'/'resumeSession'/'closeTab' → find who consumes (CLI? extension SorcarSidebarView?). Possible inconsistency/dead-broadcast finding.
1. Quick scan for duplicated near-identical helper bodies between web_server.py and server.py/task_runner.py: candidates `_get_tab`, `drop_connection_state`, `_load_last_model`, `_load_model_usage`, history file paths; grep "def \_load_last_model|def \_load_model_usage" across files. Also json_printer.py event types vs JS listeners (json_printer broadcasts thinking_delta? check).
1. Write ./tmp/candidates-E.md with final numbered findings (include Finding 1,2,3 + any new), each: TYPE, file:line both sides, snippets, grep proof, minimal fix, risk, confidence. Keep only >85%.
1. finish(success=True) with full findings list in summary. candidates-E.md stays (it is the deliverable). PROGRESS.md may remain.

## TODO (session 2 — done, superseded)

1. grep main.js for attachment limit (e.g. `32`, `maxAttach`, slice(0,32)), prompt cap, audio cap (4*1024*1024), ping/keepalive intervals, grace constants; compare values vs web_server.py constants. Also `_MAX_RESTORED_TABS` vs main.js restoredTabs slice.
1. Verify modelUrl injection: grep web_server.py for "modelUrl"/"ackAudioUrl" and chat.html; confirm vosk.js has no hardcoded model URL (grep 'alphacephei|https://' vosk.js voice.js).
1. Event-type cross-check: build sender list from server.py/commands.py/task_runner.py/json_printer.py + extension src/\*.ts (extension also sends events to webview: configData, models, history, files, completions, inputHistory, frequentTasks, droppedPaths, commitMessage, daemonStatus...). For each JS `case` with NO sender anywhere → dead listener finding; for each broadcast type with NO JS/extension listener → dead event finding. Use grep -rn over repo incl src/kiss/tests.
1. Duplicated helpers: grep for version reading (`_read_version|version(`), pid liveness (`os.kill(pid, 0)|pid_alive`), history handling, path helpers (`_kiss_dir|history_path|_HISTORY`), across web_server.py vs server.py vs task_runner.py vs helpers.py. Compare bodies. Also \_find_install_script (web_server.py:932) vs installerPath.js findInstallScript (documented twin — Python/TS duplication, note as finding #4-type redundancy). Also \_KISS_AI_ROOT (web_server.py:918) vs kissAiRoot() installerPath.js.
1. Check `mergeAction`/`run`/`resumeSession` broadcast-as-event vs command name collision (web_server broadcasts "type":"mergeAction"? line found in grep — verify context; also "type":"run" broadcast — that's probably server->CLI).
1. Write ./tmp/candidates-E.md numbered entries: TYPE, file:line both sides, snippets, grep proof, minimal fix, risk, confidence>85%. Then finish with the full list in summary.
1. Delete tmp scratch files created (only candidates-E.md is the deliverable — task says write findings there; keep it, it's in tmp per task spec; do NOT delete candidates-E.md since it IS the requested output).

______________________________________________________________________

# PROGRESS — task 6: redundancy/inconsistency audit of the 13 named sorcar files (ANALYSIS ONLY)

Deliverable: ./tmp/findings-6.md (16 numbered findings, each with TYPE, file:line ranges,
code evidence, grep proof, minimal fix, risk). No source files edited, no tests created.

## Done (single session, completed)

- Fully read: sorcar_agent.py, worktree_sorcar_agent.py, chat_sorcar_agent.py, persistence.py
  (all 3005 lines in 3 chunks), git_worktree.py, code_graph.py, mcp_servers.py, useful_tools.py,
  web_use_tool.py, skills.py, custom_commands.py, running_agent_state.py, __init__.py; plus
  vscode/diff_merge.py:100-220 (the `_git` twin) and merge_flow.py:795-815 (unstaged_files comment).
- Verified every claim by grep across src/ INCLUDING src/kiss/tests (usage of unstaged_files/
  staged_files/load_original_branch/load_baseline_commit/\_prefix_match_task/\_context_task_id/
  run_tasks_parallel/grep_hint/\_allocate_chat_id/new_chat callers/def \_git definitions...).
- Final findings (see ./tmp/findings-6.md for full detail):
  1 dual `_git` wrappers (git_worktree.py:132 vs diff_merge.py:150) — divergent timeout /
  env-scrub / surrogateescape hardening (redundancy+inconsistency, MEDIUM).
  2 copy-pasted parallel `_run_single` executors (sorcar_agent.py:1000-1148 vs
  chat_sorcar_agent.py:385-560) (redundancy, MEDIUM).
  3 `_finalize_worktree` vs `_preserve_pending_worktree_for_review` duplicate commit-and-clean
  sequence (worktree_sorcar_agent.py:327-393 vs 653-737) (redundancy).
  4 `new_chat()` does not clear one-shot `_context_task_id` seed (chat_sorcar_agent.py:125-128)
  (inconsistency).
  5 GitWorktreeOps.unstaged_files/staged_files/load_original_branch/load_baseline_commit —
  zero production callers; merge_flow.py:804-809 deliberately avoids the first two due to
  `_diff_name_only`'s strip() whitespace-mangling defect (redundancy+inconsistency).
  6 run_tasks_parallel docstring omits `totals_out` (inconsistency, doc).
  7 `_prefix_match_task` (persistence.py:1298) production-dead wrapper (redundancy).
  8 third chat-id mint site in `_add_task` (persistence.py:1071) contradicts documented
  single-`_allocate_chat_id` contract (inconsistency).
  9 stale pre-UUID/pre-column docstrings in `_load_subagent_rows_by_parent_task_id` (:2558)
  and `_load_chat_context` (:2684) (inconsistency, doc).
  10 sub-agent filtering via `_HISTORY_NOT_SUBAGENT` SQL in some readers, Python `parent_task_id`
  checks in others (inconsistency).
  11 duplicated grep_hint interception (useful_tools.py:884-899 vs sorcar_agent.py:480-496)
  (redundancy).
  12 duplicated frontmatter regex + parse pipeline + CLAUDE_CONFIG_DIR resolution
  (skills.py:62/130/109 vs custom_commands.py:61/144/128) (redundancy).
  13 stale `source` docstrings in custom_commands.py:106/150 (four values, docs list two)
  (inconsistency, doc).
  14 `_is_profile_in_use` (web_use_tool.py:295) re-implements `_read_lock_pid`(:272)+`_pid_alive`(:114)
  (redundancy).
  15 `code_graph._pid_is_running` (:1273) third PID-liveness probe (redundancy; standalone caveat).
  16 `_ensure_graph_git_excluded` (code_graph.py:823) re-implements hardened
  `GitWorktreeOps._append_info_line` (git_worktree.py:907) and re-introduces its fixed
  UnicodeDecodeError + no-lock bugs (redundancy+inconsistency, MEDIUM).
- mcp_servers.py, running_agent_state.py, __init__.py: no findings above the 90% bar.

______________________________________________________________________

# PROGRESS — analysis task C: commands.py + autocomplete.py redundancies/inconsistencies (COMPLETE)

ANALYSIS ONLY task (separate from candidates-E above). Deliverable written: ./tmp/findings-c.md — 7 grep-verified findings:

1. redundancy: `_complete_from_active_file` (autocomplete.py:96-135) production-dead — only tests call it; live path is `_complete_many` → `_active_file_identifier_matches` + `_ghost_suffix`. (low)
1. redundancy: identifier-harvest core duplicated — autocomplete.py:137-188 (`_active_file_identifier_matches`, regexes at 171/181/182, 50000-char cap at 167) ↔ cli_repl.py:580-600 (`CliCompleter._active_file_suffix`, regexes 584/592/593, cap 588). Same 3 regexes byte-identical. Fix: shared helper in vscode/helpers.py. (medium)
1. redundancy: `/help` body duplicated verbatim — commands.py:1217-1240 (`_cmd_cli_info` help subtype, "Input fast-completes" at 1232) ↔ cli_repl.py:806-829 (`_print_help`, same paragraph at 821); normalized diff = identical. Fix: `build_help_text(work_dir)`. (low)
1. redundancy: cap constant 20 duplicated — autocomplete.py:37 `_COMPLETIONS_LIMIT = 20` (comment says it "mirrors" the file-picker cap) ↔ helpers.py:317 `rank_file_suggestions(limit=20)`. (low)
1. inconsistency: stale `_parse_int` docstring (commands.py:110-128) cites `_cmd_get_adjacent_task` int-parse + `"taskId": "abc"` example, but taskId is now string-validated via `_opt_str` (commands.py:927); real callers are offset/generation (467-468) and limit (478). (low)
1. inconsistency: stale comment persistence.py:117-118 names dead `_complete_from_active_file` and wrong callee `_load_chat_context`; live: `_active_file_identifier_matches` → `_load_chat_context_text` (autocomplete.py:177). (low)
1. redundancy: task-ownership answer-queue filter duplicated — commands.py:692-700 (`_resolve_user_answer_queue`, comment admits "mirrors task_runner.\_resolve_task_answer_queue / BUG-TR2-2") ↔ task_runner.py:2022-2032 (`_resolve_task_answer_queue`). Fix: shared predicate; do NOT merge the resolvers (different lookup directions/locking). (medium)

Also verified-and-NOT-reported (correct shared imports): SLASH_COMMANDS (commands.py:1200 imports cli_repl.py:141; no TS copy exists), clip_autocomplete_suggestion, rank_file_suggestions, tricks helpers, \_prefix_match_tasks, \_record_file_usage/\_record_model_usage, custom_commands/skills/mcp helpers. No source files edited, no tests created, no commits.

## Session 4 — COMPLETE

- Read cli_client.py 900-1697 (full file now read); ran final verification greps
  (dup \_PROMPT, steering literals, dispatch-chain comments, STEER_TITLE wording,
  kiss/core/utils symbol list, \_default_kiss_dir KISS_HOME, line citations).
- Wrote FINAL ./tmp/findings-5.md: 18 verified findings (12 redundancies,
  6 inconsistencies) + a "checked and NOT reported" section for candidates that
  grep disproved (\_term_size, \_MAX_TALK_IDS parity, reset_shared_player_for_tests,
  double callback.close(), etc.). No source files edited; no tests created.

______________________________________________________________________

# MASTER TASK: find+test+remove all redundancies/inconsistencies in vscode/, core/, sorcar/

Orchestrator log (claude-fable-5 for dev, gpt-5.6-sol review pending):

1. Analysis wave (6 parallel agents): findings written to ./tmp/findings-1..6.md
   (~120 grep-verified findings + non-findings appendices).
1. Fix wave 1 (6 parallel agents, disjoint files) — ALL DONE, logs in
   ./tmp/fixlog-{models,core,sorcar,webserver,cli,vscode-misc}.md:
   models (14 fixes), core (dead utils/MultiPrinter/JUDGE_PROMPT deleted, finish()
   standardized, SUMMARIZER_PROMPT fixed), sorcar non-cli (git timeout, worktree
   cleanup helper, new_chat leak, persistence dedup/SQL predicate, skills/commands
   shared frontmatter), web_server+server (25 findings incl. byte-based prompt cap,
   taskId in \_fanout_cli_status, version-helper strictness, dead JS listeners),
   cli (escape tables/proto constants shared via cli_panel, STEER_TITLE fixed,
   \_parse_kv moved to cli_helpers, VoiceListener.final_flush), vscode-misc
   (diff_merge env-scrub+surrogateescape, metric-delta dedup + step_count fallback,
   dead autocomplete/voice code deleted).
1. Fix wave 2 (3 parallel agents) — ALL DONE, logs in ./tmp/fixlog-wave2-{git,repl,home}.md:
   \_diff_name_only -z fix + merge_flow workaround removal + porcelain parser unify +
   commit-block format single-sourced; identifier-harvest//help/model-picker/
   line-continuation/completion-dispatch consolidations across cli_repl↔vscode;
   WakeSession rename, live DEFAULT_CONFIG export (PEP 562), canonical
   kiss.core.config.kiss_home() lazy resolver (5 sites), VOICE_MODEL_CACHE derived.
1. NEXT: full test suite in parallel splits + `uv run check --full`; then set_model
   gpt-5.6-sol and run thorough review/debug wave over `git diff` for missed
   wiring/bugs; fix review findings; clean ./tmp; finish.

FINAL STATUS — TASK COMPLETE:

- pyright break from PEP 562 lazy attrs fixed via TYPE_CHECKING declarations
  (vscode_config.CONFIG_DIR/CONFIG_PATH, web_server.\_URL_FILE).
- Fixed structural test test_review_round1_bugs::test_vs_bug4 to functionally
  test \_validated_cli_task_id; fixed order-dependent freeze of the lazy
  \_URL_FILE attr in test_web_server TestRemoveUrlFileOSError/ReadOnly (restore
  via `del`, not re-assignment of the computed value).
- gpt-5.6-sol review wave (3 parallel reviewers) found+fixed 6 real bugs:
  relentless_agent circular import + finish() dedup; gemini max_tokens=0
  precedence; shared \_tool_result_to_string helper (base/v1/v2 parity);
  surrogate-safe byte truncation of prompts; \_URL_FILE override not honored by
  \_url_file_path(); autocomplete `$` vs `\\Z` trailing-newline regression;
  web_use_tool PermissionError treated as unlocked; git_worktree timeout raised
  to 300s + process-group kill. Logs: reviewlog-{core,vscode,sorcar}.md (tmp,
  now cleaned). Removed dead updateSetting union member from types.ts.
- Full suite: 8 parallel splits, 6500+ tests — ALL PASS except 7 pre-existing
  test_install_script_new_session_immunity failures (worktree baseline
  install.sh lacks the kiss-new-session-reexec block; install.sh untouched by
  this task, user is editing it) and one Playwright flake that passes alone.
- `uv run check --full`: ALL checks pass (ruff, mypy, pyright, TS typecheck,
  TS lint, mdformat, API docs, compileall).

Known deferred/skipped (documented in fixlogs): v1/v2 model class merge (intentional),
parallel-executor consolidation sorcar_agent↔chat_sorcar_agent (risky, noted),
merge_flow.\_emit_pending_worktree (caller in server.py), FLAKY_MODELS kept (public API),
audio-MIME tables kept, claude_code stream timeout deferred, viz_trajectory dup out of scope.

## gpt-5.6-sol review — requested VS Code Python diffs

Reviewed the current diffs vs HEAD for autocomplete.py, commands.py, helpers.py,
server.py, task_runner.py, merge_flow.py, diff_merge.py, and json_printer.py. Two
concrete regressions remain:

1. `autocomplete.py`'s extracted `_TRAILING_IDENT_RE` uses `$`, which matches
   before one final newline; removing the old last-character guard makes a
   multiline input ending in `"GE\n"` complete `GE` on the wrong line.
1. `diff_merge.py` now decodes invalid-byte Git paths with surrogateescape and
   serializes the lone surrogate through JSON; Node replaces it with UTF-8
   `EF BF BD`, so the native MergeManager cannot open/reject the real path and
   can falsely announce “All changes rejected.”

Adversarial review rejected the suspected stale-step fallback and 300-second
Git timeout as non-production/theoretical. Focused regression tests: 39 passed.
