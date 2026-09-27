---
title: Verified src/kiss code facts cited by papers/kisssorcar/kiss_sorcar.tex
uuid: 85c5f752-a58d-4ef3-8f3c-2b1e5095932e
summary: Code facts (with file paths) that kiss_sorcar.tex states about KISS Sorcar,
  verified 2026-09-26, plus how the paper's macros and HydraKV task list are recomputed.
created: '2026-09-26T20:45:27Z'
updated: '2026-09-26T20:45:27Z'
---
# Verified code facts behind kiss_sorcar.tex (2026-09-26)

- `src/kiss/core/kiss_agent.py`: `MAX_CONSECUTIVE_ERRORS = 3`; non-retryable errors (auth, gated/unknown model, credit) try `_try_switch_to_fallback` once (catalog fallback, OpenRouter twin) before raising; `CONTEXT_LIMIT_FRACTION = 0.7` hand-off raising `ContextWindowExceededError`; `_maybe_compact_conversation` stubs old large tool outputs from 100,000 tokens (`tool_output_compaction` on by default), never summarizes in place.
- `src/kiss/agents/sorcar/relentless_agent.py`: `MAX_ZERO_PROGRESS_SESSIONS = 2` (session that called no tool but finish, or repeated the previous summary); `max_sub_sessions` default 10,000.
- Default coding tools: Bash, bash_job, run_commands_parallel, Read, Edit (old string must occur once unless `replace_all`), Write (`useful_tools.py`). Since commit 2d6a12494 (2026-09-19) Edit and Write reject a file not in `read_files`; only Write exempts scratch paths under `tmp/` (`_is_scratch_path`). The paper's ablation results were committed 2026-09-18, before this.
- `sorcar_agent.py` `TOOL_PROFILES`: full (has set_model), assistant (has set_model), review/shell/bash omit it; `_tool_profile` gives review-like sub-agent tasks the `review` profile when `DEFAULT_CONFIG.tool_profiles` (default True, env `KISS_TOOL_PROFILES`).
- `chat_sorcar_agent.py`: `MAX_TASKS=10`, `DIGEST_FULL_RESULTS=2`, `DIGEST_MAX_PREFIX_CHARS=6000`, `DIGEST_TASK_CHARS=600`, `DIGEST_RESULT_CHARS=300` (digest appends " ..." so rendered length is limit+2); ops `new_chat()`, `resume_chat_by_id`, `resume_from_task_id`; config `chat_history_digest` / `KISS_CHAT_HISTORY_DIGEST`.
- `server/json_printer.py`: per-task state keyed by task_id; only block type / task binding thread-local.
- `task_classifier._cache_key` includes criteria + task text + model + config; memo TTL 7 days.
- Model catalog `src/kiss/core/models/MODEL_INFO.json`: 689 entries, 671 gen, 505 fc, 7 emb (fields `gen`,`fc`,`emb`).
- Channels: 45 `*_sea.py` in `agents/third_party_agents`, 42 dispatchable (minus a2a, ask, oai per `agent_dispatch._NON_CHANNEL_MODULES`); README splits 32 messaging + 10 service; "24 Muse-supported services" is a README figure (20 `*_sea.py` import muse_auth plus Google Workspace helpers).
- Welcome screen: commit 7173e60fe (2026-09-26) removed sample-task chips and MY_TASK_TEMPLATES seeding; `src/kiss/SAMPLE_TASKS.md` still has 12 prompts, `INJECTIONS.md` 8 promptlets; user files `~/.kiss/MY_INJECTION.md`, `MY_MODELS.json`.
- `SAMPLE_TASKS.md` GEPA prompt has no period after `END_RUN_GEPA`; `gepa.py` lives in `src/kiss/agents/obsolete/gepa/`.

## Paper plumbing
- `papers/kisssorcar/ks_numbers.py` now prints all Loc*, Clf* (from `benchmarkings/task_classifier/results.json`), BoSpeedupOne, and per-task HydraKV macros looked up by task id; paste its block over the macro region of kiss_sorcar.tex.
- `evidence/mine_case_studies.py` PROJECT_TASKS["hydrakv"] has 8 tasks in order: 3dbc2309 (first engine), 593b7396 (adversarial discovery), ca012ee2 (bug hunt), 307a18e8, cdde00e9, 061e7725 (workloads; 0:100 and 5:95 quoted in its prompt, one steer about FASTER), afc1c52d, be3f36f6. Opens sorcar.db read-only.
- `\DbAdvTestingTasks`: top-level `task_history` rows with task LIKE '%adversarial testing%' (13 on 2026-09-26) minus the 3 that edit the paper = ten.
- Main-text budget: References heading must stay on page 27 of the PDF (as at commit 98a5a859f).
