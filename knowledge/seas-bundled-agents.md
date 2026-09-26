---
title: Catalog of bundled SEAs (/sh, /merge, /skillopt, /write_paper, /review_paper,
  /task_update, /autoroute, /ask)
uuid: f6676a7d-e496-4a02-8a18-b6e9b8cede6f
summary: 'Catalog of bundled SEAs (/sh, /merge, /skillopt, /write_paper, /review_paper,
  /task_update, /autoroute, /ask): purpose, tools, key getters, other callers.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Catalog of bundled SEAs

All live in `src/kiss/agents/seas/` except `ask_sea.py` (`src/kiss/agents/third_party_agents/`).
Each is also a `/<name>` slash command.

| SEA | Purpose | Tools added | Key getters |
|---|---|---|---|
| `dummy_sea.py` | Empty file; default agent of `run_agent` (plain Sorcar session) | none | none |
| `sh_sea.py` (`/sh`) | Run the shell command in the prompt verbatim and return full stdout/stderr and exit code | none | `system_prompt` (replaces SYSTEM.md), `tool_profile="bash"`, worktree/commit/classify/parallel/web/memory all False |
| `merge_sea.py` (`/merge`) | Resolve git merge conflicts left by a squash merge | none | `system_prompt`, `max_budget=5.0` (`MAX_BUDGET_USD`), worktree/commit/parallel/web/memory False |
| `skillopt_sea.py` (`/skillopt`) | Optimize a skill's or SEA's prompt text against an eval set | `optimize`, `status` | `system_prompt`, `tool_profile="shell"`, worktree/commit/classify/parallel/web/memory False |
| `write_paper_sea.py` (`/write_paper`) | Write or revise a research paper under `templates/write_paper_prompt.md` rules | `check_paper` (AI-slop and consistency gates on .tex prose), `build_paper` (pdflatex/bibtex, log summary, page count) | `append_to_system_prompt` (keeps full Sorcar toolset), `use_web_tools=True`, `is_parallel=True`, `classify_tasks=False` |
| `review_paper_sea.py` (`/review_paper`) | Review a paper (PDF/.tex/.md/.txt) for a venue | `read_paper` (page-by-page text, margin line numbers stripped), `check_review` (structure, word limit default 700, slop gates) | same shape as write_paper |
| `task_update_sea.py` (`/task_update <task_id>`) | Short HTML progress report of another task from its transcript | `task_transcript` | `tool_profile="bash"`, `max_budget=1.0`, everything else off |
| `autoroute_sea.py` (`/autoroute`) | Finish a task at the lowest cost per accepted unit by routing units to the cheapest passing model tier; ledger `~/.kiss/MODEL_DECISIONS.md` | `model_menu`, `pick_model`, `estimate_cost`, `log_decision` | `system_prompt`, `is_parallel`/`classify_tasks`/web/memory False |
| `ask_sea.py` (`/ask`) | Answer questions about the running task from its persisted events, read-only | `task_overview`, `task_transcript`, `task_step` | `system_prompt` = SYSTEM_LITE-based, `append_to_system_prompt` playbook, `tool_profile="review"`, parallel/web/memory False |

## Other callers

- `merge_sea.py` is run in-process by `kiss.server.merge_conflict_resolver` when the post-task
  auto-merge of a worktree branch conflicts: it runs as a sub-agent of the failed task (cost is
  added to that task), only edits and stages the conflicted files, and the resolver verifies and
  commits.
- `task_update_sea.py` is run in-process by `kiss.server.task_update` for the task shown in the
  chat panel: on first show, every `UPDATE_INTERVAL_S` (600 s), and on refresh. Its output
  replaces the `tmp/PROGRESS.md` mirror in the task-info panel.
- `ask_sea.py` tools wrap `kiss.agents.sorcar.task_digest`. Its docstring records why: 155
  earlier `/ask` runs wasted their first steps on a missing `sqlite3` CLI and raw event JSON
  (median 9 steps, $0.91, 99 s), so the transcript is served pre-digested.
- `ask_sea` is listed in `_NON_CHANNEL_MODULES` so `run_agent` does not treat it as a channel.

## Sources
- `src/kiss/agents/seas/__init__.py`, and each SEA file named above (getters and module docstrings)
- `src/kiss/server/task_update.py` (`UPDATE_INTERVAL_S`)
- `src/kiss/agents/sorcar/agent_dispatch.py` (`_NON_CHANNEL_MODULES`)
