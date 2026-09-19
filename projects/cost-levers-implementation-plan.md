# Single-task implementation plan for the Sorcar token-cost levers

Status: proposal only, no code changed. Written 2026-09-19 from the 7-day efficiency
audit of `~/.kiss/sorcar.db` (2026-09-12 → 2026-09-19: 2,828 tasks, $8,412, 6.14 B tokens).
Edit this file freely; the condensed version lives in memory page
`sorcar-cost-levers-implementation-plan-2026-09-19`.

## 0. Grounding facts (verified in the repo)

| Already exists | Gap the task must close |
|---|---|
| `~/.kiss/SORCAR.md` is inlined into the system prompt (`RelentlessAgent.perform_task`, `src/kiss/agents/relentless_agent.py` ≈L1142) | The project `./SORCAR.md` (83 bytes) is not; `src/kiss/SYSTEM.md:105` still mandates `Read("./SORCAR.md")` as the first tool call → 1,967 wasted first steps/week |
| `src/kiss/SYSTEM_LITE.md` (4.1 KB vs 20.7 KB for `SYSTEM.md`) is selected by `SorcarAgent.run` when the Jev classifier says `is_simple` | Sub-agents always receive the full ~30-tool schema set; there are no tool profiles |
| Prompt caching is applied (`anthropic_model.py:795`, `openai_compatible_model.py:573`); usage tuples carry `cache_read`/`cache_write` (`kiss_agent.py` L1071+) | Never verified end-to-end that the static prefix hits the cache on every step |
| Cron `command` jobs (no LLM) exist in `cron_agent.py` | No one-shot / self-disable flag; polls are still written as LLM jobs |
| `fanout_guard.py`: 3 review rounds, reviewer marker `_subagent_info["reviewer"]`, strict JSON `tasks`; `run_commands_parallel` tool | No mechanical per-model budget cap; `run_parallel(tasks, max_workers)` (`sorcar_agent.py:1720`) has no `model_name`, so reviewers are started via a paid `set_model` step |
| `UsefulTools.Read(path, max_lines=2000, start_line)` (`useful_tools.py:811`) | No repeat-read dedupe, no outline mode for huge files; `KISSAgent.CONTEXT_LIMIT_FRACTION = 0.9` (`kiss_agent.py:62`); no compaction of old tool output in `KISSAgent.messages` (tool results appended in `_execute_step` ≈L943/998) |
| `ChatSorcarAgent.build_chat_prompt` (`chat_sorcar_agent.py:200`) caps history at `MAX_TASKS = 10` | Sends the full HTML result of every prior task on every step |
| Tests isolate `KISS_HOME` (`src/kiss/tests/conftest.py:79`) | 27 test tasks (`no-such-model-*`, `Qwen/QwQ-32B`, `/tmp/…` work dirs) still reached the production DB on 09-19 01:35 UTC → some path (daemon / `run_agent`) bypasses isolation; sub-agent `task_history.end_ts` stays 0 (`persistence.py` ≈L1013/2095) |

## 1. Shape of the single task

- One task, eight work packages (WP0–WP7), landed in payoff order so that if the task is
  stopped or hands off, the most valuable pieces are already in.
- Each WP ends with: end-to-end tests (repo convention: no mocks), impacted tests run via
  `run_commands_parallel`, `uv run check --full`, an entry in `tmp/PROGRESS.md`, `git add`.
- Every lever behind a config toggle (`DEFAULT_CONFIG.*` / env `KISS_*`, default on) so any
  regression is a flag flip, not a revert.
- Models: `claude-fable-5` implements. `gpt-5.6-sol` reviews read-only via `run_parallel` at
  exactly two checkpoints (after WP2, and at the end), dispatched with the model name directly
  (not `set_model`), told to "verify the listed changes; do not invent problems". Reviewer spend
  ≤ 50 % of the task budget (enforced mechanically once WP3 lands; by prompt before that).
  Reviewers cannot spawn reviewers (already enforced).
- The task practises the levers on itself: grep / line-range Reads only for `sorcar_agent.py`,
  `kiss_agent.py`, `useful_tools.py`; no LLM wrapper sub-agents; progress in `tmp/PROGRESS.md`
  so a context hand-off loses nothing.

## 2. Work packages

### WP0 — Baseline metrics + flags (do first, small)

- Script `src/kiss/scripts/cost_report.py` that computes the audit KPIs from `sorcar.db` for a
  time window: `SORCAR.md` Reads, repeat-Read ratio, cost by context bucket, sub-agent step-1
  context, reviewer share per task tree, LLM shell-wrapper sub-agents, context hand-offs,
  cache-hit ratio.
- Run it once now as the baseline; it is the acceptance test for everything below.
- Add the config toggles for WP1–WP7.

### WP1 — Fixed per-step overhead (≈ $300–500/week)

- **1a Inline `./SORCAR.md`.** Append `{work_dir}/SORCAR.md` next to the existing
  `~/.kiss/SORCAR.md` append in `perform_task`. Rewrite `SYSTEM.md:105` / `SYSTEM_LITE.md` to
  "its contents appear in the section below; do not Read it". Update docstrings in
  `daemon_client.py:440/619`, `server/README.md:376`, and the tests that assume the first Read
  (`test_system_prompt_internet_search`, stream-stall tests). Worktrees are fine: the file is in
  the git tree.
- **1b Tool profiles** in `SorcarAgent._get_tools(profile)`:
  - `full` — today's set (default).
  - `review` — Bash, Read, memory-read, decide, summary, finish. No Edit/Write/browser/talk/
    cron/run_agent/run_parallel, which also makes "read-only review" mechanical.
  - `shell` — Bash, Read, run_commands_parallel, finish.
  - `_run_single` / `agent_dispatch` pick `review` for reviewer-marked children; the parent may
    pass `tool_profile` explicitly.
- **1c Shorter tool docstrings** (schemas are generated from them). Target: sub-agent step-1
  context ≤ 7k tokens (now ≈ 12k). Measure with WP0.
- **1d Cache verification.** Assert nothing per-step-dynamic sits in the system prompt or tool
  list; add a test that step ≥ 2 of a real run reports `cache_read > 0`.

### WP2 — Context hygiene (≈ $500–800/week)

- **2a Read dedupe** in `UsefulTools.Read`: per-task map path → (mtime, size, sha, step, line
  range). Same unchanged range ⇒ return
  `"Unchanged since step N (lines a–b); pass force=True to re-read"` — only while that earlier
  content is still in the model's context (coordinated with 2c).
- **2b Outline mode for big files.** A `Read` of a file > 2,000 lines with no range returns the
  line count plus a symbol outline (`def`/`class`/`function` lines with numbers) and asks for
  ranges or grep. Any range is one call away; `media/main.js` alone drops from ≈ 10 M read-tokens
  to < 1 M per week.
- **2c Batched tool-output compaction** in `KISSAgent` before the model call: when context
  crosses 100k (then every +50k), replace `tool_result` contents older than 20 steps and larger
  than ≈ 2k chars with a stub ("output of step N compacted: first 200 chars…; re-run to see").
  Batched, not per step, so the cached prefix is invalidated at most a handful of times per
  task. Never touch the last 20 steps, `finish`, or Edit/Write results. Persisted events and
  trajectory / partial-result HTML keep the full text (already written to the DB per step).
- **2d Hand-off threshold.** `CONTEXT_LIMIT_FRACTION` 0.9 → 0.7 (configurable) so hand-offs
  happen where steps cost 2×, not 4×.

### WP3 — Mechanical per-model review budget cap + direct model dispatch

- Extend `ReviewQuota` with a shared `review_budget_fraction` (default 0.5 of the top-level
  budget). Reviewer-marked sub-tree spend (and any spend after `set_model` to another model) is
  accounted against it; a review fan-out is refused, or the child's `max_budget` clipped, when
  the allowance is exhausted.
- Add `model_name` to `run_parallel` (per fan-out) so reviewers start on `gpt-5.6-sol` without
  a paid `set_model` step.
- Record the per-step model in usage so `task_history` stops hiding reviewer spend
  (177 `set_model` calls last week).

### WP4 — Route trivial work cheaply (≈ $100–200/week; quality-safe scope only)

- **Cron:** a `one_shot` / `disable_after_delivery` flag; the cron agent converts
  "is X released?"-style polls into `command` jobs.
- **Tiering:** add a "tier" question to the existing Jev classifier and use a cheap model only
  for machine-generated LLM work (chat-history digests, cron LLM jobs, commit messages). Never
  override the model the user picked for their own task.

### WP5 — Prompt shape

- `build_chat_prompt`: full result text for the last 2 tasks, an `<h3>` / first-N-chars digest
  for older ones, total prefix ≤ ≈ 6k chars.
- Canonical reviewer prompt template in `SYSTEM.md` / `fanout_guard`: "verify the listed fixes;
  report only demonstrated issues; do not seek novel regressions".
- Memory hygiene: `SYSTEM.md` says per-round notes go to `tmp/PROGRESS.md`; `memory_write`
  warns on names matching `round\d+`.

### WP6 — Data quality

- Set `end_ts` for sub-agents at their final save (`persistence.py` ≈L1013/2095 path).
- Find the test path that wrote `no-such-model-*` tasks into the production DB despite
  `conftest.py`'s `KISS_HOME` isolation (most likely a live-daemon `run_agent` round trip) and
  make it use an isolated daemon.
- Delete the 27 artifact rows.

### WP7 — Dispatch hygiene (134 + 16 failures/week)

- In `_run_single` / `_dispatch`, rewrite parent-repo absolute paths to the worktree path in
  sub-task text when a worktree is active; have the Bash guard's error suggest the rewritten
  command.
- Fuzzy-reject generic `run_agent` names (`general`, `code-review`, `agent`, …) with the nearest
  valid name.

## 3. Acceptance criteria (measured by the WP0 script over the following 24 h / 7 d)

- `Read ./SORCAR.md` calls: 1,967/week → 0; sub-agent step-1 context: ≈ 12k → ≤ 7k tokens.
- Repeat-Read ratio: 29 % → < 5 %; steps above 200k context: 12 % of steps → < 3 %;
  context hand-offs: 7/week → 0–1.
- Reviewer share ≤ 50 % in every task tree; zero reviewer trees deeper than one level; zero LLM
  shell-wrapper sub-agents.
- Cache-read tokens present on ≥ 90 % of steps ≥ 2 on Anthropic / OpenRouter-Anthropic models.
- No quality regression: every existing e2e suite green; the levers never remove information the
  model cannot re-fetch in one call (dedupe stub, compaction stub and outline mode all say how).

## 4. Ready-to-paste task prompt

```
Implement the token-cost levers from projects/cost-levers-implementation-plan.md as work
packages WP0–WP7, in that order, each behind a config toggle (default on) with end-to-end
tests (no mocks), impacted tests run via run_commands_parallel, `uv run check --full`,
and a tmp/PROGRESS.md entry after each WP. Use grep/line-range Reads for files over
2,000 lines and never spawn an LLM sub-agent just to run a shell command.

Use 'claude-fable-5' for all implementation. After WP2 and again at the end, use
'gpt-5.6-sol' (not codex) via run_parallel for a thorough read-only review of the
listed changes only; ask it not to invent new problems and to verify wiring, missed
call sites and bugs. Keep gpt-5.6-sol under 50% of the task budget. Use the model
names literally. Finish by running the WP0 metrics script on the last 24 h and
reporting baseline vs. current KPIs.
```

## 5. Risks and containment

| Risk | Containment |
|---|---|
| Compaction invalidates the prompt cache | Batch at thresholds (100k, +50k, …) instead of per step; measured by WP1d |
| Dedupe / outline hides text the model needs | Stubs always state the one-call way to get it back; the last 20 steps are never touched |
| Cheap-model routing hurts quality | Restricted to machine-generated work and crons; the user's chosen model is never overridden |
| Tests assuming the first `Read("./SORCAR.md")` | Enumerated in WP1a and updated in the same WP |
| Task size (≈ 10 files, ≈ 1.5k LOC + tests, est. $60–120 and 3–5 h with the new guardrails) | Payoff-ordered WPs, progress file, flags: a hand-off or stop still leaves usable, tested increments |

## 6. Expected payoff

WP1–WP5 together address roughly 35–45 % of last week's $8.4k without removing any step that
produced evidence, a fix, or a verified test result. WP0/WP6 make the saving measurable; WP7
removes ≈ 150 avoidable failures per week.
