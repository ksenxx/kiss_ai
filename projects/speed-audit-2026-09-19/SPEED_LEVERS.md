# How to make Sorcar agents faster without losing quality — 7-day evidence (2026-09-12 09:56 → 2026-09-19 08:05 UTC)

Data: `~/.kiss/sorcar.db`, 2,755 tasks (252 top-level, 2,503 sub-agents), 37,911 LLM steps, 48,220 tool calls.
Method: `README.md` (derived DBs) → `speed_analysis.py` (step/tool segmentation) → `speed_report.py` (this report's numbers, `/tmp/speed7_report.txt`).
This supersedes `FINDINGS.md` (same method, window shifted 2.5 h; adds outcome/quality linkage and lever coverage vs. the WP0–WP7 cost work landed in 756ddf4c0).

## 1. Where the user's waiting time goes (top-level tasks, 111.4 h, median task 262 s)

| component | hours | share |
|---|---|---|
| agent's own LLM round trips (12,489 steps, avg 12.2 s) | 42.3 | 38% |
| waiting on `run_parallel` children (432 calls) | 37.7 | 34% |
| own `Bash` | 18.3 | 16% |
| `run_agent` (reviewer scripts 5.6 h, channel agents) | 8.1 | 7% |
| `ask_user_question` | 3.2 | 3% |
| everything else (browser, memory, Read/Edit/Write) | 1.8 | 2% |

Quality baseline: 242/252 top-level tasks finished `success: true`, 5 stopped by the user, 4 without result, 1 failed. Of 144 consecutive same-chat prompts only 4 (3%) read as a complaint/redo. The levers below are chosen so that this baseline is not touched: they remove *round trips that do no work*, *idle waiting*, and *non-converging loops*, not reasoning.

## 2. The anatomy of a step — why "fewer, fatter steps" is the main lever

* 52% of top-level steps (6,498) produce **zero thinking** and still take a median 6.0 s = 14.5 h of pure round-trip latency (prefill + tool-call JSON). Heavy-thinking steps (>2k chars) are 3% of steps and 16% of LLM time — that is where quality is made, leave it alone.
* Only **6% of top-level steps issue more than one tool call**, although the system prompt asks for independent calls in one block. 2,694 consecutive read-only steps (4.4 h) could have been single batched steps.
* Fixed latency of a trivial one-tool step grows with context (median, this week):

| model | <25k | 50–100k | 100–200k | 200–300k | >300k |
|---|---|---|---|---|---|
| claude-fable-5 | 4.3 s | 6.5 s | 8.0 s | 8.2 s | 8.3 s |
| gpt-5.6-sol | 3.6 s | 8.4 s | 9.2 s | 9.8 s | — |
| claude-fable-5-1 | 5.0 s | 7.0 s | 7.4 s | 8.7 s | 9.8 s |

  74% of top-level steps run above 50k context; the "context tax" above the <25k baseline is **8.8 h on the critical path** (21.7 h across all agents).
* Output throughput is ~85 chars/s regardless of model, so a `Write` of a whole file costs 37.5 s avg (1,236 Write steps, 12.9 h). `Edit` is 5–8 s.

## 3. Ranked levers (critical-path hours per week, quality risk, status)

| # | lever | evidence | est. saving / week | quality risk | status after WP0–WP7 |
|---|---|---|---|---|---|
| 1 | **Bound and inform the review loop.** Cap at 2 rounds by default; pass the *diff + previous verdict + what was fixed* in the reviewer prompt; run the review concurrently with the parent's next independent work item; triage findings by severity (`decide`) and fix only High/Medium before the next round. | 466 reviewer sub-agents = 66.7 h wall / 46 h LLM. 127 single-child `run_parallel` batches (22.3 h of parent waiting) are serial review rounds. In the 101 parents with ≥3 rounds, **0 of 135 rounds came back clean** — findings per round do not decline (76,34,38,33,25,15,27,28,17,20,23,24 in the 12-round race-fix task; 11,8,9,11,14,9,10,10,8,10,9,10 in the 12-round test task). Later-round findings are real but mostly *introduced by the previous fix* (fix churn). Round prompts are often context-free ("Review the staged/unstaged changes as instructed.") so each reviewer re-orients from scratch: 17–62 steps, 55% of reviewer `Read`s are files an earlier round already read; short-prompt reviewers 7 steps/46 s vs long-prompt 36 steps/615 s. | 10–15 h | Low: no round has ever converged, so a round cap changes *when* the remaining findings reach the user, not whether. Concurrency and inlined context change nothing about verdict quality. | Not done (review *budget* cap and `review` tool profile exist; no round cap, no diff inlining, no concurrency). |
| 2 | **Kill overhead-only round trips**: `summary` steps (always alone), solo `memory_search` before the first tool, `set_model` steps, step-1 `Read ./SORCAR.md`. Make `summary` a side-effect that must be called in the same block as the next real tool (or generated server-side), call `memory_search` in the same block as the first Read/Bash, pass `model_name` to `run_parallel` instead of `set_model` as step 1. | `summary`: 2,277 steps / 7.0 h (900 / 2.5 h on the top-level critical path), one per 16.6 steps. Solo memory steps 1,734 / 1.8 h. `Read ./SORCAR.md` as step 1 of 1,661 sub-agents / 2.0 h (~4.5 s added to every fan-out wave). `set_model` 85 steps. | 5–7 h | None — these steps carry no reasoning. | SORCAR.md mandate removed (WP2, not yet visible in DB — daemon not restarted before this window closed). `summary`, memory, `set_model` untouched. |
| 3 | **Batch independent tool calls** — enforce "one round trip per dependency level": Read several files at once, `grep -n` + Read line ranges instead of whole files, `run_commands_parallel` for lint + tests. | 6% multi-tool steps; 2,694 consecutive read-only steps / 4.4 h; 29% of all Reads (3,303) re-read a path already read in the same task; `main.js` (18,857 lines) read 916×. | 3–5 h | None. | Read dedupe/outline (WP3) cuts the *tokens* of repeat reads, not the round trip; prompt guidance for batching still needed. |
| 4 | **Keep context under 100k on long tasks** (compaction, hand-off, line-range reads, trimmed Bash output). | Context tax 8.8 h top-level; 37 tasks restarted at the context limit (53 h of wall); a restart costs a median 6 re-orientation steps (p90 34; 2.3 h total). | 4–6 h | Low if hand-off notes are complete (`tmp/PROGRESS.md`); compaction keeps last 20 results verbatim. | Done in WP: compaction at 100k then +50k, `KISS_CONTEXT_LIMIT_FRACTION=0.7`. Verify: the 0.7 hand-off adds restarts (each ≈ 6 steps ≈ 1 min) — net positive only if steps at 350–450k were >6 per task (they were: 618 steps / 2.6 h). |
| 5 | **Straggler-aware fan-out**: more shards than workers, size shards by measured test duration, give each child a hard step/time budget, and let the parent continue when N−1 children are back. | In batches of ≥3, max − median child wall = **9.3 h** of the 25.0 h waited (median max/median 1.65×, p90 5.6×). In 278 multi-child batches the parent waited 35.3 h while the slowest child's *tool* time was only 15.8 h: 19.5 h of the critical path is children's LLM steps. | 4–6 h | None for test shards; for reviewer/implementer children the budget must be generous (they do real work). | Not done. |
| 6 | **Zero-LLM test shards**: route "run this pytest/npm command and report" children to `run_commands_parallel` (exists since 515b81759) instead of LLM sub-agents. | 1,613 test-shard sub-agents, median 4 steps (`Read SORCAR.md → memory_search → Bash → finish`; ideal is 0); 389 shards took >4 steps (4.9 h LLM); 1,077 pure shell-wrapper sub-agents = 25.3 h wall / 5.0 h LLM. On the critical path this is only 0.9 h/week (the pytest run dominates), but it also removes the classifier start-up (2.5 s avg, p90 6.1 s per child) and the shard-that-thinks straggler. | 1–2 h + fewer stragglers | None. | Tool exists; usage this week 0.1 h. Needs a system-prompt rule: "shell-only child → run_commands_parallel". |
| 7 | **Shell hygiene**: never `sleep`-poll a background job (use `timeout N cmd` or `wait`); run `uv run check --full` once at the end; run the affected tests while iterating and the full suite once. | sleep/poll loops 242 calls / 5.2 h (single `nohup … & sleep` calls that blocked 600–1,150 s); 19 task trees ran `uv run check` ≥3× (2.6 h; one tree 14×); 121 identical commands re-executed in the same task; pytest 1,967 calls / 11.2 h total. | 3–4 h | None (the final gate still runs). | Not done. |
| 8 | **Prefer `Edit` to `Write` for existing files; keep progress notes short.** | 1,236 Write steps / 12.9 h at 85 chars/s; `PROGRESS.md` was written 81× and read 340×. | 1–2 h | None. | Not done. |

Not recommended: cutting thinking budgets or moving reasoning steps to cheaper models — heavy-thinking steps are only 16% of LLM time and are where the 96% success rate comes from. Model choice barely matters for latency (median step 5.8–7.8 s for all four main models; opus-4-8 is fastest per step but was used for 2% of steps).

## 4. What one week of the biggest wins looks like

The same "run all tests (python and javascript)" prompt ran 10 times: 0.23 h with 36 zero-review shards (claude-opus-4-8) vs 7.89 h with 12 sequential review rounds (claude-fable-5); every run reported success. The spread is entirely process (review rounds, sequential steps), not model or test time (pytest wall ≈ 0.2 h in all of them). Applying levers 1–3 alone (review cap + informed reviewer, no overhead-only steps, batched reads) removes ~20–25 of the 111 h/week on the user's critical path with no change to what the agents reason about.

## 5. Caveats

* `end_ts` is 0 for sub-agents; walls use first/last event timestamps.
* `is_error` is set on 53 tool calls only; Bash failures live in `system_output` (168 of 46,969 with Traceback/non-zero exit) — tool errors are *not* a material time sink this week (45 `run_parallel` refusals by the fan-out guard were the largest group).
* WP0–WP7 (commit 756ddf4c0) landed at the very end of this window; none of its effects are visible in these numbers yet. Re-run `speed_report.py` after a week of daemon uptime to measure them.
