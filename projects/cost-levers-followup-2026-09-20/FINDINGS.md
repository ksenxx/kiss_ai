# Cost-lever follow-up (2026-09-20): 72 h KPI re-run and the compaction question

Data: `~/.kiss/sorcar.db` snapshot (`VACUUM INTO /tmp/cost72.db`, 2026-09-20 06:08 UTC).
Lever commit `756ddf4c0` landed 2026-09-19 07:56:27 UTC (epoch 1789804587).
Baseline: `tmp/baseline_24h.json` (09-18 05:00 → 09-19 05:00 UTC, pre-lever).

Scripts (all in this directory, run with `.venv/bin/python`):

| script | purpose |
|---|---|
| `compare_kpis.py --hours 72 [--snapshot] --out tmp/compare_72h_<date>` | `cost_report` KPIs vs baseline, with the two filters (strip `/tmp/kiss_ablation` orphans, post-lever trees only = root started after the lever commit **and** steps carry the `cache_read` field that only post-lever code emits), plus a per-model **$/step split by step type** |
| `compaction_analysis.py --db … --since 1789804587` | rebuilds every session from `events`: per-step context, observed compaction drops, what was appended (tool results incl. streamed Bash output, tool-call args, thinking), share of steps ≥100k by session length, per-model $/step |
| `compaction_simulation.py --db … --since 1789804587` | replays the recorded sessions through the real `kiss.core.context_compaction.compact_tool_results` under alternative parameter sets and reports context-token totals / ≥100k share / peaks |

## 1. 72 h comparison — run now (preview) and scheduled

The task asked for the re-run "in a few days"; a one-shot cron job (`cost-levers-72h-recheck`, id `1908db14`) runs
`compare_kpis.py --hours 72 --snapshot` on 2026-09-23 06:30 UTC and posts the table.  The
preview below (2026-09-20 06:24 UTC) is what the filtered view already shows; **the raw 72 h view
is useless** — it still contains 495 ablation orphans, the pre-lever 09-19 paper tree and the 09-17/18
pre-lever traffic:

| KPI | baseline (pre-lever 24 h) | live raw 72 h | live filtered (post-lever trees, no orphans) |
|---|---|---|---|
| tasks / top-level / sub-agents | 562 / 54 / 508 | 1941 / 130 / 1811 | 52 / 42 / 10 |
| cost_usd | 1,486 | 4,045 | 871 |
| steps | 17,176 | 28,198 | 4,190 |
| sorcar_md_reads | 306 | 883 | **0** |
| shell_wrapper_subagents / $ | 258 / 74.9 | 692 / 182.7 | **0 / 0** |
| reviewer_trees / over_half | 22 / 7 | 298 / 274 (orphans) | **0 / 0** |
| cache_hit_ratio | n/a | 0.970 | 0.970 |
| repeat_read_ratio | 0.375 | 0.307 | 0.288 |
| steps ≥100k (share) | 2,441 (31.4 %) | 9,414 (41.3 %) | 2,251 (59.4 %) |
| steps_over_200k_ratio | 0.088 | 0.119 | 0.134 |
| context_handoffs / session_restarts | 2 / 14 | 12 / 46 | 2 / 15 |
| cost per step, all models | 0.087 | 0.143 | 0.208 |

Per-model, filtered, **claude-fable-5-1** (82 % of spend): 3,484 steps, own $/step **0.224**
(baseline 0.220) — flat, not lower.  The split by step type explains why:

| step type | steps | cost | share of model spend | $/step | avg ctx |
|---|---|---|---|---|---|
| normal cached step | 3,331 | $597.9 | 52 % | **0.180** | 129k |
| fan-out step (children's cost folded into parent) | 13 | $368.9 | 32 % | 28.4 | 222k |
| cache miss: ≥5 min gap since previous step (TTL expiry) | 37 | $82.8 | 7 % | **2.24** | 178k |
| cache miss: compaction step | 49 | $82.6 | 7 % | **1.69** | 133k |
| first step of a session | 53 | $14.0 | 1 % | 0.26 | 26k |

* A normal step costs $0.18 at 129k context; 86 cache-miss steps (2.5 % of steps) cost $165
  (21 % of the model's own spend).  Excluding them, $/step is 0.18 vs 0.22 baseline (−18 %).
* Fitting per-step cost on `cache_read` / non-cached tokens gives effective prices of
  $1.06/M (cache read) and $14–17/M (non-cached incl. output), R² 0.88: **73–76 % of a normal
  step's cost is re-reading the cached context** → $/step ≈ $1.06/M × context + ~$0.04.
  Per-step cost can therefore only fall if the *average context of a step* falls or the
  step count per task falls.
* Baseline $/step is also flattered by 3,090 cheap gpt-5.6-sol sub-agent steps that the live
  window lacks; compare same-model only.

## 2. Why compaction has not reduced the share of steps ≥100k

Post-lever window (123 tasks, 138 sessions, 5,585 steps; 3,090 = 55 % of steps ≥100k, carrying
78 % of context tokens).

**(a) It fires at 100k by design, then the session lives above 100k anyway.**
Sessions cross 100k at step ~20 (median).  The first compaction (57 observed) removes a median
**34k** tokens (100k → ~63k); the context re-crosses 100k a median **18.5 steps** later, and the
next trigger is `context_after + 50k` ≈ 113–115k, so from then on the session sits at ≥100k
except for brief dips.  Nothing compacts *below* 100k, so a session's ≥100k share is set by its
length:

| session length | sessions | steps | ≥100k share | peak ctx avg | cost |
|---|---|---|---|---|---|
| <20 steps | 80 | 635 | 5 % | 49k | $95 |
| 20–50 | 27 | 838 | 26 % | 110k | $103 |
| 50–100 | 15 | 1,087 | 63 % | 206k | $244 |
| 100–200 | 8 | 1,138 | 70 % | 218k | $272 |
| ≥200 | **8** | 1,887 | 72 % | 248k | $456 |

The 16 sessions ≥100 steps are 54 % of all steps, 70 % of ≥100k steps and $728.  The baseline day
had **one** session ≥200 steps; the post-lever window has eight (Windows test triage
`9d9c7c46`/`41784cfc`/`e420aed3`, paper `74cb2a67`, cost-check `69bd6c06`).  Compaction + the
0.7 hand-off also let a session run longer before handing off, which *adds* ≥100k steps.

**(b) Against the counterfactual, compaction did work — on tokens, not on the ≥100k share.**
Replaying the 48 Claude sessions ≥20 steps through the production compactor:

| policy | Σ context tokens | vs production | ≥100k share | peak ctx avg |
|---|---|---|---|---|
| A no compaction | 716M | +17 % | 0.640 | 184k |
| **B production (100k/50k, keep 20, min 2000)** | **614M** (recorded: 614M) | 1.00 | **0.573** | 160k |
| C keep 5 | 602M | −2 % | 0.555 | 157k |
| D keep 5, min 500 | 587M | −4 % | 0.548 | 154k |
| E start 50k/step 25k, keep 5, min 500 | 567M | −8 % | 0.531 | 150k |
| F D + stub old tool-call args (Bash heredocs, Write content) | 556M | −9 % | 0.510 | 143k |
| G E + F | 531M | −13.5 % | 0.490 | 138k |

Production compaction saves ~14 % of billed context tokens and 7 points of ≥100k share relative
to no compaction; the tighter settings buy only another 4–13 % *before* cache penalties (below).
"Share of steps ≥100k" is simply the wrong KPI for compaction — track Σ context tokens (or
cache-read $) per task instead.

**(c) What is left in a 100k+ context is mostly not old tool output.**
Chars appended to conversations in the window: tool results 13.5M (Bash streamed output
11.3M — capped at the call's `max_output_chars`; `Read` 1.15M, but the DB truncates non-Bash
results at ~3,000 chars so this is a lower bound), **tool-call arguments 7.9M** (Bash commands
with heredocs 4.07M, `Write` content 2.15M, `summary` 0.6M, `Edit` 0.4M), prompts 3.0M
(chat-history prefix, 21k chars/session avg), thinking summaries 2.2M.  Compaction touches only
tool results: 14 % of tool-result bytes are below `MIN_COMPACT_CHARS`, the 20 most recent results
hold 22 % of tool-result bytes of sessions that reach 100k, and everything on the assistant side
(args, text, thinking blocks, which `interleaved-thinking` keeps in context across tool calls) is
never compacted.

**(d) Every compaction is a full prompt-cache miss.**
Every detected compaction step (57 of 57) has `cache_read == 0` (the stub rewrites an early message, so the
cached prefix ends there) and cost $0.8–3.7 instead of $0.15–0.35 — the whole suffix is re-written
at $12.5/M.  Economics per compaction: saving = drop × remaining steps × $1.06/M, cost ≈
context × $12.5/M, break-even at ≈ 12 × context/drop remaining steps (≈20 steps for the first
compaction at 100k, 110–280 steps for a 12–25k drop at 230–300k):

| context before | n | median drop | downstream saving | cache-miss penalty |
|---|---|---|---|---|
| 95–130k | 18 | 35k | $71.5 | $13.5 |
| 130–180k | 17 | 31k | $66.2 | $22.3 |
| 180–230k | 12 | 33k | $30.9 | $22.0 |
| 230–400k | 10 | 36k | $37.9 | $18.4 |
| **total** | 57 | 34k | **$206.5** | **$76.3** → net ≈ +$130 (~10 % of the model's own spend) |

18 of 57 compactions were net-negative ($34.9 penalty vs $12.7 saving).  A second, unrelated
cache-miss source is as large: **37 steps whose previous step ended ≥5 min earlier** (median gap
10 min — long test runs, LaTeX builds, waiting on sub-agents) lose the 5-minute ephemeral cache
and cost $2.24 each ($82.8 total).

## 3. Recommendations (not implemented here)

1. **Make compaction cache-aware.**  Skip a compaction when `drop < 0.25 × context` or when the
   session is near its hand-off (saving ≈ 0); prefer bigger, rarer compactions
   (`keep_recent` 5–8, `min_chars` 500, step 100k instead of 50k), and piggy-back on moments
   when the cache is cold anyway (session start, after a >5 min tool call, at hand-off).
2. **Kill the TTL misses**: for tool calls expected to exceed 4 min (Bash `timeout_seconds` ≥ 300,
   `run_parallel`/`run_agent`), send a minimal keep-alive request every ~4 min (≈ context × $1.06/M
   ≈ $0.15–0.35 vs $2.2 per miss), or use the 1 h cache tier for that request ($20/M writes).
3. **Compact the assistant side too** (old `tool_use.input` of Bash heredocs / `Write` content —
   Anthropic does not sign tool_use blocks): −9 % context tokens in replay.  Needs an
   experiment for model confusion.
4. **Fix the KPI**: report Σ context tokens / cache-read $ per task and `cache_miss_steps` /
   `cache_miss_cost` (both now computed by `compare_kpis.py`) instead of "share of steps ≥100k".
5. Data hygiene found on the way: `events.tool_result.content` is truncated at ~3,000 chars for
   non-Bash tools (Bash output is streamed as `system_output`); hand-off summarizer `usage_info`
   events are persisted out of order with a lower cumulative cost (naïve deltas show a fake
   $115 "first step"); post-lever code is identifiable by the `cache_read` field.
