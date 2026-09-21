# Where the wall-clock time goes (2,786 tasks, 2026-09-12 05:41 → 2026-09-19 05:41 UTC)

Source: `~/.kiss/sorcar.db` (`task_history`, `events`). Method: `speed_analysis.py` (see README.md).

## Top-level tasks (256) — what the user waits for: 110.2 h

| component | hours | share |
|---|---|---|
| the agent's own LLM round trips (12,632 steps, avg 12.1 s) | 42.4 | 38% |
| waiting on `run_parallel` (203 calls, avg 654 s) | 36.9 | 33% |
| own `Bash` (6,937 calls) — pytest 5.8 h, `uv run check` 3.6 h (144 runs), sleep/poll loops 3.5 h, npm/js 1.8 h | 17.9 | 16% |
| `run_agent` (54 calls) — 6.8 h are `tmp/review_agent.py` reviewers, 1.3 h channel agents | 8.1 | 7% |
| `ask_user_question` (14 calls) | 3.2 | 3% |
| everything else (browser, memory, Read/Edit/Write ≈ 0 s each) | 1.7 | 2% |

Median top-level wall: 264 s. Buckets: 75 tasks <1 min (85% LLM), 54 tasks 1–5 min (66% LLM), 75 tasks 5–30 min (56% LLM), 39 tasks 30 min–2 h (43% LLM, 55% tools), 11 tasks >2 h (26% LLM, 74% waiting on sub-agents/tests).

Sub-agents (2,530): 133.6 h aggregate; 80.2 h LLM, 50.6 h tools (97% Bash). Median sub-agent: 4 steps, 60 s.

## LLM round-trip anatomy (38,010 steps)

* p50 6.9 s, p90 23.9 s, p99 76 s, max 546 s. 79% of steps issue exactly one tool call.
* Output throughput ≈ 120–135 chars/s (thinking+text+tool-call JSON) for all models.
* Fixed latency of a trivial step (one Read/Bash/Edit, <200 chars output) grows with context:
  claude-fable-5 4.4 s (<25k) → 6.2 s (50–100k) → 8.0 s (100–200k) → 9.8 s (>300k);
  gpt-5.6-sol 4.2 s (<25k) → 9.6 s (50–100k) → 13.5 s (200–300k). 37% of steps run at >100k context.
* Pure-overhead steps (a whole round trip for no work): `summary` 2,287 steps / 7.0 h (always alone in its step);
  `Read ./SORCAR.md` as step 1 of 1,666 sub-agents / 2.0 h; solo `memory_search`/`memory_pull` 1,657 steps / 1.9 h;
  2,332 consecutive single read-only steps / 3.3 h that could have been one parallel-tools step; 86 `set_model` steps.
* `Write` steps: 1,236 / 12.9 h (avg 37.5 s) — whole files streamed through the model.

## Fan-out (`run_parallel`, 385 batches, 55.2 h of parent waiting)

* 128 batches contain ONE task (20.8 h) — 120 of them are reviewer delegations (serial review rounds).
* Stragglers: in batches of ≥3, max child − median child = 10.5 h of the 25.5 h waited (median max/median 1.65×, p90 5.6×).
* 32 parents ran ≥3 sequential rounds (154 rounds).
* Dispatch overhead is negligible (1.2 s avg); each child pays a ~2–3 s LLM task-classifier call before its first step (p90 6 s, max 19 s).
* Reviews overall (single-task fan-outs + `review_agent.py` via run_agent) ≈ 27.6 h = 25% of all top-level wall time; a review round averages 10–17 min, the slowest 46 min.

## Shell time

* 25,323 Bash calls / 49.4 h; 865 calls ≥60 s account for 35.5 h. pytest 14.6 h (2,368 runs), sleep/poll loops 6.0 h (294 commands, avg 74 s — `sleep 250; for … sleep 15` waiting on background `uv run check --full` / npm), `uv run check` 105 runs ≥60 s (4.0 h; one tree ran it 11×, another 5× at ~1,000 s each), npm/js 5.5 h.
* The identical "run all tests" prompt ran 11× — 0.2 h wall with 27 parallel splits (claude-opus-4-8) vs 7.9 h when the reviewer loop was allowed to run 12 rounds.

## Channel agents

`slack` "send a test message" = 19 steps / 131 s; `google_docs` fetch = 12 steps / 180 s (170 s of it LLM); Telegram auth 58 steps / 845 s.
