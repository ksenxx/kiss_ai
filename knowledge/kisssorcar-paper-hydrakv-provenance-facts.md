---
title: 'KISS Sorcar paper: HydraKV case-study provenance facts from sorcar.db'
uuid: c42fe444-a3d3-4f1a-9630-c2623a6d2319
summary: Verified facts about the HydraKV case study in papers/kisssorcar (first engine
  written by a KISS Sorcar task, not Claude Code; task list in mine_case_studies.py
  omits 4 HydraKV tasks) plus how to rerun ks_numbers.py without touching the repo.
created: '2026-09-26T19:57:23Z'
updated: '2026-09-26T19:57:23Z'
---
# HydraKV case-study provenance (verified 2026-09-26 against ~/.kiss/sorcar.db)

## First engine (1.85 Mops/s, dense 4-byte index over 128-byte slots)
- Written by KISS Sorcar task `3dbc2309065b49859167eda3a4be6f2c` (model claude-fable-5, chat b827a569..., 2026-07-18 14:42 UTC, $12.26, 45 steps): its `Write ./tmp/hydra.cc` tool call authors "HydraKV ... written from first principles"; result reports median 1.85 Mops/s.
- `papers/kvstorepaper/hydra_kv.tex` Task 1 paragraph says the same. Both `kiss_sorcar.tex` and `ks_assistant.tex` wrongly attribute this engine to Claude Code.

## HydraKV tasks in the DB vs. the paper's six
- `papers/kisssorcar/evidence/mine_case_studies.py` PROJECT_TASKS["hydrakv"] lists 6 tasks (593b, 307a, cdde, 061e, afc1, be3f). Omitted HydraKV tasks: 3dbc2309 (first engine), bac4e437 (10 Mops rerun, $60.22, 200 steps, interrupted by server restart), ca012ee29bb94603b99b9b61945b2c1f (find-all-bugs/100% coverage, 4.78 h, $151.76, 561 steps, 3 steers incl. "did you start with most performant variant of the engine?", 3 review sub-agents), c344f77a (reference-node scored rerun, $7.07, 62 steps). kvstore paper counts seven tasks (3dbc, 593b, ca01, 307a, cdde, 061e, afc1).
- Workload task 061e7725 has 1 steering message (FASTER tests); the 0:100 and 5:95 mixes were quoted in its initial prompt, not steers.
- Steering texts live in `events` table rows with `$.type='prompt'` (2nd+ prompt without a "Task Progress (Continuation N)" header).

## Running the paper scripts read-only
- `ks_numbers.py` and `ks_tb2.py` write into `papers/kisssorcar/tables/` and `figures/`. To recompute without touching the repo: copy `papers/kisssorcar` to `/tmp/x/papers/kisssorcar`, symlink `/tmp/x/{src,projects,benchmarkings}` to the repo, and run `cd <repo> && uv run --with matplotlib python /tmp/x/papers/kisssorcar/ks_numbers.py` (needs the repo's uv env for `kiss` imports).
- As of 2026-09-26 all generated macros matched kiss_sorcar.tex except LocPackage (tex 71,557 vs 71,556).
