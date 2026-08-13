# Fast-path correctness audit tests

Adversarial unit tests that audit every precomputed "fast path" in
`engine/` for correctness on **arbitrary placeholder values** (not just the
benchmarked seeds), and confirm that every fallback-to-baseline trigger
actually engages when it should.

## How it works

1. `engine/audit.hpp` defines `AUDIT_PATH(marker)`. Every fast-path guard and
   fallback branch in `engine/query_q*.cpp` reports an `AUDIT <marker>` line
   on stderr — **only** when compiled with `-DBESPOKE_AUDIT`. Benchmark builds
   (`bench/build.sh`) never define it, so the shipped binary is unaffected
   (verified: `git diff` is purely additive and SF1 bench totals/validation
   are unchanged).
1. `build_audit.sh [<out_dir>]` builds `db_audit` (optimized engine +
   `-DBESPOKE_AUDIT`) and `db_baseline` (pristine `engine_baseline/`, the
   paper artifact used as the correctness oracle).
1. `difftest.py <manifest.jsonl>` feeds the identical request batch to both
   binaries, diffs each `result<i>.csv` pair with the paper's validation
   policy (`compare_frames` from `bench/run_bench.py`: all-column-sorted,
   atol=rtol=1e-2), and asserts each request's `expect`/`forbid` path markers.

## Manifest format (JSONL)

```json
{"line": "3 \"BUILDING\" \"1995-03-08\"", "expect": ["q3 fast"], "forbid": [], "note": "seed42"}
```

- `line` — raw request exactly as in `bench/workloads/seed*/args.txt`
- `expect` — markers that must appear among the request's AUDIT markers
- `forbid` — markers that must NOT appear
- Lines starting with `#` are comments. Matching is EXACT by default; a
  trailing `*` opts into prefix matching for a whole marker family (e.g.
  `"q9 fallback*"`). Exact matching prevents an assertion like `q13 fast`
  from silently passing on a sibling branch's marker (`q13 fast masked`).

## Running

```bash
cd projects/bespoke_tpch_x4
./tests/build_audit.sh                     # -> tests/bin/{db_audit,db_baseline}
uv run --no-project --with duckdb --with pandas python3 \
    tests/difftest.py tests/workloads/all.jsonl --bin-dir "$(pwd)/tests/bin" -v
```

Data: SF1 parquet at `<repo_root>/tmp/data/tpch_parquet/sf1` (override with
`--data`). Generate with DuckDB: `CALL dbgen(sf=1)` + `COPY ... (FORMAT PARQUET)`.

## Workloads

- `workloads/q01.jsonl` … `q06.jsonl` — Q1–Q6 (86 cases)
- `workloads/q07_12.jsonl` — Q7–Q12 (57 cases)
- `workloads/q13_17.jsonl` — Q13–Q17 (66 cases)
- `workloads/q18_22.jsonl` — Q18–Q22 (52 cases)
- `workloads/smoke.jsonl` — harness smoke test (4 cases)
- `workloads/all.jsonl` — concatenation of the above (265 cases)

Coverage per query includes: all three benchmark seeds, domain boundaries
(dates before/after the data range, exact boundary days, leap-day month
arithmetic), out-of-dictionary strings (unknown regions/segments/brands/
nations/shipmodes, lowercase variants), numeric extremes (0, negative, huge,
fractional), duplicate IN-list values, empty-result parameters, and — for
every reachable guard — at least one case where the guard is FALSE so the
fallback engages (asserted via markers). Guards whose false-branch is
unreachable for any valid placeholder (they depend only on builder/data
invariants) are documented with justification in the audit findings
(`tests/findings/audit_findings_group{A..D}.md`; the independent review
report is `tests/findings/review_gpt56sol.md`).

## Audit conclusion

265/265 cases pass; no correctness bug was found in any fast path. A
follow-up read-only review (independent model) re-ran the suite, verified
the instrumented and plain builds have identical ELF code sections, and
confirmed the harness fixes now in place (strict marker matching, empty-
frame schema checks, exact seed coverage for all three seeds). One
documented contract note: Q22 truncates >2-character phone codes to two
characters — byte-identical to the paper baseline's own `add_code`, so the
optimized engine reproduces the artifact's behavior exactly.
