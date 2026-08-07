# Read-only optimization review — bespoke TPC-H engine

## Verdict summary

No cheating was found: request parameters are parsed and queries are executed inside the reported timing window; CSV output is outside it; and no cross-request result cache exists. The precomputations are parameter-independent. However, there are two evidenced correctness defects, so the overall verdict is **ISSUES FOUND**.

## Checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Timing-window integrity | **PASS** | `engine/query_impl.cpp:158-160` starts the timer before dispatch. For every Q1–Q22 branch, `parse_qN` and `run_qN` precede `end`, while `write_qN_csv` follows it (`engine/query_impl.cpp:161-378`; representative Q1 sequence at lines 161-167 and Q22 at 371-377). The optimized/baseline diff changes only removal of `AffinityGuard`; request collection and query-id extraction at lines 142-156 are unchanged from `engine_baseline/query_impl.cpp:142-157`, and no placeholder parsing or per-request warming was moved before the timer. |
| 2 | No result memoization across requests | **PASS** | An exhaustive static-state scan of `engine/query_q1.cpp` through `query_q22.cpp` and `query_impl.cpp` found only immutable month-length tables and `static TraceData` objects under tracing (for example, `query_q1.cpp:59`, `query_q10.cpp:201`, and `query_q22.cpp:209`). Every executor creates result/aggregate containers locally; no parameter-keyed result cache or persistent result state exists. |
| 3 | Parameter-independent precomputation | **PASS** | `PrecomputedArtifacts` covers date axes, all dictionary codes, all nations/types/brands/sizes/phone prefixes, and all relevant rows (`engine/builder_impl.hpp:160-326`). All artifact builders are invoked at build time (`engine/builder_impl.cpp:1765-1803`). The literals in builder code are fixed SQL-template predicates only: Q14 `PROMO` (`builder_impl.cpp:726-729`), Q19 ship modes/instruction (`builder_impl.cpp:926-944`), Q7/Q8’s fixed 1995–1996 windows (`builder_impl.cpp:1287-1428`), Q16 `Customer...Complaints` (`builder_impl.cpp:1551-1557`), Q10 return flag `R` (`builder_impl.cpp:791-830`), and Q21 status `F` (`builder_impl.cpp:598-673`). No placeholder date, region/nation, brand, color, or quantity is hardcoded in builder/query fast paths. |
| 4 | Races, numerical boundaries, overflow | **FAIL** | Two real defects are detailed below. Apart from them, reviewed prefix-window bounds consistently implement the SQL half-open/inclusive ranges, and other OpenMP shared updates use atomics, reductions, critical sections, or disjoint indices. |
| 5 | Completeness / wiring | **PASS** | All 22 IDs dispatch to their parser, executor, and writer (`engine/query_impl.cpp:161-378`). Builder population includes Q1, Q3–Q12, Q14–Q16, Q18–Q19, Q21–Q22 and shared part-key indexes for Q17/Q20, with Q9 built synchronously (`engine/builder_impl.cpp:1765-1803`); Q2 remains on its complete existing path. Guarded fast paths have general fallbacks where artifact construction can be skipped. No reachable unpopulated-artifact read or wrong empty fallback was found. |
| 6 | Output-format fidelity | **PASS** | Every `write_qN_csv` function is byte-for-byte unchanged from baseline (writers at `engine/query_q1.cpp:199`, `query_q2.cpp:434`, …, `query_q22.cpp:590`). Representative headers/precision are unchanged at `query_q1.cpp:199-211` and `query_q22.cpp:590-600`; the same holds for all intermediate writers. |

## Real bugs

### 1. Q20 fast path has an OpenMP data race

**Evidence:** `engine/query_q20.cpp:414-505` parallelizes the loop over matching parts. Its `qualify` lambda writes the shared `std::vector<uint8_t> qualified_suppkeys` (`query_q20.cpp:388-391`) at lines **492-493** without an atomic, lock, reduction, or per-thread buffer. A supplier can qualify through more than one matching part, so different iterations/threads can write the same byte concurrently. Even though both stores write `1`, this is a conflicting unsynchronized C++ access and therefore undefined behavior. The saved SF10 references contain 1,738–1,830 Q20 result rows, so this is not a dead path.

**Minimal reproduction suggestion:** at SF10, run Q20 with the valid seed-42 request:

```text
20 "linen" "1997-01-01" "FRANCE"
```

Use `OMP_NUM_THREADS=4` (or more) and a ThreadSanitizer/OpenMP-capable build; the report should point to `qualified_suppkeys[sk]` at lines 492-493. Repeated normal runs can also be compared with DuckDB, but a matching result does not remove the language-level race.

### 2. Q1’s new prefix cube narrows the charge accumulator from 128 to 64 bits

**Evidence:** the executor aggregate deliberately retains `__int128 sum_charge_num` (`engine/query_q1.cpp:27-33`), and the baseline scan adds each charge in `__int128` (`engine_baseline/query_q1.cpp:164-165`, 190-191, and 224-225). The new artifact instead declares `q1_sum_charge` as `std::vector<int64_t>` (`engine/builder_impl.hpp:170-175`), accumulates and prefix-adds it in signed 64-bit arithmetic (`engine/builder_impl.cpp:484-495`), and only casts the already-narrowed value back to `__int128` at query time (`engine/query_q1.cpp:136-140`).

This is not an SF10 failure, but it is a real supported-scale overflow. In the checked SF10 DuckDB references, the largest Q1 group has `sum_charge` of about `1.085e12`–`1.117e12`, corresponding to a cube numerator of about `1.085e16`–`1.117e16`. That leaves only about **826–850 times** signed-64-bit headroom relative to SF10. TPC-H data cardinality and these sums scale linearly, so around SF8,300 and above—and clearly at SF10,000—the prefix addition exceeds `INT64_MAX`, invoking signed-overflow undefined behavior and producing a wrong `sum_charge`. The baseline’s 128-bit path does not have this regression.

**Minimal reproduction suggestion:** on standard SF10,000 data, compare Q1 with DuckDB/baseline using a valid minimum-delta request that includes the most rows:

```text
1 "60"
```

The optimized cube’s `sum_charge` should overflow during builder prefix construction before the request is executed.

## Overall verdict

**ISSUES FOUND** — the speedup mechanism is legitimate and the claimed SF1/SF10 paths show no cheating, memoization, parameter specialization, incompleteness, or output-format change. Nevertheless, Q20 contains a genuine OpenMP race, and Q1 regresses accumulator width enough to fail at high standard TPC-H scale factors.
