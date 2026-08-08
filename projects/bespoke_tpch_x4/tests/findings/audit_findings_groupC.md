# Fast-path correctness audit — Group C (Q13, Q14, Q15, Q16, Q17)

Scope: `projects/bespoke_tpch_x4/engine/query_q{13,14,15,16,17}.cpp` vs the
pristine baseline `engine_baseline/`, precomputes in `engine/builder_impl.cpp`
(read-only), templates in `engine/queries.txt`.

Test harness: `tests/workloads/q13_17.jsonl` (66 adversarial cases) run through
`tests/difftest.py` on SF1 parquet data with `--bin-dir tests/bin_C`
(note: pass an **absolute** bin dir; difftest launches the engines from a temp
cwd). Final status: **66/66 PASS** — every CSV matched the baseline within the
paper tolerance and every expected/forbidden AUDIT marker assertion held.

**No correctness bugs were found in any Q13–Q17 fast path; no engine logic was
changed.** The only engine edits are `#include "audit.hpp"` and `AUDIT_PATH`
markers (no-ops outside `-DBESPOKE_AUDIT`), all placed outside hot loops and
outside OpenMP parallel regions.

______________________________________________________________________

## Q13 — comment-pattern anti-join

Guards / branches (markers):

- `q13 fast masked` — single execution path. Per-customer base order counts
  come from the parameter-independent `orders_by_customer_offsets`; a parallel
  scan finds orders whose comment matches `%WORD1%WORD2%` and subtracts them.
  The precomputed per-comment `alpha_mask` (26-bit letter set, case-folded)
  and `bigram_mask` (64-bit byte-pair hash set) are **over-approximating
  prefilters**: a comment that truly contains the words always has every
  letter/bigram bit of the words set, so the prefilter can never reject a true
  match — verified that `build_alpha_mask` / `build_bigram_mask` in
  `builder_impl.cpp` are byte-identical to the copies in `query_q13.cpp`.
- `q13 fast nomask` — both words carry no letters and no bigrams (e.g. empty
  or single non-alpha chars): prefilter disabled, every comment is
  string-searched directly.

There is **no fallback branch** (the design has no parameter-dependent guard);
`orders_by_customer_offsets` is always built for TPC-H data. Correctness for
arbitrary words rests on the exact substring-search (`find_substring` handles
`needle_len == 0` and needles longer than the haystack).

Adversarial cases: seed words; both/one empty word; single-char words
(bigram mask 0); WORD1 == WORD2 and overlapping words (`abab`/`ab`,
`ests`/`ts`) exercising ordered non-overlapping search; capitalized words
(case sensitivity); LIKE metacharacters `%`/`_` treated literally (both
engines use plain substring search, so they agree by construction); digits
(alpha mask 0, bigram nonzero); embedded space; 80-char needle (OOB safety);
rare-letter words (prefilter rejects almost everything).

## Q14 — promo revenue share, 1-month window

Guards / branches (markers):

- guard `db.pre.q14_built` → fast path: per-shipdate promo/total prefix sums.
  - `q14 fast in-range` — clamped `[lo,hi]` intersects the data's shipdate
    span; two prefix-sum lookups answer the query.
  - `q14 fast empty-range` — window disjoint from the data span; both sums 0,
    `promo_revenue = 0.0` (baseline computes the same `total_sum > 0` guard).
- `q14 fallback` — **unreachable for any valid placeholder**: `q14_built` is
  false only when `lineitem` or `part` has zero rows (pure data-shape
  condition in `build_q14_artifacts`). The fast path was tested instead, and
  the fallback code is the baseline shard-scan kept intact.

Semantics checked: builder prefix sums include only lineitems whose partkey
joins `part` (`partkey_promo >= 0`), matching the fallback/baseline inner-join
semantics. Both engines share byte-identical `days_from_civil` /
`parse_date_offset_add_months` (verified with diff), so end-of-month clamping
(`1996-01-30`+1m → `1996-02-29`, `1997-01-31`+1m → `1997-02-28`) agrees.

Adversarial cases: 3 seed dates; before-min/after-max windows; exact boundary
months (1992-01-01, 1998-12-01); windows straddling either data boundary;
mid-month start; leap/non-leap clamping; window starting exactly one day past
max shipdate (1998-12-02 → empty-range).

## Q15 — top-revenue supplier, 3-month window

Guards / branches (markers):

- guard `db.pre.q15_built && q15_supp_span >= initial_capacity` → fast path
  (`q15 fast`) over the (month × suppkey) revenue cube. Per month in the
  window:
  - `q15 fast month-out-of-range` — month outside the precomputed range.
    Safe because the cube's month range equals the *actual* shipdate range of
    the data, so such months contain zero lineitem rows.
  - `q15 fast full-month` — month fully inside the window: adds the cube
    stripe (parallel loop; marker placed before the omp region).
  - `q15 fast partial-month` — window edge cuts the month (any non
    1st-of-month DATE): filtered scan of only that month's lineitem shards,
    with the exact `[start, start+span)` window predicate.
  - `q15 fast noshard-scan` — partial-month sub-branch when the lineitem
    table has no shard metadata: whole-table scan filtered on the month.
    Unreachable in this repo (the loader always builds shards); tested
    implicitly by the shard variant, documented here.
- `q15 fallback` (full baseline-style scan) — **unreachable for any valid
  placeholder**: `q15_built` is false only for empty lineitem/supplier or a
  cube larger than 2^25 slots (SF1 ≈ 84×10k, SF10 ≈ 84×100k, both far under);
  `q15_supp_span >= initial_capacity` always holds when built because
  `supp_span = max(suppkey over supplier ∪ lineitem) + 1 ≥ max(supplier.suppkey) + 1`. Pure data-shape conditions.
- `q15 empty-result` — both empty-return branches (no rows in window; max
  revenue 0). Baseline has the identical returns, so empty windows agree.

Tie semantics: all suppliers whose total equals `max_revenue` are emitted and
sorted by `s_suppkey` — the exact-integer (int64 cents) totals are identical
in both engines, so ties reproduce identically (noted in the manifest; no
parameter forces a tie on SF1, none was observed).

Adversarial cases: 3 seed requests (all month-start → cube-only, partial-month
forbidden); mid-month and month-end starts (`1996-01-31`+3m clamps to
04-30) → partial+full mix; windows entirely before/after the data (out-of-range

- empty-result); windows straddling the low/high data boundary; exact boundary
  months 1992-01 and 1998-12.

## Q16 — supplier counts per (brand, type, size)

Guards / branches (markers):

- `q16 empty-sizes` — every SIZE placeholder was `<<NULL>>`/non-numeric
  (`add_size` accepts only pure digit strings — identical lambda in the
  baseline): empty IN list → empty result in both engines. Reached with all
  sizes negative (e.g. `-1` … `-8`, still template-valid numbers).
- guard `db.pre.q16_built` → `q16 fast`: iterate precomputed distinct-supplier
  counts per (brand, type, size) group (complaint suppliers excluded at build
  time — same `Customer%Complaints` two-step find as the baseline), applying
  the three parameter filters: 64-bit `size_mask`, `banned_brand` dictionary
  code (`-1` = unknown brand bans nothing), `type_prefix_banned` per
  dictionary entry.
- `q16 fallback` (+ sub-branches `q16 fallback dense-partsupp` /
  `q16 fallback generic-partsupp`) — **unreachable for any valid
  placeholder**: `q16_built` is false only for empty part/partsupp,
  `max_size > 100000`, or `max_suppkey ≥ 2^20` (suppkey packing limit) —
  data-shape conditions only. Markers added anyway; fast path exercised by
  all parameterized filters instead.

Benign quirk (documented, not a bug): both the fast path and the fallback cap
`size_mask` at `kMaxPartSize = 50`, so a requested size in 51..63 would never
match — this is exactly the baseline's own behavior (same constant), and
TPC-H guarantees `p_size ∈ [1,50]`, so no valid data row can be affected.

Adversarial cases: 3 seed requests; 8 duplicate sizes (dedup via mask, no
duplicate output groups); unknown brand `Brand#99`; unmatched type prefix;
empty type prefix (bans every type → empty result); full type name as prefix;
lowercase prefix (case-sensitive, bans nothing); size 0 (valid number, no
part matches); all-negative sizes (empty-sizes guard); mixed −1/300/51/valid;
boundary sizes 1 and 50; off-seed brand/type mix.

## Q17 — small-quantity revenue for brand+container

Guards / branches (markers):

- `q17 null-arg` — literal `<<NULL>>` BRAND or CONTAINER: early 0-row (sum
  0 → `0.00`), identical early exit in the baseline.
- `q17 empty-part` — empty part table; **unreachable** data-shape condition
  (marker added, documented).
- `q17 unknown-code` — BRAND or CONTAINER not in the part dictionaries → sum
  over empty set, `0.00` in both engines.
- guard `!db.pre.li_by_partkey_offsets.empty()` → `q17 fast`: per-part CSR of
  lineitem rows; for each matching part compute `sum(quantity)`, `count`, then
  add `extendedprice` where `quantity*5*count < sum` (exact integer form of
  `l_quantity < 0.2*avg`, identical inequality to the fallback/baseline).
- `q17 fallback` (+ `q17 fallback no-matching-part`) — **unreachable for any
  valid placeholder**: the CSR is built whenever any partkey ≥ 0 exists across
  part/lineitem/partsupp (`build` guard is `max_partkey < 0` only).

Adversarial cases: both seed combos; unknown brand / unknown container / empty
brand / swapped arguments (all → unknown-code, empty-sum agreement); both
`<<NULL>>` sentinels; three non-seed valid brand×container combos (JUMBO PKG,
WRAP DRUM, LG CAN) exercising the CSR path off the benchmark values.

______________________________________________________________________

## Unreachable-guard summary (why each fallback cannot engage via parameters)

| Marker | Guard | Why unreachable for valid placeholders |
|---|---|---|
| `q14 fallback` | `!pre.q14_built` | builder skips only if lineitem/part empty |
| `q15 fallback` | `!pre.q15_built \|\| supp_span < capacity` | cube skipped only if tables empty or cube > 2^25; span ≥ capacity by construction |
| `q15 fast noshard-scan` | `lineitem.shards.empty()` | loader always shards lineitem in this artifact |
| `q16 fallback` (+ dense/generic) | `!pre.q16_built` | skipped only for empty tables, size > 100000, suppkey ≥ 2^20 |
| `q17 empty-part` | `max(part.partkey) < 0` | part table non-empty |
| `q17 fallback` (+ no-matching-part) | CSR offsets empty | CSR built whenever any partkey ≥ 0 exists |

All of these depend exclusively on builder/data invariants — no placeholder
value can flip them. Each fallback body is the retained baseline plan, so even
a hypothetical engagement would stay correct.

## Bugs found / fixed

- **None in the engine.** One manifest expectation was corrected during
  iteration: `14 "1998-12-02"` starts one day past the max shipdate
  (1998-12-01), so the correct path is `q14 fast empty-range` (result CSVs
  matched all along).

## Final test status

`uv run --no-project --with duckdb --with pandas python3 tests/difftest.py tests/workloads/q13_17.jsonl --bin-dir <abs>/tests/bin_C -v` →
**66/66 cases passed** (CSV diff + marker assertions).
