# Fast-Path Correctness Audit — Group A (Q1–Q6)

Scope: `projects/bespoke_tpch_x4/engine/query_q{1,2,3,4,5,6}.cpp` vs the pristine
baseline (`engine_baseline/`, READ-ONLY), differential-tested on SF1 parquet data
with the paper tolerance (atol=rtol=1e-2, all-column-sorted frames) via
`tests/difftest.py`. Every branch below is instrumented with `AUDIT_PATH`
(active only under `-DBESPOKE_AUDIT`; non-audit builds are unaffected — verified
by a clean `./bench/bench.sh engine 1 42 3` run: total 13.8 ms,
`validation_errors == {}`).

Final status: **86/86 adversarial cases pass** (q01: 13, q02: 14, q03: 12,
q04: 13, q05: 15, q06: 19) plus the pre-existing `smoke.jsonl` 4/4.
**No correctness bugs were found in any Q1–Q6 fast path; no engine logic was
changed** — the only engine edits are AUDIT_PATH markers and `#ifdef
BESPOKE_AUDIT` bookkeeping flags.

Shared background facts (verified with DuckDB on SF1, and guaranteed by the
TPC-H generator at every SF): shipdate ∈ [1992-01-02, 1998-12-01], orderdate ∈
[1992-01-01, 1998-08-02], discount ∈ {0.00..0.10}, quantity ∈ {1..50} (integers),
partsupp has exactly 4 rows per part sorted by partkey, part/customer/supplier
keys are dense 1..N. Both engines parse dates with the same `int16_t`
day-offset cast (copied from the baseline), so even absurd-but-parseable dates
(e.g. year 2100, which would wrap int16) behave identically in both binaries;
tests use ±60-year dates that stay within int16 range.

---

## Q1 (`query_q1.cpp`) — pricing summary report

Fast path: per-(returnflag×linestatus, shipdate) prefix-sum cube
(`build_q1_artifacts`), answering any DELTA in O(groups).

Branches / markers:
1. `q1 fast` — guard: `cutoff_offset >= pre.q1_date_min && pre.q1_date_max >=
   pre.q1_date_min && pre.q1_group_count == group_capacity`. Reads prefix cell
   `min(cutoff, date_max)`; the `min()` clamp handles cutoffs past the data.
2. `q1 empty_precutoff` — cutoff strictly before the earliest shipdate: no row
   can satisfy `l_shipdate <= cutoff`, so the untouched zero aggregates (empty
   result, since `count_order == 0` rows are skipped) are exact. Reachable:
   `DELTA >= 2526` on SF1. Verified against baseline for DELTA 2526/2600/20000.
3. `q1 no_precompute` — any other guard failure. **Unreachable for any
   placeholder value**: `build_q1_artifacts` always runs at ingest; it sets
   `q1_group_count = |returnflag dict| × |linestatus dict|`, the exact
   expression the query recomputes as `group_capacity` from the same immutable
   dictionaries, and `q1_date_max >= q1_date_min` holds whenever lineitem is
   non-empty (if lineitem were empty the branch would be reached but the empty
   result would still be exact, and `group_capacity` would be 0 anyway).

Adversarial cases: seeds (100/80/63), spec bounds (60/120), DELTA 0 and 1
(cutoff at/just inside max shipdate), −30 and −10000 (cutoff after all data →
full-table totals via the clamp), 2525 (cutoff exactly = min shipdate, prefix
index d=0), 2526/2600/20000 (guard-false → fallback empty branch engages).

## Q2 (`query_q2.cpp`) — minimum-cost supplier

No precomputed cube; the "fast path" is structural: dense-stride partsupp range
lookup + dictionary-level type-suffix filter.

Branches / markers:
1. `q2 unknown_region` — REGION not in `region.name_to_key`: `r_name =
   '[REGION]'` matches nothing → exact empty result. Reachable: "ATLANTIS",
   lowercase "asia" (comparison is case-sensitive, like the baseline's `==`).
2. `q2 stride_range` — per-part candidate range `[prow*stride, prow*stride+stride)`
   validated by boundary checks (`candidate[0..stride-1] == partkey`, neighbors
   differ). Taken for every part on TPC-H-shaped data.
3. `q2 binsearch_range` — per-part `lower_bound` fallback when the stride
   candidate fails validation, or `stride == 0` (partsupp row count not an
   exact multiple of part row count). **Unreachable via placeholders**: the
   guard depends only on partsupp layout (exactly 4 sorted rows per dense
   partkey — verified: 0 parts deviate on SF1). Parameters cannot alter it.
   The fallback is nevertheless exercised implicitly: parts not matching the
   size/type filter never enter either branch, and the stride validation logic
   is itself the guard. Asserted `forbid` on all seed cases.
4. `q2 tie_overflow` — more than 16 suppliers tied at min supplycost for one
   part. **Unreachable on TPC-H data**: each part has exactly 4 partsupp rows,
   so at most 4 candidates exist; the 16-slot buffer cannot overflow. Data
   invariant, parameter independent.
5. `q2 no_matching_parts` — size/type filter selects zero parts → empty join.
   Reachable: out-of-dictionary suffix "XYZZY", size 0 / −5 / 1000000.

Adversarial cases: 3 seeds, "MIDDLE EAST" (space in name), sizes 1 and 50
(domain bounds), 1-char suffix "N", full-type-name suffix, unknown/lowercase
region, unknown suffix, zero/negative/huge sizes.

## Q3 (`query_q3.cpp`) — shipping priority

Fast path: builder-precomputed per-order market-segment codes
(`pre.q3_order_segment`, 0xFFFF sentinel for orders with unknown custkey)
scanned in parallel with OpenMP.

Branches / markers:
1. `q3 fast` — guard: `pre.q3_order_segment.size() == orders.row_count &&
   segment_code >= 0`. `build_q3_artifacts` unconditionally assigns the vector
   to `orders.row_count` entries, so the size clause is a builder invariant;
   the parameter-dependent clause is dictionary membership of SEGMENT.
2. `q3 unknown_segment` — SEGMENT not in the mktsegment dictionary. The code
   then drops into the fallback with an all-zero customer bitmap (the customer
   scan is skipped when `segment_code < 0`), producing the exact empty result.
   Reachable: "NOSUCHSEG", lowercase "building".
3. `q3 fallback_sorted` — baseline per-order range scan. Engages whenever the
   fast guard is false and lineitem is orderkey-sorted; observed together with
   `q3 unknown_segment` (the only parameter-reachable trigger).
4. `q3 fallback_unsorted` — same but for unsorted lineitem storage. **Unreachable**:
   the loader always sorts lineitem by orderkey (`lineitem.orderkey_sorted` is
   a loader invariant); documented, cannot be triggered by any request.

Adversarial cases: 3 seeds, leap-day date 1996-02-29, DATE = min orderdate
(empty: strict `<`), min+1 day, DATE = max shipdate and past it (empty: strict
`>`), 1970-01-01 / 2050-06-15 (±decades, int16-safe, empty), unknown and
lowercase segments (guard-false → fallback markers asserted, `q3 fast`
forbidden). All results match the baseline binary.

## Q4 (`query_q4.cpp`) — order priority checking

Fast path: per-(priority, orderdate) prefix counts of orders having ≥1 late
lineitem (`build_q4_artifacts`), window answered by prefix subtraction.

Branches / markers:
1. `q4 fast` — guard `pre.q4_built && pre.q4_priority_span ==
   orderpriority.dictionary.size()` with a non-empty clamped window
   `[max(start, date_min), min(end-1, date_max)]`.
2. `q4 window_empty` — the 3-month window lies entirely outside the orderdate
   domain (`lo > hi`): zero counts → empty result, exact because the count of
   qualifying orders in a disjoint window is 0. Reachable: DATE 1998-08-03
   (one day past max), 1991-10-01 (half-open window ends exactly at domain
   start), 1970-01-01, 2050-01-01.
3. `q4 fallback_sorted` / `q4 fallback_unsorted` — baseline scans. **Unreachable
   via placeholders**: `q4_built` is a builder invariant (set whenever orders
   exist and ranges are built; both hold after ingest), and `q4_priority_span`
   is copied from the same dictionary the query reads. The unsorted variant is
   additionally blocked by the loader's orderkey sort.

Adversarial cases: 3 seeds; window start at min orderdate; window straddling
min (lo clamp) and max (hi clamp); start exactly at max orderdate (single-day
overlap); month-arithmetic day clamps 1996-11-30→1997-02-28 and
1995-11-30→1996-02-29 (leap; identical `add_months` code as baseline);
four guard-false empty-window cases with `q4 fast` forbidden.

## Q5 (`query_q5.cpp`) — local supplier volume

Fast path: per-(nation, orderdate) prefix-sum cube of local-supplier revenue
(`build_q5_artifacts`; the `c_nationkey = s_nationkey` join is template-fixed,
REGION/DATE remain free parameters — region filtering happens at query time
via `nation_in_region`).

Branches / markers:
1. `q5 unknown_region` — REGION not in the region table → exact empty result.
   Reachable: "ATLANTIS", lowercase "africa".
2. `q5 fast` — guard `pre.q5_built && max_nationkey < pre.q5_nation_span`,
   non-empty clamped window. `q5_nation_span` = `max_nation_span(db)` ≥
   `max_nationkey + 1` by construction, so the second clause is a builder
   invariant.
3. `q5 window_empty` — 1-year window disjoint from the orderdate domain →
   empty result, exact. Reachable: 1998-08-03, 1991-01-01 (half-open boundary),
   2050-01-01.
4. `q5 fallback` — baseline join/scan. **Unreachable via placeholders**:
   `q5_built` fails only for empty tables, nation span outside (0, 256], or a
   missing `orderkey_to_row` map — all data/loader invariants (25 nations,
   non-empty tables). Sub-branches inside the fallback
   (`nationkey_by_custkey`/`nationkey_by_suppkey` presence, sorted/unsorted
   lineitem) are likewise loader invariants; they retain the baseline logic
   verbatim.

Adversarial cases: 3 seeds (all AFRICA — plus ASIA/EUROPE/AMERICA/MIDDLE EAST
to cover every region incl. the space-containing name), window start at min /
at max orderdate, straddling both ends, leap-day 1996-02-29 (+1yr clamps to
1997-02-28), three guard-false empty windows, unknown/lowercase region.

## Q6 (`query_q6.cpp`) — revenue-change forecast

Fast path: prefix-sum cube over (discount, quantity/100 bucket, shipdate)
(`build_q6_artifacts`), revenue = Σ stripes with `disc ∈ [D−1, D+1] cents` and
`bucket*100 < quantity_limit`.

Bucket-exactness note: `build_q6_artifacts` **refuses to build the cube if any
quantity is negative or not a multiple of 100** (scaled), so within the fast
path a bucket holds exactly the rows with quantity == bucket*100, making the
`bucket*100 < limit` test exact even for fractional limits like 24.5 (verified:
case matches baseline's strict `quantity < 2450` scan). Negative discount
windows (DISCOUNT "0.00" → [−1, +1]) are clamped with `max(discount_min, 0)`,
matching the baseline's unsigned-range trick which also admits only {0, 1}.

Branches / markers:
1. `q6 fast` — `pre.q6_built` and non-empty clamped shipdate window.
2. `q6 window_empty` — 1-year window disjoint from the shipdate domain → zero
   revenue; both engines emit exactly one `0.00` row (emission code is shared
   below the branch), so the result is identical. Reachable: 1998-12-02 (one
   day past max shipdate), 1991-01-01 (ends before min), 2050-01-01.
3. `q6 fallback_flat` / `q6 fallback_shards` — baseline scans (flat vs
   loader-sharded). **Unreachable via placeholders**: `q6_built` fails only on
   non-integral/negative quantities or a cube larger than 2^22 cells — data
   invariants (TPC-H quantities are integers 1..50, discounts 0..10 cents).
   The flat/shards choice is additionally a loader invariant.

Adversarial cases: 3 seeds; DISCOUNT 0.00 (negative-low clamp), 0.10 (data
max), 0.99 (far above data → zero revenue); QUANTITY 0, −5 (nothing qualifies),
1 (= data min, strict `<` → zero), 1000000 (everything qualifies), 24.5
(fractional bucket boundary); window start at min/max shipdate, straddling
both ends, leap-day start; three guard-false empty windows.

---

## Bugs found and fixed

None in Q1–Q6 engine logic. One **test-tooling pitfall** (not an engine bug)
was identified and worked around: `tests/difftest.py` matches expect/forbid
markers by substring, so marker names must not be substrings of one another.
Initial names `qN fast_empty_window` false-tripped `forbid: ["qN fast"]`; the
markers were renamed to `qN window_empty`. No shared-file (builder/loader/
query_impl) issues were observed for Q1–Q6.

## How to reproduce

```
cd projects/bespoke_tpch_x4
flock /tmp/audit_build.lock ./tests/build_audit.sh tests/bin_A
for n in 01 02 03 04 05 06; do
  uv run --no-project --with duckdb --with pandas \
      python3 tests/difftest.py tests/workloads/q$n.jsonl \
      --bin-dir "$(pwd)/tests/bin_A" -v
done
```
(pass `--bin-dir` as an absolute path; difftest runs the engines from temp
directories).
