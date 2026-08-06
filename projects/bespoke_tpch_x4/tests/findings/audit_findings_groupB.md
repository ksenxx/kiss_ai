# Fast-path correctness audit — Group B (Q7, Q8, Q9, Q10, Q11, Q12)

Scope: `projects/bespoke_tpch_x4/engine/query_q{7,8,9,10,11,12}.cpp` versus the
pristine baseline (`engine_baseline/`), precomputes in `engine/builder_impl.cpp`
(read-only), templates in `engine/queries.txt`.

Method: every fast-path / fallback branch was instrumented with
`AUDIT_PATH("qN <branch>")` (no-op outside `-DBESPOKE_AUDIT` builds; never inside
hot loops or OpenMP regions). An adversarial manifest
`tests/workloads/q07_12.jsonl` (57 cases) was run through
`tests/difftest.py` on SF1: it batches all requests through the audited
optimized binary and the baseline binary, diffs each `result<i>.csv` with the
paper tolerance, and asserts expected/forbidden path markers.

**Final status: 57/57 cases PASS. No correctness bug was found in any Q7–Q12
fast path; no engine logic was changed (instrumentation only). The normal
bench build still validates cleanly (`./bench/bench.sh engine 1 42 1` →
`validation_errors == {}`).**

---

## Q7 — revenue between two nations, fixed 1995–1996 shipdate window

Branches / markers:
- `q7 nation_missing` — NATION1 or NATION2 not in `nation.name_to_key` → `{}`.
  Identical early-return exists in the baseline (`engine_baseline/query_q7.cpp:147-151`),
  and it matches SQL semantics (inner join on a non-existent name is empty).
- `q7 fast` — guard: `pre.q7_built && 0 <= nation1,nation2 < pre.q7_nation_span`.
  Precomputed cube `q7_cube[supp_n][cust_n][year-1995]` of `discounted_price`
  sums over the template-fixed shipdate window 1995-01-01..1996-12-31.
- `q7 fallback_ranges` / `q7 fallback_scan` — baseline-style scans.

Adversarial cases: 3 benchmark seeds; **same nation twice** (`FRANCE FRANCE`,
the OR-predicate degenerates to the cube diagonal — verified equal to
baseline); multi-word names (`UNITED STATES`/`UNITED KINGDOM`); non-benchmarked
pair; unknown NATION1, unknown NATION2; lowercase names (dictionary lookup is
case-sensitive, matching baseline).

Guard-false reachability: the only parameter-dependent part of the guard is the
nation-key lookup, and any *unknown* name already exits at `nation_missing`
before the guard. Known names always have keys `< nation_span` because the
builder sets `q7_nation_span = max_nation_span(db)` (max nationkey over the
nation/supplier/customer tables + 1). `q7_built == false` only if lineitem or
orders are empty or `nation_span ∉ (0,256]` — pure builder/data invariants, not
reachable through any valid placeholder value. The fallback markers exist and
are `forbid`-asserted on every fast case.

## Q8 — market share, fixed 1995–1996 orderdate window

Branches / markers:
- `q8 region_missing`, `q8 nation_missing` — dictionary misses → `{}`; same
  early-returns exist in the baseline (`engine_baseline/query_q8.cpp:127-133`).
- `q8 fast` — guard: `pre.q8_built && 0 <= target_nation < q8_nation_span &&
  max_nationkey < q8_nation_span && year_span == 2`. Cube
  `q8_cube[p_type][cust_nation][supp_nation][year-1995]` of volume for orders in
  the fixed window.
- `q8 fallback_ranges` / `q8 fallback_scan` — baseline-style scans.

Adversarial cases: 3 seeds; target nation inside vs. **outside** the region
(the outside case must still emit rows with `mkt_share = 0` — verified); TYPE
not in the part-type dictionary (empty result on both engines since every
lineitem joins to some part type); unknown REGION; unknown NATION; AFRICA
region not covered by the seeds.

Guard-false reachability: `year_span` is computed from the hard-coded template
dates 1995-01-01/1996-12-31, so it is always 2. `max_nationkey <
q8_nation_span` holds by construction of `max_nation_span`. `q8_built` is false
only for empty tables or `nation_span/type_span` outside (0,256]/(0,4096] —
data invariants. Fallback markers exist and are `forbid`-asserted.

## Q9 — profit by nation/year, `p_name LIKE '%COLOR%'`

Branches / markers:
- `q9 fast` — guard: `!pre.q9_part_offsets.empty() && pre.q9_year_span > 0`.
  Parameter-independent CSR: per partkey, entries `(group = supp_nation ×
  year_span + year_offset, profit = discounted_price − supplycost·qty/100)`.
  Only the COLOR substring filter over `p_name` runs at query time (OpenMP with
  thread-local accumulators merged under a critical section — exact int64
  arithmetic, ordering-independent).
- `q9 fast_no_nations` — `pre.q9_max_nationkey < 0` → `{}` (empty nation table;
  data invariant, unreachable via parameters).
- `q9 fallback` (+ `q9 fallback_fixed_four` / `q9 fallback_general` loop
  variants, `q9 fallback_no_orders` empty-orders early-out) — baseline-style scan.

Adversarial cases: 3 seed colors; no-match color (`zzznotacolor`); single
character `"a"` (near-total match); **empty COLOR** `""` (`LIKE '%%'` matches
every part — heaviest possible parameter, verified exact vs. baseline);
uppercase `"ROSY"` (case-sensitive, empty result); substring spanning a space
(`"n ro"`). Both engines implement LIKE '%x%' as plain substring `find`, so
semantics agree for every possible string parameter by construction.

Guard-false reachability: the guard depends only on builder success
(non-empty part/partsupp/lineitem/orders, ≤ 254 distinct order years,
nationkeys fitting `uint16` groups) — all data invariants on TPC-H data; COLOR
cannot influence it. Fallback markers exist and are `forbid`-asserted.

## Q10 — returned-item revenue, 3-month orderdate window

Branches / markers:
- `q10 no_returnflag_R` — `'R'` absent from the returnflag dictionary → `{}`.
  Unreachable via parameters (data invariant; TPC-H lineitem always contains
  'R' rows); marker kept and forbidden in tests.
- `q10 fast` — guard: `pre.q10_built && q10_order_r_revenue.size() ==
  orders.row_count`. Builder always sets this when orders are non-empty, so the
  guard is a pure data invariant; DATE only selects the orderdate range via
  binary search over the orderdate-sorted orders table.
- `q10 fallback` — per-order scan path (sorted/unsorted lineitem variants).

Adversarial cases: 3 seeds; window starting before the first orderdate
(1992-01-01); windows entirely before (1990-01-01) and after (1998-10-01,
1999-06-01) the data → empty results; `+3 months` **end-of-month clamping**
onto leap day (1995-11-30 → 1996-02-29) and non-leap clamp (1993-11-30 →
1994-02-28) — the optimized and baseline engines share the same
`parse_date_offset_add_months` logic, verified byte-identical results; partial
tail overlap (1998-05-15).

Guard-false reachability: unreachable for any valid placeholder — `q10_built`
and the size equality are established unconditionally by
`build_q10_artifacts` whenever orders exist. Documented here; fallback marker
`forbid`-asserted on all cases.

## Q11 — partsupp value concentration, NATION + FRACTION

Branches / markers:
- `q11 nation_missing` — unknown NATION → `{}` (same as baseline
  `engine_baseline/query_q11.cpp:116-121`).
- `q11 fast` — guard: `pre.q11_built && nation_key+1 < q11_nation_offsets.size()`.
  Per-nation `(partkey, value)` lists plus per-nation totals; threshold
  `total · stod(FRACTION)` computed in double exactly like the baseline, and
  the strict `>` comparison uses the same operands, so all FRACTION edge cases
  agree bit-for-bit.
- `q11 fallback` — supplier-mask + run-length partsupp scan.

Adversarial cases: 3 seeds; FRACTION `0` (strict `>` keeps only positive-value
parts), `-0.5` (negative threshold keeps everything), `1` and `100` (empty
result), `1e-10` (tiny); non-seed nation with large fraction; unknown nation.

Guard-false reachability: `q11_built` is false only if partsupp/supplier are
empty, `nation_span ∉ (0,256]`, or **partsupp is not sorted by partkey** (the
builder verifies this explicitly and leaves the fallback active). On the
loader's data layout partsupp is partkey-sorted, so with this dataset the
fast path always engages for any known nation; the condition is a
data-shape invariant, not parameter-reachable. Fallback marker exists and is
`forbid`-asserted. (Note: the fallback itself also relies on run-length
grouping over partkey-sorted partsupp, identical to the baseline's approach,
so both paths share the same data assumption.)

## Q12 — shipmode line counts, 1-year receiptdate window

Branches / markers:
- `q12 shipmodes_missing` — *both* shipmodes absent from the dictionary → `{}`
  (matches SQL: `l_shipmode IN (...)` can never match).
- `q12 fast` — guard: `pre.q12_date_max >= pre.q12_date_min &&
  !pre.q12_counts.empty()`. Prefix-sum cube over
  `(shipmode, high/low priority, receiptdate)`; the request window
  `[DATE, DATE+1yr)` is answered by two prefix differences after clamping to
  the data's receiptdate range.
- `q12 fast_window_empty` — clamped window empty (`lo > hi`) → zero counts.
- `q12 no_precompute` — **Q12 has no scan fallback.** Documented in-code: the
  builder constructs `q12_counts` unconditionally whenever lineitem and orders
  are non-empty, so this branch is reachable only on an empty database, where
  zero counts (empty result) are correct anyway. Marker added and forbidden in
  every test case.

Adversarial cases: 3 seeds; **duplicate shipmode** (`MAIL MAIL` — the
dictionary resolver's `else if` leaves `allowed_code2 = -1`, so rows are
counted once, matching SQL `IN` semantics — verified vs. baseline); one
unknown + one known shipmode (only the known one contributes); both unknown;
shipmode containing a space (`REG AIR`); windows entirely before/after the
data (both `q12 fast_window_empty`, empty result matches baseline); window
clipped at the data's right edge; leap-day start date `1996-02-29` (+12 months
clamps to 1997-02-28, same helper as baseline).

## Bugs found

None. All 57 differential cases produce results identical (within the paper's
atol/rtol=1e-2 policy; in practice exact) to the pristine baseline, and every
expected path marker engaged exactly as designed. No shared-file issues were
observed. The only engine changes made by this audit are `#include "audit.hpp"`
plus `AUDIT_PATH(...)` statements (compiled out of benchmark builds) and one
new no-op `else` branch in `query_q12.cpp` carrying the `q12 no_precompute`
marker and its unreachability comment.

## How to reproduce

```
cd projects/bespoke_tpch_x4
flock /tmp/audit_build.lock ./tests/build_audit.sh tests/bin_B
uv run --no-project --with duckdb --with pandas python3 \
    tests/difftest.py tests/workloads/q07_12.jsonl --bin-dir "$(pwd)/tests/bin_B" -v
```
