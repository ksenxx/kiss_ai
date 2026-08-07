# Fast-Path Correctness Audit — Group D (Q18, Q19, Q20, Q21, Q22)

Scope: `engine/query_q{18,19,20,21,22}.cpp` of the optimized bespoke TPC-H engine,
audited against the pristine baseline (`engine_baseline/`) on SF1 data with the
paper's validation tolerance. Test manifest: `tests/workloads/q18_22.jsonl`
(51 cases). Final status: **51/51 PASS** (results identical to baseline AND all
expected/forbidden path markers verified).

Build/run used:
```
flock /tmp/audit_build.lock ./tests/build_audit.sh tests/bin_D
uv run --no-project --with duckdb --with pandas python3 tests/difftest.py \
    tests/workloads/q18_22.jsonl --bin-dir "$(pwd)/tests/bin_D" -v
```
(Note: `--bin-dir` must be absolute; difftest launches the engines with
`cwd=<tempdir>`, so a relative bin dir raises FileNotFoundError. Harness file is
read-only for this audit, so the absolute path is simply passed on the command
line.)

---

## Q18 — large-volume customers

### Guards / branches (markers)
| Marker | Condition |
|---|---|
| `q18 null` | `args.QUANTITY == "<<NULL>>"` → empty result, no scan |
| `q18 fast` | `db.pre.q18_order_sum_qty.size() == orders.row_count && orders.row_count > 0` — precomputed per-order sum(l_quantity) |
| `q18 fast dense-cust` / `q18 fast mapped-cust` | inside fast path: dense custkey (1..N) direct index vs. custkey→row map |
| `q18 fallback dense` | fast guard false, orderkeys dense (1..N) — baseline-style dense-array aggregation |
| `q18 fallback sparse` | fast guard false, orderkeys sparse — baseline-style max-keyed arrays |

### Adversarial cases
Seed values 314/313/315; boundary 300; `0`, `1` (near-total qualification,
~1.5M-row result diffed OK against baseline); `10000` (empty result); `-5`
(negative threshold — exercises the `sum >= 0` sentinel: builder initializes
per-order sums to −1 so orders without lineitems are excluded even when the
threshold is negative, matching `GROUP BY ... HAVING` semantics which only
produces groups for orders that appear in lineitem); `314.5` (fractional, stod +
cents scaling); `<<NULL>>`.

### Unreachable-guard justification
`build_q18_artifacts` unconditionally does
`pre.q18_order_sum_qty.assign(orders.row_count, -1)` before any early return, so
the size always equals `orders.row_count`. The guard can only be false when
`orders.row_count == 0` (whole-table-empty data shape, impossible on TPC-H SF1
and independent of placeholder values). The `q18 fallback dense/sparse` branches
are therefore unreachable for ANY valid placeholder; they are instrumented and
`forbid`-asserted on every Q18 case. The fallback code is byte-for-byte the
baseline algorithm, so even if an exotic data shape engaged it, it computes the
reference answer.

### Bugs found
None. (Correct handling of negative thresholds via the −1 sentinel + `sum >= 0`
check was specifically verified.)

---

## Q19 — discounted revenue (three OR'd brand/container/quantity groups)

### Guards / branches (markers)
| Marker | Condition |
|---|---|
| `q19 all-groups-null` | every group has QUANTITYi or BRANDi = `<<NULL>>` → single default row |
| `q19 fast` | `db.pre.q19_built` — compact prefiltered rows (fixed shipmode/shipinstruct predicate) with brand code + container/size bits |
| `q19 fallback` | `q19_built == false` — baseline part-mask + lineitem scan |
| `q19 fallback empty-part` | fallback with `max_partkey < 0` (both part and lineitem empty) |

### Adversarial cases
All three benchmark seeds (seed7 includes a duplicate brand across groups 1&2);
negative/zero quantities (`-10/0/-1`); all-zero; huge (`1000000/2000000/3000000`
— scaled ×100 still fits int32, revenue 0); inverted `q1>q2>q3`; identical
brand+quantity in all three groups (dedup/mask logic: `row_fixed` bits AND'd
with per-group brand equality reproduces OR semantics without double counting a
row — a row is summed once via the `matches` short-circuit chain, same as the
baseline's switch/default logic); unknown brands (`Brand#99`, `Brand#00`,
`NoSuchBrand` → sentinel code 0xFFFE matches no row, revenue 0.00 row, matching
baseline); one group NULL with two active; all groups NULL; fractional
quantities.

### Unreachable-guard justification
`build_q19_artifacts` sets `q19_built = true` unless (a) the `AIR`/`AIR REG`
shipmode or `DELIVER IN PERSON` shipinstruct dictionary entries are missing, or
(b) `part.brand.dictionary.size() >= 0xFFFF`. Both are data-shape invariants of
the loaded TPC-H data (dictionaries are fixed at ingest), independent of any
placeholder value. On SF1/SF10 the codes exist and there are only 25 brands, so
`q19 fallback` is unreachable; it is `forbid`-asserted on every Q19 case. The
fallback is the baseline algorithm.

Also verified: the fast path's precomputed rows exclude lineitems whose part can
satisfy no OR-group for ANY brand (`fixed == 0 || brand == 0xFFFF`) — this
exploits only the template-fixed container/size/shipmode/shipinstruct sets,
never the placeholder brands/quantities, so it is sound for arbitrary
parameters.

### Bugs found
None.

---

## Q20 — potential part promotion

### Guards / branches (markers)
| Marker | Condition |
|---|---|
| `q20 null` | COLOR, DATE, or NATION = `<<NULL>>` → empty |
| `q20 unknown-nation` | NATION not in nation table → empty |
| `q20 empty-part` | `max_partkey < 0` (empty part table) → empty |
| `q20 fast` | `!pre.li_by_partkey_offsets.empty() && pre.ps_by_partkey_sorted` — lineitem-by-partkey CSR + partsupp offsets |
| `q20 fallback` | CSR missing or partsupp not sorted by partkey — baseline bitset + hash-map scan |

### Adversarial cases
Three benchmark seeds (incl. two-word nation `SAUDI ARABIA`); color prefix
matching no parts (`zzzznotacolor` → empty); **empty color prefix** (`""` —
`LIKE '%'` matches all 200k parts; heaviest possible COLOR, diffed OK); 1-char
prefix `l`; window at the start of the shipdate domain (`1992-01-01`); window
past the end (`1998-09-01` → empty); leap-day start `1996-02-29` (the +1-year
end date clamps to 1997-02-28 via `days_in_month`, identical to the baseline's
`parse_date_offset_add_years`); unknown nation `ATLANTIS`; NULL color.

### Unreachable-guard justification
The CSR builder computes `max_partkey` over part ∪ lineitem ∪ partsupp and
always builds `li_by_partkey_offsets` when any partkey ≥ 0 exists;
`ps_by_partkey_sorted` is set iff the loaded partsupp table is sorted by
partkey. Both are ingest-time data invariants (TPC-H partsupp parquet is sorted
by partkey), independent of placeholders, so `q20 fallback` and
`q20 empty-part` are unreachable for any valid request on this data; both are
instrumented, and `q20 fallback` is `forbid`-asserted on every Q20 case.
`ps_offsets_by_partkey` is sized from the same global `max_partkey`, and every
part-table partkey is ≤ that max, so the fast path's bounds `continue` guards
can only skip negative partkeys (which the baseline also cannot match).

### Bugs found
None. (The per-part 8-entry supplier accumulator + spill vector was checked for
overflow behavior: spill handles >8 distinct suppliers per part correctly, and
the `qualify` lambda's first-match `break` on partsupp is safe because
(ps_partkey, ps_suppkey) is a primary key.)

---

## Q21 — suppliers who kept orders waiting

### Guards / branches (markers)
| Marker | Condition |
|---|---|
| `q21 null` | NATION = `<<NULL>>` → empty |
| `q21 unknown-nation` | NATION not in nation table → empty |
| `q21 no-f-status` | no `F` code in o_orderstatus dictionary → empty |
| `q21 empty-supplier` | `max_suppkey < 0` → empty |
| `q21 fast` | `db.pre.q21_built` — precomputed (late_suppkey, count) list per qualifying order |
| `q21 fallback sorted` / `q21 fallback unsorted` | baseline orders×lineitem scan (lineitem sorted by orderkey or not) |

### Adversarial cases
RUSSIA / UNITED STATES / VIETNAM (benchmark seeds), ALGERIA, UNITED KINGDOM
(incl. two-word nations), unknown nation WAKANDA, NULL.

The NATION placeholder only selects which precomputed (suppkey,count) entries
are kept (`is_target_supplier` filter applied at query time), so the precompute
is fully parameter-independent; the per-order exists/not-exists logic in
`build_q21_artifacts` is line-for-line the baseline's.

### Unreachable-guard justification
`build_q21_artifacts` leaves `q21_built == false` only when orders/lineitem is
empty, `orders.lineitem_ranges` is missing, or there is no `F` status code —
all ingest-time data invariants. In the first and third situations the query
returns empty via its own `q21 no-f-status` / `q21 empty-supplier` guards before
reaching the fast-path branch anyway. Hence `q21 fallback *` is unreachable for
any valid placeholder on this data; `forbid`-asserted on every Q21 case.

### Bugs found
None.

---

## Q22 — global sales opportunity

### Guards / branches (markers)
| Marker | Condition |
|---|---|
| `q22 no-codes` | all seven I placeholders NULL or shorter than 2 chars → empty |
| `q22 fast` | no non-numeric 2-char-prefix codes AND `db.pre.q22_built` — per-code positive-acctbal sum/count + sorted no-order acctbal lists with suffix sums |
| `q22 numeric fallback` | numeric codes only but `q22_built == false` — candidate scan with prefix codes |
| `q22 extra fallback` | at least one non-numeric code — full phone-prefix comparison scan |

### Adversarial cases
Benchmark seed sets (both orders); duplicate codes (`13 13 13 17 17 31 31` —
dedup in `add_code` + `seen[]` ensures no double counting, matching SQL IN-set
semantics); codes matching no customers (`99..93` and `00..06` → empty);
1-char codes (`"1","3",...` — skipped by `add_code` in BOTH engines because a
2-char substring can never equal a 1-char literal); 3-char code `"133"`;
non-numeric codes `"AB"`/`"ZZ".."TT"` (**guard-false cases**: force
`q22 extra fallback`, verified the fallback engages and matches baseline —
non-numeric prefixes match no TPC-H phone, but the numeric codes in the same
request are still aggregated correctly by the fallback); single active code with
six NULLs; all NULL.

Integer-average equivalence was verified analytically: the fast path filters
no-order customers with `acctbal > floor(sum/count)` via `upper_bound`; for
integer-cent acctbals and positive sum/count, `acctbal > sum/count (rational)`
⟺ `acctbal ≥ floor(sum/count)+1`, so the floor comparison is exact (and the
baseline uses the identical floor comparison).

### Baseline-contract note (not a bug)
`add_code` truncates codes longer than 2 characters to their first two chars
(`"133"` behaves like `"13"`). Strict SQL semantics
(`substring(c_phone from 1 for 2) IN ('133')`) would match nothing. However the
**baseline engine's `add_code` is byte-identical**, so the optimized engine
reproduces the paper artifact's behavior exactly; the differential test with
`"133"` passes. Documented here instead of "fixed", since the audit's
correctness reference is the baseline artifact.

### Unreachable-guard justification
`build_q22_artifacts` sets `q22_built = true` unless customer is empty,
`customer.phone_prefix_code` was not materialized, or
`orders.orders_by_customer_offsets` is missing — ingest-time invariants
independent of placeholders. So `q22 numeric fallback` is unreachable for any
valid request on this data; `forbid`-asserted on every numeric-code Q22 case.
The `q22 extra fallback` guard-false path IS reachable (non-numeric codes) and
is positively exercised by two cases.

---

## Summary

* 51 adversarial cases, 51 pass: optimized results match the baseline within the
  paper tolerance on every case, including ~1.5M-row Q18 outputs.
* Every fast-path guard is instrumented with `AUDIT_PATH` (audit builds only;
  no-op macro in benchmark builds, placed outside hot loops / OpenMP regions).
* Every reachable fallback trigger was engaged and verified (`q18 null`,
  `q19 all-groups-null`, partial-NULL Q19, `q20 null`, `q20 unknown-nation`,
  `q21 null`, `q21 unknown-nation`, `q22 no-codes`, `q22 extra fallback`).
* Guard-false conditions that depend only on builder/data invariants
  (q18/q19/q20/q21 fallbacks, q22 numeric fallback) are documented above as
  unreachable for any valid placeholder, instrumented anyway, and
  `forbid`-asserted so a regression that accidentally disables a precompute
  would be caught as a marker failure.
* **No correctness bugs found in the Q18–Q22 fast paths; no engine logic was
  changed** — the only engine edits are the `#include "audit.hpp"` lines and
  AUDIT_PATH markers in the five Group D files.
