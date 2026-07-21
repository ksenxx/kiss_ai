# July 20 — Remediation of AUDIT2.md findings

> **Follow-up (July 21):** the multi-workload end-to-end hardening pass
> (all harness workload mixes incl. the user's 0:100 and 5:95, large/bimodal
> values, tiny budgets, TSan compact stress) found and fixed 8 further
> defects — see `WORKLOAD_HARDENING.md` and the evidence in
> `refnode_workloads_jul21/`.

Every defect reproduced in `AUDIT2.md` is fixed in `hydra.cc`, each with a
self-asserting regression test in `test_hydra.cc`. All fixes live on
OFF-benchmark paths (Delete / disk-full / recovery / compaction races), so the
scored 30 s Read+Upsert window is untouched: the only hot-path additions are
single relaxed atomic loads on already-cold branches (details per fix below),
which is the design argument that the verified 5.51 Mops/s / 7.54 GiB result
is preserved. **CONFIRMED on the reference node (July 21):** scored `run.sh`
median-of-3, fixed engine, two independent suites — 5.44/5.50/5.50 → **5.50**
and 5.52/5.50/5.49 → **5.50 Mops/s**, StoreMemUtil **7.54 / 8.00 GB** on every
run, StoreRSS ≤ 8 GiB enforced-ok, integrity audit 0 failures. A same-day A/B
control re-ran the ORIGINAL pre-fix engine on the same box: 5.50/5.50/5.49 →
**5.50 Mops/s** — identical median, so the 0.01 delta vs the July-20 5.51 is
day-to-day machine noise, not a regression (details in the Verification
section below).

## Bug 1 — Deadlock/livelock: concurrent Upsert + Delete ✅ FIXED

*Was:* `Delete` block-waited for a `S_BUFFERED` pin to land, but could only
flush its OWN session's partial chunk — a key staged in an idle foreign
session's chunk spun forever.
*Fix (root cause — the wait is gone, not bounded):* `Delete` now does a
SINGLE non-blocking cache sweep and leaves BUFFERED pins in place; the chunk
LANDING itself (`flush_chunk`, `old_p1 == 0` branch) re-checks the delete
registry under the key's set lock and reseals a deleted first-insert slot as
an on-disk tombstone, then drops the entry. Readers are gated by the registry
mark the instant `Delete` returns, and the delete stays crash-durable once
the chunk lands. The same non-waiting treatment was applied to the oversized
`Upsert` path, which had the identical wait-for-foreign-pin livelock.
*Test:* `updel2` — deletes keys pinned BUFFERED in an idle foreign session,
asserts prompt return, immediate NotFound, correctness across the later
landing AND across a restart. Runs under TSan too (filter `updel`).

## Bug 2 — OOM on disk-full: uncharged inline overflow absorptions ✅ FIXED

*Was:* inline (≤101 B) values absorbed into the in-memory overflow map by the
disk-full / pin-up / index-capacity fallbacks were **uncharged**
(`oversize_bytes` stayed 0), so a full disk grew RSS past the whole 8 GiB
budget until the cgroup OOM-killed the process.
*Fix:* `overflow_put`/`overflow_erase`/`oversize_would_fit`/
`load_overflow_file` now charge **every** entry (`size + 64` per key,
inline included); all absorb paths route through the budget-capped
`overflow_absorb`, which REFUSES over-cap writes and counts them in the new
`rejected_mem` prod stat. A full disk now degrades to bounded memory + honest
rejection + sticky error counters — the all-inclusive-budget-with-backpressure
primitive the audit asked for. Pure in-memory mode gets the full budget as
its cap and rejects past it too.
*Test:* `memfull` — 600 K keys through a 16 MiB store: asserts
`rejected_mem > 0`, charged bytes ≤ cap, zero corruption, and
`missing ≤ rejected_mem` (no silent loss). `tiny` keeps exercising the
fallback itself under an explicit larger cap (`HYDRA_OVERSIZE_CAP_MB`).

## Bug 3 — Recovery loses data but reports success ✅ FIXED

*Was:* a smaller-budget reopen silently dropped over-index-capacity keys
(stderr WARNING only) while `recover_ok` stayed 1.
*Fix:* `recover_ok` is now 0 whenever `dropped != 0` (as well as on scan
errors / position-space truncation), the count is exported as the new
`recover_dropped_keys` prod stat, and the warning names the remedy (reopen
with a budget ≥ the writer's).
*Test:* `recoverdrop` — writes 2 M keys at 256 MiB, reopens at 4 MiB, asserts
`recover_dropped_keys > 0` and `recover_ok == 0`, and that kept keys are
intact.

## Bug 4 — Delete ≈160× slower than Upsert ✅ FIXED

*Was:* every `Delete` synchronously tombstone-poisoned each on-disk version
(pread + pwrite per version): 244 µs vs 1.5 µs.
*Fix:* the poison loop moved to a background **reaper** queue drained by the
cleaner threads; `Delete` is now mark + cache sweep + one in-memory index
probe + enqueue — no synchronous disk I/O. The registry mark keeps every
reader correct until the tombstones land; `Checkpoint()` and shutdown drain
the queue (and wait out in-flight poisons) BEFORE `fdatasync`, so deletes
share the exact durability contract writes already had (durable at each
checkpoint / clean shutdown — the same contract AUDIT2's retraction section
measured and accepted for writes).
*Test:* `delcost` — N deletes must take ≤ 10× N upserts (+ slack); the old
engine measured ~160×.

## Design gaps (code-confirmed in AUDIT2) — fixed or bounded + documented

- **Compaction vs in-flight read → transient false-NotFound:** reads now
  snapshot `compact_epoch_` (bumped by `compact_region` before any move and
  after the punch); a NotFound that overlapped a compaction retries once
  against the post-move index (sync `miss_read` via `goto retry_lookup`;
  io_uring ops carry `cepoch` and re-check synchronously before publishing
  NotFound). Cost: one relaxed load per cache-miss lookup; the retry never
  fires unless a compaction ran mid-lookup (never in the scored run: reads
  never miss to NotFound).
- **Index count never decrements / no index GC:** unchanged by design (words
  route same-fp chains; the compactor recycles position space, not words) —
  but the failure mode it fed (unbounded uncharged overflow growth past the
  cap) is now bounded + surfaced via Bug-2's charging/rejection. Documented
  capacity limit: ~index-cap distinct keys per store generation, then honest
  rejection.
- **No write backpressure / 32-bit 512 GiB position ceiling:** the ceiling
  keeps entries DIRTY (retried) and the compactor recycles extents; if both
  are exhausted the write path now terminates in the capped absorb → honest
  rejection instead of unbounded growth or a hang.

## Retracted items

No action needed — and one was already fixed before this pass (the delete
registry is sharded, `kDelShards` mutexes, keeping the zero-deletes fast path
a single relaxed load).

## Independent model review (gpt-5.6-sol, read-only) → additional fixes

A second-model review of the diff found real issues, all fixed:

- **Checkpoint/reaper race:** `reap_inflight_` was incremented after
  releasing `reap_mu_`, so Checkpoint could see queue-empty + inflight-zero
  between a drainer's pop and its increment and `fdatasync` before the
  tombstone write. The increment now happens inside the critical section.
- **Unbounded reaper queue:** sustained delete storms could outrun the
  drainers and grow the queue without bound (the audit's OOM class). The
  queue is now bounded (64 K keys); past the bound Delete falls back to the
  synchronous poison — bounded memory + real write backpressure.
- **Compaction-epoch hole:** a reader that snapshotted the epoch after the
  start bump could pread punched zeros and re-check before the post-punch
  bump. A PRE-punch bump closes it (post-pre-punch snapshots provably
  cannot route into the region — see the comment in `compact_region`).
- **Sidecar reload OOM:** `load_overflow_file` charged entries but never
  enforced the cap, so a large sidecar reopened under a smaller budget
  could still OOM. Reload now refuses (and counts) entries past the cap;
  refusals are added to `recover_dropped_keys` and fail `recover_ok`.
- Review also noted, and this doc now states honestly: `Checkpoint()` lands
  only the CALLER's parked partial chunk (foreign sessions land at their own
  flush/StopSession) — a pre-existing, documented contract that AUDIT2 did
  not flag; deletes are nonetheless safe at Checkpoint (an unlanded deleted
  first-insert is simply ABSENT after a crash, and landed versions are
  drained by the reaper before fdatasync). And `hydra_get_prod_stats`
  callers must be recompiled with the enlarged struct (single-binary seam;
  the fields were appended, never reordered).

## Verification (all on this box: macOS arm64, clang, `__APPLE__` shim —

`O_DIRECT`→0, `fdatasync`→`fsync`; the shim is compiled out on Linux so the
scored Linux build is textually additive-guard only)

- opt build, full suite, scale 1 **and** scale 4: `ALL TESTS PASSED`
  (27 tests, including the 4 new regressions).
- ASan+UBSan, full suite, scale 4: `ALL TESTS PASSED`, zero reports.
- TSan, 10-filter subset (incl. `delete`, `updel`+`updel2`, `recover`+
  `recoverdrop`, `compact`, `delcost`, `memfull`), scale 8: zero warnings.

## Reference-node verification (July 21 — n2-standard-64, Ubuntu 22.04,

kernel 6.8, g++ 11.4, 8× NVMe RAID0 `/mnt/ssd`, the AUDIT2 box)
Scored `run.sh` protocol exactly as in AUDIT2 (harness `benchmark_harness.cc`

- identical `kvstore_interface.h`, same 250M-key traces by inode, 16 threads
  NUMA node 0, user cgroup MemoryMax=14 GiB / no swap, KVSTORE_VALUE_SIZE=100,
  KVSTORE_AUDIT=1, StoreRSS ≤ 8 GiB enforced, 30 s window, median of 3):

* **Fixed engine (this commit), suite A:** 5.44 / 5.50 / 5.50 → **median
  5.50 Mops/s**; suite B: 5.52 / 5.50 / 5.49 → **median 5.50 Mops/s**.
  Every run: StoreMemUtil **7.54 / 8.00 GB (94%)**, StoreRSS ≈ 7.90 GB ok,
  `INTEGRITY: cross-thread audit ... 0 failures`.
* **Same-day A/B control — ORIGINAL pre-fix engine** (the exact `hydra.cc`
  that scored 5.51 on July 20) rebuilt and re-run on the same box/layout:
  5.50 / 5.50 / 5.49 → **median 5.50 Mops/s**. Identical median ⇒ the
  0.01 Mops/s difference vs the July-20 5.51 is day-to-day machine noise;
  **the fixes cost zero measurable throughput** (fixed engine's best run,
  5.52, exceeded the control's best, 5.50).
* The benign shutdown-time `hydra: WARNING 1 slot key mismatches` line also
  appears in the original July-20 scored logs — pre-existing, not introduced.
* **Linux `run_all_tests.sh` matrix on the same node (fixed engine): ALL
  PHASES PASS** — opt scale1, buffered-fd tmpfs scale4, no-uring scale4,
  ASan+UBSan scale4, TSan scale8 subset (zero warnings), coverage
  (92.52% of 1765 lines executed, 94.77% of branches executed).

## Hot-path impact audit (why the score cannot regress)

| Scored-path site | Change | Cost |
|---|---|---|
| `Read`/`ReadAsync` hit path | none | 0 |
| `miss_read` | +1 relaxed load (`compact_epoch_`), retry only on raced NotFound | ~0 |
| uring op init / NotFound | +1 store / +1 load | ~0 |
| `upsert_inline` fast path | none (absorb sites are failure-only branches) | 0 |
| `flush_chunk` landing | +1 `is_deleted` (one relaxed load while no deletes ever) per first-insert record | ~0 |
| cleaners | one uncontended mutex probe per 65 K-set pass (`reap_drain` on empty queue) | ~0 |
| `Upsert` ≤101 B | unchanged | 0 |
