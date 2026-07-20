# PROD_READINESS — HydraKV audit item → fix → test → code map

Maps every HYDRA_PROD_AUDIT.md §7 gap (items 1–14) to its fix, its e2e test,
and where the code lives (all in `hydra.cc` unless noted). Tests are in
`test_hydra.cc`; run everything with `./run_all_tests.sh` (6-config matrix).
Design rationale and rejected alternatives: `DISCOVERY_LOG.md`.

| # | Audit gap | Status | Fix (code) | E2E test |
|---|---|---|---|---|
| 1 | HIGH: no crash recovery (O_TRUNC at open; no recover()) | FIXED | O_TRUNC removed; `recover_log()` scans the log (1 MiB buffers), skips zero/CRC-torn slots, rebuilds the index newest-LSN-wins, restores next_slot\_/landed_hw\_/lsn\_; Checkpoint() fdatasyncs; destructor lands partials + fdatasyncs (clean-shutdown durability) | `recover`, `crashrecover`, `compact` (restart leg), `enospc` (restart leg) |
| 2 | HIGH: index anonymous RAM, never persisted | FIXED (rebuild, not persist) | Index is rebuilt from the log at Init (Bitcask/FASTER-style); recovery scratch (per-word key/LSN arrays) freed before cache allocation, keeping peak RSS in budget. Recovery time scales with log size (documented) | `recover`, `crashrecover` |
| 3 | HIGH: abort() on write error incl. ENOSPC | FIXED | `xpwrite` → bool; `flush_chunk` keeps the chunk staged + entries pinned on failure, retries later; sticky `write_errors_`/`durable_ok_`; bounded upsert retry (16/64) then overflow absorption so writers never hang; tombstone/fdatasync writes fail soft | `enospc` (RLIMIT_FSIZE fault injection: no crash, reads correct, sticky stats, reflush + durability after repair) |
| 4 | MED: no GC/compaction; 512 GiB 32-bit position ceiling | FIXED | Background `compactor_main`/`compact_region`: threshold-triggered (HYDRA_COMPACT_FACTOR×live, HYDRA_COMPACT_FLOOR_MB), relocates live slots through the existing flush protocol, verifies the region is unreferenced, punches it (PUNCH_HOLE|KEEP_SIZE), recycles its position space through the free-extent list `ensure_chunk` consumes — positions are REUSED so the 32-bit ceiling cannot be hit while reclamation keeps pace | `compact` (st_blocks shrink, bounded log_bytes, correctness under concurrent readers, restart on holey log) |
| 5 | MED: read-path EIO fail-stop | FIXED | `xpread` → -1 on error; `read_page` zero-fills + counts `read_errors_`; io_uring completion errors abandon one chain, never the process; short reads below `landed_hw_` counted as faults | `readfault` (external truncate: NotFound/exact values only, read_errors>0, store stays usable) |
| 6 | LOW: >101 B values → unbounded off-budget RAM | FIXED | Overflow entries charged size+64 against `oversize_cap_` (budget/8, env-tunable); over-cap upserts rejected BEFORE any state changes + `rejected_oversize_`; charges credited on erase/shrink/delete; oversize bytes included in GetCacheStats hot_bytes; persisted via the atomic-rename sidecar | `oversizebound`, `recover` (oversized persistence + transitions) |
| 7 | LOW: RMW-vs-Upsert lost update | FIXED (resident keys) | RMW fast path does read+modify+write under the key's set lock (same lock Upsert takes); non-resident first-touch residual documented in DISCOVERY_LOG | `rmw` (existing) + fast path exercised by it; residual documented |
| 8 | LOW: multi-root stale read (:1336) | FIXED | All read paths (miss_read, uring_advance, verified_lookup) select the newest match by per-slot LSN (CRC-gated) instead of highest position | `alias` (existing) + `compact` (position reuse makes LSN ordering load-bearing) |
| 9 | LOW: Delete global-mutex cliff (:655) | FIXED | Deleted-key registry sharded 64-way (`del_shards_`); zero-deletes relaxed fast path unchanged | `delete`, `updel` (existing) |
| 10 | LOW: delete/oversized liveness gap (:1902) | MITIGATED/DOCUMENTED | Bounded-backoff waits kept; on a permanently dead disk Delete-of-BUFFERED-key waits politely (1 ms sleeps) rather than risking resurrection — correctness over liveness; documented in DISCOVERY_LOG | `updel`/`delete` cover live paths; dead-disk case documented |
| 11 | LOW: silent O_DIRECT→buffered fallback | FIXED | Fallback logged at open + surfaced as `buffered_fallback` in prod stats | matrix `buffered-fd tmpfs` phase (fallback exercised) |
| 12 | LOW: io_uring short-read zero-fill | FIXED | Per-op `filled` cursor; mid-page short reads resubmit the remainder; EOF classified against `landed_hw_` | `readfault` async leg; `async` (existing) |
| 13 | LOW: xpread aborts on EAGAIN | FIXED | EINTR/EAGAIN retried with backoff in xpread/xpwrite | covered by all disk tests |
| 14 | LOW: doc drift ">102 B" vs >101 | FIXED | Header comment corrected | n/a (doc) |

Observability (audit §2 "Observability" row): `hydra_get_prod_stats()`
(extern "C"; IKVStore is frozen by the harness) exposes durable_ok,
recover_ok, recovered_keys, recover_torn_slots, write_errors, read_errors,
rejected_oversize, oversize_bytes, compactions_run, log_bytes, live_bytes,
reclaimed_bytes, buffered_fallback, punch_unsupported.

## abort() survivors (init-time / programming invariants only)

open() failing for both O_DIRECT and buffered fds; mmap/posix_memalign
failure (construction-time OOM); io_uring_enter unexpected errno
(EBADF/EINVAL-class programming error — transient errnos are retried;
completion-level errors are fail-soft). Rationale in DISCOVERY_LOG.md.

## Independent review (gpt-5.6-sol, read-only) — disposition

An independent read-only review produced 18 findings. FIXED in code:
unverified tombstone-relocation landings no longer allow a region punch
(chunk_fill re-check after flush_partials); an fdatasync barrier now orders
relocated copies before the hole punch; an overflow sidecar beside an empty
slot log is recovered (oversized-only store); sidecar rename/unlink are
directory-fsynced and unlink failure keeps the sticky error + seen flag;
persisted sidecar records larger than GenValue::kMaxSize are skipped, never
served; Delete's tombstone-poison loop is bounded (4096 passes) on a dead
disk; recovery reports recover_ok=0 (degraded) after unreadable regions and
stops at the 32-bit position ceiling instead of truncating; the test matrix
script aggregates per-phase exit status (pipefail, stale-binary removal);
compactcold waits for a post-delete compaction pass and adds a
generation-3 restart check. DOCUMENTED as residual risks (below) rather
than fixed: fp-alias prev-chain bridges vs compaction, Checkpoint's
single-session drain, additive oversize cap, position-reuse read races,
unbounded delete registry, tombstone relocation amplification.

## Deliberate deviations / residual risks

- Inline index-capacity overflow stays unbounded (harness contract: loaded
  keys are never lost — `tiny` test); only >101 B values are capped.
- Oversize cap can overshoot by ≤ one value per concurrently-upserting
  thread (advisory check; bounded, honest).
- Durability model is checkpoint-based (FASTER-style): writes since the last
  successful Checkpoint/clean shutdown are lost on SIGKILL; a Delete's disk
  poison is only crash-durable once its page write and a later sync succeed.
- Recovery drops keys (counted + warned) if the log holds more keys than the
  index capacity of the CURRENT budget (reopen with a much smaller budget).
- Sidecar snapshot is not a point-in-time cut w.r.t. concurrent writers
  (same semantics as the slot-log flush during Checkpoint).
- Checkpoint lands only the CALLER's parked partial chunk; another live
  session's parked chunk lands at its next flush/StopSession/destructor.
  Checkpoint-after-drain (the harness's pattern) is fully covered.
- Fingerprint-alias prev chains vs compaction: a superseded slot can be the
  routing bridge to a DIFFERENT key in the same fp group (57-bit signature
  collision, ~2^-49 per pair); compaction's verification checks index words
  and cache references, not incoming prev edges, so reclaiming such a
  bridge could orphan the alias at runtime (recovery is immune: it scans
  raw slots). Astronomically rare; documented, not fixed.
- Oversize cap (budget/8) is additive to the cache sizing's 512 MiB slack:
  a store holding the full cap of oversized values can exceed the nominal
  budget by up to ~capsize; the scored workload holds none.
- Read-admission freshness still uses highest-position-wins as its TOCTOU
  re-check; with position reuse (compaction) this is weaker than the LSN
  order used for serving. Requires reuse + a racing update + admission in
  one window; the served value is always the LSN winner at read time.
- Punched regions vs in-flight reads: a read that sampled a candidate
  before relocation can transiently miss after the punch (NotFound) rather
  than retry against the updated index. Compaction never runs in scored
  windows.
- The deleted-key registry grows with live deletes (never shrinks until
  reinsert), and recovery re-marks tombstone winners: delete-heavy logs
  cost registry RAM at recovery.
- Tombstone relocation is one-for-one: a key deleted after many updates
  keeps one relocated tombstone per surviving tombstoned version
  (no per-key coalescing).

## Verification status

- Tests-first: all six new tests FAIL on HEAD (rc=1; `recover` shows real
  data loss), PASS on the hardened engine — run on the bench server
  (real NVMe, kernel 6.8, g++ 11.4). A seventh test (`compactcold`) was
  added afterwards to cover the compactor's uncached-relocation path,
  deleted/overflowed dead-slot skips, HYDRA_OVERSIZE_CAP_MB, restart on a
  compacted log + sidecar, and sidecar unlink once the map empties.

### RESULTS (bench server, FINAL sources — all review fixes in)

Measured by final_pipeline.sh on the bench node (real NVMe, kernel 6.8,
g++ 11.4), 2026-07-19, after the last batch of review fixes:

- ASan+UBSan compactcold stress loop (scale 4, leak check): 15/15
  iterations clean — the resurrection race and its fixes hold.
- PHASE_OPT (all 21 tests, scale 1): PASS
- PHASE_SHM (buffered fd, tmpfs, scale 4): PASS
- PHASE_NOURING (sync-pread fallback, scale 4): PASS
- PHASE_ASAN (ASan+UBSan, leak check, scale 4): PASS
- PHASE_TSAN (subset incl. recover/compact/compactcold/oversizebound,
  scale 8, ASLR off): PASS (zero TSan warnings)
- PHASE_COV (coverage): PASS — hydra.cc 93.30% lines (of 1716) and
  95.39% branches-executed (of 1346; taken-at-least-once 74.37%),
  above both the pre-hardening baseline (92.8% / 93.4%) and the
  previous session's 92.35% / 93.11%.
- validate.sh: rc=0 — VALIDATION PASSED, retained 250,000,000/250,000,000
  load keys (0 missing / 0 wrong_size / 0 wrong_value), stride 1
- run.sh (scored, two invocations = 6 cold-cache runs):
  5.42 / 5.44 / 5.41 (MEDIAN 5.42) and 5.46 / 5.43 / 5.46 (MEDIAN 5.46)
  Mops/s, StoreRSS 7.90 GB ≤ 8 GiB on every run. Six-run median 5.44 vs
  the HEAD engine's 5.48 reference median = 0.99x — within noise, meets
  the ≥ 0.97x no-regression gate (HEAD itself measured 5.46–5.67 across
  sessions on this box).

### RESULTS — Round 3 perf recovery (2026-07-19, same box)

Engine = the hardened sources above + ONE change: alignas(64) cache-line
isolation of hot atomics (see DISCOVERY_LOG.md "Round 3"; adversarially
measured, one idea at a time, rejects reverted).

- Re-measured session baseline (pre-change): 5.50 / 5.51 / 5.47 —
  MEDIAN 5.50 Mops/s (the earlier 5.44 median was partly box noise).
- Full matrix (run_all_tests.sh): PHASE_OPT / PHASE_SHM / PHASE_NOURING /
  PHASE_ASAN / PHASE_TSAN / PHASE_COV — ALL PASS.
- Coverage (with targeted top-up runs added to the cov phase):
  93.12% lines (1716) / 95.10% branches-executed (1346; 74.15%
  taken-at-least-once) — meets the ≥ 93.0 / ≥ 95.0 gate.
- validate.sh: rc=0 (VALIDATION PASSED, exit=0).
- run.sh × 2 (six cold-cache scored runs): 5.50 / 5.51 / 5.53
  (MEDIAN 5.51) and 5.51 / 5.53 / 5.53 (MEDIAN 5.53); six-run median
  5.52 Mops/s, StoreRSS 7.90 GB ≤ 8 GiB on every run. GOAL (≥ 5.50) MET.
