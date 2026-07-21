# HydraKV Multi-Workload Hardening (July 2026)

Follow-up to `AUDIT2_FIXES.md`. Trigger: user feedback that real-world
deployment is the only true test, plus a request to validate the exact
benchmark mixes they use (0:100 and 5:95 read:write YCSB variants, same
Zipf 0.95 skew). Goal: run every workload/program path end-to-end on the
reference node, fix every bug found, and keep the scored A_50_50 result
at or above 5.5 Mops/s.

Development model: claude-fable-5. Review model: gpt-5.6-sol (read-only,
~12% of task budget across 3 review rounds; cap 20%).

## What was tested (reference node, 64 vCPU, 8x NVMe RAID0, 8 GiB budget)

Scored protocol (run_wl.sh = same cgroup/NUMA/auditor as run.sh) over the
250M-key Zipf-0.95 traces, 16 threads, 30 s, integrity auditor ON:

| Workload (id) | Result (Mops/s) | Integrity |
|--------------------------|-----------------|-----------|
| A_50_50 (0, SCORED) | see below | 0 fail |
| RMW_100 (1) | 0.30 | 0 fail |
| B_95_5 (2) | 2.93 | 0 fail |
| C_100_0 (3) | 2.78 | 0 fail |
| W_0_100 (4, user's) | 5.09–5.13 | 0 fail |
| TIMESERIES_HD (5) | 0.08 | 0 fail (1.28M deletes: 0 resurrections, 0 lost keys) |
| W95_5_WRITE (6, user's) | 5.17–5.24 | 0 fail |

Environment variants: sync path (ASYNC_EVAL=0), no-uring, 1/32 threads,
VALUE_SIZE=1024/4096, BIMODAL_VALUES=1 (20 B/200 B), 2 GiB budget, tmpfs,
buffered-fd fallback, plus the unit matrix (opt/ASan+UBSan/TSan/coverage,
scales 1–8) and crash/restart recovery in every mode.

## Bugs found and fixed (chronological)

### 1. Oversized values (>101 B inline max) had NO disk path

VALUE_SIZE=1024/4096 showed 100% stale reads; bimodal showed millions of
false NotFounds: values that did not fit an inline slot lived only in a
budget/8-capped RAM overflow map with silent "honest rejection" past the
cap. FIX: disk-backed multi-slot x-records in the main slot log
(32 B header [key|sz|prev|lsn|dcrc|hcrc] + data, \<=33 slots, extent-
aligned, written through a buffered fd2\_ on the same file, linked into the
fingerprint index under the set lock, LSN-ordered against inline
versions). Memory-neutral; recovery is length-aware and restores them.

### 2. ext4 O_DIRECT EINVAL at the log's unaligned EOF tail page

Buffered x-appends leave i_size unaligned; O_DIRECT preads crossing that
EOF fail EINVAL, and each fault made the chain walk abandon -> false
NotFound / stale read (the 1:1 read_err==nf_walk signature). FIX: a 64 KiB
atomic x-extent bitmap (marked at reservation, before any record is
reachable; never cleared) routes every x-extent page read through fd2\_
(read_page + all four io_uring prep sites); coherency backstops re-gated
on is_x_page; compaction refuses x extents via the bitmap.

### 3. Clean-shutdown data loss (pre-existing, all workloads)

The destructor flushed only parked partial chunks; S_DIRTY cache entries
were silently reverted on reopen (test_bimodal restart verify failed
166K/300K even on the OLD engine). FIX: destructor lands all sessions'
partials, then runs a full Checkpoint (dirty sweep, delete-reaper drain,
fdatasync, sidecar).

### 4. Compactor could punch a just-reserved extent (review find)

next_slot\_ advance ran before ownership registration. FIX: reservation is
one free_mu\_ critical section (advance + owned push + x-bit mark) on both
the inline and x paths.

### 5. Recovery scan dropped 1 MiB regions after a crash (review find)

The scan read via O_DIRECT and a post-crash unaligned i_size EINVALs
silently skipped whole regions. FIX: recovery scans through fd2\_.

### 6. Destructor Checkpoint could std::terminate (review find)

32 unguarded std::thread spawns. FIX: try/catch; unspawned stripes run
synchronously.

### 7. Stale-position tombstones could silently delete an INNOCENT key

(TSan compactcold scale-8 signature: first x-record NotFound after
restart, pre-restart reads fine — the tombstone only reseals the header,
so dcrc-gated in-process reads still pass; recovery then sees the
tombstone as the key's newest LSN.) A position can go stale between
poison's verified_lookup (outside the lock) and the tombstone write
(compactor relocation + extent recycle). FIX: `tombstone_slot(pos, key)`
verifies the STORED key before resealing in both branches, requires a
live-parsed size in the inline branch (a punched page reads back zeros —
key 0 would false-match), and refuses whole-page O_DIRECT RMW on x pages
(would clobber concurrent buffered x-appends). Refusals are counted
(`tomb_ref` in HYDRA_STATS). Refusal is safe: the poison loop re-runs
verified_lookup, which cannot return a recycled position for the key.

### 8. Compaction could strand a deleted key resurrectable (review find)

compact_region skipped a still-deleted key's LIVE slots without restaging
a tombstone; a same-fp key's fp-word takeover can strand the deleted
key's copies with no routing word (so the reaper found nothing to
poison), and after the punch an older CRC-valid copy elsewhere becomes
the key's highest-LSN survivor -> resurrected at recovery. FIX: restage a
fresh tombstone (newest LSN) for is_deleted keys before reclaiming, same
protocol as the existing tombstone-relocation case.

## Architectural limits (documented, loud, not silent)

- RAM-index floor: ~8 B of budget per distinct key (index gets budget/4,
  ~97% fill hard cap). 250M keys at a 2 GiB budget CANNOT fit (~6.6M
  honest rejections, counted in prod stats); the user's 8 GiB benchmark
  budget fits. InitExtended now emits a LOUD warning at init time when
  the hash_table_size hint exceeds the index capacity, including the
  minimum budget that would fit (new `capwarn` e2e test).
- TIMESERIES delete throughput (0.08 Mops/s) is poison-I/O bound;
  correctness is clean (0 resurrections / 0 lost live keys at 1.28M
  deletes). Optimization deferred deliberately.
- RMW_100 (0.30 Mops/s) is a synchronous per-op read-modify-write;
  latency-bound by design.
- X-record extents are never compacted (documented leak class, same as
  poisoned slots).

## New regression tests (test_hydra.cc, all self-asserting e2e)

- `bimodal`: 300K mixed 20 B/200 B keys, 4 writers vs 4 readers (sync +
  async), tiny budget, forced compaction, full verify + restart verify;
  coverage asserts for sync/async x inline/x-record read paths.
- `oversizebound` (rewritten): 30K x 4 KB > whole budget: zero
  rejections, cross-session visibility, delete/RMW, restart, inline\<->
  oversized flapping LWW.
- `capwarn`: dup2-captured stderr; warn and no-warn directions.
- compactcold extended for x-records (no sidecar for x-resident values,
  gen-3 restart keeps every deleted key deleted).

## Verification (final engine, commit 5050e98f)

- Local (macOS arm64): all 27 tests pass, scale 1.
- Node: TSan compact loop 15/15 clean at commit 4138924c (the previously
  flaky combo) and re-run at 5050e98f; full matrix
  (opt/shm/no-uring/ASan+UBSan/TSan subset/coverage); scored run.sh
  median-of-3; bimodal/v1k/0:100/5:95/timeseries spot checks with
  HYDRA_STATS — see `refnode_workloads_jul21/` for the evidence logs.
- Scored A_50_50 history on this engine line: 5.53 median (chain3),
  5.50 (chain4, equals the pre-fix A/B control), chain6 = final engine.

Hot-path safety: all fixes live on Delete/compaction/oversized/recovery
paths; the scored workload (100 B values, no deletes) executes none of
them (statically confirmed in review round 3), so the 5.5+ Mops/s
result is preserved by construction and re-measured on the node.
