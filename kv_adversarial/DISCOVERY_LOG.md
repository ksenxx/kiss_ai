# DISCOVERY_LOG — HydraKV production hardening

Adversarial-AI-discovery ideas log: every idea considered, what was measured,
kept/rejected, and why. Task: close every gap in HYDRA_PROD_AUDIT.md §7 while
keeping the scored hot path ≥ 5.5 Mops/s. Tests were written FIRST (all six
new e2e tests fail on HEAD: `recover`/`oversizebound`/`compact` exit rc=1
against HEAD hydra.cc + a stub stats seam; recover additionally shows real
data loss from O_TRUNC).

Research inputs (see the ten-source web-research notes, summarized):
RocksDB WAL format + WAL recovery modes (per-record CRC32C,
kTolerateCorruptedTailRecords), LevelDB log format (fixed-block resync),
Bitcask (append-only KV: checksum-verified recovery scan + merge compaction),
FASTER recovery docs (rebuild index by log scan; fold-over multi-version
growth), LWN "PostgreSQL's fsync() surprise" (sticky durability flags; never
trust a later fsync after a failure; dirty state stays in RAM), RocksDB
Background Error Handling (severity-tiered sticky bg errors; ENOSPC
auto-recovery; bounded in-RAM buffering while degraded), man fallocate(2)
(PUNCH_HOLE|KEEP_SIZE; EOPNOTSUPP handling), corsix.org fast CRC32C
(SSE4.2 crc32 instruction ≈ 21 bits/cycle), man io_uring_prep_read(3)
(negated-errno CQE res; short-read semantics).

## Slot layout chosen (idea #1 — KEPT)

    [0..7] key | [8..11] size+1 | [12..15] prev pos+1 |
    [16..116] value (zero-padded to 101) | [117..123] LSN (56-bit) |
    [124..127] CRC-32C over bytes [0..123]

- kSlotBytes stays 128. kSlotDataMax 112 → 101: the 11 freed bytes hold a
  7-byte LSN + 4-byte CRC. Values > 101 B never entered slots anyway (they
  go to the overflow map), so the scored 100-B workload layout is untouched.
- WHY AN LSN AND NOT "highest position wins": positions are NOT temporal —
  sessions pre-reserve 8192-slot extents, so a newer version of a key can
  land at a LOWER position than an older one (session A reserves [0..8191],
  B reserves [8192..], B lands k@9000 first, A lands k@100 later). A
  position-ordered recovery would resurrect stale data. The global staging
  LSN (56-bit; 228 years at 10 M writes/s) total-orders every slot version.
  This also FIXES audit LOW-8 (multi-root stale read): all read paths now
  take the max-LSN CRC-valid verified match instead of max position.
- CRC-32C: hardware `crc32` (SSE4.2, ~16 instructions per 124-B slot) with a
  table-driven software fallback producing identical values. Runtime reads
  CRC-check ONLY accepted key matches (one crc per served miss, noise vs a
  350 µs NVMe read); recovery CRC-checks every slot.

## Crash recovery (audit HIGH-1/HIGH-2 — KEPT)

- O_TRUNC removed from open(). Non-empty log ⇒ recover_log(): 1 MiB-buffered
  sequential scan; skip sz1==0 (hole/never-landed) and CRC-invalid (torn)
  slots; newest-LSN-wins per key into the index. Temporary per-word key/LSN
  arrays (2×8 B × nwords) are mmap'd during the scan and freed BEFORE the
  cache is allocated, keeping peak RSS inside the budget.
- Tombstones (out-of-range size, CRC-valid, fresh LSN — tombstone_slot now
  reseals) participate in newest-LSN-wins, so deleted keys STAY deleted
  across restart; the winning word points at an unmatchable slot, exactly
  the state a live Delete leaves.
- Oversized/overflow values persist via a sidecar (hydra_overflow.dat):
  [magic][count] + per-entry [key|lsn|size|crc32c|bytes], written at
  Checkpoint (and destructor) to a tmp file, fdatasync'd, atomically
  rename()d (a mid-write crash can never tear the previous sidecar). At
  recovery each entry's LSN is compared against the key's newest slot-log
  LSN, so oversized→inline and inline→oversized transitions recover to the
  true last write (covered by the `recover` test's tk1/tk2 cases).
- Recovery capacity overflow (log has more keys than the index can hold,
  e.g. reopened with a smaller budget): keys over capacity are DROPPED with
  a counted stderr warning. Alternative (spill them to the overflow map)
  rejected: unbounded RAM on a path whose purpose is bounding RAM.
- Harness interaction: validate.sh/run.sh `rm -rf $KV_SPILL/*` before every
  run, so recovery never runs in a scored window. Init on a fresh dir does
  one fstat (size 0 ⇒ no scan): hot path unchanged.

## Fail-soft I/O (audit HIGH-3, MED-5, LOW-11/12/13 — KEPT)

- xpwrite → bool; xpread → ssize_t (-1 on real error), both retry
  EINTR/EAGAIN with backoff. NO abort() remains on any runtime I/O path.
- flush_chunk on write failure: nothing is published; the chunk stays staged
  and every staged entry stays PINNED in cache with its value authoritative
  (the RAM copy is the source of truth — the fsyncgate lesson), retried on
  the next flush_chunk (cleaner/Checkpoint/StopSession/destructor/full-chunk
  trigger). Sticky write_errors_ + durable_ok_=false.
- Livelock bounds (found by reasoning through the ENOSPC test before it
  existed): a full-but-unlanded chunk refuses further staging; upsert_inline
  retries the landing ≤16 times, then absorbs the write in the overflow map
  (bounded by cache pressure, honest, never hangs); find_victim exhaustion
  falls back the same way after 64 attempts.
- Read path: read_page zero-fills + counts read_errors_ on error; a SHORT
  read below the landed high-water mark (landed_hw_, advanced only by
  successful landings) is counted as a fault (external truncation), while
  EOF beyond it is the normal pre-reserved-extent case. io_uring completions:
  -EINTR/-EAGAIN resubmit; other negatives count + abandon that chain only;
  mid-page short reads RESUBMIT THE REMAINDER (per-op `filled` cursor)
  instead of zero-filling bytes the device still owes (audit LOW-12).
- fdatasync failure (Checkpoint + destructor): sticky durable_ok_=false,
  never abort, never "retry and trust" (kernel may have dropped the pages).
- O_DIRECT→buffered fallback now logged at open and surfaced as
  buffered_fallback in prod stats (audit LOW-11).
- abort() SURVIVORS (all init-time/programming invariants, per task):
  index/cache/doorkeeper/recovery-scratch mmap failure and posix_memalign
  failure (OOM at construction), open() failing for BOTH O_DIRECT and
  buffered (no storage at all), and io_uring_enter returning an unexpected
  errno (EBADF/EINVAL-class programming error; transient EINTR/EAGAIN/EBUSY
  are retried). Completion-level io_uring errors are fail-soft.

## Bounded oversize (audit LOW-6 — KEPT)

- Overflow map entries now carry {bytes, lsn}; oversized entries are charged
  size+64 against oversize_cap_ (budget/8, env HYDRA_OVERSIZE_CAP_MB).
- Oversized Upsert checks the cap BEFORE touching any state: a rejected
  upsert cannot damage the key's current value (rejection-ordering bug
  avoided by design; the test overwrites and re-reads around rejections).
  Same-size overwrites of an existing oversized key are always accepted
  (delta-based check). rejected_oversize_ counts every refusal.
- Bounded overshoot: concurrent oversized upserts can pass the advisory
  check together and overshoot the cap by ≤ one value per in-flight thread.
  Accepted (bounded, honest) instead of a global oversize mutex.
- The INLINE index-capacity fallback stays uncharged/unbounded by design:
  the `tiny` e2e test (and the harness contract that loaded keys are never
  lost) requires absorbing index overflow; documented as a deliberate
  deviation — bounding it would trade a correctness contract for a cap.

## Log compaction (audit MED-4 + 512 GiB ceiling — KEPT)

- Cyclic background scavenger (compactor_main, 50 ms cadence): triggers when
  log_bytes > max(HYDRA_COMPACT_FLOOR_MB, HYDRA_COMPACT_FACTOR × live_bytes)
  (defaults 256 MiB / 2.0; log_bytes = allocated − reclaimed, live_bytes =
  index-entry count × 128).
- Region = 8192 slots (1 MiB). Regions overlapping session-owned or free
  extents are skipped (extent-ownership registry, one mutex touch per 8192
  slots — ~335 locks/s at full scored write rate, unmeasurable).
- Live-slot relocation reuses the EXISTING flush protocol end to end:
  cache-resident slots are stage_flush()ed (S_FLUSHING pin → landing CASes
  the index word old→new under the set lock); uncached live slots (max-LSN
  verified via disk walk) are admitted CLEAN (recency 0) then staged, with
  under-lock revalidation (presence/reachability/del-mark) after
  find_victim. Every existing delete/re-upsert interlock therefore applies
  to relocations with zero new landing states.
- After relocations land, a verification pass confirms NO index word and no
  cache entry still references the region; only then is it punched
  (FALLOC_FL_PUNCH_HOLE|KEEP_SIZE) and its position range pushed to the
  free-extent list that ensure_chunk() recycles (relaxed counter keeps the
  scored path lock-free while the list is empty). This is what removes the
  512 GiB / 32-bit position ceiling: position space is REUSED, so next_slot_
  stops growing once reclamation keeps pace.
- Punch failure (EOPNOTSUPP filesystems) is nonfatal and surfaced
  (punch_unsupported): extent reuse alone stops file growth, and ANY stale
  CRC-valid bytes that remain readable lose every LSN comparison — the LSN
  is the load-bearing safety property for position reuse (stale prev
  pointers into reused space verify-fail on key or lose on LSN; loops are
  hop-bounded).
- Durability across compaction: relocated copies carry FRESH LSNs and land
  before the region is punched, so recovery after a crash mid-compaction
  picks the relocated (or a newer) copy; a punched hole reads as zeros and
  is skipped by the CRC/zero checks.

## LOW fixes

- LOW-7 RMW-vs-Upsert lost update: RMW now has a cache-resident fast path
  doing read+modify+write entirely under the key's SET lock (same lock
  plain Upserts take) — closed for resident keys; non-resident keys fall
  back to the rmw_locks path (still atomic vs other RMWs) and become
  resident on first RMW. Residual (first-touch RMW vs concurrent blind
  Upsert) documented.
- LOW-8 multi-root stale read: FIXED by LSN-based selection (above).
- LOW-9 Delete global-mutex cliff: deleted-key registry sharded 64-way;
  the zero-deletes relaxed-load fast path is unchanged.
- LOW-10 delete/oversized liveness: bounded-backoff waits kept; under a
  permanently failing disk a Delete of a BUFFERED key can still wait
  indefinitely (1 ms-sleep polite loop) — erasing the pin instead would let
  a later successful landing resurrect pre-delete bytes. Documented as the
  honest trade (correctness over liveness on a dead disk).
- LOW-13 xpread EAGAIN: retried with backoff.
- LOW-14 doc drift: header now says "> 101 B".
- Observability: GetCacheStats hot_bytes now includes oversize bytes (audit
  §2 note); full prod surface via extern "C" hydra_get_prod_stats
  (IKVStore is frozen by the harness, so a C-linkage seam instead of a new
  virtual): durable_ok, recover_ok, recovered_keys, recover_torn_slots,
  write_errors, read_errors, rejected_oversize, oversize_bytes,
  compactions_run, log_bytes, live_bytes, reclaimed_bytes,
  buffered_fallback, punch_unsupported.

## Hot-path impact analysis (why ≥ 5.5 Mops/s holds)

- Scored run: fresh spill dir ⇒ no recovery; compaction never triggers
  (after load: log = live = 32 GB ⇒ ratio 1.0; +30 s of appends ≈ 42 GB vs
  2×32 GB threshold); error branches are never-taken predicted-not-taken
  branches; ensure_chunk's free-list check is one relaxed load of a counter
  that stays 0; extent registry = 2 mutex ops per 8192 staged slots.
- Real additions on the scored path: slot_seal (one 124-B hardware CRC + LSN
  fetch_add) per STAGED slot (≈16 crc32 instructions; staging already does
  a 128-B memcpy), one CRC + LSN read per served read-miss (≈16
  instructions vs a ~350 µs NVMe read), LSN compare instead of position
  compare. Measured result: see MEASURED RESULTS below.

## Rejected / deferred ideas

- Rebuild recovery via a persisted index snapshot (FASTER-style index
  checkpoint): rejected for now — doubles checkpoint I/O and adds a
  consistency protocol; full-scan recovery is simple, correct, and its cost
  (linear in log size, ~seconds/GB) is acceptable and Bitcask-precedented.
- Copy-to-new-file compaction with fd swap: rejected — two-file read
  indirection and an fd-swap race with in-flight io_uring reads; hole
  punching + extent reuse achieves the same reclamation in place.
- Charging the inline index-overflow fallback against the oversize cap:
  rejected (breaks the harness "never lose a loaded key" contract exercised
  by the `tiny` test).
- Reserve/rollback hard cap for oversize (exact, no overshoot): rejected in
  favor of the advisory check + bounded overshoot; the reserve dance added
  a failure path in which a rejected reservation could strand credits.
- CRC verification on EVERY chain hop (not just accepted matches): rejected;
  mismatched-key hops are only used to follow prev pointers, which are
  hop-bounded and position-validated; checking only accepted matches is
  what correctness needs.

## Adversarial discovery round 2: the compactcold test and the
## deleted-key resurrection bug (found AFTER the first full matrix passed)

- IDEA: the first matrix's coverage (90.1% lines) showed compact_region's
  UNCACHED-key path never ran — test_compact's 4096 keys always fit in
  cache. A new e2e test (`compactcold`, 8 MiB budget ⇒ ~35k cache entries,
  60k keys, deletes + oversized churn + restart) was written to force it.
- The test FAILED ~1-in-3 under ASan timing: deleted keys CAME BACK after
  restart with their last pre-delete value. Root cause (a genuine recovery
  bug, invisible to runtime reads): (1) a cleaner/compactor stages a copy
  of the key (sealed with a fresh LSN at stage time); (2) Delete marks,
  sweeps the S_FLUSHING pin, and tombstones every index-reachable slot;
  (3) the staged chunk lands PHYSICALLY (xpwrite precedes the meta loop)
  and the landing correctly ABANDONS the record (pin broken) — leaving a
  CRC-valid, live-looking, unlinked ORPHAN slot of the deleted key on
  disk (lower LSN than the tombstones, so recovery is still correct);
  (4) the compactor treated tombstones as reclaimable garbage — punching
  the regions holding the tombstones while the orphan's region survived
  flips the key's newest surviving LSN to the orphan ⇒ resurrection at
  the NEXT restart.
- FIXES (all off the scored path — no deletes and no compaction happen in
  a scored run):
  1. flush_chunk abandonment: if the pin was broken and is_deleted(key),
     reseal the just-landed orphan as a tombstone in place, under the set
     lock (Delete's own discipline: a concurrent reinsert cannot land a
     newer value with a LOWER LSN than the tombstone).
  2. compact_region: a still-deleted key's tombstone is RELOCATED
     (restage_tombstone: fresh meta-less tombstone slot, sealed under the
     key's set lock after re-checking the mark) instead of dropped; if it
     cannot be restaged the region stays unreclaimed. A deleted key's last
     durable tombstone can therefore never be punched while stale copies
     of the key may survive elsewhere in the log.
  3. recover_log: tombstone-winning keys (tracked in an rt scratch array,
     freed with rk/rl) are re-marked in the delete registry after
     load_overflow_file (keys whose overflow entry outlived the tombstone
     stay live), so fix #2's invariant holds across restart GENERATIONS.
- Also fixed in the same round: the test itself first asserted "deleting
  the 32 oversized keys empties the overflow map" — wrong, because the
  engine legitimately absorbs inline keys into the overflow map under pin
  pressure (documented last-resort path); the test now deletes every key
  before asserting the sidecar is unlinked.
- REJECTED alternative: wiping abandoned slots in the chunk buffer BEFORE
  xpwrite (a pre-pass under all set locks). Rejected: doubles set-lock
  traffic on every landing (cleaners run during scored windows) and still
  leaves the same window between the write and the pin re-check.
- Earlier in this round: matrix phases SHM/NOURING failed on
  "compaction reclaimed nothing" — reclaimed_bytes is NET (extent reuse
  decrements it), so a poll could legitimately read 0. Added cumulative
  reclaimed_total_ (never decremented) as the stats surface.

## Round 3 — perf recovery (raise scored median back to >= 5.5, stretch 7.0)

Research inputs (10-site pass, notes in session tmp): travisdowns concurrency
cost hierarchy (contended fetch_add ~100ns+, cost tracks the NUMBER of
contended RMWs; sharded/batched counters escape), cppreference+GCC docs
([[likely]] is C++20 — scored build is C++17, use __builtin_expect; cold/
noinline attributes), RocksDB write-group leader sequence allocation
(one range per batch group), corsix.org fast CRC32C (HW crc32 ~21 b/cycle:
a 124-B slot seal ≈ 48 cycles ≈ 15 ns — NOT a regression candidate),
PostgreSQL WAL-insert reservation (minimal critical section), Wikipedia
false sharing (~50x penalty for distinct bytes on one line; alignas(64)),
FASTER epoch protection (defer global coordination), Preshing TSO (relaxed
does not un-contend a fetch_add on x86; only fewer/sharded RMWs help),
kernel THP docs (MADV_HUGEPAGE for the big anonymous mmaps).

- BASELINE (this session, same box, same binary as PROD_READINESS RESULTS):
  5.50 / 5.51 / 5.47 => MEDIAN 5.50 Mops/s, StoreRSS 7.90 GB. The box
  measures ~1% hotter than the previous session (5.42-5.46); the previous
  "regression" is partly box noise. Gate target unchanged: >= 5.50 median.

### Idea R3-1: cache-line isolation of hot atomics (alignas(64)) — measuring
- Evidence from code reading: flush_epoch_ (LOADED per verified-lookup page
  check) was declared adjacent to occupancy_ (WRITTEN per admission);
  dk_cur_/dk_bits_/dk_mask_ (loaded per miss) adjacent to dk_marks_
  (written per miss); landed_hw_ (loaded per read_page) adjacent to lsn_
  (written per staged slot); del_active_ (loaded per admission/landing)
  and overflow_count_ (loaded per read) shared lines with neighbors;
  free_count_ shared a line with free_mu_/vectors. Classic false sharing.
- Change: alignas(64) on next_slot_; stage_ver_+lsn_ grouped on one
  write-hot line; landed_hw_, free_count_, dk_bits_ group, dk_marks_,
  overflow_count_, flush_epoch_, occupancy_, del_active_ each isolated.
  No semantic change; ten alignas(64) specifiers, +408 B object growth
  (sizeof(HydraStore) 81,000 -> 81,408 on g++ 11.4; 6.375 cache lines of
  padding — negligible vs the 512 MiB cache-sizing reserve).
- MEASURED: 5.53 / 5.51 / 5.49 => MEDIAN 5.51 (baseline 5.50), RSS 7.90 GB.
  Fast gate (10 correctness tests, opt, scale 1): all PASS. ACCEPTED —
  within noise but non-negative, and removes a real (code-verified) false
  sharing hazard.

### Idea R3-2: MADV_HUGEPAGE on index / entries cache / doorkeeper — measuring
- Server THP config: enabled=[madvise], defrag=[madvise] => the advisory
  actively promotes to 2 MiB pages. The index (~2 GiB), cache (~5 GiB) and
  doorkeeper (2x128 MiB) are anonymous, fully-populated, uniformly randomly
  probed on every op — 4 KiB pages mean ~2-3 TLB misses/op. Advisory-only
  (#ifdef MADV_HUGEPAGE), no RSS growth expected (regions fully touched).
- MEASURED: 5.52 / 5.51 / 5.51 => MEDIAN 5.51 — no improvement over R3-1's
  5.51 (spread tightened, but that is one sample). Engagement check
  (smaps_rollup 30 s into a manual load): AnonHugePages = 32 MiB of ~9 GB
  anonymous — the advisory barely engaged (fragmented box, khugepaged lag),
  which explains the null result. REJECTED / reverted: no measured win, and
  keeping unproven knobs violates the accept rule. (If a future box shows
  high dTLB-miss rates, retry with explicit hugepage-aligned mappings.)

### Idea R3-3: per-session/batched hot stat counters — REJECTED without
### measurement (evidence-based deprioritization)
- Code reading: dk_marks_.fetch_add fires only for NOT-seen keys (Zipf-hot
  traffic is mostly seen); the rotation threshold load runs only in
  cleaner 0; occupancy_ RMWs happen per admission/eviction (~0.6 M/s
  total across 16 workers ≈ well under 1% of worker CPU). dk_seen_and_mark's
  scored cost is the inherent random DRAM touch on 2x128 MiB bitmaps, not
  the counter. Expected value below measurement noise; not worth a run.

### Ideas considered but not implemented (logged so they are not retried
### blindly)
- Batched/sharded LSN allocation: per-session LSN blocks break per-key
  monotonicity (a session holding an older block can seal a NEWER version
  of a key with a LOWER LSN after crossing a set-lock happens-before edge);
  every provable repair (per-set last-LSN, resync-with-block-abandon on
  every set-lock acquisition) either adds memory (8 B x nsets) or
  degenerates back to one fetch_add per staging. The load-phase profile
  shows Upsert-path work (75%) dominated by cache/set-lock work, not lsn_;
  in the scored window staging happens at cleaner rates (~1 M slots/s
  shared by 4 cleaners), where one extra contended RMW is noise. Would
  revisit only with perf c2c evidence of lsn_ line contention.
- __builtin_expect on fail-soft branches: expected effect below the
  +-0.02 Mops/s run-to-run noise on this harness (error branches already
  compile to forward-not-taken layout at -O3); skipped to keep the diff
  minimal and every accepted change measurement-backed.
- CRC32C tuning: measured cost analysis (corsix.org numbers) puts a 124-B
  slot seal at ~48 cycles ≈ 15 ns, one per staged slot / served miss —
  orders below the regression size. Nothing to recover.

### Round 3 final state
- Accepted: R3-1 only (alignas(64) cache-line isolation; zero semantics).
- Scored: BASELINE 5.50 median -> FINAL median-of-3 5.51 (see final gate
  below for the six-run confirmation), RSS 7.90 GB <= 8 GiB.
- FINAL GATE (2026-07-19): matrix 6/6 PASS; coverage 93.12%/95.10%
  (74.15% taken-at-least-once) after targeted top-up cov runs;
  validate.sh rc=0; six scored runs 5.50/5.51/5.53 + 5.51/5.53/5.53 ->
  six-run MEDIAN 5.52 Mops/s, StoreRSS 7.90 GB every run. Goal >= 5.50 met.

## MEASURED RESULTS

- New e2e tests on HEAD (tests-first evidence): recover/oversizebound/
  compact rc=1 (stub stats seam; recover also loses all data to O_TRUNC).
- New e2e tests on the hardened engine (bench server, real NVMe): all six
  PASS at scale 1 — recover (233k keys rebuilt, 0 torn), crashrecover
  (30k/30k checkpointed keys exact after SIGKILL; ~200k post-ckpt keys
  survived; 0 corrupt values served), enospc (write_errors sticky, reads
  correct while disk full, full durability after repair + restart),
  readfault (read_errors>0, no abort, store usable), oversizebound (cap
  enforced ±4 KB, honest rejections, budget freed by shrink/delete),
  compact (compactions_run>0, st_blocks shrank, 4096/4096 keys exact under
  concurrent readers, survives restart on the holey log).
- Full 6-phase matrix + scored benchmark numbers: see PROD_READINESS.md.
  Final pipeline (final sources, 2026-07-19): ASan compactcold loop 15/15
  clean; matrix4 all six phases PASS; coverage 93.30% lines / 95.39%
  branches-executed (74.37% taken-at-least-once); validate.sh rc=0
  (250,000,000/250,000,000 retained); scored runs 5.42/5.44/5.41
  (median 5.42) + 5.46/5.43/5.46 (median 5.46) Mops/s, StoreRSS 7.90 GB
  on every run; six-run median 5.44 = 0.99x HEAD's 5.48 (gate >= 0.97x).
