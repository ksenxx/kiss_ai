# PROGRESS — HydraKV multi-workload end-to-end hardening (Task 3)

Goal: test HydraKV (projects/kv_adversarial/hydra.cc) end-to-end on ALL harness
workloads on the reference node (ksen@34.70.16.69, /mnt/ssd/ksen/kv50-benchmark/task),
fix every bug, keep scored A_50_50 ≥ 5.5 Mops/s. Dev: claude-fable-5;
review: gpt-5.6-sol (≤20% budget; ~8% used across sessions 1–3).

## DONE (sessions 1–3)

### Session 1: x-record subsystem (commits 62867299 + c7eb4e04)
- Workload sweep found oversized values (>101 B) had NO disk path → 100% stale
  at VALUE_SIZE=1024/4096, millions of false NotFounds on bimodal.
- Implemented disk-backed multi-slot x-records + first review fixes.

### Session 2: two more root causes (commit 0c61b4ae)
1. ext4 O_DIRECT EINVAL on unaligned EOF tail page → x-extent bitmap routes
   x-page reads through buffered fd2_ (read_page + 4 uring sites), backstops
   re-gated, compaction refuses x extents.
2. Clean-shutdown data loss (pre-existing): destructor now lands all
   sessions' partials then runs full Checkpoint(). New test_bimodal.

### Session 3 (this): node verification + second gpt-5.6-sol review (commit eade81bf)
- chain3 (engine 0c61b4ae) on node: ALL CLEAN —
  - bimodal: rc=0, 0 integrity failures, nf_walk=0 (was ~1850 false NF).
  - v1k/v4k (25M keys): rc=0, 0 failures, read_err=0 (was 100% stale).
  - **SCORED: 5.51 / 5.53 / 5.53 → MEDIAN 5.53 Mops/s ≥ 5.5** (StoreRSS ok).
  - Sweep: w0100 5.12, w9505 5.24, w0100_sync 5.14, t1 2.40, t32 5.04,
    b955 2.93, c1000 2.78, rmw 0.30, ts 0.08 — all rc=0, 0 audit failures.
- gpt-5.6-sol review of 0c61b4ae (./tmp/review_session2.md) found 3 real
  issues, ALL FIXED in eade81bf (hydra.cc md5 8ea890a3, test md5 8141e8bd):
  1. CRITICAL: extent reservation raced compactor (next_slot_ fetch_add before
     ownership registration → compact_region could pass owned/free/bitmap
     gates on a just-reserved extent and punch it). Fix: reservation is ONE
     free_mu_ critical section (advance + owned push + x-bit mark) on BOTH
     the inline (ensure_chunk) and x (xrec_upsert) paths.
  2. MAJOR: recover_log scanned via O_DIRECT fd_ → post-crash unaligned
     i_size EINVALs silently dropped whole 1 MiB scan regions. Fix: recovery
     scan reads through buffered fd2_.
  3. MAJOR: destructor→Checkpoint() spawns 32 std::threads with no exception
     handling → std::terminate on clean shutdown under RLIMIT_NPROC/memory
     pressure. Fix: checkpoint_stripe() helper + try/catch spawn; unspawned
     stripes run synchronously in the caller.
  - MINOR: HYDRA_STATS xrec= now prints exact diag_xlink_ok_ census (gate
    xrec_count_ double-counted); test_bimodal hardened (min-read floor,
    sync/async × x/inline coverage asserts, forced-compaction stress
    HYDRA_COMPACT_FLOOR_MB=8/FACTOR=1.1 at store creation, N clamp).
- Local suite with eade81bf: ALL 27 TESTS PASS (scale 1, macOS).

## chain4 RESULTS (engine eade81bf) — collected
- bim4: rc=0, 0 failures, nf_walk=0, read_err=91 (benign fail-soft).
- v1k4: rc=0, 0 failures, read_err=0.
- **SCORED: 5.47 / 5.50 / 5.52 → MEDIAN 5.50 Mops/s** — meets "not below
  5.5"; identical to the July-21 pre-fix A/B control median (machine noise).
- w0100 5.09, w9505 5.17 — clean.
- Matrix: OPT/SHM/NOURING/ASAN/COV PASS (cov 89.47% lines); **TSAN "FAIL"
  = 0 data-race warnings, but 2 assertion failures in phase_tsan_compact**.

## OPEN BUG #1 — compactcold scale 8 TSan: k=0 x-record NotFound after restart
- Signature (reproduced ~1/2–1/7 with filter "compact", TSan, scale 8, node):
  `compactcold BAD k=0 del=0 ovsz=1 ok=0` after restart + consequent Delete
  fail. ALWAYS k=0 = FIRST x-record at its extent base. Pre-restart verify
  passes. May PREDATE session-3 fixes (combo never ran on x-record engines).
- Ruled out: xrec_read_data padding (reads hdr+size only; 32×17 slots = tail
  page-aligned here), bitmap indexing, landed_hw_ short-read classification,
  poison of undeleted keys via verified_lookup (key-matched).
- PRIME SUSPECT: `tombstone_slot` x-head branch tombstones the record at the
  given position WITHOUT verifying its stored key, resealing with a FRESH
  newest LSN → a mis-targeted position (stale chain pos?) would make
  recovery keep k=0 "deleted" → NotFound. Matches signature exactly.
- Instrumentation IN FLIGHT: commit 6f74578d adds TEMP env-gated
  HYDRA_DEBUG_KEY diagnostics (recover_log: per-record pos/lsn/tomb/x/sz1 +
  placement; tombstone_slot: x-tomb key/pos). Node: unit_test_area has
  hydra_dbg.cc + test_tsan_dbg; dbgloop.sh (launched, pid ~303150) loops
  `HYDRA_DEBUG_KEY=0 ./test_tsan_dbg /mnt/ssd/ksen/kv_ut compact 8` up to
  15×, appending to unit_test_area/dbg_compact.log, stops at first rc!=0.
  → NEXT SESSION: read dbg_compact.log failing trace. If a "DBG xtomb key=0"
  line appears → find the caller passing the wrong position (likely chain
  walk staleness in poison/flush-abandon path) and fix by verifying the
  header key == deleted key inside tombstone_slot (cheap, root-cause-safe).
  If recovery trace shows the x record present but losing to a higher-LSN
  tombstone → same fix. If record absent → look at shutdown/compaction.
  STRIP the TEMP diagnostics (revert 6f74578d hunks) before delivery.

## OPEN ISSUE #2 — w9505_2g (2 GiB budget, 250M keys): architectural
- 6.6M false NotFounds + dropped upserts: fingerprint index is budget/4
  bytes → ~65M entries at 2 GiB << 250M keys. RAM-index floor ≈ 8 B/key
  (needs ≥ ~8 GiB budget for 250M keys). NOT a regression; my own extra
  stress config (user's benchmark budget is 8 GiB — passes).
- Remediation planned: LOUD init-time warning when hash_table_size hint ×
  ~2 keys exceeds index cap for the given budget (InitExtended has both
  numbers), + document the floor in WORKLOAD_HARDENING.md. No silent drops:
  live-path rejections already counted (rej_mem/rejected stats).

## SESSION 4 (commit 4138924c, hydra.cc md5 c75ae033, test md5 a722011d)
- dbgloop (15× instrumented TSan compact runs) did NOT reproduce the k=0
  failure (0/15; previously ~1/2–1/7 un-instrumented) — timing-sensitive.
- BUG #1 fix implemented anyway (only identified mechanism, root-cause-safe):
  tombstone_slot(position, KEY, se) now VERIFIES the stored key before
  resealing in BOTH branches (x-head: header key under page lock; inline:
  slot key under page lock) and REFUSES the whole-page O_DIRECT RMW when
  the position lies in an x extent (is_x_page) — a stale position between
  verified_lookup and the tombstone write (compactor relocation + extent
  recycle) could otherwise reseal an INNOCENT key's newest record with a
  fresh highest LSN → silent deletion at recovery. Callers pass rec.key
  (flush_chunk ×2) / key (poison_versions). Refusal is safe: the poison
  loop re-runs verified_lookup which can't return the recycled position.
- TEMP HYDRA_DEBUG_KEY diagnostics (6f74578d) stripped.
- ISSUE #2: InitExtended(hash_table_size,...) now warns LOUDLY when the
  hint exceeds index_.cap (prints cap, budget, min budget that fits).
  New e2e test capwarn (dup2-captured stderr, warn + no-warn directions).
- Local: ALL 27 TESTS PASS (scale 1, macOS).
- chain5 launched on node (06:26 UTC): TSan compact ×15 → full matrix →
  scored run.sh (KV_SPILL=/mnt/ssd/ksen/kv_spill) → bim5/v1k5 (+stats),
  w0100_5, w9505_5. Results: task/chain5.out, matrix5.out, scored_fix5.out.

## chain5 RESULTS (engine 4138924c) — ALL GREEN
- TSan compact loop 15/15 rc=0, zero TSan warnings (previously flaky combo).
- Matrix ALL_PHASES_DONE. SCORED 5.54/5.49/5.52 → MEDIAN 5.52 ≥ 5.5 ✓.
- bim5 rc=0 (nf_walk=0, read_err=83 benign), v1k5 rc=0 (read_err=0),
  w0100_5 5.06, w9505_5 5.13 — 0 integrity failures everywhere.

## gpt-5.6-sol review round 3 (session-4 diff; ~12% total spend, cap 20%)
- Verdict: fix complete for the defect class, score-safe, no new criticals.
- Findings ALL IMPLEMENTED in commit 5050e98f (hydra md5 935b0023,
  test md5 2261e783): M1 compact_region restages tombstones for
  still-deleted keys' LIVE slots before reclaiming (fp-word takeover
  resurrection hole, pre-existing); M2 inline tombstone refusal also
  requires live-parsed size (key==0 false-match on punched zero pages);
  tomb_refused_ counter (tomb_ref in HYDRA_STATS); need_words wrap guard;
  capwarn comment fix. Local suite: ALL TESTS PASS.
- WORKLOAD_HARDENING.md written (full bug list, limits, verification).

## FINAL VERIFICATION — ALL GREEN (engine 5050e98f)
- chain6 (unit phases, correct sources): TSan compact 15/15 rc=0,
  matrix rc=0 ALL_PHASES_DONE (89.30% lines executed).
- chain6 Phase 3/4 had used a STALE kvstore_bench (run.sh never rebuilds;
  binary was from 05:02) → ran ./build.sh explicitly and chain7 re-ran
  scored + workloads on the fresh binary (tomb_ref= in stats proves it):
  **SCORED 5.51/5.52/5.49 → MEDIAN 5.51 Mops/s ≥ 5.5 ✓**; bim7 rc=0
  (nf_walk=0, tomb_ref=0, read_err=103 benign), v1k7 rc=0 (read_err=0),
  w0100_7 5.12, w9505_7 5.16, ts7 0.08 — 0 integrity failures everywhere.
- Evidence committed: refnode_workloads_jul21/ (chain5/6/7.out,
  scored_fix7.out, matrix5/6.out, README). AUDIT2_FIXES.md pointer added.
- TASK COMPLETE. Node strays and ./tmp cleaned.
3. Write projects/kv_adversarial/WORKLOAD_HARDENING.md + evidence dir with
   chain3/4/5 outputs (scored_fix3/4/5.out, matrix4/5.out); update
   AUDIT2_FIXES.md pointer; commit.
4. Clean ./tmp (chain5.sh, full_suite_s4.log, test_opt, hydra_tests, older
   session leftovers) and node strays (kv_ut, kv_spill_wl, staging_s3,
   hydra_dbg.cc, test_tsan_dbg, test_opt_fix, tsan_*repro.log, dbg_*.log,
   tsan5_*.log after harvesting).
NOTE: scored protocol reference: task/run_wl.sh + KV_SPILL=/mnt/ssd/ksen/kv_spill
for ./run.sh. Node layout: /mnt/ssd/ksen/kv50-benchmark/task.
