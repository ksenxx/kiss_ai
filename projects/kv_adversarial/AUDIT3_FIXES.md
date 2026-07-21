# AUDIT3: July-21 external audit — delete crash-durability regression

Feedback source: github.com/shubham3-ucb/baselines-kiss-sorcar, branch
`hydra-audit`, `July_21/` (4 repro programs: `minimal.cc`, `resur.cc`,
`rein.cc`, `dcost.cc`). The auditor confirmed the four July-20 bugs fixed and
the score preserved, but found that the async-delete fix (deferred reaper
tombstoning) had introduced a new regression class.

Development model: `claude-fable-5`. Read-only review + debugging model:
`gpt-5.6-sol` (2 rounds for this audit; ~2% of task budget, 20% cap).

## Bugs found

* **Bug 5 — Delete crash-resurrection (CRITICAL, confirmed).** `Delete()`
  returned after marking the RAM registry and enqueueing the key for the
  background reaper. A SIGKILL before the reaper / next `Checkpoint()` wrote
  the on-disk tombstones let recovery resurrect the key's durably
  checkpointed versions (audit: 60K of 97K keys; minimal repro 5000/5000).
  Facets: inline slots, oversized x-records, overflow-sidecar values.
* **Bug 6 — Checkpoint doesn't persist other active sessions' parked
  chunks.** Entries staged `S_BUFFERED`/`S_FLUSHING` in ANOTHER session's
  un-landed chunk are single-owner and cannot be persisted by the caller.
* **Bug 7 — delete-registry RAM uncharged.** The `del_shards_` mark registry
  grew unbounded and uncounted.

## Fixes

### Bug 5: delete-intent log ("dlog", `hydra_dlog.dat`)

`Delete()` appends a 24-byte CRC-protected intent `[key|lsn|crc32c|pad]`
through a buffered fd before returning: the record reaches the kernel page
cache, so a **process crash** (SIGKILL — the audit's scenario) cannot lose
it; `Checkpoint()`'s `fdatasync` extends it to power loss, matching the
writes' durability contract. Recovery replays intents newest-LSN-wins
(erasing stale sidecar values, deferring durable re-tombstoning to the
reaper via marks + direct enqueue), and advances `lsn_` past every intent.
Cost: ~1–2 µs per Delete (audit repro `dcost`: Delete = 1.0 µs = 1× Upsert,
vs the pre-July-20 synchronous poison at 244 µs = 160×).

`Checkpoint()` truncates the dlog only under a conservative retirement
proof; otherwise it `fdatasync`s the dlog so pending intents survive power
loss. The gate (all must hold):

1. no intent appended since the pre-drain snapshot, reaper queue empty,
   `reap_inflight_ == 0` (covers queue-full/append-failure **synchronous**
   poisons, registered in-flight before Delete drops its locks);
2. `fdatasync(fd_)` succeeded, sticky `durable_ok_` still true, and
   **lifetime-zero** `write_errors_`, `tomb_refused_`, `poison_fail_`
   (per-Checkpoint deltas were provably insufficient: a poison failure
   consumed *between* Checkpoints must also forbid retirement forever);
3. no `del_unmark` since before the stripe scan (a reinsert racing the
   Checkpoint can no-op a queued poison while its own S_DIRTY value is not
   yet durable — the intent must then outlive this Checkpoint);
4. `unlanded_chunks_ == 0` (no session holds staged-but-unlanded chunk
   data);
5. this instance recovered cleanly: no dropped keys, no skipped unreadable
   slot regions (`recover_clean_`), dlog replay not degraded
   (`dlog_degraded_`: read failures, short reads, any CRC-invalid full
   record, `fstat` failure — the last also disables the dlog so appends
   cannot clobber offset 0).

Delete's failure ladder: dlog append failure → synchronous poison for landed
keys (registered in `reap_inflight_`) or a synchronous sidecar rewrite for
overflow-only keys; every failure path is counted and permanently blocks
truncation (fail-soft, surfaced by sticky flags + stats).

**Delete linearization (review round 2, R2-C1):** the entire delete — mark,
intent-LSN allocation, overflow + cache sweep, dlog append, reaper enqueue —
now runs under the key's **set lock**, the same lock every Upsert path
publishes and `del_unmark`s under. Previously an Upsert could publish V1 and
clear the mark between Delete's mark and sweep: Delete then swept V1, poison
no-op'ed, and the key's OLD checkpointed version resurfaced at runtime while
recovery honored the intent (non-linearizable). Lock order
`set → del-shard/overflow-shard → dlog_mu_ → reap_mu_` is cycle-free.

### Bug 6

`checkpoint_stripe` counts parked `S_BUFFERED`/`S_FLUSHING` entries into
`ckpt_unflushed_`; `Checkpoint()` prints a loud actionable warning. Full
mid-run chunk stealing was rejected (single-owner chunks; hot-path/deadlock
risk); the documented contract is: quiesce sessions for full-coverage
snapshots. The `unlanded_chunks_` counter (gate item 4) makes the *delete*
correctness independent of this contract.

### Bug 7

Every mark charges `kDelMarkCharge` (64 B) into `del_bytes_`, which shares
the misc-RAM pool with the overflow map (backpressures overflow admissions)
and is included in `GetCacheStats().hot_bytes`; crossing the cap warns
loudly once. Marks themselves are never trimmed — they are load-bearing for
correctness (read gating, compaction tombstone restaging).

## Reviews (gpt-5.6-sol, read-only)

* **Round 1** (on the first dlog implementation): 7 critical, 5 major, 2
  minor — all implemented (truncation reordering after sidecar persistence,
  streamed replay, dlog-only-restart recovery + LSN advance, init-time
  direct enqueue instead of pre-cache poisoning, early intent LSN,
  `HYDRA_REAP_PAUSE` test hook, dir fsync at dlog creation, …).
* **Round 2** (on the round-1 fixes): found the R2-C1 linearization race,
  the R2-C2 reinsert-vs-Checkpoint truncation hole, the R2-C7 invisible
  synchronous poison, delta-vs-lifetime gate holes (R2-C3), uncounted
  `sync_dir` (R2-C4), unreadable-region recovery vs truncation (R2-C5),
  overflow-only append-failure ack (R2-C6), dlog `fstat`/final-CRC handling
  (R2-M2), replay memory-bound bypass (R2-M1) — all implemented as above.
  The review explicitly cleared: lock ordering (no cycles),
  `unlanded_chunks_` increment/decrement pairing on all paths, the
  `HYDRA_REAP_PAUSE` semantics, and the scored hot path (Read/Upsert never
  touch `dlog_mu_`; scored 50:50 is delete-free).

## Documented residuals (honest limits, all surfaced loudly)

* Delete durability degrades fail-soft, never fail-stop: after any
  counted I/O fault the dlog is retained forever (fdatasync'd each
  Checkpoint) and sticky flags/stats report it; acknowledged deletes whose
  *every* fallback also failed (dead disk) are only process-crash safe.
* Recovery replay queues intents past the 64K runtime reaper bound
  (~72 B/distinct key), loudly warned; requires crashing inside a massive
  uncheckpointed delete storm.
* `Checkpoint()` has no snapshot barrier for values (contract: quiesce
  sessions); delete correctness does not depend on it (gate items 1–4).
* All Deletes serialize on `dlog_mu_` for one 24-byte buffered pwrite
  (~1 µs); the scored Read/Upsert hot path is untouched.

## Verification

* All 4 audit repros clean on the reference node: `minimal` 0/5000
  resurrected, `resur` 0/97K, `rein` 200000/200000 + 0 lost in the
  16-thread flap, `dcost` Delete 1.0 µs (1× Upsert).
* New permanent test `delcrash` (T28): fork+exec SIGKILL children, 2 crash
  generations, `HYDRA_REAP_PAUSE` (recovery provably dlog-dependent, not
  reaper luck), inline + oversized facets, checkpointed reinserts,
  oversized reinsert-after-post-ckpt-delete (LSN-beats-intent),
  reinsert-racing-Checkpoint (R2-C2: recovers as the reinsert or NotFound,
  never the pre-delete value), dlog truncation by a quiescent Checkpoint,
  reinsert-LSN ordering across a clean restart.
* Full local suite (28 tests) passes; node: scored median unchanged
  (5.50 Mops/s pre-fix chain9 vs 5.51 with the final engine, machine noise
  band), workload spots w0100/w9505/ts8 clean, unit tests at scale 4, TSan
  delcrash/updel loop ×10 clean (see `refnode_workloads_jul21/` and
  chain10 logs).
