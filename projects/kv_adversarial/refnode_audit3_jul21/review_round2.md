# HydraKV delete-intent log — round-2 read-only review

**Reviewer model:** `gpt-5.6-sol`  
**Reviewed:** current `projects/kv_adversarial/hydra.cc`, current `test_hydra.cc`, `tmp/review_bug5.md`, `tmp/round1_fixes.patch`, and `tmp/test_hardening.patch`  
**Verdict: REQUEST_CHANGES**

The happy-path Bug-5 reproduction is materially improved: a successful `Delete()` appends an intent before returning, a valid dlog-only restart now invokes recovery and advances the LSN, replay no longer calls cache-dependent poisoning before cache construction, overflow persistence is attempted before dlog retirement, and `HYDRA_REAP_PAUSE` makes the main crash test genuinely dependent on replay rather than reaper luck.

Approval is nevertheless unsafe. Several round-1 critical findings are only partially fixed, and the new global truncation proof has additional holes. In particular:

1. `Delete()` and Upsert publication are still not serialized as one per-key generation, so an Upsert can remove the delete mark before `Delete()` sweeps; the completed Delete can immediately expose an old value at runtime while recovery applies the intent.
2. Checkpoint can retire an intent after a concurrent reinsert has made the reaper no-op even though that reinsert was created after its stripe was scanned and is only `S_DIRTY`, hence not durable and not covered by `unlanded_chunks_`.
3. A failed poison consumed before the current Checkpoint is forgotten. The gate compares only error-counter deltas, does not check sticky `durable_ok_`, and poisoning has no success result or retry state.
4. Sidecar `rename`/`unlink` directory-fsync failure is not reflected in `write_errors_`, so Checkpoint can truncate the only intent protecting an overflow-only delete.
5. Recovery scan errors set `recover_ok=0` but do not block dlog truncation; an intent for a key in a temporarily unreadable region can be destroyed and that key can resurrect on a later healthy reopen.
6. Dlog append failure still permits an acknowledged Delete with no recoverable evidence, especially for overflow-only keys.
7. Recovery bypasses the 64K reaper bound and the delete registry remains unbounded, so the memory-hardening claims are not met.

No source files were modified during this review.

---

## Round-1 item verification table

| Item | Status | Verification against current code |
|---|---|---|
| **C1 — sidecar delete must be durable before dlog retirement** | **PARTIAL / blocking** | The ordering is improved: `write_overflow_file()` now runs before the truncation block (`hydra.cc:3607-3612`), and most fopen/write/fdatasync/rename/unlink failures increment `write_errors_` (`3879-3933`). However, `sync_dir()` returns `void` and only clears `durable_ok_`; it does **not** increment `write_errors_` (`3935-3945`). The truncation predicate checks only current error-counter deltas, not `durable_ok_` (`3639-3648`). Thus a successful rename/unlink followed by failed directory fsync can still be followed by dlog truncation and power-loss resurrection. C1 is not completely fixed. |
| **C2 — quiescent reaper must imply successful, durable tombstones** | **NOT FIXED** | `tombstone_slot()` still returns `void` (`1523-1618`), `poison_versions()` still returns `void` and consumes work regardless of success (`3273-3293`), and `reap_drain()` always pops and discards the item (`3435-3453`). Error snapshots at `3550-3552` detect only failures occurring during that one Checkpoint; a failure consumed earlier is ignored at `3639-3646`. Sticky `durable_ok_` is not part of `safe`. There is also an uncounted second x-header `xpread` failure at `1546-1571`. |
| **C3 — foreign staged chunks / Bug 6** | **PARTIAL** | The dlog-retirement facet is substantially improved. All three chunk-staging paths increment `unlanded_chunks_` on 0→1 (`1656-1658`, `1697-1699`, `2838-2840`); successful `flush_chunk()` decrements only after landing and post-landing tombstoning (`1776-1791`, `1804-1919`); failed writes retain the count. `S_BUFFERED` and `S_FLUSHING` are both counted (`3522-3535`), and nonzero `unlanded_chunks_` blocks truncation (`3644`). However, Checkpoint still has no session/snapshot barrier, returns `void` after only an stderr warning (`3660-3670`), and still does not satisfy the interface contract to persist current in-memory state (`kvstore_interface.h:195-198`). A new `S_DIRTY` value created after its stripe was scanned is neither warned about nor counted; that is also a dlog-retirement hole described in R2-C2 below. |
| **C4 — dlog-only restart and LSN reuse** | **FIXED for the stated scenario** | A nonempty dlog now triggers `recover_log(0)` (`1281-1292`). Every valid record contributes to `max_dlsn`, even when no slot/index entry exists, and `lsn_` is advanced past it (`3370`, `3419-3424`). Therefore the stated dlog-only → reinsert → second-restart sequence no longer reuses an older LSN. Final-record corruption remains a separate edge case (R2-M2). |
| **C5 — intent for an index-capacity-dropped key** | **FIXED for capacity drops, incomplete for degraded scans** | `recover_dropped_ != 0` now permanently blocks dlog truncation in that instance (`3645`), so the exact smaller-budget/larger-budget resurrection path is prevented. However, `scan_errors` also means replay may not see an intent's key (`3714-3718`, `3858-3862`), yet `recover_ok_`/`scan_errors` is not in the truncation gate. See R2-C5. |
| **C6 — replay poisoning before `nsets_`/`set_locks_`** | **Functionally fixed, memory fix incomplete** | Replay no longer invokes `poison_versions()` during initialization. It marks and directly queues (`3395-3409`), and duplicate records for a marked key do not add another queue entry. This avoids the original null/zero cache crash beyond 64K. But direct `reap_q_.push_back()` bypasses `kReapMax`, and both queue and registry can grow with all distinct dlog keys before cache construction. See R2-M1. |
| **C7 — Delete/Upsert overlap and intent LSN timing** | **NOT FIXED** | Moving LSN allocation to immediately after `del_mark()` (`3176-3184`) is insufficient because both remain outside the set lock. Upsert publishes and calls `del_unmark()` under that lock (`2741-2759`, `2837-2867`, `2940-3007`), while Delete acquires the lock only later to sweep (`3198-3211`). A concrete violating interleaving is in R2-C1. |
| **M8 — append failure still acknowledges Delete** | **NOT FIXED** | `dlog_append()` now returns `bool` (`3301-3317`), but `Delete()` still returns `newly && existed` regardless of logging/poison success (`3236-3259`). For an overflow-only key, `landed == false`, so append failure performs no synchronous poison at all (`3246-3249`). For landed keys, poisoning has no status and no sync. Repeating Delete cannot repair the record because the mark makes `newly == false`. |
| **M9 — Bug 7 memory bound/accounting** | **NOT FIXED** | `del_mark()` still inserts before any limit and expressly never refuses or trims (`992-1024`). The number of distinct deleted keys is not a finite memory bound. `del_unmark()` subtracts 64 bytes even though `unordered_set` buckets/allocator retention need not be released (`1026-1033`). `GetCacheStats()` includes the logical charge (`4332-4334`), but `ProdStats` exposes neither registry bytes nor mark count (`688-696`, `4297-4317`). The charge only rejects future overflow admissions; it does not backpressure delete marks. |
| **M10 — unbounded dlog read and degraded recovery** | **PARTIAL** | Whole-file allocation was correctly replaced by ~1 MiB streaming (`3333-3355`), and read/short-read/internal-CRC failures generally set `dlog_degraded_`, which feeds `recover_ok_` and prevents truncation (`3344-3368`, `3646`, `3858-3862`). However, the final CRC-invalid full record is treated as benign and does not set degraded (`3362-3368`); `fstat` failure is treated as an empty dlog (`1232-1235`); and replay materializes an unbounded distinct-key registry and deque (`3400-3409`). |
| **M11 — Checkpoint warning quality/contract** | **PARTIAL** | `S_FLUSHING` is now counted along with `S_BUFFERED`, and the global unlanded counter is a stronger retirement guard. But `ckpt_unflushed_` is reset by each Checkpoint, remains racy under concurrent Checkpoints, is only stderr/optional shutdown telemetry, is absent from `ProdStats`, and cannot make the `void Checkpoint()` call fail. No full snapshot barrier was added. |
| **M12 — dlog filename creation durability** | **PARTIAL** | Init now calls `sync_dir()` after opening/creating the dlog (`1226-1238`), which is correct when it succeeds. Failure is neither returned nor logged/counted as `write_errors_` (`3941-3945`), is never retried as a prerequisite to Delete, and does not prevent future truncation. The power-loss guarantee is therefore still conditional rather than established. |
| **m13 — global dlog serialization / poisoning while locked** | **PARTIAL, worst part fixed** | Synchronous poison was correctly moved outside `dlog_mu_` (`3236-3249`), removing page I/O from the global critical section. Every successful Delete still serializes through one mutex and one 24-byte `pwrite`, and Checkpoint can hold that mutex across `fdatasync(dfd_)` on the retain path (`3630-3657`). This is a delete-heavy latency/throughput concern, not a scored Read/Upsert concern. |
| **m14 — test vacuity and missing facets** | **PARTIAL** | The core post-checkpoint test is now non-vacuous: the child is `exec`'d (`test_hydra.cc:2003-2025`), sets `HYDRA_REAP_PAUSE` before Init (`1959-1967`), and performs only ~3.9K post-checkpoint deletes, below queue backpressure, so cleaners cannot have tombstoned them before SIGKILL. The x-record reinsert leg genuinely exercises replay LSN ordering (`1986-1995`, `2050-2060`). The test still has no overflow-only value, >64K replay, foreign `S_BUFFERED`/`S_FLUSHING`, dlog-only store, smaller-index reopen, I/O/fsync/sidecar failure, concurrent Delete-vs-Upsert, or checkpoint/reinsert race. The inline clean-reinsert leg still destructively Checkpoints before reopening (`2096-2107`), and phase 2 uses a fresh dlog after the parent's destructor may truncate phase 1 (`2069-2074`). `delcrash` runs in the full opt/tmpfs/no-uring/ASan/coverage phases but is absent from the explicit TSan subset. |

---

## Blocking findings

### R2-C1 — CRITICAL: early intent LSN still does not serialize Delete with Upsert publication

**Lines:** `hydra.cc:2741-2759`, `2837-2867`, `2940-3007`, `3176-3184`, `3198-3211`, `3273-3291`.

`del_mark()` and the delete LSN are established without the per-key set lock. Upsert publication and `del_unmark()` occur under that lock. Therefore an Upsert can clear the newly created mark before Delete obtains the lock for its sweep.

Concrete completed-operation failure:

1. `V0` is checkpointed. Its cache entry is `S_CLEAN`.
2. Upsert(U) acquires the set lock and updates the cache to `V1`/`S_DIRTY`.
3. While U still owns the set lock, Delete(D) calls `del_mark(key)` and allocates dlog LSN `N`; D then waits for the set lock.
4. U calls `del_unmark(key)`, unlocks, and returns.
5. D acquires the set lock, erases `V1`, appends intent `N`, queues poison, and returns true.
6. Poison sees `is_deleted(key) == false` and stops without tombstoning `V0`.
7. A Read after both operations can return old checkpointed `V0` immediately because the mark is gone. Recovery, in contrast, sees intent `N > lsn(V0)` and reports NotFound.

U completed before D, so the only valid real-time order is U then D; returning `V0` after D is not linearizable. The oversized and fallback-overflow publication paths have the same generation problem. A second Delete that sees the old mark before a concurrent reinsert unmarks it can also sweep the new generation while `newly == false`, emit no intent, and return false.

**Required direction:** make mark/generation creation, intent timestamp assignment, cache/overflow sweep, and Upsert publication participate in one per-key serialization protocol. Merely moving the atomic LSN fetch earlier is not enough. The protocol must distinguish a delete generation from a later reinsert and prevent an older Delete from clearing/sweeping that later generation.

---

### R2-C2 — CRITICAL: Checkpoint can truncate an old intent because a concurrent non-durable reinsert made poison no-op

**Lines:** `hydra.cc:2744-2759`, `2843-2867`, `3279-3291`, `3516-3535`, `3586-3612`, `3630-3648`.

`unlanded_chunks_` covers `S_BUFFERED`/`S_FLUSHING` chunks, but not an ordinary `S_DIRTY` overwrite. Checkpoint scans each set once before draining the reaper and has no write barrier.

Concrete failure without any I/O fault:

1. `V0` is durable. Delete(D) returns and leaves a valid dlog intent plus a queued reaper key.
2. Checkpoint scans the key's set while it has no dirty live value.
3. A concurrent inline Upsert publishes `V1` against the existing index head, making the cache entry `S_DIRTY`, then unmarks the key. Since this is not a first insert, it stages no chunk and leaves `unlanded_chunks_ == 0`.
4. Checkpoint drains D's reaper item. `poison_versions()` sees the removed mark and writes no tombstone.
5. Checkpoint's earlier stripe will never revisit the new `S_DIRTY` entry. `fdatasync(fd_)` therefore persists only old `V0`.
6. Queue/inflight are empty, error deltas are zero, and the unlanded counter is zero, so Checkpoint truncates the intent.
7. Crash/reopen loses `V1` and recovers `V0`.

The Upsert was concurrent with Checkpoint and may legitimately be omitted from that checkpoint, but D completed before Checkpoint. A legal recovered prefix must therefore contain either D (NotFound) or, if U is included, `V1`; `V0` is not legal.

**Required direction:** add a real checkpoint epoch/barrier, or retain/compact intents per key until there is a proven durable tombstone or a proven durable superseding value. A globally empty reaper plus `unlanded_chunks_ == 0` is not such proof.

---

### R2-C3 — CRITICAL: poison failures consumed before a Checkpoint are forgotten and the dlog is later truncated

**Lines:** `hydra.cc:1523-1617`, `3273-3293`, `3439-3453`, `3550-3552`, `3639-3648`.

The fix snapshots counters at Checkpoint entry and rejects truncation only if those counters change during that invocation. Work items have no success state and are always removed.

Concrete failure:

1. Delete appends an intent and enqueues the key.
2. A cleaner pops it. `tombstone_slot()` suffers EIO/ENOSPC/read failure and leaves the live old slot untouched; the queue item is nevertheless consumed.
3. No Checkpoint is in progress, so `write_errors_`/`read_errors_`/`tomb_refused_` increments before the next snapshots.
4. Storage recovers. A later Checkpoint snapshots the already-incremented counters, sees no new failures, an empty queue, and a successful `fdatasync`.
5. It truncates the dlog. Crash/reopen serves the old value.

The sticky `durable_ok_ == false` is explicitly intended never to recover, but the retirement gate ignores it. In addition, when the first x-header read succeeds and the second read under the page lock fails, `tombstone_slot()` returns without incrementing any error/refusal counter (`1546-1571`), so even a same-Checkpoint failure can evade the delta gate.

**Required direction:** return a result from tombstoning/poisoning, retain or retry failed work, and associate completion with the intent it proves redundant. At minimum, any sticky durability/recovery fault must forbid global truncation; counter snapshots cannot replace per-intent success.

---

### R2-C4 — CRITICAL: directory-fsync failure can still retire the only protection for an overflow-only delete

**Lines:** `hydra.cc:3607-3648`, `3879-3945`; dlog creation at `1226-1238`.

`write_overflow_file()` correctly precedes dlog retirement, but its final durability operation is not checked by that retirement decision. `sync_dir()` only writes `durable_ok_ = false`. It neither returns status nor increments `write_errors_`.

Concrete power-loss failure:

1. An overflow-only value is durable in `hydra_overflow.dat`.
2. Delete removes it in memory and appends an intent.
3. Checkpoint successfully `unlink`s the empty sidecar (or renames a replacement without the key), but `fsync(store_directory)` fails.
4. The `write_errors_` delta remains zero, so the `safe` predicate passes and dlog is truncated.
5. Power loss restores the old directory entry/old sidecar; there is no intent, and the deleted key resurrects.

The same defect leaves initial dlog filename durability conditional: the Init call to `sync_dir()` is not a checked prerequisite for accepting crash-durable deletes.

**Required direction:** make `sync_dir()` return checked success and count failure; make sidecar persistence return a checked end-to-end result; never retire intents unless rename/unlink **and** directory fsync succeeded. Retry or fail initialization if dlog creation cannot be made durable for the claimed power-loss contract.

---

### R2-C5 — CRITICAL: `recover_ok=0` from an unreadable slot region does not prevent dlog destruction

**Lines:** `hydra.cc:3332-3415`, `3714-3718`, `3831-3862`, `3639-3646`.

The capacity-drop guard checks only `recover_dropped_`. A slot-region read error increments `scan_errors` and makes `recover_ok_ = 0`, but does not increment `recover_dropped_` and does not set `dlog_degraded_` (the latter describes only dlog replay).

Concrete failure:

1. A live old slot and a newer delete intent exist.
2. Reopen experiences a transient EIO for the slot's 1 MiB region. Recovery skips it; dlog replay cannot find the key in `rk/index_`, so it neither marks nor queues it.
3. Recovery correctly reports `recover_ok=0`, but a later Checkpoint snapshots the already-recorded read error and sees `recover_dropped_ == 0`, clean dlog replay, and no queue.
4. Checkpoint truncates the intent.
5. The storage fault clears; a later reopen reads the old slot and resurrects the key.

**Required direction:** a degraded slot/sidecar recovery must conservatively retain all dlog evidence. Gate retirement on a clean recovery proof (not merely `recover_dropped_ == 0`), and ideally resolve each intent independently of runtime index placement.

---

### R2-C6 — CRITICAL: dlog append failure still allows a successful Delete with no recovery evidence

**Lines:** `hydra.cc:3184-3185`, `3223-3259`, `3301-3317`.

Concrete overflow-only failure:

1. A sidecar-only value exists durably and has no index candidate (`landed == false`).
2. Delete erases the RAM copy, sets `existed = true`, and `dlog_append()` fails.
3. `sync_poison` is false because it is gated by `landed`; there is no slot to poison and no sidecar rewrite before return.
4. Delete returns true. SIGKILL occurs before a successful Checkpoint.
5. The old sidecar reloads and resurrects the value.

For landed keys, synchronous poison also has no result and no fdatasync guarantee. A retrying Delete cannot repair the missing intent because `del_mark()` returns false on the existing mark.

**Required direction:** do not acknowledge Delete unless a recoverable intent was appended or a checked durable fallback completed. This requires a usable failure result/backpressure path and retryable per-key intent state; a sticky metric is not recovery evidence.

---

### R2-C7 — CRITICAL: queue-full synchronous poison is invisible to the Checkpoint retirement protocol

**Lines:** `hydra.cc:1043-1057`, `3236-3249`, `3439-3452`, `3586-3648`.

When the bounded queue is full, Delete successfully appends the intent, releases `dlog_mu_`, and performs synchronous poison outside the mutex. That poison is not represented in `reap_q_` or `reap_inflight_`; `reap_inflight_` is incremented only for items popped by `reap_drain()`.

Concrete failure:

1. The queue is full (deterministic with `HYDRA_REAP_PAUSE`). Target Delete appends its intent, fails `reap_try_enqueue()`, releases `dlog_mu_`, and is descheduled before `poison_versions()`.
2. Checkpoint snapshots the append count, drains the pre-existing queue, syncs slots, sees queue/inflight empty, and truncates the dlog—including the target intent.
3. Target Delete resumes. Its poison occurs after the checkpoint's slot `fdatasync`; if the poison read/write fails, it still returns true. Even if it succeeds, that Checkpoint did not make the tombstone power-loss durable despite having snapshotted the intent.

This is a new hole created by moving expensive poisoning outside `dlog_mu_` without adding a separate in-flight token. Moving it outside was correct for lock latency, but the retirement proof must track it.

**Required direction:** register synchronous poison as in-flight while still under `dlog_mu_`/the same intent protocol, clear it only after checked completion, and make Checkpoint include it in the drain-and-sync proof.

---

## Major and minor findings

### R2-M1 — MAJOR: replay bypasses both advertised memory bounds

**Lines:** `hydra.cc:992-1024`, `1043-1057`, `3333-3340`, `3395-3409`.

Streaming bounds only the input buffer. For each distinct winning key, replay allocates an `unordered_set` node and directly pushes into `reap_q_`, bypassing `kReapMax`. A large valid dlog can therefore allocate gigabytes during Init and throw/terminate or be OOM-killed under the configured budget. Duplicate keys are deduplicated for queueing, but a large distinct-key set is exactly the delete-storm case.

The runtime delete registry has the same unbounded behavior. Charging 64 logical bytes after allocation and rejecting unrelated future overflow entries is observability, not backpressure.

**Required direction:** use a genuinely bounded recovery structure or a disk-backed/stream-merge representation, and make registry reclamation part of the durable intent/tombstone lifecycle.

### R2-M2 — MAJOR: final dlog corruption and dlog `fstat` failure can be reported as clean

**Lines:** `hydra.cc:1232-1235`, `3362-3368`, `3419-3424`, `3858-3862`.

Any `fstat` failure is treated as zero dlog bytes, allowing recovery to skip the file and future appends to overwrite offset zero. A CRC-invalid final **full** 24-byte record increments only `recover_torn_`; it does not set `dlog_degraded_`. Recovery may therefore report `recover_ok=1`, advance `lsn_` without the lost intent's unknown LSN, and later truncate the file.

It is reasonable to overwrite a non-record-aligned, unacknowledged append tail. It is not safe to assume every aligned final CRC failure was unacknowledged; corruption or a torn dlog sync is indistinguishable. At minimum this must degrade recovery and forbid retirement.

### R2-M3 — MAJOR: Checkpoint still does not implement a durable snapshot or actionable failure

**Lines:** `hydra.cc:3516-3580`, `3660-3670`; `kvstore_interface.h:195-198`.

The strengthened counters do not create a barrier. Writers can create `S_BUFFERED`, `S_FLUSHING`, or `S_DIRTY` state after their set was scanned. `ckpt_unflushed_.store(0)` also races concurrent Checkpoints, and the result is absent from the public return value and `ProdStats`. The warning is useful telemetry but not an implementation of “persist the current in-memory state.” R2-C2 shows this also affects delete correctness, not only ordinary write durability.

### R2-m4 — MINOR: delete-heavy serialization remains, while scored hot paths are clean

**Lines:** `hydra.cc:3236-3248`, `3301-3317`, `3630-3657`.

All Deletes serialize on `dlog_mu_` and one small `pwrite`. Checkpoint may hold the mutex while syncing a retained dlog. The severe round-1 problem—page poisoning under the mutex—was fixed. This can still produce a delete-throughput/latency cliff but does not touch the scored delete-free Read/Upsert hot path.

### R2-m5 — MINOR: test hardening is meaningful but far from the claimed matrix

**Lines:** `test_hydra.cc:1959-2110`; `run_all_tests.sh` TSan list.

The deterministic pause test should be kept; it is not vacuous. Add focused tests for every blocking scenario above, especially the two no-fault concurrency histories (R2-C1 and R2-C2). The new test is not included in the TSan subset, and its current post-checkpoint delete count cannot exercise the 64K queue-full path.

---

## Targeted protocol checks requested by the task

### Lock ordering / deadlocks

I found **no current lock cycle** in the reviewed paths:

- Delete takes the delete-shard lock, overflow lock, and set lock sequentially (not nested), then takes `dlog_mu_ -> reap_mu_` for append/enqueue.
- Reapers release `reap_mu_` before taking `set lock -> delete-shard lock -> page lock`.
- Flush/Upsert use `set lock -> overflow/delete-shard/page` and do not acquire dlog/reap locks.
- Checkpoint uses `dlog_mu_ -> reap_mu_`; no path takes those in reverse order.
- Sidecar persistence uses `ckpt_mu_ -> overflow-shard`; no reverse `overflow -> ckpt_mu_` path exists.
- Destructor holds `sess_mu_` while flushing parked sessions, but current write paths obtain/create their Session before set locking and do not take `sess_mu_` while holding a set lock. It releases `sess_mu_` before calling Checkpoint.

This is not approval of liveness: Checkpoint can hold `dlog_mu_` across dlog `fdatasync`, replay/destructor drain can be unbounded, and global Delete serialization remains.

### `unlanded_chunks_` pairing audit

The current pairing itself is complete for all ordinary chunk staging sites:

| Transition | Increment/decrement |
|---|---|
| Restaged tombstone first slot | increment `1656-1658` |
| Dirty-entry flush first slot | increment `1697-1699` |
| First inline insert first slot | increment `2838-2840` |
| Successful landing | decrement after all metadata handling/tombstone attempts, `1917-1919` |
| Failed chunk write | no decrement; staged state retained, `1782-1790` |
| `StopSession()` | calls `flush_partials`; failure leaves chunk and count outstanding, `1153-1165` |
| Destructor | retries every nonempty session, then Checkpoint; failed chunks remain nonzero and block dlog retirement, `1380-1394` |

No extra `chunk_fill = 0` path or unmatched `chunk_fill++` was found. X-records are intentionally not counted because they are written through before publication and are covered by slot-file `fdatasync`.

The counter is nevertheless not a checkpoint barrier and does not cover `S_DIRTY`; that semantic gap is R2-C2/R2-M3.

### Replay edge cases

- **Duplicate keys:** queueing is deduplicated by `del_mark()`. Later higher-LSN intents can still update `rl/rt`; this is correct.
- **LSN ties:** normal records use unique `fetch_add`, so ties should require corruption/wrap. Replay's strict `<` makes an equal recovered value win; duplicate equal intents are idempotent.
- **Intent for a key never in slot/sidecar:** its valid LSN still advances global `lsn_`; if the key truly never existed durably, not marking it is safe. It is unsafe when “not seen” means index drop/read failure; capacity drops are guarded, scan failures are not (R2-C5).
- **Torn/non-aligned tail:** rounding down and overwriting a partial record is reasonable. A full final CRC-invalid record is not safely distinguishable from corruption and is mishandled (R2-M2).
- **More than 64K winners:** initialization no longer calls `set_of()` before cache setup, but replay now bypasses the queue cap and can OOM (R2-M1).

### `HYDRA_REAP_PAUSE` and destructor

`reap_drain(se, false)` in cleaners honors the pause, while Checkpoint/destructor call the default `force=true` and therefore drain. The test child is exec'd, sets the environment before Init, and gets a fresh function-static pause value, so the core test hook works as intended. Destructor first stops background threads, flushes every session's chunk, then calls Checkpoint; I found no pause-specific skipped drain. The general failure/truncation defects above still apply.

### Scored-path performance

Normal `Read` and normal in-place/dirty Upsert do **not** acquire `dlog_mu_`. The only new inline staging cost is one `unlanded_chunks_` atomic transition per chunk, not per operation (`chunk_fill == 0`). Dlog open and directory sync are initialization-only. I therefore see no direct reason for a 50:50 delete-free score regression from these fixes. Delete-heavy workloads do pay one globally serialized buffered write per successful Delete.

---

## Minimum tests required before approval

1. Deterministically interleave Upsert inside its set lock with Delete's mark/sweep and assert both immediate runtime reads and crash recovery agree (R2-C1).
2. Pause a queued delete; let Checkpoint scan; publish an `S_DIRTY` reinsert before reaper drain; crash after Checkpoint and assert recovery is either deleted or the reinsert, never the old value (R2-C2).
3. Fail poison before Checkpoint, allow the next Checkpoint's I/O to succeed, and verify dlog is retained and no resurrection occurs (R2-C3).
4. Fail the second x-header pread specifically and verify the intent cannot be retired.
5. Persist an overflow-only value; fail sidecar directory fsync after rename/unlink; verify dlog retention and power-prefix recovery (R2-C4).
6. Inject a transient slot-region read error during recovery, call Checkpoint, then reopen with reads restored; verify the intent still suppresses the key (R2-C5).
7. Fail dlog append for an overflow-only and a landed key; Delete must report failure or complete a checked durable fallback (R2-C6).
8. Fill the 64K queue under `HYDRA_REAP_PAUSE`, trigger queue-full synchronous poison concurrently with Checkpoint, and verify Checkpoint cannot retire the in-flight intent (R2-C7).
9. Replay substantially more than 64K distinct intents under a strict RSS limit, plus millions of duplicate intents, and assert bounded memory/no Init crash.
10. Exercise dlog-only recovery, capacity-drop retention, final full-record CRC corruption, dlog `fstat` failure, and concurrent Checkpoints.

Until these protocol holes are fixed, the implementation does not establish the promised property that every acknowledged Delete survives SIGKILL/recovery and every successful Checkpoint safely retires its delete intents.
