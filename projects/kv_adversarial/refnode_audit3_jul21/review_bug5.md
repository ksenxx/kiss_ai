# Read-only review of `bug5_diff.patch`

## Verdict: **REQUEST_CHANGES**

The patch addresses the common-path SIGKILL reproduction, and the basic lock order (`dlog_mu_ -> reap_mu_`, with reapers releasing `reap_mu_` before taking set/page locks) does not itself form an obvious cycle. The scored delete-free hot path is also effectively unchanged. However, the delete-intent log is retired on conditions that do **not** prove that an intent is redundant, recovery has several concrete correctness/crash cases, Bug 6 is only reported rather than fixed, and Bug 7 remains an unbounded allocation. Several of these cases can resurrect an acknowledged delete.

## Findings

### CRITICAL 1 — Checkpoint can discard the dlog before overflow-sidecar deletion is durable

**Lines:** `hydra.cc:3145-3199`, `3282-3344`, `3522-3553`, `3772-3829`.

`Delete()` correctly logs a successful deletion of an overflow-only key. On recovery, `dlog_replay()` erases a stale loaded overflow value at `3310-3319`. Such a key need not have any index word, so the probe at `3323-3344` finds nothing and no reaper item is created. Checkpoint nevertheless treats an empty reaper as proof that every intent is redundant and truncates the whole dlog at `3530-3542`. Only **afterward** does it rewrite/unlink the overflow sidecar at `3552-3553`.

Concrete failure:

1. A fallback/overflow-only value is checkpointed into `hydra_overflow.dat`.
2. `Delete(key)` erases it in RAM and appends an intent; there is no slot/index version to poison.
3. `Checkpoint()` drains an empty reaper and truncates the dlog.
4. The process is killed before the sidecar rename/unlink, or `write_overflow_file()` fails at `3778-3784` / `3822-3826`.
5. Recovery loads the old sidecar and there is no intent left to suppress it: the deleted key resurrects.

This is reproducible even with SIGKILL in the same kernel if sidecar update is fault-injected after the successful `ftruncate`; it is not merely a speculative power-ordering issue. The new test never creates an overflow-only key—300-byte values are x-records in the slot log—so it does not cover the audit's sidecar facet.

**Required direction:** persist the new overflow state (including directory fsync) before retiring any intent that guards it, make `write_overflow_file()` return a checked success result, and keep/`fdatasync` the dlog on every failure. Global truncation needs proof for every intent, not merely an empty reaper queue.

---

### CRITICAL 2 — Quiescent reaper does not imply successful/durable tombstones; dlog is truncated after I/O failure

**Lines:** `hydra.cc:1499-1594`, `3224-3243`, `3362-3377`, `3509-3542`.

`tombstone_slot()` returns `void`. Read failure, stale-position refusal, and x/inline `pwrite` failure all return without a success indication (`1510-1558`, `1564-1591`). `poison_versions()` and `reap_drain()` then unconditionally consume the work item. With a failed tombstone write, the loop may retry up to 4096 times, but it eventually returns and the queue/inflight counters reach zero. Checkpoint also continues after a failed `fdatasync(fd_)` (`3512-3520`) and can immediately truncate the dlog (`3530-3542`).

Thus ENOSPC/EIO, a read fault, a refused stale position, or a failed slot `fdatasync` can produce: acknowledged delete -> no durable tombstone -> dlog truncated -> crash/reopen -> old value resurrected. Setting `durable_ok_=false` is observability, not preservation of the recovery mechanism.

**Required direction:** poisoning must return a durable/provable result, failed work must remain pending (or the intent must remain in the dlog), and dlog retirement must be forbidden if any tombstone/read/sync operation involved in the checkpoint failed. In particular, successful `fdatasync(fd_)` must be an explicit prerequisite, not a best-effort side effect.

---

### CRITICAL 3 — Foreign staged chunks make dlog truncation unsafe and Bug 6 compounds Bug 5

**Lines:** `hydra.cc:1748-1841`, `3137-3200`, `3441-3469`, `3528-3564`; interface `kvstore_interface.h:195-198`.

Checkpoint only flushes the caller's partial chunk. It counts `S_BUFFERED` entries in other sessions and prints a warning, but still returns success and still truncates the dlog. It does not even count foreign `S_FLUSHING` records; a deleted FLUSHING entry has already been changed to `S_EMPTY`, so there may be no cache evidence at all.

This creates a direct resurrection window:

1. Session A has a foreign `S_BUFFERED` first insert, or an `S_FLUSHING` record, parked in its partial chunk.
2. Session B deletes the key and logs an intent. The reaper cannot poison the not-yet-landed record.
3. Session B calls `Checkpoint()`. It warns at most, fdatasyncs the current slot file, and truncates the dlog.
4. Session A later calls `StopSession()`. `flush_chunk()` first writes a **live-looking** whole chunk at `1753-1763`, then tombstones the deleted record at `1807-1812` / `1830-1840`.
5. SIGKILL between those writes leaves a live record and no dlog; recovery resurrects it.

This also shows why `ckpt_unflushed_` cannot merely be consulted before truncation: it misses S_FLUSHING/orphan metadata. The interface says Checkpoint persists current in-memory state; a stderr warning with no return status is not an implementation of that contract. Bug 6 remains open.

**Required direction:** add a real checkpoint/session barrier or per-session chunk synchronization, flush every pre-barrier staged record, and prevent new pre-snapshot work from escaping the barrier. At minimum, never retire delete intents while any session-owned staged record that could contain the key remains unresolved.

---

### CRITICAL 4 — A dlog-only restart skips replay and reuses LSNs, allowing an old intent to delete a new reinsert

**Lines:** `hydra.cc:1205-1220`, `1260-1268`, `2800-2809`, `2906-2910`, `3282-3350`.

Initialization invokes `recover_log()` only when the slot file is nonempty or `hydra_overflow.dat` exists. It ignores a nonempty `hydra_dlog.dat`.

Concrete scenario:

1. A first inline insert is parked `S_BUFFERED`; its slot file is still empty.
2. The key is deleted. The volatile index candidate makes `Delete()` append an intent with (for example) LSN 2.
3. SIGKILL leaves only the dlog durable in the kernel page cache.
4. Reopen skips `recover_log(0)`, so `lsn_` restarts at 1 and the old dlog remains unread.
5. Reinsert the same key as an x-record; it is written through immediately with LSN 1. SIGKILL again before Checkpoint.
6. The next reopen now sees a nonempty slot file, replays the old LSN-2 intent, and deletes the post-restart reinsert.

**Required direction:** include `dlog_bytes_ != 0` in the recovery trigger, replay dlog even with an empty slot log and no sidecar, and advance `lsn_` past every valid intent before accepting writes.

---

### CRITICAL 5 — Replay cannot protect keys dropped by recovery's index-capacity limit

**Lines:** `hydra.cc:3282-3344`, recovery placement/drop around `3670-3705`, and dlog call at `3722-3727`.

`dlog_replay()` can compare an intent only with keys represented in `index_`/`rk[]`. If recovery opened with a smaller budget and dropped a key because the index was full, replay stops at an empty probe word or never finds `rk[it] == key`. The intent is neither marked nor queued.

The current small-budget instance reports degraded recovery, but its next Checkpoint sees a quiescent reaper and truncates the dlog. A later reopen with a sufficiently large budget scans the still-present old live slot and resurrects a key whose delete had returned successfully. Reporting `recover_ok=0` does not justify destroying the only delete evidence.

**Required direction:** intent resolution must not depend on successful placement in the runtime index. Scan/track the newest slot LSN for every dlog key independently (for example, parse/compact intents first and track those keys during the log scan), and never retire unresolved intents.

---

### CRITICAL 6 — Replay can call `poison_versions()` before `nsets_`/`set_locks_` exist

**Lines:** `hydra.cc:1260-1268`, `1292-1302`, `3282-3338`, `3224-3241`; queue bound at `1041-1046`.

Recovery occurs before cache sizing and before `set_locks_` allocation. `dlog_replay()` enqueues each winning record and, when the 64K queue is full, synchronously calls `poison_versions(key, session())`. That function computes `set_of(key)` while `nsets_ == 0` and dereferences `set_locks_[0]`, which is null.

More than 64K winning records can occur during a delete storm (queue contents plus in-flight records at crash), after repeated intents, or when runtime poisoning was defeated by read/write faults. The result is an Init-time crash, exactly when recovery is most needed. Duplicate dlog intents can also fill the replay queue with redundant entries.

**Required direction:** never perform set/cache-dependent poisoning until cache initialization is complete. During replay, deduplicate to the newest intent per key and stage recovery work in a structure that cannot invoke the runtime fallback prematurely.

---

### CRITICAL 7 — Delete/Upsert overlap can produce runtime/recovery disagreement because the intent LSN is allocated too late

**Lines:** `hydra.cc:3145-3199`; inline publication/unmark `2712-2729`, x-record publication/unmark `2853-2976`.

`Delete()` marks outside the set lock, releases the mark/sweep lock, and only later allocates the intent LSN. A concurrent Upsert can publish under the set lock and `del_unmark()` between `del_mark()` and dlog append.

A concrete x-record ordering is:

1. Delete marks the key.
2. Upsert acquires the set lock, writes/links an x-record with LSN N, unmarks, and returns.
3. Delete observes a landed candidate, appends intent LSN N+1, and returns. The reaper refuses to poison because the key is now unmarked.
4. Runtime reads return the Upsert value, but after crash the newer intent wins and deletes it.

There is no linearization order that explains both states: placing Delete before Upsert requires the reinsert to survive recovery; placing Delete after Upsert requires immediate runtime reads to be absent. Inline/cache interleavings have related sweep-after-unmark anomalies.

**Required direction:** make mark/sweep/intent ordering and Upsert publication share a per-key serialization/generation protocol. The intent's logical timestamp must correspond to the Delete linearization point, not to a later bookkeeping step.

---

### MAJOR 8 — An acknowledged Delete has no recovery guarantee when dlog append fails, and retries cannot repair it

**Lines:** `hydra.cc:3186-3200`, `3252-3267`.

`dlog_append()` returns `void`; on `pwrite` failure it only sets sticky stats. `Delete()` still returns true. Worse, the delete mark is now present, so a repeated idempotent Delete has `newly == false` and will not attempt another append. SIGKILL before successful background poison resurrects the value.

For the stated property (“a Delete that returned survives SIGKILL”), append failure must either make the operation report failure through a usable API, fall back to a successful synchronous durable tombstone, or retain retryable intent state. A custom sticky metric cannot reconstruct the lost intent.

---

### MAJOR 9 — Bug 7 is not actually bounded or fully charged

**Lines:** `hydra.cc:981-1022`, `1163-1171`, `2036-2049`, `4190-4228`.

`del_mark()` allocates first and merely warns after `del_bytes_ > oversize_cap_`; marks are explicitly never refused or trimmed. Therefore distinct-delete churn can still grow `unordered_set` memory without bound and OOM the process. Adding `del_bytes_` to admission checks for future overflow entries only makes overflow fail sooner; it does not enforce the store's memory budget.

The accounting is also incomplete:

* `GetCacheStats().hot_bytes` includes cache + overflow but omits `del_bytes_`, so normal memory reporting remains understated.
* `FillProdStats` does not export delete-registry bytes/counts; they appear only in an optional destructor stderr line.
* `unordered_set::erase` generally retains bucket arrays (and allocators may retain freed nodes), while `del_unmark()` subtracts the full 64-byte logical charge. Historical peak allocation can remain resident while the reported charge falls to zero.
* Recovery loads overflow values before reconstructing delete marks, so the claimed shared cap is not jointly enforced on that path.

**Required direction:** use a genuinely bounded representation or durable on-disk delete state that permits marks to be reclaimed after a proved checkpoint; account actual/container capacity conservatively; include it in all budget/stats paths; and provide backpressure before allocation rather than a post-allocation warning.

---

### MAJOR 10 — Recovery reads the entire dlog into unbudgeted RAM and does not mark recovery degraded on dlog failure

**Lines:** `hydra.cc:3282-3303`, `3752-3755`.

`std::vector<uint8_t> buf(sz)` allocates the full file. A long delete interval can create gigabytes of dlog records; recovery then allocates that amount in addition to the index and `rk`/`rl`/`rt` scratch maps, potentially throwing/terminating or being OOM-killed under the configured budget. This directly undermines the Bug 7/memory-hardening goal.

If `xpread` fails, replay simply increments `read_errors_` and returns. CRC-invalid records are skipped. Neither condition participates in `recover_ok_`, which only considers slot scan errors/drops. Recovery may therefore serve resurrected keys while reporting `recover_ok=1`.

**Required direction:** stream fixed-size chunks, bound/deduplicate intent state, and feed dlog read/CRC failures into an explicit degraded-recovery result. Do not retire a dlog that was not fully and validly replayed.

---

### MAJOR 11 — The Checkpoint warning is racy, incomplete, and not machine-actionable

**Lines:** `hydra.cc:3441-3469`, `3554-3564`; test `test_hydra.cc:1070-1110`.

Even as telemetry, `ckpt_unflushed_` is not a reliable snapshot. An active writer can create a buffered entry after its set was scanned, S_FLUSHING/orphan records are omitted, and concurrent Checkpoints can reset each other's atomic count. The warning is not exposed through `Checkpoint()` or `ProdStats`.

The existing checkpoint test stops and joins all workers before validating, and store destruction then flushes all sessions. It never crashes immediately after a Checkpoint while a foreign owner remains parked, so it cannot detect Bug 6.

---

### MAJOR 12 — Power-loss claim omits dlog file-creation/name durability

**Lines:** `hydra.cc:1211-1220`, `3246-3251`, `3543-3549`; directory sync helper `3832-3838`.

`hydra_dlog.dat` is opened with `O_CREAT`, but creation of its directory entry is never fsynced. `fdatasync(dfd_)` does not portably guarantee that a newly created filename survives power loss. The comments claim Checkpoint extends pending intents to power-loss durability, which is not established for a newly created dlog. The file should be created as part of a directory-synced store initialization (and all rename/unlink ordering must be included in the checkpoint protocol).

---

### MINOR 13 — Global dlog serialization and synchronous poisoning under that mutex can severely stall delete workloads

**Lines:** `hydra.cc:3197-3200`, `3252-3267`.

Every Delete takes one global mutex and one 24-byte `pwrite`. When the reaper queue is full, Delete holds `dlog_mu_` across `poison_versions()`, which can perform thousands of reads/page rewrites. This blocks all Deletes and Checkpoint dlog maintenance. It does not affect the scored delete-free workload, but it is a meaningful regression risk for the delete-heavy production path. Batching/per-thread append reservation or a dedicated log writer would avoid this serialization.

---

### MINOR 14 — `test_delcrash` does not establish several claims in its own header

**Lines:** `test_hydra.cc:1946-2093`.

* Background cleaners remain enabled; by the time the child signals readiness at `1987`, the reaper may already have written tombstones. The test does not prove recovery relied on dlog. A deterministic pause/failpoint is needed.
* Post-checkpoint phase-1 deletes number only about 3.9K, far below the 64K replay boundary.
* 300-byte values are x-records, not overflow-sidecar-only values.
* `delete st` at `2055` invokes the destructor's full Checkpoint, so the comment “clean close (keeps dlog)” is false; it normally drains and truncates the dlog. The second generation does not test the same surviving dlog.
* The delete/reinsert leg at `2084-2088` also cleanly stops/destructs, causing Checkpoint/truncation; it does not test replay choosing a newer reinsert while an older intent remains.
* There is no foreign-session checkpoint/crash, dlog-only store, index-drop reopen, dlog/read/write/fsync/sidecar fault injection, concurrent Delete-vs-Upsert writer, or power-loss ordering test.

---

## Wiring paths checked and dispositioned

* **RMW:** deleted-key slow path eventually calls `Upsert()` and gets normal reinsert publication; the resident fast path has a pre-lock delete check and can linearize before an overlapping Delete. No separate dlog record should be emitted by RMW itself. The broader mark/publication race in CRITICAL 7 still needs a per-key protocol.
* **Compaction/restage:** `restage_tombstone()` is maintenance for an existing delete and should not append a new user intent. Its durability-before-punch ordering is separate and appears appropriate. The dlog must nevertheless remain until all staged/orphan records covered by the original intent are resolved.
* **Other `overflow_erase()` callers:** Upsert/RMW erasures supersede overflow with a new value and should not log a delete. The Delete and replay erasures are the paths requiring the sidecar ordering in CRITICAL 1.
* **Pure in-memory mode:** there is no storage path and `Delete()` directly erases the overflow map; a disk dlog is not applicable.
* **Buffered slot-file fallback / x-record fd:** `fd_` and `fd2_` reference the same inode, so successful `fdatasync(fd_)` should cover buffered x-record/tombstone writes on Linux. The defect is that success is not required before dlog retirement.
* **Lock ordering:** current code consistently takes `dlog_mu_` before `reap_mu_`; reapers release `reap_mu_` before set/page locks. I found no present lock cycle. The long synchronous-poison critical section remains a latency/liveness concern.
* **`dlog_bytes_`:** runtime append/truncate operations are serialized by `dlog_mu_`; atomic access is sufficient for stats. The major problems are semantic retirement and recovery, not a torn in-process counter.

## Performance / score assessment

The scored workload is delete-free. The patch adds no load to normal Read/Upsert hot paths; the new file open is initialization-only, and the added state check is Checkpoint-only. I see no direct reason for a scored throughput regression from this diff. Delete-heavy throughput will decline because every successful Delete now performs a serialized buffered `pwrite`, and recovery/checkpoint costs can be much worse as described above.

## Minimum regression tests required before approval

1. Deterministically pause the reaper, Delete checkpointed inline and x-record values, SIGKILL, and verify dlog—not tombstone luck—suppresses them.
2. Force an overflow-only persisted value; Delete it; inject sidecar unlink/rename/fsync failure and SIGKILL after dlog maintenance; verify no resurrection.
3. Park foreign S_BUFFERED and S_FLUSHING records, Delete them, call Checkpoint from another session, then kill exactly after chunk write/before tombstone; verify durability.
4. Create a dlog-only store, reopen, reinsert with x-record, crash, and verify old intent cannot beat the new value.
5. Reopen under an index too small to place an intent's key, checkpoint, then reopen large; verify the delete remains permanent.
6. Replay more than 64K winning/duplicate intents and verify Init neither crashes nor exceeds memory.
7. Fail the Nth dlog write, tombstone write/read, slot `fdatasync`, dlog `fdatasync`, sidecar rename/unlink, and directory fsync; verify intent retention and recovery status.
8. Race independent Upsert/Delete writers on the same keys, crash at random points, and compare runtime/recovered outcomes with a linearizable history oracle.
9. Delete enough distinct keys to exceed the intended registry pool and assert RSS/accounting remains bounded.
10. Add crash-prefix/power-loss replay for slot tombstone, dlog append/truncate, sidecar rename/unlink, and directory-fsync ordering.
