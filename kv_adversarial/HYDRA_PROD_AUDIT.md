# KV Engines — HydraKV vs kv_5050: Code-Level Audit & Head-to-Head

*Every claim is cited to a real `file:line`, verified by direct code reading + a 10-agent Hydra audit + a 20-agent head-to-head + a final 4-agent line-exact re-check. No number is taken from either engine's own docs without confirming it in code.*

Engines: **HydraKV** (KISS Sorcar / Koushik) `hydra.cc` (2,064 lines) · **kv_5050** (reference) `src/{store,cache,cold_tier,index,record,uring}.h`.
Workload: YCSB-A 50/50, Zipf θ=0.95, 250M × 100 B, 8 GiB budget, larger-than-memory. FASTER = 0.93 Mops.

---

## ★ SUMMARY

Both are **genuine, non-reward-hacked** engines that beat FASTER on the exact workload, built on the **same core design** (compact index + CLOCK write-back cache + O_DIRECT append log + io_uring miss path; Hydra adds TinyLFU admission, ours admits-on-access). They differ almost entirely in **production-readiness**:

- **HydraKV is faster (5.9×)** because it spends everything on throughput and **drops general-prod robustness** — no crash recovery (`O_TRUNC`s its log on every start), `abort()`s the whole process on any disk error, and lets >101 B values leak to unbounded RAM. Its own paper calls it *"a benchmark-shaped engine, not a deployable store."*
- **kv_5050 is more deployable (3.49×)** — it **survives restart** (`recover()`), **fails soft** on disk EIO/ENOSPC, **rejects** oversized values honestly, and gates those properties with fault-injection tests. It trades ~40% throughput for that.
- **Both share one real gap: no disk compaction** — under continuous 50/50 upserts the log grows unbounded. This is a *workload* need (not a benchmark artifact), masked for both by the 30 s window.

**Prod bottom line:** in a real AWS/Azure-style deployment (routine restarts, transient disk faults, multi-day uptime, variable values), **HydraKV loses data on the first restart and crash-loops on the first disk hiccup; kv_5050 survives both** — but *both* eventually fill the disk without a compactor.

### Crisp scorecard

| | HydraKV | kv_5050 | Better |
|---|:--:|:--:|:--:|
| **Throughput** | **5.51 Mops (5.9×)** | 3.25 Mops (3.49×) | **Hydra** |
| Correct / verbatim / last-write-wins | ✅ | ✅ | tie |
| Bounded RAM ≤ 8 GiB | ✅ | ✅ | tie |
| Not reward-hacked | ✅ | ✅ | tie |
| **Crash recovery** | ❌ `O_TRUNC`, no `recover()` | ✅ `recover()` rebuilds index | **kv_5050** |
| **Fault handling (EIO/ENOSPC)** | ❌ `abort()` (11 sites) | ✅ zero abort; sticky flag + retry | **kv_5050** |
| **Value size (arbitrary)** | ❌ >101 B → unbounded RAM | ✅ reject + surface | **kv_5050** |
| **Disk compaction / GC** | ❌ none | ❌ none | **tie (shared gap)** |
| In-memory GC (RAM reclaim) | ✅ CLOCK evict (+unbounded overflow, off-workload) | ✅ CLOCK evict (no unbounded container) | tie (ours cleaner) |
| Index-collision safety | ✅ prev-chained, never loses | ❌ writeback first-tag-match, rare key-bury | **Hydra** |
| **Testing — sanitizers** | ✅ ASan+UBSan, TSan *subset*, 92.8% cov | ✅ ASan+UBSan, TSan *targeted paths* (not full-engine) | tie |
| **Testing — prod-invariant gates** | partial (e2e cases) | ✅ fault-injected recovery, write-error, lost-update differential (TEETH) | **kv_5050** |

**Tally (prod axes):** kv_5050 wins 3 (recovery, fault-handling, value-size) + fault-injection/differential gates · Hydra wins 1 (index-collision) · sanitizers ≈ tie · rest tie, incl. the shared no-compaction gap.

---

### Full detail (click any section to expand)

<details>
<summary><b>1 · Throughput</b></summary>

HydraKV **5.51 Mops = 5.9× FASTER** (paper-reported, non-hacked verified). kv_5050 **3.25 Mops = 3.49× FASTER** (log-backed 3.24/3.25/3.28). **Hydra wins raw speed.**

</details>

<details>
<summary><b>2 · Production readiness — deep, both engines (the important part)</b></summary>

| Property | HydraKV | kv_5050 |
|---|---|---|
| **Crash recovery** | **FAIL** — log opened `O_TRUNC` every start (`hydra.cc:750,753`); index/cache/doorkeeper are volatile `MAP_ANONYMOUS` (`:208-211`); **zero `recover()`** functions. An acked + `fdatasync`'d write is gone on restart. | **PARTIAL–PASS** — log opened **without** `O_TRUNC` (`cold_tier.h:61`); `recover()` scans every page low→high, highest-addr-wins + tombstones, restores the cursor (`cold_tier.h:96-151`, `store.h:120`). No WAL → plain Upserts are cache-only (`store.h:337`), durable at `Checkpoint()` (`store.h:442-463`); survives process restart, loses only un-checkpointed writes on power loss. |
| **Fault handling (EIO/ENOSPC)** | **FAIL** — fail-stop: **11 `abort()` sites** incl. `xpwrite` on any write error (`:319-335`), read path (`:1572,:309`), `fdatasync` (`:2037`). One disk error crashes all 16 threads. | **PASS** — **zero abort/exit sites** (whole tree); `pwrite_all` returns bool, sets sticky `write_failed_`/`durable_ok()`, leaves slot dirty + retries (`cold_tier.h:368-399`); read EIO → NotFound; process stays up. |
| **Value size** | **PARTIAL** — inline ≤101 B (`:105`); >101 B silently routed to an **unbounded** in-RAM `unordered_map` (`:601,:1181`), forfeiting the budget. | **PASS** — hard 100 B (`record.h:30`); oversize **rejected + surfaced** (`rejected_oversize_`, `store.h:339-347`), never truncated, never unbounded. |
| **Disk compaction / GC** | **GAP** — none; "a production build would compact the log here" (`:939`), "log GC out of scope" (`:1925`). Log grows unbounded; 512 GiB 32-bit ceiling. | **GAP** — none; "compaction out of scope for the 30 s window" (`cold_tier.h:16-17`, `index.h:14`). Monotonic cursor; 512 GB ceiling **surfaced** (rejects, never silent-aliases). |
| **In-memory GC** | **PASS** — write-back cache CLOCK/SLRU evicts + cleaner threads bound RAM (`find_victim`, `occupancy_`); the only unbounded container (overflow map) is unreachable at 100 B. | **PASS** — CLOCK eviction bounds cache (`cache.h:15`), index fixed calloc; **no unbounded container** on any path. |
| **Concurrency correctness** | **PASS** — per-set spinlock, genuine LWW, no lost update, no torn read; delete-resurrection closed by pin-ownership (`:1027`). | **PASS** — per-shard mutex, LWW, read-miss fill won't clobber a concurrent Upsert; flush/eviction ABA interlocks correct. |
| **Corruption detection** | PASS — full-key verify on every index hit (`:1335`), fail-stop on corrupt. | PASS — full-key verify on read; tombstone/valid byte per record (`record.h`). |
| **Observability** | Cache stats (`GetCacheStats`), excludes overflow bytes (`:2044-2059`). | `durable_ok`, `recover_ok`, `lost_writes`, `rejected_oversize`, `dropped` counters surfaced in run output (`store.h:487-524`). |
| **Index-collision harm** | **PASS** — 32-bit fp collides but aliases prev-chained; reads verify → extra I/O only, never wrong/lost. | **PARTIAL** — 32-bit tag; reads verify, but writeback `upsert_addr` uses **first-tag-match with no verify** (`index.h:145-157`) → ~1/4e9 can bury a colliding key (write-side). |

</details>

<details>
<summary><b>3 · Disk compaction vs in-memory GC — exact state</b></summary>

**Both lack disk compaction; both bound RAM by CLOCK eviction.** Symmetric — not a point against either alone:
- **Hydra:** append-only, superseded versions leak until an offline compactor (`hydra.cc:12,939,1925`); RAM bounded by cache eviction; unbounded overflow map exists but is off-workload.
- **kv_5050:** monotonic cursor, dead space for overwrites never reclaimed (`cold_tier.h:16`); RAM bounded by CLOCK; no unbounded container.

Why it matters: 50/50 = **continuous blind upserts**, so in real always-on use the log grows without bound → eventually fills the disk. Compaction is a **workload need**, masked for both by the 30 s window.

</details>

<details>
<summary><b>4 · Design idea — convergent architecture</b></summary>

Both AI-generated engines independently converged on the **same** design family — strong evidence it's the *right* answer for skewed larger-than-memory KV (not benchmark-gaming):

**Compact index (maximize cache residency) → CLOCK write-back cache tuned to the θ=0.95 hot set → O_DIRECT append log → batched io_uring miss pipeline.** (Hydra additionally gates admission with a TinyLFU doorkeeper; kv_5050 admits on every access.)

Differences: Hydra adds the **doorkeeper** (admit only on 2nd recent touch) + verifies every index hit and prev-chains aliases; kv_5050 **admits on every access** + uses a compact 8-byte lock-free index and first-tag-match on writeback.

</details>

<details>
<summary><b>5 · General vs benchmark-artifact — and how each breaks at cloud scale</b></summary>

**Workload (asked):** 50/50 read/upsert, Zipf skew, YCSB, 100 B values, larger-than-memory, keyspace fixed at 250M. **Harness artifacts (NOT the workload):** 30 s window, median-of-3, drop_caches, single-process (no restart), deletes/RMW present-but-unexercised.

- **Both are general on the workload core** (skew-aware caching, spill, index, io_uring). Neither is "overdone" for the benchmark in a gaming sense.
- **Hydra leans on harness artifacts to go faster** — skips work the workload-in-production needs but the *benchmark* doesn't test: no recovery (relies on single-process), abort-on-error (relies on fault-free 30 s), delete/RMW gaps (relies on those ops being unused).
- **kv_5050 does the opposite** — built recovery, fail-soft I/O, oversize-reject, delete/RMW correctness, none of which the 30 s benchmark rewards. Less benchmark-overfit; ~40% slower for it.

**How each breaks in a real AWS/Azure-style system:**

| Cloud reality | HydraKV | kv_5050 |
|---|---|---|
| Instance restart / pod reschedule / spot reclaim (routine) | **Total data loss** — `O_TRUNC` wipes the log every start | Survives — `recover()` rebuilds (recovery time scales with log size) |
| Transient EBS / managed-disk EIO (common) | **Crash** (`abort`) → crash-loop | Soft-fails, retries, stays up |
| Disk full / ENOSPC | **Crash** (`abort`) | Surfaces via sticky flag; stays up (writes stay dirty) |
| Multi-day uptime (no compaction) | Log fills volume → ENOSPC → crash | Log fills volume → surfaced; **shared gap** |
| Variable value sizes (real apps) | >101 B → unbounded RAM → OOM-kill | Rejected + surfaced (no data corruption) |
| Power loss / kernel panic | All data lost | Loses only writes since last `Checkpoint` |

</details>

<details>
<summary><b>6 · Testing rigor — which is better tested</b></summary>

**Both use sanitizers; the split is coverage-style vs prod-property gates:**
- **HydraKV:** 14 e2e cases (basic, lww, async, audit, oversized, delete, updel, **alias**, tiny, nodisk, rmw, twostores, **ckpt**, **rywrite**) × **6 build configs: opt / buffered / no-uring / ASan+UBSan / TSan (9-suite subset) / coverage** (paper: 92.8% line / 93.4% branch). Strong sanitizer + coverage discipline; the delete-resurrection oracle + crafted fingerprint-alias keys are genuinely good. (`test_hydra.cc` 935 lines, `run_all_tests.sh`.)
- **kv_5050:** ~20 **prod-invariant gates with TEETH**, re-run under **ASan+UBSan and TSan on targeted paths** (`AUDIT_FINDINGS.md`; op-gates run under sanitizers at the scored operating point) — `check_durability`, `check_recover_read_fault` (**fault-injected recovery**), `check_write_error_surface`, `check_checkpoint_durability_faults`, `check_lost_update_readmiss` (+ `impl_good`/`impl_bad` **differential**), `check_concurrent_rmw` (TSan teeth), `check_delete_no_resurrect`, `check_bounded_memory`, `check_value_oversize`, `check_addr_cap_surface`. **Caveat:** **not full-engine TSan** (A20 accepted-scope — io_uring isn't TSan-instrumentable).

**Net testing:** roughly even on sanitizers (both full ASan/UBSan, both **partial** TSan — Hydra a suite-subset, ours targeted-paths). Hydra edges on line/branch coverage discipline; kv_5050 edges on **fault-injection + differential prod-property gates** (it can *test* durability/recovery/fault-surface because it *implements* them). Ideal combines both.

</details>

<details>
<summary><b>7 · HydraKV — full gap list (code-cited)</b></summary>

**HIGH** — 1) No crash recovery (`:750-753,:2037`). 2) Index anonymous RAM, never persisted (`:208-212`). 3) `abort()` on write error incl. ENOSPC (`:319-335`).
**MEDIUM** — 4) No GC/compaction; 512 GiB ceiling → diverts to unbounded overflow (`:936-941,:1741`). 5) Read-path EIO fail-stop → score 0 on real HW error (`:1572`).
**LOW (general-prod)** — 6) >101 B → unbounded overflow map, off-budget (`:1181,:2044`). 7) RMW-vs-Upsert lost update (RMW unused by YCSB-A). 8) Multi-root stale read for same-signature keys (`:1336`). 9) Delete global-mutex cliff (`:655`). 10) Delete/oversized-upsert liveness gap (`:1902`). 11) Silent O_DIRECT→buffered fallback (`:750-753`). 12) io_uring short-read zero-fills (`:1574`). 13) `xpread` aborts on EAGAIN (`:302`). 14) Doc drift ">102 B" vs `>101` (`:22,:105`).

**Checked & FALSE (no invented defects):** ❌"blows 8 GiB at 250M" (2²⁸=268M, cap 260M) · ❌"global mutex every read" (gated `:655`) · ❌"delete resurrection" (pin-ownership `:1027`) · ❌"values regenerated" (verbatim `:948`) · ❌"page-cache cheat" (O_DIRECT `:750`) · ❌"reads trace" (`:750,:804`) · ❌"io_uring drops/hangs" (`:417-457`) · ❌"StoreRSS under-reported" (`:209`).

</details>

<details>
<summary><b>8 · kv_5050 — where it differs (code-cited)</b></summary>

- Durability PARTIAL: no O_TRUNC (`cold_tier.h:61`), `recover()` (`:96-151`), no WAL — checkpoint-durable (`store.h:442`).
- Fault PASS: zero abort; sticky flag + retry (`cold_tier.h:368-399`).
- Value PASS: oversize rejected (`store.h:339-347`).
- Index-collision PARTIAL: writeback first-tag-match, rare write-side bury (`index.h:145-157`).
- Shared gap: no compaction (`cold_tier.h:16`). Sanitizers: ASan+UBSan + TSan on targeted paths; **not full-engine TSan** (io_uring not instrumentable — accepted scope A20).

</details>

<details>
<summary><b>9 · Conclusion</b></summary>

Same design family, opposite priorities. **Hydra optimizes the benchmark (5.9×) by dropping prod robustness the 30 s window never tests** — it would lose data on the first cloud restart and crash on the first disk fault. **kv_5050 keeps that robustness (recovery, fail-soft I/O, oversize-reject, prod-gate + fault-injection testing, sanitizer-clean) at 3.49×** — deployable, but weaker on index-collision safety and lacks full-engine TSan. **Both owe this workload disk compaction, and neither has it.** Speed-vs-robustness trade — not one strictly better.

</details>
