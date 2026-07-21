# July 20 — HydraKV independent verification & bug reproduction

Independent review of the hardened HydraKV engine from
`ksenxx/kiss_ai/projects/kv_adversarial`, run on the reference box
(n2-standard-64, Ubuntu 22.04, kernel 6.8, g++ 11.4). Every result below was
reproduced with a self-asserting test; the honest retractions are kept in.

## What's in this folder
- **`engine/`** — Koushik's **exact codebase, as-is** (the generation): `hydra.cc`,
  `test_hydra.cc`, `kvstore_interface.h`, `run_all_tests.sh`, `gen_variant.cc`,
  `PROD_READINESS.md`, `HYDRA_PROD_AUDIT.md`, `DISCOVERY_LOG.md`, `KV_TASK.md`,
  `hydra_iter1_backup.cc`. Nothing modified.
- **`tests/`** — our independent reproduction programs (`proofs.cc`, `devil.cc`,
  `devil2.cc`, `devil3.cc`, `dcost.cc`). Each links against `engine/hydra.cc`.

## Build & run (from this folder)
```bash
cd tests
cp ../engine/{hydra.cc,kvstore_interface.h} .
g++ -O2 -g -std=c++17 -pthread -I. proofs.cc hydra.cc -o proofs
g++ -O2 -g -std=c++17 -pthread -I. devil.cc  hydra.cc -o devil
g++ -O2 -g -std=c++17 -pthread -I. dcost.cc  hydra.cc -o dcost
D=/mnt/ssd   # any fast NVMe dir
./proofs p1 $D/t1     # deadlock
./devil enospcB $D/t2 # disk-full OOM
./proofs p3 $D/t3     # recovery data-loss
./dcost   $D/t4       # delete cost
```

## What we VERIFIED as genuine ✅
- **Performance is real and in-budget:** scored median-of-3 = **5.51 Mops/s
  (5.9× FASTER), StoreRSS 7.54 GiB ≤ 8 GiB** on every run.
- **Not reward-hacked, not copied:** true O_DIRECT end-to-end, verbatim CRC'd
  values, no workload constants, honest RSS; only standard public-domain snippets
  (SplitMix64, runtime-generated CRC32C, original io_uring reimpl).
- **Clean at the scored point:** full `run_all_tests.sh` matrix passes here —
  opt / buffered / no-uring / **ASan+UBSan / TSan (0 warnings)** / coverage 92.5%.

## What FAILS — reproduced on hardware 🔴 (all OUTSIDE the 30 s / single-flow / healthy-disk / no-delete benchmark)
| # | Bug | Reproduced outcome | Root cause (engine line) |
|---|---|---|---|
| 1 | **Deadlock: concurrent Upsert + Delete** | `proofs p1`: 4 threads, 256 keys, no external locks → **froze at 6 ops in 4 s**, threads never return | Cross-session `S_BUFFERED` pin + caller-only `flush_partials`, no-timeout wait loop, no foreign-flush — `hydra.cc:2417-2439` / `:2299-2323` |
| 2 | **OOM on disk-full** | `devil enospcB`: **RSS 12.1 GiB > 8 GiB budget**, `oversize_bytes=0` | Inline (≤101 B) values absorbed into overflow map **uncharged** — `hydra.cc:1560`, callers `:2187/:2213/:2227` |
| 3 | **Recovery loses data, reports success** | `proofs p3`: reopen at smaller budget → **480K/976K keys lost, `recover_ok=1`** (only a stderr WARNING) | Index cap scales with budget; over-cap keys dropped `:2642`, `recover_ok` ignores drops `:2697` |
| 4 | **Delete ≈ 160× slower than Upsert** | `dcost`: **Upsert 1.5 µs vs Delete 244 µs** | Synchronous tombstone-poison (pread+pwrite) per Delete — `verified_lookup :1112` + `tombstone_slot :1156` |

## Code-confirmed design gaps (verified, narrower / off-benchmark)
- **Index count never decrements** (`:270`, no `fetch_sub`) → after `cap` distinct keys, index is full forever; no index GC.
- **Compaction vs in-flight read → transient false-NotFound** (`:3052-3064`, no epoch/refcount) — narrow, retry-curable, no on-disk loss.
- **No write backpressure / 32-bit 512 GiB position ceiling** (`:1239,:2857`) — real but unreachable in a 30 s run.

## Honestly RETRACTED after adversarial self-checking (not real / overstated)
- **"Every un-checkpointed write is lost on crash" — WRONG.** Measured only **~3K/195K** lost on SIGKILL without Checkpoint (background cleaners + O_DIRECT persist most within ms).
- **"admit-by-position serves stale" — REFUTED** (fail-safe; only skips a cache admission, never serves stale).
- **"101 B → single-mutex cliff" / "doorkeeper collapses throughput" / "delete registry unbounded" — OVERSTATED** (64 sharded mutexes; writes bypass admission; registry is workload-bounded).

## Summary (fact-checked)
HydraKV genuinely hits **5.9× FASTER, in the 8 GiB budget, non-hacked** — a strong,
clean benchmark engine. It is **not yet production-grade**: four defects reproduce on
hardware — a **near-instant Upsert+Delete deadlock**, a **disk-full OOM**, **silent
recovery data-loss on a smaller-budget reopen**, and **~160× delete cost**. None are
reachable by the scored benchmark, all are ordinary in real deployment. The two
hardest (deadlock, OOM) trace to two missing primitives that Redis/FASTER/RocksDB
provide: epoch/refcount reclamation, and an all-inclusive memory budget with
backpressure.