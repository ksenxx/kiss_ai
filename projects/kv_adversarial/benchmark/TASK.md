# TASK — High-throughput, production-grade Key-Value store (YCSB-A, larger-than-memory)

## Goal
Implement a **production-grade** key-value store — a real, deployable store like **Redis / RocksDB /
FASTER**, not a benchmark toy — that runs the workload below on the fixed hardware and **beats
FASTER (~0.93 Mops/s)** at an 8 GiB memory budget. Keep iterating to beat it by as much as possible
**while staying genuinely prod-grade** (a fast number that isn't correct/durable/robust does not
count). What "prod-grade" requires is specified under **"Durability / prod-grade"** below. You get
NO external help, no reference code, no prior artifacts — solve the exact problem below, by yourself.

## Workload (fixed — the harness drives it; do not modify the harness)
- **YCSB-A**, workload id **0**: each op is 50% Read / 50% blind Upsert (unseeded RNG; 50/50 in
  expectation over ~1e8 ops, not a fixed schedule).
- **Access distribution:** Zipfian theta = 0.95 over the keyspace (small hot set dominates).
- **Keys:** 250,000,000 unique 8-byte uint64. **Values:** exactly **100 bytes** (env
  KVSTORE_VALUE_SIZE=100; the harness default is 8 — the scored run sets 100).
- Value bytes = 8-byte little-endian counter + (size-8) random bytes, **stored verbatim** and
  verified by an FNV-1a-32 checksum. Values are NOT derivable from the key; regenerating or
  compressing them is a detected integrity failure.
- **Phases:** LOAD = insert all 250M keys (untimed). RUN = the timed window (below).
- Run keys are a subset of loaded keys (reads never miss). RUN Upserts are blind overwrites of
  existing keys only — the keyspace stays exactly 250M (no growth during RUN).

## Metric (exact)
- **Score = throughput over a fixed 30.0-second steady-clock window** (kRunSeconds=30):
  ops accumulated in the window divided by the window length = **Mops/s**. It is fixed-TIME, not
  fixed-ops (the run trace holds 250M keys; at these rates the window consumes far fewer — no wrap).
- **Reported number = median of 3 runs**, drop_caches before each; a run that exits non-zero, OOMs,
  or hangs scores 0 (not retried). run.sh ENFORCES StoreRSS <= 8 GiB (over-budget = score 0), wraps
  each run in a timeout (hang = score 0), and parses the harness "Total:" line (emitted on stderr,
  captured via 2>&1).
- **Baseline to beat: FASTER ~= 0.93 Mops/s** at this exact setting.

## Hardware (fixed) & where to run

**No cloud account, no authentication, no internet is needed to RUN this** — it is a local C++
benchmark (`build.sh` / `run.sh` / `validate.sh`), and the task even forbids network access. Cloud is
only relevant if you choose to *provision* a machine matching the reference node below.

**Reference node** (what the FASTER 0.93 baseline and our runs used):
- **GCP `n2-standard-64`** — 2× Intel Xeon (Cascade Lake) @ 2.80 GHz, **64 vCPU** (2 sockets × 16
  cores × 2 threads), **251 GB RAM**, 2 NUMA nodes (node0 = CPUs `0-15,32-47`, ~128 GB).
- **16 worker threads pinned to NUMA node 0** via `numactl --cpunodebind=0 --membind=0` **plus 1
  integrity-auditor thread** when `KVSTORE_AUDIT=1` — the auditor's ops are NOT counted in throughput
  (identical for every engine, including the FASTER baseline).
- **Storage:** 8× 375 GB **local NVMe SSD** in **RAID0** (`/dev/md0`, 512 KB chunk, ~2.9 TB, ext4)
  mounted at `/mnt/ssd`; non-rotational, O_DIRECT expected. Point `KV_SPILL` at a dir here (~30 GB free).
- **OS / toolchain:** Ubuntu 22.04.5 LTS, kernel 6.8; g++ 11.4, flags
  `-O3 -march=native -DNDEBUG -flto=auto -std=c++17`, pthread only, no external deps.

**Provision a matching GCP node** (optional; uses *your own* gcloud — the task itself needs none):
```
gcloud compute instances create kv-bench --zone=<zone> \
  --machine-type=n2-standard-64 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --local-ssd=interface=nvme --local-ssd=interface=nvme --local-ssd=interface=nvme \
  --local-ssd=interface=nvme --local-ssd=interface=nvme --local-ssd=interface=nvme \
  --local-ssd=interface=nvme --local-ssd=interface=nvme          # 8 × 375 GB local NVMe
# then, on the instance: RAID0 the 8 local SSDs, format, mount, install tools
sudo mdadm --create /dev/md0 --level=0 --chunk=512 --raid-devices=8 /dev/nvme0n{1,2,3,4,5,6,7,8}
sudo mkfs.ext4 -F /dev/md0 && sudo mkdir -p /mnt/ssd && sudo mount /dev/md0 /mnt/ssd && sudo chown "$USER" /mnt/ssd
sudo apt-get update && sudo apt-get install -y g++ numactl mdadm coreutils
loginctl enable-linger "$USER"      # so `systemd-run --user` (the 8 GiB cgroup) works
```

**Any comparable machine also works** — Linux + a fast NVMe (~30 GB free for the spill) + **≥ ~24 GB
RAM** (harness ~10 GB working set + the 8 GiB engine budget) + g++ 11+. On non-identical hardware the
absolute Mops differ, but the multiplier vs FASTER (measured on the same box) is what carries.

## Boundaries (all defined — nothing open)
- **Memory:** engine **data budget = 8 GiB** (8,589,934,592 bytes, passed as mem_budget_bytes).
  Your store's resident data (reported as StoreRSS) MUST stay <= 8 GiB — runs over budget are
  rejected. The whole process runs in a cgroup **MemoryMax = 14 GiB + MemorySwapMax = 0** (the extra
  ~6 GiB covers the harness's ~5 GiB key/checksum arrays + IO buffers; **no swap** — exceeding = OOM-kill).
- **Larger-than-memory:** ~25 GB of data (250M x 100B) >> 8 GiB, so you MUST spill to SSD and cache a hot subset.
- **Init:** the harness calls **only** InitExtended(hash_table_size = 1<<27, log_size_bytes = 16 GiB,
  mem_budget_bytes = 8 GiB, storage_path = <spill>). The first two are **non-binding hints** — do NOT
  allocate 16 GiB of RAM for the "log"; only mem_budget_bytes constrains RAM.
- **Storage:** use only the provided spill dir on md0; O_DIRECT expected (512B / 4KB aligned).
- **Correctness (required):** last-write-wins; values stored/returned verbatim. Certified by the
  separate untimed validate.sh (stride-1 read-back). Scored run.sh runs with validation OFF (no
  warmup); the timed window must still be a correct engine (in-run integrity guards active).
  (Note: run-phase upserted values' random suffix bytes are not re-verified — only the 8-byte
  counter and all load-phase values are — do not exploit this.)
- **Durability / prod-grade:** must behave as a real always-on KV — no crashes, hangs, or leaks over
  long runs; Checkpoint() may be a no-op for scoring but the engine must not corrupt or leak.
  Durability is not part of the throughput score but IS expected for "prod-grade".
- **Ops exercised:** Read + Upsert only (RMW/Delete/Checkpoint are in the interface, unused by id 0).
- **Scored env (exact; anything else invalidates the run):** KVSTORE_VALUE_SIZE=100,
  KVSTORE_ASYNC_EVAL=1, KVSTORE_PIPELINE_DEPTH=256, KVSTORE_COMPLETE_PENDING_INTERVAL=64,
  KVSTORE_AUDIT=1, KVSTORE_VALIDATE_LOAD_KEYS=0, KVSTORE_BIMODAL_VALUES=0, KVSTORE_BENCH_FAST_EXIT=0.
  KVSTORE_MAX_INIT_COUNT and KVSTORE_MAX_TXN_COUNT MUST be unset.
- **Forbidden:** exceeding 14 GiB / swapping; StoreRSS > 8 GiB; network; reading outside this folder;
  precomputing from the traces offline; regenerating/compressing values; returning wrong bytes.

## Interface (implement this; header in include/kvstore_interface.h)
- "extern IKVStore* create_kvstore();" returns your engine.
- Keys uint64; values GenValue{ uint8 data[<=4096]; uint32 size; }.
- Implement InitExtended, StartSession/StopSession, Refresh, Read (sync),
  ReadAsync + CompletePending (async pipelined reads — depth 256; BOTH sync and async are allowed
  and exploiting device queue depth is legitimate engineering), Upsert. RMW/Delete/Checkpoint may be
  stubs. Honor the ReadSlot contract exactly: fill out + status, THEN done.store(1, release).

## How to build & run
- ./build.sh    — compiles harness + baseline/*.cc into kvstore_bench (only include/ on -I; never uses "..").
- ./run.sh      — the SCORED median-of-3 run; prints median Mops/s + StoreRSS.
- ./validate.sh — the separate untimed correctness/integrity run (exit 0 = correct).

## You provide / we provide
- **Provided (fixed, do not modify):** include/kvstore_interface.h, harness/benchmark_harness.cc,
  traces/{load,run}_250M.dat, build.sh, run.sh, validate.sh, this TASK.md.
- **You provide:** your engine sources in baseline/*.cc (defining create_kvstore()), self-contained,
  C++17 + pthread only, nothing named kv_*.h.
