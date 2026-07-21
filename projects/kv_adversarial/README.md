# HydraKV — high-throughput, larger-than-memory KV store (YCSB-A adversarial task)

HydraKV is a production-hardened C++17 key-value store built for the task in
`benchmark/TASK.md`: beat FASTER (~0.93 Mops/s) on YCSB-A (50% read / 50% blind
upsert, Zipf θ=0.95, 250M keys × 100 B values) under an **enforced 8 GiB** memory
budget on a 64-vCPU GCP `n2-standard-64` with 8× local NVMe in RAID0.

**Final scored result: 5.51 Mops/s median-of-3 (≈5.9× FASTER), StoreRSS ≤ 8 GiB,
zero integrity failures** — see `refnode_rerun_jul21/` and `refnode_workloads_jul21/`
for the raw reference-node logs.

## Directory layout

| Path | What it is |
|---|---|
| `hydra.cc` | The engine (single translation unit; defines `create_kvstore()`). |
| `kvstore_interface.h` | Fixed harness interface (`IKVStore`) — do not modify. |
| `test_hydra.cc` | 28 self-asserting end-to-end tests (~2,000 lines). |
| `run_all_tests.sh` | Full validation matrix: opt / tmpfs / no-uring / ASan+UBSan / TSan / coverage. |
| `benchmark/` | The complete benchmark kit copied from the reference node (scripts, harness, trace generator, task spec). |
| `gen_variant.cc` | Generator for smaller/variant trace files used during hardening. |
| `hydra_iter1_backup.cc` | Historical first-iteration engine (reference only). |
| `KV_TASK.md`, `benchmark/TASK.md` | The original task specification. |
| `AUDIT2.md`, `AUDIT2_FIXES.md` | Independent audit findings and their fixes. |
| `WORKLOAD_HARDENING.md` | The 8 multi-workload bugs found+fixed (x-records, O_DIRECT EOF, shutdown flush, compactor races, recovery, tombstones). |
| `HYDRA_PROD_AUDIT.md`, `PROD_READINESS.md`, `DISCOVERY_LOG.md` | Earlier audit / design / discovery history. |
| `refnode_rerun_jul21/`, `refnode_workloads_jul21/` | Reference-node evidence: scored outputs, workload sweep, sanitizer matrix. |

## 1. Build the engine + tests (any Linux box, or macOS for development)

No external dependencies — C++17 + pthread only. `hydra.cc` uses io_uring and
O_DIRECT on Linux and compiles cleanly on macOS arm64 via a built-in `__APPLE__`
shim (buffered I/O, no uring) for development.

```bash
cd projects/kv_adversarial
g++ -O2 -g -std=c++17 -pthread hydra.cc test_hydra.cc -o test_opt   # clang++ on macOS
```

Test binary usage: `./test_opt <work_dir> [test_filter] [scale]`

```bash
./test_opt /tmp/hydra_tests "" 1        # all 28 tests, scale 1 (~minutes)
./test_opt /tmp/hydra_tests compact 4   # only tests whose name contains "compact", 4x scale
```

Exit 0 + `ALL TESTS PASSED` = pass. Every test is self-asserting; most include a
restart/recovery leg.

## 2. Validate (full test matrix — run this after ANY engine change)

```bash
cd projects/kv_adversarial
./run_all_tests.sh          # Linux; needs g++, gcov, setarch
```

Phases (each prints `PHASE_* PASS/FAIL`, final line `ALL_PHASES_DONE`):
opt scale1, buffered-fd on tmpfs scale4, `HYDRA_NO_URING=1` scale4,
ASan+UBSan scale4, TSan concurrency subset scale8 (zero-warning gate),
and gcov coverage (last measured: 93.3% lines / 95.4% branches).

Overrides: `HYDRA_TEST_DIR=<fast-disk-dir>` (put the work dir on real NVMe),
`HYDRA_SHM_DIR`, `HYDRA_ENGINE`, `HYDRA_INCLUDE_DIR`.

## 3. Set up the benchmark (reference-node protocol)

`benchmark/` mirrors the node's `task/` directory. To reproduce on a fresh machine
(ideally a GCP `n2-standard-64` with 8× local NVMe RAID0 ext4 at `/mnt/ssd` —
provisioning commands are in `benchmark/TASK.md`):

```bash
# a) lay out the kit
cp -r projects/kv_adversarial/benchmark  ~/task
mkdir -p ~/task/baseline ~/task/include
cp projects/kv_adversarial/hydra.cc            ~/task/baseline/
cp projects/kv_adversarial/kvstore_interface.h ~/task/include/

# b) generate the 250M-key traces (~2 GB each, one-time; see benchmark/traces/README.md)
cd ~/task/traces
g++ -O2 -std=c++17 gen_traces.cc -o gen_traces
./gen_traces 250000000 250000000 load_250M.dat run_250M.dat

# c) build the harness + engine
cd ~/task && ./build.sh        # g++ -O3 -march=native -DNDEBUG -flto=auto → kvstore_bench

# d) one-time: user systemd for the cgroup wrapper
loginctl enable-linger "$USER"
```

Harness invocation (what the scripts run):
`kvstore_bench <workload_id> <threads> <load.dat> <run.dat> <budget_bytes> <spill_dir>`

## 4. Run the SCORED benchmark

```bash
cd ~/task
KV_SPILL=/mnt/ssd/$USER/kv_spill ./run.sh
```

`run.sh` is the exact scoring protocol: median of 3 × (drop_caches best-effort →
user cgroup `MemoryMax=14GiB`, no swap → `numactl --cpunodebind=0 --membind=0` →
16 threads, 8 GiB engine budget, `KVSTORE_AUDIT=1` integrity auditor, 30 s timed
window). It **enforces** StoreRSS ≤ 8 GiB (over-budget ⇒ score 0), wraps each run
in a 1800 s timeout (hang ⇒ score 0), and prints per-run Mops/s + `MEDIAN:`.
Per-run logs land in `run_1..3.log`. Expected on the reference node: **≥ 5.5 Mops/s**.

## 5. Correctness validation run (untimed)

```bash
cd ~/task && KV_SPILL=/mnt/ssd/$USER/kv_spill ./validate.sh
```

Full 250M-key load with `KVSTORE_VALIDATE_LOAD_KEYS=1 KVSTORE_VALIDATE_LOAD_STRIDE=1`
(every-key stride-1 read-back + FNV-1a value checksums + cross-thread auditor).
Exit 0 = correct; 2 = value/load mismatch; 3 = in-run integrity failure.

## 6. Other workload mixes / environment variants

`run_wl.sh <workload_id> <tag> [ENV=VAL ...]` runs one full pass with the scored
protocol and writes `wl_<tag>.log` (prints rc, Mops/s, StoreRSS, and any
integrity/error lines). Workload IDs (harness `workload_id` argument):

| id | mix | reference result (250M keys, 16 thr, 8 GiB) |
|---|---|---|
| 0 | A 50:50 read:write (SCORED) | 5.51 Mops/s |
| 1 | RMW 100% read-modify-write | 0.30 Mops/s (sync-read latency-bound) |
| 2 | B 95:5 | 2.93 Mops/s |
| 3 | C 100:0 (read-only) | 2.78 Mops/s |
| 4 | W 0:100 (write-only) | 5.12 Mops/s |
| 5 | TIMESERIES_HD (delete-heavy) | 0.08 Mops/s (correctness perfect) |
| 6 | 5:95 read:write | 5.16 Mops/s |

Examples (all previously verified clean — zero integrity failures):

```bash
./run_wl.sh 4 w0100                                  # write-only
./run_wl.sh 6 w9505                                  # 5:95
./run_wl.sh 0 v1k  KVSTORE_VALUE_SIZE=1024           # 1 KiB values (x-record path)
./run_wl.sh 0 bim  KVSTORE_BIMODAL_VALUES=1          # 20 B / 200 B bimodal values
./run_wl.sh 0 sync KVSTORE_ASYNC_EVAL=0              # synchronous read path
KV_THREADS=32 ./run_wl.sh 0 t32                      # thread-count variant
KV_LOAD=traces/load_25M.dat KV_RUN=traces/run_25M.dat ./run_wl.sh 0 small  # small traces
```

Useful knobs — harness: `KVSTORE_VALUE_SIZE`, `KVSTORE_ASYNC_EVAL`,
`KVSTORE_PIPELINE_DEPTH`, `KVSTORE_AUDIT`, `KVSTORE_BIMODAL_VALUES`,
`KVSTORE_VALIDATE_LOAD_KEYS/_STRIDE`; runner: `KV_SPILL`, `KV_THREADS`,
`KV_BUDGET`, `KV_LOAD`, `KV_RUN`; engine: `HYDRA_NO_URING=1` (disable io_uring),
`HYDRA_STATS=1` (shutdown triage stats: `recover_ok`, `rejected_mem`, `tomb_ref`,
`nf_walk`, `read_errors`, …), `HYDRA_COMPACT_FLOOR_MB` / `HYDRA_COMPACT_FACTOR`
(compaction stress tuning, used by tests).

## Known, documented limits (honest, not silent — details in WORKLOAD_HARDENING.md)

- RAM-index floor: the index needs ~8 B/key within budget/4 — a 2 GiB budget cannot
  index 250M keys; init now prints a LOUD warning with the minimum budget.
- x-extents (values > 101 B) are never compacted (bounded leak class, counted).
- Delete throughput is poison-I/O bound (timeseries mix); correctness is perfect.
- 32-bit slot-position ceiling ⇒ 512 GiB max log; overflow terminates in counted,
  honest rejection (`rejected_mem`), never corruption.
