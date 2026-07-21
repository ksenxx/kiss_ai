# Reference-node verification — multi-workload hardening (July 21)

Node: the AUDIT2 reference box (GCP n2-standard-64, 64 vCPU Cascade Lake,
251 GB RAM, Ubuntu 22.04, 8x NVMe RAID0). Scored protocol identical to the
original scored run: run.sh median-of-3, 16 threads NUMA node 0, 14 GiB
cgroup, 100 B values, integrity auditor ON, StoreRSS <= 8 GiB enforced.

Engines (see `../WORKLOAD_HARDENING.md` for the fixes):
- chain5 = commit 4138924c (tombstone key-verify + capwarn).
- chain6/chain7 = commit 5050e98f (FINAL: + review fixes M1/M2/minors).

Files:
- `chain5.out` — engine 4138924c: TSan compact loop 15/15 rc=0 (the
  previously-flaky combo), full matrix ALL_PHASES_DONE, scored
  5.54/5.49/5.52 -> median 5.52 Mops/s, bimodal/v1k/0:100/5:95 spot
  checks all rc=0 with 0 integrity failures.
- `chain6.out` — FINAL engine unit verification: TSan compact loop 15/15
  rc=0, matrix rc=0 (`matrix6.out`: opt/shm/no-uring/ASan+UBSan/TSan/
  coverage all PASS, 89.30% lines executed). Its Phase-3/4 numbers were
  produced by a stale kvstore_bench and are superseded by chain7.
- `chain7.out` / `scored_fix7.out` — FINAL engine, freshly built
  kvstore_bench (tomb_ref= present in stats proves the new engine):
  - SCORED: 5.51 / 5.52 / 5.49 -> **MEDIAN 5.51 Mops/s >= 5.5**,
    StoreRSS ~7.9 GB ok on every run.
  - bimodal (20 B/200 B): rc=0, 0 integrity failures, nf_walk=0,
    tomb_ref=0, read_err=103 (benign fail-soft of 4.5M read IOPS).
  - VALUE_SIZE=1024 (25M keys): rc=0, 0 failures, read_err=0.
  - W_0_100 (user's 0:100): 5.12 Mops/s, 0 failures.
  - W95_5_WRITE (user's 5:95): 5.16 Mops/s, 0 failures.
  - TIMESERIES_HD (delete-heavy): rc=0, 0 failures.
- `matrix5.out` / `matrix6.out` — full run_all_tests.sh outputs.

Raw per-run harness logs remain on the node under
/mnt/ssd/ksen/kv50-benchmark/task (wl_*7.log, run_*.log).
