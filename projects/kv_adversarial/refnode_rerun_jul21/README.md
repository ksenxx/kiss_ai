# July 21 — Reference-node scored re-run of the AUDIT2-fixed engine

Raw logs from the confirmation re-run requested by `AUDIT2_FIXES.md`, executed
on the AUDIT2 reference box (GCP n2-standard-64, Ubuntu 22.04.5, kernel
6.8.0-1063-gcp, g++ 11.4, 8× 375 GB local NVMe RAID0 `/dev/md0` at
`/mnt/ssd`). Protocol identical to the original scored run: `run.sh`
median-of-3, 250M-key YCSB-A traces (same files by inode as the July-20 scored
run), 16 threads on NUMA node 0, user cgroup MemoryMax=14 GiB / no swap,
KVSTORE_VALUE_SIZE=100, KVSTORE_AUDIT=1, StoreRSS ≤ 8 GiB enforced, 30 s
window. Engine under test: `hydra.cc` @ commit c7745c61 (all AUDIT2 fixes),
md5 b6d4b6e3eab4ef3e1db88a678c847a6c.

| File | What it is | Result |
|---|---|---|
| `scored_fixed_A.out` | Fixed engine, scored suite A | 5.44 / 5.50 / 5.50 → **median 5.50 Mops/s** |
| `scored_fixed_B.out` | Fixed engine, scored suite B | 5.52 / 5.50 / 5.49 → **median 5.50 Mops/s** |
| `scored_old.out` | Same-day A/B control: ORIGINAL pre-fix engine (md5 4f84fd21…, the exact `hydra.cc` that scored 5.51 on July 20) | 5.50 / 5.50 / 5.49 → **median 5.50 Mops/s** |
| `run_2.log` | Representative full harness log (fixed engine, suite B run 2) | Total 5.50 Mops/s, StoreMemUtil **7.54 / 8.00 GB (94%)**, integrity audit **0 failures** |
| `linux_matrix_summary.txt` | Full Linux `run_all_tests.sh` matrix on the same node (fixed engine) | ALL PHASES PASS (opt / buffered-fd / no-uring / ASan+UBSan / TSan zero warnings / coverage 92.52% lines) |

Conclusion: the fixed engine's same-day median (5.50) equals the pre-fix
engine's same-day median (5.50) — the 0.01 Mops/s difference vs the July-20
5.51 is day-to-day machine noise, not a regression. Memory (7.54 GB
StoreMemUtil, StoreRSS ≈ 7.90 GB ≤ 8 GiB) and integrity results are identical
to the original scored run. **The 5.51 Mops/s / 7.54 GiB result is preserved.**
