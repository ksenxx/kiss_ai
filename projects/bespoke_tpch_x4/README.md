# bespoke_tpch_x4 — 34× faster than the Bespoke OLAP paper engine

This project reproduces the released TPC-H engine of the VLDB'26 paper
*"Bespoke OLAP: Synthesizing Workload-Specific One-size-fits-one Database
Engines"* (arXiv 2603.02001, artifact repo `DataManagementLab/BespokeOLAP_Artifacts`)
and then improves its total query runtime by **~34× at SF10** (goal was 4×)
through an AI-discovery optimization loop, while preserving correctness
against DuckDB reference results.

## Results (32-core machine, sum of per-query median execution times)

| Configuration | Paper engine (single-threaded) | This engine | Speedup |
|----------------------------|-------------------------------:|------------:|--------:|
| TPC-H SF10, seed 42 | 3132.2 ms | 90.8 ms | 34.5× |
| TPC-H SF10, seed 7 | — | 92.5 ms | |
| TPC-H SF10, seed 123 | — | 82.1 ms | |
| TPC-H SF10, seed 999 (held-out) | — | 85.1 ms | |
| TPC-H SF1, seed 42 | 276.8 ms | 14.2 ms | 19.5× |

The paper's own *multithreaded* artifact measures 284 ms total at SF10 seed42
on the same machine (timing-only comparison), i.e. this engine is ~3× faster
than the authors' multithreaded follow-up as well.

All configurations validate cleanly against DuckDB (policy from the paper's
validator: results sorted by all columns, atol=rtol=1e-2). Ingest time grew
only from 52 s to ~62 s at SF10.

## What was changed (discovered by the optimization loop)

1. Removed the artifact's `AffinityGuard` that pinned the whole process to one
   CPU; added `-fopenmp -march=native` (`engine/extra_flags.txt`).
1. Parameter-independent precomputation in the builder (ingest time), exploiting
   the fixed query templates of the paper's "DBMS contract": date-indexed prefix
   tables/cubes (Q1, Q4-Q8, Q12, Q14, Q15), partkey CSR join indexes (Q9, Q17,
   Q20), per-order aggregates (Q10, Q18), precomputed late-supplier and
   filtered-row artifacts (Q13, Q16, Q19, Q21, Q22).
1. OpenMP-parallel kernels for everything that still scans at query time
   (thread-local partials + merge, atomic dense accumulation, parallel sorts).
1. Two post-review fixes (found by a read-only `gpt-5.6-sol` review): an OpenMP
   race in Q20 (now `omp atomic write`) and a narrowed Q1 `sum_charge`
   accumulator (restored to `__int128`).

## Layout

- `engine_baseline/` — pristine paper artifact (only a measurement-precision
  patch: timing printed as `double` ms instead of integer ms; applied before
  baseline measurement, identical timing window).
- `engine/` — the optimized engine (same build interface + stdin protocol +
  CSV output as the baseline).
- `harness/main.cpp` — standalone host: load parquet → build database → run
  queries from stdin, print `N | Execution ms: <t>` per query.
- `bench/` — benchmark + validation harness:
  - `bench.sh <engine_dir> <sf> [seed] [reps]` — build, run, validate, JSON out.
  - `build.sh` — paper compile flags (`-g -std=c++20 -O3 -flto`) plus optional
    `<engine_dir>/extra_flags.txt`.
  - `run_bench.py` — runs the workload N times, reports per-query medians and
    validates first-repetition CSVs against `bench/ref/`.
  - `gen_ref.py` — regenerates DuckDB reference results
    (`python gen_ref.py <parquet_dir> <sf_label> <seed>...`). SF10 references
    (~268 MB) are gitignored; regenerate before validating at SF10.
  - `workloads/seed*/` — query instantiations produced by the paper's own
    generator (seed 42 matches the paper runner exactly).
- `results/` — measured JSONs: `baseline_*` (paper engine) and `final_*`
  (optimized engine, post-fix).

## Reproducing

```bash
# 1. TPC-H parquet data (DuckDB tpch extension), sf1 + sf10:
#    <repo_root>/tmp/data/tpch_parquet/sf{1,10}/{customer,...}.parquet
#    (or set TPCH_DATA_DIR)
# 2. References: uv run --with duckdb --with pandas python3 bench/gen_ref.py \
#      <data>/sf10 10 42 7 123 999
# 3. Baseline:   ./bench/bench.sh engine_baseline 10 42 5
# 4. Optimized:  ./bench/bench.sh engine 10 42 5
```

Requires Apache Arrow/Parquet C++ dev libraries and g++ with OpenMP.
