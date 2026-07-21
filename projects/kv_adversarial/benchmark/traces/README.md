# Traces (workload data) — not in git

The benchmark needs two trace files (~2 GB each), each a flat little-endian array of `uint64`
keys (the harness derives the key count from filesize / 8):

- `load_250M.dat` — 250M keys: the dense load set (a permutation of 0..N-1, each key once).
- `run_250M.dat` — 250M access keys: Zipf θ=0.95, scrambled (hotness uncorrelated with key value).

## Option A — generate them yourself (self-contained, recommended)

```
g++ -O2 -std=c++17 gen_traces.cc -o gen_traces
./gen_traces 250000000 250000000 load_250M.dat run_250M.dat   # ~2 GB each, needs ~2 GB RAM
```

This produces a fresh **equivalent** workload (same distribution/skew) — throughput is comparable.
Put both `.dat` files in this directory.

## Option B — use the exact reference files (to reproduce a specific number)

Identical bytes to our runs, if you want to match an exact result:

| file | size | md5 |
|---|---|---|
| `load_250M.dat` | 2,000,000,000 B | `8aee9e1d76995797a0fbbe22ce226b48` |
| `run_250M.dat` | 2,000,000,000 B | `90f7d4de090d68428583c10189837c02` |

- **Download:** `<DUMMY_DOWNLOAD_LINK>` ← replace with the link we send you.
- **or SCP (on request):** `scp <user>@<host>:/path/*.dat  task/traces/`
- **Verify:** `md5sum task/traces/*.dat` → must match the table.

Either way, both files must end up in `task/traces/` — that's all the harness needs.
