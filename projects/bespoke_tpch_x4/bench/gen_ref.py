"""Generate DuckDB reference results for the bench workloads.

Usage: python gen_ref.py <parquet_dir> <sf_label> <seed> [<seed> ...]

Writes bench/ref/sf<sf_label>/seed<seed>/q<id>.csv for every query in the
workload of that seed.
"""

import os
import sys

import duckdb

TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    """Run every workload query in DuckDB and store reference CSVs."""
    parquet_dir = sys.argv[1]
    sf_label = sys.argv[2]
    seeds = sys.argv[3:]

    con = duckdb.connect()
    for table in TABLES:
        path = os.path.join(parquet_dir, f"{table}.parquet")
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
        )

    for seed in seeds:
        workload_dir = os.path.join(BENCH_DIR, "workloads", f"seed{seed}")
        out_dir = os.path.join(BENCH_DIR, "ref", f"sf{sf_label}", f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(workload_dir, "order.txt")) as f:
            query_ids = f.read().split()
        for qid in query_ids:
            with open(os.path.join(workload_dir, f"q{qid}.sql")) as f:
                sql = f.read()
            df = con.execute(sql).fetchdf()
            df.to_csv(os.path.join(out_dir, f"q{qid}.csv"), index=False)
            print(f"sf{sf_label} seed{seed} q{qid}: {len(df)} rows")


if __name__ == "__main__":
    main()
