"""Benchmark + correctness harness for the bespoke TPC-H engine.

Usage:
    python run_bench.py <engine_binary> <parquet_dir> <sf_label> <seed>
        [--reps N] [--no-validate] [--out results.json]

Runs the engine once (single process: load + build + all queries repeated
N times), parses per-query execution times, validates the first repetition's
result CSVs against the DuckDB reference results with the paper's tolerance
policy (all-column-sorted frames, atol=rtol=1e-2), and prints a JSON summary
with per-query median times and their total.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile

import pandas as pd

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def load_workload(seed: str) -> tuple[list[str], list[str]]:
    """Return (arg lines, query ids) for a seeded workload."""
    workload_dir = os.path.join(BENCH_DIR, "workloads", f"seed{seed}")
    with open(os.path.join(workload_dir, "args.txt")) as f:
        args_lines = [line for line in f.read().splitlines() if line.strip()]
    with open(os.path.join(workload_dir, "order.txt")) as f:
        query_ids = f.read().split()
    assert len(args_lines) == len(query_ids)
    return args_lines, query_ids


def compare_frames(ref: pd.DataFrame, impl: pd.DataFrame) -> str | None:
    """Compare result frames using the paper's validator policy.

    Both frames are sorted by all columns and compared cell-wise with
    atol=rtol=1e-2 for numeric data. Returns None when equal, otherwise a
    short error description.
    """
    if len(ref) != len(impl):
        return f"row count mismatch: ref={len(ref)} impl={len(impl)}"
    if len(ref.columns) != len(impl.columns):
        return (
            f"column count mismatch: ref={len(ref.columns)} "
            f"impl={len(impl.columns)}"
        )
    impl = impl.copy()
    impl.columns = list(ref.columns)
    if len(ref) == 0:
        return None
    ref_sorted = ref.sort_values(by=list(ref.columns)).reset_index(drop=True)
    impl_sorted = impl.sort_values(by=list(impl.columns)).reset_index(drop=True)
    for col in ref.columns:
        ref_col = ref_sorted[col]
        impl_col = impl_sorted[col]
        if pd.api.types.is_numeric_dtype(ref_col):
            impl_num = pd.to_numeric(impl_col, errors="coerce")
            ref_num = ref_col.astype(float)
            bad = []
            for i, (a, b) in enumerate(zip(ref_num, impl_num)):
                a_nan = a is None or (isinstance(a, float) and math.isnan(a))
                b_nan = b is None or (isinstance(b, float) and math.isnan(b))
                if a_nan and b_nan:
                    continue
                if a_nan != b_nan:
                    bad.append(i)
                elif not math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2):
                    bad.append(i)
            if bad:
                i = bad[0]
                return (
                    f"column '{col}': {len(bad)} mismatches, first at row "
                    f"{i}: ref={ref_num[i]} impl={impl_num[i]}"
                )
        else:
            ref_str = ref_col.astype(str).str.strip()
            impl_str = impl_col.astype(str).str.strip()
            diff = ref_str != impl_str
            if diff.any():
                i = int(diff.idxmax())
                return (
                    f"column '{col}': {int(diff.sum())} mismatches, first at "
                    f"row {i}: ref={ref_str[i]!r} impl={impl_str[i]!r}"
                )
    return None


def validate(
    run_dir: str, query_ids: list[str], sf_label: str, seed: str
) -> dict[str, str | None]:
    """Validate result<N>.csv files of the first repetition against refs."""
    ref_dir = os.path.join(BENCH_DIR, "ref", f"sf{sf_label}", f"seed{seed}")
    errors: dict[str, str | None] = {}
    for idx, qid in enumerate(query_ids, start=1):
        ref_path = os.path.join(ref_dir, f"q{qid}.csv")
        impl_path = os.path.join(run_dir, f"result{idx}.csv")
        if not os.path.exists(impl_path):
            errors[qid] = "missing result file"
            continue
        ref = pd.read_csv(ref_path)
        try:
            impl = pd.read_csv(impl_path)
        except pd.errors.EmptyDataError:
            impl = pd.DataFrame(columns=ref.columns)
        errors[qid] = compare_frames(ref, impl)
    return errors


def main() -> None:
    """Run the benchmark and print the JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("parquet_dir")
    parser.add_argument("sf_label")
    parser.add_argument("seed")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    args_lines, query_ids = load_workload(args.seed)
    stdin_data = "\n".join(args_lines * args.reps) + "\n\n"

    binary = os.path.abspath(args.binary)
    parquet_dir = os.path.abspath(args.parquet_dir)
    if not parquet_dir.endswith("/"):
        parquet_dir += "/"

    with tempfile.TemporaryDirectory(prefix="bespoke_run_") as run_dir:
        proc = subprocess.run(
            [binary, parquet_dir],
            input=stdin_data,
            capture_output=True,
            text=True,
            cwd=run_dir,
            timeout=3600,
        )
        if proc.returncode != 0:
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            print(json.dumps({"ok": False, "error": "engine crashed"}))
            sys.exit(1)

        # parse "N | Execution ms: X" lines
        times: dict[int, float] = {}
        for line in proc.stdout.splitlines():
            if "| Execution ms:" in line:
                left, right = line.split("| Execution ms:")
                times[int(left.strip())] = float(right.strip())

        n_queries = len(query_ids)
        expected = n_queries * args.reps
        if len(times) != expected:
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"expected {expected} timings, got {len(times)}"
                        ),
                    }
                )
            )
            sys.exit(1)

        ingest_ms = None
        for line in proc.stderr.splitlines():
            if line.startswith("Ingest ms:"):
                ingest_ms = float(line.split(":")[1].strip())

        per_query: dict[str, dict] = {}
        total_median = 0.0
        for pos, qid in enumerate(query_ids):
            samples = [
                times[rep * n_queries + pos + 1] for rep in range(args.reps)
            ]
            med = statistics.median(samples)
            per_query[qid] = {"median_ms": med, "samples_ms": samples}
            total_median += med

        errors = None
        n_failed = 0
        if not args.no_validate:
            errors = validate(run_dir, query_ids, args.sf_label, args.seed)
            n_failed = sum(1 for e in errors.values() if e is not None)

        summary = {
            "ok": n_failed == 0,
            "sf": args.sf_label,
            "seed": args.seed,
            "reps": args.reps,
            "ingest_ms": ingest_ms,
            "total_median_ms": round(total_median, 3),
            "per_query": {
                q: round(v["median_ms"], 3) for q, v in per_query.items()
            },
            "samples": {q: v["samples_ms"] for q, v in per_query.items()},
            "validation_errors": (
                {q: e for q, e in (errors or {}).items() if e is not None}
                if errors is not None
                else "skipped"
            ),
        }

    out = json.dumps(summary, indent=1)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    sys.exit(0 if summary["ok"] or args.no_validate else 2)


if __name__ == "__main__":
    main()
