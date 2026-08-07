"""Adversarial fast-path audit harness for the optimized bespoke TPC-H engine.

Runs a manifest of adversarial query requests through BOTH the optimized
engine (compiled with -DBESPOKE_AUDIT so each fast-path / fallback branch
reports an ``AUDIT <marker>`` stderr line) and the pristine paper baseline
engine, then

  1. diffs each ``result<i>.csv`` pair with the paper's validation policy
     (all-column-sorted frames, atol=rtol=1e-2) imported from
     bench/run_bench.py, and
  2. asserts the execution-path markers each request was expected to take
     (e.g. that a guard really falls back to the baseline scan for
     out-of-range parameters).

Manifest format: JSON lines with keys
  line    (str)  raw request line fed to both engines, e.g. '3 "AUTO" "1992-01-01"'
  expect  (list) markers that must appear among the request's AUDIT lines
  forbid  (list) markers that must NOT appear (optional)
  note    (str)  human explanation of the edge case (optional)
Blank lines and lines starting with '#' are ignored.

Usage:
  uv run --no-project --with duckdb --with pandas python3 tests/difftest.py \
      <manifest.jsonl> [--data <parquet_dir>] [--bin-dir tests/bin] [-v]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from run_bench import compare_frames  # noqa: E402

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)), "tmp", "data", "tpch_parquet", "sf1"
)


def load_manifest(path):
    """Read the JSONL manifest, skipping blanks and '#' comments."""
    cases = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            case = json.loads(raw)
            case.setdefault("expect", [])
            case.setdefault("forbid", [])
            case.setdefault("note", "")
            case["lineno"] = lineno
            cases.append(case)
    return cases


def run_engine(binary, parquet_dir, stdin_data, run_dir):
    """Run one engine over the whole request batch; return (stdout, stderr)."""
    proc = subprocess.run(
        [binary, parquet_dir],
        input=stdin_data,
        capture_output=True,
        text=True,
        cwd=run_dir,
        timeout=3600,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"{binary} exited with {proc.returncode}")
    return proc.stdout, proc.stderr


def parse_audit_markers(stderr_text, n_requests):
    """Split 'AUDIT ...' stderr lines into a per-request list of markers."""
    markers = [[] for _ in range(n_requests)]
    current = None
    for line in stderr_text.splitlines():
        if not line.startswith("AUDIT "):
            continue
        body = line[len("AUDIT ") :].strip()
        if body.startswith("BEGIN "):
            current = int(body.split()[1]) - 1
            continue
        if current is not None and 0 <= current < n_requests:
            markers[current].append(body)
    return markers


def read_result_csv(run_dir, idx):
    """Read result<idx>.csv (1-based); empty file -> empty frame."""
    path = os.path.join(run_dir, f"result{idx}.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def marker_matches(pattern, marker):
    """Exact marker match, or prefix match when pattern ends with '*'.

    Exact matching (rather than substring) prevents an assertion like
    'q13 fast' from silently passing on a different branch's marker such as
    'q13 fast masked'. A trailing '*' opts into matching a whole marker
    family, e.g. 'q9 fallback*'.
    """
    if pattern.endswith("*"):
        return marker.startswith(pattern[:-1])
    return marker == pattern


def check_case(case, idx, base_dir, opt_dir, opt_markers):
    """Return a list of failure strings for request #idx (1-based)."""
    failures = []
    ref = read_result_csv(base_dir, idx)
    impl = read_result_csv(opt_dir, idx)
    if ref is None:
        failures.append("baseline produced no result file")
    elif impl is None:
        failures.append("optimized engine produced no result file")
    else:
        # compare_frames also validates row/column counts for empty frames,
        # so call it unconditionally (no both-empty shortcut: an empty-result
        # schema regression must still be caught).
        err = compare_frames(ref, impl)
        if err:
            failures.append(f"result mismatch vs baseline: {err}")
    got = opt_markers[idx - 1]
    for marker in case["expect"]:
        if not any(marker_matches(marker, m) for m in got):
            failures.append(f"expected path marker {marker!r} not hit (got {got})")
    for marker in case["forbid"]:
        if any(marker_matches(marker, m) for m in got):
            failures.append(f"forbidden path marker {marker!r} was hit (got {got})")
    return failures


def main():
    """Run the differential audit over a manifest and print PASS/FAIL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--bin-dir", default=os.path.join(TESTS_DIR, "bin"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cases = load_manifest(args.manifest)
    if not cases:
        print("manifest has no cases")
        sys.exit(1)
    stdin_data = "\n".join(c["line"] for c in cases) + "\n\n"
    parquet_dir = os.path.abspath(args.data)
    if not parquet_dir.endswith("/"):
        parquet_dir += "/"

    audit_bin = os.path.join(args.bin_dir, "db_audit")
    base_bin = os.path.join(args.bin_dir, "db_baseline")
    n_fail = 0
    with (
        tempfile.TemporaryDirectory(prefix="audit_base_") as base_dir,
        tempfile.TemporaryDirectory(prefix="audit_opt_") as opt_dir,
    ):
        run_engine(base_bin, parquet_dir, stdin_data, base_dir)
        _, opt_err = run_engine(audit_bin, parquet_dir, stdin_data, opt_dir)
        opt_markers = parse_audit_markers(opt_err, len(cases))
        for idx, case in enumerate(cases, start=1):
            failures = check_case(case, idx, base_dir, opt_dir, opt_markers)
            label = f"[{idx:3d}] {case['line']}"
            if case["note"]:
                label += f"   ({case['note']})"
            if failures:
                n_fail += 1
                print(f"FAIL {label}")
                for f in failures:
                    print(f"       - {f}")
            elif args.verbose:
                print(f"PASS {label}   markers={opt_markers[idx - 1]}")
    print(f"\n{len(cases) - n_fail}/{len(cases)} cases passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
