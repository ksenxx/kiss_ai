"""Reference evaluator, TusoAI contract.

Usage:
    python eval_runner.py <benchmark_name> <method_module>

Trains the method on the benchmark's TRAIN split, scores on VALIDATION, and
prints the score in TusoAI's exact format so the optimizer can parse it:

    tuso_evaluate: <score>

This is the only signal the discovery loop optimizes against; the sealed test
split is never touched here.
"""

from __future__ import annotations

import sys

from datagen import BENCHMARKS, make_benchmark
from harness import score_validation_module


def main() -> None:
    """Parse args, evaluate on validation, and print the TusoAI score line."""
    if len(sys.argv) != 3:
        print("usage: python eval_runner.py <benchmark_name> <method_module>", file=sys.stderr)
        raise SystemExit(2)
    benchmark_name, method_module = sys.argv[1], sys.argv[2]
    if benchmark_name not in BENCHMARKS:
        print(f"unknown benchmark {benchmark_name!r}; choose from {BENCHMARKS}", file=sys.stderr)
        raise SystemExit(2)
    ds = make_benchmark(benchmark_name, master_seed=0)
    score = score_validation_module(method_module, ds)
    print(f"tuso_evaluate: {float(score):.17f}")


if __name__ == "__main__":
    main()
