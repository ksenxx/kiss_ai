"""Final benchmark report for a candidate method.

Prints validation + sealed-test scores for every benchmark, plus a
generalization report (all benchmarks rebuilt from a different master seed).
Exits non-zero unless every sealed-test score AND every generalization score is
>= the pass threshold (default 99), so this doubles as the acceptance gate.

Usage:
    python run_benchmarks.py [method_module] [threshold]
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

from harness import evaluate_all

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main() -> None:
    """Evaluate a method, print a report, and enforce the >=threshold gate."""
    method_module = sys.argv[1] if len(sys.argv) > 1 else "methods.tuso_evolved"
    try:
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 99.0
    except ValueError:
        print(f"invalid threshold: {sys.argv[2]!r}", file=sys.stderr)
        raise SystemExit(2)
    if not math.isfinite(threshold):
        print(f"threshold must be finite, got {threshold!r}", file=sys.stderr)
        raise SystemExit(2)

    primary = evaluate_all(method_module, master_seed=0)
    generalization = evaluate_all(method_module, master_seed=7)

    report = {
        "method": method_module,
        "threshold": threshold,
        "primary": [
            {"benchmark": r.name, "validation": r.validation, "test": r.test}
            for r in primary
        ],
        "generalization": [
            {"benchmark": r.name, "test": r.test} for r in generalization
        ],
    }

    print(f"\n=== Biomni x TusoAI benchmark report ({method_module}) ===")
    print(f"{'benchmark':<26}{'validation':>12}{'sealed_test':>14}{'generalize':>13}")
    gmap = {r.name: r.test for r in generalization}
    for r in primary:
        print(f"{r.name:<26}{r.validation:>12.3f}{r.test:>14.3f}{gmap[r.name]:>13.3f}")

    all_test = [r.test for r in primary]
    all_gen = list(gmap.values())
    worst = min(all_test + all_gen)
    passed = worst >= threshold

    print(f"\nworst score across sealed-test + generalization: {worst:.3f}")
    print("RESULT:", "PASS ✅" if passed else "FAIL ❌", f"(threshold {threshold})")

    # Atomic write into a guaranteed-existing, script-relative directory so a
    # fresh checkout (or a crash mid-write) can never lose the gate's verdict
    # or leave a truncated report behind. The temporary file name is unique
    # per invocation so concurrent runs cannot clobber each other's writes.
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    final_path = os.path.join(_RESULTS_DIR, "last_report.json")
    fd, tmp_path = tempfile.mkstemp(dir=_RESULTS_DIR, prefix="last_report.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(report, fh, indent=2)
        os.replace(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
