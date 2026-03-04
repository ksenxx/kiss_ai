"""Find redundant tests using branch coverage with dynamic contexts.

A test is redundant if the set of branches (arcs) it covers is a subset
of the branches covered by the remaining tests in the suite.  Tests are
removed iteratively (smallest arc-set first) so the final set is safe
to delete without losing any branch coverage.

Requires: coverage.py database (.coverage) generated with
  branch = true  and  dynamic_context = "test_function"
"""

import coverage


def _load_test_arcs(
    coverage_file: str,
) -> dict[str, set[tuple[str, int, int]]]:
    cov = coverage.Coverage(data_file=coverage_file)
    cov.load()
    data = cov.get_data()
    contexts = sorted(c for c in data.measured_contexts() if c)
    test_arcs: dict[str, set[tuple[str, int, int]]] = {}
    for ctx in contexts:
        data.set_query_context(ctx)
        arcs: set[tuple[str, int, int]] = set()
        for src_file in data.measured_files():
            file_arcs = data.arcs(src_file)
            if file_arcs:
                for from_line, to_line in file_arcs:
                    arcs.add((src_file, from_line, to_line))
        if arcs:
            test_arcs[ctx] = arcs
    return test_arcs


def analyze_redundancy(coverage_file: str = ".coverage") -> list[str]:
    test_arcs = _load_test_arcs(coverage_file)

    arc_to_tests: dict[tuple[str, int, int], set[str]] = {}
    for ctx, arcs in test_arcs.items():
        for arc in arcs:
            arc_to_tests.setdefault(arc, set()).add(ctx)

    remaining = set(test_arcs)
    redundant: list[str] = []

    changed = True
    while changed:
        changed = False
        candidates = []
        for ctx in sorted(remaining):
            is_redundant = all(
                len(arc_to_tests[arc] & remaining) >= 2
                for arc in test_arcs[ctx]
            )
            if is_redundant:
                candidates.append(ctx)

        if candidates:
            victim = min(candidates, key=lambda c: len(test_arcs[c]))
            remaining.discard(victim)
            redundant.append(victim)
            changed = True

    print(f"Total test contexts: {len(test_arcs)}")
    print(f"Redundant (safe to remove): {len(redundant)}")
    for t in sorted(redundant):
        print(f"  REDUNDANT: {t}")
    return sorted(redundant)


if __name__ == "__main__":
    analyze_redundancy()
