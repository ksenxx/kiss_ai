"""Find redundant tests using branch coverage with dynamic contexts.

A test method is redundant if the set of branches (arcs) it covers
(across all its contexts: run, setup, teardown) is a subset of the
branches covered by the remaining test methods.  Methods are removed
iteratively (smallest arc-set first) so the final set is safe to delete
without losing any branch coverage.

Requires: coverage.py database (.coverage) generated with
  branch = true  and  dynamic_context = "test_function"
"""

import re

import coverage


def _method_name(context: str) -> str:
    """Strip |run, |setup, |teardown suffix to get the test method name."""
    return re.sub(r"\|(run|setup|teardown)$", "", context)


def _load_method_arcs(
    coverage_file: str,
) -> dict[str, set[tuple[str, int, int]]]:
    """Load arcs grouped by test method (union of all context suffixes)."""
    cov = coverage.Coverage(data_file=coverage_file)
    cov.load()
    data = cov.get_data()
    contexts = sorted(c for c in data.measured_contexts() if c)
    method_arcs: dict[str, set[tuple[str, int, int]]] = {}
    for ctx in contexts:
        method = _method_name(ctx)
        data.set_query_context(ctx)
        arcs: set[tuple[str, int, int]] = set()
        for src_file in data.measured_files():
            file_arcs = data.arcs(src_file)
            if file_arcs:  # pragma: no branch
                for from_line, to_line in file_arcs:
                    arcs.add((src_file, from_line, to_line))
        if arcs:  # pragma: no branch
            if method not in method_arcs:  # pragma: no branch
                method_arcs[method] = set()
            method_arcs[method].update(arcs)
    return method_arcs


def analyze_redundancy(coverage_file: str = ".coverage") -> list[str]:
    """Return sorted list of test method names that are safe to remove.

    Uses a greedy algorithm that iteratively removes the method with the
    smallest arc set, as long as every arc it covers is also covered by
    at least one other remaining method.  This guarantees that removing
    all returned methods preserves full branch coverage.
    """
    method_arcs = _load_method_arcs(coverage_file)

    arc_to_methods: dict[tuple[str, int, int], set[str]] = {}
    for method, arcs in method_arcs.items():
        for arc in arcs:
            arc_to_methods.setdefault(arc, set()).add(method)

    remaining = set(method_arcs)
    redundant: list[str] = []

    changed = True
    while changed:
        changed = False
        candidates = []
        for method in sorted(remaining):
            is_redundant = all(
                len(arc_to_methods[arc] & remaining) >= 2
                for arc in method_arcs[method]
            )
            if is_redundant:  # pragma: no branch
                candidates.append(method)

        if candidates:  # pragma: no branch
            victim = min(candidates, key=lambda m: len(method_arcs[m]))
            remaining.discard(victim)
            redundant.append(victim)
            changed = True

    print(f"Total test methods: {len(method_arcs)}")
    print(f"Redundant (safe to remove): {len(redundant)}")
    for t in sorted(redundant):  # pragma: no branch
        print(f"  REDUNDANT: {t}")
    return sorted(redundant)


if __name__ == "__main__":
    analyze_redundancy()
