# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Find redundant tests using branch coverage with dynamic contexts.

A test method is redundant when every coverage item it produces is
also produced by at least one other remaining test method.  Methods
are removed iteratively (smallest item-set first) so the final set is
safe to delete without losing any covered behavior.

Two sources of coverage are merged per test method:

1.  Python branch coverage from a ``coverage.py`` database (``.coverage``)
    generated with ``branch = true`` and
    ``dynamic_context = "test_function"``.  Each covered ``(file,
    from_line, to_line)`` arc becomes one coverage item.

2.  Coverage of ``*.ts``, ``*.js`` and any other program files emitted
    by external coverage tools (c8, nyc, istanbul, jest, vitest,
    playwright/test, ...) in the standard LCOV ``.info`` format.  Each
    LCOV record's test name (``TN:`` directive) is treated as a test
    method name; if a record has no ``TN:`` the LCOV file's stem is
    used instead.  Covered lines (``DA:``), taken branches (``BRDA:``)
    and called functions (``FNDA:``) each become coverage items.
"""

import os
import re
import sys
from collections.abc import Iterable

import coverage

# A coverage item is an opaque, hashable token that uniquely identifies
# one covered behavior (a Python arc, an LCOV line, an LCOV branch, or
# an LCOV function).  The first element is a "kind" tag so items from
# different sources never collide.
CoverageItem = tuple[object, ...]


def _method_name(context: str) -> str:
    """Strip |run, |setup, |teardown suffix to get the test method name."""
    return re.sub(r"\|(run|setup|teardown)$", "", context)


def _load_python_method_items(
    coverage_file: str,
) -> dict[str, set[CoverageItem]]:
    """Load Python branch arcs grouped by test method.

    Arcs from the ``|run``, ``|setup`` and ``|teardown`` contexts of the
    same method are unioned together so a method is only deemed
    redundant when every arc it indirectly exercises is also covered
    elsewhere.
    """
    cov = coverage.Coverage(data_file=coverage_file)
    cov.load()
    data = cov.get_data()
    contexts = sorted(c for c in data.measured_contexts() if c)
    method_items: dict[str, set[CoverageItem]] = {}
    for ctx in contexts:
        method = _method_name(ctx)
        data.set_query_context(ctx)
        items: set[CoverageItem] = set()
        for src_file in data.measured_files():
            file_arcs = data.arcs(src_file)
            if file_arcs:  # pragma: no branch
                for from_line, to_line in file_arcs:
                    items.add(("py_arc", src_file, from_line, to_line))
        if items:  # pragma: no branch
            method_items.setdefault(method, set()).update(items)
    return method_items


def _parse_lcov_file(path: str) -> dict[str, set[CoverageItem]]:
    """Parse one LCOV ``.info`` file into ``{test_name: {items}}``.

    Supported directives:

    * ``TN:<name>``     – test name for subsequent records.
    * ``SF:<file>``     – source file for the current record.
    * ``DA:<line>,<hits>[,<checksum>]`` – line ``line`` was hit
      ``hits`` times.
    * ``BRDA:<line>,<block>,<branch>,<taken>`` – the (``block``,
      ``branch``) branch at ``line`` was taken ``taken`` times.
    * ``FNDA:<hits>,<name>`` – function ``name`` was called ``hits``
      times.
    * ``end_of_record`` – closes one source-file record.

    Lines with zero hits / not-taken branches are ignored: only
    *covered* items participate in redundancy analysis.
    """
    default_test = os.path.splitext(os.path.basename(path))[0]
    method_items: dict[str, set[CoverageItem]] = {}
    cur_test = default_test
    cur_file: str | None = None
    with open(path, encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            tag, _, rest = line.partition(":")
            if tag == "TN":
                cur_test = rest.strip() or default_test
            elif tag == "SF":
                cur_file = rest.strip()
            elif tag == "end_of_record":
                cur_file = None
            elif cur_file is None:
                continue
            elif tag == "DA":
                parts = rest.split(",")
                if len(parts) >= 2 and _int_or_zero(parts[1]) > 0:
                    method_items.setdefault(cur_test, set()).add(
                        ("lcov_line", cur_file, int(parts[0]))
                    )
            elif tag == "BRDA":
                parts = rest.split(",")
                if len(parts) >= 4 and parts[3].strip() not in ("-", "0", ""):
                    method_items.setdefault(cur_test, set()).add(
                        (
                            "lcov_branch",
                            cur_file,
                            parts[0],
                            parts[1],
                            parts[2],
                        )
                    )
            elif tag == "FNDA":
                hits, _, name = rest.partition(",")
                if _int_or_zero(hits) > 0 and name:
                    method_items.setdefault(cur_test, set()).add(
                        ("lcov_func", cur_file, name.strip())
                    )
    return method_items


def _int_or_zero(s: str) -> int:
    """Parse ``s`` as int, returning 0 on failure (e.g. ``-`` in LCOV)."""
    try:
        return int(s.strip())
    except ValueError:
        return 0


def _load_lcov_method_items(
    lcov_files: Iterable[str],
) -> dict[str, set[CoverageItem]]:
    """Aggregate items from one or more LCOV ``.info`` files."""
    method_items: dict[str, set[CoverageItem]] = {}
    for path in lcov_files:
        for test, items in _parse_lcov_file(path).items():
            method_items.setdefault(test, set()).update(items)
    return method_items


def analyze_redundancy(
    coverage_file: str | None = ".coverage",
    lcov_files: Iterable[str] | None = None,
) -> list[str]:
    """Return a sorted list of test method names safe to remove.

    Uses a greedy algorithm: at each step, the method with the
    smallest coverage-item set whose every item is still covered by
    another remaining method is removed.  Iteration terminates when no
    such method exists, guaranteeing that every coverage item
    originally exercised by some test method is still exercised after
    deletion.

    Parameters
    ----------
    coverage_file:
        Path to the ``coverage.py`` database produced with branch
        coverage and ``dynamic_context = "test_function"``.  May be
        ``None`` to skip Python coverage entirely (LCOV-only mode).
    lcov_files:
        Optional iterable of LCOV ``.info`` paths produced by external
        coverage tools for ``*.ts``, ``*.js`` and other program files.
    """
    method_items: dict[str, set[CoverageItem]] = {}
    if coverage_file is not None:
        method_items.update(_load_python_method_items(coverage_file))
    if lcov_files:
        for test, items in _load_lcov_method_items(lcov_files).items():
            method_items.setdefault(test, set()).update(items)

    item_to_methods: dict[CoverageItem, set[str]] = {}
    for method, items in method_items.items():
        for item in items:
            item_to_methods.setdefault(item, set()).add(method)

    remaining = set(method_items)
    redundant: list[str] = []

    changed = True
    while changed:
        changed = False
        candidates = []
        for method in sorted(remaining):
            is_redundant = all(
                len(item_to_methods[item] & remaining) >= 2
                for item in method_items[method]
            )
            if is_redundant:  # pragma: no branch
                candidates.append(method)

        if candidates:  # pragma: no branch
            victim = min(candidates, key=lambda m: len(method_items[m]))
            remaining.discard(victim)
            redundant.append(victim)
            changed = True

    print(f"Total test methods: {len(method_items)}")
    print(f"Redundant (safe to remove): {len(redundant)}")
    for t in sorted(redundant):  # pragma: no branch
        print(f"  REDUNDANT: {t}")
    return sorted(redundant)


def _main(argv: list[str]) -> None:
    """CLI: ``redundancy_analyzer [.coverage] [extra.info ...]``."""
    coverage_file = argv[1] if len(argv) > 1 else ".coverage"
    lcov_files = argv[2:] or None
    analyze_redundancy(coverage_file=coverage_file, lcov_files=lcov_files)


if __name__ == "__main__":
    _main(sys.argv)
