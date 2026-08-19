# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Enforce the sorcar-layer packaging invariant.

Moved from the root-level ``kiss.tests.test_layering_invariants``
(a module that no longer exists)
because this test depends only on ``kiss.core`` and
``kiss.agents.sorcar`` (plus the core-only shared AST import scanner
imported below from ``kiss.tests.core.test_layering_invariants``,
which also owns the core-layer half of the invariant).

The invariant (user-specified) MUST always hold: code in
``src/kiss/agents/sorcar/`` MUST NOT depend on any code outside
``src/kiss/agents/sorcar/`` except code in ``src/kiss/core/``.
"""

from __future__ import annotations

from kiss.tests.core.test_layering_invariants import KISS_ROOT, _violations


def test_sorcar_depends_only_on_sorcar_and_core() -> None:
    """Sorcar code must import only ``kiss.core`` and ``kiss.agents.sorcar``."""
    violations = _violations(
        KISS_ROOT / "agents" / "sorcar",
        ("kiss.core", "kiss.agents.sorcar"),
    )
    assert not violations, (
        "kiss.agents.sorcar must not depend on code outside "
        "src/kiss/agents/sorcar/ and src/kiss/core/:\n"
        + "\n".join(violations)
    )
