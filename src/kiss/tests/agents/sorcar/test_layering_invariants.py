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
``src/kiss/agents/sorcar/`` except code in ``src/kiss/core/`` and the
SEA contract ``src/kiss/agents/seas/base/`` (the ``BaseSea`` class the
launcher instantiates), which in turn depends on nothing but
``kiss.core`` so the layering stays acyclic.
"""

from __future__ import annotations

from kiss.tests.core.test_layering_invariants import KISS_ROOT, _violations


def test_sorcar_depends_only_on_sorcar_core_and_sea_contract() -> None:
    """Sorcar imports only ``kiss.core``, ``kiss.agents.sorcar`` and ``kiss.agents.seas.base``."""
    violations = _violations(
        KISS_ROOT / "agents" / "sorcar",
        ("kiss.core", "kiss.agents.sorcar", "kiss.agents.seas.base"),
    )
    assert not violations, (
        "kiss.agents.sorcar must not depend on code outside "
        "src/kiss/agents/sorcar/, src/kiss/core/ and src/kiss/agents/seas/base/:\n"
        + "\n".join(violations)
    )


def test_sea_contract_depends_only_on_core() -> None:
    """The SEA contract sorcar imports must not import sorcar (or any SEA) back."""
    violations = _violations(
        KISS_ROOT / "agents" / "seas" / "base",
        ("kiss.core", "kiss.agents.seas.base"),
    )
    assert not violations, (
        "kiss.agents.seas.base must not depend on code outside "
        "src/kiss/agents/seas/base/ and src/kiss/core/:\n"
        + "\n".join(violations)
    )
