# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Targeted integration tests to cover partial branches across the codebase.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.test_partial_branch_coverage``; the non-core tests remain there.
"""

from __future__ import annotations

import pytest


class TestLazyImportNotInMap:
    """Cover the else branch where name is NOT in _LAZY_IMPORTS."""

    def test_raises_attribute_error(self) -> None:
        """Accessing a name NOT in _LAZY_IMPORTS raises AttributeError."""
        import kiss.core.models as models_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(models_mod, "NonExistentModel")
