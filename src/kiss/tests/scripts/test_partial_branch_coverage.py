# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Targeted integration tests to cover partial branches in generate_api_docs.

Split out of the root-level ``kiss.tests.test_partial_branch_coverage``
because these tests depend on ``kiss.scripts`` and therefore belong in
``tests/scripts``.

No mocks, test doubles, or fakes.
"""

from __future__ import annotations


class TestGenerateApiDocsBranches:
    """Cover partial branches in generate_api_docs.py."""

    def test_module_to_path_package(self) -> None:
        """Module path resolution for a package covers line 261 is_dir check."""
        from kiss.scripts.generate_api_docs import _module_to_path

        path = _module_to_path("kiss.core")
        assert path.exists()
