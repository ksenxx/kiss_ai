# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Targeted integration tests to cover partial branches across the codebase.

The ``drain_queue_messages`` tests moved to
``tests/agents/third_party_agents/test_partial_branch_coverage.py`` and
the ``generate_api_docs`` test to
``tests/scripts/test_partial_branch_coverage.py``, per the packaging
invariants for tests depending on those areas.  The remaining test
depends on ``kiss.agents.obsolete``, which no invariant constrains.

No mocks, test doubles, or fakes.
"""

from __future__ import annotations


class TestEscapeInvalidTemplateFieldNames:
    """Cover branches for conversion and format_spec in _escape_fragment."""

    def test_invalid_field_with_conversion_and_spec(self) -> None:
        """Invalid field with conversion+format_spec enters the escape branch (line 92->93)."""
        from kiss.agents.obsolete.gepa.template_utils import (
            escape_invalid_template_field_names,
        )

        result = escape_invalid_template_field_names("{bad!r:>10}", set())
        assert "{{bad!r:>10}}" in result
