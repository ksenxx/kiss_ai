# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (findings-2 audit), channel-CLI helper regression.

* S2-22 — ``--max_budget`` rejects nan/inf/zero/negative values.

The persistence, git_worktree and skills findings of the same audit
(S2-01/03/04/05/30/07) are covered in
``tests/agents/sorcar/test_bughunt9_findings2_core.py``.
"""

from __future__ import annotations

import pytest

from kiss.agents.third_party_agents._channel_cli import _parse_budget_value


class TestBudgetValidation:
    """S2-22: nan/inf/zero/negative budgets must be rejected."""

    def test_rejects_non_finite_and_non_positive(self) -> None:
        import argparse

        for bad in ("nan", "inf", "-inf", "0", "-3", "NaN"):
            with pytest.raises(argparse.ArgumentTypeError):
                _parse_budget_value(bad)

    def test_accepts_positive_finite(self) -> None:
        assert _parse_budget_value("12.5") == 12.5
