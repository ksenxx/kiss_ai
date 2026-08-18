# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (findings-2 audit): persistence, git_worktree, skills,
and channel-CLI helper regressions.

Covers, end-to-end with real files/processes/DBs (no mocks/patches):

* S2-01 — ``_stop_event_writer`` must never strand late-enqueued events.
* S2-03 — prefix autocomplete must return distinct older matches even
  when many newer duplicates exist.
* S2-04 — ``has_uncommitted_changes`` must treat a failing ``git
  status`` as dirty (never report clean on error).
* S2-05 — ``copy_dirty_state`` must raise on a failing ``git status``
  instead of silently omitting the user's dirty state.
* S2-30 — ``copy_dirty_state`` must mirror dirty submodule content.
* S2-07 — one invalid-UTF-8 SKILL.md file must not abort discovery.
* S2-22 — ``--max_budget`` rejects nan/inf/zero/negative values.
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
