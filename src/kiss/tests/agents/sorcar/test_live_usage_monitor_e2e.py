# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the parent task's cost/tokens header must reflect the
agent AND all of its parallel sub-agents at every turn.

Reproduces the issue where, while ``run_parallel`` blocks the parent's
turn, nothing emits ``usage_info`` on the PARENT task — so the header
(the chat webview's top bar in the VS Code extension and the remote
web client) shows a stale figure that excludes all live sub-agent
spend until every sub-agent finishes.

The fix is ``_LiveUsageMonitor`` (``sorcar_agent.py``): while parallel
sub-agents run it polls their live spend and broadcasts parent-task
``usage_info`` events whose printer-applied offsets yield the aggregate
(parent cumulative + parent live session + all sub-agents).

No mocks, patches, fakes, or test doubles: real agents, a real
``JsonPrinter`` subclass that records its own broadcasts, and (for the
slow test) real LLM calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _live_agent_usage,
    _LiveUsageMonitor,
)
from kiss.core.kiss_agent import KISSAgent


class TestLiveUsageMonitor:
    """Deterministic e2e tests of the live-usage monitor itself."""

    def test_sub_agent_live_session_spend_included(self) -> None:
        """A sub-agent's own in-flight executor session (not yet folded
        into its totals by relentless) must be counted."""
        sub: Any = KISSAgent("sub")
        sub.budget_used = 0.10
        sub.total_tokens_used = 100
        sub.total_steps = 1
        sub_executor = KISSAgent("sub-session")
        sub_executor.budget_used = 0.05
        sub_executor.total_tokens_used = 50
        sub_executor.step_count = 2
        sub._current_executor = sub_executor
        assert _live_agent_usage(sub) == (
            pytest.approx(0.15),
            150,
            3,
        )

    def test_live_agent_usage_without_executor(self) -> None:
        """Agents without a live executor report just their totals."""
        sub: Any = KISSAgent("sub")
        sub.budget_used = 0.10
        sub.total_tokens_used = 100
        assert _live_agent_usage(sub) == (pytest.approx(0.10), 100, 0)

    def test_no_printer_is_a_noop(self) -> None:
        """Without a printer the monitor never starts a thread."""
        parent = SorcarAgent("no-printer-parent")
        monitor = _LiveUsageMonitor(parent, None, interval=0.01)
        monitor.start()
        assert monitor._thread is None
        monitor.stop()
