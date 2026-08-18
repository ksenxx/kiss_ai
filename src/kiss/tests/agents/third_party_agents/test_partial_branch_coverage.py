# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Targeted integration tests to cover partial branches in _backend_utils.

Split out of the root-level ``kiss.tests.test_partial_branch_coverage``
because these tests depend on ``kiss.agents.third_party_agents`` and
therefore belong in ``tests/agents/third_party_agents``.

No mocks, test doubles, or fakes.
"""

from __future__ import annotations

import queue
from typing import Any


class TestDrainQueueMessages:
    """Cover branches in drain_queue_messages."""

    def test_drain_with_filter(self) -> None:
        """Filter keeps some messages, rejects others (line 83 keep branch)."""
        from kiss.agents.third_party_agents._backend_utils import drain_queue_messages

        q: queue.Queue[dict[str, Any]] = queue.Queue()
        q.put({"id": 1, "good": True})
        q.put({"id": 2, "good": False})
        q.put({"id": 3, "good": True})
        result = drain_queue_messages(q, limit=10, keep=lambda m: m["good"])
        assert len(result) == 2

    def test_drain_hits_limit(self) -> None:
        """Queue has more items than limit, while condition exits (line 78->85)."""
        from kiss.agents.third_party_agents._backend_utils import drain_queue_messages

        q: queue.Queue[dict[str, Any]] = queue.Queue()
        for i in range(5):
            q.put({"id": i})
        result = drain_queue_messages(q, limit=3)
        assert len(result) == 3
