# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the live session monitor never shows a torn usage triple.

Round-4 fix.  ``VSCodeServer._overlay_live_metrics`` (the live-metrics
overlay of the history/session monitor) read ``total_tokens_used``,
``budget_used`` and ``total_steps`` as three separate property reads on
a running ``RelentlessAgent``.  Each such property read sums the
agent's append-only usage ledger afresh, so a concurrent attribution
(a sub-agent bank, a server-thread abandoned-subagent reclaim) landing
between two of the reads made the monitor display an impossible mix —
for example the old cost with the new tokens and steps.

The fix routes the overlay through :func:`task_runner._subtask_metrics`
(one coherent ``usage_snapshot()`` call on ledger-bearing agents, the
same per-attribute fallback as before on plain agent-shaped objects).

Real ``VSCodeServer``, a real ``WorktreeSorcarAgent`` with the real
usage ledger, and a real writer thread performing the production
single-record attribution; no mocks.  Every attribution writes the
fixed ratio 100 tokens : 0.5 USD : 1 step, so ANY coherent prefix of
the ledger satisfies ``tokens == steps * 100`` and
``cost == steps * 0.5`` exactly (0.5 sums without rounding error) —
a torn read is the only way to break the invariant.

Round-5 repair: the original test only raced overlay reads against a
free-running writer, and under the GIL the writer usually finished in
its first timeslice — 95 of 100 measured trials performed ZERO reads
while the writer was alive, so the advertised concurrent property was
never exercised.  The writer now parks on a HANDSHAKE halfway through
its appends: the reader provably overlays while the writer is alive
and mid-stream (and, parked between two known operations, the overlay
must report exactly the half-way totals), and the test asserts a
NONZERO concurrent observation count before the quiescent check.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer

_WRITES = 4000
"""Attributions the writer thread performs (bounds the ledger sums)."""


class TestMonitorUsageSnapshot:
    """``_overlay_live_metrics`` reads one coherent usage snapshot."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_overlay_never_tears_across_concurrent_attribution(self) -> None:
        server = VSCodeServer()
        agent = WorktreeSorcarAgent("monitor snapshot agent")
        agent_state.register(agent_state.AgentState(
            "7", agent=agent, tab_id="tab-snap", server_owned=True,
        ))

        half = _WRITES // 2
        midway = threading.Event()
        resume = threading.Event()

        def attribute() -> None:
            """Bank _WRITES coherent records, parking on the handshake."""
            for _ in range(half):
                agent._attribute_usage(0.5, 100, 1)
            midway.set()
            assert resume.wait(timeout=30), "reader never released the writer"
            for _ in range(_WRITES - half):
                agent._attribute_usage(0.5, 100, 1)

        def overlay_once() -> dict[str, Any]:
            """Perform one overlay read and assert the coherence invariant."""
            session: dict[str, Any] = {"tokens": 0, "cost": 0.0, "steps": 0}
            server._overlay_live_metrics(session, "7")
            steps = session["steps"]
            assert session["tokens"] == steps * 100, f"torn read: {session!r}"
            assert session["cost"] == steps * 0.5, f"torn read: {session!r}"
            return session

        writer = threading.Thread(target=attribute, daemon=True)
        writer.start()
        concurrent_reads = 0
        try:
            # Handshake: the writer is provably ALIVE and mid-stream,
            # parked between two known attributions, so this overlay
            # read is guaranteed concurrent AND its result is exactly
            # the half-way totals — the interleaving the original test
            # almost never reached (95/100 trials had zero reads while
            # the writer lived).
            assert midway.wait(timeout=30), "writer never reached the handshake"
            assert writer.is_alive()
            session = overlay_once()
            concurrent_reads += 1
            assert session["steps"] == half, session
            resume.set()
            # Free-running race for the second half: every read that
            # lands while the writer is still appending must also be
            # coherent.
            while writer.is_alive():
                overlay_once()
                concurrent_reads += 1
        finally:
            resume.set()
            writer.join(timeout=30)
        assert not writer.is_alive()
        assert concurrent_reads > 0, (
            "the test never observed the overlay concurrently with the writer"
        )

        # The final, quiescent overlay reports the full ledger.
        session = overlay_once()
        assert session["tokens"] == _WRITES * 100
        assert session["cost"] == _WRITES * 0.5
        assert session["steps"] == _WRITES


class TestRunningRowIncludesExecutorSpend:
    """The History row of a running task shows banked PLUS in-flight spend.

    ``usage_snapshot()`` on a ``RelentlessAgent`` covers only the banked
    ledger (previous sessions, folded sub-tasks); the session in flight
    lives on ``_current_executor`` until it is banked at session end.
    The overlay used to add that executor's steps but not its cost and
    tokens, so a task's row sat at the previous sessions' total (zero
    for a first session) while its chat header showed the live spend.
    """

    def setup_method(self) -> None:
        TestMonitorUsageSnapshot.setup_method(self)  # type: ignore[arg-type]

    def teardown_method(self) -> None:
        TestMonitorUsageSnapshot.teardown_method(self)  # type: ignore[arg-type]

    def test_overlay_adds_the_in_flight_executor(self) -> None:
        server = VSCodeServer()
        agent = WorktreeSorcarAgent("running row agent")
        agent._attribute_usage(0.5, 20, 1)  # banked by an earlier session
        executor = KISSAgent("session executor")
        executor._reset(
            "gpt-6-astra", True, 10, 100.0,
            {"api_key": "sk-test", "base_url": "http://127.0.0.1:9/v1"}, verbose=False,
        )
        executor._update_tokens_and_budget_from_response(SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1_000, completion_tokens=100,
                prompt_tokens_details=None, completion_tokens_details=None,
            ),
        ))
        assert executor.budget_used == (1_000 * 10.0 + 100 * 50.0) / 1e6
        agent._current_executor = executor
        agent_state.register(agent_state.AgentState(
            "8", agent=agent, tab_id="tab-running-row", server_owned=True,
        ))

        session: dict[str, Any] = {"tokens": 0, "cost": 0.0, "steps": 0}
        server._overlay_live_metrics(session, "8")
        assert session["cost"] == 0.5 + 0.015
        assert session["tokens"] == 20 + 1_100
        assert session["steps"] == 1
