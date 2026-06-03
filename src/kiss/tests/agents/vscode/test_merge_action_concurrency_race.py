# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test for the concurrent ``mergeAction`` race.

``RemoteAccessServer`` serves two transports on the *same* asyncio event
loop: a local Unix-domain socket (the VS Code extension) and a remote
WebSocket (a browser viewer).  Both dispatch ``mergeAction`` commands to
:meth:`RemoteAccessServer._handle_web_merge_action`.

The handler used to fetch the per-tab :class:`_WebMergeState` under a
lock and then perform a multi-step read-modify-write on it **across** an
``await self._loop.run_in_executor(...)`` boundary (the ``reject``
branch rewrites the file in a worker thread).  When two clients act on
the *same* tab's merge review concurrently, the two coroutines interleave
at that ``await``:

    A: cur = current() -> (0, 0); await run_in_executor (suspends)
    B: cur = current() -> (0, 0); await run_in_executor (suspends)
    A: resumes, rejects hunk (0, 0), advances
    B: resumes, rejects hunk (0, 0) AGAIN, advances

so the *second* reject is a lost update — it re-resolves hunk (0, 0)
instead of the hunk the user meant, and hunk (0, 1) is left permanently
unresolved.

This test forces exactly that interleaving with two real coroutines on a
real event loop (no mocks) and asserts that both distinct hunks end up
resolved.  It fails before the per-tab ``asyncio.Lock`` fix and passes
after it.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kiss.agents.vscode.web_server import RemoteAccessServer, _WebMergeState


class _CoordExecutor(ThreadPoolExecutor):
    """Real executor that parks the first reject until the peer submits.

    This is pure test scaffolding (a barrier on the executor the
    production code calls into) — it controls *scheduling* only and
    never replaces production behaviour: the real
    ``_reject_hunk_in_file`` still runs.

    Each submitted job blocks until **two** jobs have been submitted (or
    a short timeout elapses).  A second submission only happens once the
    second coroutine has reached its ``run_in_executor`` call — which is
    *after* it has snapshotted ``current()``.  So when two ``reject``
    coroutines run concurrently (the racy, unfixed code path), the first
    cannot resolve its hunk before the second has read the same
    ``current()``, deterministically reproducing the lost-update race.

    When the production code correctly serialises the two coroutines
    (the per-tab ``asyncio.Lock`` fix), the second never submits while
    the first holds the lock, so the first job's wait simply times out
    and execution proceeds correctly.
    """

    def __init__(self) -> None:
        super().__init__(max_workers=4)
        self._count_lock = threading.Lock()
        self._submitted = 0
        self._both_submitted = threading.Event()

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        with self._count_lock:
            self._submitted += 1
            if self._submitted >= 2:
                self._both_submitted.set()

        def _wrapped() -> object:
            self._both_submitted.wait(timeout=0.5)
            return fn(*args, **kwargs)

        return super().submit(_wrapped)


def _build_merge_data(work_dir: Path) -> dict:
    """Create real files and a 3-hunk single-file merge data structure."""
    current = work_dir / "f_current.txt"
    base = work_dir / "f_base.txt"
    target = work_dir / "f.txt"
    lines = "".join(f"line{i}\n" for i in range(30))
    current.write_text(lines)
    base.write_text(lines)
    target.write_text(lines)

    def hunk(start: int) -> dict[str, int]:
        return {"bs": start, "bc": 1, "cs": start, "cc": 1}

    return {
        "files": [
            {
                "current": str(current),
                "base": str(base),
                "target": str(target),
                "hunks": [hunk(2), hunk(10), hunk(18)],
            }
        ]
    }


class TestConcurrentMergeActionRace(unittest.IsolatedAsyncioTestCase):
    """Two clients rejecting on the same tab must not drop a hunk."""

    async def test_two_clients_reject_same_tab_no_lost_hunk(self) -> None:
        """Concurrent ``reject`` actions resolve two distinct hunks."""
        work = Path(tempfile.mkdtemp())
        server = RemoteAccessServer(host="127.0.0.1", port=0, work_dir=str(work))
        loop = asyncio.get_running_loop()
        loop.set_default_executor(_CoordExecutor())
        server._loop = loop

        merge_data = _build_merge_data(work)
        tab_id = "tabX"
        with server._merge_states_lock:
            server._merge_states[tab_id] = _WebMergeState(merge_data)
        state = server._merge_states[tab_id]

        cmd = {"type": "mergeAction", "action": "reject", "tabId": tab_id}
        # Two independent coroutines on the same loop — the multi-viewer
        # (UDS + WSS) scenario.  gather lets each reach its
        # run_in_executor await before the other resumes.
        await asyncio.gather(
            server._handle_web_merge_action(dict(cmd)),
            server._handle_web_merge_action(dict(cmd)),
        )

        # Both reject actions must each have resolved a *distinct* hunk.
        # The race re-resolves hunk (0, 0) twice and leaves (0, 1)
        # permanently unresolved.
        self.assertTrue(state.is_resolved(0, 0))
        self.assertTrue(
            state.is_resolved(0, 1),
            "second reject was lost: hunk (0, 1) left unresolved (race)",
        )
        self.assertEqual(len(state._resolved), 2)


if __name__ == "__main__":
    unittest.main()
