# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group C): per-tab merge-action locks leak forever.

BUG-WS8-1 — every cleanup site that pops a finished review's
``_merge_states`` entry also pops its ``_merge_action_locks`` entry
(``_fire_pending_tab_close``, the client ``all-done`` branch and the
web ``closeTab`` branch of ``_dispatch_client_command``) — EXCEPT the
most common one: the completion branch at the end of
``_apply_web_merge_action`` (``remaining == 0`` after the user resolves
the last hunk).  A review finished through the merge toolbar therefore
leaks one ``asyncio.Lock`` per tab id in ``_merge_action_locks`` for
the daemon's entire lifetime (tab ids are fresh UUIDs, so entries are
never reused).

BUG-WS8-2 — ``_handle_web_merge_action`` lazily mints a lock for ANY
``tabId`` before checking that a merge review exists for it, and the
``state is None`` early-return in ``_apply_web_merge_action`` leaves
that lock behind.  An authenticated-but-buggy (or malicious) client
spamming ``mergeAction`` commands with random tab ids grows
``_merge_action_locks`` without bound.  ``_replay_merge_review``
already guards against exactly this by checking ``_merge_states``
membership BEFORE creating the lock — the action path is inconsistent
with it.

The tests drive the real ``RemoteAccessServer`` merge-action pipeline
(real event loop, real files on disk, real ``_WebMergeState``) — no
mocks or patches.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.web_server import RemoteAccessServer


class TestMergeActionLockLeak(unittest.TestCase):
    """Finished / bogus merge reviews must not leak per-tab locks."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-bh8c-mergelock-"))
        self.server = RemoteAccessServer(
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.tmpdir / "sorcar.sock",
        )

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _register_single_hunk_review(self, tab_id: str) -> None:
        """Register a real one-hunk merge review for *tab_id*."""
        base = self.tmpdir / "base.txt"
        cur = self.tmpdir / "cur.txt"
        base.write_text("old line\n")
        cur.write_text("new line\n")
        merge_data: dict[str, Any] = {
            "work_dir": str(self.tmpdir),
            "files": [
                {
                    "name": "cur.txt",
                    "base": str(base),
                    "current": str(cur),
                    "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
                },
            ],
        }
        # The real registration path used by WebPrinter.broadcast for
        # ``merge_data`` events.
        self.server._register_merge_state(tab_id, merge_data)

    def test_completed_review_releases_action_lock(self) -> None:
        """Accepting the last hunk must drop BOTH the merge state and
        the per-tab action lock (BUG-WS8-1)."""
        tab_id = "bh8c-merge-tab"
        self._register_single_hunk_review(tab_id)

        async def drive() -> None:
            self.server._loop = asyncio.get_running_loop()
            self.server._printer._loop = self.server._loop
            await self.server._handle_web_merge_action({
                "type": "mergeAction",
                "action": "accept",
                "tabId": tab_id,
            })

        asyncio.run(drive())

        self.assertNotIn(
            tab_id,
            self.server._merge_states,
            "merge state must be popped once the last hunk is resolved",
        )
        self.assertNotIn(
            tab_id,
            self.server._merge_action_locks,
            "BUG-WS8-1: the completion branch of "
            "_apply_web_merge_action popped _merge_states but leaked "
            "the per-tab entry in _merge_action_locks — one lock per "
            "finished review accumulates for the daemon's lifetime",
        )

    def test_unknown_tab_action_does_not_mint_lock(self) -> None:
        """A mergeAction for a tab with no review must not leave a
        permanent lock entry behind (BUG-WS8-2)."""
        ghost_tabs = [f"bh8c-ghost-{i}" for i in range(5)]

        async def drive() -> None:
            self.server._loop = asyncio.get_running_loop()
            self.server._printer._loop = self.server._loop
            for ghost in ghost_tabs:
                await self.server._handle_web_merge_action({
                    "type": "mergeAction",
                    "action": "next",
                    "tabId": ghost,
                })

        asyncio.run(drive())

        leaked = [
            ghost
            for ghost in ghost_tabs
            if ghost in self.server._merge_action_locks
        ]
        self.assertEqual(
            leaked,
            [],
            "BUG-WS8-2: mergeAction commands for tabs with no merge "
            f"review leaked per-tab locks: {leaked} — an authenticated "
            "client spamming random tab ids grows _merge_action_locks "
            "without bound",
        )


if __name__ == "__main__":
    unittest.main()
