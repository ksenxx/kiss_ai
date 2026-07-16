# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: ``_replay_session`` clobbers ``use_worktree`` mid-flight.

Loading a chat from history into a tab is a VIEW operation, yet
``_replay_session`` unconditionally overwrote the tab's
``use_worktree`` flag with the loaded chat's persisted
``is_worktree`` value — even while a worktree task was still RUNNING
on that tab, or while a finished worktree run was awaiting the
user's merge/discard decision (``agent._wt_pending``).

Consequence: the end-of-task cleanup in ``_TaskRunnerMixin._run_task``
keeps the agent alive only when ``tab.use_worktree and
tab.agent._wt_pending`` — after viewing a non-worktree chat in the
same tab the flag is ``False``, so the agent holding the pending
worktree state is disposed and the user's subsequent merge/discard
fails (and the worktree branch leaks).  The merge-busy guard
(``t.is_merging and t.use_worktree``) is broken the same way.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from typing import Any

import kiss.server.server as _server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


def _fake_loader(extra: dict[str, Any]) -> Any:
    """Build a stub for ``_load_latest_chat_events_by_chat_id``."""

    def loader(chat_id: str) -> dict[str, Any]:
        return {
            "events": [],
            "task": "old task",
            "task_id": None,
            "chat_id": chat_id,
            "extra": json.dumps(extra),
        }

    return loader


class TestReplayPreservesWorktreeFlag(unittest.TestCase):
    """Resuming another chat into a busy tab must not flip use_worktree."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt4-replaywt-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]
        self._orig_loader = _server_module._load_latest_chat_events_by_chat_id

    def tearDown(self) -> None:
        _server_module._load_latest_chat_events_by_chat_id = self._orig_loader
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_into_running_worktree_tab_keeps_flag(self) -> None:
        tab_id = "wt-busy-tab"
        tab = self.server._get_tab(tab_id)
        tab.chat_id = "chat-running"
        tab.use_worktree = True
        tab.is_task_active = True

        _server_module._load_latest_chat_events_by_chat_id = _fake_loader({})

        self.server._handle_command({
            "type": "resumeSession",
            "chatId": "chat-other",
            "tabId": tab_id,
        })

        assert tab.use_worktree is True, (
            "BUG: _replay_session overwrote use_worktree=False while a "
            "worktree task was still running on the tab — the end-of-task "
            "cleanup will now dispose the agent despite _wt_pending, "
            "destroying the pending worktree merge/discard state"
        )


if __name__ == "__main__":
    unittest.main()
