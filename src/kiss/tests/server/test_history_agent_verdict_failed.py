# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A run that ends with ``finish(success=False)`` must be a failed task
in the history panel, not only on its chat tab.

The chat tab strip reads ``success === false`` off the terminal event,
while the CHATS panel (task row and chat header) reads the daemon's
``failed`` flag, which used to come from the ``result`` text alone.  A
``finish(success=False)`` run keeps its HTML summary as the result
text, so the panel showed a green tick next to a task the tab strip
showed as a red cross (seen with ``/machine add ksen@host``, whose
usage error is reported through ``success=False``).

This test drives the real ``VSCodeServer._run_task`` with an agent
whose ``run`` returns the YAML result of such a finish, then asserts
the ``getHistory`` session carries ``failed: True`` and the ``failed``
tag; a ``success: true`` run through the same path stays successful.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.server import VSCodeServer

_SUMMARY = "<p>Expected <code>add &lt;name&gt; &lt;address&gt; [password]</code>.</p>"


class TestHistoryAgentVerdictFailed(unittest.TestCase):
    """``getHistory`` must flag a ``finish(success=False)`` run as failed."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-verdict-failed-")
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        th._close_db()
        th._DB_PATH = Path(self.tmpdir) / "history.db"  # type: ignore[attr-defined]
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        th._close_db()
        th._DB_PATH = self._orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, prompt: str, verdict: str) -> dict[str, Any]:
        """Run *prompt* through ``_run_task`` with a scripted result.

        Returns the ``getHistory`` session of that task.
        """
        available = get_available_models()
        if not available:
            self.skipTest("no models available in this environment")
        work_dir = Path(self.tmpdir) / re.sub(r"[^\w.-]", "_", prompt)
        work_dir.mkdir()
        tab_id = f"tab-{prompt}"
        agent = ChatSorcarAgent("Sorcar VS Code")
        state = agent_state.AgentState(
            f"state-{prompt}", agent=agent, tab_id=tab_id, server_owned=True,
        )
        agent_state.register(state)

        def scripted_run(**kwargs: object) -> str:
            task_id, chat_id = _add_task(prompt)
            agent._last_task_id = task_id
            callback = kwargs.get("_on_task_id_allocated")
            assert callable(callback)
            callback(task_id, chat_id)
            return f"success: {verdict}\nsummary: '{_SUMMARY}'\n"

        agent.run = scripted_run  # type: ignore[method-assign, assignment]
        self.server._run_task({
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "workDir": str(work_dir),
            "useWorktree": False,
            "autoCommit": False,
            "model": available[0],
        })
        self.events.clear()
        self.server._handle_command({"type": "getHistory"})
        histories = [e for e in self.events if e.get("type") == "history"]
        assert histories, "getHistory broadcast no history event"
        sessions: list[dict[str, Any]] = [
            s for s in histories[-1]["sessions"] if s["preview"] == prompt
        ]
        assert sessions, f"no history row for {prompt!r}"
        return sessions[0]

    def test_finish_success_false_is_failed_in_history(self) -> None:
        session = self._run("/machine add ksen@host", "false")
        self.assertTrue(session["failed"])
        self.assertIn("failed", session["tags"])
        row = th._get_db().execute(
            "SELECT result, success FROM task_history WHERE id = ?",
            (session["task_id"],),
        ).fetchone()
        self.assertEqual(row["result"], _SUMMARY)
        self.assertEqual(row["success"], 0)

    def test_finish_success_true_is_not_failed_in_history(self) -> None:
        session = self._run("/machine add gcp1 https://host:8787", "true")
        self.assertFalse(session["failed"])
        self.assertNotIn("failed", session["tags"])


if __name__ == "__main__":
    unittest.main()
