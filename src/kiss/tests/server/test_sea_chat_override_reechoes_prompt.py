# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A SEA ``chat_id`` override re-echoes the run's prompt before its ``clear``.

Every ``run`` is announced to the clients as ``setTaskText`` (the
prompt echo) followed by ``clear``; the chat webview opens the new
transcript with the text of the echo that precedes the ``clear``.  A
SEA whose settings pin ``chat_id`` makes ``_run_task`` broadcast a
SECOND ``clear`` for the same run (the overridden chat id), from the
worker thread, after a setup window in which the user may have
submitted a follow-up — which ``_cmd_run`` echoes as ``setTaskText``
too (it is queued as steering, no new run starts).  Without a fresh
echo of the run's own prompt right before the override ``clear``, the
webview would open the re-cleared transcript with the follow-up's
text.

No mocks: a real ``VSCodeServer``, a real agent-script file, a real
worker thread; only the LLM-driven ``run`` of the agent is stubbed.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class TestSeaChatOverrideReechoesPrompt(unittest.TestCase):
    """The override ``clear`` is preceded by the run's own prompt echo."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-sea-chat-echo-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def recording_broadcast(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = recording_broadcast  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(self_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_override_clear_follows_a_fresh_prompt_echo(self) -> None:
        """Broadcast order: echo, clear, echo (same prompt), clear (SEA chat)."""
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        script = Path(self.tmpdir) / "agent_script.py"
        script.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            "    def settings(self, settings):\n"
            "        return settings | {'chat_id': 'sea-pinned-chat'}\n",
            encoding="utf-8",
        )
        tab_id = "sea-echo-tab"
        self.server._cmd_run({
            "type": "run",
            "prompt": "the run's own prompt",
            "tabId": tab_id,
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
            "agentPath": str(script),
        })
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.task_thread is not None
        worker = state.task_thread
        worker.join(timeout=60)
        assert not worker.is_alive(), "worker did not finish"

        with self._events_lock:
            announced = [
                (e["type"], e.get("text", e.get("chat_id")))
                for e in self.events
                if e.get("tabId") == tab_id
                and e.get("type") in ("setTaskText", "clear")
            ]
        clears = [a for a in announced if a[0] == "clear"]
        assert len(clears) == 2, announced
        assert clears[1] == ("clear", "sea-pinned-chat"), announced
        assert announced[0] == ("setTaskText", "the run's own prompt"), announced
        assert announced[-2:] == [
            ("setTaskText", "the run's own prompt"),
            ("clear", "sea-pinned-chat"),
        ], (
            "the override clear must follow a fresh echo of the run's own "
            f"prompt: {announced}"
        )
