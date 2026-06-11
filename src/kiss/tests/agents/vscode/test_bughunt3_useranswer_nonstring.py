# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: non-string userAnswer leaks through Queue[str] (BUG-F).

``_cmd_user_answer`` put ``cmd.get("answer", "")`` on the tab's
``queue.Queue[str]`` verbatim, so a malformed client payload with
``"answer": null`` (or a number) made ``_await_user_response`` — and
therefore the agent's ``ask_user_question`` callback — return ``None``
where ``str`` is promised, crashing any agent code that treats the
answer as a string.
"""

from __future__ import annotations

import queue
import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


class TestUserAnswerNonString(unittest.TestCase):
    """userAnswer must always deliver a str to the waiting agent."""

    def setUp(self) -> None:
        self.server = VSCodeServer()

        def capture(event: dict[str, Any]) -> None:
            pass

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def _deliver(self, answer: Any) -> str:
        tab = self.server._get_tab("ua-tab")
        tab.user_answer_queue = queue.Queue(maxsize=1)
        self.server._cmd_user_answer({
            "type": "userAnswer", "tabId": "ua-tab", "answer": answer,
        })
        return tab.user_answer_queue.get_nowait()

    def test_none_answer_is_delivered_as_empty_string(self) -> None:
        item = self._deliver(None)
        assert isinstance(item, str), (
            f"BUG: ask_user_question would return {item!r} (not str)"
        )
        assert item == ""

    def test_numeric_answer_is_delivered_as_string(self) -> None:
        item = self._deliver(42)
        assert isinstance(item, str), (
            f"BUG: ask_user_question would return {item!r} (not str)"
        )
        assert item == "42"


if __name__ == "__main__":
    unittest.main()
