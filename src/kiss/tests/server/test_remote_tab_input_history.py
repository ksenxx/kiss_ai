# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: texts typed into a tab bound to a machine reach this daemon's composer history.

The composer's ArrowUp/ArrowDown history (``cycleHistoryUp`` in
``webapp/media/main.js``), its ghost completions and the order of its
``/`` popup all come from the daemon's ``inputHistory``: the prompts of
its own ``task_history`` plus the ``steer_inputs`` typed into running
tasks.  A tab bound to a machine forwards its ``submit`` and
``appendUserMessage`` there (:class:`kiss.server.remote_tabs.RemoteTabs`),
where the machine keeps the task row and the steer; nothing here saw
the text, so a ``/ask …`` typed during a remote run, or the prompt that
started it, dropped out of the history as soon as the next
``inputHistory`` refresh replaced the page's local copy of it.

A real local daemon, a real remote daemon ("box") with a fake model
and one client, as in :mod:`kiss.tests.server.test_remote_tabs`.

Not covered: the ``except`` branch of the recording in
``RemoteTabs.route`` (a warning when the local database refuses the
write).  Like the same branch in ``_echo_injected_prompt`` it needs a
failing SQLite write under a live daemon, which no end-to-end setup
produces without a test double.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.sorcar import machines as mach
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness, _FinishingModel

ASK_LINE = "/ask how many trials are left?"


class TestRemoteTabInputHistory(RemoteTabsHarness):
    """What the user typed into a bound tab is remembered here, its task stays the machine's."""

    async def _input_history(self) -> list[str]:
        reply = await self._ask({"type": "getInputHistory"}, "inputHistory")
        return [str(t) for t in reply["tasks"]]

    async def _steer_echo(self, tab: str) -> dict[str, Any]:
        """The machine's echo of a message typed into *tab*'s running task."""
        for _ in range(400):
            msg = await self._next()
            if msg.get("type") == "prompt" and msg.get("tabId") == tab and msg.get("steer"):
                return msg
        raise AssertionError("the machine never echoed the steer")

    async def test_prompt_and_ask_line_of_a_bound_tab_are_in_the_history(self) -> None:
        tab = "tab-history"
        await self._fan_out(tab)  # "Fan out" runs on box; its sub-agent is held
        await self._send({"type": "appendUserMessage", "tabId": tab, "prompt": ASK_LINE})
        # The echo proves the machine handled the line as a live message
        # of its running task; it is relayed after the forward, which the
        # local recording precedes, so the history asked for next is final.
        echo = await self._steer_echo(tab)
        self.assertEqual(echo["text"], ASK_LINE)

        history = await self._input_history()
        self.assertIn(ASK_LINE, history, "the /ask line typed during the remote run")
        self.assertIn("Fan out", history, "the prompt that started the remote run")
        # The task itself is still the machine's alone: no local history row.
        self.assertNotIn("Fan out", await self._titles(mach.LOCAL_MACHINE))
        self.assertIn("Fan out", await self._titles("box"))

        _FinishingModel.released.set()
        await self._run_until_idle(tab)
        # A finished run changes nothing: the texts are kept, not the task.
        history = await self._input_history()
        self.assertIn(ASK_LINE, history)
        self.assertIn("Fan out", history)
        self.assertNotIn("Fan out", await self._titles(mach.LOCAL_MACHINE))

    async def test_an_answer_to_the_machine_is_not_history(self) -> None:
        """A ``userAnswer`` forwarded to the machine is an answer, not prompt history."""
        tab = "tab-answer"
        await self._fan_out(tab)
        await self._send({"type": "userAnswer", "tabId": tab, "answer": "yes, go on"})
        _FinishingModel.released.set()
        await self._run_until_idle(tab)
        self.assertNotIn("yes, go on", await self._input_history())
