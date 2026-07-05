# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SorcarAgent ``talk`` tool.

The ``talk(language, text)`` tool broadcasts a ``{"type": "talk"}``
event through the printer so every frontend tab subscribed to the
running task — on every device — plays the text aloud through the
device's default speaker via the Web Speech API.

These tests drive a real :class:`SorcarAgent` and a real
:class:`JsonPrinter` subclass (:class:`MemoryPrinter`, which mirrors
the production ``WebPrinter`` fanout contract) — no mocks.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


def _find_tool(tools: list, name: str) -> Any:
    """Return the tool function named *name* from *tools*."""
    for t in tools:
        if callable(t) and t.__name__ == name:
            return t
    raise AssertionError(
        f"Tool {name!r} not found in "
        f"{[getattr(t, '__name__', None) for t in tools if callable(t)]}"
    )


def _make_agent(printer: Any) -> SorcarAgent:
    """Build a SorcarAgent with web tools disabled and *printer* attached."""
    agent = SorcarAgent("test-talk-tool")
    agent._use_web_tools = False
    agent.printer = printer
    return agent


class TestTalkTool(unittest.TestCase):
    """The ``talk`` tool is a default tool and broadcasts talk events."""

    def test_talk_is_a_default_tool(self) -> None:
        """``talk`` must be present in the default SorcarAgent tool list."""
        agent = _make_agent(MemoryPrinter())
        names = [
            t.__name__ for t in agent._get_tools() if callable(t)
        ]
        self.assertIn("talk", names)

    def test_talk_signature_language_then_text(self) -> None:
        """First parameter is ``language``, second is ``text``."""
        import inspect

        agent = _make_agent(MemoryPrinter())
        talk = _find_tool(agent._get_tools(), "talk")
        params = list(inspect.signature(talk).parameters)
        self.assertEqual(params[:2], ["language", "text"])

    def test_talk_signature_optional_emotion_third(self) -> None:
        """Third parameter is ``emotion`` and defaults to empty (neutral).

        The default keeps ``talk(language, text)`` calls — and every
        existing caller — working unchanged while letting the agent
        opt into an expressive delivery.
        """
        import inspect

        agent = _make_agent(MemoryPrinter())
        talk = _find_tool(agent._get_tools(), "talk")
        sig = inspect.signature(talk)
        params = list(sig.parameters)
        self.assertEqual(params[:3], ["language", "text", "emotion"])
        self.assertEqual(sig.parameters["emotion"].default, "")

    def test_talk_broadcasts_to_every_subscribed_tab(self) -> None:
        """One ``talk`` call reaches every tab subscribed to the task.

        Mirrors production: the agent thread carries a thread-local
        ``taskId``; the printer stamps one copy of the event per
        subscriber tab (i.e. per open view of the task, across all
        devices).  Each stamped copy must carry the language and text
        so each client can synthesize speech locally.
        """
        printer = MemoryPrinter()
        printer.subscribe_tab("task-1", "tab-laptop")
        printer.subscribe_tab("task-1", "tab-phone")
        agent = _make_agent(printer)
        talk = _find_tool(agent._get_tools(), "talk")

        result: dict[str, str] = {}

        def agent_thread() -> None:
            printer._thread_local.task_id = "task-1"
            result["msg"] = talk("es", "hola usuario")

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

        talk_events = [
            ev for ev in printer.emitted if ev.get("type") == "talk"
        ]
        self.assertEqual(
            {ev.get("tabId") for ev in talk_events},
            {"tab-laptop", "tab-phone"},
        )
        for ev in talk_events:
            self.assertEqual(ev.get("language"), "es")
            self.assertEqual(ev.get("text"), "hola usuario")
            self.assertEqual(ev.get("taskId"), "task-1")
        self.assertIn("es", result["msg"])

    def test_talk_broadcasts_emotion_on_every_copy(self) -> None:
        """An explicit ``emotion`` rides along on every stamped copy.

        The client shapes speech rate and pitch from this field, so
        each subscribed tab must receive it verbatim.
        """
        printer = MemoryPrinter()
        printer.subscribe_tab("task-emo", "tab-a")
        printer.subscribe_tab("task-emo", "tab-b")
        agent = _make_agent(printer)
        talk = _find_tool(agent._get_tools(), "talk")

        def agent_thread() -> None:
            printer._thread_local.task_id = "task-emo"
            talk("en-US", "We did it, nice work!", "cheerful")

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

        talk_events = [
            ev for ev in printer.emitted if ev.get("type") == "talk"
        ]
        self.assertEqual(len(talk_events), 2)
        for ev in talk_events:
            self.assertEqual(ev.get("emotion"), "cheerful")

    def test_talk_default_emotion_is_empty_neutral(self) -> None:
        """Without an ``emotion`` argument the event carries ``""``.

        An empty emotion tells the client to infer the vibe from the
        text itself (punctuation and wording) rather than a fixed one.
        """
        printer = MemoryPrinter()
        printer.subscribe_tab("task-neutral", "tab-a")
        agent = _make_agent(printer)
        talk = _find_tool(agent._get_tools(), "talk")

        def agent_thread() -> None:
            printer._thread_local.task_id = "task-neutral"
            talk("en-US", "Status update ready.")

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

        talk_events = [
            ev for ev in printer.emitted if ev.get("type") == "talk"
        ]
        self.assertEqual(len(talk_events), 1)
        self.assertEqual(talk_events[0].get("emotion"), "")

    def test_talk_copies_share_one_talk_id_unique_per_call(self) -> None:
        """All stamped copies of one ``talk()`` call share one ``talkId``.

        Every connected client receives every stamped copy (one per
        subscribed viewer tab) and dedupes playback by ``talkId`` so
        each device speaks the utterance exactly once — without the id
        a task viewed in two tabs was spoken twice on the same
        speakers.  A SECOND ``talk()`` call must carry a DIFFERENT
        ``talkId`` so repeating the same sentence intentionally still
        plays again.
        """
        printer = MemoryPrinter()
        printer.subscribe_tab("task-id", "tab-a")
        printer.subscribe_tab("task-id", "tab-b")
        agent = _make_agent(printer)
        talk = _find_tool(agent._get_tools(), "talk")

        def agent_thread() -> None:
            printer._thread_local.task_id = "task-id"
            talk("en-US", "same sentence")
            talk("en-US", "same sentence")

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

        talk_events = [
            ev for ev in printer.emitted if ev.get("type") == "talk"
        ]
        self.assertEqual(len(talk_events), 4)  # 2 calls x 2 tabs
        ids = [ev.get("talkId") for ev in talk_events]
        for talk_id in ids:
            self.assertIsInstance(talk_id, str)
            self.assertTrue(talk_id)
        # 2 distinct ids overall, and the copies of each call agree.
        self.assertEqual(len(set(ids)), 2)
        by_id: dict[str, list[dict[str, Any]]] = {}
        for ev in talk_events:
            by_id.setdefault(str(ev.get("talkId")), []).append(ev)
        for copies in by_id.values():
            self.assertEqual(
                {c.get("tabId") for c in copies}, {"tab-a", "tab-b"}
            )

    def test_talk_is_live_only_not_replayed_from_history(self) -> None:
        """Talk audio is live-only: excluded from the display recording.

        Recorded display events are replayed when a task is reopened
        from history; replaying stale audio then would be wrong.  Like
        ``askUser``, ``talk`` must fan out live to subscribed tabs but
        must NOT appear in the persisted display-event recording.
        """
        printer = MemoryPrinter()
        printer.subscribe_tab("task-rec", "tab-a")
        agent = _make_agent(printer)
        talk = _find_tool(agent._get_tools(), "talk")

        recorded: dict[str, list] = {}

        def agent_thread() -> None:
            printer._thread_local.task_id = "task-rec"
            printer.start_recording()
            talk("fr", "bonjour")
            recorded["events"] = printer.stop_recording()

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

        live = [ev for ev in printer.emitted if ev.get("type") == "talk"]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].get("language"), "fr")
        self.assertEqual(live[0].get("text"), "bonjour")
        self.assertEqual(
            [ev for ev in recorded["events"] if ev.get("type") == "talk"],
            [],
        )

    def test_talk_without_printer_reports_unavailable(self) -> None:
        """No printer (e.g. bare library use) → graceful message."""
        agent = _make_agent(None)
        talk = _find_tool(agent._get_tools(), "talk")
        msg = talk("en", "hello")
        self.assertIn("not available", msg)

    def test_talk_with_printer_that_cannot_broadcast_reports_unavailable(
        self,
    ) -> None:
        """A printer-like object without ``broadcast`` is a graceful no-op."""
        agent = _make_agent(object())
        talk = _find_tool(agent._get_tools(), "talk")
        msg = talk("en", "hello")
        self.assertIn("not available", msg)


if __name__ == "__main__":
    unittest.main()
