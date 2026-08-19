# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: talk-tool TTS spend must reach the task's reported cost.

July 2026 audit bug #2: ``synthesize_talk_audio`` ran a throwaway
``TalkSynthesisAgent`` (gpt-audio-1.5) whose ``budget_used`` was
discarded, so every ``talk()`` call's TTS spend was missing from the
calling Sorcar agent's ``budget_used`` — and therefore from the per-task
cost persisted to ``task_history.extra`` and shown in the UI.
"""

import os

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class TestTalkToolWiringE2E:
    """The actual ``talk()`` tool must fold TTS spend into the agent.

    Adversarial-review finding: the helper tests above verify
    ``_attribute_tts_usage`` and ``synthesize_talk_audio`` separately,
    but a wiring regression in ``talk()`` itself (e.g. dropping the
    ``usage_out=`` argument or the ``_attribute_tts_usage`` call) would
    slip through.  This drives the real tool end to end.
    """

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="requires OPENAI_API_KEY for a live gpt-audio-1.5 call",
    )
    def test_talk_tool_folds_tts_spend_into_agent_budget(self) -> None:
        from kiss.tests.server._memory_printer import MemoryPrinter

        printer = MemoryPrinter()
        printer.subscribe_tab("task-tts-cost", "tab-a")
        agent = SorcarAgent("talk-wiring-cost")
        agent._use_web_tools = False
        agent.printer = printer
        from typing import Any

        talk: Any = next(
            t
            for t in agent._get_tools()
            if callable(t) and t.__name__ == "talk"
        )
        printer._thread_local.task_id = "task-tts-cost"
        assert agent.budget_used == 0.0
        msg = talk("en-US", "Quick cost check.")
        assert "en-US" in msg
        assert agent.budget_used > 0
        assert agent.total_tokens_used > 0
        assert agent.total_steps == 0
