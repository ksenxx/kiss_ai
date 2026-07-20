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

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _attribute_tts_usage
from kiss.core.speech_synthesis import synthesize_talk_audio


class TestAttributeTtsUsage:
    def test_spend_lands_on_agent_budget_and_tokens(self) -> None:
        agent = SorcarAgent("tts-attribution-test")
        agent.budget_used = 0.5
        agent.total_tokens_used = 1_000
        agent.total_steps = 7
        _attribute_tts_usage(
            agent,
            {"budget_used": 0.0387, "total_tokens_used": 680, "total_steps": 1},
        )
        assert agent.budget_used == pytest.approx(0.5387)
        assert agent.total_tokens_used == 1_680
        # TTS is a single non-agentic call, not an agent step.
        assert agent.total_steps == 7

    def test_empty_usage_is_a_noop(self) -> None:
        agent = SorcarAgent("tts-attribution-noop")
        agent.budget_used = 0.25
        agent.total_tokens_used = 10
        _attribute_tts_usage(agent, {})
        assert agent.budget_used == 0.25
        assert agent.total_tokens_used == 10


class TestSynthesizeTalkAudioUsage:
    def test_empty_text_reports_no_usage(self) -> None:
        usage: dict = {}
        assert synthesize_talk_audio("   ", usage_out=usage) is None
        assert usage == {}

    def test_failed_synthesis_still_reports_usage_keys(self) -> None:
        # An unknown model makes ``agent.run`` raise BEFORE any API
        # call: the ``finally`` must still fill ``usage_out`` (with the
        # zero spend actually incurred) so the caller's attribution
        # arithmetic never KeyErrors on a failed synthesis.
        usage: dict = {}
        assert (
            synthesize_talk_audio(
                "hello", model="no-such-model-xyz", usage_out=usage,
            )
            is None
        )
        assert usage["budget_used"] == 0.0
        assert usage["total_tokens_used"] == 0

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="requires OPENAI_API_KEY for a live gpt-audio-1.5 call",
    )
    def test_live_synthesis_reports_positive_audio_billed_usage(self) -> None:
        usage: dict = {}
        result = synthesize_talk_audio("Hello there.", usage_out=usage)
        assert result is not None
        audio_b64, mime = result
        assert mime == "audio/mpeg"
        assert len(audio_b64) > 100
        # The spend must be visible to the caller...
        assert usage["budget_used"] > 0
        assert usage["total_tokens_used"] > 0
        # ...and even a two-word utterance produces >100 audio output
        # tokens; at the correct $64/M audio-output rate that is far
        # above the old text-rate floor for the same token count.
        assert usage["budget_used"] > 100 * 10.0 / 1e6


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
        from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

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
        # The synthesis spend landed on THIS agent's task accounting...
        assert agent.budget_used > 0
        assert agent.total_tokens_used > 0
        # ...without inventing an agent step for the one-shot TTS call.
        assert agent.total_steps == 0
