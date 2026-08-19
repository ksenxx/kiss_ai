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

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _attribute_tts_usage


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
        assert agent.total_steps == 7

    def test_empty_usage_is_a_noop(self) -> None:
        agent = SorcarAgent("tts-attribution-noop")
        agent.budget_used = 0.25
        agent.total_tokens_used = 10
        _attribute_tts_usage(agent, {})
        assert agent.budget_used == 0.25
        assert agent.total_tokens_used == 10
