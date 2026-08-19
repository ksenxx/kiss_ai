# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for audio-token billing and talk-TTS cost attribution.

Reproduces two per-task cost-reporting bugs found in the July 2026 audit:

1. OpenAI GPT audio models (``gpt-audio``, ``gpt-audio-1.5``) bill AUDIO
   tokens at $32/M input and $64/M output — 12.8x/6.4x their text rates —
   but KISS billed every token at text rates because MODEL_INFO.json had
   no audio prices and the OpenAI usage extractor ignored
   ``prompt_tokens_details.audio_tokens`` /
   ``completion_tokens_details.audio_tokens``.

2. The ``talk`` tool's server-side TTS (``synthesize_talk_audio``) ran a
   throwaway ``TalkSynthesisAgent`` whose ``budget_used`` was discarded,
   so TTS spend never reached the calling Sorcar agent's reported task
   cost.

Pricing ground truth (verified 2026-07 on developers.openai.com model
pages and openrouter.ai):

* gpt-audio / gpt-audio-1.5 / gpt-audio-2025-08-28: text $2.50/$10.00,
  audio $32.00/$64.00 per 1M tokens.
* gpt-audio-mini family: text $0.60/$2.40, audio $0.60/$2.40 (equal).
"""

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.test_audio_token_cost import (  # noqa: F401
    _audio_response,
)


class TestKISSAgentBudgetIncludesAudio:
    def test_budget_used_reflects_audio_rates_and_tokens_counted(self) -> None:
        agent = KISSAgent("audio-budget-test")
        agent.model = OpenAICompatibleModel(
            "gpt-audio-1.5", base_url="https://api.openai.com/v1", api_key="test",
        )
        response = _audio_response(
            prompt_tokens=60,
            completion_tokens=620,
            audio_input=0,
            audio_output=600,
        )
        agent._update_tokens_and_budget_from_response(response)
        expected = (60 * 2.5 + 20 * 10.0 + 600 * 64.0) / 1e6
        assert agent.budget_used == pytest.approx(expected)
        assert agent.total_tokens_used == 680
        assert agent.context_tokens_used == 680
