# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: talk-tool TTS spend must reach the task's reported cost.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_talk_tts_cost_attribution``; the non-core tests remain there.
"""

import os

import pytest

from kiss.core.speech_synthesis import synthesize_talk_audio


class TestSynthesizeTalkAudioUsage:
    def test_empty_text_reports_no_usage(self) -> None:
        usage: dict = {}
        assert synthesize_talk_audio("   ", usage_out=usage) is None
        assert usage == {}

    def test_failed_synthesis_still_reports_usage_keys(self) -> None:
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
        assert usage["budget_used"] > 0
        assert usage["total_tokens_used"] > 0
        assert usage["budget_used"] > 100 * 10.0 / 1e6
