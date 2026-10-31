# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``synthesize_talk_audio`` keeps its documented ``voice`` parameter.

A cleanup dropped the parameter because the repository's own callers use
the default voice.  That broke the public helper for every other caller:
``voice=`` raised ``TypeError``, and a positional voice was bound to
``usage_out``.  The voice must also reach the audio model: the live API
reads a script in a built-in voice and rejects a made-up one.
"""

import os

import pytest

from kiss.core.speech_synthesis import synthesize_talk_audio

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def test_voice_comes_between_model_and_usage_out() -> None:
    assert synthesize_talk_audio("   ", voice="alloy") is None
    usage: dict = {}
    assert synthesize_talk_audio("hello", "", "", "no-such-model-xyz", "alloy", usage) is None
    assert usage == {"budget_used": 0.0, "total_tokens_used": 0}


@requires_openai
def test_the_voice_reaches_the_audio_model() -> None:
    assert synthesize_talk_audio("Hi.", voice="alloy") is not None
    assert synthesize_talk_audio("Hi.", voice="no-such-voice") is None
