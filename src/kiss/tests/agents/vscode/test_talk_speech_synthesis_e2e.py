# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for GPT-audio speech synthesis of the ``talk`` tool.

Calls the real OpenAI default GPT audio model (``gpt-audio-1.5``, no
mocks) and checks
that :func:`kiss.agents.vscode.speech_synthesis.synthesize_talk_audio`
returns base64 MP3 audio whose decoded bytes carry a valid MP3 stream
header, plus the degradation contract (empty text / bad model → None).
"""

import base64
import os

import pytest

from kiss.agents.vscode.speech_synthesis import synthesize_talk_audio

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def _is_mp3(data: bytes) -> bool:
    """True when *data* starts with an ID3 tag or an MPEG frame sync."""
    return data.startswith(b"ID3") or (
        len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    )


@requires_openai
def test_synthesize_returns_playable_mp3() -> None:
    """A short utterance synthesizes to non-trivial base64 MP3 audio."""
    result = synthesize_talk_audio(
        "Hi there! I'm KISS Sorcar, and this is my new natural voice.",
        language="en-US",
        emotion="cheerful",
    )
    assert result is not None
    audio_b64, mime = result
    assert mime == "audio/mpeg"
    data = base64.b64decode(audio_b64)
    assert len(data) > 1000
    assert _is_mp3(data)


@requires_openai
def test_default_narrator_voice_synthesizes_multi_sentence_take() -> None:
    """The default voice reads a multi-sentence script as one MP3 clip.

    Guards the natural-narrator delivery path end to end: the default
    voice/prompt pair must be accepted by the real GPT audio model and
    yield a single substantial MP3 stream for a multi-sentence script
    (the case where breaks/stutter artifacts used to appear).
    """
    result = synthesize_talk_audio(
        "I finished refactoring the parser module. All twelve tests "
        "pass. Let me know if you would like a deeper cleanup next.",
        language="en-US",
    )
    assert result is not None
    audio_b64, mime = result
    assert mime == "audio/mpeg"
    data = base64.b64decode(audio_b64)
    assert len(data) > 10000
    assert _is_mp3(data)


def test_empty_text_returns_none() -> None:
    """Blank text must not hit the API and must return None."""
    assert synthesize_talk_audio("") is None
    assert synthesize_talk_audio("   \n") is None


def test_api_failure_degrades_to_none() -> None:
    """A nonexistent model fails the API call and returns None."""
    assert (
        synthesize_talk_audio("hello", model="no-such-model-xyz") is None
    )
