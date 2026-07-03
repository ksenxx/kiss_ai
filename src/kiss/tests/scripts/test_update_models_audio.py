# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for audio-capable chat model handling in update_models.py.

Audio-capable chat models (``gpt-audio``, ``gpt-audio-mini``, dated
snapshots, OpenRouter passthroughs) reject text-only requests on
``/v1/chat/completions``, so capability probes must attach a small audio
input for them and nothing for regular text models.
"""

import io
import wave

from kiss.scripts.update_models import _probe_attachments, _tiny_wav_bytes


def test_tiny_wav_bytes_is_valid_wav():
    """The generated probe audio must be a decodable 16-bit mono WAV."""
    with wave.open(io.BytesIO(_tiny_wav_bytes())) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000
        assert wf.getnframes() > 0


def test_probe_attachments_for_audio_models():
    """Audio chat models get exactly one WAV attachment for probing."""
    for name in (
        "gpt-audio",
        "gpt-audio-mini",
        "gpt-audio-2025-08-28",
        "gpt-audio-mini-2025-12-15",
        "gpt-4o-audio-preview",
        "openrouter/openai/gpt-audio",
    ):
        attachments = _probe_attachments(name)
        assert attachments is not None, name
        assert len(attachments) == 1, name
        assert attachments[0].mime_type == "audio/wav", name
        assert attachments[0].data == _tiny_wav_bytes(), name


def test_probe_attachments_for_text_models_is_none():
    """Regular text models are probed without attachments."""
    for name in (
        "gpt-5.4",
        "gpt-4o",
        "claude-sonnet-4-5",
        "gemini-2.5-flash",
        "openrouter/openai/gpt-5.4",
        "text-embedding-3-small",
    ):
        assert _probe_attachments(name) is None, name
