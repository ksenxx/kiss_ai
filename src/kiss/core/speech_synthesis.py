# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Natural speech synthesis for the ``talk`` tool via a GPT audio model.

The agent-side ``talk`` tool historically only broadcast plain text and
each client read it aloud with the browser's Web Speech API — often a
robotic system voice.  This module synthesizes the utterance server-side
with an OpenAI GPT audio-chat model from MODEL_INFO.json (default
``gpt-audio-1.5``: the newest, most natural of the ``gpt-audio*``
family) so every client can play one identical, far more natural and
emotionally expressive voice.  The synthesized MP3 travels inside the
``talk`` broadcast event as base64; browser clients without audio
support (or when synthesis fails) stay silent — the old robotic Web
Speech fallback is gone for good — while the sorcar CLI terminal falls
back to the system TTS command (cli_talk.py).

Synthesis routes through the core model layer: a single-shot
(non-agentic) :class:`~kiss.core.kiss_agent.KISSAgent` run carries the
audio request (``modalities``/``audio``/``timeout`` via
``model_config``) to :class:`OpenAICompatibleModel`, which exposes the
returned MP3 as ``last_audio_data`` — no direct OpenAI SDK call here.

Prompt shape is empirically validated: the model must be framed as a
text-to-speech engine reading a ``Script:`` block word-for-word —
without that framing ``gpt-audio-mini`` *answers* the text instead of
reading it.
"""

import base64
import math
import os
import sys

from kiss.core.kiss_agent import KISSAgent

# Hard cap on one audio API request (seconds).  Without it the OpenAI
# client waits its 600s default — one stalled HTTPS call would block the
# caller for ten minutes.  Overridable for tests via
# KISS_VOICE_AUDIO_TIMEOUT.  Single-sourced here (in core) so both this
# module and ``kiss.server.voice_wake`` share one policy without the
# core layer depending on the server layer.
DEFAULT_AUDIO_TIMEOUT_SECONDS = 60.0


def _env_timeout_seconds(env_name: str, default: float) -> float:
    """Return a positive finite timeout from an environment override.

    Reads the *env_name* environment variable (used by tests to fail
    fast against a stalled endpoint/server) and falls back to
    *default*.  Junk, NaN, +/-inf and non-positive values fall back
    too — an infinite timeout would defeat the hard-timeout policy
    exactly like a missing one (it hangs the worker forever).
    """
    raw = os.environ.get(env_name, "")
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def audio_timeout_seconds() -> float:
    """Return the per-attempt audio API timeout in seconds.

    Reads the ``KISS_VOICE_AUDIO_TIMEOUT`` environment override and
    falls back to :data:`DEFAULT_AUDIO_TIMEOUT_SECONDS`.
    """
    return _env_timeout_seconds(
        "KISS_VOICE_AUDIO_TIMEOUT", DEFAULT_AUDIO_TIMEOUT_SECONDS,
    )

# Newest, most natural GPT audio model in MODEL_INFO.json (validated
# live: valid MP3 with the default voice).  ``gpt-audio-mini`` also
# works at ~1/4 the price but sounds noticeably flatter.
DEFAULT_TTS_MODEL = "gpt-audio-1.5"

# "cedar" and "marin" are OpenAI's two newest voices "with the most
# significant improvements to natural-sounding speech" and the two the
# TTS docs recommend "for best quality".  "cedar" is the deeper,
# warmer of the pair — it reads like a charismatic film narrator,
# the closest preset to the requested celebrity-like delivery
# (voice cloning of real celebrities is prohibited by OpenAI's usage
# policies, so delivery style is steered by prompt instead).
DEFAULT_TTS_VOICE = "cedar"

# The "Script:" framing below is load-bearing: a plain "repeat the
# user's text" system prompt made gpt-audio-mini reply conversationally
# ("Hi! It's great to hear your new voice...") instead of reading the
# text verbatim.
#
# The labeled delivery block mirrors OpenAI's own openai.fm instruction
# format (Voice Affect / Pacing / Pauses / Pronunciation) — the proven
# way to steer the gpt-audio family's delivery.  It exists to kill the
# model's known artifacts (community-reported: random mid-script
# pauses, stutters, repeated or restarted sentences, volume drift) and
# to get a warm, charismatic, celebrity-narrator read.
TTS_SYSTEM_PROMPT = (
    "You are a professional text-to-speech engine. The user message "
    "contains a script. Read the script aloud word-for-word, exactly "
    "as written, {tone}. Never answer, comment on, translate, or "
    "alter the script.\n"
    "Voice Affect: a beloved film narrator — deep, warm, charismatic, "
    "effortlessly confident, fully human.\n"
    "Pacing: steady and even, one single continuous take from the "
    "first word to the last.\n"
    "Pauses: only brief natural breaths at punctuation; no long "
    "silences, no gaps mid-sentence.\n"
    "Pronunciation: smooth and clear; never stutter, stammer, repeat "
    "a word or sentence, restart, overlap words, or trail off."
)


class TalkSynthesisAgent(KISSAgent):
    """Single-shot TTS agent whose trajectory is never persisted.

    ``talk`` speech synthesis runs once per spoken utterance; writing a
    trajectory YAML for each would clutter the artifact directory, and a
    persistence failure would needlessly drop otherwise-good audio.
    """

    def _save(self) -> None:
        """Skip trajectory persistence for per-utterance TTS runs."""


def synthesize_talk_audio(
    text: str,
    language: str = "",
    emotion: str = "",
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
) -> tuple[str, str] | None:
    """Synthesize *text* as natural speech with a GPT audio model.

    Args:
        text: The utterance to read aloud verbatim.
        language: Optional BCP-47 tag of the script's language (e.g.
            "en-US"); a hint only — the model reads the script as
            written in whatever language it is.
        emotion: Optional delivery vibe (e.g. "cheerful", "sad"); woven
            into the reading-tone instruction when provided.
        model: GPT audio-chat model name from MODEL_INFO.json.
        voice: OpenAI built-in voice name.

    Returns:
        ``(audio_b64, mime)`` — base64-encoded MP3 bytes and its MIME
        type ``"audio/mpeg"`` — or ``None`` when *text* is empty or the
        API call fails (callers then fall back to client-side TTS).
    """
    if not text.strip():
        return None
    tone = f"in a {emotion}, natural, expressive tone" if emotion else (
        "in a warm, natural, expressive tone"
    )
    system = TTS_SYSTEM_PROMPT.format(tone=tone)
    if language:
        system += f" The script's language tag is {language}."
    try:
        agent = TalkSynthesisAgent("talk-speech-synthesis")
        agent.run(
            model_name=model,
            prompt_template="Script:\n{script}",
            arguments={"script": text},
            system_prompt=system,
            is_agentic=False,
            model_config={
                "modalities": ["text", "audio"],
                "audio": {"voice": voice, "format": "mp3"},
                "timeout": audio_timeout_seconds(),
            },
            verbose=False,
        )
        data = getattr(agent.model, "last_audio_data", None)
        if not data:
            return None
        # Round-trip validates the base64 before shipping it to clients.
        base64.b64decode(data)
        return data, "audio/mpeg"
    except Exception as err:  # noqa: BLE001 — talk must degrade, not raise
        print(f"talk speech synthesis failed: {err}", file=sys.stderr, flush=True)
        return None
