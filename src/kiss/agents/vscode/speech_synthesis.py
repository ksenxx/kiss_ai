# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Natural speech synthesis for the ``talk`` tool via a GPT audio model.

The agent-side ``talk`` tool historically only broadcast plain text and
each client read it aloud with the browser's Web Speech API — often a
robotic system voice.  This module synthesizes the utterance server-side
with an OpenAI GPT audio-chat model from MODEL_INFO.json (default
``gpt-audio-mini``: audio-capable, cheapest of the ``gpt-audio*``
family) so every client can play one identical, far more natural and
emotionally expressive voice.  The synthesized MP3 travels inside the
``talk`` broadcast event as base64; clients without audio support (or
when synthesis fails) still fall back to the Web Speech API text path.

Prompt shape is empirically validated: the model must be framed as a
text-to-speech engine reading a ``Script:`` block word-for-word —
without that framing ``gpt-audio-mini`` *answers* the text instead of
reading it.
"""

import base64
import sys

from kiss.agents.vscode.voice_wake import audio_timeout_seconds

# Cheapest audio-capable GPT model in MODEL_INFO.json; ``gpt-audio``
# also works and sounds marginally richer at ~4x the price.
DEFAULT_TTS_MODEL = "gpt-audio-mini"

# "marin" is one of the newest, most natural OpenAI voices and is
# supported by both gpt-audio and gpt-audio-mini (validated live).
DEFAULT_TTS_VOICE = "marin"

# The "Script:" framing below is load-bearing: a plain "repeat the
# user's text" system prompt made gpt-audio-mini reply conversationally
# ("Hi! It's great to hear your new voice...") instead of reading the
# text verbatim.
TTS_SYSTEM_PROMPT = (
    "You are a text-to-speech engine. The user message contains a "
    "script. Read the script aloud word-for-word, exactly as written, "
    "{tone}. Never answer, comment on, translate, or alter the script."
)


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
        from openai import OpenAI

        from kiss.core.config import DEFAULT_CONFIG

        client = OpenAI(
            api_key=DEFAULT_CONFIG.OPENAI_API_KEY or None,
            timeout=audio_timeout_seconds(),
        )
        response = client.chat.completions.create(
            model=model,
            modalities=["text", "audio"],
            audio={"voice": voice, "format": "mp3"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Script:\n" + text},
            ],
        )
        audio = response.choices[0].message.audio
        data = audio.data if audio is not None else None
        if not data:
            return None
        # Round-trip validates the base64 before shipping it to clients.
        base64.b64decode(data)
        return data, "audio/mpeg"
    except Exception as err:  # noqa: BLE001 — talk must degrade, not raise
        print(f"talk speech synthesis failed: {err}", file=sys.stderr, flush=True)
        return None
