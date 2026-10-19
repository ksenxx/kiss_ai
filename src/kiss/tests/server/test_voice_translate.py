# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Transcript post-processing of the voice pipeline (``kiss.server.speech``).

Exercises the transcript helpers — preamble/quote stripping, wake-word
prefix removal, WAV framing — with real model-output shapes observed
in probes.  The end-to-end transcription of real audio lives in
``test_voice_agent_transcribe`` and ``test_web_voice_transcribe``.
"""

from __future__ import annotations

import io
import math
import struct
import unittest
import wave

from kiss.server.speech import (
    BLOCK_SIZE,
    SAMPLE_RATE,
    clean_transcript,
    pcm_to_wav_bytes,
    strip_leading_wake_word,
)


def _loud_block() -> bytes:
    """One audio block of a clearly audible 440Hz tone (s16le)."""
    frames = [
        int(12000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        for i in range(BLOCK_SIZE)
    ]
    return struct.pack(f"<{BLOCK_SIZE}h", *frames)


class TestTranscriptHelpers(unittest.TestCase):
    """Transcript cleanup and WAV framing used by ``transcribe_pcm``."""

    def test_strip_leading_wake_word_aliases(self) -> None:
        self.assertEqual(strip_leading_wake_word("Abracadabra, fix it"), "fix it")
        self.assertEqual(
            strip_leading_wake_word("abracadabra please fix it"),
            "please fix it",
        )
        self.assertEqual(
            strip_leading_wake_word("Abrakadabra! fix the failing test"),
            "fix the failing test",
        )
        self.assertEqual(
            strip_leading_wake_word("Abra cadabra - fix it"), "fix it"
        )
        self.assertEqual(
            strip_leading_wake_word("Abracadabra. Abracadabra, fix it"), "fix it"
        )
        # Only a LEADING wake word is cut; the word elsewhere is content.
        self.assertEqual(
            strip_leading_wake_word("fix the Abracadabra bug"),
            "fix the Abracadabra bug",
        )
        # "Sorcar" is no longer part of the wake phrase, so it is plain
        # text, as are the "Hey" and the name aliases of earlier phrases.
        self.assertEqual(
            strip_leading_wake_word("Sorcar, fix it"), "Sorcar, fix it"
        )
        self.assertEqual(
            strip_leading_wake_word("Hey there, fix it"), "Hey there, fix it"
        )
        self.assertEqual(
            strip_leading_wake_word("sir car please fix it"),
            "sir car please fix it",
        )

    def test_clean_transcript_strips_preamble_and_quotes(self) -> None:
        self.assertEqual(
            clean_transcript(
                'Sure. Here is the transcription of the speech:\n\n'
                '"Fix the failing test in the parser."'
            ),
            "Fix the failing test in the parser.",
        )
        self.assertEqual(
            clean_transcript("Here's the translation: Fix the bug."),
            "Fix the bug.",
        )
        self.assertEqual(
            clean_transcript('"Hello world."'),
            "Hello world.",
        )
        self.assertEqual(
            clean_transcript("Fix the bug in the main file."),
            "Fix the bug in the main file.",
        )
        self.assertEqual(clean_transcript("  \n "), "")

    def test_pcm_to_wav_bytes_roundtrip(self) -> None:
        pcm = _loud_block()
        wav_bytes = pcm_to_wav_bytes(pcm)
        # Read back through memory: reopening a still-open
        # NamedTemporaryFile by name is refused on Windows.
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), SAMPLE_RATE)
            self.assertEqual(wf.readframes(wf.getnframes()), pcm)


if __name__ == "__main__":
    unittest.main()
