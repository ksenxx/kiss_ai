# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``KISS_VOICE_AUDIO_TIMEOUT`` override of the transcription timeout.

The gpt-audio transcription request (``kiss.server.speech.transcribe_pcm``)
is bounded by :func:`kiss.core.speech_synthesis.audio_timeout_seconds`
so a stalled network path fails fast; the environment override must
accept only finite positive values.
"""

from __future__ import annotations

import os
import unittest


class TestAudioTimeoutOverride(unittest.TestCase):
    """KISS_VOICE_AUDIO_TIMEOUT accepts only finite positive values."""

    def _timeout_with(self, raw: str | None) -> float:
        from kiss.server.speech import audio_timeout_seconds

        saved = os.environ.get("KISS_VOICE_AUDIO_TIMEOUT")
        if raw is None:
            os.environ.pop("KISS_VOICE_AUDIO_TIMEOUT", None)
        else:
            os.environ["KISS_VOICE_AUDIO_TIMEOUT"] = raw
        try:
            return audio_timeout_seconds()
        finally:
            if saved is None:
                os.environ.pop("KISS_VOICE_AUDIO_TIMEOUT", None)
            else:
                os.environ["KISS_VOICE_AUDIO_TIMEOUT"] = saved

    def test_valid_override(self) -> None:
        self.assertEqual(self._timeout_with("7.5"), 7.5)

    def test_invalid_values_fall_back_to_default(self) -> None:
        from kiss.core.speech_synthesis import DEFAULT_AUDIO_TIMEOUT_SECONDS

        for raw in (None, "", "abc", "-3", "0", "nan", "inf", "-inf"):
            self.assertEqual(
                self._timeout_with(raw),
                DEFAULT_AUDIO_TIMEOUT_SECONDS,
                msg=f"raw={raw!r}",
            )


if __name__ == "__main__":
    unittest.main()
