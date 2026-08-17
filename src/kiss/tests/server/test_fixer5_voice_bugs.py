# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixer-5 voice_wake bugs (findings F5-09, F5-10).

F5-09 — the voice-model cache dir was frozen to the real
``~/.kiss/models`` at import time, bypassing the repository-wide lazy
``$KISS_HOME`` contract; profile/test isolation was broken and model
downloads (hundreds of MB) landed in the wrong home.

F5-10 — two advertised reply-normalization shapes were mishandled by
``parse_transcription_reply``: a fenced plain-text reply fell back to
the RAW reply (forwarding literal backticks into the dictated
command), and a "Here is the transcription:" preamble before the
language-tag line defeated language parsing, leaking the tag into the
command text.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest

import kiss.server.voice_wake as voice_wake
from kiss.server.voice_wake import parse_transcription_reply


class TestModelsDirHonoursKissHome(unittest.TestCase):
    """F5-09: the models dir resolves ``$KISS_HOME`` lazily."""

    def test_default_models_dir_follows_kiss_home(self) -> None:
        import os

        old = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = "/tmp/fixer5-profile"
        try:
            self.assertEqual(
                voice_wake.default_models_dir(),
                Path("/tmp/fixer5-profile") / "models",
            )
            # The legacy module constant must resolve lazily too —
            # voice_wake was imported long before KISS_HOME changed.
            self.assertEqual(
                voice_wake.DEFAULT_MODELS_DIR,
                Path("/tmp/fixer5-profile") / "models",
            )
        finally:
            if old is None:
                os.environ.pop("KISS_HOME", None)
            else:
                os.environ["KISS_HOME"] = old

    def test_default_models_dir_without_kiss_home(self) -> None:
        import os

        old = os.environ.pop("KISS_HOME", None)
        try:
            self.assertEqual(
                voice_wake.default_models_dir(),
                Path.home() / ".kiss" / "models",
            )
        finally:
            if old is not None:
                os.environ["KISS_HOME"] = old

    def test_unknown_module_attribute_still_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = voice_wake.NO_SUCH_ATTRIBUTE  # type: ignore[attr-defined]


class TestParseTranscriptionReply(unittest.TestCase):
    """F5-10: fence fallback and preamble-before-tag parsing."""

    def test_fenced_plain_text_loses_its_fences(self) -> None:
        text, language = parse_transcription_reply("```\nhello\n```")
        self.assertEqual(text, "hello")
        self.assertIsNone(language)

    def test_fenced_plain_text_with_language_hint_fence(self) -> None:
        text, language = parse_transcription_reply(
            "```text\nopen the door\n```",
        )
        self.assertEqual(text, "open the door")
        self.assertIsNone(language)

    def test_preamble_before_language_tag_is_stripped(self) -> None:
        text, language = parse_transcription_reply(
            "Here is the transcription:\nen\nOpen the window.",
        )
        self.assertEqual(text, "Open the window.")
        self.assertEqual(language, "en")

    def test_two_line_shape_still_parses(self) -> None:
        text, language = parse_transcription_reply("en-us  \nTurn it off.")
        self.assertEqual(text, "Turn it off.")
        self.assertEqual(language, "en-us")

    def test_json_shape_still_parses(self) -> None:
        text, language = parse_transcription_reply(
            '{"language": "fr", "text": "Bonjour"}',
        )
        self.assertEqual(text, "Bonjour")
        self.assertEqual(language, "fr")

    def test_plain_text_reply_is_untouched(self) -> None:
        text, language = parse_transcription_reply("Just do the thing.")
        self.assertEqual(text, "Just do the thing.")
        self.assertIsNone(language)


if __name__ == "__main__":
    unittest.main()
