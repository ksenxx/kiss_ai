# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 V8 F4: dictated text that contains a JSON object is kept.

``parse_transcription_reply`` tried the substring from the first ``{``
to the last ``}`` before the requested two-line shape, so a compliant
``en\\n...`` reply whose dictation mentions ``{"enabled": true}`` was
replaced by that object's (missing) ``text`` field.  The language-tag
line now wins, and an object without ``text``/``language`` keys is
dictation, not an envelope.  A genuine envelope, fenced or wrapped in
prose, still parses as before.
"""

from __future__ import annotations

import unittest

from kiss.server.speech import parse_transcription_reply


class EmbeddedJsonIsDictation(unittest.TestCase):
    """The two-line shape is recognised before any JSON object inside it."""

    def test_object_without_text_key_in_tagged_dictation(self) -> None:
        text, language = parse_transcription_reply(
            'en\nPlease write {"enabled": true} into config.json.'
        )
        self.assertEqual(text, 'Please write {"enabled": true} into config.json.')
        self.assertEqual(language, "en")

    def test_object_with_text_key_in_tagged_dictation(self) -> None:
        text, language = parse_transcription_reply(
            'en\nExplain why {"text": "unexpected replacement"} is invalid here.'
        )
        self.assertEqual(text, 'Explain why {"text": "unexpected replacement"} is invalid here.')
        self.assertEqual(language, "en")

    def test_object_without_envelope_keys_and_no_tag_is_plain_text(self) -> None:
        reply = 'Please write {"enabled": true} into config.json.'
        text, language = parse_transcription_reply(reply)
        self.assertEqual(text, reply)
        self.assertIsNone(language)

    def test_object_with_envelope_keys_inside_an_untagged_sentence_is_plain_text(self) -> None:
        """An object inside a sentence is dictation whatever its keys: envelopes are whole lines."""
        for reply in (
            'Update config to {"language": "fr"}',
            'Set the response to {"text": "Welcome"} and save',
            'Both {"language": "en", "text": "x"} keys, mid-sentence',
        ):
            text, language = parse_transcription_reply(reply)
            self.assertEqual(text, reply)
            self.assertIsNone(language, reply)

    def test_fenced_tagged_dictation_with_object(self) -> None:
        text, language = parse_transcription_reply(
            '```\nfr\nSet {"debug": false} in the file.\n```'
        )
        self.assertEqual(text, 'Set {"debug": false} in the file.')
        self.assertEqual(language, "fr")

    def test_genuine_envelopes_still_parse(self) -> None:
        envelope = '{"language": "en", "text": "ordinary JSON envelope"}'
        for reply in (
            envelope,
            f"```json\n{envelope}\n```",
            f"Here is the transcription:\n{envelope}\nDone.",
        ):
            text, language = parse_transcription_reply(reply)
            self.assertEqual(text, "ordinary JSON envelope", reply)
            self.assertEqual(language, "en", reply)


if __name__ == "__main__":
    unittest.main()
