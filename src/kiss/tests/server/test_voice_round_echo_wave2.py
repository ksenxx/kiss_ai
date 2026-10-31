# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``voiceSpeech`` reply carries the round tag of the dictation it answers.

``voice.js`` tags each posted dictation with ``round`` (a number) and
pairs a reply that echoes the tag with exactly that round; an untagged
reply is paired by position with the oldest posted round.  The daemon
never echoed the tag, so the late reply of a round the page had
cancelled (mic off, flash timeout) was paired with the NEXT round once
that one had posted its audio: the cancelled words were typed into the
conversation on screen.  The reply now echoes an integer tag and sends
``null`` for anything else (a missing, boolean or string tag), which the
page treats as untagged.

Real daemon over its local WSS endpoint, one real client.  The dictations
carry no audio, so no transcription model is asked.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from kiss.tests.server.two_client_harness import Client, TwoClientHarness


def _replies(client: Client, count: int) -> list[dict[str, Any]] | None:
    """*client*'s ``voiceSpeech`` replies once there are *count* of them, else ``None``."""
    found = client.of_type("voiceSpeech")
    return found if len(found) == count else None


class VoiceRoundEchoTest(TwoClientHarness):
    """Each ``voiceSpeech`` reply names its dictation's round."""

    def test_reply_echoes_an_integer_round_and_nulls_anything_else(self) -> None:
        client = self.open_client("a")
        client.ready()
        self.wait_event(client, "models")

        tags: list[object] = [7, True, "7", None]
        for tag in tags:
            cmd: dict[str, object] = {
                "type": "voiceTranscribe", "tabId": client.tab_id, "audio": "",
            }
            if tag is not None:
                cmd["round"] = tag
            client.send(cmd)
        replies = self.wait_for(
            partial(_replies, client, len(tags)),
            what="a voiceSpeech reply to every dictation",
        )
        # Answered in send order (ServerApi.voice_transcribe chains them).
        self.assertEqual(
            [reply.get("round", "absent") for reply in replies], [7, None, None, None],
        )
        self.assertEqual({reply["text"] for reply in replies}, {""})
