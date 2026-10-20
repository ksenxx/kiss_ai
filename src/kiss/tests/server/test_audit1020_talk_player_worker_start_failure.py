# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a talk whose worker thread cannot start is not remembered as spoken.

``TalkPlayer.play`` recorded the ``talkId`` as spoken under the lock and
then started the playback worker (on the first talk).  When
``Thread.start`` raised (thread exhaustion), the id stayed in the
spoken set, so the same talk re-sent after the pressure passed was
deduped and never played (audit 2026-10-20 C5-F3).  The fix starts the
worker before recording the id, so a failed start leaves nothing
behind.

The failure is real: ``RLIMIT_NPROC`` is lowered to 1 around the first
``play`` (threads count against it for a non-root uid on Linux), so the
thread genuinely cannot start.  Playback is observed through
``KISS_SORCAR_PLAY_CMD`` pointing at a script that records its run.
"""

from __future__ import annotations

import base64
import shlex
import stat
import time
from pathlib import Path

import pytest

from kiss.server.talk_player import TalkPlayer
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved

MP3_B64 = base64.b64encode(b"ID3\x03\x00fake-mp3-" + bytes(range(32))).decode("ascii")

pytestmark = pytest.mark.skipif(
    not thread_start_can_be_starved(),
    reason="RLIMIT_NPROC does not starve threads here",
)


def test_talk_admitted_before_a_failed_worker_start_plays_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The talk is replayable after the worker could not be started for it."""
    played = tmp_path / "played.txt"
    script = tmp_path / "player.sh"
    script.write_text(f"#!/bin/sh\necho played >> {shlex.quote(str(played))}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", shlex.quote(str(script)))
    monkeypatch.delenv("KISS_SORCAR_PLAY_TIMEOUT", raising=False)
    player = TalkPlayer()
    event = {"talkId": "talk-1", "audioB64": MP3_B64, "audioMime": "audio/mpeg"}

    with nproc_limit_lowered_to_one(), pytest.raises(RuntimeError):
        player.play(event)

    player.play(event)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not played.exists():
        time.sleep(0.05)
    assert played.exists(), "the re-sent talk was deduped against the failed admission"
    time.sleep(0.5)  # a queued leftover from the failed start would play it twice
    assert played.read_text().count("played") == 1
