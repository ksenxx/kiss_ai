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

A start that a task's Stop cuts short after the thread was spawned is
the other half (audit 2026-11-10 B/V7-F1): that thread is not alive
yet, the next ``play`` starts a second one, and both drained the queue.
"""

from __future__ import annotations

import _thread
import base64
import shlex
import stat
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType

import pytest

from kiss.server.talk_player import TalkPlayer
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.lock_injection import inject

MP3_B64 = base64.b64encode(b"ID3\x03\x00fake-mp3-" + bytes(range(32))).decode("ascii")


def _wait_for(predicate: Callable[[], bool], timeout: float) -> bool:
    """Poll *predicate* until it holds or *timeout* seconds have passed."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


@pytest.mark.skipif(
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


def test_a_worker_start_cut_short_by_a_stop_leaves_one_drainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two clips queued after a Stop in ``Thread.start`` still play one after the other."""
    log, gate = tmp_path / "log", tmp_path / "gate"
    script = tmp_path / "player.py"
    script.write_text(
        "import os, time\n"
        f"open({str(log)!r}, 'a').write('start\\n')\n"
        "deadline = time.monotonic() + 60\n"
        f"while not os.path.exists({str(gate)!r}) and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        f"open({str(log)!r}, 'a').write('end\\n')\n"
    )
    monkeypatch.setenv(
        "KISS_SORCAR_PLAY_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    )
    monkeypatch.delenv("KISS_SORCAR_PLAY_TIMEOUT", raising=False)
    player = TalkPlayer()

    def stop_once_spawned(frame: FrameType, event: str, arg: object) -> None:
        if event == "c_return" and arg is _thread.start_joinable_thread:
            inject(threading.current_thread())  # the runner's Stop, the thread just spawned

    # A spawned thread comes alive once it gets the GIL: with a long switch
    # interval that is when this thread next blocks, inside the second play.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(5)
    time.sleep(0.05)  # threads that asked for the GIL under the old interval get it here
    sys.setprofile(stop_once_spawned)
    try:
        with pytest.raises(KeyboardInterrupt):
            player.play({"talkId": "cancelled", "audioB64": MP3_B64})
        sys.setprofile(None)
        spawned = player._worker
        player.play({"talkId": "second", "audioB64": MP3_B64})
    finally:
        sys.setprofile(None)
        sys.setswitchinterval(interval)
    try:
        player.play({"talkId": "third", "audioB64": MP3_B64})
        workers = {spawned, player._worker}
        assert _wait_for(lambda: sum(w in threading.enumerate() for w in workers) == 1, 10.0), (
            "the thread of the interrupted start drains the queue beside its successor"
        )
        gate.write_text("go")
        assert _wait_for(lambda: log.exists() and log.read_text().count("end") == 2, 30.0)
        assert log.read_text().split() == ["start", "end", "start", "end"]
    finally:
        gate.write_text("go")
        player.shutdown()
