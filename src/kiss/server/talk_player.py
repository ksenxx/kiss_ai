# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Daemon-side playback of agent ``talk`` events on local speakers.

The ``talk`` tool (sorcar_agent.py) broadcasts a
``{"type": "talk", "text", "language", "emotion", "talkId",
"audioB64", "audioMime"}`` event.  Browser clients play the synthesized
MP3 themselves (``media/main.js`` ``playTalkAudio``).  A headless local
client (a ``sorcar`` CLI or ``daemon_client`` run on the daemon's machine)
has no player, so the kiss-web daemon uses :func:`shared_player` to play
the clip on this machine's speakers
(``web_server.WebPrinter._play_talk_clip_locally``):

* the base64 MP3 is decoded to a temp file and played with the
  machine's audio player (``afplay`` on macOS; ``mpg123`` / ``ffplay``
  / ``mpv`` elsewhere); an event without a clip is dropped (the
  caller only hands over events that carry one);
* utterances are deduped by ``talkId`` (bounded set, as in main.js)
  and serialised on one worker thread so back-to-back ``talk()`` calls
  never speak over each other (``enqueueTalkPlayback`` parity), and
  the agent loop never blocks on playback.

The player command honours the ``KISS_SORCAR_PLAY_CMD`` environment
override (receives the audio file path as its last argument, parsed
with :func:`shlex.split`), so tests can substitute a real scripted
child process.  The hung-playback
timeout honours ``KISS_SORCAR_PLAY_TIMEOUT`` (seconds, default 600)
for the same reason.  Because an override may be a wrapper script
whose real player runs as a grandchild, each playback child is
started in its own process group (POSIX) and a timeout kills the
whole group — killing only the direct child would leave the
grandchild holding the audio device and overlapping the next clip.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import math
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any

from kiss.core.locked import locked_call
from kiss.core.processes import kill_process_group, popen_process_group

logger = logging.getLogger(__name__)

_PLAY_CMD_ENV = "KISS_SORCAR_PLAY_CMD"
_PLAY_TIMEOUT_ENV = "KISS_SORCAR_PLAY_TIMEOUT"
_MAX_TALK_IDS = 500
_PLAYBACK_TIMEOUT = 600.0
# Grace between the group SIGTERM and the SIGKILL escalation: long
# enough for a well-behaved player to release the audio device, short
# enough that the serialising worker is never stuck behind a zombie.
_KILL_GRACE = 5.0

_FALLBACK_PLAYERS: tuple[tuple[str, ...], ...] = (
    ("mpg123", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
    ("mpv", "--no-video", "--really-quiet"),
)


def player_command() -> list[str] | None:
    """Return the argv prefix used to play an audio file, or ``None``.

    Honours the ``KISS_SORCAR_PLAY_CMD`` environment variable (parsed
    with :func:`shlex.split`, so quoted arguments survive; a malformed
    override logs a warning and disables playback); otherwise picks
    the machine's audio player: ``afplay`` on macOS, else the first of
    ``mpg123`` / ``ffplay`` / ``mpv`` found on ``PATH``.  The audio
    file path is appended as the command's last argument.

    Returns:
        The player argv prefix, or ``None`` when no player exists.
    """
    override = os.environ.get(_PLAY_CMD_ENV, "").strip()
    if override:
        try:
            parts = shlex.split(override)
        except ValueError:
            logger.warning("Malformed %s: %r", _PLAY_CMD_ENV, override)
            return None
        return parts or None
    if sys.platform == "darwin":
        return ["afplay"]
    for candidate in _FALLBACK_PLAYERS:
        if shutil.which(candidate[0]):
            return list(candidate)
    return None


def _playback_timeout() -> float:
    """Return the hung-playback timeout in seconds.

    Honours the ``KISS_SORCAR_PLAY_TIMEOUT`` environment variable so
    tests can force the timeout path with a real hung child instead
    of waiting out the 600 s default; a missing, malformed,
    non-finite, or non-positive value falls back to
    ``_PLAYBACK_TIMEOUT`` — ``inf`` (or an overflowing literal such
    as ``1e999``) would otherwise pass the ``> 0`` gate and hand
    ``Popen.wait`` an infinite timeout, disabling the hung-playback
    group kill entirely (gpt-5.6-sol conc review, finding 6).

    Returns:
        The timeout in seconds (always finite and > 0).
    """
    raw = os.environ.get(_PLAY_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Malformed %s: %r", _PLAY_TIMEOUT_ENV, raw)
            return _PLAYBACK_TIMEOUT
        if math.isfinite(value) and value > 0:
            return value
    return _PLAYBACK_TIMEOUT


def _kill_playback(proc: subprocess.Popen[bytes]) -> None:
    """Kill a hung playback child and every process it spawned.

    The child was started in its own process group, so signalling
    the group (:func:`kiss.core.processes.kill_process_group`) reaches
    grandchildren too — a ``KISS_SORCAR_PLAY_CMD`` wrapper script's
    real player must die with the wrapper, or it would keep the audio
    device and play over the next clip.  SIGTERM first (players flush
    and release the device), escalating to SIGKILL after
    ``_KILL_GRACE`` seconds.
    """
    try:
        kill_process_group(proc.pid, signal.SIGTERM)
        proc.wait(timeout=_KILL_GRACE)
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        # The group could not be signalled (or is already gone): kill
        # the child alone so the caller's final wait normally ends
        # at once instead of running out its timeout.
        proc.kill()
        return
    try:
        kill_process_group(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def _run_playback(argv: list[str]) -> bool:
    """Run one playback child to completion; ``True`` on exit code 0.

    Output is suppressed so a chatty player never pollutes the
    daemon's terminal or logs.  A hung child is killed after
    :func:`_playback_timeout` seconds — together with its whole
    process group on POSIX (see :func:`_kill_playback`).
    """
    try:
        proc: subprocess.Popen[bytes] = popen_process_group(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    try:
        return proc.wait(timeout=_playback_timeout()) == 0
    except subprocess.TimeoutExpired:
        _kill_playback(proc)
        # Bounded: a SIGKILLed player stuck in the kernel (an audio
        # device or network file system that does not return) would
        # otherwise wedge the sole talk worker for good.  A child that
        # outlives the wait stays ours; ``subprocess`` reaps it later
        # from its unreaped-children list once it finally dies.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        return False


class TalkPlayer:
    """Plays broadcast ``talk`` events on the local machine's speakers.

    Thread-safe; :meth:`play` never blocks — events are queued to a
    single daemon worker thread that plays them strictly one at a
    time (the browser client's talk-queue behaviour).  Playback of the same
    ``talkId`` more than once is suppressed with a bounded set (the
    browser client's ``spokenTalkIds`` behaviour).
    """

    def __init__(self) -> None:
        # Taken on the task thread of the ``talk`` tool (``WebPrinter.
        # broadcast`` -> ``_fanout_talk`` -> ``play``), where the
        # runner's Stop injects ``KeyboardInterrupt``: a reentrant lock
        # through ``locked_call``, never a plain ``with``.
        self._lock = threading.RLock()
        self._spoken_ids: dict[str, None] = {}
        # A SimpleQueue, whose put/get are C calls: a queue.Queue's
        # put is a Python ``with self.not_full:``, which the same
        # injection leaks, wedging the worker's get and every later
        # play.
        self._queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None

    def play(self, event: dict[str, Any]) -> None:
        """Queue one ``talk`` *event* for playback (non-blocking).

        Events without a clip and already-spoken ``talkId`` values
        are dropped (the latter matching the browser client).

        Args:
            event: The broadcast ``talk`` event dictionary.
        """
        if not event.get("audioB64"):
            return
        if locked_call(self._lock, self._admit_locked, event.get("talkId")):
            self._queue.put(dict(event))

    def _admit_locked(self, talk_id: Any) -> bool:
        """Make sure the worker runs and record *talk_id*; ``False`` when already spoken."""
        if talk_id and talk_id in self._spoken_ids:
            return False
        # The worker before the record: a start that fails (thread
        # exhaustion) raises out of ``play`` with nothing remembered,
        # so the talk re-sent once the pressure is over still plays.
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._drain, name="sorcar-talk-player", daemon=True
            )
            self._worker.start()
        if talk_id:
            self._spoken_ids[talk_id] = None
            if len(self._spoken_ids) > _MAX_TALK_IDS:
                for stale in list(self._spoken_ids)[: _MAX_TALK_IDS // 2]:
                    del self._spoken_ids[stale]
        return True

    def _drain(self) -> None:
        """Worker loop: play queued events one at a time, forever."""
        while True:
            event = self._queue.get()
            try:
                self._play_clip(event)
            except Exception:
                logger.exception("talk playback failed")

    def _play_clip(self, event: dict[str, Any]) -> None:
        """Play the event's ``audioB64`` clip with the machine's audio player.

        An undecodable or empty clip, or a machine without a player, is
        skipped silently; the browser clients still play their copy.
        """
        try:
            audio = base64.b64decode(event["audioB64"], validate=True)
        except (ValueError, TypeError):
            return
        player = player_command()
        if not audio or player is None:
            return
        suffix = ".mp3" if "mpeg" in str(event.get("audioMime") or "") else ".audio"
        fd, path = tempfile.mkstemp(prefix="sorcar-talk-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            _run_playback([*player, path])
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


_shared_lock = threading.RLock()
_shared_player: TalkPlayer | None = None


def shared_player() -> TalkPlayer:
    """Return the process-wide :class:`TalkPlayer` singleton.

    One player per process mirrors the browser client's one-queue-per-device
    semantics: talks from every printer/agent in this process are
    deduped and serialised together, so parallel sub-agents can never
    speak over each other on the same speakers.  Called on the task
    thread of the ``talk`` tool, so the lock goes through
    :func:`~kiss.core.locked.locked_call`.

    Returns:
        The shared :class:`TalkPlayer`.
    """
    return locked_call(_shared_lock, _shared_player_locked)


def _shared_player_locked() -> TalkPlayer:
    global _shared_player
    if _shared_player is None:
        _shared_player = TalkPlayer()
    return _shared_player


def reset_shared_player_for_tests() -> None:
    """Discard the shared player (test isolation for talkId dedupe)."""
    global _shared_player
    with _shared_lock:
        _shared_player = None
