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
  never speak over each other (``pumpTalkQueue`` parity), and
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

A shutdown ends the clip being played the same way and drops the
queued ones: :func:`shutdown` for the daemon's, and an ``atexit`` hook
for the interpreter's exit (the worker is a daemon thread, and a player
in a group of its own would outlive the process with nobody left to
time it out).
"""

from __future__ import annotations

import atexit
import base64
import logging
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from kiss.core.locked import locked_call
from kiss.core.processes import popen_process_group, terminate_process_group

logger = logging.getLogger(__name__)

_PLAY_CMD_ENV = "KISS_SORCAR_PLAY_CMD"
_PLAY_TIMEOUT_ENV = "KISS_SORCAR_PLAY_TIMEOUT"
_MAX_TALK_IDS = 500
_PLAYBACK_TIMEOUT = 600.0
# Seconds between two looks at a playing child (``Popen.wait``'s own interval).
_EXIT_POLL = 0.05
# Seconds a shutdown waits for the worker: both graces of the group kill, and slack.
_SHUTDOWN_WAIT = 12.0

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


def _run_playback(argv: list[str], stop: threading.Event) -> None:
    """Run one playback child to completion.

    A player that cannot start or exits non-zero is skipped silently
    (the browser clients still play their copy), and its output is
    suppressed so a chatty player never pollutes the daemon's terminal
    or logs.  A hung child is killed after :func:`_playback_timeout`
    seconds, and a playing one as soon as *stop* is set (the player's
    shutdown) — together with its whole process group on POSIX.
    """
    try:
        proc: subprocess.Popen[bytes] = popen_process_group(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return
    deadline = time.monotonic() + _playback_timeout()
    # Waited for in slices on *stop*: the shutdown only sets it, so this
    # thread stays the one that signals the group and reaps its leader
    # (``terminate_process_group`` needs the leader unreaped to the end).
    while proc.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or stop.wait(min(remaining, _EXIT_POLL)):
            # The child was started in its own process group, so the kill
            # reaches grandchildren too: a ``KISS_SORCAR_PLAY_CMD`` wrapper
            # script's real player must die with the wrapper, or it would
            # keep the audio device and play over the next clip.  SIGTERM
            # first (players flush and release the device), SIGKILL five
            # seconds later; the wait after that is bounded, so a SIGKILLed
            # player stuck in the kernel (an audio device or network file
            # system that does not return) cannot wedge the sole talk
            # worker for good.  A child that outlives the wait stays ours;
            # ``subprocess`` reaps it later from its unreaped-children list
            # once it finally dies.
            terminate_process_group(proc)
            return


class TalkPlayer:
    """Plays broadcast ``talk`` events on the local machine's speakers.

    Thread-safe; :meth:`play` never blocks — events are queued to a
    single daemon worker thread that plays them strictly one at a
    time (the browser client's talk-queue behaviour).  Playback of the same
    ``talkId`` more than once is suppressed with a bounded set (the
    browser client's ``spokenTalkIds`` behaviour).  :meth:`shutdown`
    ends the player for good.
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
        # play.  ``None`` is the shutdown's wake-up.
        self._queue: queue.SimpleQueue[tuple[dict[str, Any], list[str]] | None] = (
            queue.SimpleQueue()
        )
        self._worker: threading.Thread | None = None
        # Set by ``shutdown``: nothing more is admitted or played, and
        # the clip being played is stopped.
        self._closed = threading.Event()

    def play(self, event: dict[str, Any]) -> bool:
        """Queue one ``talk`` *event* for playback (non-blocking).

        The machine's player (:func:`player_command`) is resolved here,
        once per event, and travels with the queued entry.  Events
        without a clip, events on a machine without a player and
        already-spoken ``talkId`` values are dropped (the latter
        matching the browser client).

        Args:
            event: The broadcast ``talk`` event dictionary.

        Returns:
            ``True`` when the machine has a player and the event carries
            a clip (it is queued, unless already spoken or the player
            was shut down); ``False`` otherwise.
        """
        if not event.get("audioB64"):
            return False
        player = player_command()
        if player is None:
            return False
        if locked_call(self._lock, self._admit_locked, event.get("talkId")):
            self._queue.put((dict(event), player))
        return True

    def _admit_locked(self, talk_id: Any) -> bool:
        """Make sure the worker runs and record *talk_id*; ``False`` when spoken or shut down."""
        if self._closed.is_set() or (talk_id and talk_id in self._spoken_ids):
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
        """Worker loop: play queued events one at a time, until :meth:`shutdown`."""
        # Only the published worker drains.  A Stop can cut ``start`` short
        # once this thread is spawned but before it is alive; the next
        # ``play`` then starts another, and two drainers would play two
        # clips at once.  (No Stop is injected here: a plain ``with``.)
        with self._lock:
            if self._worker is not threading.current_thread():
                return
        while (entry := self._queue.get()) is not None and not self._closed.is_set():
            try:
                self._play_clip(*entry)
            except Exception:
                logger.exception("talk playback failed")

    def shutdown(self) -> None:
        """End the player for good: stop the clip being played, drop the queued ones.

        Idempotent and bounded.  Nothing is admitted afterwards; the
        worker ends its player's whole process group (SIGTERM, SIGKILL
        after the grace), removes the clip's temp file and exits, and
        this waits for that at most ``_SHUTDOWN_WAIT`` seconds (a
        SIGKILLed player stuck in the kernel is left to the worker, as
        on a playback timeout).
        """
        worker = locked_call(self._lock, self._close_locked)
        # Alive only: a worker whose ``start`` a Stop cut short may not run
        # yet (it cannot be joined), and ends at the wake-up once it does.
        if worker is not None and worker.is_alive():
            worker.join(_SHUTDOWN_WAIT)

    def _close_locked(self) -> threading.Thread | None:
        """Close admission and wake the worker; return it."""
        self._closed.set()
        self._queue.put(None)
        return self._worker

    def _play_clip(self, event: dict[str, Any], player: list[str]) -> None:
        """Play the event's ``audioB64`` clip with the *player* argv prefix.

        An undecodable or empty clip is skipped silently; the browser
        clients still play their copy.
        """
        try:
            audio = base64.b64decode(event["audioB64"], validate=True)
        except (ValueError, TypeError):
            return
        if not audio:
            return
        suffix = ".mp3" if "mpeg" in str(event.get("audioMime") or "") else ".audio"
        fd, path = tempfile.mkstemp(prefix="sorcar-talk-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            _run_playback([*player, path], self._closed)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


_shared_lock = threading.RLock()
_shared_player: TalkPlayer | None = None
_exiting = False


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


def shutdown() -> None:
    """Shut the shared player down, if there is one: the daemon's shutdown.

    Called by ``web_server.RemoteAccessServer._shutdown_children``.
    Bounded and idempotent (:meth:`TalkPlayer.shutdown`).  The next
    :func:`shared_player` call makes a fresh player, for an embedder
    that starts another server in this process.
    """
    player = locked_call(_shared_lock, _take_shared_player)
    if player is not None:
        player.shutdown()


def _take_shared_player() -> TalkPlayer | None:
    global _shared_player
    player = _shared_player
    if not _exiting:
        _shared_player = None
    return player


def _shutdown_at_exit() -> None:
    """Interpreter exit: shut the shared player down and keep it the shared one.

    Threads run until every exit hook has, so a talk can still arrive:
    it reaches this shut-down player, never a fresh one that nothing
    would stop.
    """
    global _exiting
    _exiting = True
    shared_player().shutdown()


atexit.register(_shutdown_at_exit)
