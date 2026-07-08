# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Terminal-side playback of agent ``talk`` events for the sorcar CLI.

The ``talk`` tool (sorcar_agent.py) broadcasts a
``{"type": "talk", "text", "language", "emotion", "talkId",
"audioB64", "audioMime"}`` event.  Chat-webview clients play the
synthesized MP3 (``media/main.js`` ``playTalkAudio``) and fall back to
the Web Speech API when the clip is missing or undecodable.  A pure
CLI session has no webview, so :class:`TalkPlayer` gives the terminal
machine the same behaviour:

* the base64 MP3 is decoded to a temp file and played with the
  machine's audio player (``afplay`` on macOS; ``mpg123`` / ``ffplay``
  / ``mpv`` elsewhere);
* when there is no clip, the clip is undecodable, or the player fails,
  the text is spoken with the system TTS command (``say`` on macOS;
  ``espeak`` / ``spd-say`` elsewhere);
* utterances are deduped by ``talkId`` (bounded set, webview parity)
  and serialised on one worker thread so back-to-back ``talk()`` calls
  never speak over each other (``enqueueTalkPlayback`` parity), and
  the agent loop never blocks on playback.

Both external commands honour environment overrides —
``KISS_SORCAR_PLAY_CMD`` (receives the audio file path as its last
argument) and ``KISS_SORCAR_SAY_CMD`` (receives the utterance text as
its last argument), each parsed with :func:`shlex.split` — the same
pattern as ``KISS_SORCAR_VOICE_CMD`` in :mod:`cli_voice`, so tests can
substitute real scripted child processes.
"""

from __future__ import annotations

import base64
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

logger = logging.getLogger(__name__)

_PLAY_CMD_ENV = "KISS_SORCAR_PLAY_CMD"
_SAY_CMD_ENV = "KISS_SORCAR_SAY_CMD"
# webview parity: media/main.js caps ``spokenTalkIds`` at 500 and
# drops the oldest half when the cap is hit.
_MAX_TALK_IDS = 500
# Safety valve so one hung player can never wedge the talk queue.
_PLAYBACK_TIMEOUT = 600.0

# Linux/BSD MP3-capable players, most common first; each entry is the
# full argv prefix (the audio path is appended as the last argument).
_FALLBACK_PLAYERS: tuple[tuple[str, ...], ...] = (
    ("mpg123", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
    ("mpv", "--no-video", "--really-quiet"),
)
# Linux TTS commands (the text is appended as the last argument).
_FALLBACK_SAY: tuple[tuple[str, ...], ...] = (
    ("say",),  # macOS
    ("espeak-ng",),
    ("espeak",),
    ("spd-say", "--wait"),
)


def player_command() -> list[str] | None:
    """Return the argv prefix used to play an audio file, or ``None``.

    Honours the ``KISS_SORCAR_PLAY_CMD`` environment variable (parsed
    with :func:`shlex.split`, so quoted arguments survive); otherwise
    picks the machine's audio player: ``afplay`` on macOS, else the
    first of ``mpg123`` / ``ffplay`` / ``mpv`` found on ``PATH``.  The
    audio file path is appended as the command's last argument.

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


def say_command() -> list[str] | None:
    """Return the argv prefix used to speak text aloud, or ``None``.

    Honours the ``KISS_SORCAR_SAY_CMD`` environment variable (parsed
    with :func:`shlex.split`); otherwise picks the machine's TTS
    command: ``say`` on macOS, else the first of ``espeak-ng`` /
    ``espeak`` / ``spd-say`` found on ``PATH``.  The utterance text is
    appended as the command's last argument.

    Returns:
        The TTS argv prefix, or ``None`` when no TTS command exists.
    """
    override = os.environ.get(_SAY_CMD_ENV, "").strip()
    if override:
        try:
            parts = shlex.split(override)
        except ValueError:
            logger.warning("Malformed %s: %r", _SAY_CMD_ENV, override)
            return None
        return parts or None
    if sys.platform == "darwin":
        return ["say"]
    for candidate in _FALLBACK_SAY[1:]:
        if shutil.which(candidate[0]):
            return list(candidate)
    return None


def _run_playback(argv: list[str]) -> bool:
    """Run one playback child to completion; ``True`` on exit code 0.

    Output is suppressed so a chatty player never corrupts the
    anchored REPL's escape-coded terminal painting.  A hung child is
    killed after ``_PLAYBACK_TIMEOUT`` seconds.
    """
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    try:
        return proc.wait(timeout=_PLAYBACK_TIMEOUT) == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False


class TalkPlayer:
    """Plays broadcast ``talk`` events on the local machine's speakers.

    Thread-safe; :meth:`play` never blocks — events are queued to a
    single daemon worker thread that plays them strictly one at a
    time (the webview's talk-queue behaviour).  Playback of the same
    ``talkId`` more than once is suppressed with a bounded set (the
    webview's ``spokenTalkIds`` behaviour).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spoken_ids: dict[str, None] = {}
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None

    def play(self, event: dict[str, Any]) -> None:
        """Queue one ``talk`` *event* for playback (non-blocking).

        Blank events (no clip and no text) and already-spoken
        ``talkId`` values are dropped, matching the webview client.

        Args:
            event: The broadcast ``talk`` event dictionary.
        """
        text = str(event.get("text") or "").strip()
        if not text and not event.get("audioB64"):
            return
        talk_id = event.get("talkId")
        with self._lock:
            if talk_id:
                if talk_id in self._spoken_ids:
                    return
                self._spoken_ids[talk_id] = None
                if len(self._spoken_ids) > _MAX_TALK_IDS:
                    # Bounded memory: drop the oldest half (dicts
                    # iterate in insertion order).
                    for stale in list(self._spoken_ids)[: _MAX_TALK_IDS // 2]:
                        del self._spoken_ids[stale]
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drain, name="sorcar-talk-player", daemon=True
                )
                self._worker.start()
        self._queue.put(dict(event))

    def _drain(self) -> None:
        """Worker loop: play queued events one at a time, forever."""
        while True:
            event = self._queue.get()
            try:
                self._play_one(event)
            except Exception:
                # Playback is best-effort; a broken clip/player must
                # never take down the worker (later talks still play).
                logger.exception("talk playback failed")

    def _play_one(self, event: dict[str, Any]) -> None:
        """Play one event: synthesized clip first, then TTS fallback."""
        if self._play_clip(event):
            return
        text = str(event.get("text") or "").strip()
        if not text:
            return
        say = say_command()
        if say is None:
            return
        _run_playback([*say, text])

    def _play_clip(self, event: dict[str, Any]) -> bool:
        """Play the event's ``audioB64`` clip; ``True`` when it played."""
        audio_b64 = event.get("audioB64")
        if not audio_b64:
            return False
        try:
            audio = base64.b64decode(audio_b64, validate=True)
        except (ValueError, TypeError):
            return False
        if not audio:
            return False
        player = player_command()
        if player is None:
            return False
        suffix = ".mp3" if "mpeg" in str(event.get("audioMime") or "") else ".audio"
        fd, path = tempfile.mkstemp(prefix="sorcar-talk-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            return _run_playback([*player, path])
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


_shared_lock = threading.Lock()
_shared_player: TalkPlayer | None = None


def shared_player() -> TalkPlayer:
    """Return the process-wide :class:`TalkPlayer` singleton.

    One player per process mirrors the webview's one-queue-per-device
    semantics: talks from every printer/agent in this CLI process are
    deduped and serialised together, so parallel sub-agents can never
    speak over each other on the same speakers.

    Returns:
        The shared :class:`TalkPlayer`.
    """
    global _shared_player
    with _shared_lock:
        if _shared_player is None:
            _shared_player = TalkPlayer()
        return _shared_player


def reset_shared_player_for_tests() -> None:
    """Discard the shared player (test isolation for talkId dedupe)."""
    global _shared_player
    with _shared_lock:
        _shared_player = None
