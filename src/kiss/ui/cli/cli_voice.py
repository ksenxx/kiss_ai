# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Wake-word voice chat for the sorcar CLI REPL (``/voice``).

Typing ``/voice`` at the idle prompt spawns the wake-word listener
process (``python -m kiss.server.voice_wake``) and turns each
recognised utterance into a submitted REPL line, exactly as if the user
had typed it and pressed Enter.  In the anchored REPL voice runs in the
*background* (:class:`VoicePump`): the keyboard stays fully usable —
typing, editing and Enter-to-submit keep working — while the input
panel shows a red ``Listening ...`` indicator at the beginning of its
header (the top border) for both the idle prompt and the mid-task
steering box; the indicator starts *blinking* only once the sorcar wake
word is detected (``WAKE``) and switches to a blinking
``Transcribing ...`` while the utterance is transcribed.  Voice chat is
continuous — speech spoken while a task runs queues as a steering
follow-up — until the user toggles it off with ``/voice`` (typed or
spoken) or exits the REPL.  The plain fallback REPL keeps the modal
per-utterance capture instead, printing the indicator inline and
cancelling on Esc, Ctrl+C, Ctrl+D or Enter.

The listener's stdout speaks a line protocol: ``READY``, ``WAKE``,
``TRANSCRIBING``, ``SPEECH <json {"text","speaker","language"}>`` and
``NO_SPEECH``.  Recognised speech is decorated with the same
``Speaker #N says [in the language <lang>] that: ...`` prefix as the
chat webview via the shared :func:`speaker_prefixed_text` helper
(behavioural parity with ``media/voice.js`` ``insertSpeech``); blank
speech is never submitted.  The listener command is overridable through the
``KISS_SORCAR_VOICE_CMD`` environment variable (parsed with
:func:`shlex.split`) so tests can substitute a scripted child process.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from kiss.server.voice_wake import speaker_prefixed_text
from kiss.ui.cli.cli_panel import BLINK, CYAN, RED, RESET, YELLOW

if TYPE_CHECKING:
    from kiss.ui.cli.cli_steering import _InputBox

logger = logging.getLogger(__name__)

# Indicator texts shown (red; blinking once the wake word fires) while
# voice capture runs.
LISTENING_TEXT = "Listening ..."
TRANSCRIBING_TEXT = "Transcribing ..."
# Raw bytes that cancel PLAIN (fallback) voice capture: Esc, Ctrl+C,
# Ctrl+D, Enter (CR or LF).  Any of them anywhere in an input chunk
# exits voice mode.  The anchored REPL keeps the keyboard fully usable
# instead — ``/voice`` toggles voice mode off there.
_CANCEL_BYTES = b"\x1b\x03\x04\r\n"
# Environment variable overriding the listener command (shlex-parsed).
_CMD_ENV = "KISS_SORCAR_VOICE_CMD"
_KILL_SIGNAL = int(getattr(signal, "SIGKILL", signal.SIGTERM))


def listener_command() -> list[str]:
    """Return the argv used to spawn the wake-word listener process.

    Honours the ``KISS_SORCAR_VOICE_CMD`` environment variable (parsed
    with :func:`shlex.split`, so quoted arguments survive) and falls
    back to running :mod:`kiss.server.voice_wake` with the
    current interpreter.

    Returns:
        The listener argv list.
    """
    override = os.environ.get(_CMD_ENV, "").strip()
    if override:
        return shlex.split(override)
    return [sys.executable, "-m", "kiss.server.voice_wake"]


# How long a dead-listener final flush waits for the daemon reader
# thread to queue the child's last SPEECH line before giving up.  One
# shared constant for every flush site (the pump thread and the plain
# capture loop) so the two paths handle the same race identically.
_FINAL_FLUSH_TIMEOUT = 0.2

# User-visible message printed when the wake-word listener child dies
# unexpectedly — shared by the modal capture loop and the anchored
# pump's ``on_dead`` so the two voice modes can never diverge.
_LISTENER_DIED_MSG = (
    "✗ voice: the wake-word listener exited "
    "unexpectedly — leaving voice mode."
)


def _spawn_listener() -> VoiceListener | None:
    """Spawn the wake-word listener child, printing an error on failure.

    Shared by :func:`start_voice` (modal) and
    :func:`start_voice_anchored` (background) so both entry points
    handle a missing/broken listener command identically: the error is
    printed and ``None`` is returned so the REPL stays usable.

    Returns:
        The started :class:`VoiceListener`, or ``None`` when the
        listener process could not be spawned.
    """
    try:
        listener = VoiceListener()
        listener.start()
    except (OSError, ValueError) as exc:
        print(
            f"{YELLOW}✗ voice: could not start the wake-word listener: "
            f"{exc}{RESET}"
        )
        return None
    return listener


class VoiceListener:
    """Child wake-word listener process with a non-blocking line queue.

    A daemon reader thread pumps the child's stdout protocol lines into
    an internal queue so the REPL can poll them (:meth:`poll_line`)
    while simultaneously watching stdin for cancel keys.

    Attributes:
        proc: The spawned :class:`subprocess.Popen`, or ``None``
            before :meth:`start`.
    """

    def __init__(self, cmd: list[str] | None = None) -> None:
        self._cmd = list(cmd) if cmd is not None else listener_command()
        self.proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the listener child and start the stdout reader thread.

        Raises:
            OSError: When the listener command cannot be executed
                (missing binary, permission error, ...).
        """
        self.proc = subprocess.Popen(  # noqa: S603 - operator-supplied cmd
            self._cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            # Put POSIX listeners in their own process group so stop()
            # can kill wrapper + microphone child together (matching the
            # VS Code wake-word supervisor's leak fix for uv wrappers).
            start_new_session=os.name == "posix",
        )
        self._thread = threading.Thread(
            target=self._pump_stdout, daemon=True, name="sorcar-voice-reader",
        )
        try:
            self._thread.start()
        except BaseException:
            # If thread creation fails after Popen succeeded, do not
            # leave a microphone listener running with nobody draining it.
            self.stop()
            raise

    def _pump_stdout(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                self._lines.put(raw.strip())
        except (OSError, ValueError):  # pragma: no cover - stream torn down
            logger.debug("voice listener stdout closed", exc_info=True)

    def poll_line(self) -> str | None:
        """Return the next buffered protocol line, or ``None`` if none.

        Returns:
            The next stripped stdout line from the listener, or
            ``None`` when no line is currently queued.
        """
        try:
            return self._lines.get_nowait()
        except queue.Empty:
            return None

    def final_flush(self, timeout: float = _FINAL_FLUSH_TIMEOUT) -> str | None:
        """Return the dead child's last buffered line, if any.

        The child can exit a few milliseconds before the daemon reader
        thread queues its final ``SPEECH`` line; this gives the reader
        one short chance to flush before the caller declares failure.

        Args:
            timeout: Maximum seconds to wait for the reader thread.

        Returns:
            The next buffered protocol line, or ``None`` when none
            arrives within *timeout*.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self.poll_line()

    def alive(self) -> bool:
        """Return ``True`` while the listener child is still running."""
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Terminate the listener child/process group; idempotent.

        Every exit path of voice mode — cancel, REPL exit, errors —
        funnels through here so the child can never be leaked.  On POSIX
        the listener is started in a fresh process group and the whole
        group is signalled; this prevents ``uv run``-style wrappers from
        orphaning the real microphone-owning Python child.
        """
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            self._signal_listener(int(signal.SIGTERM))
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - slow child
                self._signal_listener(_KILL_SIGNAL, force=True)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.warning("voice listener did not die on kill")
        else:
            # The direct wrapper may already have exited while leaving
            # its microphone-owning child in the process group.  Signal
            # the group once anyway; ProcessLookupError is harmless when
            # the group is already empty.
            self._signal_listener(int(signal.SIGTERM))
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:  # pragma: no cover - defensive
                pass
        if (
            self._thread is not None
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=0.5)

    def _signal_listener(self, sig: int, *, force: bool = False) -> None:
        """Send *sig* to the listener process group (POSIX) or process."""
        proc = self.proc
        if proc is None:
            return
        if os.name == "posix" and proc.pid is not None:
            try:
                os.killpg(proc.pid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                logger.debug(
                    "voice listener process-group signal failed",
                    exc_info=True,
                )
        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
        except OSError:  # pragma: no cover - already gone
            pass


class VoicePump(threading.Thread):
    """Background thread turning listener protocol lines into UI events.

    Runs for the whole lifetime of an anchored voice session so the
    keyboard stays fully usable while voice is on: the pump keeps the
    input-panel header indicator up to date (steady red
    ``Listening ...`` before the wake word, blinking after ``WAKE``,
    blinking ``Transcribing ...`` during transcription) and hands each
    recognised utterance to *on_speech* — which injects it into the
    anchored box so it submits exactly like a typed line — then goes
    back to listening (continuous voice chat).

    A listener child that exits unexpectedly triggers *on_dead* once
    (never when :meth:`stop` already asked the pump to end).
    """

    def __init__(
        self,
        listener: VoiceListener,
        show: Callable[[str, bool], None],
        on_speech: Callable[[str], None],
        on_dead: Callable[[], None],
    ) -> None:
        super().__init__(daemon=True, name="sorcar-voice-pump")
        self._listener = listener
        self._show = show
        self._on_speech = on_speech
        self._on_dead = on_dead
        self._stop_evt = threading.Event()

    def run(self) -> None:
        """Pump protocol lines until stopped or the listener dies."""
        show_listening, show_wake, show_transcribing = _show_callbacks(
            self._show,
        )
        self._show(LISTENING_TEXT, False)
        while not self._stop_evt.is_set():
            line = self._listener.poll_line()
            if line is None:
                if self._listener.alive():
                    self._stop_evt.wait(0.05)
                    continue
                line = self._listener.final_flush()
                if line is None:
                    if not self._stop_evt.is_set():
                        self._on_dead()
                    return
            result = _handle_protocol_line(
                line, show_listening, show_wake, show_transcribing,
            )
            if result is not _NO_LINE_RESULT:
                self._on_speech(str(result))
                # The listener resumes waiting for the next wake word;
                # reflect that with the steady pre-wake indicator.
                self._show(LISTENING_TEXT, False)

    def stop(self) -> None:
        """Ask the pump to end and wait briefly for it; idempotent."""
        self._stop_evt.set()
        if self is not threading.current_thread():
            self.join(timeout=1.0)


class VoiceSession:
    """Continuous voice-chat state owned by the REPL loop.

    Two flavours exist: the anchored REPL runs a *background* session
    (:func:`start_voice_anchored`) whose :class:`VoicePump` thread
    updates the input-panel header and injects recognised speech into
    the box while the keyboard stays fully usable; the plain fallback
    REPL runs a *modal* session (:func:`start_voice`) whose
    :meth:`read` blocks per utterance.

    Attributes:
        listener: The running :class:`VoiceListener` child.
        pump: The background :class:`VoicePump` thread, or ``None``
            for a modal session.
        active: ``False`` once the listener died unexpectedly — the
            REPL loop closes and drops the session on its next pass.
    """

    def __init__(
        self,
        listener: VoiceListener,
        reader: Callable[[VoiceListener], str | None] | None = None,
        pump: VoicePump | None = None,
    ) -> None:
        self.listener = listener
        self._reader = reader
        self.pump = pump
        self.active = True
        # Optional teardown hook run by :meth:`close` after the pump
        # stopped — the anchored session uses it to clear the header
        # indicator so ``Listening ...`` never outlives voice mode.
        self.on_close: Callable[[], None] | None = None

    @property
    def background(self) -> bool:
        """``True`` for a pump-driven session (keyboard stays usable)."""
        return self._reader is None

    def read(self) -> str | None:
        """Capture the next utterance through the configured modal reader.

        Only valid for modal (non-background) sessions.

        Returns:
            The recognised speech text, or ``None`` when the user
            cancelled or the listener failed (leave voice mode).
        """
        assert self._reader is not None
        return self._reader(self.listener)

    def close(self) -> None:
        """Stop the pump (if any) and terminate the listener child."""
        if self.pump is not None:
            self.pump.stop()
        self.listener.stop()
        if self.on_close is not None:
            self.on_close()


def start_voice(
    reader: Callable[[VoiceListener], str | None],
) -> VoiceSession | None:
    """Spawn the wake-word listener and enter MODAL voice mode.

    Used by the plain (non-anchored) fallback REPL; the anchored REPL
    uses :func:`start_voice_anchored` instead so the keyboard stays
    usable.

    Args:
        reader: Per-utterance capture function
            (:func:`read_voice_line_plain`).

    Returns:
        A live :class:`VoiceSession`, or ``None`` when the listener
        process could not be spawned (an error is printed and the
        REPL stays usable).
    """
    listener = _spawn_listener()
    if listener is None:
        return None
    print(
        f"{YELLOW}🎤 Voice mode: say the wake word, then speak your task "
        f"(Esc, Ctrl+C, Ctrl+D or Enter to stop).{RESET}"
    )
    return VoiceSession(listener, reader)


_NO_LINE_RESULT = object()


def _handle_protocol_line(
    line: str,
    show_listening: Callable[[], None],
    show_wake: Callable[[], None],
    show_transcribing: Callable[[], None],
) -> str | object:
    """Handle one listener protocol line.

    ``WAKE`` (the sorcar wake word was detected) calls *show_wake* so
    the indicator starts blinking; ``TRANSCRIBING`` calls
    *show_transcribing*; ``NO_SPEECH`` and unusable ``SPEECH`` payloads
    call *show_listening* (steady, pre-wake display).

    Returns recognised speech text, or ``_NO_LINE_RESULT`` when the
    capture should keep listening.
    """
    if line == "WAKE":
        show_wake()
    elif line == "TRANSCRIBING":
        show_transcribing()
    elif line == "NO_SPEECH":
        show_listening()
    elif line.startswith("SPEECH "):
        try:
            payload = json.loads(line[len("SPEECH "):])
        except ValueError:
            logger.debug("malformed SPEECH line: %r", line)
            show_listening()
            return _NO_LINE_RESULT
        if isinstance(payload, str):
            # Legacy payload shape accepted by the VS Code wake-word
            # service: the JSON value itself is the translated text
            # (no speaker information, so no prefix).
            text = speaker_prefixed_text(payload, None, None)
            if text:
                return text
            show_listening()
            return _NO_LINE_RESULT
        if isinstance(payload, dict) and isinstance(payload.get("text"), str):
            # Shared with the chat webview (media/voice.js
            # insertSpeech): the submitted line carries the
            # "Speaker #N" prefix when the listener identified the
            # speaker, and blank speech is never submitted.
            text = speaker_prefixed_text(
                payload.get("text"),
                payload.get("speaker"),
                payload.get("language"),
            )
            if text:
                return text
            show_listening()
            return _NO_LINE_RESULT
        logger.debug("malformed SPEECH payload: %r", line)
        # A SPEECH line is terminal for a wake round.  Treat unusable
        # payloads like NO_SPEECH so the indicator cannot stay stuck in
        # Transcribing ... forever while the listener has resumed.
        show_listening()
    # READY and unknown lines are ignored.
    return _NO_LINE_RESULT


def _show_callbacks(
    show: Callable[[str, bool], None],
) -> tuple[Callable[[], None], Callable[[], None], Callable[[], None]]:
    """Build the indicator callbacks :func:`_handle_protocol_line` takes.

    Args:
        show: Renderer taking ``(text, blink)`` — the indicator text
            and whether it should blink.

    Returns:
        The ``(show_listening, show_wake, show_transcribing)`` triple:
        steady ``Listening ...``, blinking ``Listening ...`` (wake word
        detected) and blinking ``Transcribing ...``.
    """
    def show_listening() -> None:
        show(LISTENING_TEXT, False)

    def show_wake() -> None:
        show(LISTENING_TEXT, True)

    def show_transcribing() -> None:
        show(TRANSCRIBING_TEXT, True)

    return show_listening, show_wake, show_transcribing


def _capture_utterance(
    listener: VoiceListener,
    show_listening: Callable[[], None],
    show_wake: Callable[[], None],
    show_transcribing: Callable[[], None],
) -> str | None:
    """Poll the listener and stdin until speech is captured or cancelled.

    Watches stdin (raw bytes) for cancel keys and the listener's line
    queue for protocol events: ``WAKE`` (the sorcar wake word was
    detected) flips the indicator to *show_wake* — the display starts
    blinking — ``TRANSCRIBING`` flips it to *show_transcribing*,
    ``NO_SPEECH`` flips it back to the steady *show_listening*, and a
    valid ``SPEECH`` returns its text.  Malformed ``SPEECH`` payloads
    also flip back to listening so the display never stays stuck in
    transcribing state.  A dead listener (unexpected exit) prints an
    error and returns ``None``.

    Args:
        listener: The running listener child.
        show_listening: Draws the steady ``Listening ...`` indicator.
        show_wake: Draws the blinking ``Listening ...`` indicator
            (wake word detected, speech capture in progress).
        show_transcribing: Draws the blinking ``Transcribing ...``
            indicator.

    Returns:
        The recognised text, or ``None`` on cancel / listener failure.
    """
    try:
        fd: int | None = sys.stdin.fileno()
    except (OSError, ValueError):
        fd = None
    show_listening()
    while True:
        while True:
            line = listener.poll_line()
            if line is None:
                break
            result = _handle_protocol_line(
                line, show_listening, show_wake, show_transcribing,
            )
            if result is not _NO_LINE_RESULT:
                return str(result)
        if not listener.alive():
            line = listener.final_flush()
            if line is not None:
                result = _handle_protocol_line(
                    line, show_listening, show_wake, show_transcribing,
                )
                if result is not _NO_LINE_RESULT:
                    return str(result)
                continue
            print(f"{YELLOW}{_LISTENER_DIED_MSG}{RESET}")
            return None
        if fd is None:
            time.sleep(0.05)
            continue
        try:
            ready, _, _ = select.select([fd], [], [], 0.05)
        except KeyboardInterrupt:
            return None
        except InterruptedError:  # pragma: no cover - transient
            continue
        except (OSError, ValueError):
            # Some fallback environments (notably non-POSIX consoles or
            # replaced stdin objects) cannot be selected.  Keep polling
            # the listener so speech still works; only raw-key cancel is
            # unavailable until a SIGINT arrives.
            fd = None
            continue
        if not ready:
            continue
        try:
            data = os.read(fd, 4096)
        except KeyboardInterrupt:
            return None
        except InterruptedError:  # pragma: no cover - transient
            continue
        except OSError:
            fd = None
            continue
        if not data or any(key in data for key in _CANCEL_BYTES):
            # Stdin EOF behaves like Ctrl+D; any cancel key exits voice
            # mode.  Other keys are ignored while listening.
            return None


def start_voice_anchored(box: _InputBox) -> VoiceSession | None:
    """Start background voice mode on the anchored input box.

    Spawns the wake-word listener and a :class:`VoicePump` thread, then
    returns immediately so the REPL keeps reading the keyboard — typing,
    editing, completion and Enter-to-submit all keep working while
    voice is on.  The pump shows the red ``Listening ...`` /
    ``Transcribing ...`` indicator at the beginning of the panel's
    header (blinking only once the sorcar wake word is detected) for
    both the idle prompt and the steering box, and injects each
    recognised utterance into *box* so it submits exactly like a typed
    line.  Voice mode stays on — across submitted tasks — until the
    session is closed (``/voice`` again or REPL exit).

    Args:
        box: The anchored ``_InputBox`` owned by the REPL.

    Returns:
        A live background :class:`VoiceSession`, or ``None`` when the
        listener process could not be spawned (an error is printed and
        the REPL stays usable).
    """
    listener = _spawn_listener()
    if listener is None:
        return None

    def show(text: str, blink: bool) -> None:
        with box.lock:
            box.overlay = text
            box.overlay_blink = blink
            box.redraw()

    session = VoiceSession(listener)

    def on_dead() -> None:
        print(f"{YELLOW}{_LISTENER_DIED_MSG}{RESET}")
        show("", False)
        session.active = False

    def on_close() -> None:
        # Discard any utterance the pump injected while voice mode was
        # shutting down — a stale spoken line must never submit as a
        # task after the user turned voice off — then clear the header
        # indicator.  Runs after the pump thread has been joined.
        box.drain_injected()
        show("", False)

    pump = VoicePump(listener, show, box.inject_line, on_dead)
    session.pump = pump
    session.on_close = on_close
    pump.start()
    print(
        f"{YELLOW}🎤 Voice mode: say the wake word, then speak your task "
        f"— you can keep typing too; /voice again to stop.{RESET}"
    )
    return session


def read_voice_line_plain(listener: VoiceListener) -> str | None:
    """Capture one utterance in the plain (non-anchored) fallback REPL.

    Prints the red ``Listening ...`` / ``Transcribing ...`` indicator
    inline instead of styling the anchored panel's header; the
    indicator blinks only once the sorcar wake word is detected.

    Args:
        listener: The running wake-word listener child.

    Returns:
        The recognised speech text, or ``None`` on cancel / listener
        failure (leave voice mode).
    """
    def show(text: str, blink: bool) -> None:
        style = (BLINK + RED) if blink else RED
        print(f"{style}{text}{RESET}", flush=True)

    text = _capture_utterance(listener, *_show_callbacks(show))
    if text is not None:
        print(f"{CYAN}🎤 {text}{RESET}")
    return text


__all__ = [
    "LISTENING_TEXT",
    "TRANSCRIBING_TEXT",
    "VoiceListener",
    "VoicePump",
    "VoiceSession",
    "listener_command",
    "read_voice_line_plain",
    "start_voice",
    "start_voice_anchored",
]
