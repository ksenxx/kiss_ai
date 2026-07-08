# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Wake-word voice chat for the sorcar CLI REPL (``/voice``).

Typing ``/voice`` at the idle prompt spawns the wake-word listener
process (``python -m kiss.agents.vscode.voice_wake``) and turns each
recognised utterance into a submitted REPL line, exactly as if the user
had typed it and pressed Enter.  While waiting for speech the anchored
input panel shows a red ``Listening ...`` indicator at the beginning of
its header (the top border); the indicator starts *blinking* only once
the sorcar wake word is detected (``WAKE``) and switches to a blinking
``Transcribing ...`` while the utterance is transcribed.  The plain
fallback REPL prints the same indicator inline.  Voice chat is
continuous — after a submitted task completes the REPL resumes
listening — until the user cancels with Esc, Ctrl+C, Ctrl+D or Enter.

The listener's stdout speaks a line protocol: ``READY``, ``WAKE``,
``TRANSCRIBING``, ``SPEECH <json {"text","speaker","language"}>`` and
``NO_SPEECH``.  The listener command is overridable through the
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

from kiss.agents.sorcar.cli_panel import BLINK, CYAN, RED, RESET, YELLOW

if TYPE_CHECKING:
    from kiss.agents.sorcar.cli_steering import _InputBox

logger = logging.getLogger(__name__)

# Indicator texts shown (red; blinking once the wake word fires) while
# voice capture runs.
LISTENING_TEXT = "Listening ..."
TRANSCRIBING_TEXT = "Transcribing ..."
# Panel title shown while the anchored box is in voice mode.
VOICE_TITLE = (
    " voice · say the wake word, then speak your task · "
    "Esc/Ctrl+C/Ctrl+D/Enter to stop "
)
# Raw bytes that cancel voice capture: Esc, Ctrl+C, Ctrl+D, Enter (CR
# or LF).  Any of them anywhere in an input chunk exits voice mode.
_CANCEL_BYTES = b"\x1b\x03\x04\r\n"
# Environment variable overriding the listener command (shlex-parsed).
_CMD_ENV = "KISS_SORCAR_VOICE_CMD"
_KILL_SIGNAL = int(getattr(signal, "SIGKILL", signal.SIGTERM))


def listener_command() -> list[str]:
    """Return the argv used to spawn the wake-word listener process.

    Honours the ``KISS_SORCAR_VOICE_CMD`` environment variable (parsed
    with :func:`shlex.split`, so quoted arguments survive) and falls
    back to running :mod:`kiss.agents.vscode.voice_wake` with the
    current interpreter.

    Returns:
        The listener argv list.
    """
    override = os.environ.get(_CMD_ENV, "").strip()
    if override:
        return shlex.split(override)
    return [sys.executable, "-m", "kiss.agents.vscode.voice_wake"]


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


class VoiceSession:
    """Continuous voice-chat state owned by the REPL loop.

    Attributes:
        listener: The running :class:`VoiceListener` child.
    """

    def __init__(
        self,
        listener: VoiceListener,
        reader: Callable[[VoiceListener], str | None],
    ) -> None:
        self.listener = listener
        self._reader = reader

    def read(self) -> str | None:
        """Capture the next utterance through the configured reader.

        Returns:
            The recognised speech text, or ``None`` when the user
            cancelled or the listener failed (leave voice mode).
        """
        return self._reader(self.listener)

    def close(self) -> None:
        """Terminate the listener child process."""
        self.listener.stop()


def start_voice(
    reader: Callable[[VoiceListener], str | None],
) -> VoiceSession | None:
    """Spawn the wake-word listener and enter voice mode.

    Args:
        reader: Per-utterance capture function
            (:func:`read_voice_line_anchored` bound to the box, or
            :func:`read_voice_line_plain`).

    Returns:
        A live :class:`VoiceSession`, or ``None`` when the listener
        process could not be spawned (an error is printed and the
        REPL stays usable).
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
            # service: the JSON value itself is the translated text.
            return payload
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str):
                return text
        logger.debug("malformed SPEECH payload: %r", line)
        # A SPEECH line is terminal for a wake round.  Treat unusable
        # payloads like NO_SPEECH so the indicator cannot stay stuck in
        # Transcribing ... forever while the listener has resumed.
        show_listening()
    # READY and unknown lines are ignored.
    return _NO_LINE_RESULT


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
            # The process can exit a few milliseconds before the daemon
            # reader thread has queued its final SPEECH line.  Give the
            # pump one short chance to flush before declaring failure.
            if listener._thread is not None:
                listener._thread.join(timeout=0.05)
            line = listener.poll_line()
            if line is not None:
                result = _handle_protocol_line(
                    line, show_listening, show_wake, show_transcribing,
                )
                if result is not _NO_LINE_RESULT:
                    return str(result)
                continue
            print(
                f"{YELLOW}✗ voice: the wake-word listener exited "
                f"unexpectedly — leaving voice mode.{RESET}"
            )
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


def read_voice_line_anchored(
    box: _InputBox, listener: VoiceListener,
) -> str | None:
    """Capture one utterance showing the indicator inside the panel.

    Flips the anchored input box into voice mode — title swapped to
    :data:`VOICE_TITLE` and a red ``Listening ...`` /
    ``Transcribing ...`` indicator shown at the beginning of the
    panel's header (top border), blinking only once the sorcar wake
    word is detected — until speech arrives or the user cancels, then
    restores the box and echoes the recognised text above it like a
    typed line.

    Args:
        box: The anchored ``_InputBox`` owned by the REPL.
        listener: The running wake-word listener child.

    Returns:
        The recognised speech text, or ``None`` on cancel / listener
        failure (leave voice mode).
    """
    def show(text: str, blink: bool) -> None:
        with box.lock:
            box.overlay = text
            box.overlay_blink = blink
            box.redraw()

    with box.lock:
        prev_title = box.title
        box._reset_completion_state()
        box.title = VOICE_TITLE
    try:
        text = _capture_utterance(
            listener,
            lambda: show(LISTENING_TEXT, False),
            lambda: show(LISTENING_TEXT, True),
            lambda: show(TRANSCRIBING_TEXT, True),
        )
    finally:
        with box.lock:
            box.title = prev_title
            box.overlay = ""
            box.overlay_blink = False
            box.redraw()
    if text is not None:
        # Echo the utterance above the box the way typed lines echo.
        print(f"{CYAN}🎤 {text}{RESET}")
    return text


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

    text = _capture_utterance(
        listener,
        lambda: show(LISTENING_TEXT, False),
        lambda: show(LISTENING_TEXT, True),
        lambda: show(TRANSCRIBING_TEXT, True),
    )
    if text is not None:
        print(f"{CYAN}🎤 {text}{RESET}")
    return text


__all__ = [
    "LISTENING_TEXT",
    "TRANSCRIBING_TEXT",
    "VOICE_TITLE",
    "VoiceListener",
    "VoiceSession",
    "listener_command",
    "read_voice_line_anchored",
    "read_voice_line_plain",
    "start_voice",
]
