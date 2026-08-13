# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Daemon-side lifecycle control for the wake-word listener.

Runs :mod:`kiss.server.voice_wake` as a child process of the
``kiss-web`` daemon on behalf of a connected client, so the VS Code
extension host can start/stop the listener and receive its events
through the same Unix-domain socket it already uses for every other
command (the ``voiceWakeStart`` / ``voiceWakeStop`` commands of
:data:`kiss.server.sorcar.API`) instead of spawning the listener
process itself.

One listener is kept per client connection (keyed by the connection's
``conn_id``): the microphone is a per-machine resource, but tying the
process to the requesting connection means a crashed or disconnected
client can never leak a forever-running mic listener — the transport's
disconnect cleanup calls :meth:`VoiceWakeController.stop`.

The child's newline-delimited stdout protocol (``READY``, ``WAKE``,
``TRANSCRIBING``, ``NO_SPEECH``, ``SPEECH {json}`` — see
``voice_wake.emit``) is parsed here and forwarded to the client as
``voiceWakeEvent`` messages; listener start/exit is reported as
``voiceWakeState`` messages mirroring the callbacks of the extension
host's in-process ``VoiceWakeService``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_STDERR_TAIL_CHARS = 2000
"""How much trailing stderr to keep for exit diagnostics."""

_TERM_GRACE_SECONDS = 5.0
"""Seconds to wait after SIGTERM before escalating to SIGKILL."""

SendCallback = Callable[[dict[str, Any]], Awaitable[None]]
"""Async callback delivering one event dict to the owning client."""


def parse_protocol_line(line: str) -> dict[str, Any] | None:
    """Translate one listener stdout line into a client event dict.

    Mirrors the line parsing of the extension host's
    ``voiceWake.ts`` so both transports expose identical semantics.

    Args:
        line: One stripped stdout line of ``kiss.server.voice_wake``.

    Returns:
        A ``voiceWakeEvent`` dict for a recognized protocol line, or
        ``None`` for an unrecognized (diagnostic) line.
    """
    if line == "READY":
        return {"type": "voiceWakeEvent", "event": "ready"}
    if line == "WAKE":
        return {"type": "voiceWakeEvent", "event": "wake"}
    if line == "TRANSCRIBING":
        return {"type": "voiceWakeEvent", "event": "transcribing"}
    if line == "NO_SPEECH":
        return {"type": "voiceWakeEvent", "event": "no_speech"}
    if line.startswith("SPEECH "):
        text = ""
        speaker: int | None = None
        language: str | None = None
        try:
            payload = json.loads(line[len("SPEECH "):])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict) and isinstance(
            payload.get("text"), str
        ):
            text = payload["text"]
            spk = payload.get("speaker")
            if isinstance(spk, int) and not isinstance(spk, bool) and spk >= 1:
                speaker = spk
            lang = payload.get("language")
            if isinstance(lang, str) and lang:
                language = lang
        return {
            "type": "voiceWakeEvent",
            "event": "speech",
            "text": text,
            "speaker": speaker,
            "language": language,
        }
    return None


class _Listener:
    """One running wake-word listener child process."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc
        self.stderr_tail = ""
        self.stopped = False


class VoiceWakeController:
    """Per-connection lifecycle manager for wake-word listeners.

    All methods must be called on the daemon's event loop.  The
    controller owns at most one listener process per ``conn_id`` and
    guarantees the process is reaped on :meth:`stop`,
    :meth:`stop_all`, or self-exit.
    """

    def __init__(self, listener_args: list[str] | None = None) -> None:
        """Create a controller with no running listeners.

        Args:
            listener_args: Command line spawning one listener process
                (before the optional ``--sensitivity`` argument).
                Defaults to running :mod:`kiss.server.voice_wake` with
                the daemon's interpreter; tests substitute a
                protocol-speaking stand-in.
        """
        self._listener_args = listener_args or [
            sys.executable, "-m", "kiss.server.voice_wake",
        ]
        self._listeners: dict[str, _Listener] = {}

    def running(self, conn_id: str) -> bool:
        """Return whether *conn_id* currently owns a live listener.

        Args:
            conn_id: The owning connection's id.

        Returns:
            ``True`` when a listener process is running for the
            connection.
        """
        return conn_id in self._listeners

    async def start(
        self,
        conn_id: str,
        sensitivity: int | None,
        send: SendCallback,
    ) -> None:
        """Start a wake-word listener for *conn_id*.

        A second start while the connection's listener is already
        running only re-reports ``listening: true`` (mirroring the
        extension host's ``VoiceWakeService.start``); to change the
        sensitivity the client stops and restarts the listener.

        The listener's protocol events and its final state are
        delivered through *send* as they happen; this coroutine
        returns as soon as the process is spawned (or fails to
        spawn).

        Args:
            conn_id: The owning connection's id.
            sensitivity: Optional wake-word sensitivity 0..100;
                clamped here so a junk client value cannot make the
                child's argparse exit.
            send: Async callback delivering event dicts to the owning
                client (``voiceWakeEvent`` / ``voiceWakeState``).
        """
        if conn_id in self._listeners:
            await self._safe_send(
                send, {"type": "voiceWakeState", "listening": True},
            )
            return
        args = list(self._listener_args)
        if isinstance(sensitivity, (int, float)) and not isinstance(
            sensitivity, bool
        ):
            clamped = min(100, max(0, round(sensitivity)))
            args += ["--sensitivity", str(clamped)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as err:
            await self._safe_send(send, {
                "type": "voiceWakeState",
                "listening": False,
                "error": f"voice listener failed to start: {err}",
            })
            return
        listener = _Listener(proc)
        self._listeners[conn_id] = listener
        asyncio.ensure_future(self._pump_stderr(listener))
        asyncio.ensure_future(self._pump_stdout(conn_id, listener, send))

    async def stop(self, conn_id: str) -> None:
        """Stop and reap *conn_id*'s listener, if any.

        Safe to call when no listener is running (e.g. from the
        transport's unconditional disconnect cleanup).

        Args:
            conn_id: The owning connection's id.
        """
        listener = self._listeners.pop(conn_id, None)
        if listener is None:
            return
        listener.stopped = True
        await self._terminate(listener.proc)

    async def stop_all(self) -> None:
        """Stop every running listener (daemon shutdown)."""
        for conn_id in list(self._listeners):
            await self.stop(conn_id)

    async def _pump_stdout(
        self, conn_id: str, listener: _Listener, send: SendCallback,
    ) -> None:
        """Forward the child's stdout protocol lines until it exits."""
        proc = listener.proc
        assert proc.stdout is not None
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                event = parse_protocol_line(
                    raw.decode("utf-8", errors="replace").strip()
                )
                if event is None:
                    continue
                if event.get("event") == "ready":
                    await self._safe_send(
                        send, {"type": "voiceWakeState", "listening": True},
                    )
                await self._safe_send(send, event)
        except Exception:
            logger.debug("voice-wake stdout pump failed", exc_info=True)
        returncode = await proc.wait()
        # A stop() (or a stop_all/disconnect) already reported nothing;
        # its owner is gone or asked for the stop, so only a
        # self-exited listener reports its final state.
        if self._listeners.get(conn_id) is listener:
            del self._listeners[conn_id]
        if listener.stopped:
            return
        error: str | None = None
        if returncode != 0:
            detail = listener.stderr_tail.strip().split("\n")[-1].strip()
            error = f"voice listener exited (code {returncode})" + (
                f": {detail}" if detail else ""
            )
        await self._safe_send(send, {
            "type": "voiceWakeState",
            "listening": False,
            **({"error": error} if error else {}),
        })

    async def _pump_stderr(self, listener: _Listener) -> None:
        """Keep the tail of the child's stderr for exit diagnostics."""
        stderr = listener.proc.stderr
        assert stderr is not None
        try:
            while True:
                chunk = await stderr.read(4096)
                if not chunk:
                    break
                tail = listener.stderr_tail + chunk.decode(
                    "utf-8", errors="replace"
                )
                listener.stderr_tail = tail[-_STDERR_TAIL_CHARS:]
        except Exception:
            logger.debug("voice-wake stderr pump failed", exc_info=True)

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """SIGTERM the child's process group, escalating to SIGKILL."""
        if proc.returncode is not None:
            return
        pid = proc.pid
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except ProcessLookupError:
                return
        try:
            await asyncio.wait_for(proc.wait(), _TERM_GRACE_SECONDS)
        except TimeoutError:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    return
            await proc.wait()

    @staticmethod
    async def _safe_send(
        send: SendCallback, event: dict[str, Any],
    ) -> None:
        """Deliver *event*, swallowing a dead client connection."""
        try:
            await send(event)
        except Exception:
            logger.debug("voice-wake event delivery failed", exc_info=True)
