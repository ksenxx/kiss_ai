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
import functools
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

_EXIT_REPORT_JOIN_SECONDS = 5.0
"""Bound on waiting for a dead listener's in-flight exit report."""

_KILL_REAP_SECONDS = 5.0
"""Bound on waiting for the child to be reaped after SIGKILL.

A SIGKILLed listener normally reaps at once, but a detached
descendant that inherited the child's stdout/stderr keeps the pipe
transports open, and asyncio's ``Process.wait()`` then pends until the
escaped process exits — indefinitely, while the caller holds the
connection's lifecycle lock.  Past this bound the wait is abandoned
with a log line: SIGKILL was already delivered to the whole process
group, so nothing more can be done for the child itself (gpt-5.6-sol
round-4 review, finding 4).
"""

_RETIREMENT_GATE_SECONDS = 5.0
"""Bound on gating a successor's deliveries on retired pumps.

A retired predecessor pump normally completes within the stop/restart
joins; this deadline only matters when its in-flight send resists
cancellation indefinitely.  The successor then proceeds — its liveness
must not depend on arbitrary callback code ever completing — and the
generation tag (see ``VoiceWakeController.accepts``) keeps the retired
pump's stale report discardable at the delivery boundary (gpt-5.6-sol
round-3 review, finding 3).
"""

SendCallback = Callable[[dict[str, Any]], Awaitable[None]]
"""Async callback delivering one event dict to the owning client.

Every event dict carries a ``voiceGen`` tag: the listener-generation
token of its sender (see :meth:`VoiceWakeController.accepts`).  A
delivery boundary that can suspend mid-send (a slow client write)
must either deliver accepted payloads in send-start order (the
daemon's per-endpoint FIFO send lock does) or re-check
``accepts(conn_id, event["voiceGen"])`` when it resumes and drop a
retired generation's report — otherwise a wedged stale
``listening: false`` could land after a successor's
``listening: true``.  The production callback
(``web_server._handle_voice_wake_start``) checks the tag and strips
it before the payload reaches the wire.
"""


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

    def __init__(self, proc: asyncio.subprocess.Process, gen: int) -> None:
        self.proc = proc
        # The connection's listener-generation token at spawn time.
        # Every event this listener delivers is tagged with it, and
        # the delivery path drops the event once the connection has
        # moved on to a newer generation (a stop or a replacement
        # spawn) — see ``VoiceWakeController.accepts``.
        self.gen = gen
        self.stderr_tail = ""
        self.stopped = False
        self.pumps: list[asyncio.Task[None]] = []
        # Set (in a ``finally``) when the stderr pump has stopped
        # writing ``stderr_tail`` — at EOF, on error, or on
        # cancellation.  The stdout pump joins this before composing
        # the exit diagnostic; reading the tail right after
        # ``proc.wait()`` raced the sibling pump and dropped the
        # child's final stderr line on some schedules (gpt-5.6-sol
        # round-2 review, finding 4).
        self.stderr_done = asyncio.Event()


class VoiceWakeController:
    """Per-connection lifecycle manager for wake-word listeners.

    All methods must be called on the daemon's event loop.  The
    controller owns at most one listener process per ``conn_id`` and
    guarantees the process is reaped on :meth:`stop`,
    :meth:`stop_all`, or self-exit.  :meth:`start` and :meth:`stop`
    are additionally serialized per connection (a refcounted
    ``asyncio.Lock``): both span awaits while the listener slot is in
    transition, and two unserialized concurrent starts could each
    spawn a child, the second registration leaking the first.
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
        # Per-connection pump tasks that were still unfinished when
        # their listener was deregistered (a stop, or a timed-out
        # dead-listener restart).  Cancellation is cooperative: a send
        # callback may finish endpoint cleanup before honouring it, so
        # such a pump can still deliver its old ``listening: false``
        # report LATER.  A replacement listener therefore defers its
        # deliveries until every retired pump of its connection has
        # completed (:meth:`_await_retirement`) — bounded by
        # ``_RETIREMENT_GATE_SECONDS`` so one wedged retired send can
        # never suppress successors forever; ordering beyond the bound
        # is kept by the generation tag (:meth:`accepts`), which lets
        # the delivery boundary discard a retired generation's report
        # (gpt-5.6-sol round-2 finding 3, round-3 findings 2-3).
        # Registration is atomic with deregistration: ``stop`` and the
        # dead-listener path of ``start`` publish the retiring pumps
        # HERE before the listener leaves ``_listeners``, with no
        # await in between, so a concurrent ``start`` can never
        # observe both empty while an endpoint-touching pump is still
        # being joined.  Entries remove themselves on task completion,
        # so the sets stay bounded.
        self._retiring: dict[str, set[asyncio.Task[None]]] = {}
        # Per-connection listener-generation counter.  Bumped by every
        # lifecycle transition that retires the current listener's
        # right to report (a spawn, a stop): events are tagged with
        # their sender's generation and checked against this counter
        # at delivery time, so a retired pump's stale report becomes
        # unobservable instead of contradicting its successor.  An
        # entry is dropped once the connection has neither a listener
        # nor retiring pumps (every tagged event of a completed pump
        # has already been delivered), so the map stays bounded.
        self._generations: dict[str, int] = {}
        # Per-connection lifecycle serialization.  start() spans an
        # await (the subprocess spawn) while the listener slot is
        # still empty: two fully concurrent start() calls could both
        # pass the duplicate-start check and both spawn, the second
        # registration overwriting — and so leaking — the first child
        # (round-3 adjacent observation; fixed in round 4).  Every
        # start()/stop() body therefore runs under its connection's
        # lock.  Both bodies are bounded (all their internal waits
        # carry deadlines, and no client send ever runs under the
        # lock — an unbounded send callback blocking a stop() forever
        # was gpt-5.6-sol round-4 finding 2), so serialization cannot
        # become an unbounded stall.  ``_lifecycle_holds`` counts
        # holders AND waiters; an entry pair is dropped when the count
        # reaches zero, so the maps stay bounded across many
        # short-lived connections.
        self._lifecycle_locks: dict[str, asyncio.Lock] = {}
        self._lifecycle_holds: dict[str, int] = {}
        # Termination/reap tasks for children whose stop is in flight.
        # A stop() deregisters its listener and then awaits bounded
        # teardown; the teardown runs as its OWN task registered here
        # (and shield-awaited by the stop), so cancelling the stop —
        # production-reachable through the UDS handler drain — cannot
        # orphan a live child: the reap task retains ownership until
        # the child is signalled and (boundedly) reaped, and
        # :meth:`stop_all` joins every outstanding entry (gpt-5.6-sol
        # round-4 review, finding 3).  Entries remove themselves on
        # completion.
        self._reap_tasks: set[asyncio.Task[None]] = set()
        # Depth of in-progress :meth:`stop_all` calls.  While nonzero,
        # a start() that reaches its lifecycle body refuses to spawn:
        # stop_all() snapshots the connections with a listener OR any
        # lifecycle-lock activity, but a start() call that has not yet
        # taken a lifecycle reference is invisible to that snapshot and
        # could publish a live child after stop_all() returned
        # (gpt-5.6-sol round-4 review, finding 5).
        self._closing = 0

    async def _acquire_lifecycle(self, conn_id: str) -> None:
        """Acquire *conn_id*'s start/stop serialization lock.

        Reference-counted: the count covers the holder and every
        waiter, and :meth:`_drop_lifecycle_ref` frees the entry at
        zero, so the lock map stays bounded across many short-lived
        connections.  A waiter cancelled while queueing drops its
        reference without releasing anything.

        Args:
            conn_id: The owning connection's id.
        """
        lock = self._lifecycle_locks.get(conn_id)
        if lock is None:
            lock = asyncio.Lock()
            self._lifecycle_locks[conn_id] = lock
        self._lifecycle_holds[conn_id] = (
            self._lifecycle_holds.get(conn_id, 0) + 1
        )
        try:
            await lock.acquire()
        except BaseException:
            self._drop_lifecycle_ref(conn_id)
            raise

    def _release_lifecycle(self, conn_id: str) -> None:
        """Release *conn_id*'s lifecycle lock and drop its reference.

        Args:
            conn_id: The owning connection's id.
        """
        self._lifecycle_locks[conn_id].release()
        self._drop_lifecycle_ref(conn_id)

    def _drop_lifecycle_ref(self, conn_id: str) -> None:
        """Drop one lifecycle-lock reference; free the entry at zero.

        Args:
            conn_id: The owning connection's id.
        """
        remaining = self._lifecycle_holds.get(conn_id, 0) - 1
        if remaining > 0:
            self._lifecycle_holds[conn_id] = remaining
        else:
            self._lifecycle_holds.pop(conn_id, None)
            self._lifecycle_locks.pop(conn_id, None)

    def running(self, conn_id: str) -> bool:
        """Return whether *conn_id* currently owns a live listener.

        Args:
            conn_id: The owning connection's id.

        Returns:
            ``True`` when a listener process is running for the
            connection.
        """
        return conn_id in self._listeners

    def _bump_generation(self, conn_id: str) -> int:
        """Advance and return *conn_id*'s listener generation.

        Called at every lifecycle linearization point that retires the
        previous listener's right to report: a successful spawn (the
        new listener owns the fresh token) and a :meth:`stop`.

        Args:
            conn_id: The owning connection's id.

        Returns:
            The fresh (positive, strictly increasing) generation.
        """
        nxt = self._generations.get(conn_id, 0) + 1
        self._generations[conn_id] = nxt
        return nxt

    def accepts(self, conn_id: str, gen: int) -> bool:
        """Return whether *gen* is *conn_id*'s current generation.

        The delivery-boundary check for the ``voiceGen`` tag carried
        by every event this controller sends: a boundary that may
        suspend mid-send re-checks the tag when it resumes and drops a
        report whose generation was retired meanwhile, so a wedged
        stale ``listening: false`` can never land after a successor's
        ``listening: true`` (gpt-5.6-sol round-3 review, finding 3).
        The daemon's per-connection send wrapper
        (``web_server._handle_voice_wake_start``) calls this and
        strips the tag before the payload reaches the wire.

        Args:
            conn_id: The owning connection's id.
            gen: The ``voiceGen`` tag of the report to deliver.

        Returns:
            ``True`` when the report's generation is still current.
        """
        return gen == self._generations.get(conn_id, 0)

    def _maybe_drop_generation(self, conn_id: str) -> None:
        """Drop *conn_id*'s generation counter once nothing needs it.

        Safe only when the connection has neither a registered
        listener nor retiring pumps: a pump finishes only after its
        sends returned, so no in-flight tagged report can be
        mis-accepted when a later listener restarts the counter from
        zero.  Keeps the map bounded across many short-lived
        connections.

        Args:
            conn_id: The owning connection's id.
        """
        if conn_id not in self._listeners and not self._retiring.get(conn_id):
            self._generations.pop(conn_id, None)

    async def _deliver(
        self,
        conn_id: str,
        listener: _Listener,
        send: SendCallback,
        event: dict[str, Any],
    ) -> None:
        """Send *event* on *listener*'s behalf, unless it was retired.

        Tags the event with the listener's generation (``voiceGen``)
        and drops it outright when the connection has already moved on
        (:meth:`accepts` is false) — a retired pump that reaches its
        report only after cancellation therefore delivers nothing,
        and an in-flight report stays discardable at the delivery
        boundary via the tag.

        Args:
            conn_id: The owning connection's id.
            listener: The sending listener (its ``gen`` qualifies the
                report).
            send: The owning client's event callback.
            event: The event dict to deliver (not mutated).
        """
        if not self.accepts(conn_id, listener.gen):
            return
        await self._safe_send(send, {**event, "voiceGen": listener.gen})

    def _retire_pumps(
        self, conn_id: str, tasks: list[asyncio.Task[None]],
    ) -> None:
        """Track abandoned-but-unfinished pump tasks for *conn_id*.

        Called at the retirement linearization point of :meth:`stop`
        and of :meth:`start`'s dead-listener path — BEFORE the
        listener leaves ``_listeners`` and before any await, so a
        concurrent :meth:`start` that no longer sees the listener is
        guaranteed to see its unfinished pumps here (gpt-5.6-sol
        round-3 review, finding 2).  The retired pumps gate a
        successor's deliveries (:meth:`_await_retirement`); a
        completed task removes itself from the set, so the
        bookkeeping stays bounded.

        Args:
            conn_id: The owning connection's id.
            tasks: The retiring pump tasks to track (completed ones
                are skipped).
        """
        pending = [t for t in tasks if not t.done()]
        if not pending:
            return
        bucket = self._retiring.setdefault(conn_id, set())
        for task in pending:
            bucket.add(task)
            task.add_done_callback(
                functools.partial(self._discard_retired, conn_id)
            )

    def _discard_retired(
        self, conn_id: str, task: asyncio.Task[None],
    ) -> None:
        """Drop a completed retired pump from the per-conn set.

        Args:
            conn_id: The owning connection's id.
            task: The retired pump task that just completed.
        """
        bucket = self._retiring.get(conn_id)
        if bucket is not None:
            bucket.discard(task)
            if not bucket:
                self._retiring.pop(conn_id, None)
                self._maybe_drop_generation(conn_id)

    async def _await_retirement(self, conn_id: str) -> None:
        """Wait until every retired pump of *conn_id* has completed.

        A retired pump was cancelled, but cancellation is cooperative:
        its in-flight send may still complete and deliver an old
        ``listening: false`` report.  A successor listener defers its
        deliveries (its stdout pump and the duplicate-start re-report
        call this first) so that, in every schedule where the retired
        pump completes at all, the stale report lands strictly before
        the successor's ``listening: true``.

        The wait is BOUNDED by ``_RETIREMENT_GATE_SECONDS``: the
        ``SendCallback`` contract cannot guarantee a retired send ever
        completes (an event-specific endpoint may accept successor
        sends while wedging the stale one forever), and one such send
        must not suppress every successor's delivery for good
        (gpt-5.6-sol round-3 review, finding 3).  Past the deadline
        the successor proceeds; the retired pump's report remains
        unobservable through the generation tag — :meth:`_deliver`
        drops it before the send when it has not started, and the
        delivery boundary drops it via :meth:`accepts` when it was
        already in flight.

        Args:
            conn_id: The owning connection's id.
        """
        deadline = (
            asyncio.get_running_loop().time() + _RETIREMENT_GATE_SECONDS
        )
        while True:
            pending = [
                t for t in self._retiring.get(conn_id, ()) if not t.done()
            ]
            if not pending:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.wait(pending, timeout=remaining)

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
        await self._acquire_lifecycle(conn_id)
        failure: dict[str, Any] | None = None
        try:
            failure = await self._start_locked(conn_id, sensitivity, send)
        finally:
            self._release_lifecycle(conn_id)
        if failure is not None:
            # Delivered OFF the lifecycle lock: a send callback is not
            # time-bounded by the ``SendCallback`` contract, and a
            # blocked failure report holding the lock would stall
            # every later stop()/stop_all() of this connection forever
            # (gpt-5.6-sol round-4 review, finding 2).
            await self._safe_send(send, failure)

    async def _start_locked(
        self,
        conn_id: str,
        sensitivity: int | None,
        send: SendCallback,
    ) -> dict[str, Any] | None:
        """:meth:`start`'s body, under the connection's lifecycle lock.

        The lock closes the concurrent-start leak: the listener slot
        is empty across the ``create_subprocess_exec`` await, so two
        unserialized starts could both spawn and the second
        registration would orphan the first child.  Serialized, the
        second start observes the first's registration and takes the
        duplicate-start re-report path instead.

        No client send runs in this body: the lifecycle lock must
        never be held across an unbounded send callback (gpt-5.6-sol
        round-4 review, finding 2).  A failure report is returned to
        :meth:`start`, which delivers it after releasing the lock; the
        duplicate-start re-report is delegated to a pump task.

        Args:
            conn_id: The owning connection's id.
            sensitivity: Optional wake-word sensitivity 0..100.
            send: Async callback delivering event dicts to the owning
                client.

        Returns:
            A ``voiceWakeState`` failure report for :meth:`start` to
            deliver off the lock, or ``None`` when there is nothing to
            report from here (a spawned listener reports through its
            pumps).
        """
        if self._closing:
            # stop_all() is tearing the controller down: refuse to
            # spawn, or a start invisible to its snapshot could
            # publish a live child after stop_all returned
            # (gpt-5.6-sol round-4 review, finding 5).
            self._maybe_drop_generation(conn_id)
            return {
                "type": "voiceWakeState",
                "listening": False,
                "error": "voice listener not started: shutting down",
                "voiceGen": self._generations.get(conn_id, 0),
            }
        existing = self._listeners.get(conn_id)
        if existing is not None and existing.proc.returncode is not None:
            # The child already exited on its own and its stdout pump
            # is still delivering the final exit report (the pump
            # deregisters right after the report): wait for it so this
            # fresh start spawns a new listener instead of being
            # swallowed by the duplicate-start path below.  The bound
            # guards a report send wedged on a stuck endpoint; on
            # timeout the dead listener is RETIRED — marked stopped
            # and its pumps cancelled and awaited — not merely
            # deregistered.  A deregistered-but-alive pump was
            # invisible to ``stop``/``stop_all`` and could resume its
            # wedged send later, delivering the old ``listening:false``
            # AFTER the replacement's ``listening:true`` (gpt-5.6-sol
            # conc review, finding 7).  Cancellation aborts a
            # well-behaved in-flight send (``_safe_send`` swallows
            # ``Exception`` only, so ``CancelledError`` propagates and
            # the stale report is never delivered), and ``stopped``
            # suppresses a pump that has not yet reached its report.
            # Cancellation is cooperative, though: a send callback may
            # finish endpoint cleanup before honouring it, so a pump
            # still pending after the bounded wait is RETAINED in
            # ``_retiring`` — the replacement spawned below defers its
            # deliveries until it completes (bounded — see
            # ``_await_retirement``), and its stale report is
            # discarded by the generation tag once the replacement
            # spawn retires it (gpt-5.6-sol round-2 finding 3,
            # round-3 finding 3).
            pending = [t for t in existing.pumps if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=_EXIT_REPORT_JOIN_SECONDS)
            still_pending = [t for t in existing.pumps if not t.done()]
            if still_pending:
                existing.stopped = True
                # Publish the retirement BEFORE the cancellation wait —
                # and so before this listener leaves ``_listeners``
                # below, with no await between registration and
                # deregistration: a concurrent start() that no longer
                # finds the listener must find its unfinished pumps in
                # ``_retiring``, or its replacement could cross the
                # delivery gate before the old pump's stale report is
                # even visible (gpt-5.6-sol round-3 review, finding 2).
                self._retire_pumps(conn_id, still_pending)
                for task in still_pending:
                    task.cancel()
                await asyncio.wait(still_pending, timeout=1.0)
            if self._listeners.get(conn_id) is existing:
                del self._listeners[conn_id]
        current = self._listeners.get(conn_id)
        if current is not None:
            # Duplicate start: re-report ``listening: true`` from a
            # pump task, NEVER from under the lifecycle lock — the
            # send callback is unbounded, and a blocked re-report
            # holding the lock would stall stop()/stop_all() forever
            # and keep the child alive (gpt-5.6-sol round-4 review,
            # finding 2).  The task also waits out any retirement
            # first (a retired pump's stale ``listening: false`` may
            # still complete), verifies the listener is still the
            # registered one, and is registered in ``pumps`` so
            # ``stop``/restart joins and cancels it like any other
            # endpoint-touching task.
            current.pumps.append(asyncio.ensure_future(
                self._report_listening_after_retirement(
                    conn_id, current, send,
                )
            ))
            return None
        args = list(self._listener_args)
        if isinstance(sensitivity, (int, float)) and not isinstance(
            sensitivity, bool
        ):
            clamped = min(100, max(0, round(sensitivity)))
            args += ["--sensitivity", str(clamped)]
        spawn: asyncio.Task[asyncio.subprocess.Process] = (
            asyncio.ensure_future(asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # POSIX only: gives the child its own process group so
                # _terminate can reap grandchildren too.  Windows has
                # no setsid; there _terminate signals the child alone.
                start_new_session=(os.name == "posix"),
            ))
        )
        try:
            # Shielded: cancelling ``start()`` mid-spawn (the UDS
            # handler drain) must not abandon a process the event loop
            # may already have forked — the spawn runs to completion
            # under an OWNED disposal task instead (see the handler
            # below).
            proc = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            # Cancellation used to bypass the ``OSError``-only
            # generation GC below, leaving an ownerless generation
            # entry after a cancelled dead-listener restart
            # (gpt-5.6-sol round-6 review, finding 4).  Hand the
            # still-running spawn to a reap-registered disposal task —
            # it terminates the child if one materializes and repeats
            # the generation GC once nothing owns it — then clean up
            # and propagate.
            discard = asyncio.ensure_future(
                self._discard_spawn(conn_id, spawn)
            )
            self._reap_tasks.add(discard)
            discard.add_done_callback(self._reap_tasks.discard)
            self._maybe_drop_generation(conn_id)
            raise
        except OSError as err:
            # Free the connection's generation entry when nothing owns
            # it any more: the dead-listener path above may have kept
            # it alive for pumps that completed during the joins, and
            # with the replacement spawn failed no future lifecycle
            # callback would ever collect it — the map would grow one
            # ownerless entry per failed restart (gpt-5.6-sol round-4
            # review, finding 7).  Dropped BEFORE the tag is read so
            # the failure report carries the map's post-drop reading
            # and still passes the delivery boundary's ``accepts``
            # check.
            self._maybe_drop_generation(conn_id)
            return {
                "type": "voiceWakeState",
                "listening": False,
                "error": f"voice listener failed to start: {err}",
                # Tagged with the connection's CURRENT generation (no
                # listener was spawned, so none was bumped): the
                # failure report is a fresh, current report and must
                # pass the delivery boundary's ``accepts`` check.
                "voiceGen": self._generations.get(conn_id, 0),
            }
        listener = _Listener(proc, self._bump_generation(conn_id))
        self._listeners[conn_id] = listener
        listener.pumps = [
            asyncio.ensure_future(self._pump_stderr(listener)),
            asyncio.ensure_future(self._pump_stdout(conn_id, listener, send)),
        ]
        return None

    async def stop(self, conn_id: str) -> None:
        """Stop and reap *conn_id*'s listener, if any.

        Safe to call when no listener is running (e.g. from the
        transport's unconditional disconnect cleanup).

        Args:
            conn_id: The owning connection's id.
        """
        await self._acquire_lifecycle(conn_id)
        try:
            await self._stop_locked(conn_id)
        finally:
            self._release_lifecycle(conn_id)

    async def _stop_locked(self, conn_id: str) -> None:
        """:meth:`stop`'s body, under the connection's lifecycle lock.

        Serialization with :meth:`start` means a stop can never
        interleave with an in-flight spawn: it either runs before the
        spawn (the new listener then starts cleanly afterwards) or
        after the registration (the new listener is found and
        reaped).  The stop stays bounded — every wait below carries a
        deadline — so holding the lock cannot stall a later start
        indefinitely.

        Args:
            conn_id: The owning connection's id.
        """
        listener = self._listeners.get(conn_id)
        if listener is None:
            self._maybe_drop_generation(conn_id)
            return
        listener.stopped = True
        # The stop's linearization point — one synchronous block, no
        # await: publish the retirement of every unfinished pump
        # FIRST, retire the listener's generation (so a stale report
        # that has not started its send is dropped by ``_deliver`` and
        # an in-flight one stays discardable at the delivery boundary
        # via its ``voiceGen`` tag), and only then deregister.  A
        # concurrent start() that no longer sees the listener is
        # therefore guaranteed to see the retiring pumps and defer its
        # replacement's deliveries (gpt-5.6-sol round-3 review,
        # finding 2).  Registered pumps that complete inside the joins
        # below remove themselves from ``_retiring``.
        pending = [t for t in listener.pumps if not t.done()]
        self._retire_pumps(conn_id, pending)
        self._bump_generation(conn_id)
        del self._listeners[conn_id]
        # Tear down through an OWNED reap task, registered in the same
        # no-await block as the deregistration: the stop's caller can
        # be cancelled mid-teardown (the UDS handler drain cancels
        # straggling disconnect cleanups), and with the listener
        # already out of ``_listeners`` nothing else would know the
        # child exists — the shield keeps the reap running to
        # completion, and stop_all() joins outstanding reap tasks
        # (gpt-5.6-sol round-4 review, finding 3).  The task owns the
        # ENTIRE teardown — bounded process termination, the pump
        # join/cancel, and the trailing generation GC — not just the
        # child: a stop cancelled during the pump join used to leave
        # live endpoint-touching pumps (and their generation) behind,
        # invisible to stop_all(), which joins only reap tasks
        # (gpt-5.6-sol round-6 review, finding 3).
        reaper = asyncio.ensure_future(
            self._reap(conn_id, listener.proc, pending)
        )
        self._reap_tasks.add(reaper)
        reaper.add_done_callback(self._reap_tasks.discard)
        await asyncio.shield(reaper)

    async def _report_listening_after_retirement(
        self, conn_id: str, listener: _Listener, send: SendCallback,
    ) -> None:
        """Re-report ``listening: true`` once retirement has completed.

        The duplicate-start path's deferred variant: while a retired
        pump of *conn_id* may still deliver its old
        ``listening: false``, an immediate re-report could be
        contradicted by that stale report.  Waits for full retirement,
        then re-reports only when *listener* is still the connection's
        registered, un-stopped listener.

        Args:
            conn_id: The owning connection's id.
            listener: The live listener whose state is re-reported.
            send: The owning client's event callback.
        """
        await self._await_retirement(conn_id)
        if self._listeners.get(conn_id) is listener and not listener.stopped:
            await self._deliver(
                conn_id, listener, send,
                {"type": "voiceWakeState", "listening": True},
            )

    async def _reap(
        self,
        conn_id: str,
        proc: asyncio.subprocess.Process,
        pumps: list[asyncio.Task[None]],
    ) -> None:
        """Terminate *proc* and settle *pumps*, retaining ownership.

        Runs as its own task (see ``_reap_tasks``) so a cancelled
        :meth:`stop` cannot orphan the child (gpt-5.6-sol round-4
        review, finding 3).  The task owns the FULL stop teardown, not
        just the process: after the bounded termination it joins the
        listener's retiring pump tasks — the caller (disconnect
        cleanup, daemon shutdown) must be able to assume no controller
        coroutine touches the connection's endpoint once the stop has
        settled, and a stop cancelled during an unowned pump join used
        to abandon live pumps that stop_all() (which joins only reap
        tasks) never saw (gpt-5.6-sol round-6 review, finding 3).  The
        process is dead by then, so both pipes are at EOF and the
        pumps end promptly; a pump wedged on a stuck client send is
        cancelled, and one whose send resists cancellation past the
        bounded waits stays in ``_retiring`` — gating (bounded) a
        successor's deliveries (see :meth:`_await_retirement`) and
        settled again, bounded, by :meth:`stop_all`.  Finally collects
        the connection's generation entry when this reap turns out to
        be its last owner.

        Args:
            conn_id: The owning connection's id.
            proc: The deregistered listener child to terminate.
            pumps: The listener's still-unfinished pump tasks (already
                published in ``_retiring`` by the stop's linearization
                block).
        """
        try:
            await self._terminate(proc)
            if pumps:
                _, still_pending = await asyncio.wait(pumps, timeout=5.0)
                for task in still_pending:
                    task.cancel()
                if still_pending:
                    await asyncio.wait(still_pending, timeout=1.0)
        finally:
            self._maybe_drop_generation(conn_id)

    async def _discard_spawn(
        self,
        conn_id: str,
        spawn: asyncio.Task[asyncio.subprocess.Process],
    ) -> None:
        """Dispose of a listener spawn abandoned by a cancelled start.

        A ``start()`` cancelled while awaiting (shielded)
        ``create_subprocess_exec`` cannot know whether the event loop
        already forked the child.  This task — registered in
        ``_reap_tasks``, so :meth:`stop_all` joins it — retains
        ownership of the in-flight spawn: it waits for it to settle,
        terminates the child if one materialized (it was never
        registered in ``_listeners``, so nothing else would ever reap
        it), and repeats the generation GC that the cancelled start's
        ``OSError`` branch would have performed (gpt-5.6-sol round-6
        review, finding 4).

        Args:
            conn_id: The connection whose start was cancelled.
            spawn: The still-running (never-cancelled) spawn task.
        """
        try:
            proc = await spawn
        except OSError:
            logger.debug(
                "abandoned voice listener spawn failed", exc_info=True,
            )
        else:
            await self._terminate(proc)
        finally:
            self._maybe_drop_generation(conn_id)

    async def stop_all(self) -> None:
        """Stop every listener and every in-flight lifecycle (shutdown).

        Covers more than the registered listeners: a :meth:`start`
        that has entered lifecycle processing but not yet published
        its child holds (or waits on) its connection's lifecycle lock,
        so stopping every connection with lifecycle activity queues
        behind it and reaps the child it registers; a start that has
        not yet taken a lifecycle reference is refused outright by the
        ``_closing`` barrier; and children whose earlier stop was
        cancelled mid-teardown are covered by joining the outstanding
        reap tasks — each bounded (gpt-5.6-sol round-4 review,
        findings 3 and 5).

        The sweep LOOPS until the controller is quiescent — no
        listeners, no lifecycle holders or waiters, no unfinished reap
        tasks — instead of stopping a one-time snapshot: a start
        queued behind this method's own stop() takes its lifecycle
        reference only after the snapshot, and a single pass would
        lower ``_closing`` and return before the woken waiter runs,
        letting it spawn a listener after shutdown completed
        (gpt-5.6-sol round-6 review, finding 2).  Every connection
        seen by an iteration is stopped (a queued start is refused by
        the still-raised barrier once it acquires the lock), so the
        loop drains all waiters that existed before the final
        emptiness check, which runs with no await before ``return``.

        Before returning, any retired pump task still pending — e.g.
        published by a stop that was cancelled while shield-awaiting
        its reap task — is cancelled and awaited (bounded by
        ``_RETIREMENT_GATE_SECONDS``): the caller may assume no
        cancellation-responsive controller coroutine touches a
        connection's endpoint after shutdown returned (gpt-5.6-sol
        round-6 review, finding 3).  A send callback that resists
        cancellation past the bound is logged and left behind — its
        report stays unobservable through the generation tag — so one
        wedged callback cannot stall shutdown forever.
        """
        self._closing += 1
        try:
            while True:
                for conn_id in set(self._listeners) | set(
                    self._lifecycle_holds
                ):
                    await self.stop(conn_id)
                reaping = [t for t in self._reap_tasks if not t.done()]
                if reaping:
                    await asyncio.wait(reaping)
                    continue
                if not self._listeners and not self._lifecycle_holds:
                    break
            retired = [
                task
                for bucket in self._retiring.values()
                for task in bucket
                if not task.done()
            ]
            if retired:
                for task in retired:
                    task.cancel()
                await asyncio.wait(
                    retired, timeout=_RETIREMENT_GATE_SECONDS,
                )
            # Make the post-shutdown state deterministic: completed
            # retired pumps normally clean up through their done
            # callbacks, but those run via ``call_soon`` and may not
            # have fired yet when the wait above returns.
            for conn_id in list(self._retiring):
                bucket = self._retiring.get(conn_id, set())
                for task in [t for t in bucket if t.done()]:
                    bucket.discard(task)
                if not bucket:
                    self._retiring.pop(conn_id, None)
                    self._maybe_drop_generation(conn_id)
                else:
                    logger.warning(
                        "voice-wake shutdown leaving %d retired pump(s) "
                        "of connection %s behind: their sends resisted "
                        "cancellation past the %.0fs bound",
                        len(bucket), conn_id, _RETIREMENT_GATE_SECONDS,
                    )
        finally:
            self._closing -= 1

    async def _pump_stdout(
        self, conn_id: str, listener: _Listener, send: SendCallback,
    ) -> None:
        """Forward the child's stdout protocol lines until it exits."""
        # Defer delivery while a retired predecessor pump of this
        # connection can still complete an old ``listening: false``
        # send: this listener's ``listening: true`` (and every event
        # after it) waits — bounded — for retirement, so in every
        # schedule where the retired pump completes, the stale report
        # lands strictly before this listener's first report; past the
        # bound, the stale report is discarded through its generation
        # tag instead (gpt-5.6-sol round-2 finding 3, round-3
        # finding 3).  The child's stdout simply buffers in the pipe
        # meanwhile.
        await self._await_retirement(conn_id)
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
                    await self._deliver(
                        conn_id, listener, send,
                        {"type": "voiceWakeState", "listening": True},
                    )
                await self._deliver(conn_id, listener, send, event)
        except Exception:
            logger.debug("voice-wake stdout pump failed", exc_info=True)
        returncode = await proc.wait()
        # A stop() (or a stop_all/disconnect) reports nothing: its
        # owner is gone or asked for the stop, so only a self-exited
        # listener reports its final state.  Report FIRST and
        # deregister AFTER the report: while the report is in flight
        # the entry stays visible, so a concurrent stop() finds it,
        # sets ``stopped`` and joins this pump (keeping its
        # no-endpoint-touch-after-return guarantee), and a concurrent
        # start() (see the dead-listener guard there) waits for the
        # report instead of spawning a fresh listener whose state the
        # stale report would clobber.
        if listener.stopped:
            return
        error: str | None = None
        if returncode != 0:
            # ``stderr_tail`` is written by the sibling stderr pump;
            # reading it right after ``proc.wait()`` relied on a
            # scheduling order CPython does not guarantee, and the
            # child's final stderr line was intermittently missing
            # from the diagnostic (gpt-5.6-sol round-2 review,
            # finding 4).  Join the pump explicitly instead.  The
            # process is dead so stderr is normally at EOF at once;
            # the bound covers a grandchild holding the inherited fd
            # open, in which case the diagnostic uses the tail
            # captured so far.
            try:
                await asyncio.wait_for(
                    listener.stderr_done.wait(), _EXIT_REPORT_JOIN_SECONDS,
                )
            except TimeoutError:
                logger.debug(
                    "voice-wake stderr pump still draining at exit "
                    "report time; reporting with the tail so far",
                )
            detail = listener.stderr_tail.strip().split("\n")[-1].strip()
            error = f"voice listener exited (code {returncode})" + (
                f": {detail}" if detail else ""
            )
        await self._deliver(conn_id, listener, send, {
            "type": "voiceWakeState",
            "listening": False,
            **({"error": error} if error else {}),
        })
        if self._listeners.get(conn_id) is listener:
            del self._listeners[conn_id]
            self._maybe_drop_generation(conn_id)

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
        finally:
            # Signals — on EOF, error, or cancellation alike — that
            # ``stderr_tail`` will not change any more; the stdout
            # pump joins this before composing the exit diagnostic.
            listener.stderr_done.set()

    @staticmethod
    def _signal_group(pid: int, sig: signal.Signals) -> bool:
        """Best-effort signal to *pid*'s process group.

        Args:
            pid: The group leader's pid (the child was spawned with
                ``start_new_session=True`` on POSIX).
            sig: The signal to deliver.

        Returns:
            ``True`` when the group was signalled; ``False`` when the
            platform has no ``os.killpg`` (Windows) or the call failed
            — the caller then falls back to signalling the process
            alone.
        """
        killpg = getattr(os, "killpg", None)
        if killpg is None:
            return False
        try:
            killpg(pid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """SIGTERM the child's process group, escalating to SIGKILL.

        On platforms without process groups (Windows) the child alone
        is terminated/killed via the ``Process`` API.  Every wait is
        bounded: even after SIGKILL, asyncio's ``Process.wait()`` can
        pend forever when a detached descendant keeps the inherited
        stdout/stderr pipes open, and the caller may be holding the
        connection's lifecycle lock (gpt-5.6-sol round-4 review,
        finding 4) — past ``_KILL_REAP_SECONDS`` the wait is abandoned
        with a log line, the kill already delivered.
        """
        if proc.returncode is not None:
            return
        pid = proc.pid
        if not self._signal_group(pid, signal.SIGTERM):
            try:
                proc.terminate()
            except ProcessLookupError:
                return
        try:
            await asyncio.wait_for(proc.wait(), _TERM_GRACE_SECONDS)
        except TimeoutError:
            if not self._signal_group(
                pid, getattr(signal, "SIGKILL", signal.SIGTERM)
            ):
                try:
                    proc.kill()
                except ProcessLookupError:
                    return
            try:
                await asyncio.wait_for(proc.wait(), _KILL_REAP_SECONDS)
            except TimeoutError:
                logger.warning(
                    "voice listener pid %s still unreaped %.0fs after "
                    "SIGKILL (an escaped descendant may hold its "
                    "pipes); abandoning the wait",
                    pid, _KILL_REAP_SECONDS,
                )

    @staticmethod
    async def _safe_send(
        send: SendCallback, event: dict[str, Any],
    ) -> None:
        """Deliver *event*, swallowing a dead client connection."""
        try:
            await send(event)
        except Exception:
            logger.debug("voice-wake event delivery failed", exc_info=True)
