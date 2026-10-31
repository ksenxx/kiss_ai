# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Flow control of the screencast frames of one browser page, per viewer.

A page's frames (``browserFrame`` events) are produced at up to 15 a
second whenever the page repaints, each some 50-75 KB of JPEG.  Pushed
to a viewer regardless of what its link carries, they pile up in the
socket, tunnel and relay buffers between the daemon and the viewer: the
picture lags further and further behind, and the keepalive pong of the
link travels behind that backlog too, so a link slower than the stream
is declared dead after its ping timeout and the Browser tab vanishes
from every surface (the "stream browser crash").

So a frame is sent to a viewer only while at most :data:`MAX_IN_FLIGHT`
of its frames await the viewer's ``browserFrameAck``; a frame produced
meanwhile is held back, and the newest one goes out when the ack
arrives, so a slow viewer sees the latest picture at the pace its link
carries, and the backlog anywhere on the way is at most two frames.
A viewer that has acknowledged a frame once is held to that window
however long its acknowledgements take: an acknowledgement is the only
evidence that the link has carried a frame.  A viewer that has never
acknowledged anything (an older surface or relay) is not starved: a
frame of its unacknowledged for :data:`ACK_GRACE` seconds counts as
delivered, and the frame held back for it goes out then, whether or
not the page repaints again.
"""

from __future__ import annotations

import time
from collections.abc import Callable

MAX_IN_FLIGHT = 2
"""Frames a viewer may owe an acknowledgement for before the next one waits."""

ACK_GRACE = 5.0
"""Seconds after which a frame of a viewer that never acknowledges counts as delivered."""


class FrameFlow[Frame]:
    """The frames in flight to each viewer of one page, and the frame held back for it.

    Not thread-safe: the owner calls it, and runs the callbacks it
    schedules, on one thread (the browser tab service's loop, a relay
    stream under its lock).
    """

    def __init__(
        self,
        send: Callable[[str, Frame], None],
        schedule: Callable[[float, Callable[[], None]], None],
    ) -> None:
        """Set up the flow of one page.

        Args:
            send: Delivers a frame to a viewer (its connection id).
            schedule: Runs a callback after a delay in seconds (on the
                owner's thread), for the grace release of a held frame.
        """
        self._send = send
        self._schedule = schedule
        # Per viewer, when each frame it has not acknowledged was sent.
        self._in_flight: dict[str, list[float]] = {}
        self._held: dict[str, Frame] = {}
        self._acking: set[str] = set()  # the viewers that have acknowledged a frame
        self._releasing: set[str] = set()  # the viewers with a grace release scheduled

    def offer(self, viewer: str, frame: Frame) -> None:
        """Send *frame* to *viewer* now, or hold it back until the viewer catches up.

        Args:
            viewer: The viewer's connection id.
            frame: The frame just produced; it replaces any frame held
                back for the viewer.
        """
        pending = self._in_flight.setdefault(viewer, [])
        if viewer not in self._acking:
            self._expire(pending)
        if len(pending) < MAX_IN_FLIGHT:
            self._held.pop(viewer, None)  # older than this frame
            pending.append(time.monotonic())
            self._send(viewer, frame)
            return
        self._held[viewer] = frame
        if viewer not in self._acking:
            self._arm_release(viewer, pending)

    def acked(self, viewer: str) -> None:
        """Record *viewer*'s acknowledgement of its oldest frame in flight; send the held one.

        Ignored for a viewer the flow has sent nothing to.

        Args:
            viewer: The viewer's connection id.
        """
        pending = self._in_flight.get(viewer)
        if pending is None:
            return
        self._acking.add(viewer)
        if pending:
            del pending[0]
        self._send_held(viewer)

    def forget(self, viewer: str) -> None:
        """Drop everything known about *viewer* (it stopped watching the page).

        Args:
            viewer: The viewer's connection id.
        """
        self._in_flight.pop(viewer, None)
        self._held.pop(viewer, None)
        self._acking.discard(viewer)

    def close(self) -> None:
        """The page is gone: drop every viewer; a release still scheduled finds nothing."""
        self._in_flight.clear()
        self._held.clear()
        self._acking.clear()

    @staticmethod
    def _expire(pending: list[float]) -> None:
        """Drop the frames in flight whose acknowledgement is overdue (a viewer that never acks)."""
        now = time.monotonic()
        while pending and now - pending[0] >= ACK_GRACE:
            del pending[0]

    def _arm_release(self, viewer: str, pending: list[float]) -> None:
        """Schedule the grace release of the frame held back for *viewer*, once."""
        if viewer in self._releasing:
            return
        self._releasing.add(viewer)
        delay = max(0.0, pending[0] + ACK_GRACE - time.monotonic())
        self._schedule(delay, lambda: self._release(viewer))

    def _release(self, viewer: str) -> None:
        """The grace ran out: send the held frame, or wait for the next frame in flight to age.

        Nothing to do for a viewer forgotten meanwhile, or one that
        has acknowledged since: its acknowledgements release its frames.
        """
        self._releasing.discard(viewer)
        pending = self._in_flight.get(viewer)
        if pending is None or viewer in self._acking:
            return
        self._expire(pending)
        self._send_held(viewer)
        if viewer in self._held:
            self._arm_release(viewer, pending)

    def _send_held(self, viewer: str) -> None:
        """Send the frame held back for *viewer* when it may have one more in flight."""
        pending = self._in_flight.get(viewer)
        if pending is not None and viewer in self._held and len(pending) < MAX_IN_FLIGHT:
            pending.append(time.monotonic())
            self._send(viewer, self._held.pop(viewer))
