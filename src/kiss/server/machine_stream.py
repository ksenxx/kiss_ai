# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stream to a machine's daemon: the socket, its sends and the thread relaying its frames.

The base of :class:`kiss.server.remote_tabs.RemoteTab` (a bound tab's
stream) and :class:`kiss.server.remote_views.RemoteViewStream` (a
machine's terminals and browser pages).  A subclass keeps its own
bookkeeping and decodes the relayed frames (:meth:`_relay_frame`); its
owner is told when the stream ends (``owner.dropped(stream, lost)``).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar.machines import REQUEST_TIMEOUT, Machine, MachineError, link_for

logger = logging.getLogger(__name__)

DROPPED_FIELDS = ("connId", "machine")
"""Fields of a forwarded command that only mean something on this daemon."""


class MachineStream:
    """One socket to the daemon of a remote machine and the thread relaying its frames.

    Attributes:
        machine: The machine the stream talks to.
        lock: Guards the subclass's bookkeeping and ``retired``.  An
            ``RLock``: the relay thread takes it per frame, so a
            holder pauses relaying and can still route its own
            commands meanwhile (a test's way of fixing the order of a
            frame and a command).
        retired: Set (under ``lock``) once the owner took the stream
            out of service; frames still in flight are dropped.
    """

    def __init__(self, machine: Machine, owner: Any, name: str) -> None:
        """Connect to *machine* and start relaying its frames on the thread *name*.

        A subclass calls this last: the relay thread reads the
        subclass's attributes as soon as it starts.

        Raises:
            MachineError: When the machine is unreachable or rejects the
                password, or the relay thread cannot start (the socket
                is closed then).
        """
        self.machine = machine
        self.lock = threading.RLock()
        self.retired = False
        self._owner = owner
        self._closed = False
        self._ws: ClientConnection = link_for(machine).open_connection()
        self._thread = threading.Thread(target=self._relay, name=name, daemon=True)
        try:
            self._thread.start()
        except RuntimeError as exc:
            # Thread exhaustion: nothing would ever read or close the
            # socket, and the machine would keep its state for it.
            self._close_socket()
            raise MachineError(f"{machine.name}: the stream could not start: {exc}") from exc

    def _dropped_fields(self, cmd: dict[str, Any]) -> tuple[str, ...]:
        """The fields of *cmd* :meth:`send` leaves out (:data:`DROPPED_FIELDS`)."""
        return DROPPED_FIELDS

    def send(self, cmd: dict[str, Any]) -> None:
        """Forward *cmd* to the machine's daemon, without this daemon's own fields.

        Bounded by :data:`~kiss.agents.sorcar.machines.REQUEST_TIMEOUT`:
        a plain ``ws.send`` has no deadline and holds the connection's
        write lock, so one stuck on a daemon that stopped reading would
        block the client's command loop for good and wedge the
        stream's :meth:`close` (``closeTab``, daemon shutdown) behind
        it.  A write that misses the deadline closes the socket
        underneath, which ends the stream as a lost connection.

        Raises:
            MachineError: When the socket is gone or the daemon did not
                accept the command in time.
        """
        dropped = self._dropped_fields(cmd)
        frame = {k: v for k, v in cmd.items() if k not in dropped}
        # The machine's daemon checks the password on every frame of a
        # password-authenticated peer (an SSH machine's stream is
        # certificate-authenticated and sends none).
        if self.machine.password:
            frame["password"] = self.machine.password
        try:
            local_endpoint.send(self._ws, json.dumps(frame), timeout=REQUEST_TIMEOUT)
        except (ConnectionClosed, OSError) as exc:
            raise MachineError(f"{self.machine.name}: the connection is lost: {exc}") from exc

    def close(self) -> None:
        """Close the socket; the relay thread then ends without an error."""
        self._closed = True
        self._close_socket()

    def _close_socket(self) -> None:
        try:
            self._ws.close()
        except Exception:
            logger.debug("closing %s failed", self._thread.name, exc_info=True)

    def _relay(self) -> None:
        """Hand every decoded frame to :meth:`_relay_frame`; tell the owner when the stream ends.

        The socket goes with the relay, whatever ended it: a stream
        reported lost because a frame's delivery raised must not keep
        its socket (and the machine's resources for it) open.
        """
        try:
            for raw in self._ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    self._relay_frame(msg, text)
        except (ConnectionClosed, OSError):
            pass
        finally:
            lost = not self._closed
            self._close_socket()
            self._owner.dropped(self, lost=lost)

    def _relay_frame(self, msg: dict[str, Any], text: str) -> None:
        """Deliver one decoded frame (*text* is its wire form) to this daemon's clients."""
        raise NotImplementedError
