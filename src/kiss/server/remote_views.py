# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Terminal and browser tabs of a registered remote machine.

The machine drop-down of a chat header chooses where the chat's tasks
run; the Terminal and Browser buttons of such a chat open a shell, or
a page of the browser, on that same machine: ``terminalOpen`` and
``browserOpen`` carry the pick as ``machine``.  :class:`RemoteViews`
keeps one :class:`RemoteViewStream` per machine in use: an
authenticated WebSocket to the machine's daemon
(:meth:`~kiss.agents.sorcar.machines.MachineLink.open_connection`),
opened on the first such command, or when a chat tab is bound to the
machine (:meth:`RemoteViews.bind`, told by
:class:`~kiss.server.remote_tabs.RemoteTabs`), and closed once it
carries no terminal or page and no tab is bound to the machine any
more (or when this daemon stops), over which the commands are
forwarded and the machine's terminal and browser events are relayed
to this daemon's clients.  A tab's binding keeps the stream because
the agent of a task running on the machine may call for the user
(``show_browser``, a sign-in hand-off): the page it opens is
announced, focused, to every client of the machine's daemon, and the
stream relays it to every surface here the same way.  A stream lost
while a tab is still bound to its machine is opened again after
:data:`REOPEN_DELAY` (a machine that drops every socket costs one
connection per delay, not a connect loop).  A stream that opens asks
the machine for the pages its browser already has open
(``browserList``, answered with the ``browserTabs`` snapshot a
``ready`` gets, or with an ``error`` by an older daemon) and shows
them here, unfocused.

The remote daemon owns the shells and the pages exactly as it would
for its own surfaces; this daemon's clients see them as they see the
local ones, with two differences:

* **Browser tab ids are namespaced.**  The remote mints
  ``browser__N``, as this daemon does, so every relayed id is
  ``browser__N@machine`` (:func:`namespaced`) and every browser
  command naming such an id goes to that machine with the plain id
  (:func:`split_tab_id`).  A terminal tab's id is minted by the client
  and forwarded as it is; the stream remembers which connection opened
  it.
* **Every relayed event is stamped ``machine``**, so a surface labels
  the tab with the machine.  A ``ready`` is answered with one
  ``browserTabs`` snapshot flagged ``remote`` listing the pages of
  every machine (each stamped), against which the surface reconciles
  all its remote browser tabs, those of machines whose stream is gone
  included.

The remote addresses a ``terminalData`` or ``browserFrame`` to the
stream's connection; the stream hands it to the connection that opened
the terminal, or to every connection watching the page
(``browserViewport``).  A terminal takes input, a resize and a close
from the connection that opened it only, as the local
:class:`~kiss.server.terminal_tab.TerminalService` does.  A connection
that drops stops watching its pages, and its shells are hung up after
:data:`~kiss.server.terminal_tab.GRACE_SECONDS` unless a reconnecting
page claims them with ``terminalOpen`` first.  A lost stream ends its
terminals with a ``terminalError`` naming the machine, answers the
``browserOpen`` requests it never replied to with a ``browserError``,
and closes its browser tabs on every surface.

Two locks keep what this daemon recorded and what the machine sees in
step.  A command's bookkeeping and its send are one step under the
stream's ``send_lock`` (so a stale hide cannot overtake a show, a
later ``browserOpen`` cannot overtake an earlier one and take its
reply, and the grace hang-up cannot close a shell a reload reclaimed
meanwhile); the state ``lock`` is never held across a send.  Relayed
events are delivered under the state ``lock`` (the printer's sends
are scheduled, not awaited), and a ``ready`` snapshot is copied and
delivered under every stream's lock, so a surface never sees a
snapshot that predates a close it already got.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from typing import Any

from kiss.agents.sorcar.machines import Machine, MachineError, load_machines
from kiss.server.machine_stream import MachineStream
from kiss.server.terminal_tab import GRACE_SECONDS

logger = logging.getLogger(__name__)

TERMINAL_COMMANDS = frozenset({
    "terminalOpen", "terminalInput", "terminalResize", "terminalClose",
})
BROWSER_COMMANDS = frozenset({
    "browserOpen", "browserClose", "browserNavigate", "browserInput", "browserViewport",
})
VIEW_COMMANDS = TERMINAL_COMMANDS | BROWSER_COMMANDS
"""The commands :meth:`RemoteViews.route` looks at."""

TERMINAL_EVENTS = frozenset({"terminalData", "terminalOpened", "terminalExit", "terminalError"})
BROWSER_EVENTS = frozenset({
    "openBrowserTab", "browserTabs", "browserFrame", "browserState", "closeBrowserTab",
    "browserError",
})

_MACHINE_SEPARATOR = "@"

REOPEN_DELAY = 3.0
"""Seconds before a stream lost while a tab is still bound to its machine is opened again."""

_PAGES_REQUEST = "browserList"
_PAGES_UNKNOWN = f"Unknown command: {_PAGES_REQUEST}"
"""The ``error`` an older daemon answers :meth:`RemoteViewStream.request_pages` with."""


def namespaced(tab_id: str, machine_name: str) -> str:
    """The id a remote browser tab *tab_id* of *machine_name* has on this daemon."""
    return f"{tab_id}{_MACHINE_SEPARATOR}{machine_name}"


def split_tab_id(tab_id: str) -> tuple[str, str]:
    """Split a browser tab id into the remote's own id and the machine ("" when local)."""
    remote_id, sep, machine_name = tab_id.rpartition(_MACHINE_SEPARATOR)
    if not sep:
        return tab_id, ""
    return remote_id, machine_name


def registered_machine(name: str) -> Machine | None:
    """The registered machine called *name* (None when none is).

    By name only, unlike :func:`~kiss.agents.sorcar.machines.find_machine`,
    which also matches a host: the view streams here and the bound tabs
    of :mod:`kiss.server.remote_tabs` are keyed by the machine's name.
    """
    return next((m for m in load_machines() if m.name == name), None)


class RemoteViewStream(MachineStream):
    """One machine's stream: its terminals and browser pages shown here.

    Attributes:
        terminals: By terminal tab id, the connection that opened it
            (the one its output goes to).
        opening: The terminal tab ids whose ``terminalOpen`` the machine
            has not answered with ``terminalOpened`` yet: a
            ``terminalExit`` arriving meanwhile is the previous shell's
            (one a hang-up closed just before the re-open) and goes to
            nobody.  The replies carry no request id, so an open
            answered only after a close and re-open of its tab (a
            round trip's window) is taken for the re-open's answer and
            lets the old exit through; the page then shows an exit
            note its fresh shell clears.
        openers: The connections whose ``browserOpen`` awaits its
            reply (the focused ``openBrowserTab`` or a ``browserError``
            addressed to the stream), oldest first.
        browser_tabs: By the remote's own tab id, the last
            ``openBrowserTab`` of every page the machine's browser has
            open, already namespaced and stamped (unfocused): a
            connecting surface is told them (:meth:`RemoteViews.replay_to`).
        viewers: By the remote's own tab id, the connections watching
            the page (a visible ``browserViewport``): every frame of
            the page goes to each of them.
        seen: The remote ids of the pages opened or closed over the
            stream before the machine answered :meth:`request_pages`
            (None once it has, or once it answered that it does not
            know the command: an older daemon): the answer is older
            than those events, so it may not resurrect them.
        lock: Guards the attributes above and ``retired``; relayed
            events are delivered under it, nothing is sent under it.
        send_lock: Makes a command's bookkeeping and its send one step
            (taken before ``lock``, released after the send).
    """

    def __init__(self, machine: Machine, owner: RemoteViews) -> None:
        """Connect to *machine* and start relaying its frames.

        Raises:
            MachineError: When the machine is unreachable or rejects the
                password.
        """
        self.terminals: dict[str, str] = {}
        self.opening: set[str] = set()
        self.openers: list[str] = []
        self.browser_tabs: dict[str, dict[str, Any]] = {}
        self.viewers: dict[str, set[str]] = {}
        self.seen: set[str] | None = set()
        self.send_lock = threading.Lock()
        super().__init__(machine, owner, f"remote-views-{machine.name}")

    def request_pages(self) -> None:
        """Ask the machine for the pages its browser has open (``browserList``).

        Sent once the stream is the machine's registered one, so the
        ``browserTabs`` answer (:meth:`_relay_snapshot`) is relayed by
        a stream a ``ready`` snapshot sees too.  A stream lost this
        early is reported by its relay thread.
        """
        with contextlib.suppress(MachineError):
            self.send({"type": _PAGES_REQUEST})

    def _relay_frame(self, msg: dict[str, Any], text: str) -> None:
        """Deliver one decoded frame when it is a terminal or browser event."""
        kind = str(msg.get("type") or "")
        if kind in TERMINAL_EVENTS:
            self._relay_terminal(msg)
        elif kind == "browserTabs":
            self._relay_snapshot(msg)
        elif kind in BROWSER_EVENTS:
            self._relay_browser(kind, msg)
        elif kind == "error" and msg.get("text") == _PAGES_UNKNOWN:
            # No snapshot is coming: nothing to hold back for it.
            with self.lock:
                self.seen = None

    def _relay_snapshot(self, msg: dict[str, Any]) -> None:
        """Show the pages of the machine's ``browserTabs`` snapshot here, unfocused.

        The answer to :meth:`request_pages`, taken once: a page the
        stream has seen opened or closed meanwhile (``seen``) is left
        as the stream has it, as the machine took the snapshot before
        it sent those events; any other page is cached and told to
        every client as an unfocused ``openBrowserTab``.
        """
        with self.lock:
            if self.retired or self.seen is None:
                return
            seen, self.seen = self.seen, None
            for tab in msg.get("tabs") or ():
                if not isinstance(tab, dict):
                    continue
                remote_id = str(tab.get("tab_id") or "")
                if not remote_id or remote_id in seen or remote_id in self.browser_tabs:
                    continue
                event = self._stamped(tab)
                event["tab_id"] = namespaced(remote_id, self.machine.name)
                event["focus"] = False
                self.browser_tabs[remote_id] = event
                self._owner.deliver(json.dumps(event))

    def _relay_terminal(self, msg: dict[str, Any]) -> None:
        """A terminal event goes to the connection that opened the terminal."""
        tab_id = str(msg.get("tab_id") or "")
        kind = msg.get("type")
        with self.lock:
            if self.retired:
                return
            if kind == "terminalOpened":
                self.opening.discard(tab_id)
            elif kind == "terminalExit" and tab_id in self.opening:
                return  # the shell closed just before the re-open; not the new one's
            conn_id = self.terminals.get(tab_id, "")
            if conn_id:
                self._owner.deliver_to(conn_id, json.dumps(self._stamped(msg)))

    def _relay_browser(self, kind: str, msg: dict[str, Any]) -> None:
        """Relay a browser event with namespaced ids and the ``machine`` stamp.

        A frame goes to the page's viewers.  The remote strips the
        ``connId`` it addressed a frame with, so the reply to a
        ``browserOpen`` is told by its shape: a focused, non-popup
        ``openBrowserTab`` while an opener waits goes to the oldest
        waiting opener (one while none waits is an agent's call for
        attention: every client gets it, focused, as the remote's own
        surfaces do), and a ``browserError`` without a tab (the browser
        could not launch) likewise.  Everything else goes to every
        client.  The event is delivered under the lock, in the order
        the bookkeeping saw it (a ``ready`` snapshot copies the
        bookkeeping under the same lock).
        """
        remote_id = str(msg.get("tab_id") or "")
        event = self._stamped(msg)
        if "tab_id" in msg:
            event["tab_id"] = namespaced(remote_id, self.machine.name)
        to: list[str] = []  # [] = every client
        answered = False  # a browserOpen of ours failed: its opener is told
        with self.lock:
            if self.retired:
                return
            if self.seen is not None and kind in ("openBrowserTab", "closeBrowserTab"):
                self.seen.add(remote_id)
            if kind == "browserFrame":
                to = list(self.viewers.get(remote_id, ()))
                if not to:
                    return
            elif kind == "openBrowserTab":
                self.browser_tabs[remote_id] = dict(event, focus=False)
                if msg.get("focus") and not msg.get("popup") and self.openers:
                    to = [self.openers.pop(0)]
            elif kind == "browserState":
                cached = self.browser_tabs.get(remote_id)
                if cached is not None:
                    cached.update({k: v for k, v in event.items() if k in cached and k != "type"})
            elif kind == "closeBrowserTab":
                self.browser_tabs.pop(remote_id, None)
                self.viewers.pop(remote_id, None)
            elif kind == "browserError" and not remote_id:
                if not self.openers:
                    return
                to = [self.openers.pop(0)]
                answered = True
            text = json.dumps(event)
            if to:
                for conn_id in to:
                    self._owner.deliver_to(conn_id, text)
            else:
                self._owner.deliver(text)
        if kind == "closeBrowserTab" or answered:
            self._owner.close_if_idle(self)

    def _stamped(self, msg: dict[str, Any]) -> dict[str, Any]:
        """A copy of *msg* without the remote's addressing, stamped with the machine."""
        event = {k: v for k, v in msg.items() if k not in ("connId", "tabId")}
        event["machine"] = self.machine.name
        return event


class RemoteViews:
    """The terminal and browser tabs this daemon shows from remote machines."""

    def __init__(self, backend: Any) -> None:
        """*backend* is the daemon (``ServerBackend``): its printer."""
        self._backend = backend
        self._streams: dict[str, RemoteViewStream] = {}
        # By chat tab id, the machine the tab is bound to.
        self._bound: dict[str, str] = {}
        # Set by close_all: no stream is registered (or opened) from then on.
        self._closing = False
        self._lock = threading.Lock()

    def bind(self, tab_id: str, machine_name: str) -> None:
        """Record that the chat tab *tab_id* is bound to *machine_name* ("" when not any more).

        A stream to the machine stays open while a tab is bound to it,
        so a page its agent opens for the user is shown here (see the
        module).  Quick, under :class:`~kiss.server.remote_tabs.RemoteTabs`'
        registry lock, as the registry changes (a tab registered, a
        stream retired), so the bindings here are the registry's; the
        caller then opens the machine's stream (:meth:`open_for`) or
        releases the previous one (:meth:`release`) outside its lock.
        """
        with self._lock:
            self._bound.pop(tab_id, None)
            if machine_name:
                self._bound[tab_id] = machine_name

    def warm(self, machine_name: str) -> threading.Thread:
        """Start opening the stream of *machine_name* in the background; return the thread.

        For a bind (:meth:`kiss.server.remote_tabs.RemoteTabs._bind`):
        the stream is connected alongside the tab's own stream instead
        of after it (each is a TLS handshake and a login away, several
        round trips on a remote link), so the chat opened from the
        machine, or the run sent to it, waits for one round of
        handshakes, not two.  The caller joins the thread before
        :meth:`open_for`, which then finds the stream open (or opens
        it, when this could not), and calls :meth:`release` when the
        bind did not bind after all: a stream opened for nothing is
        closed when idle.  A stream that cannot be opened is reported
        by :meth:`open_for`, not here.
        """
        thread = threading.Thread(
            target=self._warm, args=(machine_name,), daemon=True,
            name=f"remote-views-warm-{machine_name}",
        )
        thread.start()
        return thread

    def _warm(self, machine_name: str) -> None:
        """:meth:`warm`'s body: open the stream, quietly."""
        with contextlib.suppress(MachineError):
            self._stream_for(machine_name)

    def open_for(self, machine_name: str) -> None:
        """Open (or keep) the stream of *machine_name* while a tab is bound to it.

        Blocks on the handshake (already done when :meth:`warm` opened
        the stream): a bound tab's task is forwarded only
        once the stream is open, so the page its agent opens for the
        user finds it subscribed.  Nothing is opened once the binding
        is gone (an unbind while the reopen of :meth:`dropped` waited)
        or the daemon is stopping; a stream opened for a binding that
        an unbind took away while it was connecting is closed again
        when idle.  A machine whose stream cannot be opened is logged
        and left alone: the tab's own stream found it reachable just
        now, and nothing of the user's is lost but the hand-off.
        """
        with self._lock:
            wanted = not self._closing and machine_name in self._bound.values()
        if not wanted:
            return
        try:
            stream = self._stream_for(machine_name)
        except MachineError as exc:
            logger.warning("the browser pages of %s are not shown here: %s", machine_name, exc)
            return
        self.close_if_idle(stream)

    def release(self, machine_name: str) -> None:
        """Close the stream of *machine_name* once no tab is bound to it and it is idle."""
        with self._lock:
            stream = self._streams.get(machine_name)
        if stream is not None:
            self.close_if_idle(stream)

    def route(self, cmd: dict[str, Any]) -> bool:
        """Send *cmd* to its machine when it addresses a remote terminal or page.

        A ``terminalOpen`` or ``browserOpen`` stamped ``machine`` opens
        the machine's stream when needed; the other terminal commands
        follow the terminal's stream, the other browser commands the
        machine of their namespaced tab id.

        Returns:
            True when the command was consumed (forwarded, refused
            with a ``terminalError``/``browserError`` to its sender, or
            ignored as another page's); False when this daemon handles
            it.
        """
        name = str(cmd["type"])
        if name not in VIEW_COMMANDS:
            return False
        tab_id = str(cmd.get("tab_id") or "")
        conn_id = str(cmd.get("connId") or "")
        frame = dict(cmd)
        if name in ("terminalOpen", "browserOpen"):
            machine_name = str(cmd.get("machine") or "")
            if not machine_name:
                return False
            # An idle stream closes under a command that reached it
            # just too late (``_book`` finds it retired): the next
            # look-up opens a fresh one.
            for _attempt in range(2):
                try:
                    stream = self._stream_for(machine_name)
                except MachineError as exc:
                    self._refuse(name, tab_id, conn_id, str(exc))
                    return True
                if self._forward(stream, name, tab_id, conn_id, cmd, frame):
                    break
            else:
                self._refuse(name, tab_id, conn_id, f"lost the connection to {machine_name}")
                return True
        elif name in TERMINAL_COMMANDS:
            found = self._terminal_stream(tab_id)
            if found is None:
                return False
            stream = found
            self._forward(stream, name, tab_id, conn_id, cmd, frame)
        else:
            remote_id, machine_name = split_tab_id(tab_id)
            if not machine_name:
                return False
            with self._lock:
                existing = self._streams.get(machine_name)
            if existing is None:
                return True  # the page went with its stream: nothing left to tell
            stream = existing
            frame["tab_id"] = remote_id
            self._forward(stream, name, remote_id, conn_id, cmd, frame)
        if name == "terminalClose":
            self.close_if_idle(stream)
        return True

    def _forward(
        self, stream: RemoteViewStream, name: str, tab_id: str, conn_id: str,
        cmd: dict[str, Any], frame: dict[str, Any],
    ) -> bool:
        """Book *cmd* on *stream* and send *frame*, as one step; False when not booked.

        A send that fails ends the stream, whose relay thread then runs
        :meth:`dropped` as lost: that tells the booked terminal owner,
        opener or page viewers (the sender among them) once.
        """
        with stream.send_lock:
            if not self._book(stream, name, tab_id, conn_id, cmd):
                return False
            try:
                stream.send(frame)
            except MachineError:
                pass
        return True

    def _book(
        self, stream: RemoteViewStream, name: str, tab_id: str, conn_id: str,
        cmd: dict[str, Any],
    ) -> bool:
        """Record what *cmd* means for *stream*; False when it must not be sent.

        A retired stream takes nothing.  A terminal's input, resize and
        close count only from the connection that opened it.  A
        ``browserViewport`` hiding the page is sent only once nobody on
        this daemon watches it: the remote tracks one viewer, this
        stream, whose last visible size it keeps.
        """
        with stream.lock:
            if stream.retired:
                return False
            if name == "terminalOpen":
                stream.terminals[tab_id] = conn_id
                stream.opening.add(tab_id)
            elif name in TERMINAL_COMMANDS:
                if stream.terminals.get(tab_id) != conn_id:
                    return False
                if name == "terminalClose":
                    del stream.terminals[tab_id]
                    stream.opening.discard(tab_id)
            elif name == "browserOpen":
                stream.openers.append(conn_id)
            elif name == "browserViewport":
                visible = bool(cmd.get("visible")) and int(cmd.get("width") or 0) > 0 and int(
                    cmd.get("height") or 0
                ) > 0
                watching = stream.viewers.setdefault(tab_id, set())
                if visible:
                    watching.add(conn_id)
                else:
                    watching.discard(conn_id)
                    if watching:
                        return False
        return True

    def _terminal_stream(self, tab_id: str) -> RemoteViewStream | None:
        """The stream whose terminal *tab_id* is (None when it is not a remote terminal)."""
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            with stream.lock:
                if tab_id in stream.terminals:
                    return stream
        return None

    def _stream_for(self, machine_name: str) -> RemoteViewStream:
        """The stream of the registered machine *machine_name*, opened when needed.

        Connecting blocks (up to the link's timeout) outside the lock;
        a stream another command opened meanwhile wins, and the one
        opened here is closed, as it is when the daemon stopped
        meanwhile (:meth:`close_all` leaves nothing behind).

        Raises:
            MachineError: When the machine is not registered,
                unreachable, or rejects the password, or the daemon is
                stopping.
        """
        with self._lock:
            stream = self._streams.get(machine_name)
            closing = self._closing
        if stream is not None:
            return stream
        if closing:
            raise MachineError(f"{machine_name}: this daemon is stopping")
        machine = registered_machine(machine_name)
        if machine is None:
            raise MachineError(f"{machine_name} is not a registered machine")
        stream = RemoteViewStream(machine, self)
        lost = False
        with self._lock:
            winner = self._streams.get(machine_name)
            closing = self._closing
            if winner is None and not closing:
                with stream.lock:
                    lost = stream.retired  # dropped before it could be registered
                if not lost:
                    self._streams[machine_name] = stream
        if winner is not None:
            stream.close()
            return winner
        if closing:
            stream.close()
            raise MachineError(f"{machine.name}: this daemon is stopping")
        if lost:
            raise MachineError(f"lost the connection to {machine.name}")
        stream.request_pages()
        return stream

    def close_if_idle(self, stream: RemoteViewStream) -> None:
        """Close *stream* once it carries no terminal, page or pending open.

        A stream whose machine a chat tab is bound to stays (see :meth:`bind`).
        """
        with self._lock:
            if stream.machine.name in self._bound.values():
                return
            with stream.lock:
                if stream.retired or stream.terminals or stream.browser_tabs or stream.openers:
                    return
                stream.retired = True
            if self._streams.get(stream.machine.name) is stream:
                del self._streams[stream.machine.name]
        stream.close()

    def _refuse(self, name: str, tab_id: str, conn_id: str, text: str) -> None:
        """Tell the sender of a command that could not be forwarded why."""
        if name in TERMINAL_COMMANDS:
            event = {"type": "terminalError", "tab_id": tab_id, "text": text}
        else:
            event = {"type": "browserError", "tab_id": "" if name == "browserOpen" else tab_id,
                     "text": text}
        self.deliver_to(conn_id, json.dumps(event))

    def replay_to(self, conn_id: str) -> None:
        """Tell the connection *conn_id* every remote browser page (its ``ready``).

        One ``browserTabs`` flagged ``remote`` lists the pages of every
        machine; the surface drops every remote tab not in it.  Copied
        and delivered under every stream's lock, so no close or open
        relayed meanwhile is told out of order.
        """
        with self._lock, contextlib.ExitStack() as held:
            tabs: list[dict[str, Any]] = []
            for stream in self._streams.values():
                held.enter_context(stream.lock)
                if not stream.retired:
                    tabs.extend(dict(t) for t in stream.browser_tabs.values())
            snapshot = {"type": "browserTabs", "remote": True, "tabs": tabs}
            self.deliver_to(conn_id, json.dumps(snapshot))

    def viewer_gone(self, conn_id: str) -> None:
        """The connection *conn_id* dropped: it watches no page, its shells get a grace period.

        Called from the server loop's connection-closed path: the work
        waits for each stream's ``send_lock`` (a command's send to the
        machine may be stalled for up to its timeout) and sends over the
        socket itself, so it runs on a thread of its own, like the grace
        timers it starts.
        """
        thread = threading.Thread(
            target=self._viewer_gone, args=(conn_id,),
            name=f"remote-views-gone-{conn_id}", daemon=True,
        )
        thread.start()

    def _viewer_gone(self, conn_id: str) -> None:
        """Do :meth:`viewer_gone`'s bookkeeping and sends, off the server loop."""
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            hidden: list[str] = []
            with stream.send_lock:
                with stream.lock:
                    for remote_id, watching in stream.viewers.items():
                        if conn_id in watching:
                            watching.discard(conn_id)
                            if not watching:
                                hidden.append(remote_id)
                    detached = [t for t, c in stream.terminals.items() if c == conn_id]
                for remote_id in hidden:
                    self._send_quietly(stream, {
                        "type": "browserViewport", "tab_id": remote_id, "visible": False,
                        "width": 0, "height": 0,
                    })
            for tab_id in detached:
                timer = threading.Timer(
                    GRACE_SECONDS, self._hang_up_detached, (stream, tab_id, conn_id),
                )
                timer.daemon = True
                timer.start()

    def _hang_up_detached(self, stream: RemoteViewStream, tab_id: str, conn_id: str) -> None:
        """Hang the shell of *tab_id* up unless another connection claimed it meanwhile.

        The check and the ``terminalClose`` are one step: a
        ``terminalOpen`` reclaiming the shell either went first (the
        shell is kept) or waits and starts a fresh shell.
        """
        with stream.send_lock:
            with stream.lock:
                if stream.retired or stream.terminals.get(tab_id) != conn_id:
                    return
                del stream.terminals[tab_id]
                stream.opening.discard(tab_id)
            self._send_quietly(stream, {"type": "terminalClose", "tab_id": tab_id})
        self.close_if_idle(stream)

    def _send_quietly(self, stream: RemoteViewStream, cmd: dict[str, Any]) -> None:
        """Send *cmd* down *stream*; a lost stream is reported by its relay thread."""
        try:
            stream.send(cmd)
        except MachineError:
            logger.debug("the stream of %s is gone", stream.machine.name, exc_info=True)

    def deliver(self, text: str) -> None:
        """Send one frame of a remote daemon to every client."""
        self._backend._printer._send_to_ws_clients(text, "")

    def deliver_to(self, conn_id: str, text: str) -> None:
        """Send one frame of a remote daemon to the connection *conn_id* only."""
        self._backend._printer._send_to_conn(conn_id, text)

    def dropped(self, stream: RemoteViewStream, lost: bool) -> None:
        """A stream ended: forget it, and when *lost* end its terminals, opens and pages here.

        A stream lost while a tab is still bound to its machine is
        opened again after :data:`REOPEN_DELAY`, if the tab is still
        bound then (the machine may have dropped the socket alone; a
        machine that is gone takes the tabs' own streams with it,
        which unbinds them).
        """
        name = stream.machine.name
        text = f"lost the connection to {name}"
        with self._lock, stream.lock:
            registered = self._streams.get(name) is stream
            if registered:
                del self._streams[name]
            still_bound = (
                lost and registered and not self._closing and name in self._bound.values()
            )
            stream.retired = True
            if lost:
                for tab_id, conn_id in stream.terminals.items():
                    self.deliver_to(conn_id, json.dumps({
                        "type": "terminalError", "tab_id": tab_id, "text": text, "machine": name,
                    }))
                for conn_id in stream.openers:
                    self.deliver_to(conn_id, json.dumps({
                        "type": "browserError", "tab_id": "", "text": text, "machine": name,
                    }))
                for remote_id in stream.browser_tabs:
                    self.deliver(json.dumps({
                        "type": "closeBrowserTab", "tab_id": namespaced(remote_id, name),
                        "machine": name,
                    }))
            stream.terminals.clear()
            stream.opening.clear()
            stream.openers.clear()
            stream.browser_tabs.clear()
            stream.viewers.clear()
        if still_bound:
            timer = threading.Timer(REOPEN_DELAY, self.open_for, (name,))
            timer.daemon = True
            timer.start()

    def close_all(self) -> None:
        """Close every stream (daemon shutdown); none is opened or registered from now on."""
        with self._lock:
            self._closing = True
            self._bound.clear()
            streams = list(self._streams.values())
            self._streams.clear()
            for stream in streams:
                with stream.lock:
                    stream.retired = True
        for stream in streams:
            stream.close()
