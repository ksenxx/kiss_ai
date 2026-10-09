# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A shell on the daemon's machine, streamed into a remote-webapp tab.

The remote webapp's "..." menu has a "Terminal" item that opens a tab
holding an xterm.js terminal (``media/terminalTab.js``).  This module
is the daemon half: one pseudo-terminal running the user's shell per
terminal tab.

Protocol (every command is a remote-only entry of the API catalog in
:mod:`kiss.server.sorcar`; every event is stamped with the owning
connection's ``connId`` so no other surface sees it):

* ``terminalOpen {tab_id, cols, rows}`` starts a shell for the tab in
  the command's work dir, or re-attaches a shell that outlived a
  dropped connection.  Answered by ``terminalOpened {tab_id, shell,
  cwd, attached}``, or ``terminalError {tab_id, text}``.
* ``terminalInput {tab_id, data}`` writes keystrokes/pastes to the pty.
* ``terminalResize {tab_id, cols, rows}`` sets the pty window size.
* ``terminalClose {tab_id}`` hangs the shell up.
* ``terminalData {tab_id, data}`` streams the shell's output;
  ``terminalExit {tab_id, code}`` reports the shell's end.

A tab is identified by a client-generated ``tab_id``.  Sessions are
bound to the WebSocket connection that opened them: when that
connection drops, the shell keeps running for ``GRACE_SECONDS`` so a
page that reconnects (a phone waking up, a network blip) gets its
shell back by sending ``terminalOpen`` with the same ``tab_id``.
Windows has no pty: there ``terminalOpen`` answers with an error.

The shell starts with the start-up files of :mod:`kiss.server.terminal_rc`
(bash: ``--rcfile``; zsh: ``ZDOTDIR``), which run the user's own
dotfiles and then write every command to the history file as it is
entered, so Up/Down in a new tab recall the commands of every earlier
one.
"""

from __future__ import annotations

import codecs
import logging
import os
import queue
import selectors
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

if sys.platform != "win32":
    import fcntl
    import pty
    import termios

from kiss.core.processes import find_bash
from kiss.server.terminal_rc import with_persistent_history

logger = logging.getLogger(__name__)

# A shell whose connection dropped is kept this long for a re-attach.
GRACE_SECONDS = 60.0
# After a hang-up the shell gets this long to exit before SIGKILL.
_HANGUP_TIMEOUT = 2.0
_READ_CHUNK = 65536
_MAX_DIM = 1000


def default_shell() -> list[str]:
    """Return the argv of the shell a terminal tab runs.

    ``$SHELL`` when it names an executable, else ``bash`` from ``PATH``,
    else ``/bin/sh``.  On macOS the shell is a login shell (``-l``), as
    in Terminal.app and VS Code, so the user's PATH set-up applies.
    """
    shell = os.environ.get("SHELL", "")
    if not (shell and os.access(shell, os.X_OK)):
        shell = find_bash() or "/bin/sh"
    return [shell, "-l"] if sys.platform == "darwin" else [shell]


def _clamp_dim(value: Any, fallback: int) -> int:
    """Return *value* as a terminal dimension in ``1..1000``, else *fallback*."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return max(1, min(_MAX_DIM, int(value)))


def _winsize(rows: int, cols: int) -> bytes:
    """Pack *rows* x *cols* as the ``struct winsize`` ``TIOCSWINSZ`` takes."""
    return struct.pack("HHHH", rows, cols, 0, 0)


@dataclass
class _Session:
    """One shell: its pty master, its process and the connection watching it."""

    tab_id: str
    conn_id: str
    pid: int
    fd: int
    shell: str
    cwd: str
    decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")(errors="replace")
    )
    writes: queue.Queue[bytes | None] = field(default_factory=queue.Queue)
    # ``time.monotonic()`` when the owning connection dropped; ``None``
    # while a connection is attached.
    detached_at: float | None = None
    # Hangs the shell up ``GRACE_SECONDS`` after the disconnect;
    # cancelled when the page re-attaches or the shell is hung up.
    grace_timer: threading.Timer | None = None
    hung_up: bool = False
    # Set under the service lock in the same step as the ``waitpid``
    # that collected the shell, so no signal is ever sent to a pid the
    # kernel may already have handed to another process.
    reaped: bool = False


class TerminalService:
    """The daemon's terminal sessions: one pty shell per terminal tab.

    Thread-safe; the public methods may be called from the server loop
    or any executor thread.  Output is delivered through
    ``printer.broadcast`` events stamped with the owning ``connId``.
    """

    def __init__(self, printer: Any) -> None:
        """Create an idle service that emits events through *printer*."""
        self._printer = printer
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def open(
        self, tab_id: str, conn_id: str, work_dir: str, cols: Any, rows: Any,
    ) -> None:
        """Start the shell of *tab_id* for *conn_id*, or re-attach a live one.

        Args:
            tab_id: The terminal tab's id (chosen by the client).
            conn_id: The WebSocket connection that owns the tab.
            work_dir: The directory the shell starts in.
            cols: Initial width in cells (``1..1000``; default 80).
            rows: Initial height in cells (``1..1000``; default 24).
        """
        cols = _clamp_dim(cols, 80)
        rows = _clamp_dim(rows, 24)
        with self._lock:
            session = self._sessions.get(tab_id)
            if session is not None:
                # A page that reconnected within the grace period (or
                # re-sent its open): the shell carries on where it was.
                session.conn_id = conn_id
                self._cancel_grace(session)
                self._set_winsize(session, cols, rows)
                self._emit_opened(session, attached=True)
                return
            if sys.platform == "win32":  # pragma: no cover — no pty there
                self._emit(
                    {
                        "type": "terminalError",
                        "tab_id": tab_id,
                        "text": "The terminal needs a pseudo-terminal, which Windows has none of.",
                    },
                    conn_id,
                )
                return
            try:
                session = self._spawn(tab_id, conn_id, work_dir, cols, rows)
            except Exception as exc:  # noqa: BLE001 — reported to the tab
                logger.warning("terminal spawn failed: %s", exc, exc_info=True)
                self._emit(
                    {
                        "type": "terminalError",
                        "tab_id": tab_id,
                        "text": f"The shell could not be started: {exc}",
                    },
                    conn_id,
                )
                return
            self._sessions[tab_id] = session
            # Announced before the reader runs, so a shell that exits
            # at once still reports ``terminalOpened`` before its
            # ``terminalExit``.
            self._emit_opened(session, attached=False)
        threading.Thread(
            target=self._write_loop, args=(session,), daemon=True,
            name=f"terminal-write-{tab_id[:8]}",
        ).start()
        threading.Thread(
            target=self._pump, args=(session,), daemon=True,
            name=f"terminal-read-{tab_id[:8]}",
        ).start()

    def input(self, tab_id: str, conn_id: str, data: Any) -> None:
        """Queue *data* (the keystrokes xterm.js produced) for the shell.

        Args:
            tab_id: The terminal tab.
            conn_id: The sending connection; must own the tab.
            data: The text to write (a non-string is ignored).
        """
        if not isinstance(data, str) or not data:
            return
        with self._lock:
            session = self._owned(tab_id, conn_id)
            if session is not None:
                session.writes.put(data.encode("utf-8", errors="surrogateescape"))

    def resize(self, tab_id: str, conn_id: str, cols: Any, rows: Any) -> None:
        """Set the pty window size (the shell gets ``SIGWINCH``).

        Args:
            tab_id: The terminal tab.
            conn_id: The sending connection; must own the tab.
            cols: New width in cells.
            rows: New height in cells.
        """
        # Under the lock: the reader deregisters a finished session under
        # it BEFORE the write loop closes the master, so the ioctl can
        # never land on a descriptor number the kernel has since reused.
        with self._lock:
            session = self._owned(tab_id, conn_id)
            if session is not None:
                self._set_winsize(session, _clamp_dim(cols, 80), _clamp_dim(rows, 24))

    def close(self, tab_id: str, conn_id: str = "") -> None:
        """Hang the shell of *tab_id* up (the tab was closed).

        The shell's process group gets ``SIGHUP`` as on a closed
        terminal window; one that is still alive ``_HANGUP_TIMEOUT``
        later is killed.  The reader thread reports the exit.

        Args:
            tab_id: The terminal tab.
            conn_id: The sending connection; when non-empty it must own
                the tab.
        """
        with self._lock:
            session = self._sessions.get(tab_id)
            if session is None or (conn_id and session.conn_id != conn_id):
                return
            self._hang_up(session)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def viewer_gone(self, conn_id: str) -> None:
        """The connection *conn_id* dropped: keep its shells for a re-attach.

        Each shell owned by the connection is detached and hung up
        ``GRACE_SECONDS`` later unless a reconnecting page claims it
        with ``terminalOpen`` first.
        """
        now = time.monotonic()
        with self._lock:
            for session in self._sessions.values():
                if session.conn_id != conn_id:
                    continue
                self._cancel_grace(session)
                session.detached_at = now
                session.grace_timer = threading.Timer(
                    GRACE_SECONDS, self._expire_detached, [session],
                )
                session.grace_timer.daemon = True
                session.grace_timer.start()

    def shutdown(self) -> None:
        """Hang up every shell (the daemon is stopping)."""
        with self._lock:
            sessions = list(self._sessions.values())
            for session in sessions:
                self._hang_up(session)
        deadline = time.monotonic() + _HANGUP_TIMEOUT + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                if not any(s.tab_id in self._sessions for s in sessions):
                    return
            time.sleep(0.05)

    def session_count(self) -> int:
        """Return how many shells are running (tests and diagnostics)."""
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _owned(self, tab_id: str, conn_id: str) -> _Session | None:
        """Return the session of *tab_id* if *conn_id* owns it (caller holds the lock)."""
        session = self._sessions.get(tab_id)
        if session is None or session.conn_id != conn_id:
            return None
        return session

    def _spawn(
        self, tab_id: str, conn_id: str, work_dir: str, cols: int, rows: int,
    ) -> _Session:
        """Fork the shell on a new pty; return its session (caller holds the lock)."""
        shell = default_shell()
        # The shell's command history outlives the tab (and reaches
        # the other tabs) through start-up files of our own.
        argv, extra_env = with_persistent_history(shell)
        cwd = work_dir if work_dir and os.path.isdir(work_dir) else os.path.expanduser("~")
        env = dict(os.environ)
        env.update(extra_env)
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover — the child execs or dies
            # Only exec-safe work here: the parent is multi-threaded.
            try:
                os.chdir(cwd)
                os.execvpe(argv[0], argv, env)
            except BaseException:  # noqa: BLE001
                pass
            os._exit(127)
        # ``forkpty`` hands back an inheritable master: without this a
        # later shell would hold every earlier terminal's master open.
        os.set_inheritable(fd, False)
        session = _Session(
            tab_id=tab_id, conn_id=conn_id, pid=pid, fd=fd,
            shell=argv[0], cwd=cwd,
        )
        # The initial size is set here, on the master, not by the child
        # before its exec: a child scheduled late (a loaded machine) would
        # apply the opening size AFTER a resize or re-attach the parent
        # had already applied, and the shell would start at the old size.
        self._set_winsize(session, cols, rows)
        return session

    def _set_winsize(self, session: _Session, cols: int, rows: int) -> None:
        try:
            fcntl.ioctl(session.fd, termios.TIOCSWINSZ, _winsize(rows, cols))
        except OSError:  # the shell just exited; the reader reports it
            pass

    def _hang_up(self, session: _Session) -> None:
        """Signal the shell to exit (caller holds the lock)."""
        if session.hung_up or session.reaped:
            return
        session.hung_up = True
        self._cancel_grace(session)
        try:
            os.killpg(os.getpgid(session.pid), signal.SIGHUP)
        except OSError:
            pass
        timer = threading.Timer(_HANGUP_TIMEOUT, self._kill_if_alive, [session])
        timer.daemon = True
        timer.start()

    def _kill_if_alive(self, session: _Session) -> None:
        with self._lock:
            self._kill_locked(session)

    def _kill_locked(self, session: _Session) -> None:
        """SIGKILL the shell unless it was reaped (caller holds the lock)."""
        if session.reaped:
            return
        try:
            os.kill(session.pid, signal.SIGKILL)
        except OSError:
            pass

    def _cancel_grace(self, session: _Session) -> None:
        """Mark *session* attached and stop its grace timer (caller holds the lock)."""
        session.detached_at = None
        if session.grace_timer is not None:
            session.grace_timer.cancel()
            session.grace_timer = None

    def _expire_detached(self, session: _Session) -> None:
        # Only the timer of the latest disconnect may hang up: a timer
        # that was already firing when the page re-attached (cancel()
        # cannot stop it) must not expire a later disconnect's grace.
        with self._lock:
            if (
                session.grace_timer is threading.current_thread()
                and session.tab_id in self._sessions
            ):
                self._hang_up(session)

    def _write_loop(self, session: _Session) -> None:
        """Write queued input to the pty in order, then close the master.

        A blocked write never stalls the server loop, and the master is
        closed HERE, after the reader's end-of-session sentinel, so no
        write can ever land on a descriptor number the kernel has since
        reused for something else.
        """
        broken = False
        while True:
            data = session.writes.get()
            if data is None:
                break
            if broken:
                continue
            try:
                while data:
                    n = os.write(session.fd, data)
                    data = data[n:]
            except OSError:
                broken = True
        try:
            os.close(session.fd)
        except OSError:
            pass

    def _pump(self, session: _Session) -> None:
        """Stream the shell's output to its connection until it exits.

        The shell's exit is polled with ``waitpid`` on every turn, not
        only at EOF: a background child that inherited the pty keeps
        the slave side open (no EOF) and may keep printing.
        """
        selector = selectors.DefaultSelector()
        selector.register(session.fd, selectors.EVENT_READ)
        code: int | None = None
        while code is None:
            if selector.select(0.5):
                data = self._read(session)
                if not data:
                    # EOF: every handle on the slave side is closed, so
                    # the shell is gone or on its way out.
                    code = self._reap(session, _HANGUP_TIMEOUT)
                    break
                self._emit_data(session, data)
            code = self._reap(session, 0.0)
        # Output written just before the exit may still sit in the pty
        # buffer; a child that keeps printing after the shell is gone
        # does not keep the tab alive.
        drain_until = time.monotonic() + 0.2
        while time.monotonic() < drain_until and selector.select(0):
            data = self._read(session)
            if not data:
                break
            self._emit_data(session, data)
        selector.close()
        with self._lock:
            self._sessions.pop(session.tab_id, None)
        session.writes.put(None)
        self._emit(
            {"type": "terminalExit", "tab_id": session.tab_id, "code": code},
            session.conn_id,
        )

    def _read(self, session: _Session) -> bytes:
        try:
            return os.read(session.fd, _READ_CHUNK)
        except OSError:  # EIO: the slave side is closed (Linux reports EOF so)
            return b""

    def _reap(self, session: _Session, timeout: float) -> int | None:
        """Collect the shell's exit code, waiting at most *timeout* seconds.

        A shell that has not exited by the deadline (it closed its tty
        but lingers) is killed; ``None`` means still running when
        *timeout* is zero.  Each ``waitpid`` runs under the lock
        together with the ``reaped`` flag it sets, so the hang-up and
        kill paths (which check that flag under the same lock) never
        signal a pid the kernel has recycled.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                try:
                    wpid, status = os.waitpid(session.pid, os.WNOHANG)
                except ChildProcessError:
                    wpid, status = session.pid, 0
                if wpid:
                    session.reaped = True
                    return os.waitstatus_to_exitcode(status)
                if timeout and time.monotonic() >= deadline:
                    self._kill_locked(session)
                    deadline = time.monotonic() + _HANGUP_TIMEOUT
            if not timeout:
                return None
            time.sleep(0.02)

    def _emit_data(self, session: _Session, data: bytes) -> None:
        text = session.decoder.decode(data)
        if text:
            self._emit(
                {"type": "terminalData", "tab_id": session.tab_id, "data": text},
                session.conn_id,
            )

    def _emit_opened(self, session: _Session, attached: bool) -> None:
        self._emit(
            {
                "type": "terminalOpened",
                "tab_id": session.tab_id,
                "shell": os.path.basename(session.shell),
                "cwd": session.cwd,
                "attached": attached,
            },
            session.conn_id,
        )

    def _emit(self, event: dict[str, Any], conn_id: str) -> None:
        """Deliver *event* to the connection *conn_id* only."""
        event["connId"] = conn_id
        self._printer.broadcast(event)


__all__ = ["GRACE_SECONDS", "TerminalService", "default_shell"]
