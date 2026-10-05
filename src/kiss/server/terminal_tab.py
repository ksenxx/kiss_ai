# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A shell on the daemon machine, streamed into a tab on every surface.

The remote webapp runs in a browser that may be far from the machine
hosting the daemon; "Terminal" in the composer's ... menu opens the
user's login shell on that machine inside a pseudo-terminal and shows
it in a content tab (xterm.js, ``media/terminalTab.js``), the way the
VS Code integrated terminal does.

Like the streamed browser (:mod:`kiss.server.browser_tab`) a terminal is
owned by the daemon, not by the surface that opened it:

* ``terminalOpen`` spawns the shell and announces ``openTerminalTab``
  on EVERY connected surface (``focus`` only for the one that asked);
* a surface that shows the tab sends ``terminalAttach`` and receives
  the scrollback kept so far, then every later chunk of output as
  ``terminalData`` (base64 of the raw bytes: xterm.js decodes UTF-8
  itself, so a multi-byte character split across two reads survives);
* ``terminalInput`` / ``terminalResize`` drive the pty; the most recent
  resize from any surface wins;
* when the shell exits, or a surface closes the tab by hand
  (``terminalClose``), ``closeTerminalTab`` closes it everywhere;
* the ``terminalTabs`` snapshot sent with ``ready`` lets a surface that
  (re)connects add the live terminals and drop the ones that died while
  it was away.

Every public method is thread-safe; output is pumped by one daemon
thread per terminal so the server loop never blocks on the pty.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import pwd
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kiss.core.printers.json_printer import JsonPrinter

try:  # Windows has no pseudo-terminals.
    import fcntl
    import struct
    import termios
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]
    struct = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Output kept per terminal for surfaces that attach later (a browser
# that reconnects, a second surface that switches to the tab).
BACKLOG_LIMIT = 256 * 1024
# A closed terminal's shell gets this long to honour SIGHUP before SIGKILL.
KILL_GRACE_SECONDS = 2.0
DEFAULT_COLS = 80
DEFAULT_ROWS = 24


@dataclass
class _Terminal:
    """One shell and the connections watching it."""

    tab_id: str
    title: str
    cwd: str
    proc: subprocess.Popen[bytes]
    master_fd: int
    backlog: bytearray = field(default_factory=bytearray)
    viewers: set[str] = field(default_factory=set)
    cols: int = DEFAULT_COLS
    rows: int = DEFAULT_ROWS
    # Set once ``closeTerminalTab`` went out (by exit or by request).
    finished: bool = False


def default_shell() -> list[str]:
    """The user's login shell as an argv (a login shell on macOS, as VS Code does)."""
    shell = os.environ.get("SHELL") or ""
    if not shell:
        with contextlib.suppress(KeyError, OSError):
            shell = pwd.getpwuid(os.getuid()).pw_shell
    if not shell or not os.path.exists(shell):
        shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    return [shell, "-l"] if sys.platform == "darwin" else [shell]


def _become_controlling_tty() -> None:
    """In the child: make the pty (already stdin) its controlling terminal.

    ``start_new_session`` made the child a session leader; a tty merely
    inherited as fd 0 does not become its controlling terminal by
    itself, and a shell without one runs without job control.
    """
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def _set_winsize(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _clamp(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(2, min(n, 1000))


class TerminalTabService:
    """Spawn shells in pseudo-terminals and stream them to the surfaces."""

    def __init__(self, printer: JsonPrinter) -> None:
        self._printer = printer
        self._lock = threading.Lock()
        self._terms: dict[str, _Terminal] = {}
        self._count = 0
        self._closed = False

    # ------------------------------------------------------------------
    # Public API (any thread)
    # ------------------------------------------------------------------

    def open(self, work_dir: str, conn_id: str) -> None:
        """Start a shell in *work_dir* and announce its tab everywhere.

        The requesting connection (*conn_id*) gets ``focus: true`` so it
        switches to the tab; every other surface adds it unfocused.  A
        failure (no pty support, no shell) is reported as
        ``terminalError`` to the requester alone.
        """
        if termios is None:
            self._emit(
                {"type": "terminalError", "tab_id": "",
                 "text": "A terminal needs a Unix daemon: this daemon runs on Windows."},
                conn_id,
            )
            return
        cwd = work_dir if work_dir and os.path.isdir(work_dir) else os.path.expanduser("~")
        try:
            term = self._spawn(cwd)
        except OSError as exc:
            logger.warning("terminal: cannot start a shell: %s", exc)
            self._emit(
                {"type": "terminalError", "tab_id": "", "text": f"The shell could not start: {exc}"},
                conn_id,
            )
            return
        with self._lock:
            if self._closed:
                self._terminate(term)
                return
            self._terms[term.tab_id] = term
        threading.Thread(
            target=self._pump, args=(term,), name=f"terminal-{term.tab_id[:8]}", daemon=True
        ).start()
        # Everyone gets the tab; only the surface that asked switches to it.
        self._emit(self._open_event(term, focus=False))
        if conn_id:
            self._emit(self._open_event(term, focus=True), conn_id)

    def attach(self, tab_id: str, conn_id: str, cols: Any, rows: Any) -> None:
        """Start streaming *tab_id* to *conn_id*, replaying the backlog first.

        The replay and the registration happen under one lock with the
        output pump, so the connection sees every byte exactly once and
        in order.  The surface's size becomes the pty's size.
        """
        with self._lock:
            term = self._terms.get(tab_id)
            if term is None or term.finished:
                return
            term.viewers.add(conn_id)
            self._emit(
                {"type": "terminalData", "tab_id": tab_id,
                 "data": base64.b64encode(bytes(term.backlog)).decode("ascii")},
                conn_id,
            )
            self._resize_locked(term, cols, rows)

    def input(self, tab_id: str, data: str, binary: bool = False) -> None:
        """Write what the user typed (or pasted) to the shell."""
        with self._lock:
            term = self._terms.get(tab_id)
            if term is None or term.finished:
                return
            fd = term.master_fd
        raw = data.encode("latin-1", "ignore") if binary else data.encode("utf-8")
        try:
            while raw:
                raw = raw[os.write(fd, raw):]
        except OSError as exc:
            logger.debug("terminal %s: write failed: %s", tab_id, exc)

    def resize(self, tab_id: str, cols: Any, rows: Any) -> None:
        """Resize the pty to the viewing surface's columns and rows."""
        with self._lock:
            term = self._terms.get(tab_id)
            if term is not None and not term.finished:
                self._resize_locked(term, cols, rows)

    def close(self, tab_id: str) -> None:
        """Close the tab everywhere and end its shell (SIGHUP, then SIGKILL)."""
        with self._lock:
            term = self._terms.get(tab_id)
            if term is None or term.finished:
                return
            self._finish_locked(term)
        threading.Thread(
            target=self._terminate, args=(term,), name="terminal-kill", daemon=True
        ).start()

    def viewer_gone(self, conn_id: str) -> None:
        """Stop streaming to *conn_id* (the client disconnected)."""
        with self._lock:
            for term in self._terms.values():
                term.viewers.discard(conn_id)

    def open_events(self) -> list[dict[str, Any]]:
        """One unfocused ``openTerminalTab`` event per live terminal."""
        with self._lock:
            return [self._open_event(t, focus=False) for t in self._terms.values() if not t.finished]

    def snapshot_event(self) -> dict[str, Any]:
        """The ``terminalTabs`` event a connecting client reconciles its tabs against."""
        return {"type": "terminalTabs", "tabs": self.open_events()}

    def shutdown(self) -> None:
        """End every shell; the daemon is stopping."""
        with self._lock:
            self._closed = True
            terms = [t for t in self._terms.values() if not t.finished]
            for term in terms:
                term.finished = True
        for term in terms:
            self._terminate(term)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self, cwd: str) -> _Terminal:
        master_fd, slave_fd = os.openpty()
        env = dict(os.environ)
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env.pop("COLUMNS", None)
        env.pop("LINES", None)
        argv = default_shell()
        try:
            _set_winsize(master_fd, DEFAULT_COLS, DEFAULT_ROWS)
            proc = subprocess.Popen(  # noqa: S603 — the user's own shell
                argv,
                cwd=cwd,
                env=env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                preexec_fn=_become_controlling_tty,  # noqa: PLW1509 — two syscalls, no locks
                close_fds=True,
            )
        except OSError:
            os.close(master_fd)
            os.close(slave_fd)
            raise
        # The child holds its own copy; ours would keep the pty from
        # reporting EOF when the shell exits.
        os.close(slave_fd)
        with self._lock:
            self._count += 1
            n = self._count
        name = os.path.basename(cwd.rstrip(os.sep)) or cwd
        title = f"Terminal: {name}" if n == 1 else f"Terminal {n}: {name}"
        return _Terminal(
            tab_id=uuid.uuid4().hex, title=title, cwd=cwd, proc=proc, master_fd=master_fd
        )

    def _pump(self, term: _Terminal) -> None:
        """Copy the shell's output to the attached surfaces until it exits."""
        while True:
            try:
                data = os.read(term.master_fd, 65536)
            except OSError:  # EIO on Linux once the last slave fd closed
                data = b""
            if not data:
                break
            with self._lock:
                term.backlog += data
                if len(term.backlog) > BACKLOG_LIMIT:
                    cut = len(term.backlog) - BACKLOG_LIMIT
                    nl = term.backlog.find(b"\n", cut)
                    del term.backlog[: nl + 1 if 0 <= nl < cut + 4096 else cut]
                viewers = list(term.viewers)
            payload = base64.b64encode(data).decode("ascii")
            for conn_id in viewers:
                self._emit({"type": "terminalData", "tab_id": term.tab_id, "data": payload}, conn_id)
        with self._lock:
            self._finish_locked(term)
        with contextlib.suppress(OSError):
            os.close(term.master_fd)
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            term.proc.wait(timeout=KILL_GRACE_SECONDS)

    def _finish_locked(self, term: _Terminal) -> None:
        """Drop *term* and close its tab on every surface (idempotent)."""
        if term.finished:
            return
        term.finished = True
        self._terms.pop(term.tab_id, None)
        self._emit({"type": "closeTerminalTab", "tab_id": term.tab_id})

    def _terminate(self, term: _Terminal) -> None:
        """Hang up the shell's process group; kill it if it lingers."""
        if term.proc.poll() is not None:
            return
        for sig in (signal.SIGHUP, signal.SIGKILL):
            with contextlib.suppress(OSError):
                os.killpg(term.proc.pid, sig)
            with contextlib.suppress(subprocess.TimeoutExpired):
                term.proc.wait(timeout=KILL_GRACE_SECONDS)
                return

    def _resize_locked(self, term: _Terminal, cols: Any, rows: Any) -> None:
        cols, rows = _clamp(cols, term.cols), _clamp(rows, term.rows)
        if (cols, rows) == (term.cols, term.rows):
            return
        term.cols, term.rows = cols, rows
        with contextlib.suppress(OSError):
            _set_winsize(term.master_fd, cols, rows)

    def _open_event(self, term: _Terminal, *, focus: bool) -> dict[str, Any]:
        return {
            "type": "openTerminalTab",
            "tab_id": term.tab_id,
            "title": term.title,
            "cwd": term.cwd,
            "focus": focus,
        }

    def _emit(self, event: dict[str, Any], conn_id: str = "") -> None:
        """Send *event* to one connection (*conn_id*) or to every client."""
        if conn_id:
            event["connId"] = conn_id
        else:
            event["tabId"] = ""
        self._printer.broadcast(event)
