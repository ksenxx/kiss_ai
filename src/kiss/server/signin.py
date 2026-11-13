# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The web app's sign-in window, driven by a brand's ``$KISS_HOME/signin`` hook.

A brand can want the user signed in to a service of its own (Seamless
Loop's hub, say).  It installs an executable ``$KISS_HOME/signin``, and
the daemon drives it over this protocol, one JSON object per stdout
line::

    signin status
        {"logged_in": bool, "email": str, "title": str,
         "providers": [{"id": str, "label": str}, ...]}

    signin login <provider id>
        first line  {"url": str}            the page to sign in on
        ... waits for the sign-in to end: the provider's redirect to the
            hook's own loopback port, or the redirect URL / code the user
            pasted into the window, which the daemon writes on its stdin
        last line   {"logged_in": true, "email": str}   and exit 0, or
                    {"error": str}                       and exit 1

Every page that sends ``ready`` while ``status`` says the user is not
logged in gets a ``signin`` event (``title``, ``providers``) once its
tabs are restored, and shows the window: one button per provider.  The
``status`` answer is cached for :data:`STATUS_CACHE_TTL` seconds and
dropped by a ``signinDone``, so a reconnect storm (every window of a
restarted daemon) runs the hook once; the answer is handed to the page
under the same lock that drops it, so a page never gets a ``signin``
after the ``signinDone`` that closed its window.  A click sends ``signinLogin``;
the daemon starts ``login`` and answers the window with ``signinUrl``,
then, when the hook exits, ``signinDone`` (broadcast, so the window
closes in every browser of this daemon) or ``signinError``.
``signinPaste`` forwards a pasted redirect URL or code to the hook's
stdin, for a browser on another machine, whose redirect to the hook's
loopback port cannot land; ``signinCancel`` ends the login.

Without the hook, or with a broken one (not executable, non-zero exit,
bad JSON, slow), the web app shows nothing: a brand's sign-in must never
lock the user out of their own daemon.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

if sys.platform != "win32":
    import fcntl
    import termios
else:  # pragma: no cover — Windows-only branch
    import _winapi
    import msvcrt

from kiss.core.config import kiss_home
from kiss.core.locked import locked_call
from kiss.core.processes import (
    exited_unreaped,
    kill_process_group,
    popen_process_group,
    run_process_group,
    terminate_process_group,
)

logger = logging.getLogger(__name__)

SIGNIN_HOOK = "signin"
"""Basename, under ``$KISS_HOME``, of the brand's sign-in hook."""

STATUS_TIMEOUT = 10.0
"""Seconds ``signin status`` may take before the daemon gives up on it."""

STATUS_CACHE_TTL = 60.0
"""Seconds one ``signin status`` answer serves every ``ready``: the hook is a process
(a Python interpreter for Seamless Loop) that every page load and reconnect would
otherwise run again, N at once after a daemon restart.  A sign-in through the window
drops the answer at once (:func:`_signed_in`); one made elsewhere (the brand's CLI)
shows after the TTL."""

HOOK_END_TIMEOUT = 2.0
"""Seconds a replaced or cancelled ``login`` hook gets to exit on SIGTERM before SIGKILL."""

HOOK_EXIT_POLL = 0.05
"""Seconds between the relay's looks, while the ``login`` hook's stdout is silent, at what
the pipe holds and whether the hook is still running (:func:`_hook_lines`)."""

PASTE_LIMIT = 4096
"""Characters of a paste handed to the hook: enough for any redirect URL, less than a pipe holds."""

ENDED_ELSEWHERE = "The sign-in was ended from another window: click a button to start it again."
"""The ``signinError`` of a window whose login another window's click or "Not now" ended."""

Reply = Callable[[dict[str, Any], bool], None]
"""Delivers an event to the window that clicked (``False``) or to every window (``True``)."""

StatusReply = Callable[[dict[str, Any]], None]
"""Delivers the ``signin`` event to the window whose ``ready`` asked for it."""


def signin_hook() -> Path | None:
    """The sign-in hook's path when ``$KISS_HOME/signin`` is an executable file, else ``None``."""
    hook = kiss_home() / SIGNIN_HOOK
    if hook.is_file() and os.access(hook, os.X_OK):
        return hook
    return None


def _hook_env() -> dict[str, str]:
    """The hook's environment: the daemon's, with ``KISS_HOME`` spelled out."""
    return {**os.environ, "KISS_HOME": str(kiss_home())}


def _parse(line: str) -> dict[str, Any]:
    """The JSON object on *line*, or ``{}`` when it holds none."""
    try:
        value = json.loads(line)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _unread_bytes(fd: int) -> int:
    """How many bytes the pipe read through *fd* holds that have not been read yet (0 at EOF)."""
    if sys.platform == "win32":  # pragma: no cover — Windows-only branch
        try:
            return int(_winapi.PeekNamedPipe(msvcrt.get_osfhandle(fd))[0])
        except BrokenPipeError:  # every write end closed and nothing left to read
            return 0
    return int(struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0\0\0\0"))[0])


def _hook_lines(proc: subprocess.Popen[str]) -> Iterator[str]:
    """The lines of *proc*'s stdout until, once *proc* has exited, what it held then.

    EOF cannot bound the read: a helper the hook started with its stdout
    inherited (a browser opener, say) keeps the pipe's write end open
    for as long as it lives, which may be long after the hook printed
    its last line and exited.  So the pipe is looked at, every
    :data:`HOOK_EXIT_POLL` seconds while it is silent, for what it holds
    and for the hook's exit (the same on every platform: Windows has no
    readiness wait for a pipe), and at the exit exactly the bytes the
    pipe holds then (everything the hook wrote is among them) are read:
    a helper that keeps writing cannot keep the read going either.  The
    helper is left alone: it may be the user's browser.  The exit is
    seen without reaping the hook: reaping is for whoever holds the
    login's lock (:meth:`SigninLogin._relay`, :meth:`SigninLogin._end_locked`),
    so no pid is freed between a look at the hook and a signal to its group.
    """
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    pending = b""
    unread = None  # bytes the pipe held at the hook's exit that are still to read
    while unread is None or unread > 0:
        held = unread
        if held is None:
            exited = exited_unreaped(proc)
            held = _unread_bytes(fd)
            if exited:
                unread = held
                continue
            if not held:
                time.sleep(HOOK_EXIT_POLL)
                continue
        chunk = os.read(fd, min(held, 65536))
        if unread is not None:
            unread -= len(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for line in lines:
            yield line.decode("utf-8", errors="replace")
    if pending:
        yield pending.decode("utf-8", errors="replace")


_status_lock = threading.RLock()
_status_cache: tuple[float, dict[str, Any] | None] | None = None
"""``(expires_at, event)`` of the last ``signin status`` run, under ``_status_lock``."""


def send_signin(reply: StatusReply) -> None:
    """Hand *reply* the ``signin`` event when the page must sign in.

    Runs ``signin status``, at most once per :data:`STATUS_CACHE_TTL`
    seconds (concurrent callers wait for the one run and share its
    answer), and calls *reply* with the answer, a copy, while still
    holding the status lock: a sign-in that succeeds meanwhile waits for
    that lock to drop the answer and announce ``signinDone``
    (:func:`_signed_in`), so an answer computed before the sign-in is
    never delivered after it.  Nothing without a hook, when the hook
    says the user is logged in, when it offers no provider, and when it
    is broken (a non-zero exit, no JSON, or more than
    :data:`STATUS_TIMEOUT` seconds: logged, never shown).  Blocking:
    callers run it off the event loop.

    Args:
        reply: Delivers the event to the page; it only queues a frame.
    """
    hook = signin_hook()
    if hook is None:
        return
    locked_call(_status_lock, _send_cached_status, hook, reply)


def _signed_in(reply: Reply, email: str) -> None:
    """Drop the cached answer and tell every window ``signinDone`` (lock held).

    One step with :func:`_send_cached_status`'s delivery, so no window
    gets a ``signin`` after the ``signinDone`` that closed its window.
    """
    global _status_cache
    _status_cache = None
    reply({"type": "signinDone", "email": email}, True)


def _send_cached_status(hook: Path, reply: StatusReply) -> None:
    """Deliver the cached ``signin`` event of *hook*, run anew once expired (lock held)."""
    global _status_cache
    now = time.monotonic()
    if _status_cache is None or now >= _status_cache[0]:
        _status_cache = (now + STATUS_CACHE_TTL, _query_status(hook))
    if _status_cache[1] is not None:
        reply(dict(_status_cache[1]))


def _query_status(hook: Path) -> dict[str, Any] | None:
    """Run ``signin status`` of *hook*; the ``signin`` event it calls for, else ``None``.

    In a process group of its own, ended whole at :data:`STATUS_TIMEOUT`:
    a child the hook started must not outlive it.
    """
    try:
        run = run_process_group([str(hook), "status"], STATUS_TIMEOUT, env=_hook_env())
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("sign-in hook %s failed: %s", hook, exc)
        return None
    status = _parse(run.stdout.strip().splitlines()[-1] if run.stdout.strip() else "")
    if run.returncode != 0 or not status:
        logger.warning(
            "sign-in hook %s: status exited %s with %r",
            hook,
            run.returncode,
            run.stdout[-200:],
        )
        return None
    if status.get("logged_in"):
        return None
    listed = status.get("providers")
    providers = [
        {"id": str(p["id"]), "label": str(p.get("label") or p["id"])}
        for p in (listed if isinstance(listed, list) else [])
        if isinstance(p, dict) and p.get("id")
    ]
    if not providers:
        return None
    return {
        "type": "signin",
        "title": str(status.get("title") or "Sign in"),
        "providers": providers,
    }


class SigninLogin:
    """The daemon's one running ``signin login``: start it, feed it a paste, end it.

    One login at a time: a new :meth:`start` ends the running one, so a
    second click (another window's, a retry, the other provider) never
    leaves two hooks waiting on two loopback ports.  A window whose
    login another window ended is told so (:data:`ENDED_ELSEWHERE`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._owner = ""
        """The window whose :meth:`start` ran ``_proc`` (``""``: none to tell), with the
        provider it clicked and its reply; all under ``_lock``."""
        self._provider = ""
        self._reply: Reply | None = None
        self._ending: list[threading.Thread] = []
        """The :meth:`_end` threads of ended hooks not yet gone, under ``_lock``."""

    def start(self, provider: str, reply: Reply, owner: str = "") -> None:
        """Run ``signin login`` *provider*; relay its lines to *reply* from a thread.

        The window that clicked gets ``signinUrl`` (``provider``, ``url``)
        as soon as the hook prints the page to open, then ``signinError``
        when the hook fails; ``signinDone`` (``email``) goes to every
        window.  A hook that cannot start, or whose relay thread cannot,
        or that prints no URL first, is a ``signinError`` too.

        Args:
            provider: The provider id from the ``signin`` event.
            reply: Delivers an event to the clicking window
                (``reply(event, False)``) or to every window
                (``reply(event, True)``).
            owner: The clicking window (its connection id): the one
                told when another window ends this login, and the one
                whose own running login this click ends silently.
        """
        hook = signin_hook()
        if hook is None:
            reply({"type": "signinError", "provider": provider, "error": "no sign-in hook"}, False)
            return
        with self._lock:
            self._end_locked(owner)
            try:
                # In a group of its own: a helper the hook starts (and
                # that inherits its stdout) is ended with it, or the
                # relay would stay blocked on the pipe the helper holds.
                proc = popen_process_group(
                    [str(hook), "login", provider],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",  # output that is not UTF-8 is not JSON, not a crash
                    env=_hook_env(),
                )
            except OSError as exc:
                reply({"type": "signinError", "provider": provider, "error": str(exc)}, False)
                return
            assert proc.stdin is not None
            os.set_blocking(proc.stdin.fileno(), False)  # see paste()
            self._proc = proc
            self._owner, self._provider, self._reply = owner, provider, reply
        thread = threading.Thread(
            target=self._relay,
            args=(proc, provider, reply),
            name="signin-login",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError as exc:  # no thread to be had: nothing would read the hook
            with self._lock:
                if self._proc is proc:  # else cancel() or a newer start() ended it: silent
                    self._end_locked()
                    reply({"type": "signinError", "provider": provider, "error": str(exc)}, False)
            assert proc.stdout is not None
            proc.stdout.close()  # the relay's job, done here for want of one

    def paste(self, text: str) -> bool:
        """Write the pasted redirect URL or code to the running hook's stdin.

        The text is cut to :data:`PASTE_LIMIT` characters (a redirect URL
        or a code is far shorter) and written to the non-blocking pipe
        in one ``os.write``: a paste that does not fit because the hook
        is not reading is refused, never parked on the pipe.  A parked
        writer would hold up :meth:`cancel`, the next :meth:`start` and
        the close of the pipe after the hook's exit (hence its
        ``signinDone``) for as long as any process holding the pipe's
        read end lives, which a descendant of the hook may well do
        after the hook itself is gone.  As nothing can block, the write
        happens under the lock, so the stream is closed only once no
        paste can reach it any more.

        Returns:
            ``False`` when no login is running (or the hook closed its
            stdin, or is not reading it), so the window can say so.  A
            paste that a nearly full pipe took only in part ends the
            login: only a write of at most ``PIPE_BUF`` bytes is all or
            nothing, and the part written, with no newline, would be
            glued to the next paste the hook reads.
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                return False
            data = (text.strip()[:PASTE_LIMIT] + "\n").encode()
            try:
                written = os.write(proc.stdin.fileno(), data)
            except (OSError, ValueError):  # the pipe is full, closed or gone
                return False
            if written < len(data):  # cut short: the stream now holds a line fragment
                self._end_locked()
                return False
        return True

    def cancel(self, owner: str = "") -> None:
        """End the running login, if any; its relay thread then stays silent.

        Returns at once: the hook is sent SIGTERM here, and a thread
        sees it gone (:meth:`_end_locked`).

        Args:
            owner: The window that cancels (its connection id); another
                window's login is told that it was ended.
        """
        with self._lock:
            self._end_locked(owner)

    def shutdown(self) -> None:
        """End the running login and wait, bounded, until every ended hook is gone.

        For the daemon's shutdown: the :meth:`_end_locked` threads are
        daemon threads, which the interpreter's exit abandons, so a hook
        that ignored SIGTERM would otherwise live on past the daemon
        with a pipe to nowhere.  Waits :data:`HOOK_END_TIMEOUT` seconds
        plus one for all of them together.
        """
        self.cancel()
        with self._lock:
            threads = list(self._ending)
        deadline = time.monotonic() + HOOK_END_TIMEOUT + 1.0
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    def _end_locked(self, by: str = "") -> None:
        """End the running hook and forget it (the lock is held).

        The window that started it is told (:data:`ENDED_ELSEWHERE`)
        when *by* names another one: it is waiting for an answer the
        hook's relay no longer sends.  Its own cancel or new click, and
        an end naming no window (the daemon's shutdown), stay silent.

        Its stdin is closed and its process group is sent SIGTERM at
        once; a thread (:meth:`_end`) then kills whatever of the group
        ignores that after :data:`HOOK_END_TIMEOUT` seconds.  (The relay
        thread cannot do this: it is reading the hook's stdout.)  With no
        thread to be had, the group is seen out here, bounded the same
        way.  A hook that has exited already (reaped here; the relay
        thread reaps only under the lock, and only the hook still current)
        is signalled nothing: a reaped pid may be a stranger's by now, and
        a helper the hook left behind is left alone as after any exit of
        its own.  Whether it has is settled before its stdin is closed: a
        hook that exits on that close (EOF on its ``read``) must still be
        ended with its group.
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        exited = proc.poll() is not None
        assert proc.stdin is not None
        proc.stdin.close()
        if by and self._owner not in ("", by):
            assert self._reply is not None
            error = {"type": "signinError", "provider": self._provider, "error": ENDED_ELSEWHERE}
            self._reply(error, False)
        if exited:
            return
        try:
            kill_process_group(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        thread = threading.Thread(target=self._end, args=(proc,), name="signin-end", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            terminate_process_group(proc, HOOK_END_TIMEOUT)
            return
        self._ending.append(thread)

    def _end(self, proc: subprocess.Popen[str]) -> None:
        """Thread body of :meth:`_end_locked`: see the ended *proc* and its group out."""
        terminate_process_group(proc, HOOK_END_TIMEOUT)
        with self._lock:
            self._ending.remove(threading.current_thread())

    def _relay(self, proc: subprocess.Popen[str], provider: str, reply: Reply) -> None:
        """Thread body of :meth:`start`: the hook's first and last lines, as events.

        A hook that :meth:`cancel` or a newer :meth:`start` has ended by
        the time a line is read sends nothing: its URL (a dead loopback
        port) must not reach the window after the replacement's, so the
        "still current" check and the send happen under the lock (the
        reply only queues the frame).  Its failure at the end is sent
        the same way: a retry that started between the check and the
        send would otherwise see the old failure reported after its own
        URL.  A sign-in that succeeded is announced through the status
        lock (:func:`_signed_in`), outside this one, and whoever ended
        the hook since: the user is signed in.  The lines are read
        until the hook's exit (:func:`_hook_lines`), so a helper that
        outlives it with its stdout cannot hold up the result.
        """
        assert proc.stdout is not None
        lines = _hook_lines(proc)
        first = _parse(next(lines, ""))
        url = first.get("url")
        with self._lock:
            if url and self._proc is proc:
                reply({"type": "signinUrl", "provider": provider, "url": str(url)}, False)
        last = first
        for line in lines:
            if line.strip():
                last = _parse(line)
        with self._lock:
            if self._proc is not proc:  # ended by cancel() or a newer start()
                proc.stdout.close()
                # _end_locked() closed its stdin and reaps it, after its group's signals.
                if not last.get("logged_in"):
                    return
            else:
                self._proc = None
                proc.wait()  # at once: the lines ran out at its exit; reaped by the lock holder
                proc.stdout.close()
                # Closed here, not left to the Popen's finaliser (a ResourceWarning
                # and a pipe held open until the collector gets to it).  No paste()
                # reaches the stream any more (it is not the current hook's), and
                # the stream buffers nothing (paste() writes the descriptor), so
                # the close cannot block or fail.
                assert proc.stdin is not None
                proc.stdin.close()
                if proc.returncode != 0 or not last.get("logged_in"):
                    error = last.get("error") or (
                        f"the sign-in hook exited with status {proc.returncode}"
                        if url
                        else "the sign-in hook printed no sign-in page"
                    )
                    reply({"type": "signinError", "provider": provider, "error": str(error)}, False)
                    return
        locked_call(_status_lock, _signed_in, reply, str(last.get("email") or ""))
