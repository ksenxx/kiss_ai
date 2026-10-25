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
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home
from kiss.core.locked import locked_call

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

PASTE_LIMIT = 4096
"""Characters of a paste handed to the hook: enough for any redirect URL, less than a pipe holds."""

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


def _drop_status() -> None:
    """Clear the cache (lock held)."""
    global _status_cache
    _status_cache = None


def _signed_in(reply: Reply, email: str) -> None:
    """Drop the cached answer and tell every window ``signinDone`` (lock held).

    One step with :func:`_send_cached_status`'s delivery, so no window
    gets a ``signin`` after the ``signinDone`` that closed its window.
    """
    _drop_status()
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
    """Run ``signin status`` of *hook*; the ``signin`` event it calls for, else ``None``."""
    try:
        run = subprocess.run(
            [str(hook), "status"],
            capture_output=True,
            text=True,
            timeout=STATUS_TIMEOUT,
            env=_hook_env(),
            check=False,
        )
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
    leaves two hooks waiting on two loopback ports.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._ending: list[threading.Thread] = []
        """The :meth:`_end` threads of ended hooks not yet gone, under ``_lock``."""

    def start(self, provider: str, reply: Reply) -> None:
        """Run ``signin login`` *provider*; relay its lines to *reply* from a thread.

        The window that clicked gets ``signinUrl`` (``provider``, ``url``)
        as soon as the hook prints the page to open, then ``signinError``
        when the hook fails; ``signinDone`` (``email``) goes to every
        window.  A hook that cannot start, or prints no URL first, is a
        ``signinError`` too.

        Args:
            provider: The provider id from the ``signin`` event.
            reply: Delivers an event to the clicking window
                (``reply(event, False)``) or to every window
                (``reply(event, True)``).
        """
        hook = signin_hook()
        if hook is None:
            reply({"type": "signinError", "provider": provider, "error": "no sign-in hook"}, False)
            return
        with self._lock:
            self._end_locked()
            try:
                proc = subprocess.Popen(
                    [str(hook), "login", provider],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=_hook_env(),
                )
            except OSError as exc:
                reply({"type": "signinError", "provider": provider, "error": str(exc)}, False)
                return
            self._proc = proc
        threading.Thread(
            target=self._relay,
            args=(proc, provider, reply),
            name="signin-login",
            daemon=True,
        ).start()

    def paste(self, text: str) -> bool:
        """Write the pasted redirect URL or code to the running hook's stdin.

        The text is cut to :data:`PASTE_LIMIT` characters (a redirect URL
        or a code is far shorter), and written outside the lock, so a
        paste the hook is not reading can never hold up :meth:`cancel`
        or the next :meth:`start`.

        Returns:
            ``False`` when no login is running (or the hook closed its
            stdin), so the window can say so.
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                return False
        try:
            proc.stdin.write(text.strip()[:PASTE_LIMIT] + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            return False
        return True

    def cancel(self) -> None:
        """End the running login, if any; its relay thread then stays silent.

        Returns at once: the hook is sent SIGTERM here, and a thread
        sees it gone (:meth:`_end_locked`).
        """
        with self._lock:
            self._end_locked()

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

    def _end_locked(self) -> None:
        """End the running hook and forget it (the lock is held).

        It is sent SIGTERM at once; a thread (:meth:`_end`) then sends
        SIGKILL after :data:`HOOK_END_TIMEOUT` seconds when it ignores
        that, and closes its stdin.  A thread, and only once the hook is
        dead, because a paste the hook never read can fill the pipe and
        leave its writer (:meth:`paste`) blocked on the stream's buffer
        lock, which closing the stream needs too: only the hook's death
        frees it.  (The relay thread cannot do this: it is blocked reading
        the hook's stdout.)
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        thread = threading.Thread(target=self._end, args=(proc,), name="signin-end", daemon=True)
        self._ending.append(thread)
        thread.start()

    def _end(self, proc: subprocess.Popen[str]) -> None:
        """Thread body of :meth:`_end_locked`: see the ended *proc* out and close its stdin."""
        try:
            proc.wait(HOOK_END_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        assert proc.stdin is not None
        try:
            proc.stdin.close()
        except OSError:  # an unread paste, and the hook is gone
            pass
        with self._lock:
            self._ending.remove(threading.current_thread())

    def _relay(self, proc: subprocess.Popen[str], provider: str, reply: Reply) -> None:
        """Thread body of :meth:`start`: the hook's first and last lines, as events.

        A hook that :meth:`cancel` or a newer :meth:`start` has ended by
        the time a line is read sends nothing: its URL (a dead loopback
        port) must not reach the window after the replacement's, so the
        "still current" check and the send happen under the lock (the
        reply only queues the frame).
        """
        assert proc.stdout is not None
        first = _parse(proc.stdout.readline())
        url = first.get("url")
        with self._lock:
            if url and self._proc is proc:
                reply({"type": "signinUrl", "provider": provider, "url": str(url)}, False)
        last = first
        for line in proc.stdout:
            if line.strip():
                last = _parse(line)
        proc.wait()
        with self._lock:
            if self._proc is not proc:  # ended by cancel() or a newer start()
                return
            self._proc = None
        if proc.returncode == 0 and last.get("logged_in"):
            locked_call(_status_lock, _signed_in, reply, str(last.get("email") or ""))
            return
        error = last.get("error") or (
            f"the sign-in hook exited with status {proc.returncode}"
            if url
            else "the sign-in hook printed no sign-in page"
        )
        reply({"type": "signinError", "provider": provider, "error": str(error)}, False)
