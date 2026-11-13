# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Browser automation tool for LLM agents using Playwright.

Drives a Chromium that no window shows by default (page analysis via the
accessibility tree, clicking, typing, screenshots).  ``show_browser()``
switches the session to a visible window when a page needs a human
(interactive login, CAPTCHA, bot check).

Bot-protection vendors (Cloudflare, Akamai, PerimeterX, Anubis, ...)
blocked 95 hosts in the task history, so the browser is made to look and
behave like a person's Chrome — see :mod:`kiss.agents.sorcar.web_stealth`:
Patchright's leak-free driver, a *headed* Chromium on a private Xvfb
display where one is available (headless mode is the strongest bot
signal), no fingerprint overrides, curved pointer paths, uneven typing
cadence, and challenge pages that are waited out and reported plainly.
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections import Counter, deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeGuard
from urllib.parse import urlparse

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.useful_tools import resolve_tool_path
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.processes import (
    IS_WINDOWS,
    SIGKILL,
    exited_unreaped,
    kill_process_tree,
    popen_process_group,
    signal_unreaped_child,
)
from kiss.core.processes import pid_alive as _pid_alive
from kiss.core.processes import process_identity as _process_identity

logger = logging.getLogger(__name__)

_SINGLETON_FILES = ("SingletonLock", "SingletonCookie", "SingletonSocket")

_ACCOUNTS_GOOGLE_URL_RE = re.compile(r"^https?://accounts\.google\.com/")


# Upper bound for every read of page state (title, text, aria snapshot).
# A page whose main thread is stuck (a script spinning forever, a
# renderer that never answers after a timed-out ``goto``) must surface as
# a tool error, never as a tool call that hangs the whole task.
_PAGE_READ_TIMEOUT_MS = 10000

# How many navigations a single page read follows (see
# ``WebUseTool._read_page``): a challenge page reloading into the real
# page, which then redirects once more, is the longest chain seen live.
_PAGE_READ_NAVIGATIONS = 3

# How long to wait for a tab the daemon opened in the Browser tab's
# browser to show up in this tool's own CDP client.
_LIVE_PAGE_TIMEOUT = 15.0

# How long a dead-driver cleanup waits for the daemon to register a popup
# this tool followed just before (it maps the popup's target id to a tab
# id only once its own client has attached to the page).
_LIVE_TAB_LOOKUP_SECS = 3.0

# Deadline, on this side of the driver, for the round trips to a
# Browser-tab page that have no usable one of their own: the liveness
# probe (``WebUseTool._is_alive``; a closed page or an exited browser
# fails it at once, the bound only caps the wait on a page that is not
# answering, which counts as alive) and the CDP session calls that read
# a page's target id (``WebUseTool._target_id``: attaching a session and
# ``Target.getTargetInfo``, both answered by the browser process in
# milliseconds; Playwright gives neither a timeout, and its driver fails
# a pending request of a CDP session only on the session's own detach
# event, never when the browser stops answering or exits under it).
_CDP_CALL_TIMEOUT_SECS = 3.0

# Longest numbered accessibility tree a tool result carries; longer
# trees are cut with a ``[truncated]`` marker.
_AX_TREE_MAX_CHARS = 50000

# Deadline for raw input operations (keyboard.press/type, mouse.move/wheel)
# that have no Playwright timeout parameter.  A page event handler that
# wedges the renderer in response to our own input (e.g. a ``keydown``
# listener entering ``while(true)``) passes the pre-input liveness probe
# and then blocks the input call forever; a watchdog kills Chromium at
# this deadline so the pending call raises instead of hanging the task.
_INPUT_WATCHDOG_SECS = _PAGE_READ_TIMEOUT_MS / 1000 + 5.0

# How long after the input watchdog's deadline the Node driver is killed as
# well.  The watchdog's recovery ends the pending call only through a
# driver that still reports it: Chromium's exit (SIGTERM, then SIGKILL two
# seconds later) or a Browser-tab page the daemon stopped or closed (12 s
# of timed steps once it holds its launch lock).  A stalled driver
# reports neither.
_INPUT_DRIVER_GRACE_SECS = 15.0

# How long a Browser-tab page whose hung script was stopped gets before it
# is asked whether it answers.  A task the stopped handler had queued is
# due at once: a read sent sooner could be answered just before that task
# hangs the page again.
_STOPPED_SCRIPT_SETTLE_MS = 300

# The driver's deadlines for an action (click, hover, screenshot) and for
# a browser launch or connect: Playwright's defaults, passed with each
# request so that the deadline ``WebUseTool._driver_call`` keeps on this
# side is the same number.
_ACTION_TIMEOUT_MS = 30000
_LAUNCH_TIMEOUT_MS = 180000

# Longest wait for another kiss process to finish its browser launch under
# ``<profile>.lock`` (``WebUseTool._profile_lock``).  A legitimate holder
# keeps it for one launch: Playwright's 30 s launch deadline, up to three
# 3 s pid-capture round trips and the cleanup of stale ``_N`` profiles.
# A bounded wait polls the lock in Python instead of blocking in one
# kernel ``flock`` call, so a Stop reaches the task thread within 50 ms
# while a holder wedged inside Playwright keeps the file.
_PROFILE_LOCK_TIMEOUT_SECS = 120.0

# Deadline for the one-time ``playwright install chromium`` download.
_CHROMIUM_INSTALL_SECS = 900.0

# How many times ``go_to_url`` issues a navigation that Chromium aborted with
# ``net::ERR_NETWORK_CHANGED`` (a host interface came or went mid-request).
_NETWORK_CHANGE_ATTEMPTS = 3

# How long a challenge widget (Turnstile's "Verify you are human" box,
# HUMAN's "Press & Hold" bar) must have been showing before it is pressed:
# a person's reading/reaction time, during which the pointer keeps drifting.
_WIDGET_REACTION_SECS = 1.2

# How long the press-and-hold bar must stay showing after a hold before it
# is held again (HUMAN's two-step variant).  A cleared bar says
# "completed, please wait" and reloads the page about 2 s after release;
# a second hold inside that window threw away a fiverr pass.  A person
# watches what happens first.
_WIDGET_RETRY_SECS = 5.0

# DevTools-grade observation (evaluate_script, list_console_messages,
# list_network_requests, wait_for): per-page ring buffers so a chatty page
# (an analytics beacon loop, a logging SPA) cannot grow memory without
# bound, and per-entry caps so one huge message or URL cannot blow up a
# tool result.
_CONSOLE_LOG_LIMIT = 200
_NETWORK_LOG_LIMIT = 300
_MESSAGE_TEXT_MAX = 500
_URL_TEXT_MAX = 300

# Caps for the results of evaluate_script, for wait_for's timeout and for
# scroll's step count: an agent must not be able to turn one tool call
# into an unbounded read or an unbounded block of the task thread.
_EVAL_RESULT_MAX_CHARS = 20000
_WAIT_FOR_MAX_SECS = 120.0
_SCROLL_MAX_STEPS = 50

# How long use_my_chrome waits for the extension when the bridge has no
# connection yet: the extension's service worker retries every 3 s while
# awake, so a bridge this call just started is usually found within this.
_BRIDGE_CONNECT_WAIT_SECS = 5.0


def _abort_route(route: Any) -> None:
    """Abort a Playwright route request (used to block accounts.google.com)."""
    route.abort()


def _get_frontmost_app() -> str | None:
    """Return the name of the frontmost macOS application, or None on failure."""
    if sys.platform != "darwin":
        return None
    try:
        r = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of first '
                "application process whose frontmost is true",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _activate_app(name: str | None) -> None:
    """Bring *name* to the foreground on macOS. No-op if name is None or non-macOS."""
    if not name or sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["osascript", "-e", f'tell application "{name}" to activate'],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        pass


INTERACTIVE_ROLES = {
    "link",
    "button",
    "textbox",
    "searchbox",
    "combobox",
    "checkbox",
    "radio",
    "switch",
    "slider",
    "spinbutton",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "treeitem",
}

# The interactive roles ARIA names by what they contain.  What the others
# contain is their value: a text field's text, a list box's options.
_NAMED_BY_CONTENT = INTERACTIVE_ROLES - {"textbox", "searchbox", "combobox", "slider", "spinbutton"}
# The roles whose content is part of the name of such an element around
# them (the others, a figure or a group, keep theirs to themselves), as
# Playwright computes names (``allowsNameFromContent`` of its roleUtils),
# with the two an AI-mode snapshot has for text and for a plain container.
_NAMING_ROLES = frozenset(
    "text generic button cell checkbox columnheader gridcell heading link menuitem"
    " menuitemcheckbox menuitemradio option radio row rowheader switch tab tooltip treeitem"
    " caption code contentinfo definition deletion emphasis insertion list listitem mark"
    " paragraph region rowgroup strong subscript superscript table term time".split()
)

# One line of Playwright's ``mode="ai"`` ARIA snapshot: its indentation,
# the quote YAML puts around a key that needs one, the node's role
# (``/url`` on the line of a property of the node above), its accessible
# name (a JSON string, or bare when it starts and ends with a slash), its
# bracketed attributes and, after a colon, its text.  A bare colon ends
# the line of a node whose children follow.
_SNAPSHOT_LINE_RE = re.compile(
    r"""^( *)- ('?)(/?\w+)( (?:"(?:[^"\\]|\\.)*"|/(?:.*?/)?))?((?: \[[^\]]*\])*)\2(:(?: .*)?)?$""",
    re.MULTILINE,
)
_NAME_UNESCAPE_RE = re.compile(r'\\(["\\])')
# What makes YAML, and so Playwright, put a node's key in single quotes.
_QUOTED_KEY_RE = re.compile(r":\s|\s#|[{}`]")

# The reference such a snapshot gives an element among its attributes:
# ``aria-ref=<ref>`` selects that very DOM node, or nothing once the node
# has left the page.
_REF_RE = re.compile(r" \[ref=(\w+)\]")
# The attributes only such a snapshot has.
_AI_ATTRS_RE = re.compile(r" \[(?:ref=\w+|cursor=pointer|active|aria-hidden)\]")

# How long a read of the tree waits for the page's iframes: the AI-mode
# snapshot reads them too and, when this runs out, returns without the
# ones that have not answered.  A page that is itself busy by then is
# given the usual ``_PAGE_READ_TIMEOUT_MS`` (``WebUseTool._aria_snapshot``).
_FRAMES_TIMEOUT_MS = 2000

# What the agent is told to do about an element id it can no longer use.
_READ_AGAIN = "Call get_page_content() to read the page again and use the IDs it shows."

_SCROLL_DELTA = {"down": (0, 300), "up": (0, -300), "right": (300, 0), "left": (-300, 0)}


_CLOSE_WATCHDOG_SECS = 15.0
# How long a new driver may take to answer its handshake before it is
# killed: ``start()`` returns only once it has.
_DRIVER_START_SECS = 30.0
# How long ``playwright.stop()`` may run before the driver is killed: the
# stop waits for the driver's stdout to close, which a driver that no
# longer drains its pipe (suspended, wedged) never does.
_DRIVER_STOP_SECS = 5.0
# How long a killed Playwright driver is given to exit so it can be reaped.
_DRIVER_REAP_SECS = 5.0


def _wait_pid_exit(pid: int, timeout: float) -> bool:
    """Poll until *pid* exits, returning True if it died within *timeout*.

    Args:
        pid: Process id to wait for.
        timeout: Maximum seconds to wait.

    Returns:
        True iff the process no longer exists.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)  # pragma: no cover — timing-dependent


def _unwatched(send: Callable[..., Any], *args: Any) -> Any:
    """Send at once: the ``act`` of a caller that is no watchdog (see :class:`_Watchdog`)."""
    return send(*args)


def _send_signal(pidfd: int, pid: int, sig: int) -> bool:
    """Send *sig* through *pidfd*, or to the bare *pid* where there is none; True once sent."""
    if pidfd >= 0 and sys.platform == "linux":
        signal.pidfd_send_signal(pidfd, sig)
    else:
        os.kill(pid, sig)
    return True


def _terminate_pid_escalating(
    pid: int, identity: str | None, act: Callable[..., Any] = _unwatched
) -> None:
    """Kill *pid* with SIGTERM then SIGKILL, verifying identity before EACH signal.

    Fails closed: signals are sent only when the process's current
    identity fingerprint is readable and equal to the one recorded at
    capture time.  This refuses recycled PIDs (the browser exited and the
    OS reassigned its PID — possibly between SIGTERM and SIGKILL) and
    unverifiable targets — killing an unrelated process would be far
    worse than leaking a browser.  Where the kernel has pidfds (Linux)
    the signals go through one opened before the first check: it names
    the process that was verified and no later holder of its pid, which
    closes the window between a check and its signal (the browser is
    the driver's child, reaped by it the moment it exits, and the
    identity read runs ``ps`` in between).  Elsewhere the bare pid is
    signalled.

    Args:
        pid: Process id to terminate.
        identity: Identity string recorded when the PID was captured.
        act: A watchdog's :meth:`_Watchdog.act`: each signal goes through
            it, and none is sent once the watchdog's operation has ended.
    """
    pidfd = -1
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is not None:
        try:
            pidfd = pidfd_open(pid)
        except ProcessLookupError:
            return
        except OSError:  # pragma: no cover — a kernel or seccomp profile without pidfd_open
            logger.debug("pidfd_open(%d) failed; signalling the pid instead", pid, exc_info=True)
    try:
        for sig in (signal.SIGTERM, SIGKILL):
            if not _pid_alive(pid):
                return
            current = _process_identity(pid)
            if current is None:
                return
            if identity is None or current != identity:
                logger.warning(
                    "Refusing to kill pid %d: cannot verify it is our browser "
                    "(recorded=%r, current=%r)",
                    pid,
                    identity,
                    current,
                )
                return
            logger.warning("Killing leaked Chromium (pid %d) with signal %d", pid, sig)
            try:
                if not act(_send_signal, pidfd, pid, sig):
                    return  # the watchdog's operation has ended: the browser is the next call's
            except OSError:  # died between the check and the signal
                return
            if _wait_pid_exit(pid, 2.0):
                return
        logger.error(  # pragma: no cover — SIGKILL cannot be ignored
            "Chromium (pid %d) could not be killed",
            pid,
        )
    finally:
        if pidfd >= 0:
            os.close(pidfd)


def _killable_live_pid(pid: int | None) -> TypeGuard[int]:
    """Return whether *pid* names a live process a watchdog may target.

    The shared arming guard of the graceful-close watchdog
    (:meth:`WebUseTool._close_browser_only`) and the raw-input watchdog
    (:meth:`WebUseTool._input_hang_watchdog`); the two previously
    duplicated it inverted, a drift hazard for a predicate whose
    failure mode is signalling the wrong process.  Refuses ``None``
    and non-positive pids (``0``/negatives address process groups) and
    this process's own pid, and requires the process to still exist.

    Args:
        pid: The recorded browser pid, or ``None`` when none was
            captured.

    Returns:
        True when a watchdog may be armed against *pid* (narrowing it to
        ``int`` for the caller).
    """
    return pid is not None and pid > 0 and pid != os.getpid() and _pid_alive(pid)


def _watchdog_kill(
    pid: int,
    identity: str | None,
    operation: str = "graceful browser close",
    deadline_secs: float = _CLOSE_WATCHDOG_SECS,
    act: Callable[..., Any] = _unwatched,
) -> None:
    """Watchdog timer body: kill a Chromium whose driver call hung.

    Killing the browser process also unwedges the hung driver call (the
    driver observes the browser exit and completes/raises the call).

    Args:
        pid: Browser process id recorded at launch.
        identity: Identity string recorded at PID capture time.
        operation: Human-readable name of the hung operation, for the log.
        deadline_secs: The deadline that expired, for the log.
        act: The watchdog's :meth:`_Watchdog.act`, for the signals.
    """
    logger.warning(
        "%s is hung after %.0fs; killing Chromium (pid %d) directly",
        operation,
        deadline_secs,
        pid,
    )
    _terminate_pid_escalating(pid, identity, act)


def _driver_is_dead(playwright: Any) -> bool:
    """Return True iff the driver process behind a sync *playwright* has gone.

    The Node driver can crash under the tool (an internal assertion in
    its CDP session on a crashed renderer has been seen).  The sync API
    fails the first call after that (its request dies with the closed
    pipe) and, unguarded, hangs every later one for good -- the
    watchdog's kill of Chromium unwedges nothing since no driver observes
    it.  :mod:`web_stealth` turns those hangs into errors; this check is
    what lets the tool *recover*: the page's and context's local state
    (``is_closed()``, ``pages``) never learns of the exit, so without it
    every later call would fail the same way instead of relaunching.
    The transport records the failure in ``on_error_future`` (also set
    when the driver's stdout closes), and its subprocess has a return
    code once reaped; both are updated only by a call that runs the
    loop, so a death while idle is noticed by the first call after it.
    """
    try:
        transport = playwright._impl_obj._connection._transport
    except AttributeError:
        return False
    future = getattr(transport, "on_error_future", None)
    if future is not None and future.done():
        return True
    proc = getattr(transport, "_proc", None)
    return proc is not None and proc.returncode is not None


def _kill_driver_process(playwright: Any, act: Callable[..., Any] = _unwatched) -> int | None:
    """Send SIGKILL to the Node driver behind a sync *playwright*; return the pid to reap.

    The driver is this process's child and is signalled only while it
    is still an unreaped one (:func:`kiss.core.processes.signal_unreaped_child`):
    asyncio's ``Process.kill`` signals the bare pid, which, once the
    loop's child watcher has reaped the driver, may belong to another
    process.  On Windows the kill goes through the process handle and
    there is nothing to reap.  A watchdog sends the signal through its
    *act* (:meth:`_Watchdog.act`): not at all once its operation ended.

    Returns:
        The driver's pid while it is this process's unreaped child
        (signalled now, or exited already), else None: reaped, Windows,
        or another Playwright layout.
    """
    try:
        proc = playwright._impl_obj._connection._transport._proc
    except AttributeError:  # another Playwright layout
        logger.debug("Could not find the Playwright driver", exc_info=True)
        return None
    pid = int(proc.pid)
    if proc.returncode is not None:
        return None
    if IS_WINDOWS:  # pragma: no cover — Windows-only branch
        act(proc.kill)
        return None
    try:
        return pid if act(signal_unreaped_child, pid, SIGKILL) else None
    except OSError:  # no descriptor for the pidfd: the driver stays rather than a stranger risked
        logger.warning("Could not kill the Playwright driver (pid %d)", pid, exc_info=True)
        return None


def _kill_driver(
    playwright: Any,
    deadline_secs: float,
    operation: str = "Raw input to the attached Chrome",
    act: Callable[..., Any] = _unwatched,
) -> None:
    """Watchdog timer body: kill the Node driver whose call hung.

    Armed around raw input to an attached Chrome (the browser is the
    user's own, which is never signalled), around raw input to a
    launched browser whose pid could not be captured, and around
    ``playwright.stop()`` (:func:`_stop_playwright`).  The driver's exit
    fails the blocked call instead (its request dies with the driver's
    pipe), and :func:`_driver_is_dead` lets the next call discard the
    driver and relaunch, or say how to reconnect; a Chromium the driver
    launched exits with it (its remote-debugging pipe closes).  It is
    also the deadline of a navigation and the failsafe of raw input.

    Args:
        playwright: The sync Playwright whose driver to kill.
        deadline_secs: The deadline that expired, for the log.
        operation: Human-readable name of the hung operation, for the log.
        act: The watchdog's :meth:`_Watchdog.act`, for the signal.
    """
    logger.warning(
        "%s is hung after %.0fs; killing the Playwright driver", operation, deadline_secs
    )
    _kill_driver_process(playwright, act)


def _stop_playwright(playwright: Any) -> None:
    """Stop a sync *playwright*; kill and reap its Node driver when ``stop()`` cannot run.

    ``stop()`` runs only on the thread that started Playwright: from any
    other (the atexit hook, or the daemon's close, of a tool created on
    a pool worker) it raises ``greenlet.error``, and its request never
    reaches the driver -- closing the driver's stdin is left to the event
    loop, which no longer turns.  Each such close left a ``node
    run-driver`` process alive until the daemon exited.  The driver is
    killed here instead and reaped (the loop's child watcher never runs
    again either, so a zombie would stay); the Chromium it launched is
    the caller's business (:meth:`WebUseTool._kill_browser_process`).
    On its own thread ``stop()`` waits for the driver's stdout to close,
    which a driver that no longer drains its pipe never does: a watchdog
    kills it after :data:`_DRIVER_STOP_SECS`, and the loop then sees the
    pipe close and reaps it.

    Args:
        playwright: The sync Playwright to stop.
    """
    watchdog = _Watchdog(
        _DRIVER_STOP_SECS,
        _kill_driver,
        args=(playwright, _DRIVER_STOP_SECS, "Stopping the Playwright driver"),
    )
    try:
        _under_timer(watchdog, playwright.stop)
    except Exception:  # noqa: BLE001 - greenlet.error off the starting thread
        logger.warning("Stopping the Playwright driver failed; killing it", exc_info=True)
        _end_driver(playwright)


def _end_driver(playwright: Any) -> None:
    """Kill the Node driver behind *playwright* and reap it, as nothing else will."""
    pid = _kill_driver_process(playwright)
    if pid is None:
        return
    deadline = time.monotonic() + _DRIVER_REAP_SECS
    while time.monotonic() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return
        except ChildProcessError:  # reaped by the loop's child watcher after all
            return
        time.sleep(0.05)
    logger.warning("Killed Playwright driver (pid %d) did not exit", pid)


class _Watchdog(threading.Timer):
    """A timer that acts for one operation, and only until that operation has ended.

    ``cancel()`` does not stop a timer whose thread is already in its
    body, and a recovery handed to the daemon's loop runs later still:
    each used to kill the driver or the browser, or stop the script,
    of whatever ran next.  *recover* therefore gets ``act=`` :meth:`act`
    and sends every signal through it, under the lock :meth:`end`
    takes: nothing is signalled for an operation that has ended.
    Nothing that can block runs under that lock (messages are logged
    outside it), so an operation that has ended waits for a signal
    under way and for nothing else.  ``acted`` tells, once the
    operation has ended, whether a signal was sent for it.
    """

    def __init__(self, secs: float, recover: Callable[..., Any], args: tuple[Any, ...]) -> None:
        super().__init__(secs, recover, args, {"act": self.act})
        self.daemon = True
        self._deciding = threading.Lock()
        self._ended = False
        self.acted = False

    def end(self) -> None:
        """Mark the operation ended, then cancel the timer."""
        with self._deciding:
            self._ended = True
        self.cancel()

    def act(self, send: Callable[..., Any], *args: Any) -> Any:
        """Run ``send(*args)``, which must not block, unless the operation has ended.

        Returns:
            What *send* returned, or None once the operation has ended.
        """
        with self._deciding:
            if self._ended:
                return None
            self.acted = True
            return send(*args)


def _under_timer(timer: _Watchdog, operation: Callable[[], Any]) -> Any:
    """Run *operation* with the watchdog *timer* armed; end it however the operation ends.

    The timer starts inside the ``try`` whose ``finally`` ends it, so
    a Stop (the ``ToolCallInterrupted`` the daemon injects into the task
    thread asynchronously) that lands once the timer's thread exists can
    only unwind through the cancel.  A context manager cannot promise
    that: a Stop raised while its ``__enter__`` returns (inside
    ``Thread.start`` after the thread was started, or at the return
    itself under a tracer) skips ``__exit__``, and the armed watchdog
    then killed the browser or driver of whatever ran next.

    Args:
        timer: The watchdog to arm; ended on return (cancelled, and
            refused its signals once it has fired).
        operation: The guarded call.

    Returns:
        *operation*'s result.
    """
    try:
        timer.start()
        return operation()
    finally:
        timer.end()


def _kill_driver_unless_done(
    done: threading.Event, playwright: Any, deadline_secs: float, operation: str
) -> None:
    """Loop timer body of :func:`_cdp_call`: :func:`_kill_driver`, unless the call is *done*.

    Args:
        done: Set once the guarded call ended, however it ended.
        playwright: The sync Playwright whose driver to kill.
        deadline_secs: The deadline that expired, for the log.
        operation: Human-readable name of the hung operation, for the log.
    """
    if not done.is_set():
        _kill_driver(playwright, deadline_secs, operation)


def _cdp_call(sync_obj: Any, coro: Any, timeout_secs: float = _CDP_CALL_TIMEOUT_SECS) -> Any:
    """Run *coro*, a Playwright ``_impl`` coroutine, on the sync API's loop with a deadline.

    Playwright's sync wrappers run every call through ``SyncBase._sync``,
    whose greenlet dispatcher keeps the event loop turning while the
    caller blocks, so the loop's own timer can cut a call short:
    ``asyncio.wait_for`` cancels the pending request and raises.  The
    sync API has no other deadline for a CDP round trip
    (``BrowserContext.new_cdp_session``, ``CDPSession.send``), and a
    helper thread cannot provide one: sync API objects belong to the
    greenlet of the thread that started Playwright.  Relies on the
    ``_sync`` / ``_impl_obj`` surface shared by Playwright and
    Patchright (verified on Playwright 1.63; the live-browser tests
    exercise it).

    The cancel needs the driver too: the request is aborted by a
    message the driver acknowledges (``Connection._abort`` awaits the
    acknowledgement), so a driver that no longer answers at all would
    hold the cancelled call as it held the call.  A loop timer
    therefore kills the driver a round trip after the deadline
    (:func:`_kill_driver`): its closed pipes fail the call at once, and
    the next tool call discards the dead driver and relaunches
    (:func:`_driver_is_dead`).  The timer's leave to kill is the
    ``done`` event, not its handle: a Stop (the ``ToolCallInterrupted``
    the daemon injects into the task thread) landing as ``call_later``
    returns has registered the timer but not assigned the handle the
    ``finally`` cancels, so the event, set there whatever happened,
    is what makes such a timer inert when the loop next turns
    (:func:`_kill_driver_unless_done`).

    Args:
        sync_obj: Any sync API object of the Playwright instance (all of
            them share one loop, dispatcher and driver).
        coro: The ``_impl_obj`` coroutine to run.
        timeout_secs: The deadline in seconds; :data:`_CDP_CALL_TIMEOUT_SECS`
            (a plain round trip) unless the call legitimately runs longer,
            as ``Runtime.evaluate`` of a page script does.

    Returns:
        *coro*'s result (an ``_impl`` object or plain data, not a sync
        wrapper).

    Raises:
        TimeoutError: *coro* did not finish within *timeout_secs*.
    """
    loop = sync_obj._loop
    done = threading.Event()
    kill = None
    try:
        if not loop.is_closed():
            grace = timeout_secs + _CDP_CALL_TIMEOUT_SECS
            kill = loop.call_later(
                grace,
                _kill_driver_unless_done,
                done,
                sync_obj,
                grace,
                "A cancelled browser call",
            )
        return sync_obj._sync(asyncio.wait_for(coro, timeout_secs))
    finally:
        done.set()
        if kill is not None:
            kill.cancel()
        # A stopped Playwright (closed loop) rejects the call before the
        # wrapper runs and closes only the wrapper, leaving *coro* itself
        # never awaited (a RuntimeWarning at collection).  Once started,
        # it finished or was cancelled on the loop and is left alone.
        if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
            coro.close()


def _page_read(sync_obj: Any, coro: Any, timeout_ms: float) -> Any:
    """Run a page read's ``_impl`` coroutine under its driver-side *timeout_ms*, bounded here too.

    The ``timeout`` of ``wait_for_function``, ``aria_snapshot`` and
    ``inner_text`` is enforced by the Node driver's progress controller,
    which bounds a renderer that no longer answers but not a driver
    that no longer does: a stalled ``node run-driver`` (alive, its pipes
    open, every request unanswered) held each of these reads for good.
    :func:`_cdp_call` adds the loop's own deadline, a round trip beyond
    the driver's so a timeout the driver reports keeps its error text.

    Args:
        sync_obj: Any sync API object of the Playwright instance.
        coro: The read's ``_impl_obj`` coroutine, created with
            ``timeout=timeout_ms``.
        timeout_ms: The driver-side timeout the coroutine was given.

    Returns:
        *coro*'s result (an ``_impl`` object or plain data).

    Raises:
        TimeoutError: The driver did not answer within the deadline.
    """
    try:
        return _cdp_call(sync_obj, coro, timeout_secs=timeout_ms / 1000 + _CDP_CALL_TIMEOUT_SECS)
    except TimeoutError:
        raise TimeoutError(
            f"Timeout {timeout_ms:.0f}ms exceeded: the browser did not answer"
        ) from None


def _wait_for_function(page: Any, expression: str, arg: Any, timeout_ms: float) -> Any:
    """``page.wait_for_function`` as a bounded page read; return the ``_impl`` handle.

    ``polling=100`` (not the default ``raf``) so throttled background
    tabs still answer.  The caller releases the handle with
    :func:`_dispose_handle`.

    Args:
        page: The sync Playwright page to read.
        expression: JavaScript function source whose truthy result ends
            the wait.
        arg: Argument passed to the function.
        timeout_ms: The wait's deadline, driver-side and here.

    Returns:
        The ``_impl`` ``JSHandle`` of the expression's result.
    """
    return _page_read(
        page,
        page._impl_obj.wait_for_function(expression, arg=arg, timeout=timeout_ms, polling=100),
        timeout_ms,
    )


def _dispose_handle(page: Any, handle: Any) -> None:
    """Release the temporary ``JSHandle`` *handle* of a page read, with a deadline.

    ``wait_for_function`` returns a handle that ``json_value()`` leaves
    registered in the driver's connection, where Python's garbage
    collector never reaches it: every title, tree, viewport, challenge
    or liveness read would otherwise keep one for the life of the page.
    The release is a renderer round trip of its own, so it runs through
    :func:`_cdp_call`; a page whose context is gone or whose renderer
    no longer answers fails or times it out, which is logged and
    ignored (the handle dies with the page).

    Args:
        page: The sync Playwright page the handle belongs to.
        handle: The ``_impl`` ``JSHandle`` to release.
    """
    try:
        _cdp_call(page, handle.dispose())
    except Exception:  # noqa: BLE001 - context gone, renderer wedged, driver dead
        logger.debug("Could not release a page-read handle", exc_info=True)


@dataclass
class _Session:
    """What moves with the browsing session between two browsers."""

    url: str
    # The context's cookies; ``None`` when they were not read (the user's
    # own Chrome, or a browser that did not answer), which is not an
    # empty jar: nothing is deleted for it on the other side.
    cookies: list[Any] | None
    # The page's origin and the ``[key, value]`` pairs of its localStorage
    # (empty origin: the storage could not be read).
    origin: str = ""
    local_storage: list[list[str]] = field(default_factory=list)


class _PageRecord:
    """What the tool keeps for one open page (:meth:`WebUseTool._record`).

    The DevTools buffers are bounded deques: a page logging in a loop or
    beaconing every second must not grow memory forever.
    """

    def __init__(self) -> None:
        # Console messages and uncaught errors (``list_console_messages``).
        self.console: deque[dict[str, str]] = deque(maxlen=_CONSOLE_LOG_LIMIT)
        # Requests and their outcome (``list_network_requests``).
        self.network: deque[dict[str, Any]] = deque(maxlen=_NETWORK_LOG_LIMIT)
        # Request object -> its entry in ``network``, until the response
        # or the failure arrives to complete it.
        self.pending: dict[Any, dict[str, Any]] = {}
        # The page's one CDP session (:meth:`WebUseTool._page_session`).
        self.session: Any = None
        # ``Runtime.enable`` was answered on ``session``: console capture runs.
        self.console_on = False
        # The network listeners are registered (:meth:`WebUseTool._observe_page`).
        self.observed = False


# Writes *items* and drops *removed* in the page's localStorage, unless
# the page landed on another origin (a redirect): a token must never be
# handed to a site it was not set by.  Returns ``[written]`` (a list, as
# ``WebUseTool._read_page`` requires).
_RESTORE_STORAGE_JS = """([origin, items, removed]) => {
  if (location.origin !== origin) return [false];
  for (const [k, v] of items) localStorage.setItem(k, v);
  for (const k of removed) localStorage.removeItem(k);
  return [true];
}"""


def _cookie_key(cookie: dict[str, Any]) -> tuple[str, str, str, str, bool]:
    """The identity of a cookie: name, domain, path and (CHIPS) partition.

    Chromium keeps one cookie per partition (the top-level site a
    third-party cookie was set under, and whether a cross-site frame
    sat between the two: the driver's ``_crHasCrossSiteAncestor``,
    true by default as in :meth:`WebUseTool._delete_cookie`): the same
    ``session`` cookie of an identity provider may be alive under one
    site and logged out under another, and one must not stand in for
    the other.
    """
    partition = cookie.get("partitionKey") or ""
    ancestor = bool(partition and cookie.get("_crHasCrossSiteAncestor", True))
    return (cookie["name"], cookie["domain"], cookie["path"], partition, ancestor)


def _is_browser_ui(page: Any) -> bool:
    """Whether *page* is part of Chrome's own UI rather than a web page.

    The new-tab page and other ``chrome://`` pages, DevTools windows and
    extension pages of a Chrome the user started are never the tab an
    agent drives there (``attach_to_chrome``).
    """
    return str(page.url).startswith(("chrome://", "devtools://", "chrome-extension://"))


# use_my_chrome's actions and the arguments each one needs, in the order
# they are checked.
_BRIDGE_ACTIONS: dict[str, tuple[str, ...]] = {
    "list_tabs": (),
    "open_tab": ("url",),
    "navigate": ("tab_id", "url"),
    "activate_tab": ("tab_id",),
    "close_tab": ("tab_id",),
    "evaluate": ("tab_id", "script"),
    "screenshot": ("tab_id",),
    "snapshot": ("tab_id",),
    "click": ("tab_id", "uid"),
    "type_text": ("tab_id", "uid", "text"),
}

# Each use_my_chrome argument: its name in the extension's protocol, and
# what the error asks for when it is missing.
_BRIDGE_ARGS: dict[str, tuple[str, str]] = {
    "tab_id": ("tabId", "tab_id (from list_tabs or open_tab)"),
    "url": ("url", "a url"),
    "script": ("script", "a script"),
    "uid": ("uid", "a uid (from a snapshot)"),
    "text": ("text", "text"),
}


def _format_bridge_result(action: str, result: Any) -> str:
    """Render a use_my_chrome action result for the agent.

    Args:
        action: The bridge action that produced *result*.
        result: The extension's decoded JSON result.

    Returns:
        Human/agent-readable one-block text: one line per tab for
        list_tabs, a confirmation for tab actions, the JSON for evaluate.
    """
    if action == "list_tabs":
        tabs = result if isinstance(result, list) else []
        if not tabs:
            return "The user's Chrome reports no open tabs."
        lines = [
            f"id={t.get('id')} {'[active] ' if t.get('active') else ''}"
            f"{t.get('title', '')} — {t.get('url', '')}"
            for t in tabs[:50]
        ]
        if len(tabs) > 50:
            lines.append(f"… and {len(tabs) - 50} more tabs")
        return "\n".join(lines)
    if action in ("open_tab", "navigate", "activate_tab") and isinstance(result, dict):
        return (
            f"Done: tab id={result.get('id')} — {result.get('title', '')} — {result.get('url', '')}"
        )
    if action == "close_tab":
        return "Tab closed."
    if result is None:
        return "undefined"
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=1)
    return str(result)


def _rmtree_logged(path: str) -> None:
    """Remove *path* recursively, logging a WARNING if it survives.

    Args:
        path: Directory to delete.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:  # pragma: no cover — permissions/filesystem races
        logger.warning("Failed to remove profile directory %s", path, exc_info=True)
    if os.path.exists(path):  # pragma: no cover — partial deletion is rare
        logger.warning("Profile directory %s still exists after removal", path)


def _install_chromium(
    command: list[str] | None = None, timeout: float = _CHROMIUM_INSTALL_SECS
) -> None:
    """Install Playwright's Chromium with a Stop-interruptible wait.

    The wait polls the child from Python in sleeps of 50 ms, so a Stop
    injected into the task thread lands within one of them;
    ``subprocess.run`` waits in one ``select`` call that
    only output, the child's exit or the deadline ends, which deferred a
    Stop for up to ``_CHROMIUM_INSTALL_SECS`` while a download stalled.
    The installer runs in its own process group (Job Object on Windows)
    and the whole group is killed when the wait ends early or times out:
    the ``python -m playwright`` launcher only spawns the Node CLI that
    does the download, so killing the launcher alone left that CLI
    running and holding the browser-cache lock.  The launcher is watched
    without being reaped and reaped only once the wait is over: a Stop
    landing after the reap kills nothing, its pid (the group's id)
    being free for a stranger's group by then.  Output is discarded as
    before (``capture_output`` never surfaced it).

    Args:
        command: The installer command; ``None`` installs Chromium with
            the configured Playwright package.
        timeout: Seconds the installer may run.

    Raises:
        RuntimeError: When the installer exits with a non-zero status.
        subprocess.TimeoutExpired: When it outlives *timeout*.
    """
    if command is None:
        command = [sys.executable, "-m", web_stealth.playwright_package(), "install", "chromium"]
    installer = popen_process_group(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + timeout
        while not exited_unreaped(installer):
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(0.05)
    except BaseException:
        # Nothing above reaps the launcher: alive or a zombie, its pid
        # (the group id) cannot have been recycled.
        kill_process_tree(installer)
        installer.wait()
        raise
    if installer.wait():
        raise RuntimeError(f"{' '.join(command)} exited with status {installer.returncode}")


def _read_lock_pid(profile_dir: str) -> int | None:
    """Return the PID recorded in a profile's ``SingletonLock`` symlink.

    Chromium's lock symlink targets ``hostname-pid``.  Returns ``None``
    when the lock is absent or unparsable.

    Args:
        profile_dir: Path to the Chromium user-data directory.

    Returns:
        The recorded PID, or ``None``.

    Raises:
        PermissionError: The symlink cannot be read.  An inaccessible
            lock may belong to a live foreign-owned Chromium, so the
            callers must not treat it as free.
    """
    lock_path = Path(profile_dir) / "SingletonLock"
    if not lock_path.is_symlink():
        return None
    try:
        target = os.readlink(str(lock_path))
        pid = int(target.rsplit("-", 1)[-1])
        return pid if pid > 0 else None
    except PermissionError:
        raise
    except (OSError, ValueError, IndexError):
        return None


def _profile_lock_state(profile_dir: str) -> tuple[int | None, bool]:
    """Return ``(pid, in_use)`` for a Chromium profile directory's lock.

    On POSIX Chromium creates a ``SingletonLock`` symlink whose target is
    ``hostname-pid`` when a profile is opened; *pid* is that PID (``None``
    when the lock is absent, unparsable or unreadable) and the profile
    is in use while the PID is alive, or while the lock cannot be read.
    On Windows Chromium instead holds ``lockfile`` open without write
    sharing (delete-on-close, so a crash leaves nothing behind); the
    profile is in use while that file cannot be opened for writing, and
    *pid* is always ``None``.

    Args:
        profile_dir: Path to the Chromium user-data directory.

    Returns:
        The lock's PID (``None`` if unknown) and whether the profile is
        currently locked by a live process.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows-only branch
        try:
            # ``r+`` never creates the file, so a probe that races the
            # browser's delete-on-close cannot leave a stale lockfile.
            with open(Path(profile_dir) / "lockfile", "r+"):
                return None, False
        except FileNotFoundError:
            return None, False
        except PermissionError:
            return None, True
    try:
        pid = _read_lock_pid(profile_dir)
    except PermissionError:
        return None, True
    return pid, pid is not None and _pid_alive(pid)


def _scalar_text(scalar: str) -> str:
    """The text of a snapshot scalar (a name, a node's text): unescaped when double-quoted."""
    return _NAME_UNESCAPE_RE.sub(r"\1", scalar[1:-1]) if scalar.startswith('"') else scalar


def _spelled_name(nodes: list[tuple[str, ...]], position: int) -> str:
    """The name that the descendants of a snapshot node spell, as it follows a role.

    Playwright leaves an element's name out of an AI-mode snapshot when
    its descendants spell all of it (a link around a heading).  The name
    is put together again from them: a descendant's own name or, of one
    that has none and a role of :data:`_NAMING_ROLES`, its text and what
    its descendants spell in turn.  What is hidden from the
    accessibility tree is no part of it.

    Args:
        nodes: The lines of the snapshot (:data:`_SNAPSHOT_LINE_RE`).
        position: The index of the node's line.

    Returns:
        A space and the name as a JSON string; ``""`` when the
        descendants spell nothing, or more than Playwright itself prints
        as a name (900 characters).
    """
    indent = len(nodes[position][0])
    words: list[str] = []
    passed_over: int | None = None  # indentation of the node whose descendants do not count
    for index in range(position + 1, len(nodes)):
        spaces, quote, role, name, attrs, colon = nodes[index]
        if len(spaces) <= indent:
            break
        if passed_over is not None and len(spaces) > passed_over:
            continue
        passed_over = None
        if " [aria-hidden]" in attrs or not (name or role in _NAMING_ROLES):
            passed_over = len(spaces)
        elif name:
            words.append(_scalar_text(name[1:].replace("''", "'") if quote else name[1:]))
            passed_over = len(spaces)
        else:
            words.append(_scalar_text(colon[2:]))
    spelled = " ".join(filter(None, words))
    return f" {json.dumps(spelled, ensure_ascii=False)}" if 0 < len(spelled) <= 900 else ""


def _inner_reference(nodes: list[tuple[str, ...]], position: int) -> str:
    """The reference of the first node inside a snapshot node that the pointer can reach.

    An element the pointer cannot reach itself (no box of its own, no
    pointer events) can hold a node that it can, a link around a
    positioned image: pressing that node presses the element.  What lies
    inside another interactive element or an iframe is not looked at:
    pressing it presses that one.

    Args:
        nodes: The lines of the snapshot (:data:`_SNAPSHOT_LINE_RE`).
        position: The index of the node's line.

    Returns:
        The reference, or ``""`` when the node holds none.
    """
    indent = len(nodes[position][0])
    passed_over: int | None = None  # indentation of the node whose descendants do not count
    for spaces, _quote, role, _name, attrs, _colon in nodes[position + 1 :]:
        if len(spaces) <= indent:
            break
        if passed_over is not None and len(spaces) > passed_over:
            continue
        passed_over = None
        if role in INTERACTIVE_ROLES or role == "iframe":
            passed_over = len(spaces)
            continue
        ref = _REF_RE.search(attrs)
        if ref:
            return ref.group(1)
    return ""


def _number_interactive_elements(snapshot: str) -> tuple[str, list[str]]:
    """Turn Playwright's ``mode="ai"`` ARIA snapshot into the tree the agent reads.

    The tree is the page as its accessibility tree has it, a line per
    node, the way Playwright's plain snapshot prints it.  What this
    snapshot adds for a model that acts through Playwright itself is
    left out: the nodes it marks as hidden from the accessibility tree
    though visible on screen (``[aria-hidden]``), the documents of the
    iframes, the ``generic`` containers (what they hold moves up a
    level) and the references.  The name it leaves to an element's
    descendants is put back (:func:`_spelled_name`).  An interactive
    element is given an id, ``[N]``, when the snapshot has a reference
    for it.  Playwright gives none to an element the pointer cannot
    reach (no box of its own, no pointer events): its id is then that
    of a node inside it that the pointer can reach
    (:func:`_inner_reference`).  An element that holds none, an option
    of a closed list, is shown without an id, since nothing else
    selects one element and no other.

    A hidden container around a single element is not in the snapshot
    (Playwright folds it into the element), so that element is shown:
    nothing tells it from one that is not hidden.

    Args:
        snapshot: The snapshot, as Playwright returned it.

    Returns:
        The tree, and the reference behind each id: that of ``[N]`` at
        index ``N - 1``.
    """
    nodes: list[tuple[str, ...]] = _SNAPSHOT_LINE_RE.findall(snapshot)
    shown: list[tuple[int, str]] = []  # the tree's lines, each with the width of its indentation
    refs: list[str] = []
    left_out: int | None = None  # indentation of the node whose descendants are not shown
    unwrapped: list[int] = []  # indentation of the generic nodes around the current line
    for position, (spaces, quote, role, name, attrs, colon) in enumerate(nodes):
        indent = len(spaces)
        if left_out is not None and indent > left_out:
            continue
        left_out = None
        while unwrapped and unwrapped[-1] >= indent:
            unwrapped.pop()
        pad = spaces[2 * len(unwrapped) :]
        if " [aria-hidden]" in attrs:
            left_out = indent
            continue
        ref = _REF_RE.search(attrs)
        if role == "generic":
            if colon == ":":
                unwrapped.append(indent)
            elif colon or (name and ref):  # its text: its name, when it is on screen to say it
                text = colon or ":" + (name.replace("''", "'") if quote else name)
                shown.append((len(pad), f"{pad}- text{text}"))
            continue
        reference = ""
        if role in INTERACTIVE_ROLES:
            reference = ref.group(1) if ref else _inner_reference(nodes, position)
        number = ""
        if reference:
            refs.append(reference)
            number = f"[{len(refs)}] "
            if not name and role in _NAMED_BY_CONTENT:
                name = _spelled_name(nodes, position)
                if _QUOTED_KEY_RE.search(name):
                    quote, name = "'", name.replace("'", "''")
        if role == "iframe":
            left_out, colon = indent, ""
        attrs = _AI_ATTRS_RE.sub("", attrs)
        shown.append((len(pad), f"{pad}- {number}{quote}{role}{name}{attrs}{quote}{colon}"))
    # A node all of whose children were left out has none: its line ends without the colon.
    widths_after = [width for width, _ in shown[1:]] + [0]
    tree = "\n".join(
        line[:-1] if line.endswith(":") and width >= after else line
        for (width, line), after in zip(shown, widths_after)
    )
    return tree, refs


class WebUseTool:
    """Browser automation tool driving Chromium through Playwright/Patchright.

    Browsing shows no window by default (``headless=True``): nothing
    steals the user's focus, and screenshots still work because Chromium
    renders off-screen exactly as it does on screen.  On Linux with Xvfb
    installed that "no window" is a *headed* Chromium on a private virtual
    display, which bot-protection vendors cannot tell from a desktop
    browser; elsewhere Chromium's real headless mode is used.  All
    browsing happens in a single Chromium instance with a persistent
    profile, so logins — and the clearance cookies challenge pages hand
    out — survive across sessions.

    When a page needs a human — an interactive login, a CAPTCHA, a bot
    check, a live demo — :meth:`show_browser` moves the session to the
    user's Browser tab: the daemon's browser (``live_browser``, a
    :class:`kiss.server.browser_tab.BrowserTabService`) opens a tab that
    every KISS surface streams, and this tool drives that page through a
    second CDP client while the user watches and interacts.  That browser
    keeps its own persistent profile across tasks and daemon restarts;
    the cookies of the headless session are carried into it (and back).
    Without a daemon, :meth:`show_browser` reopens the profile in a
    visible window instead.
    """

    _DEFAULT_USER_DATA_DIR = "__kiss_default_browser_profile__"

    # Window size of a headed launch and the viewport of a real headless one.
    viewport: tuple[int, int] = (1280, 900)

    def __init__(
        self,
        user_data_dir: str | None = _DEFAULT_USER_DATA_DIR,
        headless: bool = True,
        work_dir: str | None = None,
        ephemeral: bool = False,
        live_browser: Any = None,
        private_dir: str | None = None,
    ) -> None:
        self._live_browser = live_browser
        # A sub-agent's private folder, deleted when the sub-agent ends:
        # where its files that would go to the work dir's tmp/ go instead.
        self._private_dir = private_dir
        # True while the session lives in the Browser tab (show_browser);
        # _live_tab is the id of the tab this tool opened there.
        self._live = False
        self._live_tab: str | None = None
        # The Chromium target id of the page driven in the Browser tab,
        # read when the page is adopted: the daemon's service maps it to
        # the tab id without this tool's driver (see _live_tab_id).
        self._live_target: str | None = None
        # The DevTools endpoint _browser is connected to in the Browser
        # tab; a relaunched browser listens on a new one (see _attach_live).
        self._live_cdp_url: str | None = None
        # What the last switch of browser carried in: cookies, and the
        # localStorage keys of one origin.  Those that have died by the
        # next switch are deleted on the other side too.
        self._carried_cookies: list[Any] = []
        self._carried_storage: tuple[str, set[str]] = ("", set())
        # The session of a show_browser() that could not open its browser,
        # until a browser of this tool launches and takes it.
        self._stranded_session: _Session | None = None
        # A throwaway profile (sub-agents): the directory is created by the
        # first launch, not here, so a tool that never opens a page leaves
        # nothing behind when its process dies without ``close()``.
        self._ephemeral = ephemeral
        self._ephemeral_dir: str | None = None
        if ephemeral:
            user_data_dir = None
        elif user_data_dir == self._DEFAULT_USER_DATA_DIR:
            user_data_dir = str(kiss_home() / "browser_profile")
        self.user_data_dir = user_data_dir
        self._headless = headless
        self.work_dir = work_dir
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        # The reference of the element behind each id of the last tree read.
        self._elements: list[str] = []
        # Pages whose renderer-crash handler is armed (:meth:`_adopt_page`).
        self._crash_wired: weakref.WeakSet[Any] = weakref.WeakSet()
        self._browser_pid: int | None = None
        self._browser_identity: str | None = None
        # Chromium's ``--headless`` switch for the current launch: False
        # when the "no window" default runs headed on a virtual display.
        self._chromium_headless = headless
        # Whether this launch already discarded a remembered user agent
        # that named another Chromium build (at most once per launch).
        self._user_agent_relearned = False
        # Last pointer position the tool moved to; curved mouse paths
        # start here.  ``None`` until the first move after a launch.
        self._mouse_xy: tuple[float, float] | None = None
        # Page -> its CDP session and DevTools buffers (see _record), so
        # background tabs keep their own.
        self._page_records: dict[Any, _PageRecord] = {}
        # The CDP endpoint of a browser the user launched and this tool
        # attached to (``attach_to_chrome``); ``None`` while driving the
        # tool's own stealth Chromium.
        self._attached_url: str | None = None
        atexit.register(self.close)

    def _context_args(self) -> dict[str, Any]:
        """Return the browser-context options for the current launch mode.

        No locale, timezone or device-scale override is set: Chromium then
        reports the machine's real values, which is what a person's
        browser does, and a fixed ``America/Los_Angeles`` next to a
        different egress IP (or a Retina scale factor on a Linux server)
        is exactly the inconsistency fingerprinting scripts look for.  A
        headed window gets ``no_viewport`` so the page fills the window
        like a desktop browser; only real headless mode, which has no
        window, is given an explicit viewport.
        """
        if self._chromium_headless:
            return {"viewport": {"width": self.viewport[0], "height": self.viewport[1]}}
        return {"no_viewport": True}

    def _viewport_size(self) -> tuple[int, int]:
        """Return the live page viewport in CSS pixels.

        Playwright reports no viewport for ``no_viewport`` contexts (the
        headed window), so the page is asked for its inner size, with the
        same bounded wait as :meth:`_page_title`.
        """
        if self._page is None:
            return self.viewport
        size = self._page.viewport_size
        if size:
            return int(size["width"]), int(size["height"])
        width, height = self._read_page(self._page, "() => [innerWidth, innerHeight]")
        return int(width), int(height)

    def _is_alive(self) -> bool:
        """Return True iff the current page/context survived (not crashed/closed)."""
        if self._playwright is None or self._context is None or self._page is None:
            return False
        if _driver_is_dead(self._playwright):
            # ``is_closed()`` is local state the dead driver never updated.
            return False
        if self._live:
            # In the Browser tab the user closes tabs and navigates too, and
            # this client only hears of that while it talks to the browser:
            # a round trip first, so a closed tab, an exited browser and the
            # page's current URL are all up to date when ``is_closed()``/
            # ``url`` are read next.  A bounded page read, not a CDP
            # session's ``detach()``: the driver fails a pending request of
            # such a session only on the session's own detach event, never
            # when the browser exits under it, so that round trip hung
            # forever on a browser closing at that very moment.  A closed
            # or crashed page fails the read at once; a slow or wedged
            # page merely times out, and is alive.  The deadline is
            # ``_cdp_call``'s, on this side: the driver's own timeout
            # (``wait_for_function``'s) never fires on a page whose
            # renderer crashed while it showed ``about:blank``, and the
            # driver does not abort that call either.
            try:
                _cdp_call(self._page, self._page._impl_obj.evaluate("1"))
            except Exception:  # noqa: BLE001 - closed, exited, crashed, or not answering
                logger.debug("Exception caught", exc_info=True)
            # The round trip delivers a crash or an exit to the handlers
            # that drop the page -- or is the first to learn that the
            # driver itself died while the tool was idle.
            if self._page is None or _driver_is_dead(self._playwright):
                return False
            if self._live_tab is None:
                # The tab of a followed popup or a switched-to page is
                # learned while the driver can still be asked (the daemon
                # registers a popup a moment after the tool adopts it):
                # once the driver has died, an unidentified tab could not
                # be closed (:meth:`_discard_dead_driver`).
                self._live_tab_id()
        try:
            return not self._page.is_closed()
        except Exception:  # pragma: no cover — Playwright internals rarely throw here
            logger.debug("Exception caught", exc_info=True)
            return False

    def _adopt_page(self, page: Any) -> None:
        """Make *page* the active page and arm the renderer-crash handler.

        Every path that points ``self._page`` at a page must go through
        here: a renderer crash on an unwatched page leaves ``_page``
        referencing a crashed-but-not-closed page that ``_is_alive``
        still reports live, wedging every later call.  The per-page
        state is reset with it — the element ids of the old page, the
        pointer position, and the Browser-tab id (resolved again from
        the new page by the next call's :meth:`_is_alive`, or when
        needed, see :meth:`_live_tab_id`), so the hang watchdog
        interrupts the page it is driving, not the one it left.  In the
        Browser tab the new page's target id is read now, while the
        driver can be asked: a popup the daemon has not registered yet
        has no tab id at this point, and a driver that dies before the
        next call could not be asked for anything, yet the tab must
        still be identifiable to be closed (:meth:`_discard_dead_driver`).
        Adopting the current page again changes nothing.
        """
        if page is self._page:
            return
        self._page = page
        self._elements = []
        self._mouse_xy = None
        self._live_tab = None
        self._live_target = None
        # A page re-adopted after a tab switch already carries the handler;
        # registering it again would stack one listener per adoption.
        if page not in self._crash_wired:
            self._crash_wired.add(page)
            page.on("crash", self._on_page_crash)
        if self._live:
            try:
                self._live_target = self._target_id(page)
            except Exception:  # noqa: BLE001 - the page closed under us
                logger.debug("Exception caught", exc_info=True)
        self._observe_page(page)

    def _observe_page(self, page: Any) -> None:
        """Attach network capture to *page* once, for the DevTools tools.

        Every adopted page (including background tabs followed later)
        gets bounded ring buffers for its network traffic and console
        messages (:class:`_PageRecord`), so ``list_network_requests`` and
        ``list_console_messages`` can answer what a bare accessibility
        tree never shows: why a click did nothing (a 500, a TypeError).
        Handlers are bound to the page's record with
        :func:`functools.partial` so a page's events land in its own
        buffers; they must never raise into the driver loop.  Console
        capture starts here only in the Browser tab
        (:meth:`_capture_console`).
        """
        record = self._record(page)
        if not record.observed:
            record.observed = True
            page.on("request", partial(self._on_request, record))
            page.on("response", partial(self._on_response, record))
            page.on("requestfinished", partial(self._on_request_finished, record))
            page.on("requestfailed", partial(self._on_request_failed, record))
        if self._live:
            self._capture_console(page)

    def _record(self, page: Any) -> _PageRecord:
        """Return what the tool keeps for *page*, creating it on first use.

        The records of pages that closed since are pruned whenever one is
        created; all of them go with the browser (:meth:`_on_browser_lost`).
        """
        record = self._page_records.get(page)
        if record is None:
            self._page_records = {p: r for p, r in self._page_records.items() if not p.is_closed()}
            record = self._page_records[page] = _PageRecord()
        return record

    def _page_session(self, page: Any) -> Any:
        """Return *page*'s CDP session, attaching it on first use.

        One bare session per page serves every raw CDP need: the target
        id (:meth:`_target_id`), ``evaluate_script`` and, once enabled,
        console capture (:meth:`_capture_console`), whose listeners are
        registered here, once.  Attaching answers at once even when a
        script has wedged the renderer, but ``detach()`` waits for the
        renderer and would hang forever, so the session is never
        detached: it lives as long as the page.  The attach has no
        timeout of its own and the driver never fails it when the
        browser stops answering or exits under it, so it goes through
        :func:`_cdp_call`; a failed attach stores nothing and the next
        caller tries again.

        Raises:
            TimeoutError: The browser did not answer within
                :data:`_CDP_CALL_TIMEOUT_SECS`.
        """
        record = self._record(page)
        if record.session is None:
            context = page.context
            session = _cdp_call(context, context._impl_obj.new_cdp_session(page))
            session.on("Runtime.consoleAPICalled", partial(self._on_console_api, record))
            session.on("Runtime.exceptionThrown", partial(self._on_exception_thrown, record))
            record.session = session
        return record.session

    def _capture_console(self, page: Any) -> None:
        """Start *page*'s console capture: enable the Runtime domain on its session, once.

        Capture goes through the page's raw CDP session, not Playwright's
        ``console``/``pageerror`` events, which the stealth driver
        (patchright) does not surface: it never enables the Runtime
        domain, because a page can tell when a DevTools client has (V8
        then formats every logged object once more, so a getter on a
        logged error's ``name`` runs twice).  So on the stealth Chromium
        and in an attached Chrome, capture starts only when
        ``list_console_messages`` asks for it; ``Runtime.enable`` replays
        what the current document has logged so far, so only messages of
        documents already navigated away from are missed.  In the Browser
        tab it starts at adoption: the daemon's own client (upstream
        Playwright) has Runtime enabled there anyway.  A
        ``Runtime.enable`` that times out (a busy renderer) still
        completes later on the same session, so the next call sends it
        again there (a repeated enable is a no-op) instead of attaching
        another session whose listeners would record every message
        twice.  Best effort: browsing works without capture.
        """
        record = self._record(page)
        if record.console_on:
            return
        try:
            _cdp_call(page, self._page_session(page).send("Runtime.enable"))
        except Exception:  # noqa: BLE001 - capture is best-effort, browsing still works
            logger.debug("Could not start console capture for a page", exc_info=True)
            return
        record.console_on = True

    def _on_console_api(self, record: _PageRecord, event: dict[str, Any]) -> None:
        """Record one Runtime.consoleAPICalled event in its page's buffer."""
        try:
            parts = []
            for arg in event.get("args", []):
                if "value" in arg:
                    parts.append(str(arg["value"]))
                elif "description" in arg:
                    parts.append(str(arg["description"]))
                else:
                    parts.append(str(arg.get("type", "?")))
            where = ""
            frames = (event.get("stackTrace") or {}).get("callFrames") or []
            if frames and frames[0].get("url"):
                where = f"{frames[0]['url']}:{frames[0].get('lineNumber', 0)}"
            record.console.append(
                {
                    "level": str(event.get("type", "log")),
                    "text": " ".join(parts)[:_MESSAGE_TEXT_MAX],
                    "where": where[:_URL_TEXT_MAX],
                }
            )
        except Exception:  # a driver event must never break the driver loop
            logger.debug("Could not record a console message", exc_info=True)

    def _on_exception_thrown(self, record: _PageRecord, event: dict[str, Any]) -> None:
        """Record one Runtime.exceptionThrown event as a pageerror entry of its page."""
        try:
            details = event.get("exceptionDetails", {})
            exception = details.get("exception") or {}
            # The description starts with the error's own line and then
            # holds the stack; the first line is the message.
            text = str(exception.get("description") or details.get("text", "Uncaught"))
            text = text.splitlines()[0] if text else "Uncaught"
            where = ""
            if details.get("url"):
                where = f"{details['url']}:{details.get('lineNumber', 0)}"
            record.console.append(
                {
                    "level": "pageerror",
                    "text": text[:_MESSAGE_TEXT_MAX],
                    "where": where[:_URL_TEXT_MAX],
                }
            )
        except Exception:  # a driver event must never break the driver loop
            logger.debug("Could not record a page error", exc_info=True)

    def _on_request(self, record: _PageRecord, request: Any) -> None:
        """Record an outgoing request of a page; completed by finished/failed events."""
        try:
            entry: dict[str, Any] = {
                "method": str(request.method),
                "url": str(request.url)[:_URL_TEXT_MAX],
                "type": str(request.resource_type),
                "status": None,
                "error": "",
            }
            record.network.append(entry)
            record.pending[request] = entry
            # The ring buffer drops old entries; the pending index must
            # not keep them (and their request objects) alive forever.
            while len(record.pending) > _NETWORK_LOG_LIMIT:
                record.pending.pop(next(iter(record.pending)))
        except Exception:  # a driver event must never break the driver loop
            logger.debug("Could not record a network request", exc_info=True)

    def _on_response(self, record: _PageRecord, response: Any) -> None:
        """Record the response's status; the request may still fail mid-body."""
        try:
            entry = record.pending.get(response.request)
            if entry is not None:
                entry["status"] = response.status
        except Exception:  # a driver event must never break the driver loop
            logger.debug("Could not record a network response", exc_info=True)

    def _on_request_finished(self, record: _PageRecord, request: Any) -> None:
        """Retire the request's entry: it completed, body included."""
        try:
            record.pending.pop(request, None)
        except Exception:  # a driver event must never break the driver loop
            logger.debug("Could not retire a network request", exc_info=True)

    def _on_request_failed(self, record: _PageRecord, request: Any) -> None:
        """Complete the request's entry with its failure reason."""
        try:
            entry = record.pending.pop(request, None)
            if entry is not None:
                entry["error"] = request.failure or "failed"
        except Exception:  # a driver event must never break the driver loop
            logger.debug("Could not record a network failure", exc_info=True)

    def _on_page_crash(self, _page: Any = None) -> None:
        """Handle a renderer (page) crash without dropping the browser reference.

        When only the page's renderer sub-process dies, the main browser
        process is still alive.  We clear ``_page`` and ``_elements`` but
        keep ``_context`` and ``_browser`` so that
        :meth:`_close_browser_only` can shut down the main process cleanly
        instead of leaking it.

        Crash handlers stay armed on every page ever adopted (they are
        never removed), so this also fires when a BACKGROUND tab
        crashes; only a crash of the CURRENT page may clear the active
        page state — guard by identity, otherwise a background-tab
        crash would trigger a full teardown + relaunch of a healthy
        session.
        """
        if _page is not None and _page is not self._page:
            return
        self._page = None
        self._elements = []

    def _on_browser_lost(self, _obj: Any = None) -> None:
        """Drop page/context/browser references after a browser exit or context close.

        Called when the browser main process exits (``context.on("close")``)
        or the Browser tab's connection drops (``browser.on("disconnected")``).
        The Playwright driver (``self._playwright``) is kept running so that the
        next tool call can launch a fresh browser without paying the driver
        start-up again (:meth:`close` is what stops it).

        The listeners stay armed on every context and browser ever
        attached, so the event of a replaced one must not clear the
        references to its successor: only the event of the current
        context or browser (or a call without an object) drops them.
        """
        if _obj is not None and _obj is not self._browser and _obj is not self._context:
            return
        self._page = None
        self._context = None
        self._browser = None
        self._elements = []
        # The pages these records belonged to are gone with the browser
        # (their CDP sessions die with it; no detach needed).
        self._page_records = {}

    def _attach_live(self) -> None:
        """Open a tab in the Browser tab's browser and adopt its page over CDP.

        The daemon's service creates the tab (so every surface shows it)
        and returns the page's target id; this tool connects a second
        Playwright client to the same browser and picks that page.  A tab
        the user closed, or a browser that exited, is replaced by a fresh
        tab on the next call, and so is one whose page crashed: that tab
        is still open, so it is closed here, once the new one exists (the
        browser exits with its last tab).  A tab this client fails to
        pick up and adopt as its own, or stops picking up because a Stop
        landed (the connect can take seconds), is closed again so it does
        not linger blank in front of the user, and a page adopted halfway
        is dropped with it.  The tab is taken inside that clean-up: once
        ``open_for_agent`` has returned, the service's own clean-up no
        longer runs.
        """
        playwright = self._ensure_playwright()
        tab = page = None
        crashed = self._live_tab_id()
        try:
            tab = self._live_browser.open_for_agent()
            if crashed is not None:
                self._live_browser.close(crashed)  # a no-op for a tab that is gone
            if not self._live_connected(tab.cdp_url):
                self._disconnect_live()
                self._browser = self._driver_call(
                    partial(
                        playwright.chromium.connect_over_cdp,
                        tab.cdp_url,
                        timeout=_LAUNCH_TIMEOUT_MS,
                    ),
                    _LAUNCH_TIMEOUT_MS,
                    "Connecting to the Browser tab",
                )
                self._browser.on("disconnected", self._on_browser_lost)
                self._live_cdp_url = tab.cdp_url
            self._context = self._browser.contexts[0]
            page = self._find_live_page(tab.target_id)
            self._adopt_page(page)
            self._live_tab = tab.tab_id
        except BaseException:
            if tab is not None:
                self._live_browser.close(tab.tab_id)
            if page is not None and self._page is page:
                self._page = self._live_tab = self._live_target = None
            raise

    def _live_connected(self, cdp_url: str) -> bool:
        """Whether the CDP connection to the Browser tab's browser is still usable.

        The browser the daemon runs now listens at *cdp_url*, whose path
        carries an id minted at launch; a relaunched one (the last tab
        closed, a crash) has another, so a connection to a different
        endpoint is stale even before this client has heard that it
        dropped.  No round trip: there is none that a browser exiting at
        that moment is guaranteed to fail (see :meth:`_is_alive`).
        """
        return (
            self._browser is not None
            and self._live_cdp_url == cdp_url
            and bool(self._browser.is_connected())
        )

    def _target_id(self, page: Any) -> str:
        """Chromium's target id of *page* (what the daemon's service knows it by).

        Read over the page's CDP session (:meth:`_page_session`), which
        answers at once even when a script has wedged the renderer.  The
        read is bounded by :func:`_cdp_call` like the attach: it has no
        timeout of its own, and the driver never fails it when the
        browser stops answering or exits under it.

        Raises:
            TimeoutError: The browser did not answer within
                :data:`_CDP_CALL_TIMEOUT_SECS`.
        """
        info = _cdp_call(page, self._page_session(page).send("Target.getTargetInfo"))
        return str(info["targetInfo"]["targetId"])

    def _find_live_page(self, target_id: str) -> Any:
        """Return the shared context's page whose Chromium target id is *target_id*.

        The page was created by the daemon's client, so this client learns
        about it through a ``page`` event; the scan repeats until it
        arrives.  Pages that close mid-scan are skipped.
        """
        deadline = time.monotonic() + _LIVE_PAGE_TIMEOUT
        while True:
            for page in list(self._context.pages):
                try:
                    found = self._target_id(page) == target_id
                except Exception:  # noqa: BLE001 - the page closed under us
                    continue
                if found:
                    return page
            if time.monotonic() > deadline:
                raise RuntimeError("The Browser tab's page did not show up in this client.")
            try:
                self._context.wait_for_event("page", timeout=200)
            except Exception:  # noqa: BLE001 - timed out: rescan
                logger.debug("Exception caught", exc_info=True)

    def _live_tab_id(self) -> str | None:
        """The Browser-tab id of the current page, resolved lazily.

        Following a popup (:meth:`_check_for_new_tab`) moves ``_page`` to
        a page the daemon announced on its own; its tab id is looked up
        from the page's target id when first needed (closing it, or
        interrupting a hung script).  The target id read at adoption
        still names the tab once the page is gone: a popup that crashed
        before the lookup is closed through it.
        """
        if self._live_tab is None and (self._live_target or self._page is not None):
            try:
                target = self._live_target or self._target_id(self._page)
                self._live_tab = self._live_browser.tab_for_target(target)
            except Exception:  # noqa: BLE001 - the page is gone
                logger.debug("Exception caught", exc_info=True)
        return self._live_tab

    def _detach_live(self, close_tab: bool) -> None:
        """Disconnect from the Browser tab's browser, which keeps running.

        Args:
            close_tab: Close this tool's page on every surface too.  False
                leaves it in front of the user (end of a task).
        """
        if close_tab and self._is_alive():
            try:
                # Bounded like the liveness probe: the close is answered
                # by the browser, which may have stopped answering.
                _cdp_call(self._page, self._page._impl_obj.close())
            except Exception:  # noqa: BLE001 - closed by the user meanwhile, or not answering
                logger.debug("Could not close the Browser tab's page", exc_info=True)
        if _driver_is_dead(self._playwright):
            # The liveness probe (or the close) was the first to learn that
            # the driver died; disconnecting through it would never return.
            self._discard_dead_driver(close_live_tab=close_tab)
            return
        if close_tab and self._page is None and self._live_tab_id() is not None:
            # The page crashed (or went with its browser: a no-op then);
            # only the daemon's service can still close its tab.
            self._live_browser.close(self._live_tab)
        self._live_tab = None
        self._live_target = None
        self._disconnect_live()
        self._on_browser_lost()

    def _disconnect_live(self) -> None:
        """Drop this client's connection to the Browser tab's browser, if any.

        A ``Browser`` object left behind without ``close()`` keeps its
        ``disconnected`` listener and its place in the driver; closing a
        connected-over-CDP browser only disconnects this client, the
        daemon's browser keeps running.  The disconnect waits for the
        browser's side of the socket to close, so it is bounded like the
        page close; the driver finishes it on its own.
        """
        browser, self._browser = self._browser, None
        if browser is not None:
            try:
                _cdp_call(browser, browser._impl_obj.close())
            except Exception:  # noqa: BLE001 - the browser may already be gone
                logger.debug("Could not disconnect from the Browser tab", exc_info=True)

    def _close_browser_only(self, keep_live_tab: bool = False) -> None:
        """Close context/browser if present, leaving self._playwright running.

        In the Browser tab (``_live``) only this client is detached; the
        daemon's browser is the user's and stays up, its tab closed
        unless *keep_live_tab*.

        A failed graceful close (wedged driver connection, cross-thread
        greenlet error) is logged at WARNING and followed by
        :meth:`_kill_browser_process`, which guarantees the Chromium OS
        process actually exits instead of leaking forever.  A watchdog
        timer covers the remaining failure mode: a graceful close that
        HANGS (never returns) — after ``_CLOSE_WATCHDOG_SECS`` it kills
        the browser process directly, which also unwedges the hung call
        (the driver instead, when no pid was captured at launch: the
        browser exits with it).  The hang that kill cannot unwedge, a
        driver that no longer answers at all, ends at the loop's own
        deadline on the close (:func:`_cdp_call`, a round trip after the
        watchdog): the driver is then discarded whole, as one that has
        itself exited is without any close being sent
        (:meth:`_discard_dead_driver`).
        """
        if self._attached_url is not None:
            # A browser this tool attached to over CDP is the user's:
            # closing a connected-over-CDP browser only drops this
            # client, and no PID was ever captured, so nothing here can
            # kill it.  Its context must not be closed either (that is
            # the user's real default profile, tabs included).  This
            # branch comes FIRST: a stale _live flag must never route the
            # user's tab into _detach_live's page.close().  A driver that
            # died is discarded whole like in the launched case: a death
            # while idle is first noticed by the disconnect itself, which
            # fails (a second call into it would never return), so the
            # check is repeated after it.
            browser, self._browser = self._browser, None
            self._attached_url = None
            if browser is not None and not _driver_is_dead(self._playwright):
                try:
                    _cdp_call(browser, browser._impl_obj.close())
                except TimeoutError:
                    self._discard_dead_driver(
                        close_live_tab=False,
                        why="did not answer the disconnect from the attached Chrome",
                    )
                    return
                except Exception:  # noqa: BLE001 - the browser may already be gone
                    logger.debug("Could not disconnect from the attached Chrome", exc_info=True)
            if _driver_is_dead(self._playwright):
                self._discard_dead_driver(close_live_tab=False)
                return
            self._on_browser_lost()
            return
        if _driver_is_dead(self._playwright):
            self._discard_dead_driver(close_live_tab=not keep_live_tab)
            return
        if self._live:
            self._detach_live(close_tab=not keep_live_tab)
            return
        pid = self._browser_pid
        identity = self._browser_identity
        watchdog: _Watchdog | None = None
        if self._context is not None or self._browser is not None:
            if _killable_live_pid(pid):
                watchdog = _Watchdog(_CLOSE_WATCHDOG_SECS, _watchdog_kill, args=(pid, identity))
            else:
                watchdog = _Watchdog(
                    _CLOSE_WATCHDOG_SECS,
                    _kill_driver,
                    args=(self._playwright, _CLOSE_WATCHDOG_SECS, "Graceful browser close"),
                )
        try:
            if watchdog is not None:
                watchdog.start()
            for obj in (self._context, self._browser):
                if obj is None:
                    continue
                if _driver_is_dead(self._playwright):
                    # A driver that died while the tool was idle is first
                    # noticed by the close before this one, which fails;
                    # the next call into it would never return.
                    self._discard_dead_driver(close_live_tab=False)
                    return
                try:
                    # The watchdog fires first; a driver that answers fails
                    # the close within a round trip of its kill.
                    _cdp_call(
                        obj,
                        obj._impl_obj.close(),
                        timeout_secs=_CLOSE_WATCHDOG_SECS + _CDP_CALL_TIMEOUT_SECS,
                    )
                except TimeoutError:
                    self._discard_dead_driver(
                        close_live_tab=False, why="did not answer the graceful browser close"
                    )
                    return
                except Exception:
                    logger.warning(
                        "Graceful browser close failed; killing the Chromium "
                        "process if it survived",
                        exc_info=True,
                    )
        finally:
            if watchdog is not None:
                watchdog.end()
        self._kill_browser_process()
        self._on_browser_lost()

    def _discard_dead_driver(
        self, close_live_tab: bool, why: str = "exited under the browser tool"
    ) -> None:
        """Drop a Playwright whose driver process died, or stopped answering, under the tool.

        No graceful close goes to a dead driver (it would never return,
        see :func:`_driver_is_dead`): Chromium is killed directly when
        this tool launched it (in the Browser tab the browser is the
        daemon's and no pid is recorded), every reference is dropped and
        the driver stopped, so the next call starts a fresh driver and
        browser instead of wedging the task thread forever.  ``stop()``
        on a dead driver returns at once (its transport already finished);
        off the starting thread :func:`_stop_playwright` reaps it, and
        a driver that is alive but no longer answers is killed by its
        bounded stop.  A Stop landing on the way still stops the driver,
        which stays ``self._playwright`` until that stop has returned,
        as the browser's pid stays recorded until its kill has
        (:meth:`_kill_browser_process`): what a Stop leaves undone,
        :meth:`close` finds.

        Args:
            close_live_tab: In the Browser tab, also close this tool's tab
                on every surface -- through the daemon's service, the only
                way left without a driver.  False leaves it in front of
                the user (end of a task).
            why: What the driver did, for the log.
        """
        logger.warning("Playwright driver %s; discarding it and relaunching on the next call", why)
        playwright = self._playwright
        live_tab, self._live_tab = self._live_tab, None
        target, self._live_target = self._live_target, None
        try:
            if self._live and close_live_tab:
                try:
                    if live_tab is None and target is not None:
                        # A popup followed just before the driver died: the
                        # daemon knows the tab by the target id read at
                        # adoption -- from the moment its own client has
                        # attached to the popup, which may still be under way.
                        deadline = time.monotonic() + _LIVE_TAB_LOOKUP_SECS
                        while (
                            live_tab := self._live_browser.tab_for_target(target)
                        ) is None and time.monotonic() < deadline:
                            time.sleep(0.05)
                    if live_tab is None:
                        logger.warning(
                            "The Browser tab of the dead driver's page is unknown; it stays"
                        )
                    else:
                        self._live_browser.close(live_tab)
                except Exception:  # noqa: BLE001 - closed by the user meanwhile
                    logger.debug("Could not close the Browser tab's page", exc_info=True)
            self._kill_browser_process()
        finally:
            self._on_browser_lost()
            _stop_playwright(playwright)
            self._playwright = None

    def _kill_browser_process(self) -> None:
        """Ensure the recorded Chromium OS process is dead, escalating to signals.

        Called after the graceful Playwright close attempts.  Waits briefly
        for a clean exit, then sends SIGTERM and finally SIGKILL (after an
        identity check so a recycled PID is never signalled).  Without
        this, a close whose driver call raised (wedged connection,
        cross-thread greenlet error) silently leaked the Chromium process —
        the root cause of long-horizon tasks accumulating open browsers.
        The pid stays recorded until the process is known to be gone or
        killed: a Stop landing in the wait leaves it to the next call.
        """
        pid = self._browser_pid
        identity = self._browser_identity
        if _killable_live_pid(pid) and not _wait_pid_exit(pid, 2.0):
            logger.warning("Chromium (pid %d) survived graceful close", pid)
            _terminate_pid_escalating(pid, identity)
        self._browser_pid = None
        self._browser_identity = None

    def _capture_browser_pid(self) -> None:
        """Record the OS PID of the just-launched Chromium main process.

        Read over a browser-level CDP session (``SystemInfo.getProcessInfo``),
        which a persistent context exposes through its ``browser`` as well.
        A recorded PID lets :meth:`_kill_browser_process` guarantee the
        process dies even when the graceful Playwright close fails.  The
        round trips go through :func:`_cdp_call`: the sync CDP API has
        no deadline, and a Chromium that wedges right after launch would
        otherwise pin the task thread here forever.
        """
        self._browser_pid = None
        self._browser_identity = None
        browser = self._browser or self._context.browser
        try:
            cdp = _cdp_call(browser, browser._impl_obj.new_browser_cdp_session())
            try:
                info = _cdp_call(browser, cdp.send("SystemInfo.getProcessInfo"))
            finally:
                try:
                    _cdp_call(browser, cdp.detach())
                except Exception:  # pragma: no cover — detach rarely fails
                    logger.debug("CDP detach failed", exc_info=True)
            for proc in info.get("processInfo", []):
                if proc.get("type") == "browser":
                    self._browser_pid = int(proc["id"])
                    self._browser_identity = _process_identity(self._browser_pid)
                    return
        except Exception:  # pragma: no cover — CDP rarely fails
            logger.debug("CDP browser PID capture failed", exc_info=True)

    def _ensure_browser(self) -> None:
        """Ensure a Playwright browser page is ready, installing Chromium if needed.

        Detects and recovers from a previously-crashed Chromium by tearing down
        stale references and relaunching. This handles the common case where
        "Google Chrome for Testing quit unexpectedly" leaves the tool with a
        dead page that would otherwise fail every subsequent call.
        """
        if self._is_alive():
            return
        if _driver_is_dead(self._playwright):
            # Before anything consults the dead driver's local state
            # (``context.pages``, a Browser-tab connection that still
            # reads as connected): drop it, a fresh one starts below.  The
            # Browser tab it drove is closed, ``_attach_live`` opens a new
            # one rather than leaving the old one adrift in front of the
            # user.
            self._discard_dead_driver(close_live_tab=True)
        if self._live:
            # The user closed the tab (or it crashed, or the browser exited):
            # open a new one.
            self._attach_live()
            return
        if self._page is not None and self._context is not None:
            try:
                pages = [p for p in self._context.pages if not p.is_closed()]
            except Exception:  # pragma: no cover — context already dead
                logger.debug("Exception caught", exc_info=True)
                pages = []
            if self._attached_url is not None:
                # The user's other tabs, never their browser's own UI.
                pages = [p for p in pages if not _is_browser_ui(p)]
            if pages:
                self._adopt_page(pages[-1])
                return
        if self._attached_url is not None:
            # The driven tab's renderer crashed (only the page was dropped;
            # the connection and the user's other tabs are fine), or the
            # user's own browser exited or the connection dropped:
            # relaunching the stealth Chromium instead would silently swap
            # the logged-in session the agent was driving for a synthetic
            # profile, so fail with the way back.
            if self._page is None and self._context is not None:
                lost = "The attached tab crashed"
            elif self._page is not None and self._context is not None:
                lost = (
                    "The attached tab was closed and no other tab is open in that Chrome; "
                    "open one there"
                )
            else:
                lost = f"The Chrome at {self._attached_url} disconnected"
            raise RuntimeError(
                f'{lost}. Reconnect with attach_to_chrome("{self._attached_url}"), or call '
                "close_browser() to return to the built-in Chromium."
            )
        atexit.unregister(self.close)
        atexit.register(self.close)
        self._close_browser_only()

        # A launch that shows no window never raises one, so there is no
        # focus to save and restore.
        prev_app = None if self._headless else _get_frontmost_app()
        try:
            launcher = self._ensure_playwright().chromium
            # "No window" is implemented as a headed Chromium on a private
            # Xvfb display when one can be started: headless mode is the
            # first thing bot-protection scripts detect.  Real headless
            # is the fallback (macOS, Windows, Linux without Xvfb).
            display = web_stealth.virtual_display() if self._headless else None
            self._chromium_headless = self._headless and display is None
            self._user_agent_relearned = False
            channel = web_stealth.chrome_channel()
            kwargs: dict[str, Any] = {
                "headless": self._chromium_headless,
                # Google Chrome when installed; otherwise the "chromium"
                # channel selects the full Chromium binary, which headless
                # runs in Chrome's new headless mode (the same renderer as
                # a headed window) instead of the stripped-down
                # chrome-headless-shell.
                "channel": channel,
                "args": [
                    web_stealth.disabled_blink_features(unattended=self._headless),
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-breakpad",
                    "--noerrdialogs",
                    "--disable-dev-shm-usage",
                    f"--window-size={self.viewport[0]},{self.viewport[1]}",
                    # Without a GPU Chromium blocklists GL and ships no
                    # WebGL at all, which almost no human browser lacks
                    # and challenge scripts test first.  Off the
                    # blocklist it renders through Mesa's llvmpipe like
                    # any Linux desktop without a GPU.
                    "--ignore-gpu-blocklist",
                ],
            }
            if display is not None:
                kwargs["env"] = {**os.environ, "DISPLAY": display}
            # Real headless mode reports ``HeadlessChrome``; the headed
            # string of this build, once known, is set with Chrome's own
            # switch (``_launch_browser`` learns it on the first launch).
            switch = web_stealth.user_agent_switch(channel) if self._chromium_headless else None
            if switch is not None:
                kwargs["args"].append(switch)

            try:
                self._launch_browser(launcher, kwargs)
            except Exception as exc:  # pragma: no cover – Chromium pre-installed in CI
                message = str(exc)
                if (
                    "Executable doesn't exist" not in message
                    and "playwright install" not in message
                    and "patchright install" not in message
                ):
                    # Profile locks, missing display libraries, resource
                    # exhaustion, etc. — installing Chromium would not
                    # help and would only bury the real error.
                    raise
                logger.info("Playwright Chromium not found, installing...")
                self._close_browser_only()
                _install_chromium()
                self._launch_browser(launcher, kwargs)
            self._resume_stranded_session()
        except Exception:  # pragma: no cover — Playwright init failure
            # Only the half-launched browser goes; the driver, the atexit
            # hook and an ephemeral profile stay for the next attempt.
            self._close_browser_only()
            raise
        finally:
            _activate_app(prev_app)

    def _clean_singleton_locks(self, profile_dir: str) -> None:
        """Remove stale Singleton* files from a previously crashed Chromium.

        Chromium writes Singleton{Lock,Cookie,Socket} when a persistent profile
        is opened. If the process dies without cleaning up, the next launch
        may fail or crash. Safe to call unconditionally — live Chromium
        recreates the files during startup.

        Args:
            profile_dir: The profile directory to clean.
        """
        for name in _SINGLETON_FILES:
            try:
                (Path(profile_dir) / name).unlink(missing_ok=True)
            except OSError:  # pragma: no cover — race with another launch
                logger.debug("Exception caught", exc_info=True)

    def _resolve_user_data_dir(self) -> str | None:
        """Return a profile directory not locked by another Chromium process.

        If ``self.user_data_dir`` is ``None``, returns ``None`` (non-persistent).
        If the configured directory is already locked by a live Chromium,
        numbered variants (``<dir>_1``, ``<dir>_2``, …) are tried until a
        free one is found.  Crashed or leaked Chromiums leave such
        variants behind with dead ``SingletonLock`` PIDs; every one of
        those is deleted on the way (whether or not the base profile is
        free) so escalation dirs cannot accumulate across crash/relaunch
        cycles.  Live profiles and lock-less directories are never
        touched.  An ephemeral profile is private to this process and
        is used as is.

        Returns:
            An available profile directory path, or ``None`` to fall back to
            a non-persistent (temporary) context.
        """
        if not self.user_data_dir:
            return None
        if self._ephemeral:
            return self.user_data_dir
        chosen = None if _profile_lock_state(self.user_data_dir)[1] else self.user_data_dir
        for i in range(1, 100):
            candidate = f"{self.user_data_dir}_{i}"
            pid, in_use = _profile_lock_state(candidate)
            if pid is not None and not in_use:
                _rmtree_logged(candidate)
            if chosen is None and not in_use:
                chosen = candidate
        return chosen

    def _profile_lock(self) -> Any:
        """Return a machine-wide lock over this tool's profile family.

        The profile directory (and its ``_N`` escalation variants) is
        shared by every kiss process on the machine, so the
        check-then-use sequence in :meth:`_launch_browser` must be
        atomic across *processes*: otherwise two of them both see a
        lock-free profile, the second deletes the first's live
        ``SingletonLock``, and Chromium either opens one profile twice
        (corrupting the stored logins) or aborts with "Failed to create
        a ProcessSingleton for your profile directory".  Threads of one
        process are excluded the same way: each holder opens its own
        file description.

        One lock file sits beside the base profile and therefore covers
        every escalation variant derived from it.

        The wait is bounded by ``_PROFILE_LOCK_TIMEOUT_SECS`` so it polls
        in Python, where a Stop injected into the task thread lands at
        once; an unbounded wait sits in one kernel ``flock`` call that no
        Stop can interrupt while a holder wedged in Playwright keeps the
        file.  On timeout the ``TimeoutError`` (naming the lock file)
        surfaces as the tool's ``Error <context>: ...`` string.

        Returns:
            A context manager holding the lock, or a no-op context
            manager when this tool uses no persistent profile or an
            ephemeral one (nobody else can open it, and the lock file
            would outlive the directory).
        """
        if not self.user_data_dir or self._ephemeral:
            return nullcontext()
        return exclusive_file_lock(
            Path(f"{self.user_data_dir}.lock"), timeout=_PROFILE_LOCK_TIMEOUT_SECS
        )

    def _launch_browser(self, launcher: Any, kwargs: dict[str, Any]) -> None:
        if self._ephemeral and self._ephemeral_dir is None:
            self._ephemeral_dir = tempfile.mkdtemp(prefix="kiss_web_profile_")
            self.user_data_dir = self._ephemeral_dir
        with self._profile_lock():
            effective_dir = self._resolve_user_data_dir()
            if effective_dir:
                Path(effective_dir).mkdir(parents=True, exist_ok=True)
                self._clean_singleton_locks(effective_dir)
                launch = partial(
                    launcher.launch_persistent_context,
                    effective_dir,
                    **kwargs,
                    **self._context_args(),
                    timeout=_LAUNCH_TIMEOUT_MS,
                )
                self._context = self._driver_call(launch, _LAUNCH_TIMEOUT_MS, "Browser launch")
                self._capture_browser_pid()
                page = self._context.pages[0] if self._context.pages else self._new_page()
            else:
                launch = partial(launcher.launch, **kwargs, timeout=_LAUNCH_TIMEOUT_MS)
                self._browser = self._driver_call(launch, _LAUNCH_TIMEOUT_MS, "Browser launch")
                self._capture_browser_pid()
                self._context = self._driver_call(
                    partial(self._browser.new_context, **self._context_args()),
                    _ACTION_TIMEOUT_MS,
                    "Opening a browser context",
                )
                page = self._new_page()
        self._driver_call(
            partial(self._context.route, _ACCOUNTS_GOOGLE_URL_RE, _abort_route),
            _ACTION_TIMEOUT_MS,
            "Installing a route",
        )
        self._context.on("close", self._on_browser_lost)
        self._adopt_page(page)
        self._settle_fresh_window()
        if self._chromium_headless:
            self._reconcile_headless_user_agent(launcher, kwargs)

    def _settle_fresh_window(self) -> None:
        """Make the fresh window show its blank page before the agent's first page.

        Observed: after a watchdog kill and relaunch on Xvfb, a first
        page whose ``mousemove`` handler spins forever
        (``test_webuse_unresponsive_page_timeout``'s ``/wheelwedge``)
        wedged the renderer before any tool-issued input, so every later
        read timed out — unless the fresh window's ``about:blank`` was
        read first.  Any renderer round trip on the blank page (an
        evaluation, ``title()``, a screenshot) prevents it; a sleep or a
        ``goto("about:blank")`` does not.  The suspected mechanism is one
        stray ``mousemove`` from the window system (the Xvfb pointer sits
        at the screen centre, over the window) delivered to the first
        document the window renders; this bounded ``_read_page`` makes
        that document the blank page rather than the agent's first page.
        Until the native user-agent switch, the launch-time
        ``navigator.userAgent`` read did this by accident in every mode.
        """
        try:
            self._read_page(self._page, "() => [document.readyState]")
        except Exception:  # pragma: no cover — a fresh blank page rarely fails to answer
            logger.debug("Could not settle the fresh browser window", exc_info=True)

    def _reconcile_headless_user_agent(self, launcher: Any, kwargs: dict[str, Any]) -> None:
        """Relaunch a real-headless browser so it reports its headed user agent.

        Headless Chromium advertises ``HeadlessChrome/<version>``, which
        sites treat as a bot signal.  Rewriting it from inside the page
        is a tell of its own (``navigator.appVersion`` keeps the raw
        token, the redefined property is detectable), and Cloudflare's
        managed challenge rejected every such browser; Chrome's own
        ``--user-agent`` switch sets the header, ``navigator.userAgent``
        and ``navigator.appVersion`` consistently (see
        :func:`web_stealth.headed_user_agent`).  The headed string is only
        known once a browser of this build has reported its own, so the
        first headless launch in the process is asked and relaunched;
        later launches get the switch up front
        (:func:`web_stealth.user_agent_switch`).  A launch whose switch
        names another major version than the browser (Chrome updated
        under the running daemon) forgets the string and relaunches
        without the switch, which learns the new one — once per launch,
        so a build whose own string never matches its version is kept
        rather than relaunched forever.

        Args:
            launcher: The Playwright browser type that launched the browser.
            kwargs: The launch arguments, extended with (or stripped of)
                the switch for the relaunch.
        """
        channel = str(kwargs["channel"])
        switch = next((arg for arg in kwargs["args"] if arg.startswith("--user-agent=")), None)
        if switch is not None:
            version = self._browser_version()
            if (
                self._user_agent_relearned
                or version is None
                or web_stealth.user_agent_matches_version(switch, version)
            ):
                return
            logger.info("remembered user agent is for another Chromium build; relearning")
            self._user_agent_relearned = True
            web_stealth.forget_user_agent(channel)
            self._close_browser_only()
            kwargs["args"] = [arg for arg in kwargs["args"] if arg != switch]
            self._launch_browser(launcher, kwargs)
            return
        try:
            # Bounded like every other page read (``_read_page``), not a
            # bare ``evaluate`` with no timeout.
            user_agent = str(self._read_page(self._page, "() => [navigator.userAgent]")[0])
        except Exception:  # pragma: no cover — evaluate on a fresh page rarely fails
            logger.debug("Could not read the headless user agent", exc_info=True)
            return
        headed = web_stealth.headed_user_agent(user_agent)
        if headed is None:
            return
        web_stealth.remember_user_agent(channel, headed)
        logger.info("relaunching headless Chromium with its headed user agent")
        self._close_browser_only()
        kwargs["args"] = [*kwargs["args"], f"--user-agent={headed}"]
        self._launch_browser(launcher, kwargs)

    def _browser_version(self) -> str | None:
        """Return the running browser's version string, or ``None`` if unknown."""
        try:
            browser = self._browser
            if browser is None and self._context is not None:
                browser = self._context.browser
            return None if browser is None else str(browser.version)
        except Exception:  # pragma: no cover — driver-side failure only
            logger.debug("Could not read the browser version", exc_info=True)
            return None

    @staticmethod
    def _page_title(page: Any) -> str:
        """Return *page*'s title without hanging on an unresponsive renderer.

        Playwright's ``Page.title()`` accepts no timeout: it waits for the
        frame's execution context to answer, which never happens while the
        renderer main thread is busy (e.g. a page script that spins forever
        after ``goto`` timed out on ``domcontentloaded``). That froze a
        task inside ``get_page_content`` indefinitely.
        ``Locator.evaluate(timeout=...)`` is not a fix either: its timeout
        bounds only element-handle resolution, so a page that shadows
        ``document.title`` with a never-returning getter hangs the
        evaluation phase forever.  ``wait_for_function``'s timeout is
        enforced by the driver-side progress controller for the whole
        operation — it raises ``TimeoutError`` on schedule even while the
        renderer is wedged inside the evaluated expression.  The title is
        wrapped in a list so an empty title is still truthy and resolves
        immediately; ``polling=100`` (not the default ``raf``) so
        throttled background tabs still answer.  Works for SVG and other
        non-HTML documents (``document.title`` is defined for them).  The
        follow-up ``json_value()`` is bounded by :meth:`_read_page`: a
        getter that answers at once with an object whose own getter
        never returns passes the first wait and wedges the serialization.
        A hostile getter still leaves the renderer spinning after the
        bounded error (its loop never exits); ``close_browser()`` is the
        agent's escape hatch, exactly as for any other wedged page.
        """
        return str(WebUseTool._read_page(page, "() => [document.title]")[0])

    @staticmethod
    def _read_page(page: Any, expression: str, arg: Any = None) -> Any:
        """Evaluate *expression* (a function returning a non-empty list) on *page*.

        ``wait_for_function`` (not ``evaluate``) bounds the read (see
        :meth:`_page_title`); ``polling=100`` so throttled background
        tabs still answer.  It also survives navigations: Playwright
        re-arms it in the new document.  The follow-up ``json_value()``
        does not.  A page that navigates itself — a Cloudflare challenge
        reloading the real page once its checks pass, a redirecting
        landing page — can destroy the execution context between the two
        calls, which surfaced as ``Error navigating to ...:
        JSHandle.json_value: Execution context was destroyed``.  The read
        is then repeated on the new document, following up to
        ``_PAGE_READ_NAVIGATIONS`` navigations in a row (a challenge page
        that reloads into the real page, which then redirects again).

        ``json_value()`` is a second renderer round trip with no deadline
        of its own: serializing the answer runs the page's getters, and a
        value whose getter never returns (or a renderer that wedged in
        between) would hang it forever.  It runs under what is left of
        the read's deadline (never less than :data:`_CDP_CALL_TIMEOUT_SECS`)
        through :func:`_cdp_call`, and the handle it leaves in the driver
        is released with the same bound (:func:`_dispose_handle`).

        Args:
            page: The Playwright page to read.
            expression: JavaScript function source returning a list.
            arg: Optional argument passed to the function.

        Returns:
            The list the expression returned, as Python values.

        Raises:
            TimeoutError: The serialization of the answer did not finish
                within the read's deadline.
        """
        for attempt in range(_PAGE_READ_NAVIGATIONS + 1):
            deadline = time.monotonic() + _PAGE_READ_TIMEOUT_MS / 1000
            try:
                handle = _wait_for_function(page, expression, arg, _PAGE_READ_TIMEOUT_MS)
                try:
                    return _cdp_call(
                        page,
                        handle.json_value(),
                        timeout_secs=max(deadline - time.monotonic(), _CDP_CALL_TIMEOUT_SECS),
                    )
                except TimeoutError:
                    raise TimeoutError(
                        f"Timeout {_PAGE_READ_TIMEOUT_MS}ms exceeded: the page did not hand "
                        "over its answer (a getter that never returns?)"
                    ) from None
                finally:
                    _dispose_handle(page, handle)
            except web_stealth.playwright_api().Error as exc:
                # A page that keeps navigating surfaces its error like any other.
                if (
                    "Execution context was destroyed" not in str(exc)
                    or attempt == _PAGE_READ_NAVIGATIONS
                ):
                    raise
                logger.debug(
                    "page navigated under the read; reading the new document", exc_info=True
                )
                page.wait_for_load_state("domcontentloaded", timeout=_PAGE_READ_TIMEOUT_MS)
        raise AssertionError("unreachable")  # pragma: no cover

    def _require_responsive_renderer(self) -> None:
        """Raise ``TimeoutError`` unless the page's renderer answers promptly.

        ``keyboard.press``/``keyboard.type``, ``mouse.move``/``mouse.wheel``
        and ``Locator.count`` have no timeout parameter and block for as
        long as the renderer stays silent. Call this right before them so
        an already-unresponsive page turns into a fast, clean tool error
        (browser kept alive) instead of a tool call that never returns.
        The probe cannot bound the input call itself — a page whose event
        handler wedges the renderer in response to our input passes the
        probe first — so the callers also wrap the input in
        :meth:`_input_hang_watchdog`.
        """
        handle = _wait_for_function(self._page, "() => 1", None, _PAGE_READ_TIMEOUT_MS)
        _dispose_handle(self._page, handle)

    def _input_hang_watchdog(
        self, operation: Callable[[], Any], deadline_secs: float = _INPUT_WATCHDOG_SECS
    ) -> Any:
        """Run *operation*, a raw input call, cut off if it outlives *deadline_secs*.

        Raw input operations (``keyboard.press``/``type``,
        ``mouse.move``/``wheel``) carry no timeout in the Playwright
        protocol, so a page event handler that enters ``while(true)`` on
        ``keydown``/``mousemove``/``wheel`` blocks them forever — after the
        pre-input liveness probe already passed.  Killing the browser
        process unwedges the pending driver call (it raises), the tool
        returns its documented error string, and the next tool call
        relaunches a fresh browser via ``_ensure_browser``.  A page in
        that state is permanently wedged, so losing the session is the
        correct recovery, exactly like the graceful-close watchdog.  In
        a Chrome the user started (``attach_to_chrome``) the tool's own
        Node driver is killed instead (:func:`_kill_driver`): the user's
        browser is never signalled, and the next call reports the lost
        connection with the way back.  The driver is also what goes when
        no browser pid is known (its capture at launch timed out, a
        tolerated failure): the Chromium it launched exits with it.
        Every recovery that leaves the driver alone needs it to end the
        call, so a failsafe kills the driver
        :data:`_INPUT_DRIVER_GRACE_SECS` after the deadline; it is the
        only bound of a Browser-tab page whose tab id is unknown.

        The watchdogs are armed and ended by :func:`_under_timer`: a
        ``with`` block cannot guarantee the cancel against a Stop.

        Args:
            operation: The guarded raw input call.
            deadline_secs: Seconds *operation* may run before the
                watchdog fires.  Callers whose legitimate duration scales
                with input size (``type_text`` delay, ``scroll`` steps)
                pass a proportionally larger deadline.

        Returns:
            *operation*'s result.
        """
        failsafe_secs = deadline_secs + _INPUT_DRIVER_GRACE_SECS
        failsafe = _Watchdog(
            failsafe_secs, _kill_driver, args=(self._playwright, failsafe_secs, "Raw browser input")
        )
        guarded = partial(_under_timer, failsafe, operation)
        if self._attached_url is not None:
            timer = _Watchdog(deadline_secs, _kill_driver, args=(self._playwright, deadline_secs))
        elif self._live:
            # The browser is the user's: the daemon's service stops the
            # hung script (and closes only that tab if it stays wedged,
            # which is this thread's to see once the call is back).
            tab_id = self._live_tab_id()
            if tab_id is None:
                return guarded()
            timer = _Watchdog(deadline_secs, self._live_browser.interrupt, args=(tab_id,))
            result = _under_timer(timer, guarded)
            if timer.acted:
                self._close_tab_left_wedged(tab_id)
            return result
        elif not _killable_live_pid(self._browser_pid):
            timer = _Watchdog(
                deadline_secs,
                _kill_driver,
                args=(self._playwright, deadline_secs, "Raw browser input"),
            )
        else:
            pid, identity = self._browser_pid, self._browser_identity
            timer = _Watchdog(
                deadline_secs,
                _watchdog_kill,
                args=(pid, identity, "Raw browser input", deadline_secs),
            )
        return _under_timer(timer, guarded)

    def _close_tab_left_wedged(self, tab_id: str) -> None:
        """Have the Browser tab *tab_id* closed unless its page answers, its script stopped.

        The service stopped the script that hung a raw input call
        (:meth:`_input_hang_watchdog`), and the call came back.  The
        page may not have: a task the stopped handler had queued hangs
        it again.  The service closes a page that stays wedged only
        while the hung call lasts (afterwards it cannot tell that task
        from the script of this tool's next call, whose page a close
        must not take), so whenever the stop freed the call first the
        wedged tab stayed open for good.  On this thread the next call
        has not begun: what was queued gets
        :data:`_STOPPED_SCRIPT_SETTLE_MS` to start, and a page that
        does not answer a read after that is lost; only its tab goes.

        The service closes the tab on its own loop, where the close
        may wait its turn until this tool's next call is under way:
        the tool gives the tab up first, so that call opens a fresh
        one instead of running on a page about to be closed.

        Raises:
            RuntimeError: The page did not answer; its tab is being closed.
        """
        self._pause(_STOPPED_SCRIPT_SETTLE_MS)
        try:
            self._require_responsive_renderer()
        except (TimeoutError, web_stealth.playwright_api().TimeoutError) as exc:
            self._detach_live(close_tab=False)
            self._live_browser.close(tab_id)
            raise RuntimeError(
                "The page stayed unresponsive after its hung script was stopped; "
                "its Browser tab is being closed."
            ) from exc

    def _ensure_playwright(self) -> Any:
        """The Playwright driver (``self._playwright``), started first when there is none."""
        if self._playwright is None:
            self._start_playwright()
        return self._playwright

    def _start_playwright(self) -> Any:
        """Start a Playwright driver, keep it as ``self._playwright`` and return it.

        ``start()`` returns once the driver it spawned has answered the
        handshake.  One that has not within :data:`_DRIVER_START_SECS` is
        killed (:func:`_kill_driver`), which fails the start instead of
        waiting for good.  A start that does not return (that failure,
        or a Stop) leaves a driver no Playwright object owns and no
        ``close()`` could find: it is ended here.
        """
        manager = web_stealth.playwright_api().sync_playwright()
        # Until the handshake there is no Playwright object: the driver's
        # process hangs off the manager, where ``_kill_driver`` finds it.
        starting = SimpleNamespace(_impl_obj=manager)
        timer = _Watchdog(
            _DRIVER_START_SECS,
            _kill_driver,
            args=(starting, _DRIVER_START_SECS, "Starting the Playwright driver"),
        )
        try:
            self._playwright = _under_timer(timer, manager.start)
        except BaseException:
            _end_driver(starting)
            raise
        return self._playwright

    def _driver_call(self, call: Callable[[], Any], timeout_ms: float, operation: str) -> Any:
        """Run *call*, a request to the driver, ended on this side if the driver stalls.

        The driver enforces a request's timeout, and answers a request
        that has none within moments; one that no longer answers does
        neither, and the sync call waited on it for good.  Such a driver
        is killed a round trip after *timeout_ms* (:func:`_kill_driver`),
        which fails the pending call with its usual error; the next tool
        call relaunches.

        Args:
            call: The sync API call.
            timeout_ms: The request's timeout in milliseconds, the one
                the driver was given with it (how long a request without
                one may take at most).
            operation: Human-readable name of the request, for the log.

        Returns:
            What *call* returned.
        """
        secs = timeout_ms / 1000 + _CDP_CALL_TIMEOUT_SECS
        timer = _Watchdog(secs, _kill_driver, args=(self._playwright, secs, operation))
        return _under_timer(timer, call)

    def _pause(self, ms: float) -> None:
        """Wait *ms* milliseconds while the page's events keep being handled."""
        self._driver_call(partial(self._page.wait_for_timeout, ms), ms, "A pause")

    def _new_page(self) -> Any:
        """Open a tab in the context (a request without a timeout of its own)."""
        return self._driver_call(self._context.new_page, _ACTION_TIMEOUT_MS, "Opening a tab")

    def _goto(self, url: str, wait_until: str = "domcontentloaded") -> Any:
        """Open *url* in the active tab within the navigation's 30 s; return its response."""
        goto = partial(self._page.goto, url, wait_until=wait_until, timeout=30000)
        return self._driver_call(goto, 30000, "Navigation")

    def _move_mouse_to(self, x: float, y: float) -> None:
        """Move the pointer to (*x*, *y*) along a curved, decelerating path.

        Playwright's ``click()`` teleports the pointer onto the target,
        which behavioural bot detectors score as automation.  The path
        starts at the last position this tool moved to (or a random
        point of the viewport right after a launch).  Callers hold
        :meth:`_input_hang_watchdog` because ``mouse.move`` has no timeout.

        Args:
            x: Target x in viewport CSS pixels.
            y: Target y in viewport CSS pixels.
        """
        if self._mouse_xy is None:
            vw, vh = self._viewport_size()
            self._mouse_xy = (
                random.uniform(vw * 0.2, vw * 0.8),
                random.uniform(vh * 0.2, vh * 0.8),
            )
        for px, py in web_stealth.mouse_path(self._mouse_xy, (x, y)):
            self._page.mouse.move(px, py, steps=2)
        self._mouse_xy = (x, y)

    def _human_click(self, locator: Any, hover_only: bool = False) -> None:
        """Click (or only hover) *locator* the way a person does.

        Scrolls the element into view, glides the pointer onto a random
        point near its middle, pauses, then presses for 45-140 ms.  The
        press itself goes through Playwright's ``click``/``hover`` so its
        actionability checks (visible, stable, enabled, receives events)
        still apply; when the chosen point is not hittable within 5 s
        (e.g. the gap between two lines of a wrapped link) the plain
        centre click is used, exactly as before.  Only a ``trial`` click
        (the checks without the press) is bounded that way: a click's
        own timeout also covers the navigation it starts, and a slow
        form submission must not be submitted again at the centre.  The
        real click is issued once.  An element without a rendered box is
        clicked the plain way straight away.  Every one of these requests,
        the click and the hover themselves included, ends on this side
        when the driver stalls (:meth:`_driver_call`).

        Args:
            locator: Playwright locator of the element.
            hover_only: Move onto the element without pressing.
        """
        press_ms = random.randint(45, 140)
        api = web_stealth.playwright_api()
        # Probe before the glide, outside the fallback below: a wedged
        # renderer must surface as a 10 s error, not as a glide failure
        # followed by a 30 s plain click on the same page.
        self._require_responsive_renderer()
        try:
            position = self._glide_onto(locator)
        except api.Error:
            # A navigation in flight (e.g. a site's client-side redirect
            # right after load) destroys the element's context mid-glide;
            # Playwright's plain click below retries through that.
            logger.debug("pointer glide failed; falling back to plain click", exc_info=True)
            position = None
        if position is not None:
            try:
                if hover_only:
                    self._driver_call(
                        partial(locator.hover, position=position, timeout=5000), 5000, "Hover"
                    )
                    return
                self._driver_call(
                    partial(locator.click, position=position, trial=True, timeout=5000),
                    5000,
                    "Click",
                )
            except api.TimeoutError:
                logger.debug("point not hittable; falling back to centre click", exc_info=True)
                position = None
        if hover_only:
            action = partial(locator.hover, timeout=_ACTION_TIMEOUT_MS)
        else:
            action = partial(
                locator.click, position=position, delay=press_ms, timeout=_ACTION_TIMEOUT_MS
            )
        self._driver_call(action, _ACTION_TIMEOUT_MS, "Hover" if hover_only else "Click")

    def _glide_onto(self, locator: Any) -> dict[str, float] | None:
        """Scroll *locator* into view and glide the pointer onto it.

        Args:
            locator: Playwright locator of the element.

        Returns:
            The chosen point as an offset inside the element (for
            ``click(position=...)``), or ``None`` when the element has no
            rendered box.
        """
        self._driver_call(
            partial(locator.scroll_into_view_if_needed, timeout=_PAGE_READ_TIMEOUT_MS),
            _PAGE_READ_TIMEOUT_MS,
            "Scrolling an element into view",
        )
        box = self._driver_call(
            partial(locator.bounding_box, timeout=_PAGE_READ_TIMEOUT_MS),
            _PAGE_READ_TIMEOUT_MS,
            "Reading an element's box",
        )
        if not box or box["width"] < 1 or box["height"] < 1:  # pragma: no cover
            # scroll_into_view_if_needed already waited for visibility;
            # only an element removed between the two calls lands here.
            return None
        ox = box["width"] * random.uniform(0.35, 0.65)
        oy = box["height"] * random.uniform(0.35, 0.65)
        self._input_hang_watchdog(lambda: self._move_mouse_to(box["x"] + ox, box["y"] + oy))
        self._pause(random.randint(40, 160))
        return {"x": ox, "y": oy}

    def _idle_mouse(self) -> None:
        """Nudge the pointer a little, as a person waiting on a page does."""
        vw, vh = self._viewport_size()
        x0, y0 = self._mouse_xy or (vw / 2, vh / 2)
        x = min(max(x0 + random.uniform(-80, 80), 4), vw - 4)
        y = min(max(y0 + random.uniform(-60, 60), 4), vh - 4)
        self._input_hang_watchdog(lambda: self._move_mouse_to(x, y))

    def _challenge_vendor(self) -> str | None:
        """Return the bot-protection vendor whose interstitial is showing, if any.

        The title, the head of the body text and the page's script URLs
        (AWS WAF's blank interstitial is known by its script alone) go to
        :func:`web_stealth.challenge_vendor`.
        """
        title, body_head, scripts = self._read_page(
            self._page,
            "() => [document.title, document.body ? document.body.innerText.slice(0, 1500) : '',"
            " Array.from(document.scripts, s => s.src).filter(Boolean).join(' ').slice(0, 4000)]",
        )
        return web_stealth.challenge_vendor(str(title), str(body_head), str(scripts))

    def _frame_checkbox(self, frame_url_part: str) -> dict[str, float] | None:
        """Return the box of the checkbox in the widget frame whose URL contains *frame_url_part*.

        Args:
            frame_url_part: What tells the widget's frame apart:
                ``challenges.cloudflare.com`` for Turnstile,
                ``/recaptcha/api2/anchor`` for reCAPTCHA v2's "I'm not a
                robot" box.

        Returns:
            The checkbox's bounding box in main-frame viewport pixels, or
            ``None`` while no such frame, or only its spinner, is showing.
        """
        frame = next((f for f in self._page.frames if frame_url_part in f.url), None)
        if frame is None:
            return None
        try:
            box = self._driver_call(
                partial(frame.get_by_role("checkbox").first.bounding_box, timeout=1500),
                1500,
                "Reading an element's box",
            )
        except web_stealth.playwright_api().Error:
            # Still in the spinner phase, or the frame is being replaced.
            logger.debug("Turnstile checkbox not ready", exc_info=True)
            return None
        # ``None`` when rendered but hidden: the widget is still spinning.
        return box or None

    def _challenge_widget(self, vendor: str) -> dict[str, float] | None:
        """Return the box of the element a person would press on *vendor*'s page.

        Two widgets are pressed on the user's behalf: Cloudflare Turnstile's
        checkbox (in its cross-origin ``challenges.cloudflare.com`` iframe,
        shown when the silent checks do not rate the browser human enough;
        its accessible element spans the square and the "Verify you are
        human" label, people press the square) and HUMAN / PerimeterX's
        "Press & Hold" button, found in the visible frame of the
        ``#px-captcha`` widget (:data:`web_stealth.PRESS_AND_HOLD_BOX_JS`,
        :func:`web_stealth.press_and_hold_box`; the widget can be wider
        than its button) and pressed anywhere on it.  CAPTCHA puzzles
        (AWS WAF's "Begin" button opens one) and hard blocks have no such
        element and are reported for :meth:`show_browser`.

        Args:
            vendor: The label :meth:`_challenge_vendor` returned.

        Returns:
            The box to press, in main-frame viewport pixels, or ``None``
            when the page shows nothing to press.
        """
        if vendor in (web_stealth.CLOUDFLARE_CHALLENGE, web_stealth.STACK_OVERFLOW_CAPTCHA):
            box = self._frame_checkbox(
                "challenges.cloudflare.com"
                if vendor == web_stealth.CLOUDFLARE_CHALLENGE
                else "/recaptcha/api2/anchor"
            )
            if box is None:
                return None
            side = min(box["width"], box["height"])
            return {"x": box["x"], "y": box["y"], "width": side, "height": box["height"]}
        if vendor == web_stealth.PERIMETERX_CHECK:
            boxes = self._read_page(self._page, web_stealth.PRESS_AND_HOLD_BOX_JS)
            if not boxes:
                return None
            # Frame-owner box reads carry no protocol timeout, as raw input.
            button: dict[str, float] | None = self._input_hang_watchdog(
                lambda: web_stealth.press_and_hold_box(self._page, boxes[0])
            )
            return button
        return None

    def _puzzle_open(self, vendor: str) -> bool:
        """Whether *vendor*'s page has opened a puzzle that only a person solves.

        Stack Overflow's reCAPTCHA box, once ticked, either passes (the
        page submits itself) or opens a picture puzzle
        (:data:`web_stealth.RECAPTCHA_PUZZLE_OPEN_JS`); waiting on the
        puzzle helps nobody, so the challenge wait ends when it shows.

        Args:
            vendor: The label :meth:`_challenge_vendor` returned.

        Returns:
            ``True`` when the puzzle is showing; ``False`` for every other
            page and vendor.
        """
        if vendor != web_stealth.STACK_OVERFLOW_CAPTCHA:
            return False
        (shown,) = self._read_page(self._page, web_stealth.RECAPTCHA_PUZZLE_OPEN_JS)
        return bool(shown)

    def _release_mouse(self) -> None:
        """Let go of the mouse button after a press that a Stop or an error cut short.

        The button must not stay down for whatever uses the page next.
        The release is raw input like the press, so it has a watchdog of
        its own: a page that does not take it within a round trip is
        ended like any page that hangs raw input
        (:meth:`_input_hang_watchdog`).  A failure (the page closed, the
        watchdog ended it) is logged and ignored: the caller re-raises
        what cut the press short.
        """
        try:
            self._input_hang_watchdog(self._page.mouse.up, _CDP_CALL_TIMEOUT_SECS)
        except Exception:  # noqa: BLE001 - the page closed, or the watchdog ended it
            logger.debug("Could not release the mouse button", exc_info=True)

    def _press_at(self, box: dict[str, float], hold_secs: float) -> None:
        """Press the element *box* like a person and keep it down for *hold_secs*.

        A curved approach to the middle of the box, a short pause, then
        the button goes down and, while held, the pointer drifts as a
        resting hand does — all by pointer coordinates because the
        element is in a cross-origin frame.  A Stop or an error during
        the hold lets the button go too (:meth:`_release_mouse`).

        Args:
            box: Box to press, in main-frame viewport pixels
                (:meth:`_challenge_widget`).
            hold_secs: How long the button stays down
                (:func:`web_stealth.press_hold_secs`).
        """
        x = box["x"] + box["width"] * random.uniform(0.35, 0.65)
        y = box["y"] + box["height"] * random.uniform(0.35, 0.65)

        def hold() -> None:
            self._move_mouse_to(x, y)
            self._page.wait_for_timeout(random.randint(200, 600))
            try:
                self._page.mouse.down()
                release_at = time.monotonic() + hold_secs
                while True:
                    self._page.wait_for_timeout(random.randint(60, 140))
                    if time.monotonic() >= release_at:
                        break
                    self._page.mouse.move(*web_stealth.hold_tremor(x, y), steps=2)
                self._page.mouse.up()
            except BaseException:
                self._release_mouse()
                raise

        self._input_hang_watchdog(hold, _INPUT_WATCHDOG_SECS + hold_secs)
        logger.info("pressed the challenge widget for %.1fs on %s", hold_secs, self._page.url)

    def _settle_challenge(self) -> str:
        """Wait out a bot-protection interstitial; describe it when it stays.

        Cloudflare's managed challenge (and Anubis' proof-of-work page,
        AWS WAF's blank token challenge) run their checks in the browser
        and then reload the real page on their own — for a browser they
        rate as human.  Returning the
        interstitial's accessibility tree would make the agent retry or
        give up, so the tool waits up to
        :func:`web_stealth.challenge_wait_secs` with small pointer
        movements for the page to clear, and presses the widget a person
        would press (:meth:`_challenge_widget`) when one shows; an
        interstitial that gives way to another vendor's page (AWS WAF's
        token challenge, then its CAPTCHA puzzle) gets that page's own,
        possibly shorter, wait from then on, and a widget that answers
        the press with a puzzle (:meth:`_puzzle_open`) ends the wait.
        Presses are counted per vendor for the whole wait
        (:func:`web_stealth.widget_presses`), so two pages that keep
        handing over to each other run out of them.  A
        Google "unusual traffic" page is not waited on: it rates the
        network's IP, not the browser, so the same query is opened on
        Bing; a semanticscholar.org page kept behind AWS WAF's puzzle (the
        same kind of rating) is looked up on Crossref instead
        (:func:`web_stealth.content_fallback`); Reddit's static block page
        gets no wait either (:func:`web_stealth.challenge_wait_secs`).
        The interstitial is recognised by what the page shows
        (:func:`web_stealth.challenge_vendor`), which is also what tells
        when it has cleared.

        Returns:
            ``""`` when the page is content, otherwise a ``Note:`` line
            for the agent explaining what blocked the page and the way
            out (:func:`web_stealth.block_advice`).
        """
        vendor = self._challenge_vendor()
        if vendor is None:
            return ""
        wait_secs = web_stealth.challenge_wait_secs(vendor)
        started = time.monotonic()
        deadline = started + wait_secs
        box_seen_at: float | None = None
        presses: Counter[str] = Counter()
        while (
            vendor is not None and not vendor.startswith("Google") and time.monotonic() < deadline
        ):
            box = None
            if presses[vendor] < web_stealth.widget_presses(vendor):
                box = self._challenge_widget(vendor)
            if box is None:
                box_seen_at = None
            elif box_seen_at is None:
                box_seen_at = time.monotonic()
            reaction = _WIDGET_RETRY_SECS if presses[vendor] else _WIDGET_REACTION_SECS
            if (
                box is not None
                and box_seen_at is not None
                and time.monotonic() - box_seen_at >= reaction
            ):
                # A person reads the widget before pressing it, moving the
                # pointer meanwhile; a press the instant it appears, with
                # no pointer history, is what the vendors ignore.  The
                # verdict then takes a few seconds.
                self._press_at(box, web_stealth.press_hold_secs(vendor))
                presses[vendor] += 1
                box_seen_at = None
                deadline = max(deadline, time.monotonic() + wait_secs)
            else:
                self._idle_mouse()
            self._pause(random.randint(500, 900))
            previous, vendor = vendor, self._challenge_vendor()
            if vendor is not None and vendor != previous:
                # One interstitial replaced by another's (AWS WAF's token
                # challenge handing over to its puzzle): the new page
                # gets its own wait from now, never more than was left.
                wait_secs = web_stealth.challenge_wait_secs(vendor)
                deadline = min(deadline, time.monotonic() + wait_secs)
                # ... and its own widget: the presses spent on the previous
                # page's box (Turnstile's, cleared into Stack Overflow's
                # reCAPTCHA page) are not this page's, and a page that
                # comes back gets none of its own back.
                box_seen_at = None
            if vendor is not None and self._puzzle_open(vendor):
                logger.info("the widget on %s opened a puzzle", self._page.url)
                break
        if vendor is None:
            self._wait_for_stable()
            return ""
        fallback = web_stealth.search_fallback_url(self._page.url)
        if vendor.startswith("Google") and fallback is not None:
            self._goto(fallback)
            self._wait_for_stable()
            return (
                "Note: Google answered with its 'unusual traffic' page. It rates this "
                "network's IP address as automated traffic, which no browser setting "
                "changes, so the same query was opened on Bing instead."
            )
        host = urlparse(self._page.url).netloc
        waited = f"{time.monotonic() - started:.0f}s"
        # A page the user can see stays: they may be solving the puzzle.
        content = None if self._user_sees_page() else web_stealth.content_fallback(self._page.url)
        if content is not None:
            crossref_url, terms = content
            puzzle_url = self._page.url
            try:
                # Retried like the page itself: other jobs' Docker veth
                # links come and go on this box, and one such change
                # aborted half of the Crossref lookups of 12 Nov 2026.
                self._goto_retrying_network_change(crossref_url)
                # Bounded like every other page read, not a bare ``evaluate``.
                render = f"(terms) => [({web_stealth.CROSSREF_RENDER_JS})(terms)]"
                shown = int(self._read_page(self._page, render, terms)[0])
                if shown < 0:
                    raise ValueError("the answer is not Crossref's works list")
            except Exception as exc:  # noqa: BLE001 — best effort; the puzzle is reported then
                logger.info("Crossref fallback for %s failed: %s", host, exc)
                # Back from Crossref's (error) page to the puzzle, whose tree
                # carries the show_browser() way out.  A failed request's
                # error page commits after ``goto`` raised, whatever
                # ``page.url`` says now, and can interrupt the reopening
                # navigation: that one is tried once more after a pause.
                back: str | None = None
                for attempt in range(2):
                    try:
                        response = self._goto_retrying_network_change(puzzle_url)
                    except Exception as reopen_exc:  # noqa: BLE001 — reported, not hidden
                        back = type(reopen_exc).__name__
                    else:
                        # The puzzle itself is a 405 (the token challenge a
                        # 202); a server error is not the puzzle back.
                        status = 200 if response is None else response.status
                        back = None if status < 500 else f"HTTP {status}"
                        if back is None:
                            break
                    if not attempt:
                        self._pause(500)
                if back is not None:
                    logger.info("Reopening %s failed: %s", puzzle_url, back)
                    return (
                        f"Note: {host} answered with a bot-protection page ({vendor}) that did "
                        f"not clear within {waited}; Crossref (api.crossref.org) gave no answer "
                        f"for it and the page could not be reopened afterwards ({back}). "
                        f"Call go_to_url('{puzzle_url}') again, or use another source for "
                        "the same information."
                    )
            else:
                self._wait_for_stable()
                return (
                    f"Note: {host} answered with a bot-protection page ({vendor}) that did not "
                    f"clear within {waited}. It rates this network's IP address, which no "
                    f"browser setting changes, so '{terms}' was looked up on Crossref "
                    f"(api.crossref.org) instead: {shown} work(s) found. For the original "
                    f"page, call show_browser() and then go_to_url('{puzzle_url}') so the "
                    "user can complete the check."
                )
        # A page no wait clears (Reddit's) is not said to have been waited on.
        stayed = (
            f" that did not clear within {waited}"
            if web_stealth.challenge_wait_secs(vendor) > 0
            else ""
        )
        return (
            f"Note: {host} answered with a bot-protection page ({vendor}){stayed}. "
            f"{web_stealth.block_advice(vendor)}"
        )

    def _settled_tree(self) -> str:
        """The page's tree after a navigation, behind the ``Note:`` of a block that stayed.

        Every action that can land on a new document (``go_to_url``, a
        click on a link or a search result, a typed submission, a key
        press) goes through here, so an interstitial met on the way is
        waited out or reported the same way whichever action met it.
        """
        notice = self._settle_challenge()
        tree = self._get_ax_tree()
        return f"{notice}\n\n{tree}" if notice else tree

    def _adopt_new_tab(self, pages_before: list[Any]) -> None:
        """Switch to the tab the last action opened, if it opened one.

        A click on, or Enter over, a ``target="_blank"`` link (or a form
        submitting to one) opens a new tab; the agent's attention moves
        there, as a person's would, and so does the challenge handling.
        Only a tab the current page opened counts: the context is shared
        with the tabs of other tasks and the user (the Browser tab, an
        attached Chrome), and one of theirs appearing meanwhile is not
        this action's.

        Args:
            pages_before: The tabs the context had before the action.
        """
        opened = [
            page
            for page in self._context.pages
            if page not in pages_before and page.opener() is self._page
        ]
        if opened:
            self._adopt_page(opened[-1])
            self._wait_for_stable()

    def _user_sees_page(self) -> bool:
        """Whether the user can see the page: the Browser tab, a window, their own Chrome."""
        return self._live or not self._headless or self._attached_url is not None

    def _aria_snapshot(self, timeout_ms: int) -> str:
        """Playwright's ``mode="ai"`` ARIA snapshot of the current page, as a bounded page read.

        It gives every element the pointer can reach a reference that
        selects that very DOM node, and replaces the references of the
        snapshot before it.

        Args:
            timeout_ms: How long the page gets to answer.
        """
        body = self._page.locator("body")._impl_obj
        return str(
            _page_read(self._page, body.aria_snapshot(timeout=timeout_ms, mode="ai"), timeout_ms)
        )

    def _get_ax_tree(self) -> str:
        """The numbered accessibility tree of the current page.

        Reads the page the caller just acted on and nothing else: every
        tool method ensures the browser before acting, and a page that
        died during the action (a hung tab the daemon closed, a popup
        that closed itself) must surface as that tool's error string
        rather than be swapped for a fresh blank tab here, which would
        report the action as a success on an empty page.  The next tool
        call recovers through :meth:`_ensure_browser`.

        The tree and the reference behind each of its ids are one
        snapshot of the page (:func:`_number_interactive_elements`).
        """
        header = f"Page: {self._page_title(self._page)}\nURL: {self._page.url}\n\n"
        # The snapshot replaces the references the page holds, those of another
        # document included: a read that fails leaves no ids to be taken for its own.
        self._elements = []
        try:
            snapshot = self._aria_snapshot(_FRAMES_TIMEOUT_MS)
        except web_stealth.playwright_api().TimeoutError:
            # Not an iframe that kept the snapshot waiting: the page itself is busy.
            snapshot = self._aria_snapshot(_PAGE_READ_TIMEOUT_MS)
        tree, self._elements = _number_interactive_elements(snapshot)
        if not tree:
            return header + "(empty page)"
        if len(tree) > _AX_TREE_MAX_CHARS:
            tree = tree[:_AX_TREE_MAX_CHARS] + "\n... [truncated]"
        return header + tree

    def _wait_for_stable(self) -> None:
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:  # pragma: no cover — page load timeout is timing-dependent
            logger.debug("Exception caught", exc_info=True)
        try:
            self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:  # pragma: no cover — network idle timeout is timing-dependent
            logger.debug("Exception caught", exc_info=True)

    def _check_for_new_tab(self) -> None:
        if self._context is None:
            return
        pages = self._context.pages
        if len(pages) > 1:  # pragma: no branch
            self._adopt_page(pages[-1])

    def _resolve_locator(self, element_id: int) -> Any:
        """The locator of the element the last tree showed as ``[element_id]``.

        It selects the DOM node the id stands for (the element or, for
        one the pointer cannot reach, the node inside it that it can),
        wherever the page has moved it since, and nothing once that node
        has left the page: the id is then refused, never handed to a
        neighbour, and the agent reads the page again.

        Raises:
            ValueError: The last tree has no such id, or its element is gone.
        """
        element_id = int(element_id)
        if not 1 <= element_id <= len(self._elements):
            raise ValueError(f"Element with ID {element_id} not found. {_READ_AGAIN}")
        locator = self._page.locator(f"aria-ref={self._elements[element_id - 1]}")
        self._require_responsive_renderer()
        try:
            found = self._input_hang_watchdog(locator.count)
        except web_stealth.playwright_api().Error as exc:
            if "aria-ref" not in str(exc):
                raise
            found = 0  # the reference is of a document the tab has left
        if not found:
            raise ValueError(
                f"Element with ID {element_id} has changed since the page was read. {_READ_AGAIN}"
            )
        return locator

    def _try_ensure_browser(self, context: str) -> str | None:
        """Start the browser if needed; return an error string on failure.

        Public tools document string error returns, so browser
        startup/install failures must surface as ``Error <context>: ...``
        instead of escaping the method (S2-27).
        """
        try:
            self._ensure_browser()
            return None
        except Exception as exc:
            logger.warning("browser startup failed", exc_info=True)
            return f"Error {context}: {exc}"

    def go_to_url(self, url: str) -> str:
        """Navigate the browser to a URL and return the page accessibility tree.
        Use when you need to open a new page or switch pages. Special values: "tab:list"
        returns a list of open tabs; "tab:N" switches to tab N (0-based).

        Args:
            url: Full URL to open, or "tab:list" for tab list, or "tab:N" to switch to tab N.

        Returns:
            On success: page title, URL, and accessibility tree with [N] IDs. For "tab:list":
            list of open tabs with indices. On error: "Error navigating to <url>: <message>"."""
        err = self._try_ensure_browser(f"navigating to {url}")
        if err is not None:
            return err
        try:
            pages = self._context.pages
            if url == "tab:list":
                lines = [f"Open tabs ({len(pages)}):"]
                for i, page in enumerate(pages):
                    suffix = " (active)" if page == self._page else ""
                    try:
                        title = self._page_title(page)
                    except (TimeoutError, web_stealth.playwright_api().TimeoutError):
                        # One unresponsive tab must not make the whole
                        # listing fail; the agent needs it to switch away.
                        # The built-in TimeoutError is the serialization
                        # deadline of ``_read_page``, Playwright's the
                        # wait's.
                        logger.debug("Exception caught", exc_info=True)
                        title = "(unresponsive)"
                    lines.append(f"  [{i}] {title} - {page.url}{suffix}")
                return "\n".join(lines)
            if url.startswith("tab:"):
                idx = int(url[4:])
                if 0 <= idx < len(pages):
                    self._adopt_page(pages[idx])
                    # The tab may show an interstitial nothing settled yet
                    # (a popup the page opened, the user's own tab).
                    return self._settled_tree()
                return f"Error: Tab index {idx} out of range (0-{len(pages) - 1})."

            self._goto_retrying_network_change(url)
            self._wait_for_stable()
            return self._settled_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error navigating to {url}: {e}"

    def _goto_retrying_network_change(self, url: str) -> Any:
        """Open *url* in the active tab, retrying when a network change aborts it.

        Chromium fails every in-flight request with ``net::ERR_NETWORK_CHANGED``
        when a host interface appears or disappears (Wi-Fi hand-over, VPN
        connect, a container's veth link). The browser's own UI retries that
        error by itself; this tool does the same, up to two more times a
        second apart, so a transient event does not cost the agent a step.
        Any other error, or a third failure, propagates unchanged.

        The navigation's timeout is enforced by the driver: one that no
        longer answers is killed a round trip after it
        (:meth:`_driver_call`), which fails the pending call; the
        next tool call relaunches.
        """
        for attempt in range(_NETWORK_CHANGE_ATTEMPTS):
            try:
                return self._goto(url)
            except web_stealth.playwright_api().Error as exc:
                if (
                    "net::ERR_NETWORK_CHANGED" not in str(exc)
                    or attempt == _NETWORK_CHANGE_ATTEMPTS - 1
                ):
                    raise
                logger.debug("Navigation aborted by a network change; retrying", exc_info=True)
                time.sleep(1.0)
        raise AssertionError("unreachable")  # pragma: no cover

    def click(self, element_id: int, action: str = "click") -> str:
        """Click or hover on an interactive element by its [N] ID from the accessibility tree.
        Use after get_page_content or go_to_url to interact with links, buttons, tabs, etc.

        Args:
            element_id: Numeric ID shown in brackets [N] next to the element in the tree.
            action: "click" (default) to click the element, "hover" to only move focus.

        Returns:
            Updated accessibility tree (title, URL, numbered elements), or on error
            "Error clicking element <id>: <message>"."""
        err = self._try_ensure_browser(f"clicking element {element_id}")
        if err is not None:
            return err
        try:
            locator = self._resolve_locator(element_id)

            if action == "hover":
                self._human_click(locator, hover_only=True)
                self._pause(300)
                return self._get_ax_tree()

            pages_before = self._context.pages
            self._human_click(locator)
            self._pause(500)
            self._wait_for_stable()
            self._adopt_new_tab(pages_before)
            return self._settled_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error clicking element {element_id}: {e}"

    def type_text(self, element_id: int, text: str, press_enter: bool = False) -> str:
        """Type text into a textbox, searchbox, or other editable element by its [N] ID.
        Clears existing content then types the given text. Use for forms, search boxes, etc.

        Args:
            element_id: Numeric ID from the accessibility tree (brackets [N]).
            text: String to type into the element.
            press_enter: If True, press Enter after typing (e.g. to submit a search).

        Returns:
            Updated accessibility tree, or "Error typing into element <id>: <message>" on error."""
        err = self._try_ensure_browser(f"typing into element {element_id}")
        if err is not None:
            return err
        try:
            locator = self._resolve_locator(element_id)
            select_all = "Meta+a" if sys.platform == "darwin" else "Control+a"
            pages_before = self._context.pages
            self._human_click(locator)
            # Typing spends up to 115 ms per character (260 ms per space)
            # legitimately, so the deadline scales with the text.
            chunks = web_stealth.typing_chunks(text)
            deadline = _INPUT_WATCHDOG_SECS + web_stealth.typing_duration_secs(chunks) + 1.0

            def type_chunks() -> None:
                self._page.keyboard.press(select_all)
                self._page.keyboard.press("Backspace")
                for chunk, delay in chunks:
                    self._page.keyboard.type(chunk, delay=delay)
                if press_enter:
                    # A person pauses before submitting.
                    self._page.wait_for_timeout(random.randint(150, 450))
                    self._page.keyboard.press("Enter")

            self._input_hang_watchdog(type_chunks, deadline)
            if press_enter:
                self._pause(500)
                self._wait_for_stable()
                self._adopt_new_tab(pages_before)
                return self._settled_tree()
            return self._get_ax_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error typing into element {element_id}: {e}"

    def press_key(self, key: str) -> str:
        """Press a single key or key combination. Use for navigation, closing dialogs, shortcuts.

        Args:
            key: Key name, e.g. "Enter", "Escape", "Tab", "ArrowDown", "PageDown", "Backspace",
                 or combination like "Control+a", "Shift+Tab".

        Returns:
            Updated accessibility tree, or "Error pressing key '<key>': <message>" on error."""
        err = self._try_ensure_browser(f"pressing key {key!r}")
        if err is not None:
            return err
        try:
            self._require_responsive_renderer()
            pages_before = self._context.pages
            self._input_hang_watchdog(lambda: self._page.keyboard.press(key))
            self._pause(300)
            # Enter on a focused link or form submits: the key may have
            # navigated (or opened a tab), and onto an interstitial.
            self._adopt_new_tab(pages_before)
            return self._settled_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error pressing key '{key}': {e}"

    def scroll(self, direction: str = "down", amount: int = 3) -> str:
        """Scroll the current page to reveal more content. Use when needed elements are off-screen.

        Args:
            direction: "down", "up", "left", or "right".
            amount: Number of scroll steps of about 300 pixels (default 3, at most 50).

        Returns:
            Updated accessibility tree after scrolling, or
            "Error scrolling <direction>: <message>" on error."""
        err = self._try_ensure_browser(f"scrolling {direction}")
        if err is not None:
            return err
        try:
            dx, dy = _SCROLL_DELTA.get(direction, (0, 300))
            vw, vh = self._viewport_size()
            self._require_responsive_renderer()
            amount = min(max(int(amount), 0), _SCROLL_MAX_STEPS)
            # Each wheel step waits up to 180 ms legitimately, so the
            # deadline scales with the step count.
            deadline = _INPUT_WATCHDOG_SECS + 0.2 * amount

            def wheel() -> None:
                # Wheel from wherever the pointer lands in the middle
                # of the page, with uneven notches and pauses like a
                # person's scroll wheel.
                self._move_mouse_to(
                    random.uniform(vw * 0.3, vw * 0.7), random.uniform(vh * 0.3, vh * 0.7)
                )
                for _ in range(amount):
                    notch = random.uniform(0.8, 1.2)
                    self._page.mouse.wheel(dx * notch, dy * notch)
                    self._page.wait_for_timeout(random.randint(60, 180))

            self._input_hang_watchdog(wheel, deadline)
            self._pause(300)
            return self._get_ax_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error scrolling {direction}: {e}"

    def screenshot(self, file_path: str = "screenshot.png") -> str:
        """Capture the current viewport of the Chromium browser as an image.

        Use to verify layout, captchas, or visual state of a web page currently
        open in the browser. Works the same whether the browser is headless
        (the default) or visible. This does NOT capture or display local files,
        attached images, or PDFs — it only screenshots the browser page.

        Args:
            file_path: Path where the PNG will be saved (default "screenshot.png"). Parent
                directories are created if needed.

        Returns:
            "Screenshot saved to <resolved_path>", or
            "Error taking screenshot: <message>" on error."""
        err = self._try_ensure_browser("taking screenshot")
        if err is not None:
            return err
        try:
            # Same contract as UsefulTools.Write: a path under a worktree
            # the framework already merged and removed falls back to the
            # parent repo, or mkdir() would resurrect a zombie worktree.
            path = resolve_tool_path(file_path, self.work_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._driver_call(
                partial(
                    self._page.screenshot,
                    path=str(path),
                    full_page=False,
                    timeout=_ACTION_TIMEOUT_MS,
                ),
                _ACTION_TIMEOUT_MS,
                "Screenshot",
            )
            return f"Screenshot saved to {path}"
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error taking screenshot: {e}"

    def get_page_content(self, text_only: bool = False) -> str:
        """Get the current page content. Use to decide what to click or type next.

        Args:
            text_only: If False (default), return accessibility tree with [N] IDs for interactive
                elements. If True, return plain text only (title, URL, body text).

        Returns:
            Accessibility tree or plain text as described above, or
            "Error getting page content: <message>" on error."""
        err = self._try_ensure_browser("getting page content")
        if err is not None:
            return err
        try:
            if text_only:
                title = self._page_title(self._page)
                url = self._page.url
                body = _page_read(
                    self._page,
                    self._page._impl_obj.inner_text("body", timeout=_PAGE_READ_TIMEOUT_MS),
                    _PAGE_READ_TIMEOUT_MS,
                )
                return f"Page: {title}\nURL: {url}\n\n{body}"
            return self._get_ax_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error getting page content: {e}"

    def evaluate_script(self, script: str) -> str:
        """Evaluate JavaScript in the current page and return the result as JSON.

        The single most powerful browser tool: read state the accessibility
        tree does not show (computed styles, element attributes, JS variables),
        extract structured data, or act on the page directly (click hidden
        elements, set form values, call the page's own functions). Prefer it
        over guessing when a click did nothing — pair it with
        list_console_messages() and list_network_requests() to see why.

        Args:
            script: A JavaScript function, e.g. "() => document.title" or
                "async () => (await fetch('/api')).status". A function is
                invoked and awaited if it returns a promise. A bare
                expression ("document.title", "(1 + 2)") is returned as is.

        Returns:
            The result: a string as is, anything else as JSON (truncated at
            20000 chars); or "Error evaluating script: <message>" when the
            script throws or the page fails."""
        err = self._try_ensure_browser("evaluating script")
        if err is not None:
            return err
        src = script.strip()
        if not src:
            return "Error evaluating script: empty script."
        try:
            session = self._page_session(self._page)
        except Exception:  # noqa: BLE001 - the browser did not answer the attach
            logger.debug("Exception caught", exc_info=True)
            return "Error evaluating script: no CDP session for the current page."
        # One-shot evaluation over the page's CDP session: unlike
        # wait_for_function it is never re-armed in a new document, so a
        # script that navigates the page cannot run its side effects
        # twice (the evaluation is cancelled instead and reported as an
        # error).  The typeof check is what lets functions and bare
        # expressions both through.  awaitPromise + returnByValue move
        # promise-waiting and JSON serialization into the browser, and
        # the page-side timeout ends scripts that run long; the
        # _cdp_call deadline covers a renderer that never answers at
        # all.  The script sits on lines of its own, so a line comment
        # ending it cannot swallow the rest of the wrapper.
        expression = (
            f"(async () => {{ const f = (\n{src}\n); try {{"
            " return [true, await (typeof f === 'function' ? f() : f)];"
            " } catch (e) { return [false, String((e && e.stack) || e)]; } })()"
        )
        try:
            reply = _cdp_call(
                self._page,
                session.send(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "awaitPromise": True,
                        "returnByValue": True,
                        "timeout": _PAGE_READ_TIMEOUT_MS,
                    },
                ),
                timeout_secs=_PAGE_READ_TIMEOUT_MS / 1000 + _CDP_CALL_TIMEOUT_SECS,
            )
        except Exception as e:  # protocol errors (cyclic results), timeouts, dead pages
            logger.debug("Exception caught", exc_info=True)
            return f"Error evaluating script: {e}"
        details = reply.get("exceptionDetails")
        if details is not None:
            exception = details.get("exception") or {}
            message = exception.get("description") or details.get("text") or "unknown error"
            return f"Error evaluating script: {str(message).splitlines()[0]}"
        # The wrapper always resolves to a [ok, value] pair, and returnByValue
        # serializes it (or the call failed above with a protocol error).
        ok, value = reply["result"]["value"]
        if not ok:
            return f"Error evaluating script: {value}"
        text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
        if len(text) > _EVAL_RESULT_MAX_CHARS:
            text = text[:_EVAL_RESULT_MAX_CHARS] + "\n... [truncated]"
        return text

    def list_console_messages(self, clear: bool = False) -> str:
        """List console messages and uncaught errors of the current page.

        The accessibility tree never shows a TypeError or a failed
        console.assert; when a click or a page load did not do what it
        should, look here first.  Recording starts with the first call
        for a page: it shows what the current document has logged so
        far, and from then on the page's messages are kept (the last 200).

        Args:
            clear: If True, empty the recorded messages after listing them,
                so the next call shows only what the next action caused.

        Returns:
            One "[level] text (url:line)" line per message, "No console
            messages recorded for the current page." when there are none,
            or "Error listing console messages: <message>" on failure."""
        err = self._try_ensure_browser("listing console messages")
        if err is not None:
            return err
        try:
            self._capture_console(self._page)
            # The sync driver only delivers page events while a call is in
            # flight, so the round trip comes FIRST: it pumps events queued
            # since the last tool call into the buffers.  The snapshot and
            # the clear then happen with no driver yield between them.
            title = self._page_title(self._page)
            url = self._page.url
            log = self._record(self._page).console
            if not log:
                return "No console messages recorded for the current page."
            entries = list(log)
            if clear:
                log.clear()
            header = f"Console messages for {title} ({url}) — {len(entries)}:"
            lines = [header]
            for entry in entries:
                where = f" ({entry['where']})" if entry["where"] else ""
                lines.append(f"[{entry['level']}] {entry['text']}{where}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error listing console messages: {e}"

    def list_network_requests(self, url_filter: str = "", clear: bool = False) -> str:
        """List recent network requests of the current page with their status.

        Shows what the page actually loaded and what every action triggered:
        method, URL, resource type, HTTP status, and failure reason.  When a
        click silently does nothing, the answer is often here (a 4xx/5xx or
        a failed fetch).  The last 300 requests per page are kept.

        Args:
            url_filter: Only show requests whose URL contains this substring
                (e.g. "/api/"). Empty shows everything.
            clear: If True, empty the recorded requests after listing them,
                so the next call shows only what the next action caused.

        Returns:
            One "METHOD status URL (type)" line per request ("..." while
            pending, "FAILED: <reason>" for failed ones), "No network
            requests recorded for the current page." when there are none,
            or "Error listing network requests: <message>" on failure."""
        err = self._try_ensure_browser("listing network requests")
        if err is not None:
            return err
        try:
            # The sync driver only delivers page events while a call is in
            # flight, so the round trip comes FIRST: it pumps events queued
            # since the last tool call into the buffers.  The snapshot and
            # the clear then happen with no driver yield between them.
            title = self._page_title(self._page)
            url = self._page.url
            record = self._record(self._page)
            if not record.network:
                return "No network requests recorded for the current page."
            entries = list(record.network)
            if clear:
                record.network.clear()
                # Retired entries must not linger in the pending index.
                record.pending.clear()
            entries = [e for e in entries if url_filter in e["url"]] if url_filter else entries
            if not entries:
                return f"No network requests matching {url_filter!r}."
            header = f"Network requests for {title} ({url}) — {len(entries)}:"
            lines = [header]
            for entry in entries:
                if entry["error"]:
                    status = f"FAILED: {entry['error']}"
                elif entry["status"] is None:
                    status = "..."
                else:
                    status = str(entry["status"])
                lines.append(f"{entry['method']} {status} {entry['url']} ({entry['type']})")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error listing network requests: {e}"

    def wait_for(self, text: str, timeout: float = 15.0, disappear: bool = False) -> str:
        """Wait until text appears on (or disappears from) the current page.

        Turns "click and hope" into "click and verify": after submitting a
        form or triggering a load, wait for the success/error message
        instead of polling screenshots.  Faster and more reliable than
        re-reading the page in a loop.

        Args:
            text: Text (substring match on the visible text, any run of
                white space reading as one space) to wait for, e.g.
                "Saved" or "Something went wrong".
            timeout: Seconds to wait at most (default 15, capped at 120).
            disappear: If True, wait until the text is gone instead (e.g.
                a "Loading..." spinner).

        Returns:
            The updated accessibility tree once the condition holds, "Timed
            out after Ns waiting for text ..." when it does not, or
            "Error waiting for text ...: <message>" on failure."""
        err = self._try_ensure_browser(f"waiting for text {text!r}")
        if err is not None:
            return err
        timeout = min(max(float(timeout), 0.5), _WAIT_FOR_MAX_SECS)
        # What a viewer can actually read: copies of *text* hidden by
        # display:none, visibility:hidden or collapse, content-visibility:
        # hidden or a closed <details> (all but its first <summary>) count
        # for neither appearing nor disappearing, ALL visible
        # copies must go for a disappearance, and open shadow roots
        # (including text nodes directly under the root) and the values
        # of textareas and of inputs that render their value as text
        # (a text field, <input type="submit" value="...">; not a
        # checkbox's or radio's submission value, a password's cleartext
        # or a hidden field) count too.  The page's rendered text is rebuilt from its text
        # nodes in rendered order (shadow trees in place, a <slot> of a
        # shadow tree expanded to the light DOM assigned to it, whose
        # visibility and line box then follow the slot, not the host; a
        # <slot> outside any shadow tree renders its own children like
        # any inline element): a text node counts
        # when it has layout boxes of its own (a Range over it has
        # client rects: this leaves out display:none, undistributed
        # light DOM under a shadow host, the options of a collapsed
        # <select>, a textarea's stale default text, and keeps
        # display:contents text) and its element is not visibility- or
        # content-visibility-hidden; text inside one line box is joined
        # (<b>Sa</b>ved, a highlighted search hit, reads "Saved") while
        # a block boundary or a rendered <br> separates (two table cells
        # never read as one word); a collapsed <select> contributes its
        # selected option's label and inputs their value.  White space
        # is NOT reproduced the way CSS renders it: every run of white
        # space (source line breaks, &nbsp;, white-space:pre runs) reads
        # as one space on both sides, so "Changes saved" matches
        # "Changes\nsaved" in the source and "Changes&nbsp;saved" alike.
        # Text under a CSS text-transform is matched in both spellings:
        # as the page wrote it (what the accessibility tree shows) and
        # as it is displayed (a screenshot, the text-only content), the
        # case mapped as Chromium maps it, which is not as
        # String.toUpperCase does: upper and lower case in the language
        # computed for the element (-webkit-locale: a lang attribute or
        # the page's Content-Language), tailored for Turkish, Azeri,
        # Greek and Lithuanian only, with Georgian left without upper
        # case; for capitalize, the first letter of each word of a line
        # to title case, one UTF-16 unit at a time (a digraph and a
        # Greek letter with an iota subscript have a title case of
        # their own; a letter with no title case of one unit stays, as
        # ß does).  Text a stylesheet draws before or after an element
        # (the strings and attr() values of a ::before or ::after
        # `content`; no counter, quote mark or alternative text, and no
        # private-use glyph of an icon font) is read where it is
        # displayed, and *text* is looked for with it (the accessibility
        # tree, a screenshot) and without it (the page's own text).
        # Reading those two styles of every element costs more than the
        # rest of a poll (139 ms against 39 on a page of 28,000
        # elements, 64 against 36 on an MDN page of 5,000), so the
        # rules of the style sheets are read first and then only the
        # elements that the selectors of the rules that draw text find
        # (34 ms on that MDN page).  Reading the rules costs about a
        # microsecond each: 18 ms a poll for the 12,000 of a Stack
        # Overflow page and for the 26,000 of a GitHub one.  Every
        # element is read only where the selectors cannot tell: with a
        # sheet of another origin, whose rules a page cannot read, a
        # nested or scoped rule, one for a ::part(), a keyframe that
        # sets `content`, more than 500 such rules, and in a shadow
        # tree that has any.
        # Chromium's innerText is not used: it lists every option of a
        # collapsed <select>, skips SVG and does not cross shadow roots.
        # The rebuild reads the layout of every text node, so a cheap
        # gate runs first: the rendered letters are a subsequence of the
        # raw letters in rendered order (hidden parts deleted), so a
        # needle whose letters are not such a subsequence cannot be on
        # screen.  Letters are compared as code points (an emoji is one
        # letter on both sides) without case or accents, which is what
        # a text-transform changes: folded to lower case and then to
        # upper, so a letter and its other case fold alike whichever
        # way the transform went (ß and SS, ẞ and ß).  Polled every
        # 100 ms rather than per frame.
        predicate = (
            "([text, disappear]) => {"
            " const parentOf = (n) =>"  # flat-tree parent: the <slot> a slotted node renders in
            "   n.assignedSlot || n.parentElement"
            "   || (n.getRootNode && n.getRootNode().host) || null;"
            " const isField = (el) => el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT'"
            "   && !/^(checkbox|radio|hidden|password|file|color|range|image)$/.test(el.type));"
            " const isMenu = (el) => el.tagName === 'SELECT' && !el.multiple && el.size <= 1;"
            " const folded = (n, holder) =>"  # a closed <details> renders its first <summary> only
            "   !!holder && holder.tagName === 'DETAILS' && !holder.open"
            "   && n !== holder.querySelector(':scope > summary');"
            " const skipped = new Map();"  # inside a content-visibility:hidden subtree
            " const isSkipped = (el) => {"
            "   if (!el) return false;"
            "   if (!skipped.has(el))"
            "     skipped.set(el, getComputedStyle(el).contentVisibility === 'hidden'"
            "       || folded(el, parentOf(el)) || isSkipped(parentOf(el)));"
            "   return skipped.get(el); };"
            " const styles = new Map();"
            " const styleOf = (el) => {"
            "   if (!styles.has(el)) styles.set(el, getComputedStyle(el));"
            "   return styles.get(el); };"
            " const visible = (el) =>"  # visibility is inherited: the element's own value decides
            "   !!el && !/^(hidden|collapse)$/.test(styleOf(el).visibility) && !isSkipped(el);"
            " const boxed = (el) => visible(el) && el.getClientRects().length > 0;"
            " const laidOut = (el) => !!el && (el.getClientRects().length > 0"
            "   || (styleOf(el).display === 'contents' && laidOut(parentOf(el))));"
            " const blocks = new Map();"
            " const blockOf = (el) => {"  # the nearest element that starts its own line
            "   if (!el) return null;"
            "   if (!blocks.has(el))"
            "     blocks.set(el, /^(inline|contents|ruby)/.test(styleOf(el).display)"
            "       ? blockOf(parentOf(el)) || el : el);"
            "   return blocks.get(el); };"
            # A `content` that draws no text: none, or strings of what shows no character
            # (the private-use glyphs of an icon font, a zero-width space).
            " const noText = /^(none|normal|(\"[\\p{Co}\\p{Cf}]*\" *)+(\\/.*)?)?$/u;"
            " const textRules = new Map();"  # sheet: the selectors of its rules that draw text
            " const scan = (rules, found) => {"
            "   for (let i = 0, count = rules.length; i < count; i++) {"
            "     const rule = rules[i], selector = rule.selectorText, inner = rule.cssRules;"
            "     if (inner && inner.length) scan(inner, found);"  # @media, @keyframes, nesting
            "     if (selector === undefined) {"
            "       if (rule.styleSheet) found.push(...rulesOf(rule.styleSheet));"  # @import
            # A keyframe and the declarations nested in a rule have no selector of their own;
            # the rule of a @position-try or a @font-face has no `content`.
            "       if (rule.style && !noText.test(rule.style.content || '')) found.push('&');"
            "     } else if (/::(before|after)/.test(selector)"
            "         && !noText.test(rule.style.content))"
            "       found.push(selector);"
            "   } };"
            " const rulesOf = (sheet) => {"
            "   if (!textRules.has(sheet)) {"
            "     const found = [];"
            "     textRules.set(sheet, found);"
            "     try { scan(sheet.cssRules, found); }"
            "     catch (e) { found.push('&'); }"  # another origin's sheet: its rules are unknown
            "   }"
            "   return textRules.get(sheet); };"
            " const rulesIn = (root) =>"
            "   [...root.styleSheets, ...root.adoptedStyleSheets].flatMap(rulesOf);"
            " const targets = new Set();"  # the elements that a rule of the document may draw by
            # Where a selector does not find the elements of its rule (the base of a nested
            # one, a ::part()), every element is read; and past 500 rules, where a query for
            # each costs as much.
            " const selectors = rulesIn(document);"
            " let everywhere = selectors.length > 500 || selectors.some((selector) => {"
            "   let unknown = false;"
            # The selector without its ::before or ::after; an escape and a string are no syntax.
            "   const base = selector.replace(/\\\\[^]|\"(?:[^\"\\\\]|\\\\[^])*\""
            "|(^|[\\s>+~,])?::(?:before|after)(?=,|$)|(::|&|:scope)/g,"
            "     (all, lead, other) => {"
            "       unknown ||= !!other;"
            "       if (other || /^[\\\\\"]/.test(all)) return all;"
            "       return lead === undefined ? '' : lead + '*'; });"
            "   if (unknown) return true;"
            "   try { for (const el of document.querySelectorAll(base)) targets.add(el); }"
            "   catch (e) { return true; }"
            "   return false; }) ? 1 : 0;"
            # Whether an element renders a ::before and ::after: no replaced element or control,
            # but an image that failed, a date, time or file input, a restyled checkbox or radio.
            " const hasPseudos = (el) => !(el instanceof HTMLElement) ? false"
            "   : el.tagName === 'IMG'"
            "     ? el.complete && !el.naturalWidth && !!(el.currentSrc || el.alt)"
            "   : el.tagName === 'INPUT'"
            "     ? /^(file|date|datetime-local|month|time|week)$/.test(el.type)"
            "       || (/^(checkbox|radio)$/.test(el.type) && styleOf(el).appearance === 'none')"
            "   : !/^(AUDIO|BR|CANVAS|COL|COLGROUP|EMBED|IFRAME|METER|OBJECT|PROGRESS|SELECT"
            "|TEXTAREA|VIDEO|WBR)$/.test(el.tagName);"
            " const stringsOf = (content) => {"  # of a computed `content`, up to its alt text
            "   let out = '', depth = 0;"
            "   for (const [token, string] of"
            "       content.matchAll(/\"((?:[^\"\\\\]|\\\\[^])*)\"|[()\\/]/g)) {"
            "     if (token === '(') depth++;"
            "     else if (token === ')') depth--;"
            "     else if (depth) continue;"  # in a url() or a counter()
            "     else if (token === '/') break;"
            "     else out += string.replace(/\\\\([0-9a-f]{1,6}) ?|\\\\([^])/gi,"
            "       (all, hex, ch) => hex ? String.fromCodePoint(parseInt(hex, 16)) : ch);"
            "   }"
            "   return out; };"
            # Text nodes, <br>s, fields, menus and what a sheet draws, in rendered order.
            " const nodes = [];"
            " const addDrawn = (el, which) => {"  # what a sheet draws before or after el, if text
            "   const style = getComputedStyle(el, which);"
            "   const text = stringsOf(style.content).replace(/\\p{Co}/gu, '');"
            "   if (/^\\p{Cf}*$/u.test(text)) return;"
            "   const drawn = {by: el, text};"
            "   styles.set(drawn, style);"
            "   nodes.push(drawn); };"
            " const visit = (n) => {"
            "   if (n.nodeType === 3) { nodes.push(n); return; }"
            "   if (n.nodeType !== 1) return;"
            # The rules of its shadow tree may draw by it, in the tree and by what it slots.
            "   const inside = n.shadowRoot && rulesIn(n.shadowRoot).length ? 1 : 0;"
            "   everywhere += inside;"
            "   const read = (everywhere || targets.has(n)) && hasPseudos(n);"
            "   if (read) addDrawn(n, '::before');"
            "   if (n.tagName === 'BR' || isField(n) || isMenu(n)) nodes.push(n);"
            "   if (n.shadowRoot) walk(n.shadowRoot);"  # its light DOM shows through <slot>s only
            "   else if (n.tagName === 'SLOT' && n.getRootNode() instanceof ShadowRoot)"
            "     n.assignedNodes({flatten: true}).forEach(visit);"
            "   else walk(n);"
            "   if (read) addDrawn(n, '::after');"
            "   everywhere -= inside; };"
            " const walk = (root) => { for (const c of root.childNodes) visit(c); };"
            " const labelOf = (n) => n.nodeType === 3 ? n.data"
            "   : n.by ? n.text"
            "   : isField(n) ? n.value || ''"
            "   : isMenu(n) && n.selectedIndex >= 0 ? n.options[n.selectedIndex].label : '';"
            " const titleOf = (c) => {"  # the title case of one UTF-16 unit
            "   const p = c.charCodeAt(0);"
            "   if (p >= 0x1C4 && p <= 0x1CC)"  # the digraphs DŽ, LJ and NJ: Dž, Lj, Nj
            "     return String.fromCharCode(p - (p - 0x1C4) % 3 + 1);"
            "   if (p >= 0x1F1 && p <= 0x1F3) return '\\u01F2';"  # DZ: Dz
            "   if (p >= 0x10D0 && p <= 0x10FF) return c;"  # Georgian
            "   const up = c.toUpperCase();"
            "   if (up.length === 1) return up;"
            "   if (p >= 0x1F80 && p <= 0x1FA7 && p % 16 < 8)"  # Greek with an iota subscript
            "     return String.fromCharCode(p + 8);"
            "   if (p === 0x1FB3 || p === 0x1FC3 || p === 0x1FF3)"
            "     return String.fromCharCode(p + 9);"
            "   return c; };"
            " let words;"
            " const shownOf = (s, el, before) => {"  # *s* as el's text-transform displays it
            "   const how = styleOf(el).textTransform;"
            "   const upper = /uppercase/.test(how);"
            "   if (upper || /lowercase/.test(how)) {"
            "     const tailored = /^\"(tr|az|el|lt)[\"_@-]/i.exec(styleOf(el).webkitLocale);"
            "     const lang = tailored ? tailored[1] : 'und';"
            "     return upper ? s.toLocaleUpperCase(lang)"
            "       .replace(/[\\u1C90-\\u1CBF]/g, (c) => c.toLowerCase())"
            "       : s.toLocaleLowerCase(lang);"
            "   }"
            "   if (!/capitalize/.test(how)) return s;"
            "   words = words || new Intl.Segmenter(undefined, {granularity: 'word'});"
            "   let out = '';"  # *before*, the line's last character, may continue a word
            "   for (const w of words.segment(before + s.replace(/\\u00a0/g, ' ')))"
            "     out += w.index ? titleOf(w.segment[0]) + w.segment.slice(1) : w.segment.slice(1);"
            "   return out; };"
            " const squash = (s) => s.replace(/\\s+/g, ' ').trim();"
            " const letters = (s) => Array.from(s.toLowerCase().toUpperCase()"
            "   .normalize('NFD').replace(/\\p{M}|\\s/gu, ''));"
            " const subsequence = (needle, hay) => {"
            "   let i = 0;"
            "   for (const ch of hay) if (ch === needle[i] && ++i === needle.length) return true;"
            "   return needle.length === 0; };"
            " const range = document.createRange();"
            " const rendered = (n) => { range.selectNodeContents(n);"
            "   return range.getClientRects().length > 0; };"
            " const needle = squash(text);"
            " let found = false;"
            " if (document.body) {"
            "   const top = document.documentElement;"  # a sheet may draw by the root element too
            "   const readTop = everywhere || targets.has(top);"
            "   if (readTop) addDrawn(top, '::before');"
            "   visit(document.body);"
            "   if (readTop) addDrawn(top, '::after');"
            "   if (subsequence(letters(needle), letters(nodes.map(labelOf).join('')))) {"
            # Two readings, with and without what a sheet draws: each as written and as
            # displayed, with the last character displayed, which may continue a word.
            "     const full = {out: [], shown: [], last: ' ', box: null};"
            "     const plain = {out: [], shown: [], last: ' ', box: null};"
            "     const add = (to, n, box, holder) => {"
            "       if (to.box !== null && box !== to.box) {"
            "         to.out.push(' '); to.shown.push(' '); to.last = ' '; }"
            "       const shown = shownOf(labelOf(n), holder, to.last);"
            "       to.out.push(labelOf(n)); to.shown.push(shown);"
            "       to.last = shown.slice(-1) || to.last;"
            "       to.box = box; };"
            "     const anyDrawn = nodes.some((n) => n.by);"
            "     for (const n of nodes) {"
            "       let box = n, holder = n;"
            "       if (n.nodeType === 3) {"
            "         holder = parentOf(n);"
            "         if (!visible(holder) || folded(n, holder) || !rendered(n)) continue;"
            "         box = blockOf(holder);"
            "       } else if (n.by) {"
            "         const style = styleOf(n);"
            "         if (style.display === 'none' || /^(hidden|collapse)$/.test(style.visibility)"
            "             || isSkipped(n.by) || !laidOut(n.by)) continue;"
            "         if (/^(inline|contents|ruby)/.test(style.display)) box = blockOf(n.by);"
            "       } else if (!boxed(n)) continue;"
            "       add(full, n, box, holder);"
            "       if (anyDrawn && !n.by) add(plain, n, box, holder);"
            "     }"
            "     found = [full.out, full.shown, plain.out, plain.shown].some((parts) =>"
            "       squash(parts.join('')).includes(needle));"
            "   }"
            " }"
            " return disappear ? !found : found; }"
        )
        try:
            handle = _wait_for_function(self._page, predicate, [text, disappear], timeout * 1000)
            _dispose_handle(self._page, handle)
            return self._get_ax_tree()
        except web_stealth.playwright_api().TimeoutError:
            verb = "disappear" if disappear else "appear"
            return f"Timed out after {timeout:g}s waiting for text {text!r} to {verb}."
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error waiting for text {text!r}: {e}"

    def attach_to_chrome(self, cdp_url: str = "http://localhost:9222") -> str:
        """Attach to a real Chrome/Chromium the user started, and drive its tabs.

        Some tasks need the user's own logged-in browser sessions (SSO,
        corporate apps, accounts with passkeys).  The user starts Chrome
        with remote debugging enabled — e.g.
        ``chrome --remote-debugging-port=9222`` — and this tool connects
        to it over the Chrome DevTools Protocol and adopts its newest
        tab.  From then on every browser tool (click, type_text,
        evaluate_script, list_console_messages, list_network_requests,
        screenshot, ...) operates on the user's real browser, with its
        real cookies and logins.  close_browser() detaches again WITHOUT
        closing the user's Chrome; the built-in stealth Chromium takes
        over on the next call.

        WARNING: this acts with the full authority of the user's logged-in
        sessions.  Only attach when the user asked for it, and prefer
        reading over acting.

        Args:
            cdp_url: The DevTools endpoint of the user's browser
                (default "http://localhost:9222").

        Returns:
            The accessibility tree of the adopted tab, or
            "Error attaching to Chrome at <url>: <message>" on failure."""
        if self._live:
            # Live mode and an attached user browser are mutually
            # exclusive: leave the Browser tab first (its tab belongs to
            # the daemon's browser; closing it here would be wrong, and
            # a leftover _live flag would route the NEXT teardown into
            # _detach_live's page.close() on the user's tab).
            self._detach_live(close_tab=False)
            self._live = False
        self._close_browser_only()
        try:
            playwright = self._ensure_playwright()
            browser = self._driver_call(
                partial(playwright.chromium.connect_over_cdp, cdp_url, timeout=15000),
                15000,
                "Attaching to Chrome",
            )
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error attaching to Chrome at {cdp_url}: {e}"
        try:
            pages = [
                p
                for context in browser.contexts
                for p in context.pages
                if not p.is_closed() and not _is_browser_ui(p)
            ]
            if not pages:
                raise RuntimeError("no open tabs found")
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            try:
                _cdp_call(browser, browser._impl_obj.close())
            except Exception:  # noqa: BLE001 - already gone
                logger.debug("Exception caught", exc_info=True)
            return f"Error attaching to Chrome at {cdp_url}: {e}"
        browser.on("disconnected", self._on_browser_lost)
        self._browser = browser
        # The newest regular tab wins; its own context becomes this
        # tool's context (tab listing, popup detection) so a browser
        # with several contexts still behaves.
        self._context = pages[-1].context
        self._attached_url = cdp_url
        # The user's Chrome took over: a session stranded by a failed
        # show_browser() is not brought back after it.
        self._stranded_session = None
        try:
            self._adopt_page(pages[-1])
            return self._get_ax_tree()
        except Exception as e:  # a tab that cannot be adopted or read (wedged, no <body>)
            logger.debug("Exception caught", exc_info=True)
            self._close_browser_only()
            return f"Error attaching to Chrome at {cdp_url}: {e}"

    def use_my_chrome(
        self,
        action: str,
        tab_id: int = 0,
        url: str = "",
        script: str = "",
        uid: int = 0,
        text: str = "",
        timeout: int = 30,
    ) -> str:
        """Drive the user's everyday Chrome through the bridge Chrome extension.

        This is the way into the browser the user is actually logged into
        (their default-profile Chrome with Gmail, SSO, passkeys): Chrome
        since v136 ignores --remote-debugging-port on the default profile,
        so attach_to_chrome cannot reach it, but the bridge extension can.
        The user loads the extension once (chrome://extensions → Developer
        mode → Load unpacked → the directory named in this tool's error
        while no extension is connected); this tool then talks to it
        over a token-gated loopback WebSocket.  The daemon
        serves that socket from start-up when the "Drive my Chrome"
        setting is on; otherwise the first call starts it and may wait a
        few seconds for the extension to find it.

        Actions:
            "list_tabs": every tab as "id=<id> [active] <title> — <url>".
            "open_tab": open url in a new tab, wait for load, return its id.
            "navigate": point tab_id at url, wait for load.
            "activate_tab": bring tab_id to the front.
            "close_tab": close tab_id.
            "evaluate": run script in tab_id's page and return the result
                (must be JSON-serializable; return a string when in doubt).
                Uses the page's main world where the site's CSP allows it,
                so page variables and fetch() with the site's session work.
            "screenshot": capture tab_id's viewport; returns the path of
                the saved PNG (under the work dir's tmp/).
            "snapshot": list the visible interactive elements of tab_id's
                page as a compact numbered list, e.g.
                '[3] <button> "Save"'. The numbers (uids) are the handles
                for click and type_text; every snapshot re-assigns them,
                so re-snapshot after the page changes.
            "click": click element uid in tab_id (from a snapshot).
                Scrolls it into view and dispatches the full pointer/mouse
                event sequence so React and other frameworks see a real
                click.
            "type_text": replace the value of the input/textarea/
                contenteditable element uid in tab_id with text,
                dispatching the input/change events frameworks listen for.

        The normal rhythm is: snapshot, click/type_text by uid, snapshot
        again. Use evaluate for anything the snapshot cannot express.

        Args:
            action: One of the actions listed above.
            tab_id: Target tab id from list_tabs/open_tab (not needed for
                list_tabs/open_tab).
            url: Target URL for open_tab/navigate.
            script: JavaScript for evaluate.
            uid: Element number from snapshot, for click/type_text.
            text: Replacement value for type_text (empty clears the field).
            timeout: Seconds to wait for the extension's answer (1 to 120).

        Returns:
            The action result, or "Error ..." explaining what to do (e.g.
            how to install the extension when none is connected).

        WARNING: this acts with the full authority of the user's logged-in
        sessions.  Only use it when the user asked for it, and prefer
        reading over acting."""
        from kiss.agents.sorcar import chrome_bridge  # deferred: websockets import cost

        needed = _BRIDGE_ACTIONS.get(action)
        if needed is None:
            return f"Error: unknown action '{action}'. Use one of: {', '.join(_BRIDGE_ACTIONS)}."
        given: dict[str, Any] = {
            "tab_id": tab_id,
            "url": url,
            "script": script,
            "uid": uid,
            "text": text,
        }
        args: dict[str, Any] = {}
        for name in needed:
            wire_name, wanted = _BRIDGE_ARGS[name]
            # An empty text is a value (it clears the field); the rest must be given.
            if not given[name] and name != "text":
                return f"Error: action '{action}' needs {wanted}."
            args[wire_name] = given[name]
        try:
            bridge = chrome_bridge.ChromeBridge.instance()
        except chrome_bridge.ChromeBridgeError as e:
            return f"Error starting the Chrome bridge: {e}"
        if not bridge.extension_connected:
            # Started by this very call (daemon setting off, or a CLI run):
            # the extension finds the new config on its next retry.
            bridge.wait_for_extension(_BRIDGE_CONNECT_WAIT_SECS)
        # Bounded like wait_for (the agent's timeout must not park the task
        # for hours), and never under a second: a deadline already past
        # would drop the extension's connection while the user's Chrome
        # still runs the action.
        wait_secs = min(max(float(timeout), 1.0), _WAIT_FOR_MAX_SECS)
        try:
            result = bridge.call(action, args, timeout=wait_secs)
        except chrome_bridge.ChromeBridgeError as e:
            return f"Error using my Chrome: {e}"
        if action == "screenshot":
            return self._save_bridge_shot(result)
        return _format_bridge_result(action, result)

    def _save_bridge_shot(self, data_url: Any) -> str:
        """Decode a screenshot data URL from the bridge into a file and return its path."""
        import base64

        if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
            return (
                "Error: the extension returned an unexpected screenshot payload: "
                f"{str(data_url)[:120]}"
            )
        try:
            png = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        except Exception as e:  # noqa: BLE001 - a buggy/compromised extension must not crash the tool
            return f"Error: the extension returned an undecodable screenshot payload: {e}"
        # ns: no same-second collisions.  Under the task's work dir like
        # screenshot(), not the daemon's CWD; a sub-agent's goes to its
        # private folder, so that it leaves nothing in the work tree.
        name = f"chrome-shot-{time.time_ns()}.png"
        out = (Path(self._private_dir, name) if self._private_dir
               else resolve_tool_path(f"tmp/{name}", self.work_dir))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
        return f"Screenshot saved to {out} ({out.stat().st_size} bytes)."

    def close(self) -> str:
        """Close the browser and release resources. Call when done with the session or before exit.

        A page shown in the Browser tab stays open there for the user;
        only this tool's connection to it is dropped.

        Returns:
            "Browser closed." (always, even if nothing was open)."""
        self._close_browser_only(keep_live_tab=True)
        self._stranded_session = None
        if self._playwright:
            _stop_playwright(self._playwright)
        self._playwright = None
        atexit.unregister(self.close)
        if self._ephemeral_dir:
            _rmtree_logged(self._ephemeral_dir)
        return "Browser closed."

    def close_browser(self) -> str:
        """Close the Chromium browser and free its OS process.

        Use when you are done with web browsing for now (its purpose is
        over) so the browser does not stay running for the rest of a
        long task. Safe to call anytime: the next web tool call (e.g.
        go_to_url) automatically relaunches a fresh browser with the same
        profile, so logins are preserved. While the page is shown in the
        Browser tab this closes that tab on every surface.

        Returns:
            "Browser closed. It will relaunch automatically on the next web tool call."."""
        self._close_browser_only()
        self._stranded_session = None
        return "Browser closed. It will relaunch automatically on the next web tool call."

    def show_browser(self, visible: bool = True) -> str:
        """Show the page to the user, live, in a Browser tab on every KISS surface.

        Call this whenever the user should see or interact with a page:
        an interactive login or OAuth consent, a CAPTCHA or "unusual
        traffic" bot check, a demo or test of a web app the user wants to
        watch, or any browsing the user asked to follow live. The page you
        are on reopens as a Browser tab that every surface switches to;
        the user can click and type in it while you keep driving the same
        page with the other web tools (screenshot, click, type_text,
        get_page_content, ...). That browser keeps its cookies and logins
        across tasks and daemon restarts, and the cookies of the current
        headless session are carried into it. Without a KISS daemon the
        browser is shown as a window on the machine instead. Pass
        visible=False when the human part is done: the tab closes and
        browsing continues headless with the cookies carried back.

        Args:
            visible: True to show the page to the user, False to return to
                headless browsing.

        Returns:
            The accessibility tree of the reopened page, or
            "Browser is now visible in the Browser tab."/"Browser is now
            headless." when no page was open, or
            "Error <doing something>: <message>" on failure."""
        state = "visible" if visible else "headless"
        alive = self._is_alive()
        shown = self._live or not self._headless
        # A Chrome the user started is theirs, not the Browser tab: the page
        # moves into the tab on visible=True, and visible=False leaves that
        # Chrome (running) for the headless browser instead of being a no-op.
        already = shown if visible else not shown and self._attached_url is None
        if already and alive:
            return f"Browser is already {state}."
        # With no browser to capture from, the session an earlier failed
        # switch left behind is still the one to carry.
        session = self._capture_session() if alive else self._stranded_session
        self._stranded_session = None
        self._close_browser_only()
        self._live = visible and self._live_browser is not None
        self._headless = not visible or self._live
        err = self._try_ensure_browser(f"making the browser {state}")
        if err is not None:
            if self._live:
                self._detach_live(close_tab=True)
            # Browsing goes on headless: a window that cannot open (no X
            # server) would fail every later tool call the same way.  The
            # browser the session was captured from is closed: the session
            # waits for the headless one the next tool call launches.
            self._live, self._headless = False, True
            self._stranded_session = session
            return err
        self._restore_session(session)
        if session is not None and session.url:
            return self.go_to_url(session.url)
        if self._live:
            return "Browser is now visible in the Browser tab."
        return f"Browser is now {state}."

    def _resume_stranded_session(self) -> None:
        """Move the session of a failed ``show_browser()`` into the browser just launched.

        The switch closed the browser it captured the session from before
        the other one failed to open; without this the headless browser
        that takes over starts on a blank page, logged out.  A page that
        cannot be reopened is logged: the tool call that launched the
        browser goes on with the cookies restored.
        """
        session, self._stranded_session = self._stranded_session, None
        if session is None:
            return
        self._restore_session(session)
        if session.url:
            try:
                self._goto(session.url)
            except Exception:  # noqa: BLE001 - unreachable right now
                logger.debug("Could not reopen the page of a failed switch", exc_info=True)

    def _capture_session(self) -> _Session:
        """Return what to carry into the browser that takes over the session.

        Chromium cannot switch between headless and visible without being
        restarted, and the Browser tab is a different browser altogether,
        so the page's URL, the context's cookies (a login or bot-check
        flow is usually mid-way through setting them) and the page
        origin's ``localStorage`` (token-based logins) travel along; they
        are handed to the new browser by :meth:`_restore_session`.  The
        caller checks :meth:`_is_alive` first.

        Returns:
            The session to restore (``url`` empty when nothing worth
            reopening is loaded).
        """
        # The user may have opened a tab themselves while the window was
        # visible; that newest tab is the one worth carrying over.  While
        # no window shows, nobody but the agent opens tabs, and the newest
        # is not its page (a popup it followed and left again); in the
        # Browser tab, and in a Chrome the user started, the newest page
        # may be an unrelated tab of the user's.
        if not self._live and self._attached_url is None and not self._headless:
            self._check_for_new_tab()
        url = self._page.url
        if url.startswith("about:"):
            url = ""
        session = _Session(url, None)
        if self._attached_url is not None:
            # The cookie jar and localStorage are the user's own Chrome
            # profile (attach_to_chrome): only the URL may leave it.
            return session
        try:
            # Bounded like the liveness probe: the Browser tab's browser
            # may have stopped answering (the caller's probe counts a
            # silent one as alive).
            session.cookies = _cdp_call(self._context, self._context._impl_obj.cookies())
        except Exception:  # noqa: BLE001 - closed under the switch, or not answering
            logger.debug("Could not read cookies before relaunch", exc_info=True)
        if url:
            try:
                session.origin, session.local_storage = self._read_page(
                    self._page, "() => [location.origin, Object.entries(localStorage)]"
                )
            except Exception:  # noqa: BLE001 - opaque origin (data:) or a stuck page
                logger.debug("Could not read localStorage before relaunch", exc_info=True)
        return session

    def _restore_session(self, session: _Session | None) -> None:
        """Move *session* into the freshly launched browser.

        Cookies are added, and the cookies carried in at the previous
        switch that have since died (logged out, expired) are deleted so
        they do not come back to life; other cookies of the destination
        are left alone.  The origin's ``localStorage`` is reconciled the
        same way by opening the page once before the caller navigates to
        it properly.  With nothing captured (the page was already closed)
        nothing is deleted either, nor when the jar could not be read
        (the user's own Chrome, a browser that did not answer): an empty
        capture is not a logout, and that switch carries no cookies.
        Nor does a switch whose import failed: only cookies the
        destination took can later die there.

        Args:
            session: What :meth:`_capture_session` returned.
        """
        if session is None:
            self._carried_cookies, self._carried_storage = [], ("", set())
            return
        if session.cookies is not None:
            alive = {_cookie_key(c) for c in session.cookies}
            for cookie in self._carried_cookies:
                if _cookie_key(cookie) not in alive:
                    try:
                        self._delete_cookie(cookie)
                    except Exception:  # noqa: BLE001 - the page's CDP session did not answer
                        logger.debug("Could not drop a dead cookie after relaunch", exc_info=True)
        self._carried_cookies = []
        if session.cookies:
            try:
                # Bounded like the capture: the Browser tab's browser may
                # stop answering right after it opened the tab.
                _cdp_call(self._context, self._context._impl_obj.add_cookies(session.cookies))
            except Exception:  # noqa: BLE001 - a malformed cookie, or not answering
                logger.debug("Could not restore cookies after relaunch", exc_info=True)
            else:
                self._carried_cookies = session.cookies
        self._restore_local_storage(session)

    def _delete_cookie(self, cookie: dict[str, Any]) -> None:
        """Delete *cookie* (one name, domain, path and partition) from the current browser.

        ``BrowserContext.clear_cookies`` knows no partition: it takes
        every partition's cookie of that name, domain and path with it,
        another top-level site's live login among them.  The CDP call
        names the partition the way the driver does when it adds a
        partitioned cookie (``hasCrossSiteAncestor`` is the driver's
        non-standard ``_crHasCrossSiteAncestor``, true by default); without
        one, only the unpartitioned cookie goes.  Bounded like every CDP
        round trip (:func:`_cdp_call`).
        """
        params: dict[str, Any] = {
            "name": cookie["name"],
            "domain": cookie["domain"],
            "path": cookie["path"],
        }
        if cookie.get("partitionKey"):
            params["partitionKey"] = {
                "topLevelSite": cookie["partitionKey"],
                "hasCrossSiteAncestor": cookie.get("_crHasCrossSiteAncestor", True),
            }
        session = self._page_session(self._page)
        _cdp_call(self._page, session.send("Network.deleteCookies", params))

    def _restore_local_storage(self, session: _Session) -> None:
        """Write the captured origin's localStorage into the current browser.

        Keys carried in at the previous switch for the same origin that
        are gone now are removed.  The page is opened once for the write,
        which is skipped if it lands on another origin.  The write is
        bounded like every page read (:meth:`_read_page`): a page whose
        script wedges the renderer right after commit must not hang the
        switch of browser forever.
        """
        keys = {key for key, _ in session.local_storage}
        carried_origin, carried_keys = self._carried_storage
        removed = sorted(carried_keys - keys) if carried_origin == session.origin else []
        # Only keys actually written count as carried: a skipped write must
        # not make their later absence look like a logout.
        self._carried_storage = ("", set())
        if not session.origin or not (session.local_storage or removed):
            return
        try:
            self._goto(session.url, wait_until="commit")
            [written] = self._read_page(
                self._page, _RESTORE_STORAGE_JS, [session.origin, session.local_storage, removed]
            )
        except Exception:  # noqa: BLE001 - the origin is unreachable right now
            logger.debug("Could not restore localStorage after relaunch", exc_info=True)
            return
        if written:
            self._carried_storage = (session.origin, keys)
        else:
            logger.debug("localStorage not restored: %s moved to another origin", session.url)

    def get_tools(self) -> list[Callable[..., str]]:
        """Return callable web tools for registration with an agent.

        Returns:
            List of callables: go_to_url, click, type_text, press_key, scroll, screenshot,
            get_page_content, evaluate_script, list_console_messages,
            list_network_requests, wait_for, attach_to_chrome, use_my_chrome,
            show_browser, close_browser. Does not include close."""
        return [
            self.go_to_url,
            self.click,
            self.type_text,
            self.press_key,
            self.scroll,
            self.screenshot,
            self.get_page_content,
            self.evaluate_script,
            self.list_console_messages,
            self.list_network_requests,
            self.wait_for,
            self.attach_to_chrome,
            self.use_my_chrome,
            self.show_browser,
            self.close_browser,
        ]
