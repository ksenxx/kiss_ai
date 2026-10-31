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
from typing import Any, TypeGuard
from urllib.parse import urlparse

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.useful_tools import resolve_tool_path
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.processes import SIGKILL, kill_process_tree, popen_process_group
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

# Caps for the results of evaluate_script and for wait_for's timeout: an
# agent must not be able to turn one tool call into an unbounded read or
# an unbounded block of the task thread.
_EVAL_RESULT_MAX_CHARS = 20000
_WAIT_FOR_MAX_SECS = 120.0

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

_ROLE_LINE_RE = re.compile(r"^(\s*)-\s+('?)([\w]+)\s*(.*)")

_NAME_RE = re.compile(r'"((?:\\.|[^"\\])*)"')
_NAME_UNESCAPE_RE = re.compile(r'\\(["\\])')

_SCROLL_DELTA = {"down": (0, 300), "up": (0, -300), "right": (300, 0), "left": (-300, 0)}


_CLOSE_WATCHDOG_SECS = 15.0


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


def _terminate_pid_escalating(pid: int, identity: str | None) -> None:
    """Kill *pid* with SIGTERM then SIGKILL, verifying identity before EACH signal.

    Fails closed: signals are sent only when the process's current
    identity fingerprint is readable and equal to the one recorded at
    capture time.  This refuses recycled PIDs (the browser exited and the
    OS reassigned its PID — possibly between SIGTERM and SIGKILL) and
    unverifiable targets — killing an unrelated process would be far
    worse than leaking a browser.

    Args:
        pid: Process id to terminate.
        identity: Identity string recorded when the PID was captured.
    """
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
            os.kill(pid, sig)
        except OSError:  # pragma: no cover — died between checks
            return
        if _wait_pid_exit(pid, 2.0):
            return
    logger.error(  # pragma: no cover — SIGKILL cannot be ignored
        "Chromium (pid %d) could not be killed",
        pid,
    )


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
) -> None:
    """Watchdog timer body: kill a Chromium whose driver call hung.

    Killing the browser process also unwedges the hung driver call (the
    driver observes the browser exit and completes/raises the call).

    Args:
        pid: Browser process id recorded at launch.
        identity: Identity string recorded at PID capture time.
        operation: Human-readable name of the hung operation, for the log.
        deadline_secs: The deadline that expired, for the log.
    """
    logger.warning(
        "%s is hung after %.0fs; killing Chromium (pid %d) directly",
        operation,
        deadline_secs,
        pid,
    )
    _terminate_pid_escalating(pid, identity)


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


def _kill_driver(playwright: Any, deadline_secs: float) -> None:
    """Watchdog timer body: kill the Node driver whose raw input to an attached Chrome hung.

    The browser is the user's own Chrome, which is never signalled.  The
    driver's exit fails the blocked call instead (its request dies with
    the driver's pipe), and :func:`_driver_is_dead` lets the next call
    discard the driver and say how to reconnect.  ``Process.kill`` polls
    the child first and skips one already reaped (Popen's guard against
    signalling a recycled pid).

    Args:
        playwright: The sync Playwright whose driver to kill.
        deadline_secs: The deadline that expired, for the log.
    """
    logger.warning(
        "Raw input to the attached Chrome is hung after %.0fs; killing the Playwright driver",
        deadline_secs,
    )
    try:
        playwright._impl_obj._connection._transport._proc.kill()
    except (AttributeError, ProcessLookupError):  # already gone, or another Playwright layout
        logger.debug("Could not kill the Playwright driver", exc_info=True)


class _TimerGuard:
    """Context manager that starts a ``threading.Timer`` on enter and cancels it on exit."""

    def __init__(self, timer: threading.Timer) -> None:
        self._timer = timer

    def __enter__(self) -> None:
        """Start the guarded timer."""
        self._timer.start()

    def __exit__(self, *exc: object) -> None:
        """Cancel the timer (a no-op if it already fired)."""
        self._timer.cancel()


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

    Args:
        sync_obj: Any sync API object of the Playwright instance (all of
            them share one loop and dispatcher).
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
    try:
        return sync_obj._sync(asyncio.wait_for(coro, timeout_secs))
    finally:
        # A stopped Playwright (closed loop) rejects the call before the
        # wrapper runs and closes only the wrapper, leaving *coro* itself
        # never awaited (a RuntimeWarning at collection).  Once started,
        # it finished or was cancelled on the loop and is left alone.
        if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
            coro.close()


@dataclass
class _Session:
    """What moves with the browsing session between two browsers."""

    url: str
    cookies: list[Any]
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


def _cookie_key(cookie: dict[str, Any]) -> tuple[str, str, str]:
    """The identity of a cookie: name, domain and path."""
    return (cookie["name"], cookie["domain"], cookie["path"])


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

    ``Popen.wait(timeout=...)`` polls the child from Python in sleeps of
    at most 50 ms, so a Stop injected into the task thread lands within
    one of them; ``subprocess.run`` waits in one ``select`` call that
    only output, the child's exit or the deadline ends, which deferred a
    Stop for up to ``_CHROMIUM_INSTALL_SECS`` while a download stalled.
    The installer runs in its own process group (Job Object on Windows)
    and the whole group is killed when the wait ends early or times out:
    the ``python -m playwright`` launcher only spawns the Node CLI that
    does the download, so killing the launcher alone left that CLI
    running and holding the browser-cache lock.  Output is discarded as
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
        installer.wait(timeout=timeout)
    except BaseException:
        # The launcher is alive or an unreaped zombie here, so its pid
        # (the group id) cannot have been recycled.
        kill_process_tree(installer)
        installer.wait()
        raise
    if installer.returncode:
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


def _is_profile_in_use(profile_dir: str) -> bool:
    """Check whether a Chromium profile directory is locked by a running process.

    See :func:`_profile_lock_state`.

    Args:
        profile_dir: Path to the Chromium user-data directory.

    Returns:
        True if the profile is currently locked by a live process.
    """
    return _profile_lock_state(profile_dir)[1]


def _number_interactive_elements(snapshot: str) -> tuple[str, list[dict[str, str]]]:
    result_lines: list[str] = []
    elements: list[dict[str, str]] = []
    counter = 0
    # Running occurrence tallies (this runs on every accessibility-tree
    # render; rescanning the accumulated element list per element would
    # be O(n^2) on large pages).
    pair_counts: Counter[tuple[str, str]] = Counter()
    role_counts: Counter[str] = Counter()
    for line in snapshot.splitlines():
        m = _ROLE_LINE_RE.match(line)
        if not m:
            result_lines.append(line)
            continue
        indent, quote, role, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        if role not in INTERACTIVE_ROLES:
            result_lines.append(line)
            continue
        counter += 1
        name_match = _NAME_RE.match(rest)
        name = _NAME_UNESCAPE_RE.sub(r"\1", name_match.group(1)) if name_match else ""
        if quote:
            name = name.replace("''", "'")
        # Record which occurrence of this (role, name) pair — and of the
        # bare role — this element is, in snapshot (document) order, so
        # resolving an ID targets *this* element rather than the first
        # visible one that happens to share its role and name.
        occurrence = pair_counts[(role, name)]
        role_occurrence = role_counts[role]
        pair_counts[(role, name)] += 1
        role_counts[role] += 1
        elements.append(
            {
                "role": role,
                "name": name,
                "occurrence": str(occurrence),
                "role_occurrence": str(role_occurrence),
            }
        )
        result_lines.append(f"{indent}- [{counter}] {quote}{role} {rest}".rstrip())
    return "\n".join(result_lines), elements


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
    ) -> None:
        self._live_browser = live_browser
        # True while the session lives in the Browser tab (show_browser);
        # _live_tab is the id of the tab this tool opened there.
        self._live = False
        self._live_tab: str | None = None
        # The Chromium target id of the page driven in the Browser tab,
        # read when the page is adopted: the daemon's service maps it to
        # the tab id without this tool's driver (see _live_tab_id).
        self._live_target: str | None = None
        # The profile directory the last launch actually used (the
        # configured one or an ``<dir>_N`` escalation); a test seam.
        self.effective_user_data_dir: str | None = None
        # The DevTools endpoint _browser is connected to in the Browser
        # tab; a relaunched browser listens on a new one (see _attach_live).
        self._live_cdp_url: str | None = None
        # What the last switch of browser carried in: cookies, and the
        # localStorage keys of one origin.  Those that have died by the
        # next switch are deleted on the other side too.
        self._carried_cookies: list[Any] = []
        self._carried_storage: tuple[str, set[str]] = ("", set())
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
        self._elements: list[dict[str, str]] = []
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
        tab on the next call.  A tab this client fails to pick up and
        adopt as its own, or stops picking up because a Stop landed (the
        connect can take seconds), is closed again so it does not linger
        blank in front of the user, and a page adopted halfway is
        dropped with it.  The tab is taken inside that clean-up: once
        ``open_for_agent`` has returned, the service's own clean-up no
        longer runs.
        """
        if self._playwright is None:
            self._playwright = web_stealth.playwright_api().sync_playwright().start()
        tab = page = None
        try:
            tab = self._live_browser.open_for_agent()
            if not self._live_connected(tab.cdp_url):
                self._disconnect_live()
                self._browser = self._playwright.chromium.connect_over_cdp(tab.cdp_url)
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
        interrupting a hung script).
        """
        if self._live_tab is None and self._page is not None:
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
                self._page.close()
            except Exception:  # noqa: BLE001 - closed by the user meanwhile
                logger.debug("Could not close the Browser tab's page", exc_info=True)
        if self._playwright is not None and _driver_is_dead(self._playwright):
            # The liveness probe (or the close) was the first to learn that
            # the driver died; disconnecting through it would never return.
            self._discard_dead_driver(close_live_tab=close_tab)
            return
        self._live_tab = None
        self._live_target = None
        self._disconnect_live()
        self._on_browser_lost()

    def _disconnect_live(self) -> None:
        """Drop this client's connection to the Browser tab's browser, if any.

        A ``Browser`` object left behind without ``close()`` keeps its
        ``disconnected`` listener and its place in the driver; closing a
        connected-over-CDP browser only disconnects this client, the
        daemon's browser keeps running.
        """
        browser, self._browser = self._browser, None
        if browser is not None:
            try:
                browser.close()
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
        the browser process directly, which also unwedges the hung call.
        The one hang that kill cannot unwedge, a close sent to a driver
        that has itself exited, is never attempted: a dead driver is
        discarded whole (:meth:`_discard_dead_driver`).
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
                    browser.close()
                except Exception:  # noqa: BLE001 - the browser may already be gone
                    logger.debug("Could not disconnect from the attached Chrome", exc_info=True)
            if self._playwright is not None and _driver_is_dead(self._playwright):
                self._discard_dead_driver(close_live_tab=False)
                return
            self._on_browser_lost()
            return
        if self._playwright is not None and _driver_is_dead(self._playwright):
            self._discard_dead_driver(close_live_tab=not keep_live_tab)
            return
        if self._live:
            self._detach_live(close_tab=not keep_live_tab)
            return
        pid = self._browser_pid
        identity = self._browser_identity
        watchdog: threading.Timer | None = None
        if (self._context is not None or self._browser is not None) and _killable_live_pid(pid):
            watchdog = threading.Timer(
                _CLOSE_WATCHDOG_SECS,
                _watchdog_kill,
                args=(pid, identity),
            )
            watchdog.daemon = True
            watchdog.start()
        try:
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
                    obj.close()
                except Exception:
                    logger.warning(
                        "Graceful browser close failed; killing the Chromium "
                        "process if it survived",
                        exc_info=True,
                    )
        finally:
            if watchdog is not None:
                watchdog.cancel()
        self._kill_browser_process()
        self._on_browser_lost()

    def _discard_dead_driver(self, close_live_tab: bool) -> None:
        """Drop a Playwright whose driver process died under the tool.

        No graceful close goes to a dead driver (it would never return,
        see :func:`_driver_is_dead`): Chromium is killed directly when
        this tool launched it (in the Browser tab the browser is the
        daemon's and no pid is recorded), every reference is dropped and
        the driver stopped, so the next call starts a fresh driver and
        browser instead of wedging the task thread forever.  ``stop()``
        on a dead driver returns at once (its transport already finished).

        Args:
            close_live_tab: In the Browser tab, also close this tool's tab
                on every surface -- through the daemon's service, the only
                way left without a driver.  False leaves it in front of
                the user (end of a task).
        """
        logger.warning(
            "Playwright driver exited under the browser tool; discarding it"
            " and relaunching on the next call"
        )
        playwright, self._playwright = self._playwright, None
        live_tab, self._live_tab = self._live_tab, None
        target, self._live_target = self._live_target, None
        if self._live and close_live_tab:
            try:
                if live_tab is None and target is not None:
                    # A popup followed just before the driver died: the
                    # daemon knows the tab by the target id read at adoption
                    # -- from the moment its own client has attached to the
                    # popup, which may still be under way.
                    deadline = time.monotonic() + _LIVE_TAB_LOOKUP_SECS
                    while (
                        live_tab := self._live_browser.tab_for_target(target)
                    ) is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                if live_tab is None:
                    logger.warning("The Browser tab of the dead driver's page is unknown; it stays")
                else:
                    self._live_browser.close(live_tab)
            except Exception:  # noqa: BLE001 - closed by the user meanwhile
                logger.debug("Could not close the Browser tab's page", exc_info=True)
        self._kill_browser_process()
        self._on_browser_lost()
        try:
            playwright.stop()
        except Exception:
            logger.debug("Stopping the dead Playwright driver failed", exc_info=True)

    def _kill_browser_process(self) -> None:
        """Ensure the recorded Chromium OS process is dead, escalating to signals.

        Called after the graceful Playwright close attempts.  Waits briefly
        for a clean exit, then sends SIGTERM and finally SIGKILL (after an
        identity check so a recycled PID is never signalled).  Without
        this, a close whose driver call raised (wedged connection,
        cross-thread greenlet error) silently leaked the Chromium process —
        the root cause of long-horizon tasks accumulating open browsers.
        """
        pid = self._browser_pid
        identity = self._browser_identity
        self._browser_pid = None
        self._browser_identity = None
        if not _killable_live_pid(pid) or _wait_pid_exit(pid, 2.0):
            return
        logger.warning("Chromium (pid %d) survived graceful close", pid)
        _terminate_pid_escalating(pid, identity)

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
        if self._playwright is not None and _driver_is_dead(self._playwright):
            # Before anything consults the dead driver's local state
            # (``context.pages``, a Browser-tab connection that still
            # reads as connected): drop it, a fresh one starts below.  The
            # Browser tab it drove is closed, ``_attach_live`` opens a new
            # one rather than leaving the old one adrift in front of the
            # user.
            self._discard_dead_driver(close_live_tab=True)
        if self._live:
            # The user closed the tab (or the browser exited): open a new one.
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
            else:
                lost = f"The Chrome at {self._attached_url} disconnected"
            raise RuntimeError(
                f'{lost}. Reconnect with attach_to_chrome("{self._attached_url}"), or call '
                "close_browser() to return to the built-in Chromium."
            )
        atexit.unregister(self.close)
        atexit.register(self.close)
        self._close_browser_only()
        api = web_stealth.playwright_api()

        # A launch that shows no window never raises one, so there is no
        # focus to save and restore.
        prev_app = None if self._headless else _get_frontmost_app()
        try:
            if self._playwright is None:
                self._playwright = api.sync_playwright().start()
            launcher = self._playwright.chromium
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
            path = Path(profile_dir) / name
            try:
                if path.is_symlink() or path.exists():
                    path.unlink()
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
        chosen = None if _is_profile_in_use(self.user_data_dir) else self.user_data_dir
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
            self.effective_user_data_dir = effective_dir
            if effective_dir:
                Path(effective_dir).mkdir(parents=True, exist_ok=True)
                self._clean_singleton_locks(effective_dir)
                self._context = launcher.launch_persistent_context(
                    effective_dir, **kwargs, **self._context_args()
                )
                self._capture_browser_pid()
                page = self._context.pages[0] if self._context.pages else self._context.new_page()
            else:
                self._browser = launcher.launch(**kwargs)
                self._capture_browser_pid()
                self._context = self._browser.new_context(**self._context_args())
                page = self._context.new_page()
        self._context.route(_ACCOUNTS_GOOGLE_URL_RE, _abort_route)
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
        follow-up ``json_value()`` has no timeout, but it only runs after
        the renderer answered the same expression an instant earlier.
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

        Args:
            page: The Playwright page to read.
            expression: JavaScript function source returning a list.
            arg: Optional argument passed to the function.

        Returns:
            The list the expression returned, as Python values.
        """
        for attempt in range(_PAGE_READ_NAVIGATIONS + 1):
            try:
                handle = page.wait_for_function(
                    expression, arg=arg, timeout=_PAGE_READ_TIMEOUT_MS, polling=100
                )
                return handle.json_value()
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
        self._page.wait_for_function("() => 1", timeout=_PAGE_READ_TIMEOUT_MS, polling=100)

    def _input_hang_watchdog(self, deadline_secs: float = _INPUT_WATCHDOG_SECS) -> Any:
        """Cut off the guarded raw input call if it outlives *deadline_secs*.

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
        connection with the way back.

        Args:
            deadline_secs: Seconds the guarded block may run before the
                watchdog fires.  Callers whose legitimate duration scales
                with input size (``type_text`` delay, ``scroll`` steps)
                pass a proportionally larger deadline.

        Returns:
            A ``threading.Timer``-cancelling context manager guarding the
            ``with`` block.
        """
        if self._attached_url is not None:
            timer = threading.Timer(
                deadline_secs, _kill_driver, args=(self._playwright, deadline_secs)
            )
            timer.daemon = True
            return _TimerGuard(timer)
        if self._live:
            # The browser is the user's: the daemon's service stops the
            # hung script (and closes only that tab if it stays wedged).
            tab_id = self._live_tab_id()
            if tab_id is None:
                return nullcontext()
            timer = threading.Timer(deadline_secs, self._live_browser.interrupt, args=(tab_id,))
            timer.daemon = True
            return _TimerGuard(timer)
        pid = self._browser_pid
        identity = self._browser_identity
        if not _killable_live_pid(pid):
            return nullcontext()
        timer = threading.Timer(
            deadline_secs,
            _watchdog_kill,
            args=(pid, identity, "Raw browser input", deadline_secs),
        )
        timer.daemon = True
        return _TimerGuard(timer)

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
        centre click is used, exactly as before.  An element without a
        rendered box is clicked the plain way straight away.

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
                    locator.hover(position=position, timeout=5000)
                else:
                    locator.click(position=position, delay=press_ms, timeout=5000)
                return
            except api.TimeoutError:
                logger.debug("point not hittable; falling back to centre click", exc_info=True)
        if hover_only:
            locator.hover()
        else:
            locator.click(delay=press_ms)

    def _glide_onto(self, locator: Any) -> dict[str, float] | None:
        """Scroll *locator* into view and glide the pointer onto it.

        Args:
            locator: Playwright locator of the element.

        Returns:
            The chosen point as an offset inside the element (for
            ``click(position=...)``), or ``None`` when the element has no
            rendered box.
        """
        locator.scroll_into_view_if_needed(timeout=_PAGE_READ_TIMEOUT_MS)
        box = locator.bounding_box(timeout=_PAGE_READ_TIMEOUT_MS)
        if not box or box["width"] < 1 or box["height"] < 1:  # pragma: no cover
            # scroll_into_view_if_needed already waited for visibility;
            # only an element removed between the two calls lands here.
            return None
        ox = box["width"] * random.uniform(0.35, 0.65)
        oy = box["height"] * random.uniform(0.35, 0.65)
        with self._input_hang_watchdog():
            self._move_mouse_to(box["x"] + ox, box["y"] + oy)
        self._page.wait_for_timeout(random.randint(40, 160))
        return {"x": ox, "y": oy}

    def _idle_mouse(self) -> None:
        """Nudge the pointer a little, as a person waiting on a page does."""
        vw, vh = self._viewport_size()
        x0, y0 = self._mouse_xy or (vw / 2, vh / 2)
        x = min(max(x0 + random.uniform(-80, 80), 4), vw - 4)
        y = min(max(y0 + random.uniform(-60, 60), 4), vh - 4)
        with self._input_hang_watchdog():
            self._move_mouse_to(x, y)

    def _challenge_vendor(self) -> str | None:
        """Return the bot-protection vendor whose interstitial is showing, if any."""
        title, body_head = self._read_page(
            self._page,
            "() => [document.title, document.body ? document.body.innerText.slice(0, 1500) : '']",
        )
        return web_stealth.challenge_vendor(str(title), str(body_head))

    def _turnstile_checkbox(self) -> dict[str, float] | None:
        """Return the box of Cloudflare Turnstile's "Verify you are human" checkbox.

        Cloudflare's managed challenge first runs silent checks; when they
        do not rate the browser human enough it shows a single checkbox
        (no puzzle) for the visitor to tick.  The box lives in a
        cross-origin ``challenges.cloudflare.com`` iframe.

        Returns:
            The checkbox's bounding box in main-frame viewport pixels, or
            ``None`` while no widget, or only its spinner, is showing.
        """
        frame = next((f for f in self._page.frames if "challenges.cloudflare.com" in f.url), None)
        if frame is None:
            return None
        try:
            box = frame.get_by_role("checkbox").first.bounding_box(timeout=1500)
        except web_stealth.playwright_api().Error:
            # Still in the spinner phase, or the frame is being replaced.
            logger.debug("Turnstile checkbox not ready", exc_info=True)
            return None
        # ``None`` when rendered but hidden: the widget is still spinning.
        return box or None

    def _challenge_widget(self, vendor: str) -> dict[str, float] | None:
        """Return the box of the element a person would press on *vendor*'s page.

        Two widgets are pressed on the user's behalf: Cloudflare Turnstile's
        checkbox (:meth:`_turnstile_checkbox`; its accessible element spans
        the square and the "Verify you are human" label, people press the
        square) and HUMAN / PerimeterX's "Press & Hold" button, found in
        the visible frame of the ``#px-captcha`` widget
        (:data:`web_stealth.PRESS_AND_HOLD_BOX_JS`,
        :func:`web_stealth.press_and_hold_box`; the widget can be wider
        than its button) and pressed anywhere on it.  CAPTCHA puzzles and
        hard blocks have no such element and are reported for
        :meth:`show_browser`.

        Args:
            vendor: The label :meth:`_challenge_vendor` returned.

        Returns:
            The box to press, in main-frame viewport pixels, or ``None``
            when the page shows nothing to press.
        """
        if vendor == web_stealth.CLOUDFLARE_CHALLENGE:
            box = self._turnstile_checkbox()
            if box is None:
                return None
            side = min(box["width"], box["height"])
            return {"x": box["x"], "y": box["y"], "width": side, "height": box["height"]}
        if vendor == web_stealth.PERIMETERX_CHECK:
            boxes = self._read_page(self._page, web_stealth.PRESS_AND_HOLD_BOX_JS)
            if not boxes:
                return None
            # Frame-owner box reads carry no protocol timeout, as raw input.
            with self._input_hang_watchdog():
                return web_stealth.press_and_hold_box(self._page, boxes[0])
        return None

    def _press_at(self, box: dict[str, float], hold_secs: float) -> None:
        """Press the element *box* like a person and keep it down for *hold_secs*.

        A curved approach to the middle of the box, a short pause, then
        the button goes down and, while held, the pointer drifts as a
        resting hand does — all by pointer coordinates because the
        element is in a cross-origin frame.

        Args:
            box: Box to press, in main-frame viewport pixels
                (:meth:`_challenge_widget`).
            hold_secs: How long the button stays down
                (:func:`web_stealth.press_hold_secs`).
        """
        x = box["x"] + box["width"] * random.uniform(0.35, 0.65)
        y = box["y"] + box["height"] * random.uniform(0.35, 0.65)
        with self._input_hang_watchdog(_INPUT_WATCHDOG_SECS + hold_secs):
            self._move_mouse_to(x, y)
            self._page.wait_for_timeout(random.randint(200, 600))
            self._page.mouse.down()
            release_at = time.monotonic() + hold_secs
            while True:
                self._page.wait_for_timeout(random.randint(60, 140))
                if time.monotonic() >= release_at:
                    break
                self._page.mouse.move(*web_stealth.hold_tremor(x, y), steps=2)
            self._page.mouse.up()
        logger.info("pressed the challenge widget for %.1fs on %s", hold_secs, self._page.url)

    def _settle_challenge(self) -> str:
        """Wait out a bot-protection interstitial; describe it when it stays.

        Cloudflare's managed challenge (and Anubis' proof-of-work page)
        run their checks in the browser and then reload the real page on
        their own — for a browser they rate as human.  Returning the
        interstitial's accessibility tree would make the agent retry or
        give up, so the tool waits up to
        :func:`web_stealth.challenge_wait_secs` with small pointer
        movements for the page to clear, and presses the widget a person
        would press (:meth:`_challenge_widget`) when one shows.  A Google
        "unusual traffic" page is not waited on: it rates the network's
        IP, not the browser, so the same query is opened on Bing.  The
        interstitial is recognised by what the page shows
        (:func:`web_stealth.challenge_vendor`), which is also what tells
        when it has cleared.

        Returns:
            ``""`` when the page is content, otherwise a ``Note:`` line
            for the agent explaining what blocked the page.
        """
        vendor = self._challenge_vendor()
        if vendor is None:
            return ""
        wait_secs = web_stealth.challenge_wait_secs(vendor)
        started = time.monotonic()
        deadline = started + wait_secs
        box_seen_at: float | None = None
        presses = 0
        while (
            vendor is not None and not vendor.startswith("Google") and time.monotonic() < deadline
        ):
            box = None
            if presses < web_stealth.widget_presses(vendor):
                box = self._challenge_widget(vendor)
            if box is None:
                box_seen_at = None
            elif box_seen_at is None:
                box_seen_at = time.monotonic()
            reaction = _WIDGET_RETRY_SECS if presses else _WIDGET_REACTION_SECS
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
                presses += 1
                box_seen_at = None
                deadline = max(deadline, time.monotonic() + wait_secs)
            else:
                self._idle_mouse()
            self._page.wait_for_timeout(random.randint(500, 900))
            vendor = self._challenge_vendor()
        if vendor is None:
            self._wait_for_stable()
            return ""
        fallback = web_stealth.search_fallback_url(self._page.url)
        if vendor.startswith("Google") and fallback is not None:
            self._page.goto(fallback, wait_until="domcontentloaded", timeout=30000)
            self._wait_for_stable()
            return (
                "Note: Google answered with its 'unusual traffic' page. It rates this "
                "network's IP address as automated traffic, which no browser setting "
                "changes, so the same query was opened on Bing instead."
            )
        host = urlparse(self._page.url).netloc
        return (
            f"Note: {host} answered with a bot-protection page ({vendor}) that did not "
            f"clear within {time.monotonic() - started:.0f}s. Call show_browser() so the user "
            "can complete the check, or use another source for the same information."
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
        """
        header = f"Page: {self._page_title(self._page)}\nURL: {self._page.url}\n\n"
        snapshot = self._page.locator("body").aria_snapshot(timeout=_PAGE_READ_TIMEOUT_MS)
        if not snapshot:
            self._elements = []
            return header + "(empty page)"
        numbered, self._elements = _number_interactive_elements(snapshot)
        if len(numbered) > _AX_TREE_MAX_CHARS:
            numbered = numbered[:_AX_TREE_MAX_CHARS] + "\n... [truncated]"
        return header + numbered

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
        element_id = int(element_id)
        if element_id < 1 or element_id > len(self._elements):
            snapshot = self._page.locator("body").aria_snapshot(timeout=_PAGE_READ_TIMEOUT_MS)
            if snapshot:
                _, self._elements = _number_interactive_elements(snapshot)
            if element_id < 1 or element_id > len(self._elements):
                raise ValueError(f"Element with ID {element_id} not found.")
        entry = self._elements[element_id - 1]
        role = entry["role"]
        name = entry["name"]
        if name:
            locator = self._page.get_by_role(role, name=name, exact=True)
            occurrence = int(entry.get("occurrence", "0"))
        else:
            # get_by_role(role) matches named and unnamed elements alike,
            # so an unnamed element's index counts every element of the
            # role in snapshot order.
            locator = self._page.get_by_role(role)
            occurrence = int(entry.get("role_occurrence", "0"))
        self._require_responsive_renderer()
        with self._input_hang_watchdog():
            n = locator.count()
        if n == 0:  # pragma: no cover — race between snapshot and DOM
            raise ValueError(f"Element with ID {element_id} not found on page.")
        if n == 1:
            return locator
        if occurrence < n:
            # Both the aria snapshot and get_by_role() enumerate the
            # accessibility tree in document order, so the recorded
            # occurrence picks the exact element this ID was assigned
            # to — not merely the first visible role/name match.
            return locator.nth(occurrence)
        for i in range(n):  # pragma: no branch — first visible element always found
            try:
                if locator.nth(i).is_visible():
                    return locator.nth(i)
            except Exception:  # pragma: no cover — Playwright is_visible rarely throws
                logger.debug("Exception caught", exc_info=True)
                continue
        return locator.first  # pragma: no cover — all elements invisible is rare

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
                    except web_stealth.playwright_api().TimeoutError:
                        # One unresponsive tab must not make the whole
                        # listing fail; the agent needs it to switch away.
                        logger.debug("Exception caught", exc_info=True)
                        title = "(unresponsive)"
                    lines.append(f"  [{i}] {title} - {page.url}{suffix}")
                return "\n".join(lines)
            if url.startswith("tab:"):
                idx = int(url[4:])
                if 0 <= idx < len(pages):
                    self._adopt_page(pages[idx])
                    return self._get_ax_tree()
                return f"Error: Tab index {idx} out of range (0-{len(pages) - 1})."

            self._goto_retrying_network_change(url)
            self._wait_for_stable()
            notice = self._settle_challenge()
            tree = self._get_ax_tree()
            return f"{notice}\n\n{tree}" if notice else tree
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
        """
        for attempt in range(_NETWORK_CHANGE_ATTEMPTS):
            try:
                return self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
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
                self._page.wait_for_timeout(300)
                return self._get_ax_tree()

            pages_before = len(self._context.pages)
            self._human_click(locator)
            self._page.wait_for_timeout(500)
            self._wait_for_stable()
            pages = self._context.pages
            if len(pages) > pages_before:
                self._adopt_page(pages[-1])
                self._wait_for_stable()
            return self._get_ax_tree()
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
            self._human_click(locator)
            # Typing spends up to 115 ms per character (260 ms per space)
            # legitimately, so the deadline scales with the text.
            chunks = web_stealth.typing_chunks(text)
            deadline = _INPUT_WATCHDOG_SECS + web_stealth.typing_duration_secs(chunks) + 1.0
            with self._input_hang_watchdog(deadline):
                self._page.keyboard.press(select_all)
                self._page.keyboard.press("Backspace")
                for chunk, delay in chunks:
                    self._page.keyboard.type(chunk, delay=delay)
                if press_enter:
                    # A person pauses before submitting.
                    self._page.wait_for_timeout(random.randint(150, 450))
                    self._page.keyboard.press("Enter")
            if press_enter:
                self._page.wait_for_timeout(500)
                self._wait_for_stable()
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
            with self._input_hang_watchdog():
                self._page.keyboard.press(key)
            self._page.wait_for_timeout(300)
            return self._get_ax_tree()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error pressing key '{key}': {e}"

    def scroll(self, direction: str = "down", amount: int = 3) -> str:
        """Scroll the current page to reveal more content. Use when needed elements are off-screen.

        Args:
            direction: "down", "up", "left", or "right".
            amount: Number of scroll steps (default 3).

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
            # Each wheel step waits up to 180 ms legitimately, so the
            # deadline scales with the step count.
            deadline = _INPUT_WATCHDOG_SECS + 0.2 * max(int(amount), 0)
            with self._input_hang_watchdog(deadline):
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
            self._page.wait_for_timeout(300)
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
            self._page.screenshot(path=str(path), full_page=False)
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
                body = self._page.inner_text("body", timeout=_PAGE_READ_TIMEOUT_MS)
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
        # display:none, visibility:hidden, content-visibility:hidden or a
        # closed <details> (all but its first <summary>) count for
        # neither appearing nor disappearing, ALL visible
        # copies must go for a disappearance, and open shadow roots
        # (including text nodes directly under the root) and
        # input/textarea values (e.g. <input type="submit" value="...">)
        # count too.  The page's rendered text is rebuilt from its text
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
        # Chromium's innerText is not used: it lists every option of a
        # collapsed <select>, skips SVG and does not cross shadow roots.
        # The rebuild reads the layout of every text node, so a cheap
        # gate runs first: the rendered letters are a subsequence of the
        # raw letters in rendered order (hidden parts deleted), so a
        # needle whose letters are not such a subsequence cannot be on
        # screen.  Letters are compared as code points (an emoji is one
        # letter on both sides).  Polled every 100 ms rather than per
        # frame.
        predicate = (
            "([text, disappear]) => {"
            " const parentOf = (n) =>"  # flat-tree parent: the <slot> a slotted node renders in
            "   n.assignedSlot || n.parentElement"
            "   || (n.getRootNode && n.getRootNode().host) || null;"
            " const isField = (el) => el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';"
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
            "   !!el && styleOf(el).visibility !== 'hidden' && !isSkipped(el);"
            " const boxed = (el) => visible(el) && el.getClientRects().length > 0;"
            " const blocks = new Map();"
            " const blockOf = (el) => {"  # the nearest element that starts its own line
            "   if (!el) return null;"
            "   if (!blocks.has(el))"
            "     blocks.set(el, /^(inline|contents|ruby)/.test(styleOf(el).display)"
            "       ? blockOf(parentOf(el)) || el : el);"
            "   return blocks.get(el); };"
            " const nodes = [];"  # text nodes, <br>s, fields and menus in rendered order
            " const visit = (n) => {"
            "   if (n.nodeType === 3) { nodes.push(n); return; }"
            "   if (n.nodeType !== 1) return;"
            "   if (n.tagName === 'BR' || isField(n) || isMenu(n)) nodes.push(n);"
            "   if (n.shadowRoot) walk(n.shadowRoot);"  # its light DOM shows through <slot>s only
            "   else if (n.tagName === 'SLOT' && n.getRootNode() instanceof ShadowRoot)"
            "     n.assignedNodes({flatten: true}).forEach(visit);"
            "   else walk(n); };"
            " const walk = (root) => { for (const c of root.childNodes) visit(c); };"
            " const labelOf = (n) => n.nodeType === 3 ? n.data"
            "   : isField(n) ? n.value || ''"
            "   : isMenu(n) && n.selectedIndex >= 0 ? n.options[n.selectedIndex].label : '';"
            " const squash = (s) => s.replace(/\\s+/g, ' ').trim();"
            " const letters = (s) => Array.from(s.replace(/\\s+/g, ''));"
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
            "   walk(document.body);"
            "   if (subsequence(letters(needle), letters(nodes.map(labelOf).join('')))) {"
            "     let out = '', prev = null;"
            "     for (const n of nodes) {"
            "       let box;"
            "       if (n.nodeType === 3) {"
            "         const holder = parentOf(n);"
            "         if (!visible(holder) || folded(n, holder) || !rendered(n)) continue;"
            "         box = blockOf(holder);"
            "       } else {"
            "         if (!boxed(n)) continue;"
            "         box = n;"
            "       }"
            "       if (prev !== null && box !== prev) out += ' ';"
            "       out += labelOf(n);"
            "       prev = box;"
            "     }"
            "     found = squash(out).includes(needle);"
            "   }"
            " }"
            " return disappear ? !found : found; }"
        )
        try:
            self._page.wait_for_function(
                predicate, arg=[text, disappear], timeout=timeout * 1000, polling=100
            )
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
        if self._playwright is None:
            self._playwright = web_stealth.playwright_api().sync_playwright().start()
        try:
            browser = self._playwright.chromium.connect_over_cdp(cdp_url, timeout=15000)
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
                browser.close()
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
            text: Replacement value for type_text.
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
            if not given[name]:
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
        # Bounded like wait_for: a Stop cannot interrupt the wait, and a
        # deadline already past would drop the extension's connection while
        # the user's Chrome still runs the action.
        wait_secs = min(max(float(timeout), 1.0), _WAIT_FOR_MAX_SECS)
        try:
            result = bridge.call(action, args, timeout=wait_secs)
        except chrome_bridge.ChromeBridgeError as e:
            return f"Error using my Chrome: {e}"
        if action == "screenshot":
            return self._save_bridge_shot(result)
        return _format_bridge_result(action, result)

    def _save_bridge_shot(self, data_url: Any) -> str:
        """Decode a screenshot data URL from the bridge into ./tmp/ and return its path."""
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
        # screenshot(), not the daemon's CWD.
        out = resolve_tool_path(f"tmp/chrome-shot-{time.time_ns()}.png", self.work_dir)
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
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:  # pragma: no cover — Playwright stop rarely fails
                logger.debug("Exception caught", exc_info=True)
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
        if visible == (self._live or not self._headless) and alive:
            return f"Browser is already {state}."
        session = self._capture_session() if alive else None
        self._close_browser_only()
        self._live = visible and self._live_browser is not None
        self._headless = not visible or self._live
        err = self._try_ensure_browser(f"making the browser {state}")
        if err is not None:
            if self._live:
                self._detach_live(close_tab=True)
                self._live, self._headless = False, True
            return err
        self._restore_session(session)
        if session is not None and session.url:
            return self.go_to_url(session.url)
        if self._live:
            return "Browser is now visible in the Browser tab."
        return f"Browser is now {state}."

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
        # visible; that newest tab is the one worth carrying over.  In the
        # Browser tab, and in a Chrome the user started, the newest page
        # may be an unrelated tab of the user's.
        if not self._live and self._attached_url is None:
            self._check_for_new_tab()
        url = self._page.url
        if url.startswith("about:"):
            url = ""
        session = _Session(url, [])
        if self._attached_url is not None:
            # The cookie jar and localStorage are the user's own Chrome
            # profile (attach_to_chrome): only the URL may leave it.
            return session
        try:
            session.cookies = self._context.cookies()
        except Exception:  # pragma: no cover — reading cookies rarely fails
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
        nothing is deleted either: an empty capture is not a logout.

        Args:
            session: What :meth:`_capture_session` returned.
        """
        if session is None:
            self._carried_cookies, self._carried_storage = [], ("", set())
            return
        alive = {_cookie_key(c) for c in session.cookies}
        for cookie in self._carried_cookies:
            if _cookie_key(cookie) not in alive:
                try:
                    self._context.clear_cookies(
                        name=cookie["name"], domain=cookie["domain"], path=cookie["path"]
                    )
                except Exception:  # pragma: no cover — clearing a cookie rarely fails
                    logger.debug("Could not drop a dead cookie after relaunch", exc_info=True)
        self._carried_cookies = session.cookies
        if session.cookies:
            try:
                self._context.add_cookies(session.cookies)
            except Exception:  # pragma: no cover — a malformed cookie is rare
                logger.debug("Could not restore cookies after relaunch", exc_info=True)
        self._restore_local_storage(session)

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
            self._page.goto(session.url, wait_until="commit")
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
