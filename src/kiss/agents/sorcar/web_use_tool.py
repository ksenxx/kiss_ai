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

import atexit
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
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar._concurrency import pid_alive as _pid_alive
from kiss.agents.sorcar.persistence import _default_kiss_dir
from kiss.agents.sorcar.useful_tools import (
    _absolutize,
    _active_worktree_remap,
    _file_lock,
    _stale_worktree_fallback,
)
from kiss.core.processes import SIGKILL
from kiss.core.processes import process_identity as _process_identity

logger = logging.getLogger(__name__)

_SINGLETON_FILES = ("SingletonLock", "SingletonCookie", "SingletonSocket")

_ACCOUNTS_GOOGLE_URL_RE = re.compile(r"^https?://accounts\.google\.com/")

# Headless Chromium reports "HeadlessChrome/<version>" in its user agent.
# Many sites use that token alone to serve a bot challenge instead of the
# page, so it is rewritten to the equivalent headed token.
_HEADLESS_UA_TOKEN = "HeadlessChrome"
_HEADED_UA_TOKEN = "Chrome"

# Upper bound for every read of page state (title, text, aria snapshot).
# A page whose main thread is stuck (a script spinning forever, a
# renderer that never answers after a timed-out ``goto``) must surface as
# a tool error, never as a tool call that hangs the whole task.
_PAGE_READ_TIMEOUT_MS = 10000

# Deadline for raw input operations (keyboard.press/type, mouse.move/wheel)
# that have no Playwright timeout parameter.  A page event handler that
# wedges the renderer in response to our own input (e.g. a ``keydown``
# listener entering ``while(true)``) passes the pre-input liveness probe
# and then blocks the input call forever; a watchdog kills Chromium at
# this deadline so the pending call raises instead of hanging the task.
_INPUT_WATCHDOG_SECS = _PAGE_READ_TIMEOUT_MS / 1000 + 5.0

# How long ``go_to_url`` waits for a bot-protection interstitial (Cloudflare
# "Just a moment...", Anubis) to clear on its own before reporting it.
# Cloudflare's managed challenge takes 3-8 s in a browser it rates human.
_CHALLENGE_WAIT_SECS = 12.0

# How long the Turnstile "Verify you are human" box must have been showing
# before it is pressed: a person's reading/reaction time, during which the
# pointer keeps drifting.
_TURNSTILE_REACTION_SECS = 1.2


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

_LAUNCH_LOCK = threading.RLock()

_BROWSER_CMD_MARKERS = ("chrom", "playwright", "headless")


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
        "Chromium (pid %d) could not be killed", pid,
    )


def _killable_live_pid(pid: int | None) -> bool:
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
        True when a watchdog may be armed against *pid*.
    """
    return (
        pid is not None and pid > 0 and pid != os.getpid() and _pid_alive(pid)
    )


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


def _read_lock_pid(
    profile_dir: str, *, propagate_permission_error: bool = False,
) -> int | None:
    """Return the PID recorded in a profile's ``SingletonLock`` symlink.

    Chromium's lock symlink targets ``hostname-pid``.  Returns ``None``
    when the lock is absent or unparsable.

    Args:
        profile_dir: Path to the Chromium user-data directory.
        propagate_permission_error: Re-raise a permission failure while
            reading the symlink.  The profile-in-use check enables this
            conservative mode because an inaccessible lock may belong
            to a live foreign-owned Chromium and must not look free.

    Returns:
        The recorded PID, or ``None``.
    """
    lock_path = Path(profile_dir) / "SingletonLock"
    if not lock_path.is_symlink():
        return None
    try:
        target = os.readlink(str(lock_path))
        pid = int(target.rsplit("-", 1)[-1])
        return pid if pid > 0 else None
    except PermissionError:
        if propagate_permission_error:
            raise
        return None
    except (OSError, ValueError, IndexError):
        return None


def _is_profile_in_use(profile_dir: str) -> bool:
    """Check whether a Chromium profile directory is locked by a running process.

    On POSIX Chromium creates a ``SingletonLock`` symlink whose target is
    ``hostname-pid`` when a profile is opened.  If the symlink exists and
    the referenced PID is alive, the profile is considered in use.  On
    Windows Chromium instead holds ``lockfile`` open without write
    sharing (delete-on-close, so a crash leaves nothing behind); the
    profile is in use while that file cannot be opened for writing.

    Args:
        profile_dir: Path to the Chromium user-data directory.

    Returns:
        True if the profile is currently locked by a live process.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows-only branch
        try:
            # ``r+`` never creates the file, so a probe that races the
            # browser's delete-on-close cannot leave a stale lockfile.
            with open(Path(profile_dir) / "lockfile", "r+"):
                return False
        except FileNotFoundError:
            return False
        except PermissionError:
            return True
    try:
        pid = _read_lock_pid(profile_dir, propagate_permission_error=True)
    except PermissionError:
        return True
    return pid is not None and _pid_alive(pid)


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
        elements.append({
            "role": role,
            "name": name,
            "occurrence": str(occurrence),
            "role_occurrence": str(role_occurrence),
        })
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
    check — :meth:`show_browser` reopens the same profile in a visible
    window and re-navigates to the current page.
    """

    _DEFAULT_USER_DATA_DIR = "__kiss_default_browser_profile__"

    def __init__(
        self,
        viewport: tuple[int, int] = (1280, 900),
        user_data_dir: str | None = _DEFAULT_USER_DATA_DIR,
        headless: bool = True,
        work_dir: str | None = None,
        ephemeral: bool = False,
        **_kwargs: Any,
    ) -> None:
        self._ephemeral_dir: str | None = None
        if ephemeral:
            self._ephemeral_dir = tempfile.mkdtemp(prefix="kiss_web_profile_")
            user_data_dir = self._ephemeral_dir
        elif user_data_dir == self._DEFAULT_USER_DATA_DIR:
            user_data_dir = str(_default_kiss_dir() / "browser_profile")
        self.viewport = viewport
        self.user_data_dir = user_data_dir
        self._headless = headless
        self.work_dir = work_dir
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._elements: list[dict[str, str]] = []
        self._browser_pid: int | None = None
        self._browser_identity: str | None = None
        # Chromium's ``--headless`` switch for the current launch: False
        # when the "no window" default runs headed on a virtual display.
        self._chromium_headless = headless
        # Last pointer position the tool moved to; curved mouse paths
        # start here.  ``None`` until the first move after a launch.
        self._mouse_xy: tuple[float, float] | None = None
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
        handle = self._page.wait_for_function(
            "() => [innerWidth, innerHeight]", timeout=_PAGE_READ_TIMEOUT_MS, polling=100
        )
        width, height = handle.json_value()
        return int(width), int(height)

    def _is_alive(self) -> bool:
        """Return True iff the current page/context survived (not crashed/closed)."""
        if self._playwright is None or self._context is None or self._page is None:
            return False
        try:
            return not self._page.is_closed()
        except Exception:  # pragma: no cover — Playwright internals rarely throw here
            logger.debug("Exception caught", exc_info=True)
            return False

    def _adopt_page(self, page: Any) -> None:
        """Make *page* the active page and arm the renderer-crash handler.

        Every path that points ``self._page`` at a page must register
        ``_on_page_crash`` too: a renderer crash on an unwatched page
        leaves ``_page`` referencing a crashed-but-not-closed page that
        ``_is_alive`` still reports live, wedging every later call.
        """
        self._page = page
        self._page.on("crash", self._on_page_crash)

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

        Called when the browser main process exits (``context.on("close")``).
        The Playwright driver (``self._playwright``) is kept running so that the
        next tool call can launch a fresh browser without restarting the driver
        (sync_playwright cannot be restarted in the same process).
        """
        self._page = None
        self._context = None
        self._browser = None
        self._elements = []

    def _close_browser_only(self) -> None:
        """Close context/browser if present, leaving self._playwright running.

        A failed graceful close (wedged driver connection, cross-thread
        greenlet error) is logged at WARNING and followed by
        :meth:`_kill_browser_process`, which guarantees the Chromium OS
        process actually exits instead of leaking forever.  A watchdog
        timer covers the remaining failure mode: a graceful close that
        HANGS (never returns) — after ``_CLOSE_WATCHDOG_SECS`` it kills
        the browser process directly, which also unwedges the hung call.
        """
        pid = self._browser_pid
        identity = self._browser_identity
        watchdog: threading.Timer | None = None
        if (
            self._context is not None or self._browser is not None
        ) and _killable_live_pid(pid):
            watchdog = threading.Timer(
                _CLOSE_WATCHDOG_SECS, _watchdog_kill, args=(pid, identity),
            )
            watchdog.daemon = True
            watchdog.start()
        try:
            for obj in (self._context, self._browser):
                if obj is None:
                    continue
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
        if pid is None or pid <= 0 or pid == os.getpid():
            return
        if _wait_pid_exit(pid, 2.0):
            return
        logger.warning("Chromium (pid %d) survived graceful close", pid)
        _terminate_pid_escalating(pid, identity)

    def _capture_browser_pid(self, profile_dir: str | None) -> None:
        """Record the OS PID of the just-launched Chromium main process.

        Primary source: a browser-level CDP session
        (``SystemInfo.getProcessInfo``), which works for both persistent
        and non-persistent contexts.  Fallback: the profile's
        ``SingletonLock`` symlink (older Chromium versions).  A recorded
        PID lets :meth:`_kill_browser_process` guarantee the process dies
        even when the graceful Playwright close fails.

        Args:
            profile_dir: The effective user-data directory of the launch,
                or ``None`` for a non-persistent context.
        """
        self._browser_pid = None
        self._browser_identity = None
        browser = self._browser
        if browser is None and self._context is not None:
            browser = getattr(self._context, "browser", None)
        if browser is not None:
            try:
                cdp = browser.new_browser_cdp_session()
                try:
                    info = cdp.send("SystemInfo.getProcessInfo")
                finally:
                    try:
                        cdp.detach()
                    except Exception:  # pragma: no cover — detach rarely fails
                        logger.debug("CDP detach failed", exc_info=True)
                for proc in info.get("processInfo", []):
                    if proc.get("type") == "browser":
                        self._browser_pid = int(proc["id"])
                        self._browser_identity = _process_identity(
                            self._browser_pid,
                        )
                        return
            except Exception:  # pragma: no cover — CDP rarely fails
                logger.debug("CDP browser PID capture failed", exc_info=True)
        if profile_dir:  # pragma: no cover — lock-file fallback path
            pid = _read_lock_pid(profile_dir)
            if pid is None:
                return
            identity = _process_identity(pid)
            if identity and any(
                marker in identity.lower() for marker in _BROWSER_CMD_MARKERS
            ):
                self._browser_pid = pid
                self._browser_identity = identity

    def _cleanup_stale_escalation_dirs(self) -> None:
        """Delete stale ``<user_data_dir>_N`` escalation profile directories.

        ``_resolve_user_data_dir`` escalates to numbered profile variants
        when the base profile is locked by a live Chromium.  Crashed or
        leaked Chromiums leave those directories behind with dead
        ``SingletonLock`` PIDs; remove them so escalation dirs cannot
        accumulate across crash/relaunch cycles.  Only directories whose
        lock PID is provably dead are removed — the base profile, live
        profiles, and lock-less directories are never touched.
        """
        if not self.user_data_dir or self._ephemeral_dir:
            return
        for i in range(1, 100):
            candidate = f"{self.user_data_dir}_{i}"
            pid = _read_lock_pid(candidate)
            if pid is None or _pid_alive(pid):
                continue
            _rmtree_logged(candidate)

    def _ensure_browser(self) -> None:
        """Ensure a Playwright browser page is ready, installing Chromium if needed.

        Detects and recovers from a previously-crashed Chromium by tearing down
        stale references and relaunching. This handles the common case where
        "Google Chrome for Testing quit unexpectedly" leaves the tool with a
        dead page that would otherwise fail every subsequent call.
        """
        if self._is_alive():
            return
        if self._page is not None and self._context is not None:
            try:
                pages = [p for p in self._context.pages if not p.is_closed()]
            except Exception:  # pragma: no cover — context already dead
                logger.debug("Exception caught", exc_info=True)
                pages = []
            if pages:
                self._adopt_page(pages[-1])
                self._elements = []
                return
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
            kwargs: dict[str, Any] = {
                "headless": self._chromium_headless,
                # Google Chrome when installed; otherwise the "chromium"
                # channel selects the full Chromium binary, which headless
                # runs in Chrome's new headless mode (the same renderer as
                # a headed window) instead of the stripped-down
                # chrome-headless-shell.
                "channel": web_stealth.chrome_channel(),
                "args": [
                    "--disable-blink-features=AutomationControlled",
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
                subprocess.run(
                    [sys.executable, "-m", web_stealth.playwright_package(),
                     "install", "chromium"],
                    check=True,
                    capture_output=True,
                    timeout=900,
                )
                self._launch_browser(launcher, kwargs)
        except Exception:  # pragma: no cover — Playwright init failure
            self.close()
            raise
        finally:
            _activate_app(prev_app)

    def _clean_singleton_locks(self, profile_dir: str | None = None) -> None:
        """Remove stale Singleton* files from a previously crashed Chromium.

        Chromium writes Singleton{Lock,Cookie,Socket} when a persistent profile
        is opened. If the process dies without cleaning up, the next launch
        may fail or crash. Safe to call unconditionally — live Chromium
        recreates the files during startup.

        Args:
            profile_dir: Directory to clean.  Falls back to ``self.user_data_dir``
                when *None*.
        """
        target = profile_dir or self.user_data_dir
        if not target:
            return
        for name in _SINGLETON_FILES:
            path = Path(target) / name
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
        free one is found.

        Returns:
            An available profile directory path, or ``None`` to fall back to
            a non-persistent (temporary) context.
        """
        if not self.user_data_dir:
            return None
        if not _is_profile_in_use(self.user_data_dir):
            return self.user_data_dir
        for i in range(1, 100):
            candidate = f"{self.user_data_dir}_{i}"
            if not _is_profile_in_use(candidate):
                return candidate
        return None  # pragma: no cover — 100 concurrent instances is unlikely

    def _profile_lock(self) -> Any:
        """Return a machine-wide lock over this tool's profile family.

        The profile directory (and its ``_N`` escalation variants) is
        shared by every kiss process on the machine, so the
        check-then-use sequence in :meth:`_launch_browser` must be
        atomic across *processes*: otherwise two of them both see a
        lock-free profile, the second deletes the first's live
        ``SingletonLock``, and Chromium either opens one profile twice
        (corrupting the stored logins) or aborts with "Failed to create
        a ProcessSingleton for your profile directory".

        One lock file sits beside the base profile and therefore covers
        every escalation variant derived from it.

        Returns:
            A context manager holding the lock, or a no-op context
            manager when this tool uses no persistent profile.
        """
        if not self.user_data_dir:
            return nullcontext()
        return _file_lock(Path(f"{self.user_data_dir}.lock"))

    def _launch_browser(self, launcher: Any, kwargs: dict[str, Any]) -> None:
        with _LAUNCH_LOCK, self._profile_lock():
            self._cleanup_stale_escalation_dirs()
            effective_dir = self._resolve_user_data_dir()
            self.effective_user_data_dir = effective_dir
            if effective_dir:
                Path(effective_dir).mkdir(parents=True, exist_ok=True)
                self._clean_singleton_locks(effective_dir)
                self._context = launcher.launch_persistent_context(
                    effective_dir, **kwargs, **self._context_args()
                )
                self._capture_browser_pid(effective_dir)
                page = (
                    self._context.pages[0] if self._context.pages
                    else self._context.new_page()
                )
            else:
                self._browser = launcher.launch(**kwargs)
                self._capture_browser_pid(None)
                self._context = self._browser.new_context(**self._context_args())
                page = self._context.new_page()
        self._context.route(_ACCOUNTS_GOOGLE_URL_RE, _abort_route)
        self._context.on("close", self._on_browser_lost)
        self._adopt_page(page)
        self._mouse_xy = None
        self._mask_headless_user_agent()

    def _mask_headless_user_agent(self) -> None:
        """Rewrite the ``HeadlessChrome`` user-agent token to ``Chrome``.

        Headless Chromium advertises ``HeadlessChrome/<version>``, and many
        sites treat that token alone as a bot signal and answer with a
        challenge page instead of content.  It is rewritten in both places
        a site can read it: the ``User-Agent`` request header (context
        wide) and ``navigator.userAgent`` (an init script that runs in
        every page and frame of the context).  A headed browser reports no
        such token, so this is a no-op there.
        """
        try:
            user_agent = self._page.evaluate("navigator.userAgent")
            if _HEADLESS_UA_TOKEN not in user_agent:
                return
            headed = user_agent.replace(_HEADLESS_UA_TOKEN, _HEADED_UA_TOKEN)
            self._context.set_extra_http_headers({"User-Agent": headed})
            self._context.add_init_script(
                "Object.defineProperty(navigator, 'userAgent', "
                f"{{get: () => {json.dumps(headed)}}});",
            )
        except Exception:  # pragma: no cover — evaluate on a fresh page rarely fails
            logger.debug("Could not mask the headless user agent", exc_info=True)

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
        handle = page.wait_for_function(
            "() => [document.title]", timeout=_PAGE_READ_TIMEOUT_MS, polling=100
        )
        return str(handle.json_value()[0])

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
        self._page.wait_for_function(
            "() => 1", timeout=_PAGE_READ_TIMEOUT_MS, polling=100
        )

    def _input_hang_watchdog(self, deadline_secs: float = _INPUT_WATCHDOG_SECS) -> Any:
        """Kill Chromium if the guarded raw input call outlives *deadline_secs*.

        Raw input operations (``keyboard.press``/``type``,
        ``mouse.move``/``wheel``) carry no timeout in the Playwright
        protocol, so a page event handler that enters ``while(true)`` on
        ``keydown``/``mousemove``/``wheel`` blocks them forever — after the
        pre-input liveness probe already passed.  Killing the browser
        process unwedges the pending driver call (it raises), the tool
        returns its documented error string, and the next tool call
        relaunches a fresh browser via ``_ensure_browser``.  A page in
        that state is permanently wedged, so losing the session is the
        correct recovery, exactly like the graceful-close watchdog.

        Args:
            deadline_secs: Seconds the guarded block may run before the
                watchdog fires.  Callers whose legitimate duration scales
                with input size (``type_text`` delay, ``scroll`` steps)
                pass a proportionally larger deadline.

        Returns:
            A ``threading.Timer``-cancelling context manager guarding the
            ``with`` block.
        """
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
                random.uniform(vw * 0.2, vw * 0.8), random.uniform(vh * 0.2, vh * 0.8)
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
        self._require_responsive_renderer()
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
        handle = self._page.wait_for_function(
            "() => [document.title, document.body ? document.body.innerText.slice(0, 1500) : '']",
            timeout=_PAGE_READ_TIMEOUT_MS,
            polling=100,
        )
        title, body_head = handle.json_value()
        return web_stealth.challenge_vendor(str(title), str(body_head))

    def _turnstile_checkbox(self) -> dict[str, float] | None:
        """Return the box of Cloudflare Turnstile's "Verify you are human" checkbox.

        Cloudflare's managed challenge first runs silent checks; when they
        do not rate the browser human enough it shows a single checkbox
        (no puzzle) for the visitor to tick.  The box lives in a
        cross-origin ``challenges.cloudflare.com`` iframe.  Nothing else is
        ever solved on the user's behalf: CAPTCHA puzzles, press-and-hold
        widgets and hard blocks are reported for :meth:`show_browser`.

        Returns:
            The checkbox's bounding box in main-frame viewport pixels, or
            ``None`` while no widget, or only its spinner, is showing.
        """
        frame = next(
            (f for f in self._page.frames if "challenges.cloudflare.com" in f.url), None
        )
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

    def _press_at(self, box: dict[str, float]) -> None:
        """Press the left-hand square of the accessible element *box* like a person.

        The Turnstile element spans the square and its "Verify you are
        human" label; people press the square.  A curved approach, a short
        pause and a held button, all by pointer coordinates because the
        element is in a cross-origin frame.

        Args:
            box: Bounding box in main-frame viewport pixels.
        """
        side = min(box["width"], box["height"])
        x = box["x"] + side * random.uniform(0.35, 0.65)
        y = box["y"] + box["height"] * random.uniform(0.35, 0.65)
        with self._input_hang_watchdog():
            self._move_mouse_to(x, y)
            self._page.wait_for_timeout(random.randint(200, 600))
            self._page.mouse.down()
            self._page.wait_for_timeout(random.randint(60, 140))
            self._page.mouse.up()
        logger.info("ticked the Cloudflare Turnstile checkbox on %s", self._page.url)

    def _settle_challenge(self, response: Any) -> str:
        """Wait out a bot-protection interstitial; describe it when it stays.

        Cloudflare's managed challenge (and Anubis' proof-of-work page)
        run their checks in the browser and then reload the real page on
        their own — for a browser they rate as human.  Returning the
        interstitial's accessibility tree would make the agent retry or
        give up, so the tool waits up to :data:`_CHALLENGE_WAIT_SECS`
        with small pointer movements for the page to clear.  A Google
        "unusual traffic" page is not waited on: it rates the network's
        IP, not the browser, so the same query is opened on Bing.

        Args:
            response: The navigation response from ``page.goto`` (may be
                ``None`` for ``about:`` and same-document navigations).

        Returns:
            ``""`` when the page is content, otherwise a ``Note:`` line
            for the agent explaining what blocked the page.
        """
        mitigated = response is not None and response.headers.get("cf-mitigated") == "challenge"
        vendor = self._challenge_vendor()
        if vendor is None and not mitigated:
            return ""
        deadline = time.monotonic() + _CHALLENGE_WAIT_SECS
        box_seen_at: float | None = None
        ticked = False
        while (
            vendor is not None
            and not vendor.startswith("Google")
            and time.monotonic() < deadline
        ):
            box = None
            if vendor == "Cloudflare challenge" and not ticked:
                box = self._turnstile_checkbox()
            if box is not None and box_seen_at is None:
                box_seen_at = time.monotonic()
            if (
                box is not None
                and box_seen_at is not None
                and time.monotonic() - box_seen_at >= _TURNSTILE_REACTION_SECS
            ):
                # A person reads the box before pressing it, moving the
                # pointer meanwhile; a press the instant it appears, with
                # no pointer history, is what Turnstile ignores.  The
                # verdict then takes a few seconds.
                self._press_at(box)
                ticked = True
                deadline = max(deadline, time.monotonic() + _CHALLENGE_WAIT_SECS)
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
            f"clear within {_CHALLENGE_WAIT_SECS:.0f}s. Call show_browser() so the user "
            "can complete the check, or use another source for the same information."
        )

    def _get_ax_tree(self, max_chars: int = 50000) -> str:
        self._ensure_browser()
        header = f"Page: {self._page_title(self._page)}\nURL: {self._page.url}\n\n"
        snapshot = self._page.locator("body").aria_snapshot(timeout=_PAGE_READ_TIMEOUT_MS)
        if not snapshot:
            self._elements = []
            return header + "(empty page)"
        numbered, self._elements = _number_interactive_elements(snapshot)
        if len(numbered) > max_chars:
            numbered = numbered[:max_chars] + "\n... [truncated]"
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
        if len(pages) > 1 and pages[-1] != self._page:  # pragma: no branch
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

            response = self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self._wait_for_stable()
            notice = self._settle_challenge(response)
            tree = self._get_ax_tree()
            return f"{notice}\n\n{tree}" if notice else tree
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error navigating to {url}: {e}"

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
            if len(self._context.pages) > pages_before:
                self._check_for_new_tab()
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
            path = Path(_absolutize(file_path, self.work_dir)).resolve()
            remapped = _active_worktree_remap(path, self.work_dir)
            if remapped is not None:
                path = remapped
            else:
                # Same contract as UsefulTools.Write: a path under a
                # worktree the framework already merged and removed must
                # fall back to the parent repo, or mkdir() would
                # resurrect a zombie worktree whose contents are never
                # merged and are deleted by the next prune.
                fallback = _stale_worktree_fallback(path)
                if fallback is not None:
                    path = fallback
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

    def close(self) -> str:
        """Close the browser and release resources. Call when done with the session or before exit.

        Returns:
            "Browser closed." (always, even if nothing was open)."""
        self._close_browser_only()
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
        profile, so logins are preserved.

        Returns:
            "Browser closed. It will relaunch automatically on the next web tool call."."""
        self._close_browser_only()
        return (
            "Browser closed. It will relaunch automatically on the next "
            "web tool call."
        )

    def show_browser(self, visible: bool = True) -> str:
        """Show the Chromium window on screen. Browsing is headless by default.

        Call this when a page needs the human in front of the screen: an
        interactive login or OAuth consent, a CAPTCHA, an "unusual traffic"
        bot check, or when the user asks to watch what you are doing. The
        same browser profile is reused, so cookies and logins carry over,
        and the page you are on is reopened in the visible window. Pass
        visible=False to go back to the headless window when the human
        part is done.

        Args:
            visible: True to reopen the browser in a window the user can see
                and interact with, False to return to headless browsing.

        Returns:
            The accessibility tree of the reopened page, or
            "Browser is now visible."/"Browser is now headless." when no page
            was open, or "Error <doing something>: <message>" on failure."""
        state = "visible" if visible else "headless"
        if self._headless == (not visible) and self._is_alive():
            return f"Browser is already {state}."
        url, cookies = self._capture_session()
        self._close_browser_only()
        self._headless = not visible
        err = self._try_ensure_browser(f"making the browser {state}")
        if err is not None:
            return err
        self._restore_cookies(cookies)
        if url:
            return self.go_to_url(url)
        return f"Browser is now {state}."

    def _capture_session(self) -> tuple[str, list[Any]]:
        """Return the page to reopen and the cookies to carry across a relaunch.

        Chromium cannot switch between headless and visible without being
        restarted, and a restart drops every session-only cookie — exactly
        the cookies a login or bot-check flow is in the middle of setting.
        They are handed back to the new browser by
        :meth:`_restore_cookies`.

        Returns:
            The URL to reopen (empty when nothing worth reopening is
            loaded) and the cookies of the current context.
        """
        if not self._is_alive():
            return "", []
        # The user may have opened a tab themselves while the window was
        # visible; that newest tab is the one worth carrying over.
        self._check_for_new_tab()
        url = self._page.url
        if url.startswith("about:"):
            url = ""
        try:
            return url, self._context.cookies()
        except Exception:  # pragma: no cover — reading cookies rarely fails
            logger.debug("Could not read cookies before relaunch", exc_info=True)
            return url, []

    def _restore_cookies(self, cookies: list[Any]) -> None:
        """Add *cookies* to the freshly launched context.

        Args:
            cookies: Cookies captured by :meth:`_capture_session`.
        """
        if not cookies:
            return
        try:
            self._context.add_cookies(cookies)
        except Exception:  # pragma: no cover — a malformed cookie is rare
            logger.debug("Could not restore cookies after relaunch", exc_info=True)

    def get_tools(self) -> list[Callable[..., str]]:
        """Return callable web tools for registration with an agent.

        Returns:
            List of callables: go_to_url, click, type_text, press_key, scroll, screenshot,
            get_page_content, show_browser, close_browser. Does not include close."""
        return [
            self.go_to_url,
            self.click,
            self.type_text,
            self.press_key,
            self.scroll,
            self.screenshot,
            self.get_page_content,
            self.show_browser,
            self.close_browser,
        ]
