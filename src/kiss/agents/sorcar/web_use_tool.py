# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Browser automation tool for LLM agents using Playwright.

Uses non-headless Playwright Chromium for page analysis and automation
(accessibility tree, clicking, typing, screenshots).
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.persistence import _default_kiss_dir
from kiss.core.useful_tools import _absolutize, _active_worktree_remap

logger = logging.getLogger(__name__)

_SINGLETON_FILES = ("SingletonLock", "SingletonCookie", "SingletonSocket")

_ACCOUNTS_GOOGLE_URL_RE = re.compile(r"^https?://accounts\.google\.com/")


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

# Role lines look like ``- button "Name"`` — but when the accessible
# name contains ``": "`` Playwright single-quote-wraps the whole YAML
# key (``- 'link "colon: \"q\""':``), so an optional leading ``'`` must
# be accepted or the element is silently never numbered.
_ROLE_LINE_RE = re.compile(r"^(\s*)-\s+('?)([\w]+)\s*(.*)")

# Accessible names are double-quoted with YAML escaping: ``\"`` for an
# embedded quote and ``\\`` for a backslash.  A naive ``"([^"]*)"``
# stops at the first embedded quote and records a corrupted name that
# get_by_role(name=..., exact=True) can never match.
_NAME_RE = re.compile(r'"((?:\\.|[^"\\])*)"')
_NAME_UNESCAPE_RE = re.compile(r'\\(["\\])')

_SCROLL_DELTA = {"down": (0, 300), "up": (0, -300), "right": (300, 0), "left": (-300, 0)}


def _pid_alive(pid: int) -> bool:
    """Return True iff the OS process *pid* currently exists.

    Args:
        pid: Process id to probe with a null signal.

    Returns:
        True when the process exists (even if owned by another user),
        False when it does not.
    """
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:  # pragma: no cover — foreign-owned process
        return True
    except OSError:
        return False


# Seconds a graceful Playwright close may take before the watchdog kills
# the Chromium process directly (a wedged driver connection can otherwise
# hang the close call forever, leaking the browser).
_CLOSE_WATCHDOG_SECS = 15.0

# Serializes stale-dir cleanup + profile resolution + launch within this
# process so two concurrently-launching tools can never clean/select the
# same profile directory out from under each other.
_LAUNCH_LOCK = threading.RLock()

# Substrings identifying a Chromium/Playwright browser process command
# line; the SingletonLock PID fallback only trusts PIDs whose process
# matches (the lock could be corrupt or its PID recycled).
_BROWSER_CMD_MARKERS = ("chrom", "playwright", "headless")


def _process_identity(pid: int) -> str | None:
    """Return a stable identity string (start time + command) for *pid*.

    Used to detect PID reuse before sending kill signals: two different
    processes can never share both a start timestamp and a command line.

    Args:
        pid: Process id to fingerprint.

    Returns:
        The ``ps`` ``lstart``+``command`` line, or ``None`` when the
        process is gone or ``ps`` failed.
    """
    try:
        r = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:  # pragma: no cover — ps missing/unresponsive
        logger.debug("Exception caught", exc_info=True)
        return None


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
    sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sig_kill):
        if not _pid_alive(pid):
            return
        current = _process_identity(pid)
        if current is None:
            # Process vanished between the checks (or ps failed): there
            # is nothing that can be safely signalled.
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
        logger.warning("Killing leaked Chromium (pid %d) with %s", pid, sig.name)
        try:
            os.kill(pid, sig)
        except OSError:  # pragma: no cover — died between checks
            return
        if _wait_pid_exit(pid, 2.0):
            return
    logger.error(  # pragma: no cover — SIGKILL cannot be ignored
        "Chromium (pid %d) could not be killed", pid,
    )


def _watchdog_kill(pid: int, identity: str | None) -> None:  # pragma: no cover
    """Watchdog timer body: kill a Chromium whose graceful close hung.

    Killing the browser process also unwedges the hung driver call (the
    driver observes the browser exit and completes/raises the close).

    Args:
        pid: Browser process id recorded at launch.
        identity: Identity string recorded at PID capture time.
    """
    logger.warning(
        "Graceful browser close is hung after %.0fs; killing Chromium "
        "(pid %d) directly",
        _CLOSE_WATCHDOG_SECS,
        pid,
    )
    _terminate_pid_escalating(pid, identity)


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

    Chromium creates a ``SingletonLock`` symlink whose target is
    ``hostname-pid`` when a profile is opened.  If the symlink exists and
    the referenced PID is alive, the profile is considered in use.

    Args:
        profile_dir: Path to the Chromium user-data directory.

    Returns:
        True if the profile is currently locked by a live process.
    """
    # Composition of the two shared helpers: ``_read_lock_pid``
    # already rejects absent/unparsable/corrupt locks (including the
    # ``pid <= 0`` case — ``os.kill(0, 0)`` signals the caller's own
    # process group and always succeeds, which would mark the profile
    # permanently in use), and ``_pid_alive`` implements the
    # EPERM-means-alive liveness probe.
    try:
        pid = _read_lock_pid(profile_dir, propagate_permission_error=True)
    except PermissionError:
        # Preserve the old inline check's conservative behaviour: a
        # lock symlink we cannot inspect may belong to another user's
        # live Chromium, so never launch a second browser into it.
        return True
    return pid is not None and _pid_alive(pid)


def _number_interactive_elements(snapshot: str) -> tuple[str, list[dict[str, str]]]:
    result_lines: list[str] = []
    elements: list[dict[str, str]] = []
    counter = 0
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
            # The whole key is a YAML *single-quoted* scalar, in which
            # an embedded apostrophe is escaped by doubling it
            # (``- 'link "Bob''s: list"':``).  Collapse the doubling or
            # get_by_role(name=..., exact=True) can never match.
            name = name.replace("''", "'")
        elements.append({"role": role, "name": name})
        result_lines.append(f"{indent}- [{counter}] {quote}{role} {rest}".rstrip())
    return "\n".join(result_lines), elements


class WebUseTool:
    """Browser automation tool using non-headless Playwright Chromium.

    The user can see and interact with the Chromium window directly.
    All browsing (including user-interaction flows like OAuth, CAPTCHAs)
    happens in this single Chromium instance.
    """

    # Sentinel meaning "use the default profile dir under the KISS home".
    # The actual path is resolved lazily at construction time via
    # ``_default_kiss_dir()`` so it respects the ``KISS_HOME`` env var even
    # when KISS_HOME is set after package import (as the test conftest does).
    _DEFAULT_USER_DATA_DIR = "__kiss_default_browser_profile__"

    def __init__(
        self,
        viewport: tuple[int, int] = (1280, 900),
        user_data_dir: str | None = _DEFAULT_USER_DATA_DIR,
        headless: bool = False,
        work_dir: str | None = None,
        ephemeral: bool = False,
        **_kwargs: Any,
    ) -> None:
        # Ephemeral mode (parallel sub-agents): a throwaway profile in a
        # fresh temp directory, deleted by close().  This keeps sub-agents
        # off the user's persistent profile so they never escalate to
        # ``browser_profile_N`` dirs (one leaked visible window each).
        self._ephemeral_dir: str | None = None
        if ephemeral:
            self._ephemeral_dir = tempfile.mkdtemp(prefix="kiss_web_profile_")
            user_data_dir = self._ephemeral_dir
        elif user_data_dir == self._DEFAULT_USER_DATA_DIR:
            user_data_dir = str(_default_kiss_dir() / "browser_profile")
        self.viewport = viewport
        self.user_data_dir = user_data_dir
        self._headless = headless
        # Agent working directory: relative screenshot paths are
        # anchored here, consistent with the Read/Write/Edit/Bash tools.
        self.work_dir = work_dir
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._elements: list[dict[str, str]] = []
        # OS PID + identity fingerprint of the Chromium main process,
        # recorded at launch so a failed graceful close can escalate to
        # killing the process (identity guards against PID reuse).
        self._browser_pid: int | None = None
        self._browser_identity: str | None = None
        atexit.register(self.close)

    def _context_args(self) -> dict[str, Any]:
        return {
            "viewport": {"width": self.viewport[0], "height": self.viewport[1]},
            "locale": "en-US",
            "timezone_id": "America/Los_Angeles",
            "java_script_enabled": True,
            "has_touch": False,
            "is_mobile": False,
            "device_scale_factor": 2,
        }

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
            (self._context is not None or self._browser is not None)
            and pid is not None
            and pid > 0
            and pid != os.getpid()
            and _pid_alive(pid)
        ):
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
        # A graceful close that succeeded needs only a moment to finish.
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
            # The lock could be corrupt or its PID recycled: only trust
            # it when the process actually looks like a browser.
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
        # Active tab closed (user hit the tab's ✕ / window.close()) but
        # the context survives with other tabs: adopt the most recent
        # surviving tab instead of tearing down the whole session and
        # discarding every open tab.  ``self._page is None`` means a
        # renderer *crash* (see _on_page_crash) — that path must still
        # fall through to a full teardown + relaunch.
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
        # Re-arm the atexit safety net (close() unregisters it).  The
        # unregister-then-register pattern keeps exactly one entry even
        # when the browser is relaunched many times.
        atexit.unregister(self.close)
        atexit.register(self.close)
        self._close_browser_only()
        from playwright.sync_api import sync_playwright

        prev_app = _get_frontmost_app()
        try:
            if self._playwright is None:
                self._playwright = sync_playwright().start()
            launcher = self._playwright.chromium
            kwargs: dict[str, Any] = {
                "headless": self._headless,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-infobars",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-breakpad",
                    "--noerrdialogs",
                    "--disable-dev-shm-usage",
                ],
            }

            try:
                self._launch_browser(launcher, kwargs)
            except Exception:  # pragma: no cover – Chromium always pre-installed in CI
                logger.info("Playwright Chromium not found, installing...")
                # A partially-launched browser (launch succeeded, later
                # setup raised) must be torn down before retrying or it
                # would leak alongside the retry's browser.
                self._close_browser_only()
                subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    check=True,
                    capture_output=True,
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

    def _launch_browser(self, launcher: Any, kwargs: dict[str, Any]) -> None:
        # The launch lock serializes cleanup + profile resolution + launch
        # so two concurrently-launching tools in this process can never
        # clean/select the same profile directory out from under each
        # other.
        with _LAUNCH_LOCK:
            self._cleanup_stale_escalation_dirs()
            effective_dir = self._resolve_user_data_dir()
            if effective_dir:
                Path(effective_dir).mkdir(parents=True, exist_ok=True)
                self._clean_singleton_locks(effective_dir)
                self._context = launcher.launch_persistent_context(
                    effective_dir, **kwargs, **self._context_args()
                )
                # Record the Chromium OS PID immediately — before any
                # post-launch setup that could raise — so close() can
                # guarantee the process dies even when setup fails or a
                # later graceful close fails.  NOTE: _on_browser_lost
                # deliberately does NOT clear the PID — after the driver
                # drops its references the PID is the only remaining
                # handle to a possibly-still-running process.
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
        # The accounts.google.com block applies to the tool as a whole,
        # not just persistent profiles: install it on every context.
        self._context.route(_ACCOUNTS_GOOGLE_URL_RE, _abort_route)
        self._context.on("close", self._on_browser_lost)
        self._adopt_page(page)

    def _get_ax_tree(self, max_chars: int = 50000) -> str:
        self._ensure_browser()
        header = f"Page: {self._page.title()}\nURL: {self._page.url}\n\n"
        snapshot = self._page.locator("body").aria_snapshot()
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
            snapshot = self._page.locator("body").aria_snapshot()
            if snapshot:
                _, self._elements = _number_interactive_elements(snapshot)
            if element_id < 1 or element_id > len(self._elements):
                raise ValueError(f"Element with ID {element_id} not found.")
        role = self._elements[element_id - 1]["role"]
        name = self._elements[element_id - 1]["name"]
        if name:
            locator = self._page.get_by_role(role, name=name, exact=True)
        else:
            locator = self._page.get_by_role(role)
        n = locator.count()
        if n == 0:  # pragma: no cover — race between snapshot and DOM
            raise ValueError(f"Element with ID {element_id} not found on page.")
        if n == 1:
            return locator
        for i in range(n):  # pragma: no branch — first visible element always found
            try:
                if locator.nth(i).is_visible():
                    return locator.nth(i)
            except Exception:  # pragma: no cover — Playwright is_visible rarely throws
                logger.debug("Exception caught", exc_info=True)
                continue
        return locator.first  # pragma: no cover — all elements invisible is rare

    def go_to_url(self, url: str) -> str:
        """Navigate the browser to a URL and return the page accessibility tree.
        Use when you need to open a new page or switch pages. Special values: "tab:list"
        returns a list of open tabs; "tab:N" switches to tab N (0-based).

        Args:
            url: Full URL to open, or "tab:list" for tab list, or "tab:N" to switch to tab N.

        Returns:
            On success: page title, URL, and accessibility tree with [N] IDs. For "tab:list":
            list of open tabs with indices. On error: "Error navigating to <url>: <message>"."""
        self._ensure_browser()
        try:
            pages = self._context.pages
            if url == "tab:list":
                lines = [f"Open tabs ({len(pages)}):"]
                for i, page in enumerate(pages):
                    suffix = " (active)" if page == self._page else ""
                    lines.append(f"  [{i}] {page.title()} - {page.url}{suffix}")
                return "\n".join(lines)
            if url.startswith("tab:"):
                idx = int(url[4:])
                if 0 <= idx < len(pages):
                    self._adopt_page(pages[idx])
                    return self._get_ax_tree()
                return f"Error: Tab index {idx} out of range (0-{len(pages) - 1})."

            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self._wait_for_stable()
            return self._get_ax_tree()
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
        self._ensure_browser()
        try:
            locator = self._resolve_locator(element_id)

            if action == "hover":
                locator.hover()
                self._page.wait_for_timeout(300)
                return self._get_ax_tree()

            pages_before = len(self._context.pages)
            locator.click()
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
        self._ensure_browser()
        try:
            locator = self._resolve_locator(element_id)
            select_all = "Meta+a" if sys.platform == "darwin" else "Control+a"
            locator.click()
            self._page.keyboard.press(select_all)
            self._page.keyboard.press("Backspace")
            self._page.keyboard.type(text, delay=50)
            if press_enter:
                self._page.keyboard.press("Enter")
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
        self._ensure_browser()
        try:
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
        self._ensure_browser()
        try:
            dx, dy = _SCROLL_DELTA.get(direction, (0, 300))
            vw, vh = self.viewport[0] // 2, self.viewport[1] // 2
            self._page.mouse.move(vw, vh)
            for _ in range(amount):
                self._page.mouse.wheel(dx, dy)
                self._page.wait_for_timeout(100)
            self._page.wait_for_timeout(300)
            return self._get_ax_tree()
        except Exception as e:  # pragma: no cover — Playwright scroll rarely fails
            logger.debug("Exception caught", exc_info=True)
            return f"Error scrolling {direction}: {e}"

    def screenshot(self, file_path: str = "screenshot.png") -> str:
        """Capture the visible viewport of the Chromium browser as an image.

        Use to verify layout, captchas, or visual state of a web page currently
        open in the browser. This does NOT capture or display local files,
        attached images, or PDFs — it only screenshots the browser window.

        Args:
            file_path: Path where the PNG will be saved (default "screenshot.png"). Parent
                directories are created if needed.

        Returns:
            "Screenshot saved to <resolved_path>", or
            "Error taking screenshot: <message>" on error."""
        self._ensure_browser()
        try:
            # Anchor the path the same way the file tools do: expand
            # ``~`` to the user's home (never a literal ``./~/`` dir),
            # then resolve relative paths against the agent work_dir
            # (NOT the daemon process cwd — in worktree mode that
            # would silently escape the worktree).  Reuses the exact
            # helper the Read/Write/Edit tools use so the two path
            # policies can never drift.
            path = Path(_absolutize(file_path, self.work_dir)).resolve()
            # Active-worktree remap, mirroring Write/Edit: an absolute
            # parent-repo path (model ignored the ``Work dir:`` hint)
            # must land inside the live ``.kiss-worktrees/kiss_wt-*``
            # worktree — never dirty the user's main checkout.
            remapped = _active_worktree_remap(path, self.work_dir)
            if remapped is not None:
                path = remapped
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
        self._ensure_browser()
        try:
            if text_only:
                title = self._page.title()
                url = self._page.url
                body = self._page.inner_text("body")
                return f"Page: {title}\nURL: {url}\n\n{body}"
            return self._get_ax_tree()
        except Exception as e:  # pragma: no cover — Playwright get content rarely fails
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
        # Drop the atexit registration so closed tools are not retained
        # for the process lifetime (one leaked entry per agent run).
        # ``_ensure_browser`` re-registers if this tool is revived.
        atexit.unregister(self.close)
        if self._ephemeral_dir:
            # Keep _ephemeral_dir set: a closed tool can be revived by
            # the next web tool call (_ensure_browser relaunches and
            # mkdir-recreates the dir), and the revived browser's profile
            # must be deleted again by the next close().
            _rmtree_logged(self._ephemeral_dir)
        return "Browser closed."

    def close_browser(self) -> str:
        """Close the Chromium browser window and free its OS process.

        Use when you are done with web browsing for now (its purpose is
        over) so the browser window does not stay open for the rest of a
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

    def get_tools(self) -> list[Callable[..., str]]:
        """Return callable web tools for registration with an agent.

        Returns:
            List of callables: go_to_url, click, type_text, press_key, scroll, screenshot,
            get_page_content, close_browser. Does not include close."""
        return [
            self.go_to_url,
            self.click,
            self.type_text,
            self.press_key,
            self.scroll,
            self.screenshot,
            self.get_page_content,
            self.close_browser,
        ]
