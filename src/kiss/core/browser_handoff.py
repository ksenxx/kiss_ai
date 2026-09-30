# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hand sign-in pages to the USER: the streamed Browser tab, else their browser.

Every connector's authentication ends with a page only the user may
complete: an OAuth consent screen, a device-code page, a developer
portal where a bot token is created.  The agent must never drive those
pages itself (they sit behind the user's login and second factor), so
the page is put in front of the user instead.  :func:`open_for_user`
tries, in order:

1. The Browser tab.  Inside the kiss-web daemon the machine's browser is
   streamed as a tab on every KISS surface (the web app and the VS Code
   extension, see :mod:`kiss.server.browser_tab`); the daemon registers
   its opener with :func:`set_browser_tab_opener`, and the page opens
   there with every surface switched to it.  The user is looking at the
   page already, so the agent must NOT tell them to open a URL.
2. The user's default browser on this machine
   (:func:`open_in_default_browser`).  The agent still shows the URL in
   the chat so the user can finish manually when no window appeared.
3. Nothing: the process runs headless without a daemon; the user has to
   open the URL themselves.

:func:`open_in_default_browser` honours ``$BROWSER`` (the ``webbrowser``
module convention: a command, with an optional ``%s`` placeholder for
the URL) and otherwise uses the platform opener (``open`` on macOS,
``os.startfile`` on Windows, ``xdg-open`` elsewhere).  It never waits
for the browser to finish (only a few seconds to catch an opener that
fails at once) and never raises; a headless environment simply yields
``False``.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Where open_for_user() put the page.
BROWSER_TAB = "browser_tab"  # the Browser tab streamed to every KISS surface
DEFAULT_BROWSER = "default_browser"  # the user's default browser on this machine
NOT_OPENED = ""

# Only pages we built ourselves (file://) or public sign-in pages.
_ALLOWED_SCHEMES = ("http", "https", "file")
# A URL opened this recently is not opened again: check_<service>_auth()
# may run several times while a credential is still missing, and one
# tab per attempt is what the user wants, not one per tool call.
_REOPEN_AFTER = 120.0
# How long to watch the opener for an immediate failure (unknown
# command, xdg-open without a handler) before assuming it succeeded.
_LAUNCH_GRACE = 3.0

# Longest a caller waits for another caller's in-flight launch of the
# same page (the Browser tab alone may take up to its 90 s open timeout).
_SETTLE_TIMEOUT = 120.0


@dataclass
class _Opening:
    """One page being (or recently) opened for the user.

    Attributes:
        opened_at: ``time.monotonic()`` when the launch was claimed.
        surface: :data:`BROWSER_TAB` or :data:`DEFAULT_BROWSER`.
        settled: ``True`` once the launch succeeded; a failed launch is
            removed from :data:`_recent` instead.
    """

    opened_at: float
    surface: str
    settled: bool = False


_recent: dict[str, _Opening] = {}
_lock = threading.Condition()
# Registered by the kiss-web daemon: opens a URL in the streamed Browser
# tab, focused on every surface; returns whether it could.
_browser_tab_opener: Callable[[str], bool] | None = None


def set_browser_tab_opener(opener: Callable[[str], bool] | None) -> None:
    """Register the daemon's Browser-tab opener (``None`` unregisters it).

    Args:
        opener: Called with a URL; opens it as a Browser tab that every
            KISS surface switches to and returns ``True``, or returns
            ``False`` (or raises) when the browser cannot be launched.
            Blocking is fine: the tool waits for the page to open.
    """
    global _browser_tab_opener
    _browser_tab_opener = opener


def _allowed(url: str) -> bool:
    """Whether *url* is a page the user may be handed (http(s) or file)."""
    try:
        return urlsplit(url).scheme.lower() in _ALLOWED_SCHEMES
    except ValueError:
        return False


def _claim(url: str, surface: str) -> str | None:
    """Record that *url* is being opened on *surface*, unless it is open already.

    Args:
        url: The page.
        surface: :data:`BROWSER_TAB` or :data:`DEFAULT_BROWSER`.

    Returns:
        Where *url* was opened within the last :data:`_REOPEN_AFTER`
        seconds (the caller must not open it again), or ``None`` once
        the claim is recorded.  While another caller's launch of the
        same page is in flight this waits for its outcome: a success is
        shared, a failure (the claim is released) lets this caller open
        the page itself.  So concurrent callers never open a second
        copy, nor report a page that never appeared.
    """
    deadline = time.monotonic() + _SETTLE_TIMEOUT
    with _lock:
        while True:
            now = time.monotonic()
            opening = _recent.get(url)
            if opening is None or now - opening.opened_at >= _REOPEN_AFTER:
                _recent[url] = _Opening(now, surface)
                return None
            if opening.settled or now >= deadline:
                return opening.surface
            _lock.wait(deadline - now)


def _settle(url: str) -> None:
    """Mark the claimed launch of *url* as succeeded and wake waiting callers."""
    with _lock:
        opening = _recent.get(url)
        if opening is not None:
            opening.settled = True
        _lock.notify_all()


def _release(url: str) -> None:
    """Forget a claim whose launch failed, so the next attempt opens again."""
    with _lock:
        _recent.pop(url, None)
        _lock.notify_all()


def is_headless_environment() -> bool:
    """Return True when running in a headless/Docker/Linux environment.

    Checks in order:
    1. KISS_HEADLESS env var (explicit override, "1"/"true"/"yes" → headless)
    2. Presence of /.dockerenv (running inside Docker)
    3. Linux with no $DISPLAY and no $WAYLAND_DISPLAY set
    """
    env = os.environ.get("KISS_HEADLESS", "").lower()
    if env in ("1", "true", "yes"):  # pragma: no branch
        return True
    if env in ("0", "false", "no"):  # pragma: no branch
        return False
    if Path("/.dockerenv").exists():  # pragma: no branch
        return True
    if sys.platform.startswith("linux"):  # pragma: no branch
        if (  # pragma: no branch
            not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
        ):
            return True
    return False


def _launch_commands(url: str) -> list[list[str]]:
    """Return the command lines that may open *url*, in order of preference.

    ``$BROWSER`` wins when set: every ``os.pathsep``-separated entry is a
    candidate (``%s`` is replaced by the URL, otherwise the URL is
    appended), tried in order until one starts.  Without it the platform
    opener is used.

    Args:
        url: The page to open.

    Returns:
        The candidate command lines; empty when ``$BROWSER`` is malformed.
    """
    custom = os.environ.get("BROWSER", "")
    if not custom.strip():
        return [["open", url]] if sys.platform == "darwin" else [["xdg-open", url]]
    commands = []
    for entry in custom.split(os.pathsep):
        if not entry.strip():
            continue
        try:
            # Non-POSIX splitting keeps Windows path backslashes; it also
            # keeps the quotes around a quoted path, hence the strip.
            argv = [arg.strip('"') for arg in shlex.split(entry, posix=os.name != "nt")]
        except ValueError:
            continue
        if any("%s" in arg for arg in argv):
            commands.append([arg.replace("%s", url) for arg in argv])
        else:
            commands.append([*argv, url])
    return commands


def _popen_args(command: list[str]) -> list[str] | str:
    """Return *command* in the form ``subprocess.Popen`` must receive it.

    A batch-file opener (``BROWSER=chrome.cmd``, the shape of every npm
    shim) is run by ``cmd.exe``, which parses the argument line itself
    and treats ``&``, ``|``, ``^`` and ``%`` as operators -- and
    ``Popen`` only quotes arguments containing whitespace, so an OAuth
    URL would be cut at its first ``&``.  Every argument of a batch
    file is therefore wrapped in double quotes, which ``cmd.exe`` and
    the script's ``%*`` pass through verbatim.

    Args:
        command: The opener command line.

    Returns:
        *command* itself, or the quoted command-line string for a
        ``.bat`` / ``.cmd`` opener on Windows.

    Raises:
        ValueError: If an argument for a batch file contains a double
            quote or a line break.  ``cmd.exe`` has no escape for a
            quote inside a quoted argument, so such an argument could
            end the quote and run the rest as commands; a URL never
            legitimately contains either (they are ``%22`` / ``%0A``).
    """
    if os.name != "nt" or not command[0].lower().endswith((".bat", ".cmd")):
        return command
    for arg in command[1:]:  # pragma: no cover
        if '"' in arg or "\r" in arg or "\n" in arg:
            raise ValueError(f"unsafe argument for a batch-file opener: {arg!r}")
    return " ".join(  # pragma: no cover
        [subprocess.list2cmdline(command[:1]), *(f'"{arg}"' for arg in command[1:])]
    )


def _launch(command: list[str]) -> bool:
    """Start *command* detached and report whether it looks successful.

    Args:
        command: The opener command line.

    Returns:
        ``True`` when the process exited 0 within the grace period or is
        still running after it (the opener is the browser itself, e.g.
        ``BROWSER=firefox``); ``False`` when it could not be started or
        exited with an error right away.
    """
    try:
        proc = subprocess.Popen(
            _popen_args(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError):
        return False
    try:
        return proc.wait(timeout=_LAUNCH_GRACE) == 0
    except subprocess.TimeoutExpired:
        return True


def open_in_default_browser(url: str) -> bool:
    """Try to open *url* in the user's default browser on this machine.

    Never raises: a malformed URL or ``$BROWSER`` simply counts as not
    opened.

    Args:
        url: An ``http(s)://`` sign-in page or a ``file://`` page this
            program wrote.  Any other scheme is refused.

    Returns:
        ``True`` when the page is (or was very recently) opened in a
        browser the user can see; ``False`` when this process runs
        headless (no display, Docker, ``KISS_HEADLESS=1``), the scheme
        is not allowed, or no opener could be started or all failed
        right away.  The caller must still show the URL to the user in
        both cases.
    """
    if not _allowed(url) or is_headless_environment():
        return False
    if _claim(url, DEFAULT_BROWSER) is not None:
        return True
    opened = False
    windows_shell = sys.platform == "win32" and not os.environ.get("BROWSER", "").strip()
    try:
        if windows_shell:  # pragma: no cover
            # Windows has no opener executable; the shell association is
            # only reachable through os.startfile (not testable on Linux).
            try:
                os.startfile(url)  # type: ignore[attr-defined]
                opened = True
            except OSError:
                opened = False
        else:
            opened = any(_launch(command) for command in _launch_commands(url))
    finally:
        # Settle or release even when interrupted, so no waiter hangs on
        # a claim that will never be resolved.
        if opened:
            _settle(url)
        else:
            _release(url)
    return opened


def open_for_user(url: str) -> str:
    """Put *url* in front of the user: in the Browser tab, else their browser.

    The Browser tab (registered with :func:`set_browser_tab_opener` by
    the kiss-web daemon) wins because every KISS surface switches to it,
    wherever the daemon runs; it is not subject to the headless check,
    since the streamed browser is headless by design.  Without a daemon,
    or when its browser cannot be launched, the user's default browser
    on this machine is tried.  Never raises.

    Args:
        url: An ``http(s)://`` sign-in page or a ``file://`` page this
            program wrote.  Any other scheme is refused.

    Returns:
        :data:`BROWSER_TAB` when the page is (or was very recently)
        opened in the streamed Browser tab, :data:`DEFAULT_BROWSER` when
        it is open in the user's default browser, :data:`NOT_OPENED`
        (empty string) when neither could be done.
    """
    if not _allowed(url):
        return NOT_OPENED
    opener = _browser_tab_opener
    if opener is not None:
        recent = _claim(url, BROWSER_TAB)
        if recent is not None:
            return recent
        opened = False
        try:
            opened = opener(url)
        except Exception as exc:  # noqa: BLE001 — a failed launch falls back
            logger.warning("browser tab: cannot open %s: %s", url, exc)
        finally:
            if opened:
                _settle(url)
            else:
                _release(url)
        if opened:
            return BROWSER_TAB
    return DEFAULT_BROWSER if open_in_default_browser(url) else NOT_OPENED


def browser_handoff_note(url: str, opened_in: str) -> str:
    """Describe to the agent what happened to *url* and what it must do next.

    Args:
        url: The page the user has to complete.
        opened_in: The result of :func:`open_for_user`.

    Returns:
        One or two sentences to embed in tool output: in the Browser tab
        the user is already looking at the page and must NOT be told to
        open a URL; in the default browser the URL is still shown via
        ask_user_question() as a fallback; when nothing opened the user
        must open it themselves.
    """
    if opened_in == BROWSER_TAB:
        return (
            f"{url} is now open in the Browser tab that every KISS surface (web app and "
            "VS Code) has just switched to, so the user is already looking at it. Tell "
            "the user what to do on that page and do NOT ask them to open a URL or "
            "give them this URL to open; only if the user says they cannot see the "
            "page, show it with ask_user_question()."
        )
    if opened_in == DEFAULT_BROWSER:
        return (
            f"{url} has just been opened in the user's default browser on this "
            "machine. Still show the user this exact URL with ask_user_question() "
            "so they can open it themselves if no browser window appeared."
        )
    return (
        "No browser could be opened from this machine (headless or remote), so "
        f"the user must open {url} themselves: show them this exact URL with "
        "ask_user_question() to open in their OWN browser."
    )


def portal_handoff(url: str) -> str:
    """Open a developer portal for the user and describe the outcome.

    For connectors whose credential is a token or key the user copies
    out of a web console: the console is opened for the user with
    :func:`open_for_user`, and the returned sentence tells the agent
    whether the user already sees it or must be given the URL.

    Args:
        url: The portal page where the credential is created.

    Returns:
        The :func:`browser_handoff_note` for *url*.
    """
    return browser_handoff_note(url, open_for_user(url))
