# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hand sign-in pages to the USER's default browser.

Every connector's authentication ends with a page only the user may
complete: an OAuth consent screen, a device-code page, a developer
portal where a bot token is created.  The agent must never drive those
pages itself (they sit behind the user's login and second factor), but
it can do the next best thing: open the page in the user's default
browser on this machine so that the user only has to approve, and then
show the URL (and code) in the chat so the user can finish manually
when no window appeared -- because the agent runs on a remote or
headless host, or because the launch failed.

:func:`open_in_default_browser` is that best effort.  It honours
``$BROWSER`` (the ``webbrowser`` module convention: a command, with an
optional ``%s`` placeholder for the URL) and otherwise uses the
platform opener (``open`` on macOS, ``os.startfile`` on Windows,
``xdg-open`` elsewhere).  It never waits for the browser to finish
(only a few seconds to catch an opener that fails at once) and never
raises; a headless environment simply yields ``False``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from kiss.agents.third_party_agents._backend_utils import is_headless_environment

# Only pages we built ourselves (file://) or public sign-in pages.
_ALLOWED_SCHEMES = ("http", "https", "file")
# A URL opened this recently is not opened again: check_<service>_auth()
# may run several times while a credential is still missing, and one
# tab per attempt is what the user wants, not one per tool call.
_REOPEN_AFTER = 120.0
# How long to watch the opener for an immediate failure (unknown
# command, xdg-open without a handler) before assuming it succeeded.
_LAUNCH_GRACE = 3.0

_opened_at: dict[str, float] = {}
_lock = threading.Lock()


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
    try:
        scheme = urlsplit(url).scheme.lower()
    except ValueError:
        return False
    if scheme not in _ALLOWED_SCHEMES or is_headless_environment():
        return False
    with _lock:
        # Claim the URL before launching so a concurrent caller sees it
        # as opened instead of launching a second tab.
        now = time.monotonic()
        if now - _opened_at.get(url, -_REOPEN_AFTER) < _REOPEN_AFTER:
            return True
        _opened_at[url] = now
    if sys.platform == "win32" and not os.environ.get("BROWSER", "").strip():  # pragma: no cover
        # Windows has no opener executable; the shell association is
        # only reachable through os.startfile (not testable on Linux).
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            opened = True
        except OSError:
            opened = False
    else:
        opened = any(_launch(command) for command in _launch_commands(url))
    if not opened:
        with _lock:
            _opened_at.pop(url, None)
    return opened


def browser_handoff_note(url: str, opened: bool) -> str:
    """Describe to the agent what happened to *url* and what it must do next.

    Args:
        url: The page the user has to complete.
        opened: The result of :func:`open_in_default_browser`.

    Returns:
        One or two sentences to embed in tool output: whether the page
        is already open in the user's default browser, and that the
        URL must be shown to the user via ask_user_question() anyway.
    """
    if opened:
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
    out of a web console: the console is opened in the user's default
    browser when possible, and the returned sentence tells the agent to
    show the URL regardless.

    Args:
        url: The portal page where the credential is created.

    Returns:
        The :func:`browser_handoff_note` for *url*.
    """
    return browser_handoff_note(url, open_in_default_browser(url))
