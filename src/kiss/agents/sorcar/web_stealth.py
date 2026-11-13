# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Make the KISS browser look like a person's Chrome to bot-protection vendors.

Analysis of the task history (``history.db`` in the brand's home directory,
Sept 2026) found 95 hosts answering the browser tool with a
bot-protection page: Cloudflare managed challenges
(stackoverflow, dl.acm.org, npmjs, readthedocs sites, congress.gov, ...),
Cloudflare hard blocks (medium.com), Akamai "Access Denied" (carmax,
costco), Anubis (dblp, archwiki), Imperva (cato.org), PerimeterX
(walmart) and Google's ``/sorry/`` "unusual traffic" page.  Measured
against 14 of those URLs from the same machine, the stock configuration
(Playwright driver, headless Chromium) got 4 pages; Patchright driving a
headed Chromium on a virtual display got 11.  Re-measured in Oct 2026
(Patchright 1.63, Google Chrome 154 on Xvfb) all 13 hosts load; what
remained was PerimeterX's press-and-hold widget on walmart product pages
and fiverr (now pressed like a person, see :func:`press_hold_secs`),
Anubis proof-of-work pages that needed more than 12 s on a loaded VM
(see :func:`challenge_wait_secs`), and two article pages that merely
*talked* about "press and hold" being mistaken for the widget (the
patterns below now require the block page's own wording).  The hold
itself is what HUMAN's widget wants: whenever the widget's state could
be read, the bar said "HUMAN CHALLENGE COMPLETED, PLEASE WAIT" after
about 4 s of holding, and fiverr then served the real page (three
natural challenges; a challenge forced with a ``PhantomJS`` user agent
re-serves itself, as HUMAN's docs say).  Whether the vendor *accepts* a
solution is decided on its side from the IP's and session's score: in
two sessions on this datacenter IP the challenge stayed after two holds,
and the page is then reported for the user, as a person stuck in the
same loop would be.  By late Oct 2026 fiverr's widget frame spanned a
600 px column with its 310 px button at the left edge, so holds at the
middle of ``#px-captcha`` often missed the button (fiverr blocked after
two holds on 30 Oct); the button itself is pressed now
(:func:`press_and_hold_box`).  On 30 Oct 2026 semanticscholar.org, which
had served paper and search pages all September, began answering them
with AWS WAF's CAPTCHA action (HTTP 405, title "Human Verification",
"Let's confirm you are human", a "Begin" button opening a picture
puzzle; https://docs.aws.amazon.com/waf/latest/developerguide/waf-captcha-and-challenge.html).
An un-automated Google Chrome with a fresh profile gets the same puzzle
from this machine, so it rates the network's IP address, not the
browser: the page is recognised (by its wording, or by its title and
``captcha.awswaf.com`` script for other languages) and reported after a
3 s grace -- nothing is pressed, since the puzzle needs a person -- and the
same paper or search is then looked up on Crossref, the DOI registry's
open API, which served every probe from this network (:func:`content_fallback`;
Semantic Scholar's own API answers 429 to nine of ten unauthenticated
calls and OpenAlex rate-limits anonymous search under load).  AWS WAF's
*Challenge* action (HTTP 202) is a blank interstitial whose
``token.awswaf.com/.../challenge.js`` acquires a token and reloads the
page in about a second; read before that, it was an "(empty page)" three
times in the task history, so a blank page carrying that script is now
waited out like a Cloudflare challenge.  On 3 Nov 2026 a research task
met two more pages that came back as content, unrecognised: Stack
Overflow's own "Human verification" page (``/nocaptcha?s=<id>``, where
``/search?q=`` redirects a visitor it rates a script; a Google reCAPTCHA
v2 box, "Check the CAPTCHA box, and we'll be out of your way") and
Reddit's "You've been blocked by network security" page (HTTP 403 on
every reddit.com host -- www, old.reddit.com's "whoa there, pardner!",
the ``.json`` API -- its network policy on data-centre, cloud and VPN
addresses, lifted only for logged-in accounts).  The reCAPTCHA box is
ticked like Turnstile's; from this machine the tick opens a picture
puzzle at once (:data:`RECAPTCHA_PUZZLE_OPEN_JS` ends the wait there),
on a network Google rates better the page submits itself.  Reddit's
page is static, so it gets no wait; both are reported with the way out
(:func:`block_advice`).  The same task also got a Cloudflare interstitial
back from a ``click`` on a search result: only ``go_to_url`` settled
challenges, so clicks, typed submissions and key presses that navigate
now do too.  Three groups of helpers, all consumed by
:mod:`kiss.agents.sorcar.web_use_tool`:

* **Engine**: :func:`playwright_api` prefers Patchright, a patched
  Playwright driver that does not send the ``Runtime.enable`` CDP command
  every anti-bot script fingerprints (https://github.com/Kaliiiiiiiiii-Vinyzu/patchright);
  :func:`chrome_channel` prefers an installed Google Chrome over the
  bundled Chromium.
* **Virtual display**: :func:`virtual_display` starts one Xvfb per process
  so Chromium runs *headed* on a server without a screen.  Headless mode
  is the single strongest bot signal (benchmark: 100% -> 40% bypass for
  the same engine, https://github.com/techinz/browsers-benchmark).
* **Human input**: :func:`mouse_path` (Bezier pointer paths),
  :func:`typing_chunks` (uneven typing cadence), :func:`hold_tremor`
  (the resting hand on a held button) and :func:`challenge_vendor`
  (recognise a bot-protection page so the tool can wait for it to clear,
  press its widget, and report it plainly), with
  :func:`challenge_wait_secs` / :func:`press_hold_secs` /
  :func:`widget_presses` / :data:`PRESS_AND_HOLD_BOX_JS` /
  :func:`press_and_hold_box` describing how each vendor's page is waited
  on and pressed.
"""

from __future__ import annotations

import atexit
import logging
import os
import random
import re
import selectors
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from importlib import import_module
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse, urlsplit

from kiss.core.locked import locked
from kiss.core.processes import terminate_process_group
from kiss.core.tool_interrupt import result_interruptible

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

_API_MODULE: ModuleType | None = None
_API_LOCK = threading.RLock()


@locked(_API_LOCK)
def playwright_api() -> ModuleType:
    """Return the sync Playwright API module to drive Chromium with.

    Patchright is a drop-in replacement whose driver avoids the
    ``Runtime.enable`` / ``Console.enable`` CDP leaks that Cloudflare,
    Akamai and DataDome fingerprint; it is used whenever it is importable
    and the stock ``playwright`` package is the fallback.

    Task threads call this concurrently; the lock makes the import and
    the dead-driver guard install one critical section, and the module
    is published only once it is guarded, so no caller drives an
    unguarded API and ``SyncBase._sync`` is wrapped exactly once.

    Returns:
        ``patchright.sync_api`` or ``playwright.sync_api``.
    """
    global _API_MODULE
    if _API_MODULE is None:
        try:
            api = import_module("patchright.sync_api")
        except ImportError:  # pragma: no cover — patchright is a declared dependency
            api = import_module("playwright.sync_api")
        _guard_sync_api_against_dead_driver(api)
        _API_MODULE = api
    return _API_MODULE


DRIVER_EXITED_MESSAGE = "Playwright driver exited; the browser must be relaunched"


class _DispatcherGuard:
    """Stand-in for the sync API's dispatcher greenlet during one ``_sync`` call.

    The sync API runs its asyncio loop in a *dispatcher* greenlet and
    waits for a call's task with ``while not task.done(): dispatcher.switch()``.
    When the driver process exits, the loop's main coroutine returns, the
    dispatcher greenlet finishes, and from then on ``switch()`` returns at
    once without running anything: the wait becomes a busy spin that
    never ends (seen in the full-suite run of 2026-10-25 after a driver
    crash, 900 s in ``BrowserContext.close``).  This guard switches like
    the greenlet and raises as soon as it is found dead.  It replaces a
    wrapper object's reference to the greenlet for good (never nested,
    never restored): sync calls on one object can overlap across
    greenlets -- an event handler's call while a navigation is pending
    -- and finish in any order, so a per-call swap would leak.
    """

    __slots__ = ("_fiber",)

    def __init__(self, fiber: object) -> None:
        self._fiber = fiber

    def switch(self, *args: object, **kwargs: object) -> object:
        """Run the dispatcher; raise instead of returning once it has finished."""
        fiber: object = self._fiber
        result = fiber.switch(*args, **kwargs)  # type: ignore[attr-defined]
        if getattr(fiber, "dead", False):
            raise _sync_api_error(DRIVER_EXITED_MESSAGE)
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self._fiber, name)


def _sync_api_error(message: str) -> Exception:
    """The sync API's own ``Error`` (so callers' ``except`` clauses see the usual type)."""
    pkg = playwright_api().__name__.split(".")[0]
    error: Exception = import_module(f"{pkg}._impl._errors").Error(message)
    return error


def _guard_sync_api_against_dead_driver(api: ModuleType) -> None:
    """Make every sync API call fail fast, not spin forever, once its driver has exited.

    Wraps ``SyncBase._sync`` of *api*'s package: the wrapper object's
    dispatcher reference is replaced, once, by a :class:`_DispatcherGuard`
    (``_sync`` and ``EventInfo.value`` reach the greenlet through it), so
    a driver exit noticed mid-call raises, and a call finding the
    greenlet already dead raises at once.  Without this, only the first
    call after the exit fails (the read of the closed pipe fails its
    request) and every later one hangs the calling thread for good,
    which no watchdog on the browser process can unwedge.  Applied once
    per process; the tool then discards the dead driver and relaunches.
    """
    pkg = api.__name__.split(".")[0]
    try:
        sync_base = import_module(f"{pkg}._impl._sync_base").SyncBase
    except (ImportError, AttributeError):  # pragma: no cover — another API layout
        logger.debug("No SyncBase to guard in %s", pkg, exc_info=True)
        return
    if getattr(sync_base, "_kiss_dead_driver_guard", False):
        return
    original = sync_base._sync

    def _sync(self: object, coro: object) -> object:
        fiber = self._dispatcher_fiber  # type: ignore[attr-defined]
        if not isinstance(fiber, _DispatcherGuard):
            fiber = self._dispatcher_fiber = _DispatcherGuard(fiber)  # type: ignore[attr-defined]
        if getattr(fiber, "dead", False):
            close = getattr(coro, "close", None)
            if close is not None:
                close()
            raise _sync_api_error(DRIVER_EXITED_MESSAGE)
        return original(self, coro)

    sync_base._sync = _sync
    sync_base._kiss_dead_driver_guard = True


def playwright_package() -> str:
    """Return the distribution name whose ``install`` CLI fetches browsers.

    Returns:
        ``"patchright"`` or ``"playwright"``, matching :func:`playwright_api`.
    """
    return playwright_api().__name__.split(".")[0]


_CHROME_PATHS = (
    "/opt/google/chrome/chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def chrome_channel() -> str:
    """Return the Playwright ``channel`` to launch: real Chrome when present.

    Google Chrome ships codecs, brand strings and APIs the open-source
    Chromium build lacks, and bot-protection vendors score the
    ``"Chromium"`` brand lower.  The bundled Chromium is the fallback so
    nothing has to be installed by hand.

    Returns:
        ``"chrome"`` if a Google Chrome installation is found, else
        ``"chromium"``.
    """
    if shutil.which("google-chrome") or shutil.which("google-chrome-stable"):
        return "chrome"  # pragma: no cover — needs Google Chrome installed
    if any(os.path.exists(path) for path in _CHROME_PATHS):
        return "chrome"  # pragma: no cover — needs Google Chrome installed
    return "chromium"


def disabled_blink_features(unattended: bool) -> str:
    """Return the ``--disable-blink-features=...`` switch for a Chromium launch.

    ``AutomationControlled`` is always disabled: Chromium forces
    ``navigator.webdriver`` on for any ``--remote-debugging-*`` flag and
    only this switch clears it.  An *unattended* browser (headless, or
    headed on an Xvfb display nobody looks at) also disables ``WebAuth``:
    a security-key or passkey request there opens Chrome's native modal
    dialog, which is neither streamed to the Browser tab nor visible to
    the agent and which swallows every click on the page underneath
    (Duo's "Use your security key" prompt wedged the Browser tab this
    way).  No key can be plugged into such a browser anyway; without the
    API, sign-in pages offer their other methods instead.

    Args:
        unattended: True when no human can see the browser's own window.

    Returns:
        The complete command-line switch.
    """
    features = ["AutomationControlled"]
    if unattended:
        features.append("WebAuth")
    return "--disable-blink-features=" + ",".join(features)


# Headless Chrome's user agent carries ``HeadlessChrome/<version>`` where
# a window carries ``Chrome/<version>``.  Rewriting the token from inside
# the page (an init script redefining ``navigator.userAgent``) is itself
# a tell: ``navigator.appVersion`` keeps the raw token, so the two
# strings disagree, and the redefined property is an *own* property of
# ``navigator`` where a real browser has it on the prototype.  Measured
# in Oct 2026 on this machine (Chrome 154, Patchright 1.63): with that
# script, real headless mode failed every Cloudflare managed challenge
# (dl.acm.org, sciencedirect, patents.justia.com, espacenet: 0 of 4);
# with Chrome's own ``--user-agent`` switch and no script, 4 of 4 — the
# switch sets the header, ``userAgent`` and ``appVersion`` natively and
# leaves the client-hint brands (``Sec-CH-UA``) untouched.  Those
# blocks had hit the macOS daemon, where there is no Xvfb and the
# browser must run in real headless mode.
HEADLESS_UA_TOKEN = "HeadlessChrome"
_HEADED_UA_TOKEN = "Chrome"
# Channel -> headed user agent; single dict operations, no lock needed.
_HEADED_USER_AGENTS: dict[str, str] = {}


def headed_user_agent(user_agent: str) -> str | None:
    """Return *user_agent* as the same browser would report it headed.

    Args:
        user_agent: ``navigator.userAgent`` of a freshly launched browser.

    Returns:
        The string with ``HeadlessChrome`` replaced by ``Chrome``, or
        ``None`` when *user_agent* carries no headless token (a headed
        browser, or one already launched with :func:`user_agent_switch`).
    """
    if HEADLESS_UA_TOKEN not in user_agent:
        return None
    return user_agent.replace(HEADLESS_UA_TOKEN, _HEADED_UA_TOKEN)


def user_agent_switch(channel: str) -> str | None:
    """Return the ``--user-agent=...`` switch remembered for *channel*, if any.

    The headed user agent of a browser build is only known once that
    build has been launched and asked (see :func:`remember_user_agent`);
    this returns it for every later launch in the process so a headless
    browser starts with the right user agent the first time instead of
    being relaunched.

    Args:
        channel: The Playwright channel (:func:`chrome_channel`).

    Returns:
        The complete command-line switch, or ``None`` when no headless
        launch of *channel* has been seen yet in this process.
    """
    user_agent = _HEADED_USER_AGENTS.get(channel)
    return None if user_agent is None else f"--user-agent={user_agent}"


def remember_user_agent(channel: str, user_agent: str) -> None:
    """Record the headed user agent of *channel* for :func:`user_agent_switch`.

    Args:
        channel: The Playwright channel (:func:`chrome_channel`).
        user_agent: The headed string from :func:`headed_user_agent`.
    """
    _HEADED_USER_AGENTS[channel] = user_agent


def forget_user_agent(channel: str) -> None:
    """Drop the user agent remembered for *channel* (the build has changed).

    Args:
        channel: The Playwright channel (:func:`chrome_channel`).
    """
    _HEADED_USER_AGENTS.pop(channel, None)


_CHROME_MAJOR_RE = re.compile(r"\bChrome/(\d+)\.")


def user_agent_matches_version(user_agent: str, version: str) -> bool:
    """Whether *user_agent* names the major version of a browser reporting *version*.

    A daemon keeps running while Chrome updates underneath it; the
    remembered string would then carry the old major version while the
    client hints, which the switch does not touch, carry the new one — a
    mismatch a challenge script can see.

    Args:
        user_agent: A user-agent string or ``--user-agent=`` switch.
        version: ``Browser.version`` such as ``"154.0.8037.57"``.

    Returns:
        True when the ``Chrome/<major>`` token equals *version*'s major.
    """
    match = _CHROME_MAJOR_RE.search(user_agent)
    return match is not None and match.group(1) == version.split(".", 1)[0]


# ---------------------------------------------------------------------------
# Virtual display (Xvfb)
# ---------------------------------------------------------------------------

_XVFB_LOCK = threading.RLock()
_XVFB_PROC: subprocess.Popen[bytes] | None = None
_XVFB_DISPLAY: str | None = None
_XVFB_START_TIMEOUT_SECS = 15.0

# The wrapper keeps Xvfb alive exactly as long as the Python process that
# started it: if Python dies without running atexit (SIGKILL, hard
# crash) the loop notices within two seconds and kills Xvfb, so no
# display server is ever leaked.  ``$1`` is the Python PID, ``$2`` the
# write end of the ``-displayfd`` pipe.
_XVFB_WRAPPER = (
    'Xvfb -displayfd "$2" -screen 0 1920x1080x24 -nolisten tcp -noreset & X=$!; '
    'while kill -0 "$1" 2>/dev/null && kill -0 "$X" 2>/dev/null; do sleep 2; done; '
    'kill "$X" 2>/dev/null'
)


def virtual_display() -> str | None:
    """Return the ``DISPLAY`` of a process-wide Xvfb, starting it on demand.

    Chromium's headless mode is what most bot-protection vendors detect
    first, so on Linux the tool runs a *headed* Chromium on an invisible
    X server instead.  One Xvfb is shared by every browser this process
    launches and is torn down at interpreter exit (or, if the process is
    killed, by the wrapper shell loop within two seconds).

    The browser tools call this on the task thread, where the runner's
    Stop is an exception raised at any check point of the bytecode:
    inside ``Popen`` once the wrapper is forked, or as a call returns
    and before its result is bound, where nothing could reap the
    wrapper or close its pipe (and a plain ``with`` leaks the lock,
    CPython gh-74174).  The display is therefore started and published
    by a thread of its own, which receives no injection
    (:func:`_provide_display`); the caller only waits for it, and a
    start-up whose caller was stopped ends published or terminated.

    Returns:
        A display name such as ``":1"``, or ``None`` when this is not
        Linux or the ``Xvfb`` binary is not installed (the caller then
        falls back to real headless mode).

    Raises:
        ToolCallInterrupted: When the calling thread's tool call was
            interrupted while the display was starting.
    """
    abandoned = threading.Event()
    provided: Future[str | None] = Future()
    try:
        threading.Thread(target=_provide_display, args=(provided, abandoned), daemon=True).start()
        return result_interruptible(provided)
    finally:
        abandoned.set()


def _provide_display(provided: Future[str | None], abandoned: threading.Event) -> None:
    """Thread body of :func:`virtual_display`: resolve *provided* with the shared display.

    *provided* is resolved with the lock released: a caller stopped
    inside ``Future.result`` may have leaked the future's own lock, and
    a thread parked on that one must not hold the display's.

    Args:
        provided: Receives the display, ``None``, or the error raised.
        abandoned: Set once the caller stopped waiting.
    """
    try:
        with _XVFB_LOCK:
            display = _shared_display(abandoned)
    except BaseException as error:  # the caller's to report
        provided.set_exception(error)
    else:
        provided.set_result(display)


def _shared_display(abandoned: threading.Event) -> str | None:
    """Return the display of the running Xvfb, starting one when none runs.

    Called with ``_XVFB_LOCK`` held, on the thread of :func:`_provide_display`.

    Args:
        abandoned: Set once the caller of :func:`virtual_display` stopped
            waiting: no Xvfb is started for it, and one that has not
            reported its display yet is given up.

    Returns:
        What :func:`virtual_display` returns.
    """
    global _XVFB_PROC, _XVFB_DISPLAY
    if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
        return _XVFB_DISPLAY
    _XVFB_PROC, _XVFB_DISPLAY = None, None
    if abandoned.is_set() or not sys.platform.startswith("linux") or shutil.which("Xvfb") is None:
        return None
    read_fd = write_fd = -1
    proc: subprocess.Popen[bytes] | None = None
    display = None
    try:
        try:
            read_fd, write_fd = os.pipe()
            proc = subprocess.Popen(
                ["sh", "-c", _XVFB_WRAPPER, "xvfb", str(os.getpid()), str(write_fd)],
                pass_fds=(write_fd,),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Forgotten before it is closed: after an error at the close, the
            # ``finally`` must not close a number that may be another thread's by then.
            fd, write_fd = write_fd, -1
            os.close(fd)
            display = _read_display_number(read_fd, proc, abandoned)
        except OSError:
            # Descriptor or process exhaustion (the pipe, Popen's own pipe or
            # the selector): browse headless rather than fail the whole tool.
            logger.warning("could not start Xvfb; falling back to headless", exc_info=True)
            return None
        if display is not None:
            _XVFB_DISPLAY = display
            _XVFB_PROC = proc
    finally:
        # Both pipe ends are ours to close whether or not Popen raised.
        if write_fd >= 0:
            os.close(write_fd)
        if read_fd >= 0:
            os.close(read_fd)
        if proc is not None and _XVFB_PROC is not proc and proc.poll() is None:
            # Timed out, abandoned or failed: the wrapper (and its Xvfb)
            # must not outlive this call unowned, since ``_XVFB_PROC`` is
            # still ``None`` and nothing could reap it.  A wrapper that
            # ``_read_display_number`` already saw exit is reaped and its
            # group gone (the wrapper kills Xvfb before it exits), so its
            # pgid may belong to someone else by now: no signal.
            terminate_process_group(proc)
    if display is not None:
        logger.info("virtual display %s started for headed Chromium", display)
    elif not abandoned.is_set():  # else nobody is left to fall back
        logger.warning("Xvfb did not start; falling back to headless Chromium")
    return display


def _read_display_number(
    read_fd: int, proc: subprocess.Popen[bytes], abandoned: threading.Event
) -> str | None:
    """Read the display number Xvfb writes once it accepts connections.

    The wait uses the platform's readiness API, not ``select``, whose
    ``FD_SETSIZE`` limit rejects a pipe numbered 1024 or higher (a
    daemon with that many sockets and files open).

    Args:
        read_fd: Read end of the ``-displayfd`` pipe.
        proc: The wrapper process, polled so a crashed Xvfb is noticed.
        abandoned: Set once the caller of :func:`virtual_display`
            stopped waiting: the start-up is given up.

    Returns:
        ``":N"`` or ``None`` on timeout / early exit / *abandoned*.
    """
    deadline = time.monotonic() + _XVFB_START_TIMEOUT_SECS
    data = b""
    with selectors.DefaultSelector() as ready:
        ready.register(read_fd, selectors.EVENT_READ)
        while not data.endswith(b"\n"):
            if abandoned.is_set():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0 or proc.poll() is not None:  # pragma: no cover — timing
                return None
            if ready.select(min(remaining, 0.5)):
                chunk = os.read(read_fd, 16)
                if not chunk:
                    return None
                data += chunk
    return ":" + data.decode().strip()


def stop_virtual_display() -> None:
    """Stop the process-wide Xvfb, if one is running.

    Registered once with :mod:`atexit` at import; also callable directly so
    a test or a long-lived daemon can release the X server early.  The
    next :func:`virtual_display` call starts a fresh one.
    """
    global _XVFB_PROC, _XVFB_DISPLAY
    with _XVFB_LOCK:
        proc, _XVFB_PROC, _XVFB_DISPLAY = _XVFB_PROC, None, None
    if proc is not None:
        terminate_process_group(proc)


atexit.register(stop_virtual_display)


# ---------------------------------------------------------------------------
# Human-like input
# ---------------------------------------------------------------------------


def mouse_path(start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
    """Return waypoints of a curved, decelerating pointer move from *start* to *end*.

    Real pointer traces are arcs that start fast and settle onto the
    target, not straight teleports; behavioural detectors (Cloudflare,
    PerimeterX, DataDome) score straight-line or instant moves as
    automation.  The path is a cubic Bezier whose control points are
    pushed sideways by a random fraction of the distance, sampled with an
    ease-out so points crowd near the target, plus sub-pixel jitter.

    Args:
        start: Current pointer position (viewport CSS pixels).
        end: Target position.

    Returns:
        Waypoints excluding *start*, ending exactly on *end*.
    """
    (x0, y0), (x3, y3) = start, end
    dx, dy = x3 - x0, y3 - y0
    distance = (dx * dx + dy * dy) ** 0.5
    if distance < 1.0:
        return [end]
    # Perpendicular unit vector: the curve bows to one side of the chord.
    px, py = -dy / distance, dx / distance
    bow = distance * random.uniform(0.08, 0.28) * random.choice((-1, 1))
    x1 = x0 + dx * random.uniform(0.2, 0.4) + px * bow
    y1 = y0 + dy * random.uniform(0.2, 0.4) + py * bow
    x2 = x0 + dx * random.uniform(0.6, 0.8) + px * bow * random.uniform(0.2, 0.7)
    y2 = y0 + dy * random.uniform(0.6, 0.8) + py * bow * random.uniform(0.2, 0.7)
    count = max(6, min(24, int(distance / 40)))
    points: list[tuple[float, float]] = []
    for i in range(1, count + 1):
        t = i / count
        t = 1 - (1 - t) ** 2.2  # ease-out: dense samples near the target
        u = 1 - t
        bx = u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3
        by = u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3
        if i < count:
            bx += random.uniform(-0.8, 0.8)
            by += random.uniform(-0.8, 0.8)
        points.append((bx, by))
    points[-1] = end
    return points


def typing_chunks(text: str) -> list[tuple[str, int]]:
    """Split *text* into runs typed at their own cadence.

    ``keyboard.type(text, delay=50)`` emits keystrokes on a metronome,
    which is a well-known automation tell.  People type in bursts: each
    word at a slightly different speed, with a longer pause at every
    space.

    Args:
        text: The string to type.

    Returns:
        ``(chunk, per_key_delay_ms)`` pairs whose chunks concatenate to
        *text*; a chunk that is a single space carries the inter-word
        pause as its delay.
    """
    chunks: list[tuple[str, int]] = []
    for piece in re.split(r"( )", text):
        if piece == " ":
            chunks.append((piece, random.randint(90, 260)))
        elif piece:
            chunks.append((piece, random.randint(35, 115)))
    return chunks


def typing_duration_secs(chunks: list[tuple[str, int]]) -> float:
    """Return the wall-clock time the chunks from :func:`typing_chunks` need.

    Args:
        chunks: Output of :func:`typing_chunks`.

    Returns:
        Seconds; used to scale the raw-input watchdog deadline.
    """
    return sum(len(chunk) * delay for chunk, delay in chunks) / 1000.0


# ---------------------------------------------------------------------------
# Bot-protection page recognition
# ---------------------------------------------------------------------------

# Interstitials are recognised by their *title* (a page's own title is a
# strong signal: no article is titled "Just a moment...") or by two
# co-occurring body phrases.  Standalone prose such as the word "captcha"
# or "You don't have permission" appears in ordinary articles and must not
# trigger the challenge wait.
CLOUDFLARE_CHALLENGE = "Cloudflare challenge"
ANUBIS_CHECK = "Anubis proof-of-work check"
PERIMETERX_CHECK = "PerimeterX press-and-hold check"
# AWS WAF's two interstitials (https://docs.aws.amazon.com/waf/latest/developerguide/waf-captcha-and-challenge-actions.html):
# the Challenge action's blank page (HTTP 202) that acquires a token in
# the browser and reloads on its own, and the CAPTCHA action's picture
# puzzle (HTTP 405), which only a person solves.
AWS_WAF_CHALLENGE = "AWS WAF challenge"
AWS_WAF_CAPTCHA = "AWS WAF CAPTCHA puzzle"
# Stack Overflow's own "Human verification" page (``/nocaptcha?s=<id>``,
# the search endpoint redirects to it for a visitor it rates a script):
# a Google reCAPTCHA v2 box, "Check the CAPTCHA box, and we'll be out of
# your way".  Reddit's two block pages (HTTP 403, every reddit.com host:
# www, old.reddit.com, the ``.json`` API), its network policy on data
# centre, cloud and VPN addresses for visitors who are not logged in.
STACK_OVERFLOW_CAPTCHA = "Stack Overflow 'Human verification' CAPTCHA"
REDDIT_BLOCK = "Reddit network-policy block"

_TITLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Cloudflare block", re.compile(r"^Attention Required! \| Cloudflare$", re.I)),
    (CLOUDFLARE_CHALLENGE, re.compile(r"^Just a moment\.\.\.$", re.I)),
    (ANUBIS_CHECK, re.compile(r"^Making sure you're not a bot!?$", re.I)),
    ("Akamai block", re.compile(r"^Access Denied$", re.I)),
    # walmart.com/blocked, fiverr, and PerimeterX's stock block page.
    (
        PERIMETERX_CHECK,
        re.compile(
            r"^Robot or human\?$|^It needs a human touch$|^Access to this page has been denied\.?$",
            re.I,
        ),
    ),
    ("CAPTCHA", re.compile(r"^Are you a robot\?$|^Security check required$", re.I)),
)
# (vendor label, both patterns must match the body head)
_BODY_PATTERNS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        "Google 'unusual traffic' page",
        re.compile(r"unusual traffic from your computer network", re.I),
        re.compile(r"IP address:", re.I),
    ),
    (
        "Cloudflare block",
        re.compile(r"Sorry, you have been blocked", re.I),
        re.compile(r"Ray ID|Cloudflare", re.I),
    ),
    (
        CLOUDFLARE_CHALLENGE,
        re.compile(
            r"Performing security verification|Verify(?:ing)? you are human|"
            r"Enable JavaScript and cookies to continue",
            re.I,
        ),
        re.compile(r"Ray ID|Cloudflare", re.I),
    ),
    (
        ANUBIS_CHECK,
        re.compile(r"Sad Anubis|Oh noes!", re.I),
        re.compile(r"Anubis", re.I),
    ),
    (
        "Akamai block",
        re.compile(r"You don't have permission to access", re.I),
        re.compile(r"errors\.edgesuite\.net|Reference #", re.I),
    ),
    (
        "Imperva Incapsula block",
        re.compile(r"Request unsuccessful", re.I),
        re.compile(r"Incapsula incident ID", re.I),
    ),
    (
        # The widget's label or the page's heading, plus the block page's
        # own sentence.  Articles about the widget ("a simple press and
        # hold challenge", "Please verify you are Human: Press & Hold")
        # quote the label but not the sentence.
        PERIMETERX_CHECK,
        re.compile(r"Press & Hold|Activate and hold the button|Robot or human\?", re.I),
        re.compile(
            r"confirm that you.re human|Please verify you are a human|"
            r"using automation tools to browse|Reference ID",
            re.I,
        ),
    ),
    (
        # The stock puzzle page's heading and its own sentence (title
        # "Human Verification", too generic on its own), as semanticscholar
        # served it on 30 Oct 2026.
        AWS_WAF_CAPTCHA,
        re.compile(r"Let.s confirm you are human", re.I),
        re.compile(
            r"Complete the security check before continuing|verifies that you are not a bot",
            re.I,
        ),
    ),
)
# (vendor label, pattern the whole title must match, pattern the body
# head must match): pages whose title alone is a question's or an
# article's (a Stack Overflow question can be titled "Human verification")
# and whose wording alone is quoted by articles and meta posts about
# the page.  Reddit's block pages carry nothing but their sentence, so
# the body must *start* with it, which an article's never does.
_PAGE_PATTERNS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        STACK_OVERFLOW_CAPTCHA,
        re.compile(
            r"^Human verification - (?:Stack Overflow.*|.*Stack Exchange|Super User|"
            r"Ask Ubuntu|Server Fault|MathOverflow)$",
            re.I,
        ),
        re.compile(r"Check the CAPTCHA box", re.I),
    ),
    (
        REDDIT_BLOCK,
        re.compile(r"^$"),
        re.compile(r"^\s*You.ve been blocked by network security\b", re.I),
    ),
    (
        REDDIT_BLOCK,
        re.compile(r"^Blocked$", re.I),
        re.compile(r"^\s*whoa there, pardner!", re.I),
    ),
)
# AWS WAF's interstitials carry no vendor name in their text: the
# Challenge action's page has an empty title and no text at all, only the
# token script, and the CAPTCHA page's wording is rendered by its puzzle
# script in the browser's language under the fixed title "Human
# Verification".  Both scripts are also embedded by sites that use the
# AWS WAF SDK on their own pages
# (https://docs.aws.amazon.com/waf/latest/developerguide/waf-js-challenge-api.html),
# which have their own titles and text -- hence the blank-page and title
# requirements in :func:`challenge_vendor`.
_AWS_WAF_SCRIPTS = frozenset({"challenge.js", "captcha.js"})
_AWS_WAF_CAPTCHA_TITLE = "human verification"


def _aws_waf_scripts(scripts: str) -> set[str]:
    """The AWS WAF interstitial scripts among *scripts*, by file name.

    A script counts when served over HTTPS from a host under
    ``awswaf.com`` (the vendor's own, not ``awswaf.com.evil.example``)
    and named ``challenge.js`` or ``captcha.js`` (not ``captcha.js.map``,
    nor a URL that merely mentions one in its query string).  Chromium
    keeps a malformed ``src`` as written; one that does not parse (a bad
    port, an unclosed ``[``) is skipped, not raised on.

    Args:
        scripts: Space-separated script URLs.

    Returns:
        A subset of ``{"challenge.js", "captcha.js"}``.
    """
    seen: set[str] = set()
    for url in scripts.split():
        try:
            # ``urlsplit`` keeps ``;params`` in the path, so the file name
            # is exact; reading ``port`` validates it.
            parts = urlsplit(url)
            host = parts.hostname or ""
            parts.port  # noqa: B018 — raises ValueError on a bad port
        except ValueError:
            continue
        if parts.scheme == "https" and (host == "awswaf.com" or host.endswith(".awswaf.com")):
            name = parts.path.rsplit("/", 1)[-1]
            if name in _AWS_WAF_SCRIPTS:
                seen.add(name)
    return seen


def challenge_vendor(title: str, body_head: str, scripts: str = "") -> str | None:
    """Recognise a bot-protection interstitial from what the page shows.

    Args:
        title: ``document.title``.
        body_head: The first ~1500 characters of ``document.body.innerText``.
        scripts: The ``src`` URLs of the page's ``<script>`` elements,
            space-separated (empty when the caller has not read them).

    Returns:
        A short vendor label (``"Cloudflare challenge"``, ``"Akamai
        block"``, ...) or ``None`` when the page looks like content.
    """
    stripped = title.strip()
    for label, pattern in _TITLE_PATTERNS:
        if pattern.search(stripped):
            return label
    for label, first, second in _BODY_PATTERNS:
        if first.search(body_head) and second.search(body_head):
            return label
    for label, title_pattern, body_pattern in _PAGE_PATTERNS:
        if title_pattern.search(stripped) and body_pattern.search(body_head):
            return label
    aws = _aws_waf_scripts(scripts)
    if "captcha.js" in aws and stripped.lower() == _AWS_WAF_CAPTCHA_TITLE:
        return AWS_WAF_CAPTCHA
    if "challenge.js" in aws and not stripped and not body_head.strip():
        return AWS_WAF_CHALLENGE
    return None


# How long a vendor's page is given to clear on its own.  Cloudflare's
# managed challenge takes 3-8 s in a browser it rates human.  Anubis makes
# the browser find a SHA-256 hash with (by default) five leading zero
# nibbles, about a million hashes in a Web Worker
# (https://anubis.techaro.lol/docs/design/how-anubis-works), which took
# longer than 12 s on this loaded VM six times in the task history.
_CHALLENGE_WAIT_SECS = 12.0
_ANUBIS_WAIT_SECS = 45.0
# AWS WAF's CAPTCHA page renders its "Begin" button only after its token
# script has run, so once its wording shows nothing is pending: the
# puzzle stays until a person solves it.  A short grace for a page in
# the middle of reloading itself, then it is reported.
_PUZZLE_WAIT_SECS = 3.0


def challenge_wait_secs(vendor: str) -> float:
    """How long *vendor*'s interstitial is given to clear before it is reported.

    Args:
        vendor: A label returned by :func:`challenge_vendor`.

    Returns:
        Seconds; longer for Anubis, whose proof-of-work is CPU-bound,
        shorter for AWS WAF's puzzle, which no wait clears, none for
        Reddit's static 403 page.
    """
    if vendor == ANUBIS_CHECK:
        return _ANUBIS_WAIT_SECS
    if vendor == AWS_WAF_CAPTCHA:
        return _PUZZLE_WAIT_SECS
    if vendor == REDDIT_BLOCK:
        return 0.0
    return _CHALLENGE_WAIT_SECS


def press_hold_secs(vendor: str) -> float:
    """How long the button stays down when *vendor*'s widget is pressed.

    Cloudflare's Turnstile box is a click (a person's 60-140 ms).  HUMAN's
    "Press & Hold" bar fills over several seconds and the hold must
    outlast it (https://docs.humansecurity.com/applications/human-challenge:
    "clicking / tapping on it until the bar is full and then release
    it"); 8 s cleared fiverr's widget from this machine.

    Args:
        vendor: A label returned by :func:`challenge_vendor`.

    Returns:
        Seconds, drawn afresh on each call.
    """
    if vendor == PERIMETERX_CHECK:
        return random.uniform(8.0, 10.0)
    return random.uniform(0.06, 0.14)


def widget_presses(vendor: str) -> int:
    """How many times *vendor*'s widget is pressed before the page is reported.

    Turnstile's box is ticked once: a second tick on a box that has not
    cleared is what a person does not do (and unticks it).  So is
    reCAPTCHA's "I'm not a robot" box on Stack Overflow's page: ticked,
    it either passes (the page submits itself) or opens a picture puzzle
    (:data:`RECAPTCHA_PUZZLE_OPEN_JS`).  HUMAN's two-step variant asks to
    "press again when prompted".

    Args:
        vendor: A label returned by :func:`challenge_vendor`.

    Returns:
        2 for the press-and-hold bar, 1 for the Turnstile and reCAPTCHA
        boxes, 0 for pages with nothing to press.
    """
    if vendor == PERIMETERX_CHECK:
        return 2
    return 1 if vendor in (CLOUDFLARE_CHALLENGE, STACK_OVERFLOW_CAPTCHA) else 0


# Whether reCAPTCHA v2 has opened its picture puzzle: the challenge
# iframe (``/recaptcha/api2/bframe``, titled "recaptcha challenge expires
# in two minutes") is on the page from the start inside a wrapper kept
# ``visibility: hidden`` until the box's verdict calls for a puzzle, when
# the wrapper is shown.  Read from the main frame, whatever the puzzle's
# own markup; once it is open no wait clears the page (only a person
# solves it), so the wait ends there.  Answers a one-element list, as
# the tool's bounded page read wants (it polls until the value is truthy).
RECAPTCHA_PUZZLE_OPEN_JS = """() => {
  const frame = document.querySelector('iframe[src*="/recaptcha/api2/bframe"]');
  if (!frame) return [false];
  const box = frame.getBoundingClientRect();
  return [getComputedStyle(frame).visibility === 'visible' && box.width > 0 && box.height > 0];
}"""

_SHOW_BROWSER_ADVICE = (
    "Call show_browser() so the user can complete the check, or use another source "
    "for the same information."
)
_BLOCK_ADVICE: dict[str, str] = {
    REDDIT_BLOCK: (
        "Reddit refuses this network's IP address for visitors who are not logged in, "
        "on every reddit.com host (old.reddit.com and the .json API included), which no "
        "browser setting changes. Call show_browser() so the user can log in to Reddit, "
        "or use another source for the same information."
    ),
    STACK_OVERFLOW_CAPTCHA: (
        "Its reCAPTCHA box opens a picture puzzle for this network's IP address, which "
        "only a person solves. Call show_browser() so the user can solve it, or search "
        "the web with site:stackoverflow.com for the same questions."
    ),
}


def block_advice(vendor: str) -> str:
    """The way out that the ``Note:`` about *vendor*'s page suggests to the agent.

    Args:
        vendor: A label returned by :func:`challenge_vendor`.

    Returns:
        One or two sentences: why the page will not clear and what to do
        instead, for the pages that rate the network's address (Reddit,
        Stack Overflow's reCAPTCHA); the generic ``show_browser()`` advice
        otherwise.
    """
    return _BLOCK_ADVICE.get(vendor, _SHOW_BROWSER_ADVICE)


def hold_tremor(x: float, y: float) -> tuple[float, float]:
    """Where a resting finger has drifted to while holding a button at (*x*, *y*).

    A pointer that stays on the exact same pixel for eight seconds is a
    script; a hand drifts by a pixel or so.

    Args:
        x: Pressed x in viewport pixels.
        y: Pressed y in viewport pixels.

    Returns:
        A point within about 1.5 px of the pressed one.
    """
    return x + random.uniform(-1.5, 1.5), y + random.uniform(-1.5, 1.5)


# The box of HUMAN / PerimeterX's press-and-hold widget, in main-frame
# viewport pixels, as a one-element list, or an empty list (a list because
# the tool's bounded page read waits for a truthy result).  The widget is
# the ``#px-captcha`` element of the block page (walmart.com/blocked,
# fiverr's "It needs a human touch", PerimeterX's stock page); the button
# to press is inside it (:func:`press_and_hold_box`).  Content pages of
# those sites also carry an empty ``#px-captcha`` placeholder, which is why
# the page text decides that a check is showing and this only locates the
# widget.
# Smallest box taken for the bar: content pages' empty ``#px-captcha``
# placeholders and collapsed frames are narrower or shorter.
_MIN_BAR_WIDTH = 40.0
_MIN_BAR_HEIGHT = 20.0
PRESS_AND_HOLD_BOX_JS = f"""() => {{
  const el = document.querySelector('#px-captcha');
  if (!el) return [];
  const r = el.getBoundingClientRect();
  if (r.width < {_MIN_BAR_WIDTH:g} || r.height < {_MIN_BAR_HEIGHT:g}) return [];
  return [{{x: r.x, y: r.y, width: r.width, height: r.height}}];
}}"""

# Buttons looked at in the widget's visible frame (the real one has one).
_MAX_BAR_BUTTONS = 4


def _usable_box(box: dict[str, float], within: dict[str, float]) -> bool:
    """Whether *box* is big enough to be the bar and centred inside *within*.

    Args:
        box: A rendered element's bounding box in viewport pixels.
        within: The box that must contain *box*'s centre.

    Returns:
        ``True`` when a person could see and press *box*.
    """
    if box["width"] < _MIN_BAR_WIDTH or box["height"] < _MIN_BAR_HEIGHT:
        return False
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    return (
        within["x"] <= cx <= within["x"] + within["width"]
        and within["y"] <= cy <= within["y"] + within["height"]
    )


def press_and_hold_box(page: Any, widget: dict[str, float]) -> dict[str, float] | None:
    """Return the box of the "Press & Hold" button inside HUMAN's widget.

    HUMAN renders the button in an iframe of ``#px-captcha``.  On fiverr
    (Oct 2026) that iframe sits in the element's closed shadow root next
    to four ``display: none`` copies (the "multiple iframes" honeypots of
    https://docs.humansecurity.com/applications/human-challenge) and spans
    the whole element, a 600 px block, while its button is 310 px at the
    left edge: holds at the middle of *widget* landed right of the button
    on empty frame, never reached the bar, and the page stayed blocked.
    The page's frame tree also lists frames in closed shadow roots, so the
    widget's visible frame is found there (hidden copies have no box) and
    the button in it by its ARIA role.

    Args:
        page: The Playwright page showing the check.
        widget: The ``#px-captcha`` element's box
            (:data:`PRESS_AND_HOLD_BOX_JS`), in main-frame viewport pixels.

    Returns:
        The button's box in main-frame viewport pixels, or the visible
        frame's box when no button in it can be pressed, or ``None`` while
        no frame of the widget shows (still rendering, or the page is
        reloading under the read).
    """
    error = playwright_api().Error
    for frame in page.frames:
        if frame.parent_frame is None:
            continue
        try:
            owner = frame.frame_element()
            try:
                box: dict[str, float] | None = owner.bounding_box()
            finally:
                owner.dispose()  # the widget is polled: a kept handle per frame adds up
            if box is None or not _usable_box(box, widget):
                continue  # a hidden copy, a collapsed frame, or one elsewhere
            buttons = frame.get_by_role("button")
            for i in range(min(buttons.count(), _MAX_BAR_BUTTONS)):
                button: dict[str, float] | None = buttons.nth(i).bounding_box(timeout=1500)
                if button is not None and _usable_box(button, box):
                    return button
        except error:
            # The frame detached or navigated under the read.
            logger.debug("press-and-hold frame unreadable", exc_info=True)
            continue
        return box
    return None


_GOOGLE_SEARCH_HOST_RE = re.compile(r"^(www\.)?google\.[a-z.]+$", re.I)


def search_fallback_url(url: str) -> str | None:
    """Return a Bing URL for the same query when *url* is a Google search.

    Google answers search traffic from cloud / VPN address space with its
    ``/sorry/`` "unusual traffic" page regardless of how human the
    browser looks (https://support.google.com/websearch/answer/86640),
    and from a cloud VM it usually shows no solvable CAPTCHA at all.
    Bing served 72 of 73 searches in the task history from this network,
    so a blocked Google query is retried there.

    Args:
        url: The URL that was navigated to.

    Returns:
        ``https://www.bing.com/search?q=...`` or ``None`` when *url* is
        not a Google search with a query.
    """
    parsed = urlparse(url)
    if not _GOOGLE_SEARCH_HOST_RE.match(parsed.netloc):
        return None
    query = parse_qs(parsed.query)
    if parsed.path.startswith("/sorry/"):
        # /sorry/index?continue=<original search url>&q=<opaque token>:
        # the search terms live in the ``continue`` URL, not in ``q``.
        parsed = urlparse((query.get("continue") or [""])[0])
        query = parse_qs(parsed.query)
    if parsed.path != "/search":
        return None
    terms = query.get("q") or []
    if not terms:
        return None
    return "https://www.bing.com/search?q=" + quote_plus(terms[0])


_SEMANTIC_SCHOLAR_HOST_RE = re.compile(r"^(www\.)?semanticscholar\.org$", re.I)
# /paper/<title words and first authors, hyphenated>/<40-hex paper id>; the
# id alone (no slug) names no words to search for.
_SEMANTIC_SCHOLAR_PAPER_RE = re.compile(r"^/paper/(?:([^/]+)/)?[0-9a-f]{40}/?$", re.I)
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_CROSSREF_FIELDS = "DOI,title,author,container-title,issued,abstract,is-referenced-by-count,URL"


def content_fallback(url: str) -> tuple[str, str] | None:
    """Return a Crossref query answering a semanticscholar.org page the site's CAPTCHA keeps back.

    Since 30 Oct 2026 semanticscholar.org answers paper and search pages
    from this network's address space with AWS WAF's CAPTCHA puzzle, which
    an un-automated Chrome gets too, so no browser setting clears it.
    Crossref's public works API (https://api.crossref.org/swagger-ui/index.html,
    no key, no CAPTCHA) finds the same paper from the words of the page's
    URL slug -- ``query.bibliographic`` matches title words and author
    names together -- and answers a search query with its own ranked list;
    it served every probe from this network (Semantic Scholar's own API
    answers 429 to most unauthenticated calls, OpenAlex rate-limits
    anonymous search under load).

    Args:
        url: The URL that was navigated to.

    Returns:
        ``(crossref_url, terms)`` -- the API URL to open and the words
        searched for -- or ``None`` when *url* is not a semanticscholar.org
        search with a query or paper page with a title slug.
    """
    parsed = urlparse(url)
    # ``hostname``: lower-cased, without port or userinfo (so an authority of
    # ``semanticscholar.org@evil.example`` is evil.example).
    if not _SEMANTIC_SCHOLAR_HOST_RE.match(parsed.hostname or ""):
        return None
    if parsed.path.rstrip("/") == "/search":
        terms = (parse_qs(parsed.query).get("q") or [""])[0]
        rows = 10
    else:
        match = _SEMANTIC_SCHOLAR_PAPER_RE.match(parsed.path)
        if match is None or not match.group(1):
            return None
        terms = unquote(match.group(1)).replace("-", " ")
        rows = 5
    terms = " ".join(terms.split())
    if not terms:
        return None
    return (
        f"{CROSSREF_WORKS_URL}?query.bibliographic={quote_plus(terms)}"
        f"&rows={rows}&select={_CROSSREF_FIELDS}",
        terms,
    )


# Turns the Crossref JSON the browser shows as text into a readable page:
# one article per work (title, authors, year, venue, DOI link, citation
# count, abstract when deposited) under a heading naming the terms, with
# the document title set so the agent and the bench can tell the page.
# Evaluated with the search terms as its argument; returns the number of
# works shown, or -1 when the page is not Crossref's JSON.
CROSSREF_RENDER_JS = """(terms) => {
  const pre = document.querySelector("pre");
  let data;
  try {
    data = JSON.parse((pre ? pre.textContent : document.body.innerText) || "");
  } catch (e) {
    return -1;
  }
  const items = data && data.message && Array.isArray(data.message.items)
    ? data.message.items : null;
  if (items === null) { return -1; }
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g,
    (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const first = (v) => Array.isArray(v) ? (v.length ? v[0] : "") : (v == null ? "" : v);
  const articles = items.map((w) => {
    const authors = (w.author || []).map((a) =>
      [a.given, a.family].filter(Boolean).join(" ") || a.name || "").filter(Boolean).join(", ");
    const parts = w.issued && w.issued["date-parts"] && w.issued["date-parts"][0];
    const year = parts && parts[0] ? parts[0] : "";
    const venue = first(w["container-title"]);
    const abstract = String(w.abstract || "").replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ").trim();
    const doi = w.DOI ? "https://doi.org/" + w.DOI : "";
    return "<article><h2>" + esc(first(w.title)) + "</h2>"
      + "<p>" + esc(authors) + (year ? " (" + esc(year) + ")" : "")
      + (venue ? ". " + esc(venue) : "") + "</p>"
      + (doi ? '<p>DOI: <a href="' + esc(doi) + '">' + esc(doi) + "</a></p>" : "")
      + "<p>Cited by " + esc(w["is-referenced-by-count"] || 0) + " (Crossref)</p>"
      + (abstract ? "<p>" + esc(abstract) + "</p>" : "")
      + "</article>";
  });
  document.title = "Crossref: " + terms;
  const heading = items.length + " works on Crossref for \\"" + terms + "\\"";
  document.body.innerHTML = "<h1>" + esc(heading) + "</h1>" + articles.join("");
  return items.length;
}"""
