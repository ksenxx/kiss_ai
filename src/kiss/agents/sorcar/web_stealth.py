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
same loop would be.  Three groups of helpers, all consumed by
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
  :func:`widget_presses` / :data:`PRESS_AND_HOLD_BOX_JS` describing how
  each vendor's page is waited on and pressed.
"""

from __future__ import annotations

import atexit
import logging
import os
import random
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from importlib import import_module
from types import ModuleType
from urllib.parse import parse_qs, quote_plus, urlparse

from kiss.agents.sorcar._concurrency import locked
from kiss.core.processes import terminate_process_group

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
    return match is not None and match.group(1) == version.split(".")[0]


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


@locked(_XVFB_LOCK)
def virtual_display() -> str | None:
    """Return the ``DISPLAY`` of a process-wide Xvfb, starting it on demand.

    Chromium's headless mode is what most bot-protection vendors detect
    first, so on Linux the tool runs a *headed* Chromium on an invisible
    X server instead.  One Xvfb is shared by every browser this process
    launches and is torn down at interpreter exit (or, if the process is
    killed, by the wrapper shell loop within two seconds).

    The lock is taken through :func:`locked`: the browser tools call
    this on the task thread, where the runner's Stop injection would
    leak a plain ``with`` (CPython gh-74174) and hang every later
    browser launch of the daemon.

    Returns:
        A display name such as ``":1"``, or ``None`` when this is not
        Linux or the ``Xvfb`` binary is not installed (the caller then
        falls back to real headless mode).
    """
    global _XVFB_PROC, _XVFB_DISPLAY
    if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
        return _XVFB_DISPLAY
    _XVFB_PROC, _XVFB_DISPLAY = None, None
    if not sys.platform.startswith("linux") or shutil.which("Xvfb") is None:
        return None
    read_fd, write_fd = os.pipe()
    proc: subprocess.Popen[bytes] | None = None
    display = None
    try:
        try:
            proc = subprocess.Popen(
                ["sh", "-c", _XVFB_WRAPPER, "xvfb", str(os.getpid()), str(write_fd)],
                pass_fds=(write_fd,),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            # Descriptor or process exhaustion: browse headless rather
            # than fail the whole tool.
            logger.warning("could not start Xvfb; falling back to headless", exc_info=True)
            return None
        os.close(write_fd)
        write_fd = -1
        display = _read_display_number(read_fd, proc)
    finally:
        # Both pipe ends are ours to close whether or not Popen raised.
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)
        if proc is not None and display is None and proc.poll() is None:
            # Timed out, or a Stop injected into the task thread while
            # the display number was awaited: the wrapper (and its Xvfb)
            # must not outlive this call unowned, since ``_XVFB_PROC`` is
            # still ``None`` and nothing could reap it.  A wrapper that
            # ``_read_display_number`` already saw exit is reaped and its
            # group gone (the wrapper kills Xvfb before it exits), so its
            # pgid may belong to someone else by now: no signal.
            terminate_process_group(proc)
    if display is None:
        logger.warning("Xvfb did not start; falling back to headless Chromium")
        return None
    _XVFB_PROC, _XVFB_DISPLAY = proc, display
    logger.info("virtual display %s started for headed Chromium", display)
    return display


def _read_display_number(read_fd: int, proc: subprocess.Popen[bytes]) -> str | None:
    """Read the display number Xvfb writes once it accepts connections.

    Args:
        read_fd: Read end of the ``-displayfd`` pipe.
        proc: The wrapper process, polled so a crashed Xvfb is noticed.

    Returns:
        ``":N"`` or ``None`` on timeout / early exit.
    """
    deadline = time.monotonic() + _XVFB_START_TIMEOUT_SECS
    data = b""
    while not data.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or proc.poll() is not None:  # pragma: no cover — timing
            return None
        ready, _, _ = select.select([read_fd], [], [], min(remaining, 0.5))
        if ready:
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
)


def challenge_vendor(title: str, body_head: str) -> str | None:
    """Recognise a bot-protection interstitial from what the page shows.

    Args:
        title: ``document.title``.
        body_head: The first ~1500 characters of ``document.body.innerText``.

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
    return None


# How long a vendor's page is given to clear on its own.  Cloudflare's
# managed challenge takes 3-8 s in a browser it rates human.  Anubis makes
# the browser find a SHA-256 hash with (by default) five leading zero
# nibbles, about a million hashes in a Web Worker
# (https://anubis.techaro.lol/docs/design/how-anubis-works), which took
# longer than 12 s on this loaded VM six times in the task history.
_CHALLENGE_WAIT_SECS = 12.0
_ANUBIS_WAIT_SECS = 45.0


def challenge_wait_secs(vendor: str) -> float:
    """How long *vendor*'s interstitial is given to clear before it is reported.

    Args:
        vendor: A label returned by :func:`challenge_vendor`.

    Returns:
        Seconds; longer for Anubis, whose proof-of-work is CPU-bound.
    """
    return _ANUBIS_WAIT_SECS if vendor == ANUBIS_CHECK else _CHALLENGE_WAIT_SECS


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
    cleared is what a person does not do (and unticks it).  HUMAN's
    two-step variant asks to "press again when prompted".

    Args:
        vendor: A label returned by :func:`challenge_vendor`.

    Returns:
        2 for the press-and-hold bar, 1 for the Turnstile box, 0 for pages
        with nothing to press.
    """
    if vendor == PERIMETERX_CHECK:
        return 2
    return 1 if vendor == CLOUDFLARE_CHALLENGE else 0


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
# fiverr's "It needs a human touch", PerimeterX's stock page); its button
# fills the element.  Content pages of those sites also carry an empty
# ``#px-captcha`` placeholder, which is why the page text decides that a
# check is showing and this only locates the widget.
PRESS_AND_HOLD_BOX_JS = """() => {
  const el = document.querySelector('#px-captcha');
  if (!el) return [];
  const r = el.getBoundingClientRect();
  if (r.width < 40 || r.height < 20) return [];
  return [{x: r.x, y: r.y, width: r.width, height: r.height}];
}"""


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
