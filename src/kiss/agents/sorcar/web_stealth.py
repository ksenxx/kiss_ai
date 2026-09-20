# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Make the KISS browser look like a person's Chrome to bot-protection vendors.

Analysis of ``~/.kiss/sorcar.db`` (Sept 2026) found 95 hosts answering the
browser tool with a bot-protection page: Cloudflare managed challenges
(stackoverflow, dl.acm.org, npmjs, readthedocs sites, congress.gov, ...),
Cloudflare hard blocks (medium.com), Akamai "Access Denied" (carmax,
costco), Anubis (dblp, archwiki), Imperva (cato.org), PerimeterX
(walmart) and Google's ``/sorry/`` "unusual traffic" page.  Measured
against 14 of those URLs from the same machine, the stock configuration
(Playwright driver, headless Chromium) got 4 pages; Patchright driving a
headed Chromium on a virtual display got 11.  Three groups of helpers,
all consumed by :mod:`kiss.agents.sorcar.web_use_tool`:

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
  :func:`typing_chunks` (uneven typing cadence) and
  :func:`challenge_vendor` (recognise a bot-protection page so the tool
  can wait for it to clear and report it plainly).
"""

from __future__ import annotations

import atexit
import logging
import os
import random
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from importlib import import_module
from types import ModuleType
from urllib.parse import parse_qs, quote_plus, urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

_API_MODULE: ModuleType | None = None


def playwright_api() -> ModuleType:
    """Return the sync Playwright API module to drive Chromium with.

    Patchright is a drop-in replacement whose driver avoids the
    ``Runtime.enable`` / ``Console.enable`` CDP leaks that Cloudflare,
    Akamai and DataDome fingerprint; it is used whenever it is importable
    and the stock ``playwright`` package is the fallback.

    Returns:
        ``patchright.sync_api`` or ``playwright.sync_api``.
    """
    global _API_MODULE
    if _API_MODULE is None:
        try:
            _API_MODULE = import_module("patchright.sync_api")
        except ImportError:  # pragma: no cover — patchright is a declared dependency
            _API_MODULE = import_module("playwright.sync_api")
    return _API_MODULE


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


# ---------------------------------------------------------------------------
# Virtual display (Xvfb)
# ---------------------------------------------------------------------------

_XVFB_LOCK = threading.Lock()
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

    Returns:
        A display name such as ``":1"``, or ``None`` when this is not
        Linux or the ``Xvfb`` binary is not installed (the caller then
        falls back to real headless mode).
    """
    global _XVFB_PROC, _XVFB_DISPLAY
    with _XVFB_LOCK:
        if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
            return _XVFB_DISPLAY
        _XVFB_PROC, _XVFB_DISPLAY = None, None
        if not sys.platform.startswith("linux") or shutil.which("Xvfb") is None:
            return None
        read_fd, write_fd = os.pipe()
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
        if display is None:
            _killpg(proc)
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


def _killpg(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the wrapper shell and its Xvfb child together.

    Args:
        proc: Wrapper started with ``start_new_session=True``.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:  # pragma: no cover — group already reaped
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover — Xvfb ignores SIGTERM only when wedged
        os.killpg(proc.pid, signal.SIGKILL)


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
        _killpg(proc)


atexit.register(stop_virtual_display)


# ---------------------------------------------------------------------------
# Human-like input
# ---------------------------------------------------------------------------


def mouse_path(
    start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
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
_TITLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Cloudflare block", re.compile(r"^Attention Required! \| Cloudflare$", re.I)),
    ("Cloudflare challenge", re.compile(r"^Just a moment\.\.\.$", re.I)),
    ("Anubis proof-of-work check", re.compile(r"^Making sure you're not a bot!?$", re.I)),
    ("Akamai block", re.compile(r"^Access Denied$", re.I)),
    ("PerimeterX press-and-hold check", re.compile(r"^Robot or human\?$", re.I)),
    ("CAPTCHA", re.compile(r"^Are you a robot\?$|^Security check required$", re.I)),
)
# (vendor label, both patterns must match the body head)
_BODY_PATTERNS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    ("Google 'unusual traffic' page",
     re.compile(r"unusual traffic from your computer network", re.I),
     re.compile(r"IP address:", re.I)),
    ("Cloudflare block",
     re.compile(r"Sorry, you have been blocked", re.I),
     re.compile(r"Ray ID|Cloudflare", re.I)),
    ("Cloudflare challenge",
     re.compile(r"Performing security verification|Verify(?:ing)? you are human|"
                r"Enable JavaScript and cookies to continue", re.I),
     re.compile(r"Ray ID|Cloudflare", re.I)),
    ("Anubis proof-of-work check",
     re.compile(r"Sad Anubis|Oh noes!", re.I),
     re.compile(r"Anubis", re.I)),
    ("Akamai block",
     re.compile(r"You don't have permission to access", re.I),
     re.compile(r"errors\.edgesuite\.net|Reference #", re.I)),
    ("Imperva Incapsula block",
     re.compile(r"Request unsuccessful", re.I),
     re.compile(r"Incapsula incident ID", re.I)),
    ("PerimeterX press-and-hold check",
     re.compile(r"Press & Hold|press and hold", re.I),
     re.compile(r"human", re.I)),
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
        cont = (query.get("continue") or [""])[0]
        parsed, query = urlparse(cont), parse_qs(urlparse(cont).query)
    if parsed.path != "/search":
        return None
    terms = query.get("q") or []
    if not terms:
        return None
    return "https://www.bing.com/search?q=" + quote_plus(terms[0])
