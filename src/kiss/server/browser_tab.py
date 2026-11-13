# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stream the daemon machine's browser into a tab on every surface.

The daemon may run on a remote machine the user reaches only over ssh.
This service launches that machine's default browser (or the closest
Chromium-family browser, see :mod:`kiss.core.default_browser`) with a
KISS-owned persistent profile, and turns every browser page into one
KISS tab: JPEG frames from the DevTools ``Page.startScreencast`` API are
pushed to every client that shows the tab, and the clients' mouse,
wheel, keyboard and paste events are replayed through ``Input.*``.
A client acknowledges every frame it gets (``browserFrameAck``), and
no more than two frames are in flight to it at a time
(:mod:`kiss.server.frame_flow`): a slow link gets the newest frame at
its own pace instead of a backlog that stalls the link's keepalive
and drops the connection.

The browser is this daemon's machine's.  A ``browserOpen`` stamped
``machine`` (the chat is bound to a registered machine) goes to that
machine's daemon instead, through :mod:`kiss.server.remote_views`,
which shows the page here as the tab ``browser__N@machine`` and
forwards the tab-addressed commands by that id.

Playwright is asyncio-based, so the service owns a private event loop
thread; the public methods are thread-safe and may be called from the
daemon's command executor threads or from its server loop.

Events broadcast to clients (``tabId: ""`` reaches every client; frames
and ``browserTouch`` carry ``connId`` and reach only the clients that
show the tab):

- ``openBrowserTab``  {tab_id, url, title, browser, note, focus, popup}
- ``browserTabs``     {tabs: [openBrowserTab events]} - the snapshot a
  connecting client reconciles against (drops tabs closed while it
  was away)
- ``browserState``    {tab_id, url, title, canGoBack, canGoForward}
- ``browserFrame``    {tab_id, data (base64 JPEG), width, height} -
  answered by the client with ``browserFrameAck`` {tab_id}
- ``browserTouch``    {tab_id, id, scrollsLeft} - the answer to a
  ``browserInput`` event {kind: "touch", id, x, y} (a finger went
  down on the page at that point): whether a drag to the right from
  there would scroll something on the page.  The phone layout of the
  webapp turns such a drag into its swipe back only when it would not.
- ``closeBrowserTab`` {tab_id}
- ``browserError``    {tab_id, text}
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ipaddress
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import quote, urlsplit

from kiss.agents.sorcar.web_stealth import (
    disabled_blink_features,
    headed_user_agent,
    virtual_display,
)
from kiss.core.browser_handoff import is_headless_environment
from kiss.core.default_browser import ResolvedBrowser, resolve_browser
from kiss.core.locked import locked_call
from kiss.server.frame_flow import FrameFlow

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, CDPSession, Frame, Page, Playwright

    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

TAB_ID_PREFIX = "browser__"
_FALLBACK_HOME = "https://www.google.com"
_DEFAULT_VIEWPORT = (1280, 800)
_JPEG_QUALITY = 60
_MIN_FRAME_INTERVAL = 1 / 15
_OPEN_TIMEOUT = 90.0
_CALL_TIMEOUT = 30.0
# How long attaching a CDP session to a new page gets (_register): two
# round trips to a browser that has just created the page.  A browser
# that stops answering would otherwise hold every waiter on the
# attachment for good: the open, a second registration of the page and
# the cleanup of a cancelled open.
_ATTACH_TIMEOUT = 10.0
# Longest an agent thread waiting for an open goes without a bytecode
# check point, i.e. without noticing the task runner's Stop.
_WAIT_SLICE = 0.5
# After terminating a hung script, how long the page gets to answer a probe.
_INTERRUPT_PROBE_TIMEOUT = 3.0
# How long ``page.close()`` gets before the page is closed through
# DevTools instead (a normal close takes milliseconds; see _close_page).
_CLOSE_TIMEOUT = 2.0
# How long closing the browser (``context.close()``: Playwright waits for
# the browser *process* to exit, with no deadline of its own) and stopping
# the driver each get in _teardown_locked.  A healthy browser exits in well
# under a second; one that does not exit on ``Browser.close`` must not
# keep the lifecycle lock, or every later open would queue behind it.
_TEARDOWN_TIMEOUT = 5.0
# Longest line kept from a browser's own stderr in a launch-failure summary.
_FAILURE_LINE_LIMIT = 300

_MODIFIER_BITS = {"alt": 1, "ctrl": 2, "meta": 4, "shift": 8}
_KEY_TEXT = {"Enter": "\r"}
# What a drag to the right from a point of a document would scroll, for
# _scrolls_left.  True when something around the point scrolls sideways
# and stands more than a pixel (sizes are rounded) from its left end:
# the element at the point, one of its ancestors (through open shadow
# roots and slots) or the window.  As the browser has it:
# - an element scrolls when its overflow-x is auto or scroll, and a
#   text field always;
# - the window scrolls unless the overflow-x of the root element says
#   otherwise, or the body's while the root's overflow is visible; the
#   element that speaks for the window is no scroller of its own;
# - offsets start at the right end, and are then 0 or less, where the
#   text runs from right to left or the lines do: for the window, where
#   the body's do.
# The frame element when the point is on an embedded page, whose own
# answer then decides.  True as well on what a plugin draws (a PDF, an
# <object>): nothing here can look into it, and a drag nobody can vouch
# for stays the page's.  False otherwise.
_SCROLLS_LEFT_JS = """([x, y]) => {
  let el = document.elementFromPoint(x, y);
  while (el && el.shadowRoot) {
    const inner = el.shadowRoot.elementFromPoint(x, y);
    if (inner === el) break;
    el = inner;
  }
  const offLeftEnd = (box, left, flow) => {
    const fromRight = flow.writingMode === 'horizontal-tb'
      ? flow.direction === 'rtl'
      : flow.writingMode.endsWith('-rl');
    return left + (fromRight ? box.scrollWidth - box.clientWidth : 0) > 1;
  };
  const root = document.documentElement;
  const body = document.body;
  const rootStyle = getComputedStyle(root);
  const rootIsSilent = rootStyle.overflowX === 'visible' && rootStyle.overflowY === 'visible';
  const speaker = body && rootIsSilent ? body : root;
  for (let node = el; node; node = node.assignedSlot || node.parentNode || node.host) {
    if (node.nodeType !== 1 || node === speaker) continue;
    const style = getComputedStyle(node);
    const scrolls = style.overflowX === 'auto' || style.overflowX === 'scroll'
      || node instanceof HTMLInputElement;
    if (scrolls && offLeftEnd(node, node.scrollLeft, style)) return true;
  }
  const overflow = getComputedStyle(speaker).overflowX;
  if (overflow !== 'hidden' && overflow !== 'clip'
      && offLeftEnd(root, scrollX, getComputedStyle(body || root))) return true;
  if (el instanceof HTMLIFrameElement || el instanceof HTMLFrameElement) return el;
  return el instanceof HTMLEmbedElement || el instanceof HTMLObjectElement;
}"""
# Where the point (x, y) of a document lies in the page a frame element
# of it embeds, however the frame is drawn (scaled, turned, zoomed).
# The browser maps it: a mouse event on the element carries the point
# as offsets from the element's padding edge, in zoomed pixels.  They
# are read while the event is on its way, when it still has its target
# (one in a shadow tree loses it afterwards).  A browser without
# currentCSSZoom (Chromium before 128) gives no point: the question
# then fails, and a finger without an answer scrolls the page.
_FRAME_POINT_JS = """(frame, [x, y]) => {
  let at;
  frame.addEventListener('kiss-touch', e => { at = [e.offsetX, e.offsetY]; }, {once: true});
  frame.dispatchEvent(new MouseEvent('kiss-touch', {clientX: x, clientY: y}));
  const style = getComputedStyle(frame);
  const zoom = frame.currentCSSZoom;
  return [
    at[0] / zoom - parseFloat(style.paddingLeft),
    at[1] / zoom - parseFloat(style.paddingTop),
  ];
}"""
_HOST_RE = re.compile(r"^[\w.-]+(\.[a-z]{2,})(:\d+)?(/.*)?$", re.IGNORECASE)
# The ``BrowserType.launch_persistent_context: `` prefix of a Playwright
# message: a capitalised class name, a dot and a method (a lower-case
# ``chrome.exe: Access is denied.`` is a file name, not an API).
_PLAYWRIGHT_API_PREFIX_RE = re.compile(r"^[A-Z]\w+\.\w+: ")


@dataclass
class _PageRecord:
    """One browser page and the clients currently looking at it."""

    tab_id: str
    page: Page
    cdp: CDPSession | None = None
    # Chromium's target id: how an agent's own CDP client names this page.
    target_id: str = ""
    # Set once the CDP session is attached (or attaching failed): a
    # second registration of the same page waits for it.
    attached: asyncio.Event = field(default_factory=asyncio.Event)
    viewers: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Per viewer, the frames in flight and the one held back (see
    # frame_flow); set by _reserve, which creates the record.
    flow: FrameFlow[dict[str, Any]] = field(init=False)
    casting: bool = False
    last_frame_at: float = 0.0
    pending_frame: dict[str, Any] | None = None
    pending_session: int | None = None
    flush_handle: asyncio.TimerHandle | None = None
    url: str = ""
    title: str = ""
    can_go_back: bool = False
    can_go_forward: bool = False
    # Set when the service closes the page itself: a navigation this
    # aborts is no error to report.
    closing: bool = False


@dataclass
class _OpenClaim:
    """Who owns the tab an agent thread's open makes; read and written on the loop only.

    The open records its tab here unless the thread stopped waiting
    (``abandoned``), and closes the tab itself when it did; the
    thread's give-up closes a tab already recorded.  Both steps run on
    the loop without an ``await`` inside, so exactly one of them sees
    the other's mark and the tab is never left without an owner.
    """

    abandoned: bool = False
    tab_id: str = ""


@dataclass(frozen=True)
class AgentTab:
    """A browser tab opened for an agent, which drives it through its own CDP client.

    Attributes:
        tab_id: The KISS tab id (``browser__N``) shown on every surface.
        target_id: Chromium's target id of the page, so a second CDP
            client can tell this page from the user's other tabs.
        cdp_url: ``ws://127.0.0.1:<port>/devtools/browser/<id>``, the
            browser's DevTools endpoint (``connect_over_cdp`` accepts it);
            the id is minted at launch, so it names this browser instance.
    """

    tab_id: str
    target_id: str
    cdp_url: str


def home_url() -> str:
    """The page a new browser tab opens on: ``$KISS_BROWSER_HOME`` or Google."""
    return os.environ.get("KISS_BROWSER_HOME", "").strip() or _FALLBACK_HOME


def normalize_url(text: str) -> str:
    """Turn address-bar input into a URL.

    Args:
        text: What the user typed: a URL, a bare host name, or words.

    Returns:
        ``text`` unchanged when it has a scheme, ``https://`` + text
        for a host name, ``http://`` + text for an IP address or
        ``localhost`` (as browsers do: a LAN device or local server has
        no certificate), otherwise a Google search for the words.
    """
    text = text.strip()
    if not text:
        return home_url()
    if "://" in text or text.startswith(("about:", "chrome:", "data:", "file:")):
        return text
    if " " not in text:
        if _is_ip_address(text):
            return "http://" + text
        if _HOST_RE.match(text):
            return "https://" + text
    return "https://www.google.com/search?q=" + quote(text)


def _result_in_slices[T](future: concurrent.futures.Future[T], timeout: float) -> T:
    """Return *future*'s result, waiting in :data:`_WAIT_SLICE` slices.

    One ``future.result(timeout)`` is a C-level wait that an exception
    injected into the thread (the task runner's Stop) cannot interrupt:
    it would land only once the open finished, possibly minutes later.

    Raises:
        TimeoutError: *timeout* passed first.
    """
    deadline = time.monotonic() + timeout
    while not future.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no answer from the browser within {timeout:g}s")
        concurrent.futures.wait([future], min(remaining, _WAIT_SLICE))
    return future.result()


def _start(request: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Start *request* on the running loop; it runs up to its first wait before this returns."""
    return asyncio.Task(request, loop=asyncio.get_running_loop(), eager_start=True)


def _dispatch(
    act: Callable[..., Any] | None, request: Coroutine[Any, Any, Any]
) -> asyncio.Task[Any] | None:
    """Send *request* under the leave *act* gives: its task, or None once the leave is refused.

    *act* (the agent watchdog's, see :meth:`BrowserTabService.interrupt`)
    runs the start under the lock the end of the hung call takes, and
    the task is eager: *request* runs up to its first wait, which is
    after Playwright wrote it to its driver.  The leave and the send
    are thus one step, which the end of the hung call comes wholly
    before or after: a recovery never reaches the script, or the page,
    of the call that follows.  Without *act*, *request* is just sent.
    """
    task: asyncio.Task[Any] | None = _start(request) if act is None else act(_start, request)
    if task is None:
        request.close()
    return task


def launch_failure_summary(exc: BaseException) -> str:
    """Return the lines of a browser launch failure a person needs to read.

    Playwright's message for a browser that would not start repeats the
    whole command line and the call log.  The headline ("Failed to
    launch the browser process.", without the API name Playwright puts
    in front), the first line of the browser's own stderr (tagged
    ``[pid=N][err]``: a library it cannot load, a corrupt install) and
    the way it ended (``<process did exit: ...>``) say what happened;
    everything else is noise in a notification or a tooltip.

    Args:
        exc: The exception the launch raised.

    Returns:
        One to three lines; ``""`` for an exception without a message.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return ""
    summary = [_PLAYWRIGHT_API_PREFIX_RE.sub("", lines[0])]
    for marker in ("[err]", "<process did exit"):
        for line in lines[1:]:
            if marker in line:
                summary.append(line.split("] ", 1)[-1][:_FAILURE_LINE_LIMIT])
                break
    return "\n".join(summary)


def _is_ip_address(text: str) -> bool:
    """Whether *text* is an IP address (IPv6 in brackets) or ``localhost``.

    An optional ``:port`` and path may follow.
    """
    try:
        parts = urlsplit("//" + text)
        if parts.hostname != "localhost":
            ipaddress.ip_address(parts.hostname or "")
        _ = parts.port  # a port that is not a number raises
    except ValueError:
        return False
    return True


class BrowserTabService:
    """Launch the machine's browser and mirror its pages as KISS tabs.

    One instance lives on the daemon's :class:`AgentServer`.  Nothing
    starts until a tab is first opened (:meth:`open`,
    :meth:`open_for_agent`, :meth:`open_for_user`); :meth:`shutdown`
    stops the browser and the private event loop.
    """

    def __init__(self, printer: JsonPrinter, profile_dir: Path) -> None:
        """Create an idle service.

        Args:
            printer: The daemon printer whose thread-safe ``broadcast``
                delivers events to the connected clients.
            profile_dir: Persistent ``--user-data-dir`` for the browser
                (cookies and logins survive between sessions).
        """
        self._printer = printer
        self._profile_dir = profile_dir
        # Reached from agent task threads (``open_for_agent``,
        # ``tab_for_target``), where the runner's Stop injects
        # ``KeyboardInterrupt``: those entries go through ``locked_call``
        # on a reentrant lock, never a plain ``with``.
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._browser: ResolvedBrowser | None = None
        self._pages: dict[str, _PageRecord] = {}
        self._by_page: dict[int, _PageRecord] = {}
        self._next_id = 1
        self._closed = False
        # Serialises every browser lifecycle step on the private loop:
        # launching, creating a page, closing a page and tearing the
        # browser down.  Two opens racing must not start two browsers;
        # a launch on a profile whose previous browser is still exiting
        # fails; and whether the last close empties the browser is only
        # known once the page creation racing it has reserved its page.
        self._launch_lock = asyncio.Lock()
        # Pages being created by _open (not popups) - see _on_page.
        self._creating = 0

    # ------------------------------------------------------------------
    # Thread-safe public API
    # ------------------------------------------------------------------

    def open(self, url: str, conn_id: str, open_id: str = "") -> None:
        """Open *url* in a new browser page and announce the tab everywhere.

        Returns at once; the first call also launches the browser, which
        can take seconds.  Launch failures reach *conn_id* as a
        ``browserError`` event, as does an open for whose loop no thread
        could be started; one arriving after :meth:`shutdown` is dropped.

        Args:
            url: Address to load (normalised with :func:`normalize_url`).
            conn_id: Connection that asked.
            open_id: The ``openId`` of the command, echoed on its answer
                (see :meth:`_answer`).
        """
        try:
            future = self._run_threadsafe(
                self._open_reporting(normalize_url(url), conn_id, open_id)
            )
        except RuntimeError as exc:
            if not self._closed:
                logger.warning("browser tab: cannot open %s: %s", url, exc)
                self._answer(
                    {"type": "browserError", "tab_id": "", "text": str(exc)}, conn_id, open_id
                )
            return
        future.add_done_callback(_log_future_error)

    def open_for_agent(self) -> AgentTab:
        """Open a blank tab for an agent and switch every surface to it.

        Blocks until the tab is announced (launching the browser first
        when needed).  The agent then attaches its own Playwright client
        to :attr:`AgentTab.cdp_url` and drives the page whose target id
        is :attr:`AgentTab.target_id`, live in front of the user.

        Returns:
            The new tab's ids and the browser's DevTools endpoint.

        Raises:
            RuntimeError: The service is shut down or the browser cannot
                be launched.
        """
        claim = _OpenClaim()
        try:
            future = self._run_threadsafe(self._open_for_agent(claim), _OPEN_TIMEOUT)
            return _result_in_slices(future, _OPEN_TIMEOUT + _CALL_TIMEOUT)
        except BaseException as exc:
            # Also the task runner's Stop: nobody is left to use the tab.
            self._submit(self._abandon(claim))
            if isinstance(exc, Exception):
                raise RuntimeError(f"Cannot open a Browser tab: {exc}") from exc
            raise

    def open_for_user(self, url: str) -> bool:
        """Open *url* in a new tab that every surface switches to, for the user to complete.

        The hand-off behind :func:`kiss.core.browser_handoff.open_for_user`:
        a sign-in page, consent screen or developer portal the agent may
        not drive itself.  Blocks until the page has started loading
        (launching the browser first when needed), so the caller knows
        the user is looking at it.

        Args:
            url: The page to load, as given (no address-bar normalisation).

        Returns:
            ``True`` once the tab is announced and the navigation
            committed; ``False`` when the service is shut down, the
            browser cannot be launched or *url* cannot be loaded (the
            blank tab is closed again).
        """
        claim = _OpenClaim()
        try:
            future = self._run_threadsafe(self._open_for_user(url, claim), _OPEN_TIMEOUT)
            _result_in_slices(future, _OPEN_TIMEOUT + _CALL_TIMEOUT)
        except BaseException as exc:
            # Also the task runner's Stop: nobody is left to use the tab.
            self._submit(self._abandon(claim))
            if not isinstance(exc, Exception):
                raise
            # The caller falls back to another browser.
            logger.warning("browser tab: cannot open %s for the user: %s", url, exc)
            return False
        return True

    def tab_for_target(self, target_id: str) -> str | None:
        """The tab id of the attached page whose Chromium target id is *target_id*, if any."""
        return locked_call(self._lock, self._tab_for_target_locked, target_id)

    def _tab_for_target_locked(self, target_id: str) -> str | None:
        for rec in self._pages.values():
            if rec.target_id == target_id and rec.cdp is not None:
                return rec.tab_id
        return None

    def interrupt(self, tab_id: str, act: Callable[..., Any] | None = None) -> None:
        """Unwedge a page whose script hangs: stop the script, or close the tab.

        An agent's raw input call (a key press, a wheel event) blocks
        forever while a page handler spins in ``while(true)``.  Killing
        the browser is not an option here (it is the user's), so the
        running script is terminated, and the page closed when it still
        does not answer.  Fire and forget.

        Args:
            tab_id: The tab whose page hangs.
            act: The agent watchdog's leave to act (``act(send,
                *args)`` runs *send* unless the hung call has ended).
                The recovery sends its terminate, and its close, through
                it once it runs on the service's loop, and does nothing
                when refused: the script running by then, and its
                page, belong to the next call (:func:`_dispatch`).
                A page still wedged once the terminate has freed the
                hung call is thus the caller's to have closed
                (:meth:`close`), before its next call.
        """
        self._submit(self._interrupt(tab_id, act))

    def close(self, tab_id: str) -> None:
        """Close the browser page behind *tab_id* (every surface drops the tab)."""
        self._submit(self._close(tab_id))

    def navigate(self, tab_id: str, action: str, url: str) -> None:
        """Drive the address bar: ``go`` (to *url*), ``back``, ``forward`` or ``reload``."""
        self._submit(self._navigate(tab_id, action, url))

    def input(self, tab_id: str, event: dict[str, Any]) -> None:
        """Replay a client mouse/key/text event on the page (fire and forget).

        A ``touch`` event replays nothing: it asks what a drag from a
        finger's point would scroll, and the tab's viewers get the
        answer as a ``browserTouch`` event.
        """
        self._submit(self._input(tab_id, event))

    def viewport(self, tab_id: str, conn_id: str, width: int, height: int, visible: bool) -> None:
        """Record that *conn_id* shows (or hid) the tab at *width* x *height* CSS px.

        Frames are streamed only while at least one connection shows
        the tab; the page viewport follows the most recent visible size.
        """
        self._submit(self._viewport(tab_id, conn_id, width, height, visible))

    def frame_acked(self, tab_id: str, conn_id: str) -> None:
        """Record that *conn_id* received a frame of the tab (its ``browserFrameAck``).

        The frame held back for the connection while it was behind, if
        any, is sent now (see :mod:`kiss.server.frame_flow`).
        """
        self._submit(self._frame_acked(tab_id, conn_id))

    def viewer_gone(self, conn_id: str) -> None:
        """Forget every tab *conn_id* was watching (the client disconnected)."""
        self._submit(self._viewer_gone(conn_id))

    def open_events(self) -> list[dict[str, Any]]:
        """Return one unfocused ``openBrowserTab`` event per live, attached tab."""
        with self._lock:
            return [
                self._open_event(rec, focus=False)
                for rec in self._pages.values()
                if rec.cdp is not None
            ]

    def snapshot_event(self) -> dict[str, Any]:
        """The ``browserTabs`` event a connecting client reconciles its tabs against."""
        return {"type": "browserTabs", "tabs": self.open_events()}

    def send_snapshot(self, conn_id: str) -> None:
        """Send *conn_id* the ``browserTabs`` snapshot it reconciles its tabs against.

        Captured and sent in one step of the private loop, the thread
        every page's open and close is announced from in one step too:
        the snapshot either lists a page and precedes its
        ``closeBrowserTab``, or omits it and follows.  A snapshot
        captured off the loop and sent later could follow a close it
        predates, and the client, taking it as authoritative, would
        reopen the closed page on that surface alone.  Without a loop
        (no browser launched yet, or the service shut down) nothing is
        in flight and the empty snapshot goes at once; decided under
        the lock the first open starts the loop under and the shutdown
        stops it under, so the send lands on a running loop or goes
        directly, never on neither.

        Args:
            conn_id: The connection that asked (a ``ready``, a
                ``browserList``); ``""`` reaches every client.
        """
        with self._lock:
            if self._loop is None:
                self._emit(self.snapshot_event(), conn_id)
            else:
                self._submit(self._send_snapshot(conn_id))

    async def _send_snapshot(self, conn_id: str) -> None:
        self._emit(self.snapshot_event(), conn_id)

    def shutdown(self) -> None:
        """Close the browser, stop the private loop and join its thread."""
        with self._lock:
            self._closed = True
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None
        if loop is None or thread is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._teardown(announce=True), loop)
            future.result(_CALL_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — shutdown must not raise
            logger.debug("browser tab: shutdown error: %s", exc)
        # Teardown can leave independent navigation/input callbacks
        # suspended. Settle their submitted futures and finalizers before
        # closing the loop, including a teardown that missed its deadline.
        drain = asyncio.run_coroutine_threadsafe(self._cancel_pending(), loop)
        try:
            drain.result(_CLOSE_TIMEOUT + 1)
        except Exception as exc:  # noqa: BLE001 — a wedged loop must not block shutdown
            drain.cancel()
            logger.debug("browser tab: draining shutdown tasks: %s", exc)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        if thread.is_alive():
            # A callback is wedged on the loop: closing a running loop
            # raises, and the rest of the daemon's shutdown must go on.
            logger.warning("browser tab: the loop thread did not stop; leaving it behind")
            return
        loop.close()

    async def _cancel_pending(self) -> None:
        """Cancel outstanding service operations while the private loop still runs."""
        current = asyncio.current_task()
        pending = {task for task in asyncio.all_tasks() if task is not current}
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=_CLOSE_TIMEOUT)

    # ------------------------------------------------------------------
    # Loop plumbing
    # ------------------------------------------------------------------

    def _run_threadsafe(
        self, coro: Coroutine[Any, Any, _T], timeout: float | None = None, start: bool = True
    ) -> concurrent.futures.Future[_T]:
        """Hand *coro* to the private loop (started on first use) and return its future.

        The shutdown check and the hand-over are one critical section
        under ``_lock``, the lock :meth:`shutdown` flips ``_closed``
        under before it stops the loop: a coroutine accepted here is
        queued ahead of the teardown and runs (finding the service
        closed under ``_launch_lock`` at worst); one that arrives later
        is refused at once instead of landing on a stopped loop, where
        nothing would ever resolve its future and the caller would
        wait out its whole timeout.

        Args:
            coro: The coroutine to run.
            timeout: Seconds after which the loop cancels *coro*
                (``asyncio.wait_for``); ``None`` for no limit.
            start: Whether to start the loop when it does not run yet.

        Raises:
            RuntimeError: The service is shut down, the loop does not
                run and is not to be started, or no thread could be
                started for it (*coro* is closed).
        """
        try:
            return locked_call(self._lock, self._run_threadsafe_locked, coro, timeout, start)
        except RuntimeError:
            coro.close()
            raise

    def _run_threadsafe_locked(
        self, coro: Coroutine[Any, Any, _T], timeout: float | None, start: bool
    ) -> concurrent.futures.Future[_T]:
        loop = self._ensure_loop_locked(start)
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout)
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _ensure_loop_locked(self, start: bool = True) -> asyncio.AbstractEventLoop:
        """Return the private event loop, started on first use if *start*; ``_lock`` held."""
        if self._closed:
            raise RuntimeError("The browser service is shut down.")
        if self._loop is None:
            if not start:
                raise RuntimeError("The browser service has not started.")
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="kiss-browser-tab", daemon=True
            )
            try:
                thread.start()
            except RuntimeError:
                loop.close()  # no thread will ever run it
                raise
            self._loop, self._thread = loop, thread
        return self._loop

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run *coro* on the private loop without waiting; log failures.

        For what acts on the tabs the browser has: before an open has
        started the loop there is none, and *coro* is dropped.
        """
        try:
            future = self._run_threadsafe(coro, start=False)
        except RuntimeError:
            return
        future.add_done_callback(_log_future_error)

    def _emit(self, event: dict[str, Any], conn_id: str = "") -> None:
        """Broadcast *event* to one connection (*conn_id*) or to every client."""
        if conn_id:
            event["connId"] = conn_id
        else:
            event["tabId"] = ""
        self._printer.broadcast(event)

    # ------------------------------------------------------------------
    # Browser lifecycle (private loop)
    # ------------------------------------------------------------------

    async def _launch(self) -> BrowserContext:
        """Launch the resolved browser with the persistent profile, once.

        The caller holds ``_launch_lock``.
        """
        if self._context is None:
            self._context = await self._launch_new()
        return self._context

    async def _launch_new(self) -> BrowserContext:
        # Heavy import, deferred until the first browser tab opens.
        from playwright.async_api import async_playwright  # noqa: PLC0415

        self._browser = resolve_browser()
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        playwright = await async_playwright().start()
        self._playwright = playwright
        try:
            context = await self._launch_resolved(playwright)
            # Still inside the try: the mask's own awaits are the last
            # points where an abandoned open's cancellation can land, and
            # the launched browser would then hold the profile with no
            # ``_context`` to tear it down.
            await self._mask_headless_user_agent(context)
        except (Exception, asyncio.CancelledError):
            # The driver (a node process) must not outlive the failed
            # (or, past ``_OPEN_TIMEOUT``, abandoned) launch: stopping it
            # also closes a browser it already launched, so the next open
            # starts a fresh one on a free profile.  A shutdown that ran
            # meanwhile has already taken and stopped it.  The stop is
            # bounded as in ``_teardown_locked``: a driver that does not
            # exit would otherwise hold ``_launch_lock`` for good.
            if self._playwright is playwright:
                self._playwright = None
                try:
                    await asyncio.wait_for(playwright.stop(), _TEARDOWN_TIMEOUT)
                except Exception as exc:  # noqa: BLE001 — the driver exits on its own
                    logger.warning("browser tab: driver stop after failed launch: %r", exc)
            raise
        logger.info(
            "browser tab: launched %s (%s) with profile %s",
            self._browser.name, self._browser.executable, self._profile_dir,
        )
        context.on("page", self._on_page)
        context.on("close", self._on_context_close)
        return context

    async def _launch_resolved(self, playwright: Playwright) -> BrowserContext:
        """Launch the resolved browser, else Playwright's bundled Chromium.

        An installed browser can still be unable to start: a Chrome whose
        update replaced the launcher but lost the framework it loads
        aborts on every start until it is reinstalled, for instance.
        The bundled Chromium then shows the page, and the tab's badge
        names the browser that failed and why (``ResolvedBrowser.note``).
        """
        assert self._browser is not None
        try:
            return await self._launch_browser(playwright)
        except Exception as exc:  # noqa: BLE001 — any failure of that binary means "use ours"
            if self._browser.executable is None:
                raise
            reason = launch_failure_summary(exc) or repr(exc)
            logger.warning(
                "browser tab: %s (%s) cannot start, using the bundled Chromium: %s",
                self._browser.name, self._browser.executable, reason,
            )
            note = f"{self._browser.name} could not be started: {reason}"
            self._browser = ResolvedBrowser("Chromium (bundled)", None, False, note)
            return await self._launch_browser(playwright)

    async def _launch_browser(self, playwright: Playwright) -> BrowserContext:
        """Launch headed on the desktop (or a shared Xvfb), else headless."""
        display = None
        headless = is_headless_environment()
        if headless:
            # No window server: run a *headed* browser on a shared Xvfb
            # when one can be started.  Chromium brands headless mode
            # "HeadlessChrome" in the User-Agent and the Sec-CH-UA client
            # hints on purpose, and Google's sign-in answers that (and
            # navigator.webdriver) with "Couldn't sign you in", which
            # would break every OAuth hand-off streamed through the tab.
            display = virtual_display()
            headless = display is None
        try:
            return await self._launch_context(playwright, headless, display)
        except Exception:
            if headless:
                raise
            # A desktop session without a reachable window server (a Mac
            # reached over ssh, for instance): fall back to headless.
            return await self._launch_context(playwright, True, None)

    async def _launch_context(
        self, playwright: Playwright, headless: bool, display: str | None
    ) -> BrowserContext:
        assert self._browser is not None
        width, height = _DEFAULT_VIEWPORT
        return await playwright.chromium.launch_persistent_context(
            str(self._profile_dir),
            executable_path=self._browser.executable,
            headless=headless,
            viewport={"width": width, "height": height},
            env={**os.environ, "DISPLAY": display} if display else None,
            args=[
                # Clears navigator.webdriver (dropping --enable-automation
                # is not enough) and, when no one can see this browser's
                # own window, WebAuthn: its native security-key dialog
                # would block the streamed page invisibly.
                disabled_blink_features(unattended=headless or display is not None),
                "--no-first-run",
                "--no-default-browser-check",
                # The debugging port lets an agent attach a second CDP
                # client (open_for_agent); 0 picks a free port, recorded
                # in the profile's DevToolsActivePort file.
                "--remote-debugging-port=0",
            ],
            ignore_default_args=["--enable-automation"],
        )

    async def _mask_headless_user_agent(self, context: BrowserContext) -> None:
        """Rewrite a headless browser's ``HeadlessChrome`` UA token to ``Chrome``.

        Many sites answer that token with a bot challenge instead of
        content, which would hit the user in every streamed tab.  Both
        places a site reads it are patched: the ``User-Agent`` header
        and ``navigator.userAgent``.  A headed browser has no such token.
        """
        try:
            probe = context.pages[0] if context.pages else await context.new_page()
            headed = headed_user_agent(await probe.evaluate("navigator.userAgent"))
            if headed is None:
                return
            await context.set_extra_http_headers({"User-Agent": headed})
            await context.add_init_script(
                "Object.defineProperty(navigator, 'userAgent', "
                f"{{get: () => {json.dumps(headed)}}});"
            )
        except Exception:  # pragma: no cover - a fresh page rarely refuses evaluate
            logger.debug("browser tab: could not mask the headless user agent", exc_info=True)

    def _cdp_url(self) -> str:
        """The browser's DevTools websocket endpoint, from the profile's ``DevToolsActivePort``.

        The path carries an id minted at launch, so the URL also tells
        one browser instance from the next on the same port.
        """
        port, path = (self._profile_dir / "DevToolsActivePort").read_text().split()[:2]
        return f"ws://127.0.0.1:{port}{path}"

    async def _open_reporting(self, url: str, conn_id: str, open_id: str = "") -> None:
        try:
            await asyncio.wait_for(self._open(url, conn_id, open_id), _OPEN_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — any launch failure is reported to the user
            logger.warning("browser tab: cannot open %s: %s", url, exc)
            self._answer(
                {"type": "browserError", "tab_id": "", "text": launch_failure_summary(exc)},
                conn_id, open_id,
            )

    def _answer(self, event: dict[str, Any], conn_id: str, open_id: str) -> None:
        """Send *event* to *conn_id* as the answer to its ``browserOpen``, echoing *open_id*.

        A daemon showing this machine's pages forwards its clients'
        opens over one connection and tells their answers apart by the
        id it stamped each with (:mod:`kiss.server.remote_views`).
        """
        if open_id:
            event["openId"] = open_id
        self._emit(event, conn_id)

    async def _open(self, url: str, conn_id: str, open_id: str = "") -> str:
        rec = await self._new_tab()
        # Everyone gets the tab; only the surface that asked switches to it.
        if conn_id:
            self._answer(self._open_event(rec, focus=True), conn_id, open_id)
        try:
            await rec.page.goto(url, wait_until="commit")
        except Exception as exc:  # noqa: BLE001 — a bad URL must not kill the tab
            # A tab closed while still loading is no error.
            if not (rec.closing or rec.page.is_closed()):
                self._emit({"type": "browserError", "tab_id": rec.tab_id, "text": str(exc)})
        return rec.tab_id

    async def _open_for_agent(self, claim: _OpenClaim) -> AgentTab:
        rec = await self._claimed_tab(claim)
        return AgentTab(rec.tab_id, rec.target_id, self._cdp_url())

    async def _open_for_user(self, url: str, claim: _OpenClaim) -> None:
        rec = await self._claimed_tab(claim)
        try:
            await rec.page.goto(url, wait_until="commit")
        except Exception:
            # A page that cannot load is no hand-off: drop the blank tab
            # so the caller can fall back to another browser.
            await self._close_page(rec)
            raise

    async def _claimed_tab(self, claim: _OpenClaim) -> _PageRecord:
        """Open a tab for the agent thread behind *claim* and switch every surface to it.

        The agent wants the user's attention, or the user has to act on
        the page.  A thread that stopped waiting meanwhile gets no tab:
        it is closed again before it is ever focused.

        Raises:
            RuntimeError: The thread stopped waiting (see :class:`_OpenClaim`).
        """
        rec = await self._new_tab()
        if claim.abandoned:
            await self._close_page(rec)
            raise RuntimeError("Nobody is waiting for the tab any more.")
        claim.tab_id = rec.tab_id
        self._emit(self._open_event(rec, focus=True))
        return rec

    async def _abandon(self, claim: _OpenClaim) -> None:
        """Give up the open behind *claim*: close its tab now, or let the open close it."""
        claim.abandoned = True
        if claim.tab_id:
            await self._close(claim.tab_id)

    async def _new_tab(self) -> _PageRecord:
        """Launch the browser if needed, create a page and announce it unfocused.

        Under ``_launch_lock``: a close of the last tab that is still in
        flight finishes first (its teardown included, so the launch below
        finds a free profile), and the page is created before any later
        close or teardown runs.
        """
        async with self._launch_lock:
            if self._closed:
                # Queued just ahead of the shutdown's teardown: launching
                # now would hold the lock the teardown waits for.
                raise RuntimeError("The browser service is shut down.")
            context = await self._launch()
            creation = asyncio.ensure_future(self._create_page(context))
            try:
                page = await asyncio.shield(creation)
            except asyncio.CancelledError:
                # The open timed out (``_OPEN_TIMEOUT``) while the browser
                # was creating the page.  The creation goes on, and the
                # context's ``page`` event may announce the page: nobody
                # wants it, so it is closed once the creation returns it.
                self._schedule(self._close_created(creation))
                raise
        rec = None
        try:
            rec = await self._register(page, popup=False)
        finally:
            if rec is None:
                # The open timed out (``_OPEN_TIMEOUT``) while the page
                # was being attached, or the attach failed (the browser
                # closed the page, refused a CDP session or stopped
                # answering, ``_ATTACH_TIMEOUT``): nobody wants the page,
                # and a claim cannot close it (its tab id is unknown
                # until here), so it goes once its attachment has settled.
                self._schedule(self._close_unwanted(page))
        if rec is None:
            raise RuntimeError("The browser closed the new page before it could be attached.")
        return rec

    async def _create_page(self, context: BrowserContext) -> Page:
        """``context.new_page()``, counted in ``_creating`` until it returns (see _on_page)."""
        self._creating += 1
        try:
            return await context.new_page()
        finally:
            self._creating -= 1

    async def _close_created(self, creation: asyncio.Task[Page]) -> None:
        """Close the page of *creation*, whose open was cancelled before it returned."""
        try:
            page = await creation
        except Exception as exc:  # noqa: BLE001 — no page was created
            logger.debug("browser tab: abandoned page creation failed: %s", exc)
            return
        await self._close_unwanted(page)

    async def _close_unwanted(self, page: Page) -> None:
        """Close *page*, whose open gave it up while it was being created or attached.

        The open was cancelled, or its registration came back without a
        record.  The page event's own registration may still be
        attaching it: once that has settled (within ``_ATTACH_TIMEOUT``,
        whatever the browser does), a page it announced closes like any
        tab (its listener tells every surface); one never announced has
        no listener and is closed directly.
        """
        with self._lock:
            rec = self._by_page.get(id(page))
        if rec is not None:
            await rec.attached.wait()
            if rec.cdp is not None:
                await self._close_page(rec)
                return
        try:
            await asyncio.wait_for(page.close(), _CLOSE_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — the page may be gone already
            logger.debug("browser tab: unwanted page not closed: %s", exc)

    async def _register(self, page: Page, popup: bool) -> _PageRecord | None:
        """Give *page* a tab id, wire its events and announce it (idempotent).

        Both ``context.new_page()`` in :meth:`_open` and the context's
        ``page`` event call this for the same page, so the record is
        reserved under the lock before the first ``await``.  A page that
        closes (or refuses a CDP session, or whose browser does not
        answer within ``_ATTACH_TIMEOUT``) while being attached is
        forgotten again and never announced; the caller gets ``None``.
        The bound is what lets every waiter on ``attached`` settle: the
        second registration below, and :meth:`_close_unwanted`.

        Neither is a tab a teardown withdrew during an ``await`` here (a
        shutdown tells every surface the tab is closed and only then has
        the browser exit, so the page is still open): announced, or
        handed to the open that goes on to answer its surface, the tab
        would stay there with nothing left to close it.
        """
        rec, is_new = self._reserve(page)
        if not is_new:
            await rec.attached.wait()
            return rec if rec.cdp is not None and self._record(rec.tab_id) is rec else None
        try:
            async with asyncio.timeout(_ATTACH_TIMEOUT):
                cdp = await page.context.new_cdp_session(page)
                info = await cdp.send("Target.getTargetInfo")
            rec.target_id = info["targetInfo"]["targetId"]
            # Checked AFTER the last await: the transport dispatches every
            # message it has buffered without yielding, so the page's
            # ``close`` can be delivered (with no listener yet) between
            # the ``getTargetInfo`` reply and this coroutine resuming.
            # A record completed for such a page would be a tab nobody
            # can close.  Nothing yields between here and ``page.on``
            # below, so a close that arrives later reaches the listener.
            if page.is_closed():
                raise RuntimeError("page closed while attaching")
            if self._record(rec.tab_id) is not rec:
                raise RuntimeError("tab withdrawn while attaching")
        except BaseException as exc:  # noqa: BLE001 — a vanished popup is not an error
            # ``BaseException``: the ``asyncio.wait_for`` around
            # :meth:`_open` cancels this coroutine on timeout, and a
            # cancelled attach must forget the page too, or a second
            # ``_register`` for it would wait on ``attached`` forever.
            logger.debug("browser tab: %s not attached: %s", rec.tab_id, exc)
            with self._lock:
                self._pages.pop(rec.tab_id, None)
                self._by_page.pop(id(page), None)
            rec.attached.set()
            if isinstance(exc, asyncio.CancelledError):
                raise
            if popup:
                # No open waits for a popup (``_new_tab`` closes the
                # page of a failed open itself): close it here, or it
                # lives on with no record, shown by no surface.
                self._schedule(self._close_unwanted(page))
            return None
        rec.cdp = cdp
        rec.attached.set()
        cdp.on("Page.screencastFrame", lambda params: self._on_frame(rec, params))
        page.on("framenavigated", lambda frame: self._on_navigated(rec, frame))
        page.on("load", lambda _page: self._schedule_state(rec))
        page.on("close", lambda _page: self._schedule(self._on_page_closed(rec)))
        self._emit(self._open_event(rec, focus=popup, popup=popup))
        return rec

    def _reserve(self, page: Page) -> tuple[_PageRecord, bool]:
        """Return the record of *page* and whether this call created it."""
        with self._lock:
            existing = self._by_page.get(id(page))
            if existing is not None:
                return existing, False
            tab_id = f"{TAB_ID_PREFIX}{self._next_id}"
            self._next_id += 1
            rec = _PageRecord(tab_id=tab_id, page=page, url=page.url)
            rec.flow = FrameFlow(
                send=lambda conn_id, frame: self._emit(dict(frame), conn_id),
                schedule=self._call_later,
            )
            self._pages[tab_id] = rec
            self._by_page[id(page)] = rec
            return rec, True

    def _open_event(self, rec: _PageRecord, focus: bool, popup: bool = False) -> dict[str, Any]:
        """The ``openBrowserTab`` announcement of *rec*.

        ``focus`` asks the receiving surface to switch to the tab;
        ``popup`` marks a page the browser opened itself (a client only
        honours ``focus`` for a popup while it is showing a browser tab,
        so a popup never yanks a surface out of its chat).
        """
        browser = self._browser
        return {
            "type": "openBrowserTab",
            "tab_id": rec.tab_id,
            "url": rec.url,
            "title": rec.title,
            "browser": browser.name if browser else "",
            "isDefaultBrowser": bool(browser and browser.is_default),
            "note": browser.note if browser else "",
            "focus": focus,
            "popup": popup,
            "canGoBack": rec.can_go_back,
            "canGoForward": rec.can_go_forward,
        }

    def _on_page(self, page: Page) -> None:
        """A new page appeared: mirror it.

        Fires for popups and ``target=_blank`` links, but also for the
        pages :meth:`_open` creates; the counter tells them apart (so a
        page the user asked for is not announced as a focus-stealing
        popup, whichever registration runs first).
        """
        self._schedule(self._register(page, popup=self._creating == 0))

    def _on_context_close(self, context: BrowserContext) -> None:
        """The browser exited (headed window closed, crash): drop every tab.

        A context the service closed itself was taken out of ``_context``
        before its close, so its event is stale: acting on it would tear
        down the browser launched since.  The check is made again under
        the lock (:meth:`_teardown`): the close of the last tab, which
        the same exit fires first, tears the browser down ahead of this,
        and a reopen queued between the two launches a new one.
        """
        if context is self._context:
            logger.warning("browser tab: the browser exited on its own; closing every tab")
            self._schedule(self._teardown(announce=True, context=context))

    async def _on_page_closed(self, rec: _PageRecord) -> None:
        with self._lock:
            known = self._pages.pop(rec.tab_id, None) is not None
            self._by_page.pop(id(rec.page), None)
            remaining = len(self._pages)
        if rec.flush_handle is not None:
            rec.flush_handle.cancel()
        rec.flow.close()
        if not known:
            # Already withdrawn by a teardown, whose context close is
            # what fired this event: nothing left to announce or close.
            return
        logger.info(
            "browser tab: %s closed (%s); %d tab(s) left", rec.tab_id, rec.url, remaining,
        )
        self._emit({"type": "closeBrowserTab", "tab_id": rec.tab_id})
        if remaining:
            return
        # The last tab closes the browser.  Decided under the lock: a
        # page being created meanwhile (the user closed one tab while
        # opening another) has been reserved by the time it is free.
        async with self._launch_lock:
            if not self._pages:
                await self._teardown_locked(announce=False)

    async def _teardown(self, announce: bool, context: BrowserContext | None = None) -> None:
        """Close the browser and forget every page; given *context*, only while it is current."""
        async with self._launch_lock:
            if context is None or context is self._context:
                await self._teardown_locked(announce)

    async def _teardown_locked(self, announce: bool) -> None:
        """Close the browser and forget every page (caller holds ``_launch_lock``).

        Both awaits are bounded by ``_TEARDOWN_TIMEOUT``: a browser that
        does not exit on ``Browser.close`` (wedged, stopped, waiting on a
        native modal) would otherwise hold ``_launch_lock`` for good and
        kill the feature until the daemon restarts.  Stopping the driver
        closes its stdin: the driver then SIGKILLs a browser whose close
        is still pending and exits, which frees the profile for the next
        launch.  A driver that does not even manage that is abandoned (it
        is a child process; it force-exits by itself within 30 s).
        """
        with self._lock:
            records = list(self._pages.values())
            self._pages.clear()
            self._by_page.clear()
            context, playwright = self._context, self._playwright
            self._context, self._playwright = None, None
        if context is not None:
            logger.info(
                "browser tab: closing the browser (%s)",
                "it exited or the daemon stops" if announce else "its last tab closed",
            )
        for rec in records:
            rec.closing = True  # the close below aborts a load: no error to report
            if rec.flush_handle is not None:
                rec.flush_handle.cancel()
            rec.flow.close()
            if announce:
                self._emit({"type": "closeBrowserTab", "tab_id": rec.tab_id})
        if context is not None:
            try:
                await asyncio.wait_for(context.close(), _TEARDOWN_TIMEOUT)
            except TimeoutError:
                logger.warning("browser tab: the browser did not exit; stopping the driver")
            except Exception as exc:  # noqa: BLE001 — browser may already be gone
                logger.debug("browser tab: context close: %s", exc)
        if playwright is not None:
            try:
                await asyncio.wait_for(playwright.stop(), _TEARDOWN_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 — the driver exits on its own
                logger.warning("browser tab: driver stop: %r", exc)

    # ------------------------------------------------------------------
    # Navigation and state
    # ------------------------------------------------------------------

    def _record(self, tab_id: str) -> _PageRecord | None:
        with self._lock:
            return self._pages.get(tab_id)

    async def _close(self, tab_id: str) -> None:
        rec = self._record(tab_id)
        if rec is not None:
            await self._close_page(rec)

    async def _close_page(self, rec: _PageRecord, act: Callable[..., Any] | None = None) -> None:
        """Close *rec*'s page (under ``_launch_lock``, like every lifecycle step).

        A recovery closes under its watchdog's leave *act*: the close is
        sent in the step that gets the leave (:func:`_dispatch`), and
        the page is left alone once it is refused.

        Chromium now and then swallows the close of a page whose
        navigation is committing (a tab closed right after it opened):
        ``page.close()`` then never returns, and neither does a second
        one, which only re-awaits the same closed promise.  The page
        would live on with no surface showing it, keeping the browser
        alive.  The DevTools ``Page.close`` command still closes such a
        page, so it is sent when the close is overdue.

        Every step is bounded: a browser that stops answering (and may
        exit meanwhile, which leaves Playwright's request pending for
        good) must not keep the lock, or its own close events and every
        later open would queue behind this close forever.
        """
        if act is None:
            rec.closing = True  # at once: a load that fails while this waits is no error
        async with self._launch_lock:
            close = _dispatch(act, rec.page.close())
            if close is None:
                logger.info("browser tab: %s stays open: its hung call has ended", rec.tab_id)
                return
            rec.closing = True
            try:
                await asyncio.wait_for(close, _CLOSE_TIMEOUT)
                return
            except TimeoutError:
                logger.info("browser tab: %s: close swallowed; closing via DevTools", rec.tab_id)
            try:
                if rec.cdp is not None:
                    await asyncio.wait_for(rec.cdp.send("Page.close"), _CLOSE_TIMEOUT)
                await asyncio.wait_for(rec.page.close(), _CLOSE_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 — the tab stays; the user can close it again
                logger.warning("browser tab: %s did not close: %s", rec.tab_id, exc)

    async def _interrupt(self, tab_id: str, act: Callable[..., Any] | None = None) -> None:
        rec = self._record(tab_id)
        if rec is None or rec.cdp is None:
            return
        terminate = _dispatch(act, rec.cdp.send("Runtime.terminateExecution"))
        if terminate is None:
            return
        try:
            # Bounded like the probe: a browser that stops answering
            # would otherwise keep the recovery (and the agent's blocked
            # input call behind it) waiting on the terminate request
            # for good, the bounded close below never reached.
            await asyncio.wait_for(terminate, _INTERRUPT_PROBE_TIMEOUT)
            await asyncio.wait_for(rec.page.evaluate("1"), _INTERRUPT_PROBE_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — still wedged: the tab is lost
            logger.warning("browser tab: %s does not answer, closing it: %s", tab_id, exc)
            await self._close_page(rec, act)

    async def _navigate(self, tab_id: str, action: str, url: str) -> None:
        rec = self._record(tab_id)
        if rec is None:
            return
        try:
            if action == "back":
                await rec.page.go_back(wait_until="commit")
            elif action == "forward":
                await rec.page.go_forward(wait_until="commit")
            elif action == "reload":
                await rec.page.reload(wait_until="commit")
            else:
                await rec.page.goto(normalize_url(url), wait_until="commit")
        except Exception as exc:  # noqa: BLE001 — navigation errors are shown, not raised
            # A tab closed while still loading is no error (as in _open).
            if not (rec.closing or rec.page.is_closed()):
                self._emit({"type": "browserError", "tab_id": tab_id, "text": str(exc)})

    def _on_navigated(self, rec: _PageRecord, frame: Any) -> None:
        if frame == rec.page.main_frame:
            rec.url = frame.url
            self._schedule_state(rec)

    def _schedule_state(self, rec: _PageRecord) -> None:
        self._schedule(self._emit_state(rec))

    async def _emit_state(self, rec: _PageRecord) -> None:
        if rec.cdp is None or rec.page.is_closed():
            return
        try:
            rec.title = await rec.page.title()
            history = await rec.cdp.send("Page.getNavigationHistory")
        except Exception:  # noqa: BLE001 — page navigated away mid-query
            return
        index = int(history.get("currentIndex", 0))
        entries = history.get("entries", [])
        # A page created through CDP starts on an about:blank entry that
        # a real new tab would not show; going "back" to it is pointless.
        real_before = [e for e in entries[:index] if e.get("url") != "about:blank"]
        rec.url = rec.page.url
        rec.can_go_back = bool(real_before)
        rec.can_go_forward = index < len(entries) - 1
        self._emit(
            {
                "type": "browserState",
                "tab_id": rec.tab_id,
                "url": rec.url,
                "title": rec.title,
                "canGoBack": rec.can_go_back,
                "canGoForward": rec.can_go_forward,
            }
        )

    # ------------------------------------------------------------------
    # Screencast
    # ------------------------------------------------------------------

    async def _viewport(
        self, tab_id: str, conn_id: str, width: int, height: int, visible: bool
    ) -> None:
        rec = self._record(tab_id)
        if rec is None:
            return
        if visible and width > 0 and height > 0:
            # A resize keeps the viewer's frames in flight on its
            # account: the window is the viewer's, not the size's.
            rec.viewers[conn_id] = (width, height)
            if rec.page.viewport_size != {"width": width, "height": height}:
                await rec.page.set_viewport_size({"width": width, "height": height})
        else:
            rec.viewers.pop(conn_id, None)
            rec.flow.forget(conn_id)
        await self._sync_cast(rec)

    async def _frame_acked(self, tab_id: str, conn_id: str) -> None:
        rec = self._record(tab_id)
        if rec is not None:
            rec.flow.acked(conn_id)

    async def _viewer_gone(self, conn_id: str) -> None:
        with self._lock:
            records = list(self._pages.values())
        for rec in records:
            rec.flow.forget(conn_id)
            if rec.viewers.pop(conn_id, None) is not None:
                await self._sync_cast(rec)

    async def _sync_cast(self, rec: _PageRecord) -> None:
        """Start or stop the screencast to match whether anyone is watching."""
        if rec.cdp is None:
            return
        want = bool(rec.viewers) and not rec.page.is_closed()
        if want == rec.casting:
            if want:
                # A new or resized viewer needs a fresh frame right away.
                await self._restart_cast(rec)
            return
        rec.casting = want
        try:
            if want:
                await self._start_cast(rec)
            else:
                await rec.cdp.send("Page.stopScreencast")
        except Exception as exc:  # noqa: BLE001 — page may be closing
            logger.debug("browser tab: screencast toggle: %s", exc)

    async def _restart_cast(self, rec: _PageRecord) -> None:
        assert rec.cdp is not None
        try:
            await rec.cdp.send("Page.stopScreencast")
            await self._start_cast(rec)
        except Exception as exc:  # noqa: BLE001 — page may be closing
            logger.debug("browser tab: screencast restart: %s", exc)

    async def _start_cast(self, rec: _PageRecord) -> None:
        assert rec.cdp is not None
        width = max(w for w, _ in rec.viewers.values())
        height = max(h for _, h in rec.viewers.values())
        await rec.cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": _JPEG_QUALITY,
                "maxWidth": width,
                "maxHeight": height,
                "everyNthFrame": 1,
            },
        )

    def _on_frame(self, rec: _PageRecord, params: dict[str, Any]) -> None:
        """Queue a screencast frame for the viewers, rate limited.

        The frame is acknowledged only when it is sent (or superseded by
        a newer one), so Chrome's own in-flight limit throttles capture
        to what the daemon actually forwards.
        """
        if rec.pending_session is not None:
            self._ack(rec, rec.pending_session)
        rec.pending_session = int(params["sessionId"])
        meta = params.get("metadata", {})
        rec.pending_frame = {
            "type": "browserFrame",
            "tab_id": rec.tab_id,
            "data": params["data"],
            "width": meta.get("deviceWidth"),
            "height": meta.get("deviceHeight"),
        }
        now = time.monotonic()
        wait = rec.last_frame_at + _MIN_FRAME_INTERVAL - now
        if wait <= 0:
            self._flush_frame(rec)
        elif rec.flush_handle is None:
            loop = asyncio.get_running_loop()
            rec.flush_handle = loop.call_later(wait, self._flush_frame, rec)

    def _flush_frame(self, rec: _PageRecord) -> None:
        rec.flush_handle = None
        frame, session = rec.pending_frame, rec.pending_session
        rec.pending_frame, rec.pending_session = None, None
        if session is not None:
            self._ack(rec, session)
        if frame is None or not rec.viewers:
            return
        rec.last_frame_at = time.monotonic()
        for conn_id in list(rec.viewers):
            rec.flow.offer(conn_id, frame)

    def _ack(self, rec: _PageRecord, session: int) -> None:
        if rec.cdp is not None and not rec.page.is_closed():
            self._schedule(rec.cdp.send("Page.screencastFrameAck", {"sessionId": session}))

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    async def _input(self, tab_id: str, ev: dict[str, Any]) -> None:
        rec = self._record(tab_id)
        if rec is None or rec.cdp is None or rec.page.is_closed():
            return
        kind = ev.get("kind")
        try:
            if kind == "mouse":
                await rec.cdp.send("Input.dispatchMouseEvent", _mouse_params(ev))
            elif kind == "key":
                await rec.cdp.send("Input.dispatchKeyEvent", _key_params(ev))
            elif kind == "text":
                await rec.cdp.send("Input.insertText", {"text": str(ev.get("text", ""))})
            elif kind == "touch":
                await self._answer_touch(rec, ev)
        except Exception as exc:  # noqa: BLE001 — input on a closing page is dropped
            logger.debug("browser tab: input %s: %s", kind, exc)

    async def _answer_touch(self, rec: _PageRecord, ev: dict[str, Any]) -> None:
        """Tell *rec*'s viewers what a drag to the right from the finger of *ev* would scroll.

        Every viewer is told: the one whose finger it is knows the id.
        """
        point = float(ev.get("x", 0)), float(ev.get("y", 0))
        scrolls = await _scrolls_left(rec.page.main_frame, *point)
        for conn_id in list(rec.viewers):
            self._emit(
                {
                    "type": "browserTouch",
                    "tab_id": rec.tab_id,
                    "id": ev.get("id"),
                    "scrollsLeft": scrolls,
                },
                conn_id,
            )

    @staticmethod
    def _call_later(delay: float, callback: Callable[[], None]) -> None:
        """Run *callback* on the private loop after *delay* seconds (called on the loop)."""
        asyncio.get_running_loop().call_later(delay, callback)

    def _schedule(self, coro: Any) -> None:
        """Run *coro* from a Playwright callback (already on the private loop)."""
        task = asyncio.ensure_future(coro)
        task.add_done_callback(_log_future_error)


async def _scrolls_left(frame: Frame, x: float, y: float) -> bool:
    """Whether a drag to the right from (*x*, *y*) of *frame*'s viewport would scroll something.

    Looks at what is under the point, in *frame* and in the pages
    embedded there (of other sites too), see :data:`_SCROLLS_LEFT_JS`.
    """
    found = await frame.evaluate_handle(_SCROLLS_LEFT_JS, [x, y])
    try:
        owner = found.as_element()
        if owner is None:
            return bool(await found.json_value())
        embedded = await owner.content_frame()
        # A frame element has a frame, unless it left the page since
        # (the caller drops an input that fails).
        assert embedded is not None
        return await _scrolls_left(embedded, *await owner.evaluate(_FRAME_POINT_JS, [x, y]))
    finally:
        await found.dispose()


def _modifiers(ev: dict[str, Any]) -> int:
    return sum(bit for name, bit in _MODIFIER_BITS.items() if ev.get(name))


def _mouse_params(ev: dict[str, Any]) -> dict[str, Any]:
    """Translate a client pointer/wheel event into ``Input.dispatchMouseEvent`` params."""
    params: dict[str, Any] = {
        "type": str(ev.get("action", "mouseMoved")),
        "x": float(ev.get("x", 0)),
        "y": float(ev.get("y", 0)),
        "button": str(ev.get("button", "none")),
        "buttons": int(ev.get("buttons", 0)),
        "clickCount": int(ev.get("clickCount", 0)),
        "modifiers": _modifiers(ev),
    }
    if params["type"] == "mouseWheel":
        params["deltaX"] = float(ev.get("deltaX", 0))
        params["deltaY"] = float(ev.get("deltaY", 0))
    return params


def _key_params(ev: dict[str, Any]) -> dict[str, Any]:
    """Translate a client ``keydown``/``keyup`` into ``Input.dispatchKeyEvent`` params.

    Printable keys (and Enter) carry ``text`` so the page receives a
    character; with Ctrl/Alt/Meta held the key is a shortcut and no
    text is sent, exactly as a physical keyboard behaves.
    """
    key = str(ev.get("key", ""))
    key_code = int(ev.get("keyCode", 0) or 0)
    params: dict[str, Any] = {
        "key": key,
        "code": str(ev.get("code", "")),
        "windowsVirtualKeyCode": key_code,
        "nativeVirtualKeyCode": key_code,
        "modifiers": _modifiers(ev),
        "autoRepeat": bool(ev.get("repeat", False)),
    }
    if ev.get("action") == "up":
        params["type"] = "keyUp"
        return params
    shortcut = bool(ev.get("ctrl") or ev.get("alt") or ev.get("meta"))
    text = "" if shortcut else (_KEY_TEXT.get(key) or (key if len(key) == 1 else ""))
    if text:
        params.update({"type": "keyDown", "text": text, "unmodifiedText": text})
    else:
        params["type"] = "rawKeyDown"
    return params


def _log_future_error(future: Any) -> None:
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:
        logger.debug("browser tab: background task failed: %s", exc)
