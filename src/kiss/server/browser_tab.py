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

Playwright is asyncio-based, so the service owns a private event loop
thread; the public methods are thread-safe and may be called from the
daemon's command executor threads or from its server loop.

Events broadcast to clients (``tabId: ""`` reaches every client; frames
carry ``connId`` and reach only the clients that show the tab):

- ``openBrowserTab``  {tab_id, url, title, browser, note, focus, popup}
- ``browserTabs``     {tabs: [openBrowserTab events]} - the snapshot a
  connecting client reconciles against (drops tabs closed while it
  was away)
- ``browserState``    {tab_id, url, title, canGoBack, canGoForward}
- ``browserFrame``    {tab_id, data (base64 JPEG), width, height}
- ``closeBrowserTab`` {tab_id}
- ``browserError``    {tab_id, text}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from kiss.core.browser_handoff import is_headless_environment
from kiss.core.default_browser import ResolvedBrowser, resolve_browser

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, CDPSession, Page, Playwright

    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)

TAB_ID_PREFIX = "browser__"
_FALLBACK_HOME = "https://www.google.com"
_DEFAULT_VIEWPORT = (1280, 800)
_JPEG_QUALITY = 60
_MIN_FRAME_INTERVAL = 1 / 15
_OPEN_TIMEOUT = 90.0
_CALL_TIMEOUT = 30.0
_HEADLESS_UA_TOKEN = "HeadlessChrome"
# After terminating a hung script, how long the page gets to answer a probe.
_INTERRUPT_PROBE_TIMEOUT = 3.0

_MODIFIER_BITS = {"alt": 1, "ctrl": 2, "meta": 4, "shift": 8}
_KEY_TEXT = {"Enter": "\r"}
_HOST_RE = re.compile(r"^[\w.-]+(\.[a-z]{2,})(:\d+)?(/.*)?$", re.IGNORECASE)


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
    casting: bool = False
    last_frame_at: float = 0.0
    pending_frame: dict[str, Any] | None = None
    pending_session: int | None = None
    flush_handle: asyncio.TimerHandle | None = None
    url: str = ""
    title: str = ""
    can_go_back: bool = False
    can_go_forward: bool = False


@dataclass(frozen=True)
class AgentTab:
    """A browser tab opened for an agent, which drives it through its own CDP client.

    Attributes:
        tab_id: The KISS tab id (``browser__N``) shown on every surface.
        target_id: Chromium's target id of the page, so a second CDP
            client can tell this page from the user's other tabs.
        cdp_url: ``http://127.0.0.1:<port>`` of the browser's DevTools
            endpoint (``connect_over_cdp`` accepts it).
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
        for a host name, otherwise a Google search for the words.
    """
    text = text.strip()
    if not text:
        return home_url()
    if "://" in text or text.startswith(("about:", "chrome:", "data:", "file:")):
        return text
    if " " not in text and (_HOST_RE.match(text) or text.startswith("localhost")):
        return "https://" + text
    return "https://www.google.com/search?q=" + quote(text)


class BrowserTabService:
    """Launch the machine's browser and mirror its pages as KISS tabs.

    One instance lives on the daemon's :class:`VSCodeServer`.  Nothing
    starts until :meth:`open` is first called; :meth:`shutdown` stops
    the browser and the private event loop.
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
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._browser: ResolvedBrowser | None = None
        self._pages: dict[str, _PageRecord] = {}
        self._by_page: dict[int, _PageRecord] = {}
        self._next_id = 1
        self._closed = False
        self._launch_lock: asyncio.Lock | None = None
        # Pages being created by _open (not popups) - see _on_page.
        self._creating = 0

    # ------------------------------------------------------------------
    # Thread-safe public API
    # ------------------------------------------------------------------

    def open(self, url: str, conn_id: str) -> None:
        """Open *url* in a new browser page and announce the tab everywhere.

        Returns at once; the first call also launches the browser, which
        can take seconds.  Launch failures reach *conn_id* as a
        ``browserError`` event.

        Args:
            url: Address to load (normalised with :func:`normalize_url`).
            conn_id: Connection that asked.
        """
        self._submit(self._open_reporting(normalize_url(url), conn_id))

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
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._open_for_agent(), loop)
        try:
            return future.result(_OPEN_TIMEOUT)
        except Exception as exc:
            raise RuntimeError(f"Cannot open a Browser tab: {exc}") from exc

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
        try:
            loop = self._ensure_loop()
            future = asyncio.run_coroutine_threadsafe(self._open_for_user(url), loop)
            future.result(_OPEN_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — the caller falls back to another browser
            logger.warning("browser tab: cannot open %s for the user: %s", url, exc)
            return False
        return True

    def tab_for_target(self, target_id: str) -> str | None:
        """The tab id of the attached page whose Chromium target id is *target_id*, if any."""
        with self._lock:
            for rec in self._pages.values():
                if rec.target_id == target_id and rec.cdp is not None:
                    return rec.tab_id
        return None

    def interrupt(self, tab_id: str) -> None:
        """Unwedge a page whose script hangs: stop the script, or close the tab.

        An agent's raw input call (a key press, a wheel event) blocks
        forever while a page handler spins in ``while(true)``.  Killing
        the browser is not an option here (it is the user's), so the
        running script is terminated, and the page closed when it still
        does not answer.  Fire and forget.
        """
        self._submit(self._interrupt(tab_id))

    def close(self, tab_id: str) -> None:
        """Close the browser page behind *tab_id* (every surface drops the tab)."""
        self._submit(self._close(tab_id))

    def navigate(self, tab_id: str, action: str, url: str) -> None:
        """Drive the address bar: ``go`` (to *url*), ``back``, ``forward`` or ``reload``."""
        self._submit(self._navigate(tab_id, action, url))

    def input(self, tab_id: str, event: dict[str, Any]) -> None:
        """Replay a client mouse/key/text event on the page (fire and forget)."""
        self._submit(self._input(tab_id, event))

    def viewport(self, tab_id: str, conn_id: str, width: int, height: int, visible: bool) -> None:
        """Record that *conn_id* shows (or hid) the tab at *width* x *height* CSS px.

        Frames are streamed only while at least one connection shows
        the tab; the page viewport follows the most recent visible size.
        """
        self._submit(self._viewport(tab_id, conn_id, width, height, visible))

    def viewer_gone(self, conn_id: str) -> None:
        """Forget every tab *conn_id* was watching (the client disconnected)."""
        if self._loop is not None:
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
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()

    # ------------------------------------------------------------------
    # Loop plumbing
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the private event loop thread on first use."""
        with self._lock:
            if self._closed:
                raise RuntimeError("The browser service is shut down.")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="kiss-browser-tab", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def _submit(self, coro: Any) -> None:
        """Run *coro* on the private loop without waiting; log failures."""
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            coro.close()
            return
        future = asyncio.run_coroutine_threadsafe(coro, loop)
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

        Serialised: two opens racing (a double-click on "Browser") must
        not start two browsers on one profile.
        """
        if self._launch_lock is None:
            self._launch_lock = asyncio.Lock()
        async with self._launch_lock:
            if self._context is None:
                self._context = await self._launch_new()
            return self._context

    async def _launch_new(self) -> BrowserContext:
        # Heavy import, deferred until the first browser tab opens.
        from playwright.async_api import async_playwright  # noqa: PLC0415

        self._browser = resolve_browser()
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        headless = is_headless_environment()
        try:
            context = await self._launch_context(headless)
        except Exception:
            if headless:
                raise
            # A desktop session without a reachable window server (a Mac
            # reached over ssh, for instance): fall back to headless.
            context = await self._launch_context(True)
        await self._mask_headless_user_agent(context)
        context.on("page", self._on_page)
        context.on("close", self._on_context_close)
        return context

    async def _launch_context(self, headless: bool) -> BrowserContext:
        assert self._playwright is not None and self._browser is not None
        width, height = _DEFAULT_VIEWPORT
        return await self._playwright.chromium.launch_persistent_context(
            str(self._profile_dir),
            executable_path=self._browser.executable,
            headless=headless,
            viewport={"width": width, "height": height},
            # The debugging port lets an agent attach a second CDP client
            # (open_for_agent); 0 picks a free port, recorded in the
            # profile's DevToolsActivePort file.
            args=["--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0"],
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
            user_agent = await probe.evaluate("navigator.userAgent")
            if _HEADLESS_UA_TOKEN not in user_agent:
                return
            headed = user_agent.replace(_HEADLESS_UA_TOKEN, "Chrome")
            await context.set_extra_http_headers({"User-Agent": headed})
            await context.add_init_script(
                "Object.defineProperty(navigator, 'userAgent', "
                f"{{get: () => {json.dumps(headed)}}});"
            )
        except Exception:  # pragma: no cover - a fresh page rarely refuses evaluate
            logger.debug("browser tab: could not mask the headless user agent", exc_info=True)

    def _cdp_url(self) -> str:
        """The browser's DevTools HTTP endpoint, from the profile's ``DevToolsActivePort``."""
        port = (self._profile_dir / "DevToolsActivePort").read_text().split()[0]
        return f"http://127.0.0.1:{port}"

    async def _open_reporting(self, url: str, conn_id: str) -> None:
        try:
            await asyncio.wait_for(self._open(url, conn_id), _OPEN_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — any launch failure is reported to the user
            logger.warning("browser tab: cannot open %s: %s", url, exc)
            self._emit({"type": "browserError", "tab_id": "", "text": str(exc)}, conn_id)

    async def _open(self, url: str, conn_id: str) -> str:
        rec = await self._new_tab()
        # Everyone gets the tab; only the surface that asked switches to it.
        if conn_id:
            self._emit(self._open_event(rec, focus=True), conn_id)
        try:
            await rec.page.goto(url, wait_until="commit")
        except Exception as exc:  # noqa: BLE001 — a bad URL must not kill the tab
            self._emit({"type": "browserError", "tab_id": rec.tab_id, "text": str(exc)})
        return rec.tab_id

    async def _open_for_agent(self) -> AgentTab:
        rec = await self._new_tab()
        # The agent wants the user's attention: every surface switches to the tab.
        self._emit(self._open_event(rec, focus=True))
        return AgentTab(rec.tab_id, rec.target_id, self._cdp_url())

    async def _open_for_user(self, url: str) -> None:
        rec = await self._new_tab()
        # The user has to act on this page: every surface switches to the tab.
        self._emit(self._open_event(rec, focus=True))
        try:
            await rec.page.goto(url, wait_until="commit")
        except Exception:
            # A page that cannot load is no hand-off: drop the blank tab
            # so the caller can fall back to another browser.
            await rec.page.close()
            raise

    async def _new_tab(self) -> _PageRecord:
        """Launch the browser if needed, create a page and announce it unfocused."""
        context = await self._launch()
        self._creating += 1
        try:
            page = await context.new_page()
        finally:
            self._creating -= 1
        rec = await self._register(page, popup=False)
        if rec is None:
            raise RuntimeError("The browser closed the new page before it could be attached.")
        return rec

    async def _register(self, page: Page, popup: bool) -> _PageRecord | None:
        """Give *page* a tab id, wire its events and announce it (idempotent).

        Both ``context.new_page()`` in :meth:`_open` and the context's
        ``page`` event call this for the same page, so the record is
        reserved under the lock before the first ``await``.  A page that
        closes (or refuses a CDP session) while being attached is
        forgotten again and never announced; the caller gets ``None``.
        """
        rec, is_new = self._reserve(page)
        if not is_new:
            await rec.attached.wait()
            return rec if rec.cdp is not None else None
        try:
            cdp = await page.context.new_cdp_session(page)
            if page.is_closed():
                raise RuntimeError("page closed while attaching")
            rec.target_id = (await cdp.send("Target.getTargetInfo"))["targetInfo"]["targetId"]
        except Exception as exc:  # noqa: BLE001 — a vanished popup is not an error
            logger.debug("browser tab: %s not attached: %s", rec.tab_id, exc)
            with self._lock:
                self._pages.pop(rec.tab_id, None)
                self._by_page.pop(id(page), None)
            rec.attached.set()
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
            self._pages[tab_id] = rec
            self._by_page[id(page)] = rec
            return rec, True

    def _open_event(
        self, rec: _PageRecord, focus: bool, popup: bool = False
    ) -> dict[str, Any]:
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

    def _on_context_close(self, _context: BrowserContext) -> None:
        """The browser exited (headed window closed, crash): drop every tab."""
        self._schedule(self._teardown(announce=True))

    async def _on_page_closed(self, rec: _PageRecord) -> None:
        with self._lock:
            self._pages.pop(rec.tab_id, None)
            self._by_page.pop(id(rec.page), None)
            remaining = len(self._pages)
        if rec.flush_handle is not None:
            rec.flush_handle.cancel()
        self._emit({"type": "closeBrowserTab", "tab_id": rec.tab_id})
        if remaining == 0:
            await self._teardown(announce=False)

    async def _teardown(self, announce: bool) -> None:
        """Close the browser and forget every page."""
        with self._lock:
            records = list(self._pages.values())
            self._pages.clear()
            self._by_page.clear()
            context, playwright = self._context, self._playwright
            self._context, self._playwright = None, None
        for rec in records:
            if rec.flush_handle is not None:
                rec.flush_handle.cancel()
            if announce:
                self._emit({"type": "closeBrowserTab", "tab_id": rec.tab_id})
        if context is not None:
            try:
                await context.close()
            except Exception as exc:  # noqa: BLE001 — browser may already be gone
                logger.debug("browser tab: context close: %s", exc)
        if playwright is not None:
            await playwright.stop()

    # ------------------------------------------------------------------
    # Navigation and state
    # ------------------------------------------------------------------

    def _record(self, tab_id: str) -> _PageRecord | None:
        with self._lock:
            return self._pages.get(tab_id)

    async def _close(self, tab_id: str) -> None:
        rec = self._record(tab_id)
        if rec is not None:
            await rec.page.close()

    async def _interrupt(self, tab_id: str) -> None:
        rec = self._record(tab_id)
        if rec is None or rec.cdp is None:
            return
        try:
            await rec.cdp.send("Runtime.terminateExecution")
            await asyncio.wait_for(rec.page.evaluate("1"), _INTERRUPT_PROBE_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — still wedged: the tab is lost
            logger.warning("browser tab: %s does not answer, closing it: %s", tab_id, exc)
            await rec.page.close()

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
            rec.viewers[conn_id] = (width, height)
            if rec.page.viewport_size != {"width": width, "height": height}:
                await rec.page.set_viewport_size({"width": width, "height": height})
        else:
            rec.viewers.pop(conn_id, None)
        await self._sync_cast(rec)

    async def _viewer_gone(self, conn_id: str) -> None:
        with self._lock:
            records = list(self._pages.values())
        for rec in records:
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
            self._emit(dict(frame), conn_id)

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
        except Exception as exc:  # noqa: BLE001 — input on a closing page is dropped
            logger.debug("browser tab: input %s: %s", kind, exc)

    def _schedule(self, coro: Any) -> None:
        """Run *coro* from a Playwright callback (already on the private loop)."""
        task = asyncio.ensure_future(coro)
        task.add_done_callback(_log_future_error)


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


def decode_frame(event: dict[str, Any]) -> bytes:
    """Return the JPEG bytes of a ``browserFrame`` event (used by tests and tools)."""
    return base64.b64decode(event["data"])
