# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The victory jingle, end to end: a real page, a real daemon, a registered machine.

The webapp served by a local ``RemoteAccessServer`` is opened in a
real headless Chromium under its normal autoplay policy (no flag): the
user picks the registered machine "box" in the chat header, types a
task and presses Enter, as they would.  "box" is a second daemon with
a fake model whose every answer is a ``finish`` call (see
:mod:`kiss.tests.server.test_remote_tabs`); its ``task_done`` is
relayed to the page, which then plays ``media/victory.mp3``
(``playVictorySound`` in ``main.js``), the typing having been the
gesture the policy wants.

Nothing in the page is replaced.  The proof of playback is Chromium's
own media log (what ``chrome://media-internals`` shows), read over the
DevTools protocol's ``Media`` domain: the player the page created for
the clip's URL goes through ``kPlay`` and reaches ``kEnded``.  A player
that was only created (``new Audio(url)`` preloads the clip) logs its
``kLoad`` but never ``kPlay``, so a fetch alone would not pass.

A second page opened to watch (tabs are the same on every surface, so
it shows the same chat) sounds the next finish too, on its own player.
Last, the jingle muted through the first page's Settings checkbox (the
"..." menu > Settings): no player there for the next task, while the
watching page, unmuted, still sounds it.

The autoplay refusal itself (a ``play()`` rejected with
``NotAllowedError`` on a page without a gesture, sounded at the next
one) is not reachable here: headless Chromium resolves ``play()`` on
an untouched page whatever ``--autoplay-policy`` says (measured on
1243), so that path is exercised by the jsdom suite
``test/victorySound.test.js`` with the rejection the browser would
give.

Under full-suite load Chromium drops requests in flight with
``net::ERR_NETWORK_CHANGED`` whenever a host interface appears or
goes away (other jobs' Docker containers add and remove veth links
all the time).  A navigation that dies that way is retried as the
sync tests do with ``conftest.goto_retrying_network_change``; a
clip download cut short that way is the page's own business: it
fetches the clip again on a fresh player (``startVictoryPlayer``),
so a finish may leave a second clip player that never started
beside the one that played.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from kiss.tests.server.test_remote_tabs import RemoteTabsHarness

CLIP = "/media/victory.mp3?v="


class MediaLog:
    """One page's media players and their events, from the ``Media`` CDP domain."""

    def __init__(self) -> None:
        self.events: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    async def attach(cls, context: BrowserContext, page: Page) -> MediaLog:
        """Start recording *page*'s media log."""
        log = cls()
        cdp = await context.new_cdp_session(page)
        cdp.on("Media.playerEventsAdded", log.on_events)
        await cdp.send("Media.enable")
        return log

    def on_events(self, msg: dict[str, Any]) -> None:
        """Record a ``Media.playerEventsAdded`` notification."""
        player = self.events.setdefault(str(msg["playerId"]), [])
        player.extend(json.loads(ev["value"]) for ev in msg["events"])

    def players_of(self, url_part: str) -> list[str]:
        """The ids of the players that loaded a URL containing *url_part*."""
        return [
            pid for pid, evs in self.events.items()
            if any(ev.get("event") == "kLoad" and url_part in ev.get("url", "") for ev in evs)
        ]

    def names(self, player_id: str) -> list[str]:
        """The event names of *player_id*, in order."""
        return [str(ev.get("event")) for ev in self.events[player_id]]

    def played_to_the_end(self) -> list[str]:
        """The clip players that logged both ``kPlay`` and ``kEnded``.

        Not "``kEnded`` after ``kPlay``": the media pipeline and the
        player report through separate batches, and for the ~0.5 s clip
        the pipeline's ``kEnded`` is often delivered before the player's
        ``kPlay`` (seen on Linux under load).  A preloaded player that
        was never played logs neither.
        """
        return [
            pid for pid in self.players_of(CLIP)
            if {"kPlay", "kEnded"} <= set(self.names(pid))
        ]

    async def wait_played(self, count: int, timeout: float = 15.0) -> None:
        """Wait until *count* clip players have played to their end."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.played_to_the_end()) >= count:
                return
            await asyncio.sleep(0.1)
        raise AssertionError(f"{count} clip(s) never played to the end: {self.events}")


class TestVictorySoundLive(RemoteTabsHarness):
    """The served page plays the jingle to its end when its remote task finishes."""

    async def _open_page(self, browser: Browser) -> tuple[Page, MediaLog]:
        """A fresh, untouched page on the served webapp and its media log.

        The navigation is made up to three times while it fails with
        ``net::ERR_NETWORK_CHANGED`` (see the module docstring): a
        ``goto`` raising it, or a page whose script died of it and
        whose self-reload (``media/assetLoadGuard.js``) was cut the
        same way, so it ends on Chromium's error page or never shows
        its input.  Any other failure propagates unchanged.
        """
        context = await browser.new_context(
            ignore_https_errors=True, service_workers="block",
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()
        log = await MediaLog.attach(context, page)
        failures: list[str] = []

        def note_failed_request(request: Any) -> None:
            failures.append(f"{request.url} ({request.resource_type}): {request.failure}")

        page.on("requestfailed", note_failed_request)
        url = f"https://127.0.0.1:{self.server.port}/"
        for attempt in range(3):
            failures.clear()
            try:
                await page.goto(url)
                if page.url.startswith("chrome-error://"):
                    raise PlaywrightError(f"{url} ended on {page.url}")
                await page.wait_for_selector("#task-input", state="visible", timeout=30000)
            except PlaywrightError as exc:
                network_changed = "net::ERR_NETWORK_CHANGED" in str(exc) or any(
                    "net::ERR_NETWORK_CHANGED" in failure for failure in failures
                )
                if not network_changed or attempt == 2:
                    raise AssertionError(
                        f"{url}: {exc} (page at {page.url}; failed requests: {failures})",
                    ) from exc
                await asyncio.sleep(1.0)
                continue
            break
        page.remove_listener("requestfailed", note_failed_request)
        await page.wait_for_function(
            "() => [...document.querySelectorAll('#chat-machine option')]"
            ".some(o => o.value === 'box')",
            timeout=30000,
        )
        return page, log

    async def _type_task_for_box(self, page: Page, tab: str, prompt: str) -> None:
        """Pick "box", type *prompt*, press Enter, and wait for the run to end.

        The end is the daemon's own ``status running:false`` for *tab*
        on the harness connection, which addressed no tab and so gets
        every tab's frames (``_local_clients_for_tab``).
        """
        await page.select_option("#chat-machine", "box")
        await page.click("#task-input")
        await page.keyboard.type(prompt)
        await page.keyboard.press("Enter")
        await self._run_until_idle(tab)

    def _assert_sounded(self, log: MediaLog, count: int, url: str) -> None:
        """*count* finishes sounded on *log*'s page, each once.

        *count* clip players played to their end, every one of them
        created for the clip's *url*; any other clip player never
        started (``kLoad`` without ``kPlay``): its download was cut
        short (``ERR_NETWORK_CHANGED``, see the module docstring) and
        the page fetched the clip again on one of those that played.
        """
        played = log.played_to_the_end()
        self.assertEqual(len(played), count, log.events)
        for pid in log.players_of(CLIP):
            loaded = next(ev for ev in log.events[pid] if ev.get("event") == "kLoad")
            self.assertEqual(loaded["url"], f"https://127.0.0.1:{self.server.port}{url}")
            if pid not in played:
                self.assertNotIn("kPlay", log.names(pid), log.events)

    async def test_remote_task_finish_plays_the_jingle(self) -> None:
        """Typed task on "box": one clip player, played to its end under the
        browser's own autoplay policy; a second surface sounds it too;
        muted in Settings, no player on that page."""
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page, log = await self._open_page(browser)
                tab = str(await page.evaluate("window._testApi.getActiveTabId()"))
                self.assertTrue(tab)
                url = str(await page.evaluate("window.__VOICE__.victoryAudioUrl"))
                self.assertTrue(url.startswith(CLIP), url)
                self.assertTrue(
                    await page.evaluate("document.getElementById('cfg-victory-sound').checked"),
                    "the jingle is on by default",
                )

                # 1. The typed task: the typing is the gesture the policy wants.
                await self._type_task_for_box(page, tab, "Say hello")
                await log.wait_played(1)
                self._assert_sounded(log, 1, url)

                # 2. A page opened to watch shows the same chat (tabs are
                #    shared across surfaces) and sounds the next finish
                #    on its own player, once.
                watcher, watch_log = await self._open_page(browser)
                await self._type_task_for_box(page, tab, "Say hello again")
                await log.wait_played(2)
                await watch_log.wait_played(1)
                self._assert_sounded(log, 2, url)
                self._assert_sounded(watch_log, 1, url)

                # 3. Muted on the first page, the way the user does it:
                #    "..." > Settings > the box.  Per page: the watcher
                #    still sounds the finish.
                await page.click("#more-btn")
                await page.click("#settings-btn")
                await page.click("#cfg-victory-sound")
                self.assertEqual(
                    await page.evaluate("localStorage.getItem('kissVictorySound')"), "off",
                )
                await page.click("#settings-panel-close")
                muted_players = len(log.players_of(CLIP))
                await self._type_task_for_box(page, tab, "Say hello a third time")
                await watch_log.wait_played(2)
                self._assert_sounded(watch_log, 2, url)
                self.assertEqual(len(log.players_of(CLIP)), muted_players, "muted: no new player")
            finally:
                await browser.close()
