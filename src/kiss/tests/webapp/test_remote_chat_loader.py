# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
# test_explorer_scm_commands and re-exported for pytest's collector)
"""E2E: a remote machine's chat says it is on its way while it is fetched.

A click on a CHATS-panel row of a remote machine opens a tab that shows
the task's text at once, but its transcript only after the daemon has
connected to the machine and the machine has replayed the chat, a few
round trips later.  Until then the tab shows "Loading the chat from
<machine>..." under the task panel; the line goes with the answer's
first frame (the tab's status), the transcript, or the error of a
machine that could not be reached.

The real remote webapp (an in-process ``RemoteAccessServer``) with a
real second daemon (:mod:`kiss.tests.remote_daemon`) registered as
machine ``box`` behind a TCP proxy that delays every byte, so the
fetch takes long enough to watch; driven by headless Chromium.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.agents.sorcar import machines as mach
from kiss.tests.conftest import find_free_port
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    harness,  # noqa: F401  (module fixture)
)

DELAY = 0.3
"""One-way delay of the proxy, seconds."""


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


class DelayProxy:
    """A TCP proxy to ``host:port`` adding DELAY seconds each way, on its own loop thread."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.local_port = find_free_port()
        self.loop = asyncio.new_event_loop()
        self.writers: set[asyncio.StreamWriter] = set()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="delay-proxy")
        self.thread.start()
        self.started.wait(10)

    async def _pump(self, src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                await asyncio.sleep(DELAY)
                dst.write(data)
                await dst.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            dst.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(self.host, self.port)
        except OSError:
            # Nothing listens behind the proxy (the daemon stopped): the
            # peer sees the connection dropped, as a dead host's would be.
            writer.close()
            return
        self.writers.update((writer, upstream_writer))
        await asyncio.gather(
            self._pump(reader, upstream_writer), self._pump(upstream_reader, writer),
        )
        self.writers.difference_update((writer, upstream_writer))

    async def _serve(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", self.local_port)
        self.started.set()
        try:
            await self.server.serve_forever()
        except asyncio.CancelledError:
            pass
        # The relays of the connections still open end with their
        # sockets (stop closed them): wait for them rather than leave
        # them pending when the loop closes.
        relays = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*relays, return_exceptions=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        finally:
            self.loop.close()

    def _close(self) -> None:
        """Close every relayed connection, then the listener (on the proxy's loop)."""
        for writer in list(self.writers):
            writer.close()
        self.server.close()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self._close)
        self.thread.join(10)


class Box:
    """The remote daemon ``box`` (behind the delay proxy) and its seeded chat."""

    def __init__(self, harness: ExplorerHarness) -> None:
        root = Path(harness.tmpdir)
        self.home = root / "box-home"
        self.daemon: RemoteDaemon = start_remote_daemon(
            self.home, password="pw-box",
            # One chat per test: a tab opened for a chat stays in the
            # daemon's registry after its page is gone, and a later page
            # adopts it, so a second click on the same chat would show
            # that tab instead of fetching the chat.
            seed=[
                {"chat_id": f"box-chat-{n}", "task": f"Box task {n}", "result": "done"}
                for n in (1, 2)
            ],
        )
        target = urlsplit(self.daemon.url)
        self.proxy = DelayProxy(str(target.hostname), int(target.port or 0))
        url = self.daemon.url.replace(f":{target.port}", f":{self.proxy.local_port}")
        mach.save_machines([])
        mach.add_machine("box", url, "pw-box")

    def stop(self) -> None:
        self.proxy.stop()
        self.daemon.stop()
        mach.save_machines([])


@pytest.fixture(scope="module")
def box(harness: ExplorerHarness):
    b = Box(harness)
    yield b
    b.stop()


def _open(browser: Browser, harness: ExplorerHarness) -> Page:
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": 1280, "height": 900},
        service_workers="block",
    )
    context.add_init_script("localStorage.setItem('kissTipsOptOut', '1');")
    page = context.new_page()
    page.goto(harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector(
        "#history-list .sidebar-item[data-machine='box']", state="attached", timeout=30000,
    )
    return page


def _click_box_chat(page: Page, chat_id: str = "box-chat-1") -> None:
    group = page.locator(
        f"#history-list .history-chat-group[data-chat-id='{chat_id}']",
        has=page.locator(".sidebar-item[data-machine='box']"),
    )
    group.locator(".history-chat-header").click(timeout=20000)


def test_the_tab_says_the_chat_is_on_its_way_until_it_arrives(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        t0 = time.monotonic()
        _click_box_chat(page)
        loader = page.locator("#output #remote-chat-loader")
        loader.wait_for(state="visible", timeout=5000)
        assert loader.text_content() == "Loading the chat from box\u2026"
        # Under the task panel, not in its place.
        assert page.locator("#output .task-panel").count() == 1
        # The answer takes at least the proxy's round trips; then the
        # line is gone and the chat is bound to box.
        loader.wait_for(state="detached", timeout=30000)
        assert time.monotonic() - t0 > 2 * DELAY
        page.wait_for_function(
            "() => document.getElementById('chat-machine').value === 'box'", timeout=20000,
        )
        assert page.locator("#output #remote-chat-loader").count() == 0
    finally:
        page.context.close()


def test_the_line_goes_with_the_error_of_a_machine_out_of_reach(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    box.daemon.stop()
    try:
        _click_box_chat(page, "box-chat-2")
        error = page.locator("#output .ev.err")
        error.first.wait_for(state="visible", timeout=30000)
        assert "box" in str(error.first.text_content())
        assert page.locator("#output #remote-chat-loader").count() == 0
    finally:
        page.context.close()
