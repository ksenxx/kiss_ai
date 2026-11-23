# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the Settings button of the remote webapp, and the composer's
"..." menu it left.

The Settings control used to be a gear pill that ``renderTabBar()``
appended to ``#tab-bar`` (and once regressed into rendering as a tiny
dot there), then an item of the input footer's "..." overflow menu, then
a button of the strip under the Chats list.  It now is the first icon
button of the strip under the task-info panel (``#meta-footer`` /
``#settings-btn`` in ``chat.html``), left of the Terminal button; the
menu (``#more-btn`` / ``#more-menu``) keeps mic, share, attach and the
Home Screen item.

Static tests pin the CSS wiring (the menu surface and item rules exist
in ``remote-codex.css``); the live test boots the production
``RemoteAccessServer`` + headless Chromium and asserts the Settings
button renders on the strip at a clickable size with a visible gear
icon, and that clicking it opens the settings panel and closes an open
"..." menu.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.design_tokens import inline_design_tokens

MEDIA_DIR = (
    Path(__file__).resolve().parents[2] / "webapp" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"


def _find_rule(css: str, selector: str) -> str:
    """Union of declaration bodies of every ``body.remote-chat``-scoped
    rule for *selector*, or fail."""
    pattern = (
        r"body\.remote-chat[^{,]*"
        + re.escape(selector)
        + r"\s*(?:,[^{]*)?\{([^}]*)\}"
    )
    bodies = re.findall(pattern, css)
    assert bodies, f"body.remote-chat scoped rule for {selector!r} missing"
    return "\n".join(bodies)


def test_remote_codex_defines_more_menu_surface() -> None:
    """remote-codex.css must restyle the "..." menu as a floating
    surface (rounded, elevated, on the remote palette)."""
    css = CODEX_CSS.read_text(encoding="utf-8")
    rule = _find_rule(css, "#more-menu")
    assert "border-radius" in rule, rule
    assert "background" in rule, rule
    assert "box-shadow" in rule, rule


def test_remote_codex_styles_more_menu_items() -> None:
    """remote-codex.css must give the menu items comfortable touch
    padding and a hover treatment."""
    css = inline_design_tokens(CODEX_CSS.read_text(encoding="utf-8"))
    rule = _find_rule(css, ".more-menu-item")
    m_pad = re.search(r"padding:\s*(\d+)px", rule)
    assert m_pad and int(m_pad.group(1)) >= 6, (
        ".more-menu-item needs touch-friendly padding; " f"got: {rule!r}"
    )
    hover = _find_rule(css, ".more-menu-item:hover:not(:disabled)")
    assert "background" in hover, hover


def _start_live_server(
    tmp_path: Path,
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Boot the production ``RemoteAccessServer`` until *done* is set.

    Stores the bound ephemeral port in ``state['port']`` (or the
    startup exception in ``state['error']``) and sets *ready*.
    """
    from kiss.server.web_server import RemoteAccessServer
    from kiss.tests.local_ws import _generate_self_signed_cert

    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            local_endpoint_file=tmp_path / "sorcar-local.json",
        )
        started = False
        try:
            await server.start_async()
            started = True
            assert server._ws_server is not None
            state["port"] = next(
                iter(server._ws_server.sockets)
            ).getsockname()[1]
            ready.set()
            while not done.is_set():
                await asyncio.sleep(0.02)
        except BaseException as exc:  # pragma: no cover - defensive
            state["error"] = exc
            ready.set()
        finally:
            if started:
                await server.stop_async()


    asyncio.run(scenario())


_MEASURE_JS = r"""
(() => {
  const menu = document.getElementById('more-menu');
  const item = document.getElementById('settings-btn');
  const svg = item ? item.querySelector('svg') : null;
  if (!menu || !item || !svg) {
    return {hasMenu: !!menu, hasItem: !!item, hasSvg: !!svg};
  }
  const itemRect = item.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();
  const hit = document.elementFromPoint(
    itemRect.left + itemRect.width / 2,
    itemRect.top + itemRect.height / 2,
  );
  return {
    hasMenu: true,
    hasItem: true,
    hasSvg: true,
    menuOpen: menu.classList.contains('open'),
    onStrip: !!item.closest('#meta-footer'),
    nextId: item.nextElementSibling ? item.nextElementSibling.id : '',
    inMenu: menu.contains(item),
    reachable: !!hit && item.contains(hit),
    itemWidth: itemRect.width,
    itemHeight: itemRect.height,
    svgWidth: svgRect.width,
    svgHeight: svgRect.height,
    label: item.getAttribute('aria-label'),
  };
})()
"""


@pytest.mark.timeout(180)
def test_live_settings_button_on_the_task_info_strip(
    tmp_path: Path,
) -> None:
    """On the live remote page the Settings button of the task-info strip
    is a real clickable icon button with a visible gear, left of the
    Terminal button; clicking it opens the settings panel and closes the
    "..." menu left open."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server,
        args=(tmp_path, ready, done, state),
        daemon=True,
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError(
                "RemoteAccessServer startup failed"
            ) from startup_error
        port = state["port"]

        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--ignore-certificate-errors"]
            )
            try:
                page = browser.new_page(
                    ignore_https_errors=True,
                    viewport={"width": 1400, "height": 900},
                )
                goto_retrying_network_change(
                    page,
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#more-btn", state="attached")
                page.wait_for_timeout(200)
                page.click("#more-btn")
                page.wait_for_selector("#more-menu.open", state="visible")
                measured = page.evaluate(_MEASURE_JS)
                page.click("#settings-btn")
                page.wait_for_selector(
                    "#settings-panel.open", state="attached"
                )
                after = page.evaluate(_MEASURE_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
    thread_error = state.get("error")
    if isinstance(thread_error, BaseException):
        raise AssertionError(
            "RemoteAccessServer thread failed"
        ) from thread_error

    assert measured["hasMenu"] and measured["hasItem"], repr(measured)
    assert measured["menuOpen"], (
        'clicking "..." must open the overflow menu; ' + repr(measured)
    )
    assert measured["hasSvg"], (
        "the Settings item must contain a gear SVG child; "
        + repr(measured)
    )
    assert measured["svgWidth"] >= 14 and measured["svgHeight"] >= 14, (
        "settings gear SVG must render at ~16x16; got: " + repr(measured)
    )
    assert measured["onStrip"] and not measured["inMenu"], (
        "Settings lives on the strip under the task-info panel; " + repr(measured)
    )
    assert measured["nextId"] == "terminal-btn", (
        "Settings stands left of the Terminal button; " + repr(measured)
    )
    assert measured["itemWidth"] == measured["itemHeight"] >= 28, (
        "settings button must be a square, clickable icon button; got: "
        + repr(measured)
    )
    assert measured["reachable"], (
        "nothing may cover the settings button; " + repr(measured)
    )
    assert measured["label"] == "Settings", repr(measured)
    assert not after["menuOpen"], (
        "a click on Settings must close the open menu; " + repr(after)
    )
