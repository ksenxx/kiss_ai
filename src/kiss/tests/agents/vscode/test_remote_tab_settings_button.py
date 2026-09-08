# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the Settings entry of the composer's "..." menu on the
remote webapp.

The Settings control used to be a gear pill that ``renderTabBar()``
appended to ``#tab-bar`` (and once regressed into rendering as a tiny
dot there).  It now lives as a labelled item inside the input footer's
"..." overflow menu (``#more-btn`` / ``#more-menu`` / ``#settings-btn``
in ``chat.html``), together with mic, share, attach, Git Commit and the
remote-only theme toggle.

Static tests pin the CSS wiring (the menu surface and item rules exist
in ``remote-codex.css``, and the theme toggle stays hidden outside the
remote webapp); the live test boots the production
``RemoteAccessServer`` + headless Chromium, opens the "..." menu and
asserts the Settings item renders at a clickable size with a visible
gear icon, and that clicking it opens the settings panel and closes the
menu.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
MAIN_CSS = MEDIA_DIR / "main.css"


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
    css = CODEX_CSS.read_text(encoding="utf-8")
    rule = _find_rule(css, ".more-menu-item")
    m_pad = re.search(r"padding:\s*(\d+)px", rule)
    assert m_pad and int(m_pad.group(1)) >= 6, (
        ".more-menu-item needs touch-friendly padding; " f"got: {rule!r}"
    )
    hover = _find_rule(css, ".more-menu-item:hover:not(:disabled)")
    assert "background" in hover, hover


def test_main_css_hides_theme_toggle_outside_remote() -> None:
    """The theme toggle is remote-webapp-only: main.css must hide
    ``#theme-btn`` by default and unhide it under ``body.remote-chat``
    (the VS Code webview always follows the editor theme)."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    assert re.search(r"#theme-btn\s*\{\s*display:\s*none;", css), (
        "#theme-btn must be hidden outside the remote webapp"
    )
    assert re.search(
        r"body\.remote-chat\s+#theme-btn\s*\{\s*display:\s*flex;", css
    ), "#theme-btn must be shown on the remote webapp"


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
    from kiss.server.web_server import (
        RemoteAccessServer,
        _generate_self_signed_cert,
    )

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
            uds_path=tmp_path / "sorcar.sock",
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
  return {
    hasMenu: true,
    hasItem: true,
    hasSvg: true,
    menuOpen: menu.classList.contains('open'),
    itemWidth: itemRect.width,
    itemHeight: itemRect.height,
    svgWidth: svgRect.width,
    svgHeight: svgRect.height,
    label: (item.textContent || '').trim(),
  };
})()
"""


@pytest.mark.timeout(180)
def test_live_remote_more_menu_settings_item(
    tmp_path: Path,
) -> None:
    """On the live remote page the "..." button opens the overflow menu,
    whose Settings item is a real clickable row with a visible gear
    icon; clicking it opens the settings panel and closes the menu."""
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
                page.goto(
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
    assert measured["itemWidth"] >= 100 and measured["itemHeight"] >= 24, (
        "settings item must be a clickable row; got: " + repr(measured)
    )
    assert "Settings" in str(measured["label"]), repr(measured)
    assert not after["menuOpen"], (
        "activating the Settings item must close the menu; " + repr(after)
    )
