# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the tab-bar Settings gear button on the remote webapp.

Bug: on the remote webapp the tab-bar's Settings button (the gear
icon appended by ``renderTabBar()`` in ``main.js``) renders as a tiny
dot instead of the 18x18 gear the VS Code extension shows.

Root cause: ``main.css`` caps ``.chat-tab-settings`` at
``min/max-width: 30px`` with ``padding: 4px 8px 4px 0`` (~22px of
content space for the 18x18 SVG).  ``remote-codex.css`` has a higher-
specificity ``body.remote-chat #tab-bar .chat-tab`` rule that
overrides the padding to ``5px 12px``, collapsing the 30px-wide flex
item to ~6px of content and shrinking the SVG flex-child to a dot.

Fix: ``remote-codex.css`` adds a ``body.remote-chat #tab-bar
.chat-tab-settings`` pill rule (mirroring the ``.chat-tab-add``
treatment) with enough padding/width for the gear, plus an SVG rule
that pins the child SVG's intrinsic 18x18 size and disables
``flex-shrink``.

Static tests pin the CSS wiring; the live test boots the production
``RemoteAccessServer`` + headless Chromium and asserts that
``renderTabBar()``'s Settings tab and its gear SVG render at a
plausible clickable size (not "a dot").
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


# ── Static: remote-codex.css restores gear geometry ─────────────────


def test_remote_codex_defines_chat_tab_settings_pill() -> None:
    """remote-codex.css must define an explicit .chat-tab-settings
    override so the .chat-tab pill padding cannot collapse the gear."""
    css = CODEX_CSS.read_text(encoding="utf-8")
    rule = _find_rule(css, ".chat-tab-settings")
    # Pill styling to visually match the .chat-tab-add pill.
    assert "border-radius: 999px" in rule, (
        ".chat-tab-settings must be a pill on the remote webapp; "
        f"got: {rule!r}"
    )
    # Restore enough content width for the 18x18 SVG.
    m_min = re.search(r"min-width:\s*(\d+)px", rule)
    m_max = re.search(r"max-width:\s*(\d+)px", rule)
    assert m_min and m_max, (
        ".chat-tab-settings must pin min/max width so the gear has "
        f"room; got: {rule!r}"
    )
    assert int(m_min.group(1)) >= 32, rule
    assert int(m_max.group(1)) >= 32, rule


def test_remote_codex_pins_settings_svg_size() -> None:
    """remote-codex.css must pin the child SVG's intrinsic 18x18 size
    and stop the flex layout from shrinking it."""
    css = CODEX_CSS.read_text(encoding="utf-8")
    rule = _find_rule(css, ".chat-tab-settings svg")
    assert "width: 18px" in rule, rule
    assert "height: 18px" in rule, rule
    assert "flex-shrink: 0" in rule, (
        "the gear SVG must not be shrunk by the flex layout; "
        f"got: {rule!r}"
    )


def test_remote_codex_settings_rule_comes_after_generic_chat_tab() -> None:
    """The new .chat-tab-settings rule must live LATER in the file
    than the generic ``body.remote-chat #tab-bar .chat-tab`` rule so
    the cascade (equal specificity across id + 2 classes) resolves in
    the settings rule's favor."""
    css = CODEX_CSS.read_text(encoding="utf-8")
    generic = css.find("body.remote-chat #tab-bar .chat-tab {")
    settings = css.find("body.remote-chat #tab-bar .chat-tab-settings {")
    assert generic != -1, "generic .chat-tab pill rule missing"
    assert settings != -1, ".chat-tab-settings pill rule missing"
    assert generic < settings, (
        "the .chat-tab-settings override must appear after the "
        "generic .chat-tab rule in remote-codex.css"
    )


# ── Live: RemoteAccessServer + real Chromium ────────────────────────


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
    from kiss.agents.vscode.web_server import (
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
  const tab = document.querySelector('.chat-tab-settings');
  const svg = tab ? tab.querySelector('svg') : null;
  if (!tab || !svg) {
    return {
      hasTab: !!tab,
      hasSvg: !!svg,
    };
  }
  const tabRect = tab.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();
  const cs = getComputedStyle(tab);
  const svgCs = getComputedStyle(svg);
  return {
    hasTab: true,
    hasSvg: true,
    tabWidth: tabRect.width,
    tabHeight: tabRect.height,
    svgWidth: svgRect.width,
    svgHeight: svgRect.height,
    tabDisplay: cs.display,
    tabBorderRadius: cs.borderTopLeftRadius,
    svgWidthCss: svgCs.width,
    svgHeightCss: svgCs.height,
    svgFlexShrink: svgCs.flexShrink,
    tabVisible: tabRect.width > 0 && tabRect.height > 0,
  };
})()
"""


@pytest.mark.timeout(180)
def test_live_remote_tab_settings_button_renders_gear(
    tmp_path: Path,
) -> None:
    """The remote page's tab-bar Settings tab is a real clickable pill
    that contains an 18x18 SVG — not a dot."""
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
                page.wait_for_selector("#tab-bar", state="attached")
                # init() runs renderTabBar() at boot, which appends
                # .chat-tab-add and .chat-tab-settings to #tab-bar.
                page.wait_for_selector(
                    ".chat-tab-settings", state="attached"
                )
                page.wait_for_selector(
                    ".chat-tab-settings svg", state="attached"
                )
                # Let layout settle after the injected tab render.
                page.wait_for_timeout(200)
                measured = page.evaluate(_MEASURE_JS)
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

    assert measured["hasTab"], (
        "renderTabBar() must append .chat-tab-settings to #tab-bar; "
        + repr(measured)
    )
    assert measured["hasSvg"], (
        ".chat-tab-settings must contain a gear SVG child; "
        + repr(measured)
    )
    # The gear SVG must render at its intrinsic 18x18 size — not
    # collapsed to a dot by the outer flex container.
    assert measured["svgWidth"] >= 16 and measured["svgHeight"] >= 16, (
        "settings gear SVG must render at ~18x18; got: "
        + repr(measured)
    )
    assert measured["svgWidthCss"] == "18px", measured
    assert measured["svgHeightCss"] == "18px", measured
    assert measured["svgFlexShrink"] == "0", (
        "the gear SVG must be flex-shrink:0 so it cannot be squeezed "
        + repr(measured)
    )
    # The clickable pill must be roomy enough to hit reliably.
    assert measured["tabWidth"] >= 32, (
        "settings pill must be a clickable size; got: " + repr(measured)
    )
    assert measured["tabHeight"] >= 22, (
        "settings pill must be tall enough to hit; got: "
        + repr(measured)
    )
