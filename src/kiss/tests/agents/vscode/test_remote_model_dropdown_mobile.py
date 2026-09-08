# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the mobile model picker is fully usable.

Two regressions on the remote webapp at phone widths, driven against
the production ``RemoteAccessServer`` + a real headless Chromium:

* ``#model-dropdown`` anchors to the model pill's right edge (CSS
  ``right: 0``). On a narrow phone the pill sits mid-row, so a 280px+
  dropdown used to poke past the LEFT viewport edge (185px off screen
  at 320px width) and the model list was clipped.  ``positionModelDD``
  in main.js now shifts the open dropdown right just enough to keep it
  fully inside the viewport, re-fitting on every re-render (search
  filtering, ``models`` refreshes).

* ``#model-name`` used to truncate from the END, hiding exactly the
  distinctive part of a long model name (variant/size/date). The pill
  now renders as an RTL line with LRM-pinned LTR text, so a truncated
  name keeps its END visible and the ellipsis sits at the START.

Both tests inject the model list the way production does — a
``models`` message through ``window.postMessage`` (the WebSocket shim
delivers daemon events exactly like that) — and click the real pill.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

# The served page boots with #app hidden until the daemon connection
# reveals it, and — with no daemon behind this harness — pops the
# remote-password modal, which intercepts pointer events. Neither is
# this layout test's business: reveal the app the way
# setServerLoading(false) does and keep the auth modal out of the way.
_REVEAL_APP_JS = """
() => {
  const overlay = document.getElementById('kiss-server-loading');
  if (overlay) overlay.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
  const auth = document.getElementById('auth-modal');
  if (auth) {
    auth.style.display = 'none';
    new MutationObserver(() => {
      if (auth.style.display !== 'none') auth.style.display = 'none';
    }).observe(auth, {attributes: true});
  }
}
"""

_MODEL_NAMES = [
    "claude-ultra-long-model-name-with-many-qualifiers-20261199-128k",
    "claude-fable-5",
    "claude-opus-4-5-20260114",
    "gpt-5.6-sol",
    "gemini-3.5-pro-preview-super-long-name",
    "grok-4.5-fast",
    "deepseek-reasoner-v4",
    "o5-mini-high",
    "llama-4.5-maverick",
    "mistral-large-3",
    "qwen3-max-instruct",
    "kimi-k3-thinking",
    "claude-sonnet-4-8",
    "gpt-5.6-codex-sol",
    "nova-premier-2",
]

_LONG_SELECTED = "claude-opus-4-5-20260114-thinking"

_INJECT_MODELS_JS = f"""
() => {{
  const names = {_MODEL_NAMES!r};
  window.postMessage({{type: 'models',
    models: names.map(n => ({{name: n, vendor: 'Anthropic', inp: 1,
                              out: 5, uses: 0}})),
    selected: {_LONG_SELECTED!r}}}, '*');
}}
"""

_DROPDOWN_GEOMETRY_JS = """
() => {
  const r = document.getElementById('model-dropdown')
    .getBoundingClientRect();
  return {
    left: r.left, right: r.right, top: r.top, bottom: r.bottom,
    vw: window.innerWidth, vh: window.innerHeight,
    items: document.querySelectorAll('#model-list .model-item').length,
  };
}
"""

# Horizontal-scroll geometry of the model list: with overflow-x hidden
# and every name shrinking inside its row (.model-item-name), the list
# must never be horizontally scrollable, and a long name must ellipsize
# from the START (its end character stays inside the row box).
_LIST_GEOMETRY_JS = """
() => {
  const list = document.getElementById('model-list');
  const dd = document.getElementById('model-dropdown');
  const rows = Array.from(
    document.querySelectorAll('#model-list .model-item'));
  const longRow = rows.find(r =>
    r.textContent.includes('claude-ultra-long-model-name'));
  const name = longRow.querySelector('.model-item-name');
  const tn = name.firstChild;
  const at = (i) => {
    const r = document.createRange();
    r.setStart(tn, i); r.setEnd(tn, i + 1);
    return r.getBoundingClientRect();
  };
  const nameBox = name.getBoundingClientRect();
  return {
    listClientWidth: list.clientWidth,
    listScrollWidth: list.scrollWidth,
    listOverflowX: getComputedStyle(list).overflowX,
    ddWidth: dd.getBoundingClientRect().width,
    maxRowWidth: Math.max(
      ...rows.map(r => r.getBoundingClientRect().width)),
    nameClipped: name.scrollWidth > name.clientWidth,
    nameBoxLeft: nameBox.left, nameBoxRight: nameBox.right,
    firstLeft: at(1).left,
    lastRight: at(tn.length - 2).right,
    priceVisible: longRow.querySelector('.model-cost')
      .getBoundingClientRect().right <= dd.getBoundingClientRect().right,
  };
}
"""

# Character-level geometry of the pill label: which end of the name
# survives the truncation. The label text is wrapped in two U+200E
# marks, so the first/last name characters sit at offsets 1 and
# length-2 of the text node.
_PILL_GEOMETRY_JS = """
() => {
  const name = document.getElementById('model-name');
  const tn = name.firstChild;
  const box = name.getBoundingClientRect();
  const at = (i) => {
    const r = document.createRange();
    r.setStart(tn, i); r.setEnd(tn, i + 1);
    return r.getBoundingClientRect();
  };
  const first = at(1);
  const last = at(tn.length - 2);
  return {
    text: tn.textContent.replace(/\\u200e/g, ''),
    boxLeft: box.left, boxRight: box.right,
    firstLeft: first.left, firstRight: first.right,
    lastLeft: last.left, lastRight: last.right,
    clientWidth: name.clientWidth, scrollWidth: name.scrollWidth,
  };
}
"""


def _start_live_server(
    tmp_path: Path,
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Run the production RemoteAccessServer until *done* is set.

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
    # macOS caps AF_UNIX paths at 104 bytes; pytest's tmp_path can
    # exceed that, so the socket gets its own short temp name.
    uds_path = Path(tempfile.gettempdir()) / f"kmdd-{uuid.uuid4().hex[:8]}.sock"

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            uds_path=uds_path,
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
            uds_path.unlink(missing_ok=True)

    asyncio.run(scenario())


def _open_dropdown(page) -> None:
    """Deliver the model list and open the dropdown via the real pill."""
    page.wait_for_selector("#model-btn", state="attached")
    page.evaluate(_REVEAL_APP_JS)
    page.wait_for_timeout(200)
    page.evaluate(_INJECT_MODELS_JS)
    page.wait_for_timeout(200)
    page.click("#model-btn")
    page.wait_for_selector("#model-dropdown.open", state="visible")
    page.wait_for_timeout(100)


def _assert_on_screen(
    geo: dict, label: str, items: int, margin: float = 8.0
) -> None:
    assert geo["items"] == items, (
        f"{label}: expected {items} rendered model rows; got {geo!r}"
    )
    assert geo["left"] >= margin, (
        f"{label}: dropdown clipped at the left viewport edge: {geo!r}"
    )
    assert geo["right"] <= geo["vw"] - margin, (
        f"{label}: dropdown clipped at the right viewport edge: {geo!r}"
    )
    assert geo["top"] >= 0 and geo["bottom"] <= geo["vh"], (
        f"{label}: dropdown clipped vertically: {geo!r}"
    )


@pytest.mark.timeout(180)
def test_mobile_model_dropdown_fully_visible(tmp_path: Path) -> None:
    """Served page + real Chromium at phone widths: the opened model
    dropdown sits fully inside the viewport (no clipping), including
    after a search-driven re-render, at 390px and at a cramped 320px."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server,
        args=(tmp_path, ready, done, state),
        daemon=True,
    )
    thread.start()
    geos: dict[str, dict] = {}
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError(
                "RemoteAccessServer startup failed"
            ) from startup_error
        port = state["port"]

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                for width, height in [(390, 844), (320, 700)]:
                    page = browser.new_page(
                        ignore_https_errors=True,
                        viewport={"width": width, "height": height},
                    )
                    page.goto(
                        f"https://127.0.0.1:{port}/",
                        wait_until="domcontentloaded",
                    )
                    _open_dropdown(page)
                    geos[f"{width}px"] = page.evaluate(_DROPDOWN_GEOMETRY_JS)
                    # A search re-renders (and re-fits) the open list.
                    page.fill("#model-search", "claude")
                    page.wait_for_timeout(100)
                    geos[f"{width}px filtered"] = page.evaluate(
                        _DROPDOWN_GEOMETRY_JS
                    )
                    page.close()
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    claude_rows = sum("claude" in n for n in _MODEL_NAMES)
    for label in ("390px", "320px"):
        _assert_on_screen(geos[label], label, items=len(_MODEL_NAMES))
        _assert_on_screen(
            geos[f"{label} filtered"],
            f"{label} after search re-render",
            items=claude_rows,
        )


@pytest.mark.timeout(180)
def test_model_pill_truncates_from_start(tmp_path: Path) -> None:
    """A pill label too long for the capped mobile pill keeps the END
    of the model name visible and hides the START (leading ellipsis);
    a digit-ending name keeps left-to-right character order."""
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
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                page = browser.new_page(
                    ignore_https_errors=True,
                    viewport={"width": 390, "height": 844},
                )
                page.goto(
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#model-btn", state="attached")
                page.evaluate(_REVEAL_APP_JS)
                page.wait_for_timeout(200)
                page.evaluate(_INJECT_MODELS_JS)
                page.wait_for_timeout(200)
                overflowing = page.evaluate(_PILL_GEOMETRY_JS)

                # Pick a short, digit-ending name through the real
                # dropdown: no truncation, and — the bidi trap of the
                # RTL-line trick — the characters must stay in
                # left-to-right order ("claude-fable-5", never
                # "5-claude-fable").
                page.click("#model-btn")
                page.wait_for_selector(
                    "#model-dropdown.open", state="visible"
                )
                page.click(".model-item:has-text('claude-fable-5')")
                page.wait_for_timeout(100)
                # The 390px pill caps at ~34px of text, so even this
                # short name truncates there; widen the window until
                # the pill fits it whole for the character-order check.
                page.set_viewport_size({"width": 800, "height": 844})
                page.wait_for_timeout(100)
                digits = page.evaluate(_PILL_GEOMETRY_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert overflowing["text"] == _LONG_SELECTED
    assert overflowing["scrollWidth"] > overflowing["clientWidth"], (
        f"the long name must overflow the capped pill: {overflowing!r}"
    )
    assert overflowing["lastRight"] <= overflowing["boxRight"] + 1, (
        f"the END of a truncated model name must stay visible: "
        f"{overflowing!r}"
    )
    assert overflowing["firstLeft"] < overflowing["boxLeft"], (
        f"the START of a truncated model name must be the clipped "
        f"part (leading ellipsis): {overflowing!r}"
    )

    assert digits["text"] == "claude-fable-5"
    assert digits["scrollWidth"] <= digits["clientWidth"] + 1, (
        f"'claude-fable-5' must fit the pill untruncated: {digits!r}"
    )
    assert digits["firstLeft"] < digits["lastLeft"], (
        f"bidi reorder: the trailing digit of 'claude-fable-5' must "
        f"render at the RIGHT end of the pill: {digits!r}"
    )


@pytest.mark.timeout(180)
def test_mobile_model_list_never_scrolls_horizontally(
    tmp_path: Path,
) -> None:
    """At phone widths the open model list is never horizontally
    scrollable: rows shrink long names (start-ellipsis) instead of
    widening the scroller, and the price stays inside the dropdown."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server,
        args=(tmp_path, ready, done, state),
        daemon=True,
    )
    thread.start()
    geos: dict[str, dict] = {}
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError(
                "RemoteAccessServer startup failed"
            ) from startup_error
        port = state["port"]

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                for width, height in [(390, 844), (320, 700)]:
                    page = browser.new_page(
                        ignore_https_errors=True,
                        viewport={"width": width, "height": height},
                    )
                    page.goto(
                        f"https://127.0.0.1:{port}/",
                        wait_until="domcontentloaded",
                    )
                    _open_dropdown(page)
                    geos[f"{width}px"] = page.evaluate(_LIST_GEOMETRY_JS)
                    page.close()
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    for label, geo in geos.items():
        assert geo["listOverflowX"] == "hidden", f"{label}: {geo!r}"
        assert geo["listScrollWidth"] <= geo["listClientWidth"], (
            f"{label}: the model list is horizontally scrollable: {geo!r}"
        )
        assert geo["maxRowWidth"] <= geo["ddWidth"] + 1, (
            f"{label}: a row is wider than the dropdown: {geo!r}"
        )
        assert geo["priceVisible"], (
            f"{label}: the price fell off the dropdown's right edge: "
            f"{geo!r}"
        )
        # The 64-char name cannot fit a phone-width row: it must be
        # clipped, keeping its END on screen (leading ellipsis).
        assert geo["nameClipped"], (
            f"{label}: the long name unexpectedly fits, so this test "
            f"no longer exercises truncation: {geo!r}"
        )
        assert geo["lastRight"] <= geo["nameBoxRight"] + 1, (
            f"{label}: the END of the long name must stay visible: "
            f"{geo!r}"
        )
        assert geo["firstLeft"] < geo["nameBoxLeft"], (
            f"{label}: the START of the long name must be the clipped "
            f"part: {geo!r}"
        )
