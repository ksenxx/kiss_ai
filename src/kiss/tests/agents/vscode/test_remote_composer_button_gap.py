# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: the remote webapp composer buttons sit at QUARTERED gaps.

The remote webapp composer row (``#model-picker`` in
``media/chat.html``) holds five controls — burger menu (``#menu-btn``),
model pill (``#model-btn``), attach files (``#upload-btn``), inject
promptlet (``#tricks-btn``) and mic (``#voice-btn``).  Each circular
button is 36px wide around a 16px icon, so with the historical layout
adjacent icons sat ~21px apart (icon to icon) and ~11px from the model
pill's border.  The restyle in ``remote-codex.css`` pulls the circles
together with body.remote-chat-scoped negative margins; the gaps were
first halved (-5px per inner edge) and then halved AGAIN (-8px per
inner edge), so the VISIBLE gaps are now ~5px and ~3px — a quarter of
the historical values — while keeping the full 36px touch targets and
the row's outer alignment.

This test drives the production ``RemoteAccessServer`` + headless
Chromium and asserts on the rendered geometry: the visible gap between
each adjacent pair of controls (measured between the icon <svg> edges,
or the pill's own border box for ``#model-btn``) must be at most a
QUARTER of the historical value (plus a 0.5px layout tolerance) and
must stay positive (no glyph overlap).  The VS Code webview is
unaffected: the override lives in ``remote-codex.css``, which only the
remote page loads and which is scoped under ``body.remote-chat``.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

# Composer controls in DOM order.
_BTN_IDS = ["menu-btn", "model-btn", "upload-btn", "tricks-btn", "voice-btn"]

# Historical VISIBLE gap (px) between each adjacent pair before the
# halving: 10px icon padding + 1px flex gap (+10px more when the next
# control is another padded circle rather than the bordered pill).
_OLD_VISIBLE_GAPS = {
    ("menu-btn", "model-btn"): 11.0,
    ("model-btn", "upload-btn"): 11.0,
    ("upload-btn", "tricks-btn"): 21.0,
    ("tricks-btn", "voice-btn"): 21.0,
}

_VISIBLE_EDGES_JS = """
(ids) => {
  const out = {};
  for (const id of ids) {
    const el = document.getElementById(id);
    const target = id === 'model-btn' ? el : el.querySelector('svg');
    const r = target.getBoundingClientRect();
    out[id] = {left: r.left, right: r.right};
  }
  return out;
}
"""

_TOUCH_TARGET_JS = """
(ids) => {
  const out = {};
  for (const id of ids) {
    const r = document.getElementById(id).getBoundingClientRect();
    out[id] = {width: r.width, height: r.height};
  }
  return out;
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


@pytest.mark.timeout(180)
def test_remote_composer_button_gaps_quartered(tmp_path: Path) -> None:
    """Served page + real Chromium: every adjacent composer-control
    visible gap is at most a QUARTER of its historical value (i.e. the
    once-halved gap halved again), stays positive, and the 36px touch
    targets survive."""
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
                    viewport={"width": 420, "height": 900},
                )
                page.goto(
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#input-footer", state="attached")
                page.wait_for_timeout(300)
                edges = page.evaluate(_VISIBLE_EDGES_JS, _BTN_IDS)
                targets = page.evaluate(_TOUCH_TARGET_JS, _BTN_IDS)
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

    for (a, b), old_gap in _OLD_VISIBLE_GAPS.items():
        gap = edges[b]["left"] - edges[a]["right"]
        assert gap <= old_gap / 4 + 0.5, (
            f"visible gap {a} -> {b} is {gap:.2f}px; expected at most "
            f"a quarter of the historical {old_gap:.0f}px "
            "(+0.5px tolerance)"
        )
        assert gap > 0, (
            f"visible gap {a} -> {b} is {gap:.2f}px; controls must "
            "not overlap visually"
        )

    # The halving must come from spacing, not from shrinking the
    # touch targets: every circular control keeps its 36px hit area.
    for bid in ("menu-btn", "upload-btn", "tricks-btn", "voice-btn"):
        assert targets[bid]["width"] == 36, targets
        assert targets[bid]["height"] == 36, targets
