# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: hovering the pinned task text on the REMOTE web app shows
a tooltip with the ENTIRE task text at the SAME font size as the task
text itself.

The remote web app (``RemoteAccessServer``) serves the exact same
``chat.html`` + ``main.js`` + ``main.css`` as the VS Code extension
webview, plus ``remote-codex.css`` overrides.  This live test boots the
production server, opens the page in headless Chromium, sets a long
task through the production ``setTaskText`` event path, hovers
``#task-panel-text`` with a real mouse move, and asserts:

* the shared ``#custom-tooltip`` becomes visible and contains the
  ENTIRE task text;
* the tooltip's computed ``font-size`` EQUALS the computed
  ``font-size`` of the task text in the fixed panel.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

LONG_TASK = (
    "Refactor the payment pipeline to support multi-currency "
    "settlement. Step 1: normalize every ledger entry to minor "
    "units. Step 2: add an FX-rate snapshot table keyed by "
    "(currency, day). Step 3: migrate historical rows in batches "
    "of 10k with checkpoints. " + "x" * 600 + " Finally run the "
    "full reconciliation suite and attach the report."
)


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
    uds_dir = tempfile.mkdtemp(prefix="kiss-tt-")

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            # AF_UNIX socket paths are limited to ~104 bytes on macOS;
            # pytest tmp_path can exceed that, so bind in a short dir
            # (removed in the finally below).
            uds_path=Path(uds_dir) / "sorcar.sock",
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

    try:
        asyncio.run(scenario())
    finally:
        shutil.rmtree(uds_dir, ignore_errors=True)


@pytest.mark.timeout(180)
def test_live_remote_task_panel_hover_tooltip(tmp_path: Path) -> None:
    """Served page + real Chromium: hovering the pinned task text pops
    a tooltip with the ENTIRE task text at the task text's font size."""
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
                page.wait_for_selector("#task-panel", state="attached")

                # Pin the task through the PRODUCTION event path — the
                # same window 'message' handler the backend drives.
                page.evaluate(
                    "task => window.postMessage("
                    "{type: 'setTaskText', text: task}, '*')",
                    LONG_TASK,
                )
                page.wait_for_selector("#task-panel.visible")

                # The full text must ride on the hover tooltip source.
                tooltip_attr = page.get_attribute(
                    "#task-panel-text", "data-tooltip"
                )
                assert tooltip_attr == LONG_TASK, (
                    "#task-panel-text must carry the ENTIRE task text "
                    f"in data-tooltip; got: {tooltip_attr!r}"
                )

                # Real hover: move the mouse onto the task text and
                # wait out the 400 ms show delay + fade-in.
                page.hover("#task-panel-text")
                page.wait_for_selector("#custom-tooltip.visible")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById("
                    "'custom-tooltip')).opacity === '1'"
                )

                probes = page.evaluate(
                    """() => {
                        const tip = document.getElementById(
                            'custom-tooltip');
                        const txt = document.getElementById(
                            'task-panel-text');
                        return {
                            tipText: tip.textContent,
                            tipFontSize: getComputedStyle(tip).fontSize,
                            taskFontSize: getComputedStyle(txt).fontSize,
                            taskPanelClass: tip.classList.contains(
                                'task-panel-tooltip'),
                        };
                    }"""
                )

                # Moving the mouse away must hide the tooltip again.
                page.mouse.move(5, 5)
                page.wait_for_function(
                    "() => !document.getElementById('custom-tooltip')"
                    ".classList.contains('visible')"
                )
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

    assert probes["tipText"] == LONG_TASK, (
        "the tooltip must contain the ENTIRE task text; got: "
        + repr(probes["tipText"])
    )
    assert probes["taskPanelClass"] is True, (
        "the task tooltip must carry .task-panel-tooltip: " + repr(probes)
    )
    assert probes["tipFontSize"] == probes["taskFontSize"], (
        "the tooltip font size must EQUAL the task text font size; "
        + repr(probes)
    )
