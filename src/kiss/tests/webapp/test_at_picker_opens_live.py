# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: typing ``@`` in the chat composer opens the file picker.

The production RemoteAccessServer serves the page to a real headless
Chromium.  The test types ``@`` into ``#task-input`` and waits for the
``#autocomplete`` popup to list the working directory's files: the
``getFiles`` request, the daemon's ``loading`` placeholder, the index
build on the registry's worker thread and the populated ``files`` reply
all have to happen for that.

Regression (2026-10-31): on macOS the daemon's index worker sat for
minutes in the home-directory scan (a five-second privacy-protection
wait per ``~/Library`` directory, then an ``open`` of a ``.gitignore``
inside iCloud Drive that never returned), so no ``files`` reply was ever
sent and ``@`` showed nothing.  Here the working directory is a root of
its own and the home index is never waited for, as in the daemon; the
test pins the end-to-end path the user sees.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.conftest import touch
from kiss.tests.webapp.test_codex_task_panel_style import _start_live_server

_ITEM_TEXTS_JS = """
() => Array.from(document.querySelectorAll('#autocomplete .ac-item'))
  .map(el => el.dataset.text)
"""


@pytest.mark.timeout(180)
def test_typing_at_opens_the_file_picker_with_the_work_dirs_files(tmp_path: Path) -> None:
    touch(tmp_path, "notes.md", "src/app.py", "src/util.py")
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True,
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from startup_error
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                page = browser.new_page(
                    ignore_https_errors=True, viewport={"width": 1280, "height": 900},
                )
                goto_retrying_network_change(
                    page, f"https://127.0.0.1:{state['port']}/", wait_until="load",
                )
                # The composer is usable once the socket is up and the
                # start-up overlay has gone.
                page.wait_for_selector("#kiss-server-loading", state="hidden")
                page.wait_for_selector("#task-input", state="visible")
                page.click("#task-input")
                page.keyboard.type("@")
                page.wait_for_selector("#autocomplete .ac-item", state="visible", timeout=30_000)
                items = page.evaluate(_ITEM_TEXTS_JS)
                expected = {"./notes.md", "./src/", "./src/app.py", "./src/util.py"}
                assert set(items) >= expected, items
                # Narrowing the mention filters the list.
                page.keyboard.type("util")
                page.wait_for_function(
                    "() => Array.from(document.querySelectorAll('#autocomplete .ac-item'))"
                    ".map(el => el.dataset.text).join('|') === './src/util.py'",
                    timeout=30_000,
                )
                # Tab accepts the highlighted entry into the composer.
                page.keyboard.press("Tab")
                assert page.input_value("#task-input") == "./src/util.py "
                assert page.is_hidden("#autocomplete")
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
