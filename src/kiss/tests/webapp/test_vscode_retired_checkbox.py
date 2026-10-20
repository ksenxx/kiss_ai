# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from other test modules)
"""End-to-end test of the "Do not show this again" checkbox on the "VS Code
extension removed" toast, in a REAL headless Chromium against a REAL
:class:`RemoteAccessServer`: the toast renders with the box, ticking
it removes the toast, writes ``$KISS_HOME/.vscode-retired-dismissed``,
closes the toast in a second window, and a fresh page gets no toast.
"""

from __future__ import annotations

import time

import pytest

from kiss.core.config import kiss_home
from kiss.server.vscode_retired import VSCODE_RETIRED_DISMISSED, VSCODE_RETIRED_FLAG
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_workspace_sections import (
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)

TOAST = '.kiss-notification[data-notification-id="vscode-retired"]'
BOX = TOAST + " .kiss-notification-opt-out-box"


@pytest.fixture
def retired_flag():
    flag = kiss_home() / VSCODE_RETIRED_FLAG
    marker = kiss_home() / VSCODE_RETIRED_DISMISSED
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("2026-10-16T00:00:00Z\n")
    marker.unlink(missing_ok=True)
    yield marker
    flag.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)


def test_do_not_show_this_retires_the_notice_everywhere(browser, harness, retired_flag):
    first, page, _ = _open_page(browser, harness)
    second, other, _ = _open_page(browser, harness)
    try:
        for p in (page, other):
            p.wait_for_selector(TOAST, timeout=15000)
            label = p.locator(TOAST + " label.kiss-notification-opt-out")
            assert label.inner_text().strip() == "Do not show this again"
            assert p.locator(BOX).is_checked() is False
        # Tick it in the first window.
        page.click(BOX)
        page.wait_for_selector(TOAST, state="detached", timeout=5000)
        # The daemon recorded the choice...
        deadline = time.time() + 5
        while not retired_flag.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert retired_flag.is_file()
        # ...and closed the toast in the other window too.
        other.wait_for_selector(TOAST, state="detached", timeout=5000)
        # A fresh page gets no toast while the other bootstrap toasts
        # and the page itself still come up.
        page.reload()
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        page.wait_for_timeout(1500)
        assert page.locator(TOAST).count() == 0
    finally:
        first.close()
        second.close()
