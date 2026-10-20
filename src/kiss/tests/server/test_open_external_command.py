# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The ``openExternal`` command: a chat link opened from the app window.

The standalone app window has no browser UI, so the page hands every
web link it shows to the daemon, which opens it in the user's default
browser (``kiss.core.browser_handoff.open_in_default_browser``) and
answers ``openExternalResult`` to the requesting connection.  The
opener honours ``$BROWSER``, so the tests point it at a script that
records the URL it was given: a real launch, no mocks.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.tests.server.conftest import make_capturing_server

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX $BROWSER script")


@pytest.fixture
def recorder(tmp_path: Path) -> Iterator[Path]:
    """A ``$BROWSER`` that appends its argument to a log file."""
    log = tmp_path / "opened.log"
    script = tmp_path / "browser.sh"
    script.write_text(f'#!/bin/sh\nprintf \'%s\\n\' "$1" >> "{log}"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    saved = {k: os.environ.get(k) for k in ("BROWSER", "KISS_HEADLESS")}
    os.environ["BROWSER"] = str(script)
    os.environ["KISS_HEADLESS"] = "0"
    try:
        yield log
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _wait_for(events: list[dict[str, Any]], etype: str, timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ev in list(events):
            if ev.get("type") == etype:
                return ev
        time.sleep(0.05)
    raise AssertionError(f"no {etype} event: {events}")


def test_link_is_opened_in_the_default_browser(recorder: Path) -> None:
    server, events = make_capturing_server()
    server._cmd_open_external({"url": "https://example.com/docs?x=1", "connId": "c-app"})
    result = _wait_for(events, "openExternalResult")
    assert result["opened"] is True
    assert result["url"] == "https://example.com/docs?x=1"
    assert result["connId"] == "c-app", "the answer goes to the requesting window only"
    assert recorder.read_text(encoding="utf-8").splitlines() == ["https://example.com/docs?x=1"]


def test_a_second_click_on_the_same_link_opens_it_again(recorder: Path) -> None:
    """The handoff's once-per-URL guard (for the app's own launch) does not apply to links."""
    server, events = make_capturing_server()
    server._cmd_open_external({"url": "https://example.com/twice", "connId": "c-app"})
    _wait_for(events, "openExternalResult")
    events.clear()
    server._cmd_open_external({"url": "https://example.com/twice", "connId": "c-app"})
    result = _wait_for(events, "openExternalResult")
    assert result["opened"] is True
    assert recorder.read_text(encoding="utf-8").splitlines() == [
        "https://example.com/twice",
        "https://example.com/twice",
    ]


def test_headless_daemon_reports_nothing_opened(recorder: Path) -> None:
    os.environ["KISS_HEADLESS"] = "1"
    server, events = make_capturing_server()
    server._cmd_open_external({"url": "http://example.com/", "connId": "c-h"})
    result = _wait_for(events, "openExternalResult")
    assert result["opened"] is False
    assert result["url"] == "http://example.com/"
    assert not recorder.exists(), "nothing was launched"


def test_only_web_links_are_opened(recorder: Path) -> None:
    server, events = make_capturing_server()
    for bad in ("file:///etc/passwd", "javascript:alert(1)", "ftp://x/y", "", None, 7):
        server._cmd_open_external({"url": bad, "connId": "c-bad"})
    server._cmd_open_external({"connId": "c-bad"})
    time.sleep(0.3)
    assert events == []
    assert not recorder.exists()
