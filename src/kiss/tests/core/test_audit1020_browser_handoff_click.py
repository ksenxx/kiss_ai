# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``open_in_default_browser(once=False)`` shares an in-flight launch (audit 2026-10-20, A4-F1).

A link the user clicks is opened again even when the page was opened a
moment ago.  But while another caller's launch of the same page is still
in flight, the click must join that launch like every other caller: the
old code released the live claim first, which woke the waiters and let
the page open two or three times.

The race is established by handshakes, not by timing: the launcher
reports that it is running (its log line) and exits only when the test
releases it, and the test releases it only after both the plain caller
and the click are seen waiting on the still-unsettled claim.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kiss.core import browser_handoff
from kiss.core.browser_handoff import open_in_default_browser
from kiss.tests.conftest import IS_WINDOWS, install_cli_script

# A browser launcher that records its argv, then exits 0 only once the
# test writes the ``release`` file beside it: the launch is "in flight"
# until then (bounded by ``_LAUNCH_GRACE``, after which the opener counts
# as a success anyway) and then settles as a success.
_GATED_BROWSER = f"""#!{sys.executable}
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "browser-log"), "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
deadline = time.monotonic() + 30
while not os.path.exists(os.path.join(here, "release")) and time.monotonic() < deadline:
    time.sleep(0.01)
"""


@pytest.fixture()
def gated_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Install the gated recording browser as ``$BROWSER`` on a non-headless host."""
    home = tmp_path / "browser"
    home.mkdir()
    script = home / "browser"
    install_cli_script(script, _GATED_BROWSER)
    launcher = script.with_name("browser.cmd") if IS_WINDOWS else script
    monkeypatch.setenv("BROWSER", str(launcher))
    monkeypatch.setenv("KISS_HEADLESS", "0")
    with browser_handoff._lock:
        browser_handoff._recent.clear()
    yield home
    (home / "release").touch()


def _launches(home: Path) -> list[list[str]]:
    """Return the argv of every browser launch recorded so far."""
    log = home / "browser-log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _wait_until(condition: Callable[..., bool], what: str, *args: object) -> None:
    """Poll ``condition(*args)`` until it holds, failing with *what* after 10 s."""
    deadline = time.monotonic() + 10.0
    while not condition(*args):
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def _launched_once(home: Path) -> bool:
    """Whether exactly one browser launch has been recorded."""
    return len(_launches(home)) == 1


def _both_joined(url: str) -> bool:
    """Whether two callers are blocked in ``_claim`` on the unsettled launch of *url*."""
    with browser_handoff._lock:
        opening = browser_handoff._recent.get(url)
        if opening is None or opening.settled:
            return False
        return len(getattr(browser_handoff._lock, "_waiters")) == 2


def _open(url: str, once: bool, results: list[bool]) -> None:
    results.append(open_in_default_browser(url, once=once))


def test_click_during_an_in_flight_launch_joins_it(gated_browser: Path) -> None:
    url = f"https://example.test/page/{uuid.uuid4()}"
    results: list[bool] = []
    first = threading.Thread(target=_open, args=(url, True, results), daemon=True)
    first.start()
    _wait_until(_launched_once, "the first launch never started", gated_browser)
    waiter = threading.Thread(target=_open, args=(url, True, results), daemon=True)
    click = threading.Thread(target=_open, args=(url, False, results), daemon=True)
    waiter.start()
    click.start()
    # Both must be parked on the claim while the launch is still in flight.
    deadline = time.monotonic() + 10.0
    while not _both_joined(url):
        assert _launches(gated_browser) == [[url]], (
            "the click opened the page again instead of joining the launch in flight"
        )
        assert time.monotonic() < deadline, "the callers never joined the in-flight launch"
        time.sleep(0.01)
    (gated_browser / "release").touch()
    for thread in (first, waiter, click):
        thread.join(timeout=15)
        assert not thread.is_alive()
    # All three callers report the page open; the browser launched once.
    assert results == [True, True, True]
    assert _launches(gated_browser) == [[url]]

    # Once the launch has settled, a click opens the page again while a
    # plain caller keeps sharing the recent launch.
    assert open_in_default_browser(url) is True
    assert _launches(gated_browser) == [[url]]
    assert open_in_default_browser(url, once=False) is True
    assert _launches(gated_browser) == [[url], [url]]
