# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (leftovers): reading a Browser-tab page's target id is bounded.

``WebUseTool._target_id`` attaches a CDP session to the page and sends
``Target.getTargetInfo``; Playwright's sync API gives neither call a
timeout, and its driver fails a pending CDP-session request only on
the session's own detach event, never when the browser stops answering
or exits under it.  ``_find_live_page`` (``show_browser``, a tab the
user closed) and ``_live_tab_id`` (the input watchdog, closing the tab)
therefore hung the task for good on a browser that froze or died at
that moment.  Both round trips now go through ``_cdp_call`` with
``_CDP_CALL_TIMEOUT_SECS``.

A REAL daemon ``BrowserTabService`` (its own Chromium) and a REAL
``WebUseTool`` (a second CDP client on it); the browser is frozen with
``SIGSTOP`` and, in the second test, killed while the request is
pending.  Before the fix each test hangs until its own 40 s watchdog
resumes the browser (first) or until pytest's timeout (second).
"""

from __future__ import annotations

import gc
import os
import signal
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import web_use_tool as wut
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.agents.sorcar.test_web_use_tool_live_browser import _daemon_browser_pid
from kiss.tests.conftest import posix_only
from kiss.tests.server._memory_printer import MemoryPrinter

pytestmark = pytest.mark.skipif(
    not (Path.home() / ".cache" / "ms-playwright").is_dir(),
    reason="Playwright browsers not installed",
)

_BOUND = wut._CDP_CALL_TIMEOUT_SECS + 2.0


@pytest.fixture
def service(tmp_path: Path) -> Any:
    svc = BrowserTabService(MemoryPrinter(), tmp_path / "tab-profile")
    yield svc
    svc.shutdown()


@pytest.fixture
def tool(tmp_path: Path, service: BrowserTabService) -> Any:
    web = WebUseTool(user_data_dir=str(tmp_path / "agent-profile"), live_browser=service)
    yield web
    web.close()


def _timed_target_id(tool: WebUseTool) -> tuple[Exception | None, float]:
    started = time.monotonic()
    try:
        tool._target_id(tool._page)
    except Exception as exc:  # noqa: BLE001 - the bound, or whatever Playwright raises first
        return exc, time.monotonic() - started
    return None, time.monotonic() - started


@posix_only("the browser is frozen with SIGSTOP")
def test_target_id_of_a_browser_that_does_not_answer_is_bounded(tool, service):
    """Attaching the session and the read both come back on their own while
    the browser is stopped, with and without a session already attached,
    and the session still works once the browser answers again."""
    tool.show_browser()
    page = tool._page
    pid = _daemon_browser_pid(tool)
    # ``show_browser`` attached a session while finding the page; forget it
    # (as ``_detach_live`` does) so the next read has to attach again.
    tool._cdp_sessions.clear()
    os.kill(pid, signal.SIGSTOP)
    resume = threading.Timer(40, os.kill, args=(pid, signal.SIGCONT))
    resume.daemon = True
    resume.start()
    try:
        attach_exc, attach_elapsed = _timed_target_id(tool)
        os.kill(pid, signal.SIGCONT)
        time.sleep(0.2)
        first = tool._target_id(page)  # attaches for real
        assert page in tool._cdp_sessions
        os.kill(pid, signal.SIGSTOP)
        send_exc, send_elapsed = _timed_target_id(tool)
    finally:
        resume.cancel()
        os.kill(pid, signal.SIGCONT)
    assert isinstance(attach_exc, TimeoutError), attach_exc
    assert attach_elapsed < _BOUND, f"attaching waited {attach_elapsed:.1f}s for the browser"
    assert isinstance(send_exc, TimeoutError), send_exc
    assert send_elapsed < _BOUND, f"the read waited {send_elapsed:.1f}s for the browser"
    # The session survived the abandoned request.
    assert tool._target_id(page) == first
    assert service.tab_for_target(first) == tool._live_tab


@posix_only("the browser is frozen with SIGSTOP")
def test_target_id_of_a_browser_exiting_under_the_read_is_bounded(tool, service):
    """The documented hang: the driver never fails the pending request of a
    browser that exits during it.  The bound does, and ``_live_tab_id``
    (which swallows the failure) comes back ``None`` in time."""
    tool.show_browser()
    tool._target_id(tool._page)
    pid = _daemon_browser_pid(tool)
    os.kill(pid, signal.SIGSTOP)
    kill = threading.Timer(1, os.kill, args=(pid, signal.SIGKILL))
    kill.daemon = True
    kill.start()
    exc, elapsed = _timed_target_id(tool)
    assert isinstance(exc, TimeoutError), exc
    assert elapsed < _BOUND, f"the read waited {elapsed:.1f}s for a dead browser"
    tool._live_tab = None
    started = time.monotonic()
    assert tool._live_tab_id() is None
    assert time.monotonic() - started < _BOUND


def test_cdp_call_after_playwright_stopped_leaves_no_unawaited_coroutine():
    """Review 2026-10-12 (core): a call on a stopped Playwright is rejected cleanly.

    ``SyncBase._sync`` on a closed loop closes only the ``wait_for``
    wrapper it was handed and raises; the ``CDPSession.send`` coroutine
    inside stayed unawaited and was reported as such when collected.
    ``_cdp_call`` closes it itself.
    """
    from playwright.sync_api import Error, sync_playwright

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        session = wut._cdp_call(context, context._impl_obj.new_cdp_session(page))
        info = wut._cdp_call(page, session.send("Target.getTargetInfo"))
        assert info["targetInfo"]["type"] == "page"
        browser.close()
    finally:
        pw.stop()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(Error, match="Event loop is closed"):
            wut._cdp_call(page, session.send("Target.getTargetInfo"))
        gc.collect()
    assert [str(w.message) for w in caught] == []
