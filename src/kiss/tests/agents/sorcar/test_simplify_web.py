# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``web_use_tool`` simplification (W1-W6, S3).

Every test drives a real Chromium (or real files and processes) and fails
on the code as it was:

* W1 — an ephemeral profile left a ``<dir>.lock`` file beside the deleted
  directory, one per sub-agent, forever.
* W2 — the ephemeral directory was created in ``__init__``; a tool that
  never opened a page leaked it when its process died without ``close()``.
* W3 — ``go_to_url("tab:N")`` kept the Browser-tab id of the page it
  left, so the hang watchdog would interrupt the wrong tab.
* W4 — the localStorage restore used an unbounded ``evaluate``; a page
  whose script wedges the renderer right after commit hung
  ``show_browser`` forever.  So did, right after it, the liveness round
  trip of ``_is_alive`` and ``_target_id``: a page-level CDP session's
  ``detach()`` waits for the wedged renderer.
* S3 — escalation-dir cleanup and profile resolution are one walk.

W6 (re-registering the crash handler on the page already adopted) was a
redundancy, not a leak: Playwright caches the wrapper of a bound-method
handler and the emitter keys handlers by identity, so the repeated
``on("crash", ...)`` was a no-op.  Its test pins that re-adoption changes
nothing (now by returning early, which also keeps the per-page state).

W5 (the process-local ``_LAUNCH_LOCK`` around the launch, redundant with
the file lock) has no test of its own: its only observable effect was
timing, and ``test_browser_profile_cross_process_lock.py`` already proves
the file lock excludes concurrent launches.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.conftest import PLAYWRIGHT_CHROMIUM_INSTALLED, posix_only
from kiss.tests.server._memory_printer import MemoryPrinter

pytestmark = pytest.mark.skipif(
    not PLAYWRIGHT_CHROMIUM_INSTALLED,
    reason="Playwright browsers not installed",
)


class _Handler(BaseHTTPRequestHandler):
    """``/sticky`` stores a localStorage key the first time it is served and
    wedges the renderer (``for(;;){}`` at parse time) every time after;
    anything else is an inert page."""

    sticky_served = False

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Answer according to the path."""
        body = "<title>Inert</title>a page"
        if self.path == "/sticky" and _Handler.sticky_served:
            body = "<title>Wedge</title><script>for(;;){}</script>"
        elif self.path == "/sticky":
            _Handler.sticky_served = True
            body = "<title>Sticky</title><script>localStorage.setItem('tok', '1')</script>ok"
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Keep the test output quiet."""


@pytest.fixture
def server() -> Any:
    _Handler.sticky_served = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def service(tmp_path: Path) -> Any:
    printer = MemoryPrinter()
    svc = BrowserTabService(printer, tmp_path / "tab-profile")
    yield svc, printer
    svc.shutdown()


@pytest.fixture
def live_tool(tmp_path: Path, service: Any) -> Any:
    web = WebUseTool(user_data_dir=str(tmp_path / "agent-profile"), live_browser=service[0])
    yield web
    web.close()


def _wait(pred: Any, what: str, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = pred()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _opened_tabs(printer: MemoryPrinter) -> list[str]:
    return [e["tab_id"] for e in list(printer.emitted) if e["type"] == "openBrowserTab"]


def _dead_pid() -> int:
    """A pid that no process has (a reaped child's)."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@posix_only("the ephemeral profile and its lock file are POSIX paths")
def test_ephemeral_profile_leaves_no_lock_file_behind(server: str) -> None:
    """W1 + W2: nothing exists before the first launch; after ``close()``
    neither the directory nor a ``<dir>.lock`` beside it remains."""
    tool = WebUseTool(headless=True, ephemeral=True)
    assert tool.user_data_dir is None and tool._ephemeral_dir is None
    try:
        assert tool.go_to_url(f"{server}/inert").startswith("Page: Inert")
        profile = tool.user_data_dir
        assert profile is not None and os.path.isdir(profile)
        assert tool.effective_user_data_dir == profile
    finally:
        tool.close()
    assert not os.path.exists(profile), "ephemeral profile dir leaked"
    assert not os.path.exists(f"{profile}.lock"), "ephemeral profile lock file leaked"


def test_switching_to_the_users_tab_retargets_the_live_tab_id(
    live_tool: WebUseTool, service: Any, server: str
) -> None:
    """W3: after ``go_to_url("tab:N")`` the tool's Browser-tab id is the
    tab it now drives (the user's), not the one it opened itself."""
    svc, printer = service
    svc.open(f"{server}/inert", "user")
    _wait(lambda: _opened_tabs(printer), "user tab")
    user_tab = _opened_tabs(printer)[0]
    live_tool.show_browser()
    agent_tab = live_tool._live_tab
    assert agent_tab is not None and agent_tab != user_tab

    pages = live_tool._context.pages
    idx = next(i for i, page in enumerate(pages) if page.url == f"{server}/inert")
    assert live_tool.go_to_url(f"tab:{idx}").startswith("Page: Inert")
    assert live_tool._live_tab_id() == user_tab
    assert live_tool._mouse_xy is None
    # The watchdog of the next raw input targets the page being driven.
    live_tool.press_key("Tab")
    assert live_tool._live_tab_id() == user_tab
    # Leaving the user's tab in place: the tool must not close it on exit.
    live_tool.close()
    assert user_tab in svc._pages


def test_wedged_page_bounds_the_switch_of_browser(
    live_tool: WebUseTool, service: Any, server: str
) -> None:
    """W4: the localStorage restore of the switch is bounded like any page
    read; a page that wedges its renderer at commit does not hang it."""
    assert live_tool.go_to_url(f"{server}/sticky").startswith("Page: Sticky")
    assert live_tool._page.evaluate("localStorage.getItem('tok')") == "1"

    started = time.monotonic()
    result = live_tool.show_browser()
    elapsed = time.monotonic() - started
    assert isinstance(result, str)
    assert elapsed < 90, f"show_browser took {elapsed:.0f}s on a wedged page"
    # The write was abandoned, so its keys are not treated as carried over.
    assert live_tool._carried_storage == ("", set())


def test_adopting_the_current_page_again_adds_no_crash_listener(server: str) -> None:
    """W6: ``tab:N`` on the current tab leaves the page, its crash handler
    and the pointer position alone."""
    tool = WebUseTool(headless=True, user_data_dir=None)
    try:
        assert tool.go_to_url(f"{server}/inert").startswith("Page: Inert")
        page = tool._page
        # Playwright keeps a listener of its own; the tool's is the other.
        armed = len(page._impl_obj.listeners("crash"))
        tool._mouse_xy = (12.0, 34.0)
        for _ in range(3):
            assert tool.go_to_url("tab:0").startswith("Page: Inert")
        assert tool._page is page
        assert len(page._impl_obj.listeners("crash")) == armed
        assert tool._mouse_xy == (12.0, 34.0)
    finally:
        tool.close()


@posix_only("Chromium's SingletonLock symlink is the POSIX profile lock")
def test_resolution_escalates_past_live_and_reclaims_stale_variants(tmp_path: Path) -> None:
    """S3: with the base profile in use, a stale variant is reclaimed and
    chosen, a live one is skipped and kept, and the walk also sweeps stale
    variants beyond the chosen one."""
    base = tmp_path / "profile"
    base.mkdir()
    os.symlink(f"testhost-{os.getpid()}", str(base / "SingletonLock"))
    live = tmp_path / "profile_1"
    live.mkdir()
    os.symlink(f"testhost-{os.getpid()}", str(live / "SingletonLock"))
    dead = _dead_pid()
    for i in (2, 3):
        stale = tmp_path / f"profile_{i}"
        stale.mkdir()
        os.symlink(f"testhost-{dead}", str(stale / "SingletonLock"))

    tool = WebUseTool(user_data_dir=str(base), headless=True)
    assert tool._resolve_user_data_dir() == str(tmp_path / "profile_2")
    assert live.is_dir(), "live escalation dir was deleted"
    assert not (tmp_path / "profile_2").exists() and not (tmp_path / "profile_3").exists()
    tool.close()
