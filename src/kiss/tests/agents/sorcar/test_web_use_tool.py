"""Tests for web_use_tool.py module."""

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.web_use_tool import (
    _ACCOUNTS_GOOGLE_URL_RE,
    _SINGLETON_FILES,
    WebUseTool,
    _abort_route,
    _activate_app,
    _get_frontmost_app,
    _is_profile_in_use,
)


class _FailureCollector:
    """Callable that records (url, errorText) for requestfailed events."""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str | None]] = []

    def __call__(self, request: Any) -> None:
        self.failures.append((request.url, request.failure))


class _AbortCallTracker:
    """Stand-in for a Playwright Route that records abort() calls."""

    def __init__(self) -> None:
        self.aborted = 0

    def abort(self) -> None:
        self.aborted += 1

FORM_PAGE = b"""<!DOCTYPE html>
<html><head><title>Test Form</title></head>
<body>
  <h1>Test Form Page</h1>
  <a href="/second">Go to second page</a>
  <form>
    <label for="username">Username</label>
    <input type="text" id="username" name="username" placeholder="Enter username">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" placeholder="Enter password">
    <label for="color">Color</label>
    <select id="color" name="color">
      <option value="red">Red</option>
      <option value="green">Green</option>
      <option value="blue">Blue</option>
    </select>
    <label for="bio">Bio</label>
    <textarea id="bio" name="bio" placeholder="Bio"></textarea>
    <button type="submit">Submit</button>
  </form>
  <button id="action-btn" onclick="document.title='Clicked!'">Action</button>
  <div id="hover-target" onmouseover="this.textContent='Hovered!'"
       style="padding:20px;background:#eee;" role="button" tabindex="0">Hover me</div>
</body></html>"""

SECOND_PAGE = b"""<!DOCTYPE html>
<html><head><title>Second Page</title></head>
<body>
  <h1>Second Page</h1>
  <a href="/">Back to form</a>
  <p>Content on second page.</p>
</body></html>"""

LONG_PAGE = b"""<!DOCTYPE html>
<html><head><title>Long Page</title></head>
<body style="height: 5000px;">
  <h1>Top of page</h1>
  <div style="position: absolute; top: 3000px;">
    <p>Bottom content</p>
  </div>
</body></html>"""

ROLE_PAGE = b"""<!DOCTYPE html>
<html><head><title>Role Page</title></head>
<body>
  <div role="button" tabindex="0">Role Button</div>
  <div role="link" tabindex="0">Role Link</div>
  <div contenteditable="true" role="textbox" aria-label="Editable div">Editable div</div>
</body></html>"""

EMPTY_PAGE = b"""<!DOCTYPE html>
<html><head><title>Empty</title></head>
<body></body></html>"""

NEW_TAB_PAGE = b"""<!DOCTYPE html>
<html><head><title>New Tab Page</title></head>
<body>
  <a href="/second" target="_blank" id="newtab-link">Open in new tab</a>
</body></html>"""

KEY_PAGE = b"""<!DOCTYPE html>
<html><head><title>Key Test</title></head>
<body>
  <input type="text" id="key-input" onkeydown="this.value=event.key">
  <div id="key-result"></div>
</body></html>"""

@pytest.fixture(scope="module")
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            pages = {
                "/": FORM_PAGE,
                "/second": SECOND_PAGE,
                "/long": LONG_PAGE,
                "/roles": ROLE_PAGE,
                "/empty": EMPTY_PAGE,
                "/newtab": NEW_TAB_PAGE,
                "/keytest": KEY_PAGE,
            }
            content = pages.get(self.path, FORM_PAGE)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="module")
def web_tool():
    tool = WebUseTool(user_data_dir=None, headless=True)
    yield tool
    tool.close()


class TestNavigation:

    def test_go_to_invalid_url(self, web_tool):
        result = web_tool.go_to_url("http://localhost:99999/nonexistent")
        assert "Error" in result


class TestCrashRecovery:
    """Verify the tool auto-recovers after Chromium/context dies unexpectedly.

    Simulates the "Google Chrome for Testing quit unexpectedly" scenario by
    closing the browser context out from under the tool.
    """

    def test_auto_relaunch_after_context_close(self, web_tool, http_server):
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()
        web_tool._context.close()
        assert not web_tool._is_alive()
        result = web_tool.go_to_url(http_server + "/")
        assert "Test" in result
        assert web_tool._is_alive()

    def test_on_page_crash_preserves_context(self, web_tool, http_server):
        """_on_page_crash must keep _context alive for cleanup.

        Before the fix, both page crash and browser close used the same
        _on_browser_lost handler which cleared _context, making the main
        browser process unreachable and leaking it.
        """
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()
        ctx = web_tool._context
        brw = web_tool._browser

        # Simulate a renderer crash event (only page dies, browser alive)
        web_tool._on_page_crash()

        assert web_tool._page is None
        assert web_tool._context is ctx  # Must be preserved
        assert web_tool._browser is brw  # Must be preserved
        assert not web_tool._is_alive()

    def test_page_crash_closes_old_browser(self, web_tool, http_server):
        """After a page renderer crash, recovery must close the old browser.

        Reproduces the root cause of "Google Chrome for Testing quit
        unexpectedly": when only the renderer crashes, the old browser
        process must be terminated during recovery.  Before the fix,
        _on_browser_lost cleared the context reference, leaking the
        main browser process.
        """
        # Recover from previous test's crash state first
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()

        # Keep a reference to the old browser
        old_browser = web_tool._browser
        assert old_browser is not None

        # Simulate a renderer crash (only page dies, browser still running)
        web_tool._on_page_crash()
        assert not web_tool._is_alive()
        # Context still alive — browser process still running
        assert old_browser.is_connected()

        # Recovery: _ensure_browser should close the old browser, then relaunch
        result = web_tool.go_to_url(http_server + "/")
        assert "Test Form" in result
        assert web_tool._is_alive()

        # The old browser must have been closed (not leaked)
        assert not old_browser.is_connected()


class TestSingletonLockCleanup:
    """Stale Singleton{Lock,Cookie,Socket} from a previously crashed Chromium
    must be removed before launching a persistent context."""

    def test_cleans_stale_singleton_files(self, tmp_path):
        (tmp_path / _SINGLETON_FILES[0]).symlink_to("stale-host-99999")
        (tmp_path / _SINGLETON_FILES[1]).write_text("stale")
        tool = WebUseTool(user_data_dir=str(tmp_path), headless=True)
        try:
            tool._clean_singleton_locks()
            for name in _SINGLETON_FILES:
                assert not (tmp_path / name).exists()
                assert not (tmp_path / name).is_symlink()
        finally:
            tool.close()

    def test_clean_singleton_locks_no_profile(self):
        """Called on an in-memory tool — no-op, does not raise."""
        tool = WebUseTool(user_data_dir=None, headless=True)
        tool._clean_singleton_locks()
        tool.close()


class TestFocusHelpers:
    """Tests for _get_frontmost_app and _activate_app focus management."""

    def test_get_frontmost_app_returns_string_on_macos(self):
        """On macOS, _get_frontmost_app should return the current app name."""
        result = _get_frontmost_app()
        if sys.platform == "darwin":
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            assert result is None

    def test_activate_app_none_is_noop(self):
        """_activate_app(None) should silently do nothing."""
        _activate_app(None)

    def test_activate_app_with_valid_app(self):
        """_activate_app with a real app should not raise."""
        if sys.platform == "darwin":
            _activate_app("Finder")

    def test_activate_app_with_nonexistent_app(self):
        """_activate_app with a bogus name should not raise (best-effort)."""
        _activate_app("NonExistentApp12345")

    def test_ensure_browser_calls_focus_helpers(self, web_tool):
        """_ensure_browser should save and restore focus even in headless mode."""
        web_tool._context.close()
        assert not web_tool._is_alive()
        web_tool._ensure_browser()
        assert web_tool._is_alive()


class TestConcurrentProfileAccess:
    """Two WebUseTool instances sharing the same user_data_dir must not collide.

    Reproduces the "Something went wrong when opening your profile" error that
    occurs when multiple tasks (e.g. local extension + remote web server) try to
    use Chromium with the same profile directory simultaneously.
    """

    def test_second_instance_uses_different_profile(self, http_server, tmp_path):
        """When the profile is locked by one instance, a second must auto-select a variant.

        Each tool runs in its own thread (realistic: real tasks come from
        separate extension/web-server threads). Thread A locks the profile,
        signals readiness, then thread B launches and must detect the lock
        and use a numbered variant instead of crashing with the Chromium
        profile error.
        """
        profile_dir = str(tmp_path / "shared_profile")

        a_ready = threading.Event()
        b_done = threading.Event()
        results: dict[str, str] = {}
        errors: dict[str, Exception] = {}

        def _run_a() -> None:
            tool = WebUseTool(user_data_dir=profile_dir, headless=True)
            try:
                results["a"] = tool.go_to_url(http_server + "/")
                a_ready.set()
                b_done.wait(timeout=60)
            except Exception as exc:
                errors["a"] = exc
                a_ready.set()
            finally:
                tool.close()

        def _run_b() -> None:
            a_ready.wait(timeout=60)
            tool = WebUseTool(user_data_dir=profile_dir, headless=True)
            try:
                results["b"] = tool.go_to_url(http_server + "/")
            except Exception as exc:
                errors["b"] = exc
            finally:
                tool.close()
                b_done.set()

        t1 = threading.Thread(target=_run_a)
        t2 = threading.Thread(target=_run_b)
        t1.start()
        t2.start()
        t2.join(timeout=90)
        b_done.set()  # unblock t1 if t2 failed
        t1.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert "Test Form" in results["a"]
        assert "Test Form" in results["b"]

    def test_is_profile_in_use_no_lock(self, tmp_path):
        """Profile without SingletonLock is not in use."""
        assert not _is_profile_in_use(str(tmp_path))

    def test_is_profile_in_use_stale_lock(self, tmp_path):
        """Profile with SingletonLock pointing to a dead PID is not in use."""
        (tmp_path / "SingletonLock").symlink_to("hostname-999999999")
        assert not _is_profile_in_use(str(tmp_path))

    def test_is_profile_in_use_live_lock(self, tmp_path):
        """Profile with SingletonLock pointing to this process is in use."""
        import os

        (tmp_path / "SingletonLock").symlink_to(f"hostname-{os.getpid()}")
        assert _is_profile_in_use(str(tmp_path))

    def test_resolve_skips_locked_profiles(self, tmp_path):
        """_resolve_user_data_dir skips the configured dir when it's locked."""
        import os

        profile = str(tmp_path / "profile")
        Path(profile).mkdir()
        (Path(profile) / "SingletonLock").symlink_to(f"hostname-{os.getpid()}")

        tool = WebUseTool(user_data_dir=profile, headless=True)
        try:
            resolved = tool._resolve_user_data_dir()
            assert resolved == f"{profile}_1"
        finally:
            tool.close()

    def test_resolve_skips_multiple_locked(self, tmp_path):
        """_resolve_user_data_dir skips numbered variants that are also locked."""
        import os

        profile = str(tmp_path / "profile")
        for suffix in ["", "_1", "_2"]:
            d = Path(f"{profile}{suffix}")
            d.mkdir(parents=True)
            (d / "SingletonLock").symlink_to(f"hostname-{os.getpid()}")

        tool = WebUseTool(user_data_dir=profile, headless=True)
        try:
            resolved = tool._resolve_user_data_dir()
            assert resolved == f"{profile}_3"
        finally:
            tool.close()

    def test_resolve_returns_none_for_no_dir(self):
        """_resolve_user_data_dir returns None when user_data_dir is None."""
        tool = WebUseTool(user_data_dir=None, headless=True)
        try:
            assert tool._resolve_user_data_dir() is None
        finally:
            tool.close()


class TestAccountsGoogleUrlRegex:
    """Pure unit tests for the accounts.google.com URL regex."""

    def test_matches_https_root(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match("https://accounts.google.com/")

    def test_matches_http_root(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match("http://accounts.google.com/")

    def test_matches_signin_path(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match(
            "https://accounts.google.com/signin/v2/identifier"
        )

    def test_matches_oauth_path(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match(
            "https://accounts.google.com/o/oauth2/v2/auth?foo=bar"
        )

    def test_rejects_other_google_host(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match("https://www.google.com/")

    def test_rejects_subdomain_prefix(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match("https://myaccounts.google.com/")

    def test_rejects_homograph_suffix(self):
        # Must not match accounts.google.com.evil.com
        assert not _ACCOUNTS_GOOGLE_URL_RE.match(
            "https://accounts.google.com.evil.com/"
        )

    def test_rejects_ftp_scheme(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match("ftp://accounts.google.com/")


class TestAbortRoute:
    """Unit tests for the _abort_route helper."""

    def test_calls_route_abort_once(self):
        tracker = _AbortCallTracker()
        _abort_route(tracker)
        assert tracker.aborted == 1


def _run_block_test(profile: str, http_url: str, out: dict) -> None:
    """Worker for test_request_to_accounts_google_is_aborted.

    Runs in its own thread so it gets a fresh asyncio thread-local state,
    which is required because the module-scoped ``web_tool`` fixture parks
    a Playwright greenlet on the main thread that leaves
    ``asyncio._running_loop`` set, causing a fresh
    ``sync_playwright().start()`` on the main thread to fail with
    "Playwright Sync API inside the asyncio loop".
    """
    tool = WebUseTool(user_data_dir=profile, headless=True)
    try:
        tool.go_to_url(http_url + "/")
        out["alive_before"] = tool._is_alive()

        collector = _FailureCollector()
        tool._context.on("requestfailed", collector)

        result = tool.go_to_url("https://accounts.google.com/signin")
        out["result_type"] = type(result).__name__
        tool._page.wait_for_timeout(500)
        out["failures"] = list(collector.failures)
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    finally:
        tool.close()


def _run_other_host_test(profile: str, http_url: str, out: dict) -> None:
    """Worker for test_other_hosts_are_not_blocked (runs in its own thread)."""
    tool = WebUseTool(user_data_dir=profile, headless=True)
    try:
        out["body"] = tool.go_to_url(http_url + "/")
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    finally:
        tool.close()


class TestAccountsGoogleRouteBlocking:
    """Verify the route registered in _launch_browser blocks accounts.google.com.

    These integration tests launch a real persistent-context browser
    (the only path where the route is currently registered) and confirm
    that any request to accounts.google.com is aborted before it leaves
    the browser.

    They run in a worker thread because the module-scoped ``web_tool``
    fixture leaves a Playwright greenlet parked on the main thread, which
    makes the next ``sync_playwright().start()`` on that thread raise.
    """

    def test_request_to_accounts_google_is_aborted(self, tmp_path, http_server):
        profile = str(tmp_path / "profile")
        out: dict = {}
        t = threading.Thread(
            target=_run_block_test, args=(profile, http_server, out)
        )
        t.start()
        t.join(timeout=120)
        assert not t.is_alive(), "worker thread hung"
        assert "error" not in out, f"worker raised: {out.get('error')}"
        assert out["alive_before"] is True
        assert out["result_type"] is not None

        matches = [
            (url, err)
            for url, err in out["failures"]
            if "accounts.google.com" in url
        ]
        assert matches, (
            "No requestfailed event captured for accounts.google.com; "
            f"all failures: {out['failures']}"
        )
        # Playwright's route.abort() default reason produces ERR_FAILED
        # (not ERR_NAME_NOT_RESOLVED, which would indicate the request
        # actually went out).
        url, err = matches[0]
        assert err is not None
        assert "ERR_FAILED" in err or "ABORTED" in err, (
            f"Expected an aborted-request error, got: {err!r}"
        )

    def test_other_hosts_are_not_blocked(self, tmp_path, http_server):
        """Sanity check: requests to other hosts must still succeed."""
        profile = str(tmp_path / "profile2")
        out: dict = {}
        t = threading.Thread(
            target=_run_other_host_test, args=(profile, http_server, out)
        )
        t.start()
        t.join(timeout=120)
        assert not t.is_alive(), "worker thread hung"
        assert "error" not in out, f"worker raised: {out.get('error')}"
        assert "Test Form" in out["body"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
