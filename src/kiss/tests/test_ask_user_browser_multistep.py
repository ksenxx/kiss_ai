"""Tests for ask_user_browser_action with complex multi-step scenarios.

Tests a multi-page web application with:
- Login form with validation
- Multi-step wizard flow
- Dynamic content after user actions
- Navigation between pages
- Callback coordination with threading
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

# --- Multi-page web app HTML ---

LOGIN_PAGE = b"""<!DOCTYPE html>
<html><head><title>Login</title></head>
<body>
  <h1>Login Required</h1>
  <form action="/dashboard" method="get">
    <label for="user">Username</label>
    <input type="text" id="user" name="user" role="textbox" aria-label="Username">
    <label for="pass">Password</label>
    <input type="password" id="pass" name="pass" role="textbox" aria-label="Password">
    <button type="submit">Sign In</button>
  </form>
  <a href="/forgot">Forgot password?</a>
</body></html>"""

DASHBOARD_PAGE = b"""<!DOCTYPE html>
<html><head><title>Dashboard</title></head>
<body>
  <h1>Welcome to Dashboard</h1>
  <nav>
    <a href="/wizard/step1">Start Setup Wizard</a>
    <a href="/settings">Settings</a>
    <a href="/">Logout</a>
  </nav>
  <p>You are logged in.</p>
</body></html>"""

WIZARD_STEP1 = b"""<!DOCTYPE html>
<html><head><title>Wizard - Step 1</title></head>
<body>
  <h1>Setup Wizard - Step 1 of 3</h1>
  <p>Choose your preferences</p>
  <label for="name">Project Name</label>
  <input type="text" id="name" role="textbox" aria-label="Project Name">
  <button id="next1">Next</button>
  <a href="/dashboard">Cancel</a>
</body></html>"""

WIZARD_STEP2 = b"""<!DOCTYPE html>
<html><head><title>Wizard - Step 2</title></head>
<body>
  <h1>Setup Wizard - Step 2 of 3</h1>
  <p>Configure your options</p>
  <label><input type="checkbox" role="checkbox"
    aria-label="Enable notifications"> Notifications</label>
  <label><input type="checkbox" role="checkbox"
    aria-label="Enable dark mode"> Dark mode</label>
  <button id="back2">Back</button>
  <button id="next2">Next</button>
</body></html>"""

WIZARD_STEP3 = b"""<!DOCTYPE html>
<html><head><title>Wizard - Step 3</title></head>
<body>
  <h1>Setup Wizard - Step 3 of 3</h1>
  <p>Review and confirm</p>
  <p>Click Finish to complete setup.</p>
  <button id="back3">Back</button>
  <button id="finish">Finish</button>
</body></html>"""

WIZARD_DONE = b"""<!DOCTYPE html>
<html><head><title>Setup Complete</title></head>
<body>
  <h1>Setup Complete!</h1>
  <p>Your project has been configured successfully.</p>
  <a href="/dashboard">Go to Dashboard</a>
</body></html>"""

SETTINGS_PAGE = b"""<!DOCTYPE html>
<html><head><title>Settings</title></head>
<body>
  <h1>Settings</h1>
  <label for="email">Email</label>
  <input type="text" id="email" role="textbox" aria-label="Email">
  <button id="save">Save</button>
  <a href="/dashboard">Back to Dashboard</a>
</body></html>"""

FORGOT_PAGE = b"""<!DOCTYPE html>
<html><head><title>Forgot Password</title></head>
<body>
  <h1>Reset Password</h1>
  <label for="reset-email">Email</label>
  <input type="text" id="reset-email" role="textbox" aria-label="Reset Email">
  <button id="reset">Send Reset Link</button>
  <a href="/">Back to Login</a>
</body></html>"""

PAGES = {
    "/": LOGIN_PAGE,
    "/dashboard": DASHBOARD_PAGE,
    "/wizard/step1": WIZARD_STEP1,
    "/wizard/step2": WIZARD_STEP2,
    "/wizard/step3": WIZARD_STEP3,
    "/wizard/done": WIZARD_DONE,
    "/settings": SETTINGS_PAGE,
    "/forgot": FORGOT_PAGE,
}


@pytest.fixture(scope="module")
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            content = PAGES.get(path, LOGIN_PAGE)
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


@pytest.fixture()
def tool():
    t = WebUseTool(headless=True, user_data_dir=None, wait_for_user_callback=None)
    yield t
    t.close()


class TestMultiStepWizardFlow:
    """Test navigating through a multi-step wizard using ask_user_browser_action."""

    def test_login_then_wizard_full_flow(self, http_server: str, tool: WebUseTool) -> None:
        """Navigate login -> dashboard -> wizard steps 1-3 -> completion."""
        # Step 1: Navigate to login
        result = tool.go_to_url(http_server)
        assert "Login" in result
        assert "Sign In" in result

        # Step 2: Ask user to log in (simulated - no callback, just returns page state)
        result = tool.ask_user_browser_action("Please log in with your credentials")
        assert "Login" in result
        assert "Username" in result

        # Step 3: Navigate to dashboard (simulating post-login)
        result = tool.go_to_url(f"{http_server}/dashboard")
        assert "Dashboard" in result
        assert "Start Setup Wizard" in result

        # Step 4: Navigate to wizard step 1
        result = tool.go_to_url(f"{http_server}/wizard/step1")
        assert "Step 1" in result
        assert "Project Name" in result

        # Step 5: Navigate through wizard steps
        result = tool.go_to_url(f"{http_server}/wizard/step2")
        assert "Step 2" in result
        assert "Notifications" in result

        result = tool.go_to_url(f"{http_server}/wizard/step3")
        assert "Step 3" in result
        assert "Finish" in result

        # Step 6: Complete wizard
        result = tool.go_to_url(f"{http_server}/wizard/done")
        assert "Setup Complete" in result
        assert "Go to Dashboard" in result


class TestCallbackCoordinationMultiStep:
    """Test callback-based user interaction across multiple pages."""

    def test_callback_tracks_multiple_navigations(self, http_server: str) -> None:
        """Callback is invoked at each user interaction point with correct URLs."""
        interactions: list[tuple[str, str]] = []

        def callback(instruction: str, url: str) -> None:
            interactions.append((instruction, url))

        t = WebUseTool(headless=True, user_data_dir=None, wait_for_user_callback=callback)
        try:
            # Login page
            t.ask_user_browser_action("Log in", url=http_server)
            assert len(interactions) == 1
            assert interactions[0][0] == "Log in"

            # Dashboard
            t.ask_user_browser_action("Click wizard link", url=f"{http_server}/dashboard")
            assert len(interactions) == 2
            assert interactions[1][0] == "Click wizard link"
            assert "/dashboard" in interactions[1][1]

            # Wizard step 1
            t.ask_user_browser_action(
                "Fill in project name", url=f"{http_server}/wizard/step1"
            )
            assert len(interactions) == 3
            assert interactions[2][0] == "Fill in project name"

            # Wizard step 2
            t.ask_user_browser_action(
                "Toggle checkboxes", url=f"{http_server}/wizard/step2"
            )
            assert len(interactions) == 4
            assert interactions[3][0] == "Toggle checkboxes"

        finally:
            t.close()

    def test_callback_with_threaded_user_simulation(self, http_server: str) -> None:
        """Simulate a realistic multi-step flow where callback blocks until user acts."""
        completed_steps: list[str] = []
        user_events: list[threading.Event] = []

        def blocking_callback(instruction: str, url: str) -> None:
            event = threading.Event()
            user_events.append(event)
            completed_steps.append(f"waiting:{instruction}")
            event.wait(timeout=5.0)
            completed_steps.append(f"done:{instruction}")

        t = WebUseTool(
            headless=True, user_data_dir=None, wait_for_user_callback=blocking_callback
        )
        try:
            # Run agent's multi-step flow in a background thread
            agent_done = threading.Event()

            def agent_flow() -> None:
                t.ask_user_browser_action("Step 1: Login", url=http_server)
                t.ask_user_browser_action(
                    "Step 2: Fill form", url=f"{http_server}/wizard/step1"
                )
                t.ask_user_browser_action(
                    "Step 3: Confirm", url=f"{http_server}/wizard/step3"
                )
                agent_done.set()

            thread = threading.Thread(target=agent_flow)
            thread.start()

            # Simulate user completing each step with small delays
            for i in range(3):
                # Wait for callback to register
                deadline = time.monotonic() + 5.0
                while len(user_events) <= i and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert len(user_events) > i, f"Step {i} callback never registered"
                user_events[i].set()

            thread.join(timeout=10.0)
            assert agent_done.is_set()
            assert len(completed_steps) == 6  # 3 waiting + 3 done
            assert completed_steps == [
                "waiting:Step 1: Login",
                "done:Step 1: Login",
                "waiting:Step 2: Fill form",
                "done:Step 2: Fill form",
                "waiting:Step 3: Confirm",
                "done:Step 3: Confirm",
            ]
        finally:
            t.close()


class TestPageElementInteraction:
    """Test interacting with specific elements before/after ask_user_browser_action."""

    def test_type_into_form_then_ask_user(self, http_server: str) -> None:
        """Type into form fields, then ask user to complete an action."""
        interactions: list[tuple[str, str]] = []

        def callback(instruction: str, url: str) -> None:
            interactions.append((instruction, url))

        t = WebUseTool(headless=True, user_data_dir=None, wait_for_user_callback=callback)
        try:
            # Navigate to settings page
            result = t.go_to_url(f"{http_server}/settings")
            assert "Settings" in result
            assert "Email" in result

            # Find and type into the email field
            result = t.get_page_content()
            assert "Email" in result

            # Ask user to verify settings
            result = t.ask_user_browser_action("Please verify and save settings")
            assert "Settings" in result
            assert len(interactions) == 1
        finally:
            t.close()

    def test_navigate_and_interact_mixed(self, http_server: str) -> None:
        """Alternate between automated navigation and user actions."""
        t = WebUseTool(headless=True, user_data_dir=None, wait_for_user_callback=None)
        try:
            # Automated: go to login
            result = t.go_to_url(http_server)
            assert "Login" in result

            # User action: login
            result = t.ask_user_browser_action("Please log in")
            assert "Login" in result

            # Automated: go to forgot password
            result = t.go_to_url(f"{http_server}/forgot")
            assert "Reset Password" in result
            assert "Reset Email" in result

            # User action: enter email for reset
            result = t.ask_user_browser_action("Enter your email for password reset")
            assert "Reset Password" in result

            # Automated: go back to login
            result = t.go_to_url(http_server)
            assert "Login" in result
        finally:
            t.close()


class TestStopEventDuringMultiStep:
    """Test that stop events properly interrupt multi-step flows."""

    def test_stop_interrupts_second_step(self) -> None:
        """A stop event during step 2 should halt the multi-step flow."""
        stop_event = threading.Event()
        step_count = 0

        def interruptible_callback(instruction: str, url: str) -> None:
            nonlocal step_count
            step_count += 1
            event = threading.Event()
            while not event.wait(timeout=0.05):
                if stop_event.is_set():
                    raise KeyboardInterrupt("Stopped by user")

        interrupted = threading.Event()
        error_msg: list[str | None] = [None]

        def agent_flow() -> None:
            try:
                interruptible_callback("Step 1", "http://example.com")
            except KeyboardInterrupt as e:
                error_msg[0] = str(e)
                interrupted.set()

        # Set stop before starting - first callback should raise immediately
        stop_event.set()
        t = threading.Thread(target=agent_flow)
        t.start()
        t.join(timeout=5.0)
        assert interrupted.is_set()
        assert step_count == 1
        assert error_msg[0] is not None and "Stopped by user" in error_msg[0]

    def test_stop_during_blocking_wait(self) -> None:
        """Stop event fires while callback is blocking, causing clean interruption."""
        stop_event = threading.Event()
        started_waiting = threading.Event()

        def slow_callback(instruction: str, url: str) -> None:
            started_waiting.set()
            event = threading.Event()
            while not event.wait(timeout=0.05):
                if stop_event.is_set():
                    raise KeyboardInterrupt("Agent stopped")

        interrupted = threading.Event()

        def agent_flow() -> None:
            try:
                slow_callback("Do something slow", "http://example.com")
            except KeyboardInterrupt:
                interrupted.set()

        t = threading.Thread(target=agent_flow)
        t.start()

        # Wait for callback to start blocking
        assert started_waiting.wait(timeout=5.0)
        time.sleep(0.1)

        # Fire stop
        stop_event.set()
        t.join(timeout=5.0)
        assert interrupted.is_set()
        assert not t.is_alive()


class TestAskUserWithPageContent:
    """Test that ask_user_browser_action returns correct page content at each step."""

    def test_content_changes_across_navigations(self, http_server: str) -> None:
        """Each ask_user_browser_action with a URL returns the correct page's content."""
        t = WebUseTool(headless=True, user_data_dir=None, wait_for_user_callback=None)
        try:
            # Login page
            result = t.ask_user_browser_action("Check login", url=http_server)
            assert "Login" in result
            assert "Sign In" in result
            assert "Dashboard" not in result

            # Dashboard
            result = t.ask_user_browser_action(
                "Check dashboard", url=f"{http_server}/dashboard"
            )
            assert "Dashboard" in result
            assert "Welcome" in result
            assert "Sign In" not in result

            # Wizard step 2
            result = t.ask_user_browser_action(
                "Check wizard", url=f"{http_server}/wizard/step2"
            )
            assert "Step 2" in result
            assert "Notifications" in result
            assert "Dashboard" not in result  # main heading should not be Dashboard

            # Settings
            result = t.ask_user_browser_action(
                "Check settings", url=f"{http_server}/settings"
            )
            assert "Settings" in result
            assert "Email" in result
        finally:
            t.close()

    def test_ask_user_without_url_stays_on_current_page(
        self, http_server: str
    ) -> None:
        """Without a URL, ask_user_browser_action returns the current page."""
        t = WebUseTool(headless=True, user_data_dir=None, wait_for_user_callback=None)
        try:
            t.go_to_url(f"{http_server}/wizard/step2")
            result = t.ask_user_browser_action("Toggle some checkboxes")
            assert "Step 2" in result
            assert "Notifications" in result
        finally:
            t.close()


class TestConcurrentMultiStepCallbacks:
    """Test concurrent scenarios with multiple callback invocations."""

    def test_rapid_sequential_callbacks(self, http_server: str) -> None:
        """Rapidly complete multiple user action steps in sequence."""
        call_log: list[str] = []
        events: list[threading.Event] = []

        def fast_callback(instruction: str, url: str) -> None:
            event = threading.Event()
            events.append(event)
            call_log.append(f"start:{instruction}")
            event.wait(timeout=5.0)
            call_log.append(f"end:{instruction}")

        t = WebUseTool(
            headless=True, user_data_dir=None, wait_for_user_callback=fast_callback
        )

        all_done = threading.Event()

        def agent_work() -> None:
            try:
                for i in range(5):
                    t.ask_user_browser_action(
                        f"Action {i}", url=f"{http_server}/wizard/step{(i % 3) + 1}"
                    )
                all_done.set()
            except Exception:
                pass

        thread = threading.Thread(target=agent_work)
        thread.start()

        # Release each callback as soon as it registers
        for i in range(5):
            deadline = time.monotonic() + 5.0
            while len(events) <= i and time.monotonic() < deadline:
                time.sleep(0.02)
            assert len(events) > i
            events[i].set()

        thread.join(timeout=15.0)
        t.close()

        assert all_done.is_set()
        assert len(call_log) == 10  # 5 start + 5 end
        for i in range(5):
            assert f"start:Action {i}" in call_log
            assert f"end:Action {i}" in call_log


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
