"""Tests for ask_user_browser_action, ask_user_question, and multi-step scenarios."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool


TEST_PAGE = b"""<!DOCTYPE html>
<html><head><title>CAPTCHA Page</title></head>
<body>
  <h1>Solve CAPTCHA</h1>
  <button id="verify">Verify</button>
</body></html>"""

OTHER_PAGE = b"""<!DOCTYPE html>
<html><head><title>Other Page</title></head>
<body>
  <h1>Other Page</h1>
  <a href="/">Back</a>
</body></html>"""

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
    "/other": OTHER_PAGE,
    "/captcha": TEST_PAGE,
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


class TestAskUserBrowserAction:
    def test_with_url_navigates_first(self, http_server: str) -> None:
        tool = WebUseTool(user_data_dir=None, wait_for_user_callback=None)
        try:
            tool.go_to_url(http_server)
            result = tool.ask_user_browser_action(
                "Check this page", url=f"{http_server}/other"
            )
            assert "Other Page" in result
        finally:
            tool.close()


class TestWaitForUserBrowserCallback:
    def test_event_based_callback(self) -> None:
        broadcasts: list[dict] = []
        user_action_event: threading.Event | None = None

        def _wait_for_user_browser(instruction: str, url: str) -> None:
            nonlocal user_action_event
            event = threading.Event()
            user_action_event = event
            broadcasts.append({
                "type": "user_browser_action",
                "instruction": instruction,
                "url": url,
            })
            while not event.wait(timeout=0.1):
                pass
            user_action_event = None

        done = threading.Event()

        def agent_thread() -> None:
            _wait_for_user_browser("Solve CAPTCHA", "http://example.com")
            done.set()

        t = threading.Thread(target=agent_thread)
        t.start()
        time.sleep(0.2)
        assert len(broadcasts) == 1
        assert broadcasts[0]["instruction"] == "Solve CAPTCHA"
        assert user_action_event is not None
        user_action_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert not t.is_alive()

    def test_stop_event_interrupts_callback(self) -> None:
        current_stop_event = threading.Event()

        def _wait_for_user_browser(instruction: str, url: str) -> None:
            event = threading.Event()
            while not event.wait(timeout=0.1):
                if current_stop_event.is_set():
                    raise KeyboardInterrupt("Agent stopped while waiting for user")

        current_stop_event.set()
        with pytest.raises(KeyboardInterrupt, match="Agent stopped"):
            _wait_for_user_browser("Solve CAPTCHA", "http://example.com")


class TestPageElementInteraction:
    def test_type_into_form_then_ask_user(self, http_server: str) -> None:
        interactions: list[tuple[str, str]] = []

        def callback(instruction: str, url: str) -> None:
            interactions.append((instruction, url))

        t = WebUseTool(user_data_dir=None, wait_for_user_callback=callback)
        try:
            result = t.go_to_url(f"{http_server}/settings")
            assert "Settings" in result
            assert "Email" in result
            result = t.get_page_content()
            assert "Email" in result
            result = t.ask_user_browser_action("Please verify and save settings")
            assert "Settings" in result
            assert len(interactions) == 1
        finally:
            t.close()


class TestStopEventDuringMultiStep:
    def test_stop_interrupts_second_step(self) -> None:
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

        stop_event.set()
        t = threading.Thread(target=agent_flow)
        t.start()
        t.join(timeout=5.0)
        assert interrupted.is_set()
        assert step_count == 1
        assert error_msg[0] is not None and "Stopped by user" in error_msg[0]

    def test_stop_during_blocking_wait(self) -> None:
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
        assert started_waiting.wait(timeout=5.0)
        time.sleep(0.1)
        stop_event.set()
        t.join(timeout=5.0)
        assert interrupted.is_set()
        assert not t.is_alive()


class TestConcurrentMultiStepCallbacks:
    def test_rapid_sequential_callbacks(self, http_server: str) -> None:
        call_log: list[str] = []
        events: list[threading.Event] = []

        def fast_callback(instruction: str, url: str) -> None:
            event = threading.Event()
            events.append(event)
            call_log.append(f"start:{instruction}")
            event.wait(timeout=5.0)
            call_log.append(f"end:{instruction}")

        t = WebUseTool(
            user_data_dir=None, wait_for_user_callback=fast_callback
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

        for i in range(5):
            deadline = time.monotonic() + 5.0
            while len(events) <= i and time.monotonic() < deadline:
                time.sleep(0.02)
            assert len(events) > i
            events[i].set()

        thread.join(timeout=15.0)
        t.close()

        assert all_done.is_set()
        assert len(call_log) == 10
        for i in range(5):
            assert f"start:Action {i}" in call_log
            assert f"end:Action {i}" in call_log


class TestAskUserQuestionCallback:
    def test_event_based_question_flow(self) -> None:
        broadcasts: list[dict] = []
        user_question_event: threading.Event | None = None
        user_question_answer = ""

        def _ask_user_question(question: str) -> str:
            nonlocal user_question_event, user_question_answer
            event = threading.Event()
            user_question_event = event
            user_question_answer = ""
            broadcasts.append({"type": "user_question", "question": question})
            while not event.wait(timeout=0.1):
                pass
            answer = user_question_answer
            user_question_event = None
            user_question_answer = ""
            return answer

        result_holder: list[str] = [""]
        done = threading.Event()

        def agent_thread() -> None:
            result_holder[0] = _ask_user_question("What is the API key?")
            done.set()

        t = threading.Thread(target=agent_thread)
        t.start()
        time.sleep(0.2)
        assert len(broadcasts) == 1
        assert broadcasts[0]["question"] == "What is the API key?"
        assert user_question_event is not None
        user_question_answer = "sk-abc123"
        user_question_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert result_holder[0] == "sk-abc123"

    def test_stop_event_interrupts_question(self) -> None:
        current_stop_event = threading.Event()

        def _ask_user_question(question: str) -> str:
            event = threading.Event()
            while not event.wait(timeout=0.1):
                if current_stop_event.is_set():
                    raise KeyboardInterrupt(
                        "Agent stopped while waiting for user answer"
                    )
            return ""

        current_stop_event.set()
        with pytest.raises(KeyboardInterrupt, match="Agent stopped"):
            _ask_user_question("What color?")

    def test_empty_answer(self) -> None:
        user_question_event: threading.Event | None = None
        user_question_answer = ""

        def _ask_user_question(question: str) -> str:
            nonlocal user_question_event, user_question_answer
            event = threading.Event()
            user_question_event = event
            user_question_answer = ""
            while not event.wait(timeout=0.1):
                pass
            answer = user_question_answer
            user_question_event = None
            user_question_answer = ""
            return answer

        result_holder: list[str] = ["unset"]
        done = threading.Event()

        def agent_thread() -> None:
            result_holder[0] = _ask_user_question("Any preference?")
            done.set()

        t = threading.Thread(target=agent_thread)
        t.start()
        time.sleep(0.2)
        assert user_question_event is not None
        user_question_answer = ""
        user_question_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert result_holder[0] == ""

    def test_multiline_answer(self) -> None:
        user_question_event: threading.Event | None = None
        user_question_answer = ""

        def _ask_user_question(question: str) -> str:
            nonlocal user_question_event, user_question_answer
            event = threading.Event()
            user_question_event = event
            user_question_answer = ""
            while not event.wait(timeout=0.1):
                pass
            answer = user_question_answer
            user_question_event = None
            user_question_answer = ""
            return answer

        result_holder: list[str] = [""]
        done = threading.Event()

        def agent_thread() -> None:
            result_holder[0] = _ask_user_question("Describe the issue")
            done.set()

        t = threading.Thread(target=agent_thread)
        t.start()
        time.sleep(0.2)
        assert user_question_event is not None
        multiline = "Line 1\nLine 2\nLine 3"
        user_question_answer = multiline
        user_question_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert result_holder[0] == multiline


class TestSorcarEndpointIntegration:
    @pytest.fixture()
    def app_client(self, tmp_path):
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        class State:
            event: threading.Event | None = None
            answer: str = ""

        state = State()

        async def user_question_done(request: Request) -> JSONResponse:
            if state.event is not None:
                body = await request.json()
                state.answer = body.get("answer", "")
                state.event.set()
                return JSONResponse({"status": "ok"})
            return JSONResponse({"error": "No pending question"}, status_code=404)

        app = Starlette(
            routes=[
                Route(
                    "/user-question-done",
                    user_question_done,
                    methods=["POST"],
                ),
            ]
        )
        client = TestClient(app)
        yield client, state

    def test_no_pending_question_returns_404(self, app_client) -> None:
        client, state = app_client
        resp = client.post("/user-question-done", json={"answer": "test"})
        assert resp.status_code == 404
        assert resp.json()["error"] == "No pending question"

    def test_answer_submitted_sets_event(self, app_client) -> None:
        client, state = app_client
        state.event = threading.Event()
        resp = client.post("/user-question-done", json={"answer": "my answer"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert state.event.is_set()
        assert state.answer == "my answer"

    def test_empty_answer_accepted(self, app_client) -> None:
        client, state = app_client
        state.event = threading.Event()
        resp = client.post("/user-question-done", json={"answer": ""})
        assert resp.status_code == 200
        assert state.event.is_set()
        assert state.answer == ""

    def test_missing_answer_field_defaults_empty(self, app_client) -> None:
        client, state = app_client
        state.event = threading.Event()
        resp = client.post("/user-question-done", json={})
        assert resp.status_code == 200
        assert state.event.is_set()
        assert state.answer == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
