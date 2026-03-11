"""Tests for the ask_user_question feature.

Tests the full flow: agent tool -> callback -> server endpoint -> UI event.
"""

import threading
import time

import pytest


class TestAskUserQuestionCallback:
    """Test the callback pattern used by sorcar.py for ask_user_question."""

    def test_event_based_question_flow(self) -> None:
        """Full flow: agent asks question, user answers, agent gets answer."""
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

        # Simulate agent calling the callback in a thread
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

        # Simulate user answering
        user_question_answer = "sk-abc123"
        user_question_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert result_holder[0] == "sk-abc123"

    def test_stop_event_interrupts_question(self) -> None:
        """If stop event is set while waiting, KeyboardInterrupt is raised."""
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
        """User can submit an empty answer."""
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
        # Submit empty answer
        user_question_answer = ""
        user_question_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert result_holder[0] == ""

    def test_multiline_answer(self) -> None:
        """User can submit a multi-line answer."""
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


class TestSorcarAgentAskUserQuestion:
    """Test that SorcarAgent exposes ask_user_question as a tool."""

    def test_callback_passed_through(self) -> None:
        """SorcarAgent stores the ask_user_question_callback."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        def my_callback(question: str) -> str:
            return "answer"

        agent = SorcarAgent("test", ask_user_question_callback=my_callback)
        assert agent._ask_user_question_callback is my_callback

    def test_default_no_callback(self) -> None:
        """SorcarAgent defaults to no ask_user_question_callback."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("test")
        assert agent._ask_user_question_callback is None

    def test_ask_user_question_tool_in_tools(self) -> None:
        """ask_user_question tool is included in _get_tools()."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            tool_names = [t.__name__ for t in tools]
            assert "ask_user_question" in tool_names
        finally:
            agent.web_use_tool.close()

    def test_ask_user_question_tool_without_callback(self) -> None:
        """Without callback, the tool returns a 'not available' message."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            result = ask_tool("What is your name?")
            assert "not available" in result
        finally:
            agent.web_use_tool.close()

    def test_ask_user_question_tool_with_callback(self) -> None:
        """With callback, the tool calls through to the callback."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        def my_callback(question: str) -> str:
            return f"answer to: {question}"

        agent = SorcarAgent("test", ask_user_question_callback=my_callback)
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            result = ask_tool("What color?")
            assert result == "answer to: What color?"
        finally:
            agent.web_use_tool.close()

    def test_ask_user_question_tool_docstring(self) -> None:
        """The tool has a proper docstring for tool registration."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            assert ask_tool.__doc__ is not None
            assert "question" in ask_tool.__doc__.lower()
        finally:
            agent.web_use_tool.close()


class TestSorcarEndpointIntegration:
    """Test the /user-question-done endpoint via the Starlette app."""

    @pytest.fixture()
    def app_client(self, tmp_path):
        """Create a test app with the user question endpoints."""
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
        """Without a pending question, returns 404."""
        client, state = app_client
        resp = client.post(
            "/user-question-done",
            json={"answer": "test"},
        )
        assert resp.status_code == 404
        assert resp.json()["error"] == "No pending question"

    def test_answer_submitted_sets_event(self, app_client) -> None:
        """When a question is pending, submitting answer sets the event."""
        client, state = app_client
        state.event = threading.Event()
        resp = client.post(
            "/user-question-done",
            json={"answer": "my answer"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert state.event.is_set()
        assert state.answer == "my answer"

    def test_empty_answer_accepted(self, app_client) -> None:
        """An empty answer string is accepted."""
        client, state = app_client
        state.event = threading.Event()
        resp = client.post(
            "/user-question-done",
            json={"answer": ""},
        )
        assert resp.status_code == 200
        assert state.event.is_set()
        assert state.answer == ""

    def test_missing_answer_field_defaults_empty(self, app_client) -> None:
        """If 'answer' field is missing, defaults to empty string."""
        client, state = app_client
        state.event = threading.Event()
        resp = client.post(
            "/user-question-done",
            json={},
        )
        assert resp.status_code == 200
        assert state.event.is_set()
        assert state.answer == ""


class TestChatbotUIUserQuestion:
    """Test that the chatbot UI HTML contains the user_question event handler."""

    def test_user_question_case_in_html(self) -> None:
        """The generated HTML contains a case for 'user_question' event."""
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test", "", "/tmp")
        assert "user_question" in html
        assert "Agent Question" in html
        assert "user-question-done" in html

    def test_user_question_has_textarea(self) -> None:
        """The user_question card includes a textarea for the user to type."""
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test", "", "/tmp")
        # Find the user_question section
        idx = html.index("user_question")
        section = html[idx : idx + 2000]
        assert "textarea" in section
        assert "Type your answer" in section


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
