"""Tests for ask_user_question callback and endpoint."""

import threading
import time

import pytest


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
