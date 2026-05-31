"""Integration test: stop button cancels the model's in-flight API request.

When the user clicks the stop button while the agent is blocked in a
model API call (e.g. waiting for a streaming response from
Anthropic/OpenAI/Gemini), the ``_stop_task`` method must cancel the
model's HTTP client so the network call fails immediately, unblocking
the agent thread.

Previously, ``_stop_task`` only set the cooperative ``stop_event`` and
(after a delay) injected ``KeyboardInterrupt`` via
``PyThreadState_SetAsyncExc``.  When the thread was blocked inside a
C-extension HTTP client (httpx), the async exception could not be
delivered, leaving the agent stuck until the model eventually responded.

The fix adds ``Model.cancel()`` (closes the HTTP client) and calls it
from ``_stop_task`` via ``_cancel_model_async``.
"""

from __future__ import annotations

import os
import threading
import unittest
from typing import Any
from unittest import mock

from kiss.core.models.model import Model


class _FakeModel(Model):
    """A minimal Model subclass that tracks cancel() calls."""

    def __init__(self) -> None:
        super().__init__("fake-model")
        self.cancel_called = threading.Event()

    def cancel(self) -> None:
        self.cancel_called.set()

    def initialize(self, prompt: str, attachments: Any = None) -> None:
        pass

    def generate(self) -> tuple[str, Any]:
        return "", None

    def generate_and_process_with_tools(
        self, function_map: Any, tools_schema: Any = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        return [], "", None

    def extract_input_output_token_counts_from_response(
        self, response: Any,
    ) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        return []


class _FakeExecutor:
    """Mimics KISSAgent with a .model attribute."""

    def __init__(self, model: _FakeModel) -> None:
        self.model = model


class _FakeAgent:
    """Mimics WorktreeSorcarAgent with a _current_executor attribute."""

    def __init__(self, executor: _FakeExecutor | None) -> None:
        self._current_executor = executor
        self._last_task_id: int | None = None


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer
    return VSCodeServer()


class TestStopCancelsModel(unittest.TestCase):
    """Verify that _stop_task calls Model.cancel() to unblock the agent thread."""

    def test_stop_task_calls_model_cancel(self) -> None:
        """When _stop_task is called, it must invoke cancel() on the live model."""
        server = _make_server()
        tab = server._get_tab("tab-stop")
        fake_model = _FakeModel()
        executor = _FakeExecutor(fake_model)
        fake_agent = _FakeAgent(executor)

        # Set up a running task
        with server._state_lock:
            tab.stop_event = threading.Event()
            # Create a dummy thread so _stop_task thinks a task is running
            dummy = threading.Thread(target=lambda: None, daemon=True)
            dummy.start()
            dummy.join()
            tab.task_thread = dummy
            tab.agent = fake_agent  # type: ignore[assignment]

        server._stop_task("tab-stop")

        assert fake_model.cancel_called.is_set(), (
            "Model.cancel() was not called by _stop_task"
        )

    def test_stop_task_sets_stop_event_and_cancels_model(self) -> None:
        """Both stop_event and model.cancel() must be triggered."""
        server = _make_server()
        tab = server._get_tab("tab-both")
        fake_model = _FakeModel()
        executor = _FakeExecutor(fake_model)
        fake_agent = _FakeAgent(executor)

        stop_event = threading.Event()
        with server._state_lock:
            tab.stop_event = stop_event
            dummy = threading.Thread(target=lambda: None, daemon=True)
            dummy.start()
            dummy.join()
            tab.task_thread = dummy
            tab.agent = fake_agent  # type: ignore[assignment]

        server._stop_task("tab-both")

        assert stop_event.is_set(), "stop_event was not set"
        assert fake_model.cancel_called.is_set(), "Model.cancel() was not called"

    def test_stop_task_no_agent_does_not_crash(self) -> None:
        """_stop_task with no agent (tab.agent is None) must not crash."""
        server = _make_server()
        tab = server._get_tab("tab-none")
        with server._state_lock:
            tab.stop_event = threading.Event()
            tab.agent = None
        # Should not raise
        server._stop_task("tab-none")

    def test_stop_task_no_executor_does_not_crash(self) -> None:
        """_stop_task with agent but no _current_executor must not crash."""
        server = _make_server()
        tab = server._get_tab("tab-noexec")
        fake_agent = _FakeAgent(None)  # type: ignore[arg-type]
        fake_agent._current_executor = None
        with server._state_lock:
            tab.stop_event = threading.Event()
            tab.agent = fake_agent  # type: ignore[assignment]
        # Should not raise
        server._stop_task("tab-noexec")


class TestCancelModelAsync(unittest.TestCase):
    """Direct tests for _cancel_model_async static method."""

    def test_calls_cancel_on_model(self) -> None:
        """_cancel_model_async calls model.cancel() through the agent hierarchy."""
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin

        fake_model = _FakeModel()
        executor = _FakeExecutor(fake_model)
        fake_agent = _FakeAgent(executor)

        _TaskRunnerMixin._cancel_model_async(fake_agent)  # type: ignore[arg-type]
        assert fake_model.cancel_called.is_set()

    def test_none_agent_is_noop(self) -> None:
        """_cancel_model_async(None) is a no-op."""
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin

        _TaskRunnerMixin._cancel_model_async(None)  # Should not raise

    def test_no_executor_is_noop(self) -> None:
        """When agent has no _current_executor, cancel is a no-op."""
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin

        fake_agent = _FakeAgent(None)  # type: ignore[arg-type]
        fake_agent._current_executor = None
        _TaskRunnerMixin._cancel_model_async(fake_agent)  # type: ignore[arg-type]


class TestModelCancelDefault(unittest.TestCase):
    """Test Model.cancel() default implementation."""

    def test_cancel_calls_close_on_client(self) -> None:
        """Model.cancel() calls client.close() when available."""
        fake_model = _FakeModel()
        mock_client = mock.MagicMock()
        fake_model.client = mock_client

        # Call the base class cancel (not the overridden one)
        Model.cancel(fake_model)

        mock_client.close.assert_called_once()

    def test_cancel_no_client_is_noop(self) -> None:
        """Model.cancel() is a no-op when client is None."""
        fake_model = _FakeModel()
        fake_model.client = None
        Model.cancel(fake_model)  # Should not raise

    def test_cancel_client_without_close_is_noop(self) -> None:
        """Model.cancel() is a no-op when client has no close method."""
        fake_model = _FakeModel()
        fake_model.client = "not-a-real-client"  # Has no close()
        Model.cancel(fake_model)  # Should not raise

    def test_cancel_swallows_exceptions(self) -> None:
        """Model.cancel() swallows exceptions from client.close()."""
        fake_model = _FakeModel()
        mock_client = mock.MagicMock()
        mock_client.close.side_effect = RuntimeError("connection pool error")
        fake_model.client = mock_client
        Model.cancel(fake_model)  # Should not raise


class TestStopResolvesThroughSubscriber(unittest.TestCase):
    """When stop is issued from a subscriber tab, it should resolve
    to the source tab and cancel that tab's model."""

    def test_stop_subscriber_tab_cancels_source_model(self) -> None:
        """Stopping from a viewer tab resolves to source and calls cancel."""
        server = _make_server()

        # Source tab with a running task
        source_tab = server._get_tab("source-tab")
        fake_model = _FakeModel()
        executor = _FakeExecutor(fake_model)
        fake_agent = _FakeAgent(executor)
        stop_event = threading.Event()

        with server._state_lock:
            source_tab.stop_event = stop_event
            dummy = threading.Thread(target=lambda: None, daemon=True)
            dummy.start()
            dummy.join()
            source_tab.task_thread = dummy
            source_tab.agent = fake_agent  # type: ignore[assignment]

        # Viewer tab subscribed to the same task
        server._get_tab("viewer-tab")
        server.printer.subscribe_tab("task-123", "source-tab")
        server.printer.subscribe_tab("task-123", "viewer-tab")

        # Stop from the viewer tab
        server._stop_task("viewer-tab")

        assert stop_event.is_set(), "Source tab's stop_event was not set"
        assert fake_model.cancel_called.is_set(), (
            "Source tab's model.cancel() was not called via subscriber resolution"
        )


if __name__ == "__main__":
    unittest.main()
