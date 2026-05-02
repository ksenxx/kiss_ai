"""Integration tests: task run without available models emits result-panel error.

When no API keys are configured (no models available), running a task
must broadcast a ``result`` event with ``success=False`` and the text
"No model available.  Set at least one API key in the environment."
so the frontend displays the error in the rich result panel.
"""

from __future__ import annotations

import os
import queue
import threading
import unittest
from typing import Any
from unittest import TestCase


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer

    return VSCodeServer()


class TestNoModelAvailableResultEvent(TestCase):
    """When no models are available, a ``result`` event with the correct
    error message must be broadcast before the agent even starts."""

    def test_no_model_emits_result_event(self) -> None:
        """With all API keys cleared, running a task should immediately
        broadcast a result event with the no-model error message."""
        # Save and clear all API keys
        from kiss.core import config as config_module

        keys = config_module.DEFAULT_CONFIG
        saved = {
            "ANTHROPIC_API_KEY": keys.ANTHROPIC_API_KEY,
            "OPENAI_API_KEY": keys.OPENAI_API_KEY,
            "GEMINI_API_KEY": keys.GEMINI_API_KEY,
            "TOGETHER_API_KEY": keys.TOGETHER_API_KEY,
            "OPENROUTER_API_KEY": keys.OPENROUTER_API_KEY,
            "MINIMAX_API_KEY": getattr(keys, "MINIMAX_API_KEY", ""),
        }
        # Also clear PATH so shutil.which("claude") (and any other CLI
        # provider lookups in get_available_models) returns None — without
        # this, machines with the Claude Code CLI installed report cc/*
        # models as available even when every API key is empty, which
        # bypasses the no-model gate this test is exercising.
        saved_path = os.environ.get("PATH", "")
        try:
            keys.ANTHROPIC_API_KEY = ""
            keys.OPENAI_API_KEY = ""
            keys.GEMINI_API_KEY = ""
            keys.TOGETHER_API_KEY = ""
            keys.OPENROUTER_API_KEY = ""
            keys.MINIMAX_API_KEY = ""
            os.environ["PATH"] = ""

            server = _make_server()
            events: list[dict[str, Any]] = []
            lock = threading.Lock()

            orig_broadcast = server.printer.broadcast

            def capture(e: dict[str, Any]) -> None:
                with lock:
                    events.append(dict(e))
                orig_broadcast(e)

            server.printer.broadcast = capture  # type: ignore[assignment]

            tab_id = "no-model-test-1"
            tab = server._get_tab(tab_id)

            # The agent's run should NOT be called at all
            run_called = False

            def fake_run(**kwargs: Any) -> None:
                nonlocal run_called
                run_called = True

            tab.agent.run = fake_run  # type: ignore[assignment]

            stop_event = threading.Event()
            tab.stop_event = stop_event
            tab.user_answer_queue = queue.Queue()

            task_thread = threading.Thread(
                target=server._run_task,
                args=({"type": "run", "prompt": "do something", "tabId": tab_id},),
                daemon=True,
            )
            tab.task_thread = task_thread
            task_thread.start()
            task_thread.join(timeout=10)

            with lock:
                result_events = [e for e in events if e.get("type") == "result"]

            assert not run_called, "Agent's run() should not be called when no models available"

            assert len(result_events) >= 1, (
                f"Expected at least one result event, got {len(result_events)}. "
                f"All events: {[e.get('type') for e in events]}"
            )
            result_ev = result_events[-1]
            assert result_ev.get("success") is False, (
                f"Result event should have success=False, got {result_ev.get('success')}"
            )
            expected_text = (
                "No model available.  Set at least one API key in the environment."
            )
            assert expected_text in (result_ev.get("text") or ""), (
                f"Result text should contain '{expected_text}', got: {result_ev.get('text')}"
            )
        finally:
            for attr, val in saved.items():
                setattr(keys, attr, val)
            os.environ["PATH"] = saved_path

    def test_with_model_proceeds_normally(self) -> None:
        """When API keys are set, task should proceed to agent.run()."""
        from kiss.core import config as config_module

        keys = config_module.DEFAULT_CONFIG
        saved_key = keys.ANTHROPIC_API_KEY
        try:
            keys.ANTHROPIC_API_KEY = "test-key-for-model-check"

            server = _make_server()
            events: list[dict[str, Any]] = []
            lock = threading.Lock()

            orig_broadcast = server.printer.broadcast

            def capture(e: dict[str, Any]) -> None:
                with lock:
                    events.append(dict(e))
                orig_broadcast(e)

            server.printer.broadcast = capture  # type: ignore[assignment]

            tab_id = "model-ok-test-1"
            tab = server._get_tab(tab_id)

            run_called = False

            def fake_run(**kwargs: Any) -> None:
                nonlocal run_called
                run_called = True
                tab.agent.total_tokens_used = 100
                tab.agent.budget_used = 0.01
                tab.agent.step_count = 1
                printer = kwargs.get("printer", server.printer)
                printer.print(
                    "success: true\nsummary: Done",
                    type="result",
                    total_tokens=100,
                    cost="$0.0100",
                    step_count=1,
                )

            tab.agent.run = fake_run  # type: ignore[assignment]

            stop_event = threading.Event()
            tab.stop_event = stop_event
            tab.user_answer_queue = queue.Queue()

            task_thread = threading.Thread(
                target=server._run_task,
                args=({"type": "run", "prompt": "test task", "tabId": tab_id},),
                daemon=True,
            )
            tab.task_thread = task_thread
            task_thread.start()
            task_thread.join(timeout=10)

            assert run_called, "Agent's run() should be called when a model is available"

            with lock:
                result_events = [e for e in events if e.get("type") == "result"]

            # Should have the success result, not the no-model error
            assert len(result_events) >= 1
            no_model_results = [
                e for e in result_events
                if "No model available" in (e.get("text") or "")
            ]
            assert len(no_model_results) == 0, (
                "Should not emit no-model error when a key is configured"
            )
        finally:
            keys.ANTHROPIC_API_KEY = saved_key


if __name__ == "__main__":
    unittest.main()
