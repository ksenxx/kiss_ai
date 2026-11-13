# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: task run without available models emits result-panel error.

When no API keys are configured (no models available), running a task
must broadcast a ``result`` event with ``success=False`` and the text
"No model available.  Set at least one API key in the environment."
so the frontend displays the error in the rich result panel.  When
other models are usable and only the run's own model is not, the text
names that model instead.
"""

from __future__ import annotations

import os
import queue
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import TestCase

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.tests.conftest import install_cli_script
from kiss.tests.server._memory_printer import MemoryPrinter

# Every credential ``_configured_providers`` consults, the Bedrock key
# pair and bearer token included: the autouse conftest fixture hides the
# ambient AWS chain, but the config object captured the exported
# variables at import time.
_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def _make_server(printer: Any = None) -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.server import AgentServer

    return AgentServer(printer)


def _register_tab_state(task_id: str, tab_id: str, agent: ChatSorcarAgent) -> Any:
    """Register a server-owned AgentState for *tab_id* carrying *agent*."""
    st = agent_state.AgentState(
        task_id,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
    )
    st.user_answer_queue = queue.Queue()
    agent_state.register(st)
    return st


def _unregister_state(st: Any) -> None:
    """Remove *st* from the registry regardless of any mid-run re-keying."""
    for s in agent_state.snapshot():
        if s is st:
            agent_state.unregister(s.task_id, s)


class TestNoModelAvailableResultEvent(TestCase):
    """When no models are available, a ``result`` event with the correct
    error message must be broadcast before the agent even starts."""

    def test_no_model_emits_result_event(self) -> None:
        """With all API keys cleared, running a task should immediately
        broadcast a result event with the no-model error message."""
        from kiss.core import config as config_module

        keys = config_module.DEFAULT_CONFIG
        saved = {name: getattr(keys, name) for name in _CREDENTIALS}
        from kiss.core.models import codex_model as codex_module

        saved_path = os.environ.get("PATH", "")
        saved_codex_paths = codex_module._UI_CANDIDATE_PATHS
        try:
            for name in _CREDENTIALS:
                setattr(keys, name, "")
            os.environ["PATH"] = ""
            codex_module._UI_CANDIDATE_PATHS = ()

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
            agent = ChatSorcarAgent("Sorcar")

            run_called = False

            def fake_run(**kwargs: Any) -> None:
                nonlocal run_called
                run_called = True

            agent.run = fake_run  # type: ignore[method-assign, assignment]
            st = _register_tab_state("no-model-task-1", tab_id, agent)

            task_thread = threading.Thread(
                target=server._run_task,
                args=({"type": "run", "prompt": "do something", "tabId": tab_id},),
                daemon=True,
            )
            task_thread.start()
            task_thread.join(timeout=10)
            _unregister_state(st)

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
            codex_module._UI_CANDIDATE_PATHS = saved_codex_paths

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
            agent = ChatSorcarAgent("Sorcar")

            run_called = False

            def fake_run(**kwargs: Any) -> None:
                nonlocal run_called
                run_called = True
                agent.total_tokens_used = 100
                agent.budget_used = 0.01
                agent.step_count = 1
                printer = kwargs.get("printer", server.printer)
                printer.print(
                    "success: true\nsummary: Done",
                    type="result",
                    total_tokens=100,
                    cost="$0.0100",
                    step_count=1,
                )

            agent.run = fake_run  # type: ignore[method-assign, assignment]
            st = _register_tab_state("model-ok-task-1", tab_id, agent)

            task_thread = threading.Thread(
                target=server._run_task,
                args=({"type": "run", "prompt": "test task", "tabId": tab_id},),
                daemon=True,
            )
            task_thread.start()
            task_thread.join(timeout=10)
            _unregister_state(st)

            assert run_called, "Agent's run() should be called when a model is available"

            with lock:
                result_events = [e for e in events if e.get("type") == "result"]

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


def test_unusable_model_is_named_when_other_models_are_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With only the ``claude``/``codex`` CLIs present, an API model is refused by name."""
    from kiss.core import config as config_module

    for name in _CREDENTIALS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, name, "")
    for cli in ("claude", "codex"):
        install_cli_script(tmp_path / cli, "#!/bin/sh\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    available = get_available_models()
    assert available and "claude-opus-4-6" not in available

    printer = MemoryPrinter()
    tab_id = "unusable-model-test-1"
    st = _register_tab_state("unusable-model-task-1", tab_id, ChatSorcarAgent("Sorcar"))
    try:
        _make_server(printer)._run_task(
            {"type": "run", "prompt": "do something", "tabId": tab_id, "model": "claude-opus-4-6"}
        )
    finally:
        _unregister_state(st)
    results = [e for e in printer.emitted if e.get("type") == "result"]
    assert [(e["tabId"], e["success"], e["text"]) for e in results] == [(
        tab_id,
        False,
        'No model available under the name "claude-opus-4-6".  Pick another model'
        " or configure its provider (API key or CLI).",
    )]


if __name__ == "__main__":
    unittest.main()
