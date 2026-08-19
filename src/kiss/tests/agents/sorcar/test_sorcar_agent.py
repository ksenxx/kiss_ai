# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for sorcar_agent.py: prompt construction, arg parsing, task resolution,
CLI callbacks, callback wiring, bash streaming, and autocomplete clipping."""

from __future__ import annotations

from typing import Any, cast

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.web_use_tool import WebUseTool


class TestSorcarAgentCallbackWiring:
    def test_ask_user_question_without_callback(self) -> None:
        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(user_data_dir=None, headless=True)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            result = ask_tool("hello?")
            assert "not available" in result
        finally:
            agent.web_use_tool.close()

    def test_run_sets_callbacks_temporarily(self) -> None:
        agent = SorcarAgent("test")
        parent_class = cast(Any, agent.__class__.__mro__[1])
        original_perform = parent_class.perform_task
        captured: dict[str, object] = {}

        def ask_callback(question: str) -> str:
            return f"UI: {question}"

        def fake_perform(
            self: object, tools: list, attachments: list | None = None,
        ) -> str:
            del self, attachments
            captured["ask"] = getattr(agent, "_ask_user_question_callback", None)
            callables = [t for t in tools if callable(t)]
            ask_tool = next(t for t in callables if t.__name__ == "ask_user_question")
            captured["answer"] = ask_tool("hello")
            return "success: true\nis_continue: false\nsummary: ok\n"

        parent_class.perform_task = fake_perform  # type: ignore[method-assign]
        try:
            result = agent.run(
                prompt_template="task",
                ask_user_question_callback=ask_callback,
            )
        finally:
            parent_class.perform_task = original_perform  # type: ignore[method-assign]

        assert "success: true" in result
        assert captured["ask"] is ask_callback
        assert captured["answer"] == "UI: hello"
        assert getattr(agent, "_ask_user_question_callback", None) is None
        assert agent.web_use_tool is None
